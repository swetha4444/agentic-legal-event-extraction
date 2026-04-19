#!/usr/bin/env python3
"""Chunk-level legal event-graph extraction (force per-sentence events when verbs exist)."""
import argparse
import json
import os
import re
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __name__ == "__main__":
    _src = Path(__file__).resolve().parents[1]
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from openai import OpenAI

from agents.config_loader import get_llm_config
from rebuilt_event_graphs_force_sentence_events.prompts import (
    EMPTY_RETRY_SUFFIX,
    PARSE_RETRY_SUFFIX,
    build_event_graph_messages,
)

TRUNCATION_PARSE_MARKERS = (
    "expecting value",
    "unterminated string",
    "expecting ',' delimiter",
    "expecting property name enclosed in double quotes",
    "unterminated object",
    "unterminated array",
)


def _strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()
    return text


def _extract_payload(raw: str) -> tuple[dict[str, Any], list[str], str | None]:
    text = _strip_code_fences(raw)
    if not text:
        return {}, [], "Empty response"
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return {}, [], str(exc)
    if not isinstance(data, dict):
        return {}, [], "Response is not a JSON object"
    keys = list(data.keys())
    payload = None
    if any(k in data for k in ("entities", "events", "temporal_edges", "causal_edges")):
        payload = data
    else:
        for value in data.values():
            if isinstance(value, dict) and any(k in value for k in ("entities", "events", "temporal_edges", "causal_edges")):
                payload = value
                break
    if payload is None:
        payload = data
    return payload, keys, None


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _normalize_sentence_id(value: Any) -> Any:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            return int(text)
        match = re.fullmatch(r"[Ss](\d+)", text)
        if match:
            return int(match.group(1))
    return value


def _normalize_span(span: Any) -> dict:
    span = _as_dict(span)
    out = {}
    if "start" in span:
        out["start"] = span.get("start")
    if "end" in span:
        out["end"] = span.get("end")
    return out


def _normalize_entities(entities: list[Any]) -> list[dict[str, Any]]:
    out = []
    for idx, entity in enumerate(_as_list(entities), start=1):
        if not isinstance(entity, dict):
            continue
        aliases = _as_list(entity.get("aliases"))
        if not aliases:
            mentions = _as_list(entity.get("mentions"))
            aliases = [
                mention.get("span") or mention.get("text") or mention.get("span_text")
                for mention in mentions
                if isinstance(mention, dict) and (mention.get("span") or mention.get("text") or mention.get("span_text"))
            ]
        out.append(
            {
                "entity_id": str(entity.get("entity_id") or f"E{idx}"),
                "name": entity.get("name") or entity.get("label") or entity.get("entity") or f"Entity {idx}",
                "kind": entity.get("kind") or entity.get("type") or "UNKNOWN",
                "canonical_role": entity.get("canonical_role") or entity.get("role") or "UNKNOWN",
                "aliases": aliases,
            }
        )
    return out


def _normalize_snippets(snippets: Any) -> list[dict[str, Any]]:
    out = []
    if isinstance(snippets, str):
        return [{"text": snippets}]
    if isinstance(snippets, dict):
        snippets = [snippets]
    for snippet in _as_list(snippets):
        if isinstance(snippet, str):
            out.append({"text": snippet})
            continue
        if not isinstance(snippet, dict):
            continue
        item = {}
        if "sentence_id" in snippet:
            item["sentence_id"] = _normalize_sentence_id(snippet.get("sentence_id"))
        if "text" in snippet:
            item["text"] = snippet.get("text")
        elif "snippet" in snippet:
            item["text"] = snippet.get("snippet")
        elif "span" in snippet and isinstance(snippet.get("span"), str):
            item["text"] = snippet.get("span")
        if isinstance(snippet.get("span"), dict):
            item["span"] = _normalize_span(snippet.get("span"))
        out.append(item)
    return out


def _normalize_participants(participants: Any) -> list[dict[str, Any]]:
    out = []
    for participant in _as_list(participants):
        if not isinstance(participant, dict):
            continue
        mention = participant.get("mention")
        if isinstance(mention, str):
            mention = {"span_text": mention}
        mention = _as_dict(mention)
        item = {
            "entity_id": participant.get("entity_id") or participant.get("entity") or participant.get("name") or "",
            "role": participant.get("role") or "CONTEXT",
            "mention": {
                k: v
                for k, v in {
                    "sentence_id": _normalize_sentence_id(mention.get("sentence_id")),
                    "span_text": mention.get("span_text") or mention.get("text"),
                    "span": _normalize_span(mention.get("span")) if isinstance(mention.get("span"), dict) else None,
                }.items()
                if v not in (None, {}, "")
            },
        }
        out.append(item)
    return out


def _normalize_time(time_value: Any) -> dict[str, Any]:
    time_value = _as_dict(time_value)
    return {
        "kind": time_value.get("kind") or "UNKNOWN",
        "raw_span": time_value.get("raw_span") or "",
        "normalized": time_value.get("normalized"),
        **({"anchor_event_id": time_value.get("anchor_event_id")} if time_value.get("anchor_event_id") is not None else {}),
        **({"relation": time_value.get("relation")} if time_value.get("relation") is not None else {}),
        "confidence": float(time_value.get("confidence", 0.0) or 0.0),
    }


def _normalize_evidence(evidence: Any) -> dict[str, Any]:
    evidence = _as_dict(evidence)
    sentence_ids = [_normalize_sentence_id(sid) for sid in _as_list(evidence.get("sentence_ids")) if sid is not None]
    snippets = _normalize_snippets(evidence.get("snippets"))
    if not snippets and evidence.get("snippet"):
        snippets = _normalize_snippets(
            {
                "text": evidence.get("snippet"),
                "sentence_id": sentence_ids[0] if sentence_ids else None,
            }
        )
    return {
        "sentence_ids": sentence_ids,
        "snippets": snippets,
    }


def _normalize_events(events: list[Any]) -> list[dict[str, Any]]:
    out = []
    for idx, event in enumerate(_as_list(events), start=1):
        if not isinstance(event, dict):
            continue
        trigger = _as_dict(event.get("trigger"))
        out.append(
            {
                "event_id": str(event.get("event_id") or f"EV{idx}"),
                "event_type": event.get("event_type") or "UNKNOWN_EVENT",
                "main_verb": event.get("main_verb") or trigger.get("span_text") or "",
                "trigger": {
                    k: v
                    for k, v in {
                        "sentence_id": _normalize_sentence_id(trigger.get("sentence_id")),
                        "span_text": trigger.get("span_text") or trigger.get("text"),
                        "span": _normalize_span(trigger.get("span")) if isinstance(trigger.get("span"), dict) else None,
                    }.items()
                    if v not in (None, {}, "")
                },
                "participants": _normalize_participants(event.get("participants")),
                "time": _normalize_time(event.get("time")),
                "evidence": _normalize_evidence(event.get("evidence")),
                "confidence": float(event.get("confidence", 0.0) or 0.0),
            }
        )
    return out


def _normalize_edges(edges: Any) -> list[dict[str, Any]]:
    out = []
    for edge in _as_list(edges):
        if not isinstance(edge, dict):
            continue
        out.append(
            {
                "from_event": edge.get("from_event") or "",
                "to_event": edge.get("to_event") or "",
                "relation": edge.get("relation") or "",
                "evidence": _normalize_evidence(edge.get("evidence")),
                "confidence": float(edge.get("confidence", 0.0) or 0.0),
            }
        )
    return out


def _normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "entities": _normalize_entities(payload.get("entities")),
        "events": _normalize_events(payload.get("events")),
        "temporal_edges": _normalize_edges(payload.get("temporal_edges")),
        "causal_edges": _normalize_edges(payload.get("causal_edges")),
    }


def _build_structures(graph: dict[str, Any]) -> list[dict[str, Any]]:
    structures = []
    temporal = graph.get("temporal_edges") or []
    causal = graph.get("causal_edges") or []
    for event in graph.get("events") or []:
        event_id = event.get("event_id")
        structures.append(
            {
                "structure_id": f"STR_{event_id}",
                "event_id": event_id,
                "event_type": event.get("event_type"),
                "main_verb": event.get("main_verb"),
                "entity_ids": [p.get("entity_id") for p in event.get("participants") or [] if p.get("entity_id")],
                "temporal_links": [deepcopy(edge) for edge in temporal if edge.get("from_event") == event_id or edge.get("to_event") == event_id],
                "causal_links": [deepcopy(edge) for edge in causal if edge.get("from_event") == event_id or edge.get("to_event") == event_id],
                "evidence_sentence_ids": list(event.get("evidence", {}).get("sentence_ids") or []),
            }
        )
    return structures


def _build_render_graph(graph: dict[str, Any]) -> dict[str, Any]:
    nodes = []
    edges = []
    seen_nodes = set()

    for entity in graph.get("entities") or []:
        node_id = f"entity:{entity.get('entity_id')}"
        if node_id not in seen_nodes:
            seen_nodes.add(node_id)
            nodes.append(
                {
                    "id": node_id,
                    "node_type": "entity",
                    "label": entity.get("name") or entity.get("entity_id"),
                    "properties": {
                        "entity_id": entity.get("entity_id"),
                        "kind": entity.get("kind"),
                        "canonical_role": entity.get("canonical_role"),
                        "aliases": entity.get("aliases") or [],
                    },
                }
            )

    for event in graph.get("events") or []:
        event_node_id = f"event:{event.get('event_id')}"
        if event_node_id not in seen_nodes:
            seen_nodes.add(event_node_id)
            nodes.append(
                {
                    "id": event_node_id,
                    "node_type": "event",
                    "label": event.get("event_type") or event.get("event_id"),
                    "properties": {
                        "event_id": event.get("event_id"),
                        "trigger_text": event.get("trigger", {}).get("span_text"),
                        "time": event.get("time"),
                        "evidence_sentence_ids": event.get("evidence", {}).get("sentence_ids") or [],
                    },
                }
            )
        verb = event.get("main_verb") or event.get("trigger", {}).get("span_text") or ""
        if verb:
            verb_node_id = f"verb:{verb}"
            if verb_node_id not in seen_nodes:
                seen_nodes.add(verb_node_id)
                nodes.append(
                    {
                        "id": verb_node_id,
                        "node_type": "verb",
                        "label": verb,
                        "properties": {"normalized": verb},
                    }
                )
            edges.append(
                {
                    "source": event_node_id,
                    "target": verb_node_id,
                    "edge_type": "HAS_VERB",
                    "label": "HAS_VERB",
                    "properties": {},
                }
            )
        for participant in event.get("participants") or []:
            if not participant.get("entity_id"):
                continue
            edges.append(
                {
                    "source": event_node_id,
                    "target": f"entity:{participant.get('entity_id')}",
                    "edge_type": "PARTICIPATES_IN",
                    "label": participant.get("role") or "CONTEXT",
                    "properties": {"role": participant.get("role") or "CONTEXT"},
                }
            )

    for edge in graph.get("temporal_edges") or []:
        edges.append(
            {
                "source": f"event:{edge.get('from_event')}",
                "target": f"event:{edge.get('to_event')}",
                "edge_type": "TEMPORAL",
                "label": edge.get("relation") or "TEMPORAL",
                "properties": {
                    "relation": edge.get("relation"),
                    "evidence_sentence_ids": edge.get("evidence", {}).get("sentence_ids") or [],
                },
            }
        )

    for edge in graph.get("causal_edges") or []:
        edges.append(
            {
                "source": f"event:{edge.get('from_event')}",
                "target": f"event:{edge.get('to_event')}",
                "edge_type": "CAUSAL",
                "label": edge.get("relation") or "CAUSAL",
                "properties": {
                    "relation": edge.get("relation"),
                    "evidence_sentence_ids": edge.get("evidence", {}).get("sentence_ids") or [],
                },
            }
        )

    return {"nodes": nodes, "edges": edges}


def _make_client(model_name: str | None) -> tuple[OpenAI, str, float]:
    cfg = get_llm_config()
    api_key = cfg.get("api_key")
    api_base = cfg.get("api_base")
    if not api_key:
        raise ValueError("Missing AGENT_API_KEY / llm.api_key in config")
    model = model_name or cfg.get("model") or "gpt4o"
    model_lower = model.lower()
    temperature = 1.0 if "gpt-5" in model_lower or model_lower == "gpt5" else float(cfg.get("temperature", 0.0) or 0.0)
    return OpenAI(api_key=api_key, base_url=api_base, timeout=120.0), model, temperature


def _is_transient_transport_error(exc: Exception) -> bool:
    text = str(exc).lower()
    markers = (
        "504",
        "gateway time-out",
        "gateway timeout",
        "timed out",
        "timeout",
        "connection error",
        "502",
        "503",
        "rate limit",
        "temporarily unavailable",
    )
    return any(marker in text for marker in markers)


def _looks_truncated_json_response(raw: str, parse_error: str | None) -> bool:
    text = (raw or "").strip()
    if not text:
        return False
    lowered = (parse_error or "").lower()
    if not any(marker in lowered for marker in TRUNCATION_PARSE_MARKERS):
        return False
    if text.endswith("```"):
        return False
    stripped = re.sub(r"```(?:json)?\s*", "", text)
    tail = stripped.rstrip()
    if not tail:
        return False
    return tail[-1] in {",", ":", "{", "["} or not tail.endswith(("}", "]"))


def _request_json(
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    max_completion_tokens: int | None,
    enforce_json_response_format: bool,
) -> str:
    token_budget = max_completion_tokens
    if token_budget:
        token_budget = min(int(token_budget), 4500)

    last_exc: Exception | None = None
    for transport_attempt in range(3):
        request = {"model": model, "messages": messages, "temperature": temperature}
        if token_budget:
            request["max_completion_tokens"] = token_budget
        if enforce_json_response_format:
            request["response_format"] = {"type": "json_object"}
        try:
            response = client.chat.completions.create(**request)
            return (response.choices[0].message.content or "").strip()
        except TypeError:
            request.pop("response_format", None)
            response = client.chat.completions.create(**request)
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            last_exc = exc
            exc_text = str(exc).lower()
            if "only temperature=1 is supported" in exc_text or "temperature=0.0" in exc_text:
                request["temperature"] = 1.0
                response = client.chat.completions.create(**request)
                return (response.choices[0].message.content or "").strip()
            if enforce_json_response_format and "response_format" in exc_text:
                request.pop("response_format", None)
                response = client.chat.completions.create(**request)
                return (response.choices[0].message.content or "").strip()
            if token_budget and ("max_completion_tokens" in exc_text or "unknown parameter" in exc_text):
                request.pop("max_completion_tokens", None)
                response = client.chat.completions.create(**request)
                return (response.choices[0].message.content or "").strip()
            if _is_transient_transport_error(exc) and transport_attempt < 2:
                time.sleep(2 ** transport_attempt)
                if token_budget:
                    token_budget = max(1200, int(token_budget * 0.7))
                continue
            raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("LLM request failed without returning a response")


def _extract_chunk(
    client: OpenAI,
    model: str,
    temperature: float,
    chunk: dict[str, Any],
    case_name: str,
    docket_number: str,
    max_events_per_chunk: int,
    max_completion_tokens: int | None,
    max_parse_retries: int,
    max_empty_retries: int,
    include_raw_llm: bool,
    enforce_json_response_format: bool,
) -> dict[str, Any]:
    parse_attempt = 0
    empty_attempt = 0
    raw = ""
    payload_keys: list[str] = []
    parse_error = None
    max_events_default = max_events_per_chunk
    max_completion_tokens_default = max_completion_tokens

    while True:
        retry_suffix = ""
        max_events_this_attempt = max_events_default
        max_completion_tokens_this_attempt = max_completion_tokens_default
        if parse_attempt > 0:
            retry_suffix = PARSE_RETRY_SUFFIX
            if _looks_truncated_json_response(raw, parse_error):
                max_events_this_attempt = max(3, min(max_events_default, max_events_default - 1))
                if max_completion_tokens_this_attempt:
                    max_completion_tokens_this_attempt = max(max_completion_tokens_this_attempt, 4200)
                retry_suffix = (
                    retry_suffix.strip()
                    + "\nReturn a more compact JSON object. Keep only the most important entities and events, keep evidence snippets very short,"
                    " use at most one short snippet per event or edge, and do not include optional explanatory fields."
                )
        elif empty_attempt > 0:
            retry_suffix = EMPTY_RETRY_SUFFIX
        messages = build_event_graph_messages(
            text=chunk.get("text") or "",
            case_name=case_name,
            docket_number=docket_number,
            chunk_id=chunk.get("chunk_id") or "chunk_0",
            theme=chunk.get("theme") or "",
            max_events_per_chunk=max_events_this_attempt,
            retry_suffix=retry_suffix,
        )
        use_response_format = enforce_json_response_format and parse_attempt == 0 and empty_attempt == 0
        try:
            raw = _request_json(
                client=client,
                model=model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_completion_tokens_this_attempt,
                enforce_json_response_format=use_response_format,
            )
        except Exception as exc:
            return {
                **chunk,
                "graph": {"entities": [], "events": [], "temporal_edges": [], "causal_edges": []},
                "extraction": {"entities": [], "events": [], "temporal_edges": [], "causal_edges": []},
                "structures": [],
                "render_graph": {"nodes": [], "edges": []},
                "llm_payload_keys": payload_keys,
                **({"raw_llm_response": raw} if include_raw_llm else {}),
                "parse_error": str(exc),
            }

        payload, payload_keys, parse_error = _extract_payload(raw)
        if parse_error:
            if _looks_truncated_json_response(raw, parse_error) and parse_attempt < max_parse_retries:
                parse_attempt += 1
                continue
            if parse_attempt < max_parse_retries:
                parse_attempt += 1
                continue
            return {
                **chunk,
                "graph": {"entities": [], "events": [], "temporal_edges": [], "causal_edges": []},
                "extraction": {"entities": [], "events": [], "temporal_edges": [], "causal_edges": []},
                "structures": [],
                "render_graph": {"nodes": [], "edges": []},
                "llm_payload_keys": payload_keys,
                **({"raw_llm_response": raw} if include_raw_llm else {}),
                "parse_error": parse_error,
                "parse_retries_used": parse_attempt,
                "empty_retries_used": empty_attempt,
            }

        graph = _normalize_payload(payload)
        if not any(graph.get(name) for name in ("entities", "events", "temporal_edges", "causal_edges")):
            if empty_attempt < max_empty_retries:
                empty_attempt += 1
                continue
            result = {
                **chunk,
                "graph": graph,
                "extraction": deepcopy(graph),
                "structures": [],
                "render_graph": {"nodes": [], "edges": []},
                "llm_payload_keys": payload_keys,
                **({"raw_llm_response": raw} if include_raw_llm else {}),
                "extraction_warning": "LLM returned valid JSON but no structured items were extracted.",
                "parse_retries_used": parse_attempt,
                "empty_retries_used": empty_attempt,
            }
            return result

        structures = _build_structures(graph)
        render_graph = _build_render_graph(graph)
        return {
            **chunk,
            "graph": graph,
            "extraction": deepcopy(graph),
            "structures": structures,
            "render_graph": render_graph,
            "llm_payload_keys": payload_keys,
            **({"raw_llm_response": raw} if include_raw_llm else {}),
            "parse_retries_used": parse_attempt,
            "empty_retries_used": empty_attempt,
        }


def _iter_docs(path: str):
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuilt chunk-level event-graph extraction.")
    parser.add_argument("--input", required=True, help="Input chunk JSONL, e.g. events_llm_chunks.jsonl")
    parser.add_argument("--output", required=True, help="Output JSONL for extracted chunk event graphs")
    parser.add_argument("--model", default=None, help="Override extraction model")
    parser.add_argument("--doc-id", default=None, help="Process only this doc_id")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--max-chunks-per-doc", type=int, default=None)
    parser.add_argument("--max-chunks-total", type=int, default=None)
    parser.add_argument("--max-events-per-chunk", type=int, default=8)
    parser.add_argument("--max-completion-tokens", type=int, default=7000)
    parser.add_argument("--max-parse-retries", type=int, default=1)
    parser.add_argument("--max-empty-retries", type=int, default=1)
    parser.add_argument("--include-raw-llm", action="store_true")
    parser.add_argument("--no-response-format", action="store_true", help="Disable response_format=json_object")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    client, model, temperature = _make_client(args.model)

    total_chunks = 0
    written_docs = 0
    generated_at = datetime.now(timezone.utc).isoformat()

    with open(output_path, "w") as out_fh:
        for doc in _iter_docs(args.input):
            if args.doc_id and doc.get("doc_id") != args.doc_id:
                continue
            if args.max_docs is not None and written_docs >= args.max_docs:
                break
            chunks = list(doc.get("chunks") or [])
            if args.max_chunks_per_doc is not None:
                chunks = chunks[: args.max_chunks_per_doc]
            if args.max_chunks_total is not None:
                remaining = args.max_chunks_total - total_chunks
                if remaining <= 0:
                    break
                chunks = chunks[:remaining]
            out_chunks = []
            for chunk in chunks:
                out_chunks.append(
                    _extract_chunk(
                        client=client,
                        model=model,
                        temperature=temperature,
                        chunk=chunk,
                        case_name=doc.get("case_name") or doc.get("title") or doc.get("doc_id") or "",
                        docket_number=doc.get("docket_number") or "",
                        max_events_per_chunk=args.max_events_per_chunk,
                        max_completion_tokens=args.max_completion_tokens,
                        max_parse_retries=args.max_parse_retries,
                        max_empty_retries=args.max_empty_retries,
                        include_raw_llm=args.include_raw_llm,
                        enforce_json_response_format=not args.no_response_format,
                    )
                )
                total_chunks += 1
            out_doc = {
                "doc_id": doc.get("doc_id") or doc.get("case_id") or doc.get("title") or f"doc_{written_docs}",
                "model": model,
                "generated_at_utc": generated_at,
                "num_chunks": len(out_chunks),
                "chunks": out_chunks,
            }
            out_fh.write(json.dumps(out_doc, default=str) + "\n")
            written_docs += 1
            if args.max_chunks_total is not None and total_chunks >= args.max_chunks_total:
                break


if __name__ == "__main__":
    main()
