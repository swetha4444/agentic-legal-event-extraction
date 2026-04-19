#!/usr/bin/env python3
"""Core evaluator and optimizer for rebuilt two-stage GEPA chunk event graphs."""
import copy
import hashlib
import json
import math
import re
import string
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rebuilt_event_graphs.extract_chunk_event_graphs_rebuilt import (
    _build_render_graph,
    _build_structures,
    _extract_payload,
    _make_client,
    _normalize_payload,
)
from rebuilt_two_stage_gepa_event_graphs.prompts import (
    build_judge_messages,
    build_reflection_messages,
    build_stage1_messages,
    build_stage2_messages,
)
from rebuilt_two_stage_gepa_event_graphs.coverage_enforcer import enforce_verb_sentence_coverage

DEFAULT_MIN_SCHEMA_VALIDITY = 0.8
DEFAULT_MIN_NON_EMPTY_RATE = 0.8
DEFAULT_MIN_JUDGE_SCORE_MEAN = 0.5
DEFAULT_FRONTIER_PARENT_LIMIT = 3
DEFAULT_PLATEAU_PATIENCE = 3
DEFAULT_MIN_QUALITY_IMPROVEMENT = 0.005
TRUNCATION_PARSE_MARKERS = (
    "expecting value",
    "unterminated string",
    "expecting ',' delimiter",
    "expecting property name enclosed in double quotes",
    "unterminated object",
    "unterminated array",
)


@dataclass
class EvalArtifacts:
    metrics: dict[str, Any]
    predictions_docs: list[dict[str, Any]]
    asi_text: str
    judge_details: list[dict[str, Any]]
    trace_bundle: dict[str, Any]
    predictions_path: Path
    metrics_path: Path
    asi_path: Path
    judge_details_path: Path


def load_candidate(path: str) -> dict[str, Any]:
    with open(path) as fh:
        candidate = json.load(fh)
    return sanitize_candidate(candidate)


def _escape_literal_braces_in_template(template: str, allowed_fields: set[str]) -> str:
    text = str(template or "")
    escaped = text.replace("{", "{{").replace("}", "}}")
    formatter = string.Formatter()
    for literal_text, field_name, format_spec, conversion in formatter.parse(text):
        if field_name is None:
            continue
        normalized_field = str(field_name).strip()
        if normalized_field not in allowed_fields:
            continue
        original = "{" + normalized_field
        if conversion:
            original += "!" + conversion
        if format_spec:
            original += ":" + format_spec
        original += "}"
        escaped_original = original.replace("{", "{{").replace("}", "}}")
        escaped = escaped.replace(escaped_original, original)
    return escaped


def sanitize_candidate(candidate: dict[str, Any], *, default_name: str | None = None) -> dict[str, Any]:
    candidate = copy.deepcopy(candidate or {})
    candidate.setdefault("name", default_name or "two_stage_event_graph_candidate")
    legacy_model = candidate.get("model")
    candidate.setdefault("stage1_model", legacy_model or "claude-sonnet-4-5")
    candidate.setdefault("stage2_model", legacy_model or candidate.get("stage1_model") or "claude-sonnet-4-5")
    candidate.setdefault("stage1_system_prompt", candidate.get("system_prompt", ""))
    candidate.setdefault("stage1_user_prompt_template", candidate.get("user_prompt_template", ""))
    candidate.setdefault("stage2_system_prompt", "")
    candidate.setdefault("stage2_user_prompt_template", "")
    retry = candidate.setdefault("retry_suffix_templates", {})
    if not isinstance(retry, dict):
        retry = {}
        candidate["retry_suffix_templates"] = retry
    retry.setdefault("stage1_parse_retry", retry.get("parse_retry", ""))
    retry.setdefault("stage1_empty_retry", retry.get("empty_retry", ""))
    retry.setdefault("stage2_parse_retry", "")
    retry.setdefault("stage2_empty_retry", "")
    cfg = candidate.setdefault("config", {})
    if not isinstance(cfg, dict):
        cfg = {}
        candidate["config"] = cfg
    cfg.setdefault("max_events_per_chunk", 8)
    cfg.setdefault("stage1_max_completion_tokens", cfg.get("max_completion_tokens", 2600))
    cfg.setdefault("stage2_max_completion_tokens", cfg.get("max_completion_tokens", 2600))
    cfg.setdefault("enforce_json_response_format", True)
    cfg.setdefault("max_stage1_parse_retries", cfg.get("max_parse_retries", 1))
    cfg.setdefault("max_stage1_empty_retries", cfg.get("max_empty_retries", 1))
    cfg.setdefault("max_stage2_parse_retries", 1)
    cfg.setdefault("max_stage2_empty_retries", 1)
    cfg.setdefault("judge_max_completion_tokens", 1200)
    cfg.setdefault("enforce_verb_sentence_coverage_postprocess", False)
    cfg.setdefault("fallback_event_type", "SUPPORTING_FACT")
    cfg.setdefault("fallback_event_confidence", 0.35)
    candidate["stage1_user_prompt_template"] = _escape_literal_braces_in_template(
        candidate.get("stage1_user_prompt_template") or "",
        {"case_name", "docket_number", "chunk_id", "theme", "max_events_per_chunk", "text"},
    )
    candidate["stage2_user_prompt_template"] = _escape_literal_braces_in_template(
        candidate.get("stage2_user_prompt_template") or "",
        {"case_name", "docket_number", "chunk_id", "theme", "text", "intermediate_json"},
    )
    candidate["model"] = candidate.get("stage2_model")
    return candidate


def candidate_hash(candidate: dict[str, Any]) -> str:
    payload = json.dumps(candidate, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value or "candidate")
    return value.strip("_") or "candidate"


def save_candidate_json(candidate: dict[str, Any], candidates_dir: Path) -> Path:
    candidates_dir.mkdir(parents=True, exist_ok=True)
    name = safe_name(candidate.get("name") or "candidate")
    path = candidates_dir / f"{candidate_hash(candidate)}_{name}.json"
    if not path.exists():
        path.write_text(json.dumps(candidate, indent=2))
    return path


def _iter_docs(path: str, max_docs: int | None = None, max_total_chunks: int | None = None) -> list[dict[str, Any]]:
    docs = []
    total_chunks = 0
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            chunks = list(doc.get("chunks") or [])
            if max_total_chunks is not None:
                remaining = max_total_chunks - total_chunks
                if remaining <= 0:
                    break
                chunks = chunks[:remaining]
            doc = copy.deepcopy(doc)
            doc["chunks"] = chunks
            docs.append(doc)
            total_chunks += len(chunks)
            if max_docs is not None and len(docs) >= max_docs:
                break
            if max_total_chunks is not None and total_chunks >= max_total_chunks:
                break
    return docs


def _usage_field(usage: Any, *names: str) -> int:
    for name in names:
        if isinstance(usage, dict) and usage.get(name) is not None:
            try:
                return int(usage.get(name) or 0)
            except (TypeError, ValueError):
                return 0
        value = getattr(usage, name, None)
        if value is not None:
            try:
                return int(value or 0)
            except (TypeError, ValueError):
                return 0
    return 0


def _extract_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    prompt_tokens = _usage_field(usage, "prompt_tokens", "input_tokens")
    completion_tokens = _usage_field(usage, "completion_tokens", "output_tokens")
    total_tokens = _usage_field(usage, "total_tokens")
    if total_tokens == 0 and (prompt_tokens or completion_tokens):
        total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _extract_response_text(response: Any) -> str:
    try:
        content = response.choices[0].message.content
    except Exception:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            else:
                text = getattr(item, "text", None)
                if text:
                    parts.append(str(text))
        return "\n".join(part for part in parts if part).strip()
    return str(content or "").strip()


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


def _request_json_with_stats(
    *,
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    max_completion_tokens: int | None,
    enforce_json_response_format: bool,
    request_kind: str,
) -> tuple[str, dict[str, Any]]:
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
        request_start = time.perf_counter()
        try:
            response = client.chat.completions.create(**request)
        except TypeError:
            request.pop("response_format", None)
            response = client.chat.completions.create(**request)
        except Exception as exc:
            last_exc = exc
            exc_text = str(exc).lower()
            if "only temperature=1 is supported" in exc_text or "temperature=0.0" in exc_text:
                request["temperature"] = 1.0
                response = client.chat.completions.create(**request)
            elif enforce_json_response_format and "response_format" in exc_text:
                request.pop("response_format", None)
                response = client.chat.completions.create(**request)
            elif token_budget and ("max_completion_tokens" in exc_text or "unknown parameter" in exc_text):
                request.pop("max_completion_tokens", None)
                response = client.chat.completions.create(**request)
            elif _is_transient_transport_error(exc) and transport_attempt < 2:
                time.sleep(2 ** transport_attempt)
                if token_budget:
                    token_budget = max(800, int(token_budget * 0.7))
                continue
            else:
                raise
        latency_seconds = time.perf_counter() - request_start
        usage = _extract_usage(response)
        text = _extract_response_text(response)
        return text, {
            "request_kind": request_kind,
            "transport_attempt": transport_attempt,
            "latency_seconds": round(latency_seconds, 6),
            "used_response_format": "response_format" in request,
            "used_max_completion_tokens": request.get("max_completion_tokens"),
            "temperature": request.get("temperature"),
            **usage,
        }
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("LLM request failed without returning a response")


def _strip_internal_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_internal_fields(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, list):
        return [_strip_internal_fields(item) for item in value]
    return value


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


def _unwrap_single_nested_object(payload: dict[str, Any]) -> dict[str, Any]:
    current = payload
    for _ in range(3):
        if not isinstance(current, dict) or len(current) != 1:
            break
        only_value = next(iter(current.values()))
        if not isinstance(only_value, dict):
            break
        current = only_value
    return current if isinstance(current, dict) else payload


def _empty_graph() -> dict[str, Any]:
    return {"entities": [], "events": [], "temporal_edges": [], "causal_edges": []}


def _graph_has_signal(graph: dict[str, Any], active_keys: tuple[str, ...]) -> bool:
    return any((graph.get(name) or []) for name in active_keys)


def _merge_stage_graphs(stage1_graph: dict[str, Any], stage2_graph: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(stage2_graph or _empty_graph())
    if not merged.get("entities") and stage1_graph.get("entities"):
        merged["entities"] = copy.deepcopy(stage1_graph.get("entities") or [])
    if not merged.get("events") and stage1_graph.get("events"):
        merged["events"] = copy.deepcopy(stage1_graph.get("events") or [])
    merged.setdefault("temporal_edges", [])
    merged.setdefault("causal_edges", [])
    return merged


def _coerce_two_stage_payload(payload: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(payload or {})
    for edge_key in ("temporal_edges", "causal_edges"):
        coerced_edges = []
        for edge in payload.get(edge_key) or []:
            if not isinstance(edge, dict):
                continue
            coerced = copy.deepcopy(edge)
            if not coerced.get("from_event"):
                coerced["from_event"] = (
                    coerced.get("source_event_id")
                    or coerced.get("source")
                    or coerced.get("from")
                    or ""
                )
            if not coerced.get("to_event"):
                coerced["to_event"] = (
                    coerced.get("target_event_id")
                    or coerced.get("target")
                    or coerced.get("to")
                    or ""
                )
            coerced_edges.append(coerced)
        if edge_key in payload:
            payload[edge_key] = coerced_edges
    return payload


def _run_extraction_stage(
    *,
    client: Any,
    model: str,
    temperature: float,
    stage_name: str,
    build_messages_fn: Any,
    candidate: dict[str, Any],
    doc: dict[str, Any],
    chunk: dict[str, Any],
    include_raw_llm: bool,
    active_keys: tuple[str, ...],
    max_events_per_chunk: int,
    max_completion_tokens: int,
    max_parse_retries: int,
    max_empty_retries: int,
    parse_retry_text: str,
    empty_retry_text: str,
    system_prompt: str,
    user_prompt_template: str,
    stage_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stage_kwargs = stage_kwargs or {}
    parse_attempt = 0
    empty_attempt = 0
    raw = ""
    payload_keys: list[str] = []
    parse_error: str | None = None
    call_records: list[dict[str, Any]] = []

    while True:
        retry_suffix = ""
        max_events_this_attempt = max_events_per_chunk
        max_completion_tokens_this_attempt = max_completion_tokens
        if parse_attempt > 0:
            retry_suffix = parse_retry_text or ""
            if _looks_truncated_json_response(raw, parse_error):
                max_events_this_attempt = max(3, min(max_events_per_chunk, max_events_per_chunk - 1))
                max_completion_tokens_this_attempt = max(max_completion_tokens_this_attempt, 4200)
                retry_suffix = (
                    retry_suffix.rstrip()
                    + "\nReturn a more compact JSON object. Keep only the most important grounded items,"
                    " keep evidence snippets extremely short, and avoid optional fields."
                ).strip()
        elif empty_attempt > 0:
            retry_suffix = empty_retry_text or ""

        messages = build_messages_fn(
            text=chunk.get("text") or "",
            case_name=doc.get("case_name") or doc.get("title") or doc.get("doc_id") or "",
            docket_number=doc.get("docket_number") or "",
            chunk_id=chunk.get("chunk_id") or "chunk_0",
            theme=chunk.get("theme") or "",
            max_events_per_chunk=max_events_this_attempt,
            system_prompt=system_prompt or None,
            user_prompt_template=user_prompt_template or None,
            retry_suffix=retry_suffix,
            **stage_kwargs,
        )
        use_response_format = bool((candidate.get("config") or {}).get("enforce_json_response_format", True)) and parse_attempt == 0 and empty_attempt == 0

        try:
            raw, call_stats = _request_json_with_stats(
                client=client,
                model=model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_completion_tokens_this_attempt,
                enforce_json_response_format=use_response_format,
                request_kind=stage_name,
            )
            call_stats["stage"] = stage_name
            call_stats["parse_attempt"] = parse_attempt
            call_stats["empty_attempt"] = empty_attempt
            call_records.append(call_stats)
        except Exception as exc:
            return {
                "stage_name": stage_name,
                "graph": _empty_graph(),
                "payload_keys": payload_keys,
                "raw": raw,
                "error": str(exc),
                "warning": None,
                "parse_retries_used": parse_attempt,
                "empty_retries_used": empty_attempt,
                "call_records": call_records,
                "final_status": "exception",
            }

        payload, payload_keys, parse_error = _extract_payload(raw)
        if parse_error:
            if parse_attempt < max_parse_retries:
                parse_attempt += 1
                continue
            return {
                "stage_name": stage_name,
                "graph": _empty_graph(),
                "payload_keys": payload_keys,
                "raw": raw,
                "error": parse_error,
                "warning": None,
                "parse_retries_used": parse_attempt,
                "empty_retries_used": empty_attempt,
                "call_records": call_records,
                "final_status": "parse_error",
            }

        graph = _normalize_payload(_coerce_two_stage_payload(payload))
        if not _graph_has_signal(graph, active_keys):
            if empty_attempt < max_empty_retries:
                empty_attempt += 1
                continue
            return {
                "stage_name": stage_name,
                "graph": graph,
                "payload_keys": payload_keys,
                "raw": raw,
                "error": None,
                "warning": "LLM returned valid JSON but no structured items were extracted.",
                "parse_retries_used": parse_attempt,
                "empty_retries_used": empty_attempt,
                "call_records": call_records,
                "final_status": "empty_valid_json",
            }

        return {
            "stage_name": stage_name,
            "graph": graph,
            "payload_keys": payload_keys,
            "raw": raw,
            "error": None,
            "warning": None,
            "parse_retries_used": parse_attempt,
            "empty_retries_used": empty_attempt,
            "call_records": call_records,
            "final_status": "ok",
        }


def _extract_chunk_with_candidate(
    chunk: dict[str, Any],
    doc: dict[str, Any],
    candidate: dict[str, Any],
    include_raw_llm: bool,
) -> dict[str, Any]:
    config = candidate.get("config") or {}
    retry_templates = candidate.get("retry_suffix_templates") or {}
    max_events = int(config.get("max_events_per_chunk", 8) or 8)

    stage1_client, stage1_model, stage1_temperature = _make_client(candidate.get("stage1_model"))
    stage2_client, stage2_model, stage2_temperature = _make_client(candidate.get("stage2_model"))

    stage1 = _run_extraction_stage(
        client=stage1_client,
        model=stage1_model,
        temperature=stage1_temperature,
        stage_name="stage1_extract",
        build_messages_fn=build_stage1_messages,
        candidate=candidate,
        doc=doc,
        chunk=chunk,
        include_raw_llm=include_raw_llm,
        active_keys=("entities", "events"),
        max_events_per_chunk=max_events,
        max_completion_tokens=int(config.get("stage1_max_completion_tokens", 2600) or 2600),
        max_parse_retries=int(config.get("max_stage1_parse_retries", 1) or 1),
        max_empty_retries=int(config.get("max_stage1_empty_retries", 1) or 1),
        parse_retry_text=retry_templates.get("stage1_parse_retry") or "",
        empty_retry_text=retry_templates.get("stage1_empty_retry") or "",
        system_prompt=candidate.get("stage1_system_prompt") or "",
        user_prompt_template=candidate.get("stage1_user_prompt_template") or "",
    )
    stage1_graph = stage1.get("graph") or _empty_graph()
    stage1_graph["temporal_edges"] = []
    stage1_graph["causal_edges"] = []

    if stage1.get("error"):
        return {
            **chunk,
            "graph": _empty_graph(),
            "extraction": _empty_graph(),
            "intermediate_graph": stage1_graph,
            "structures": [],
            "render_graph": {"nodes": [], "edges": []},
            "llm_payload_keys": [],
            **(
                {
                    "raw_llm_response": stage1.get("raw", ""),
                    "stage1_raw_llm_response": stage1.get("raw", ""),
                    "stage2_raw_llm_response": "",
                }
                if include_raw_llm
                else {}
            ),
            "parse_error": f"stage1: {stage1['error']}",
            "parse_retries_used": int(stage1.get("parse_retries_used", 0) or 0),
            "empty_retries_used": int(stage1.get("empty_retries_used", 0) or 0),
            "stage1_parse_retries_used": int(stage1.get("parse_retries_used", 0) or 0),
            "stage1_empty_retries_used": int(stage1.get("empty_retries_used", 0) or 0),
            "stage2_parse_retries_used": 0,
            "stage2_empty_retries_used": 0,
            "_trace_meta": {
                "stage1_calls": stage1.get("call_records", []),
                "stage2_calls": [],
                "extraction_calls": stage1.get("call_records", []),
                "final_status": "stage1_error",
            },
        }

    if not _graph_has_signal(stage1_graph, ("entities", "events")):
        return {
            **chunk,
            "graph": _empty_graph(),
            "extraction": _empty_graph(),
            "intermediate_graph": stage1_graph,
            "structures": [],
            "render_graph": {"nodes": [], "edges": []},
            "llm_payload_keys": [],
            **(
                {
                    "raw_llm_response": stage1.get("raw", ""),
                    "stage1_raw_llm_response": stage1.get("raw", ""),
                    "stage2_raw_llm_response": "",
                }
                if include_raw_llm
                else {}
            ),
            "extraction_warning": stage1.get("warning") or "Stage 1 returned no grounded structure.",
            "parse_retries_used": int(stage1.get("parse_retries_used", 0) or 0),
            "empty_retries_used": int(stage1.get("empty_retries_used", 0) or 0),
            "stage1_parse_retries_used": int(stage1.get("parse_retries_used", 0) or 0),
            "stage1_empty_retries_used": int(stage1.get("empty_retries_used", 0) or 0),
            "stage2_parse_retries_used": 0,
            "stage2_empty_retries_used": 0,
            "_trace_meta": {
                "stage1_calls": stage1.get("call_records", []),
                "stage2_calls": [],
                "extraction_calls": stage1.get("call_records", []),
                "final_status": "stage1_empty",
            },
        }

    stage2 = _run_extraction_stage(
        client=stage2_client,
        model=stage2_model,
        temperature=stage2_temperature,
        stage_name="stage2_graph",
        build_messages_fn=build_stage2_messages,
        candidate=candidate,
        doc=doc,
        chunk=chunk,
        include_raw_llm=include_raw_llm,
        active_keys=("entities", "events", "temporal_edges", "causal_edges"),
        max_events_per_chunk=max_events,
        max_completion_tokens=int(config.get("stage2_max_completion_tokens", 2600) or 2600),
        max_parse_retries=int(config.get("max_stage2_parse_retries", 1) or 1),
        max_empty_retries=int(config.get("max_stage2_empty_retries", 1) or 1),
        parse_retry_text=retry_templates.get("stage2_parse_retry") or "",
        empty_retry_text=retry_templates.get("stage2_empty_retry") or "",
        system_prompt=candidate.get("stage2_system_prompt") or "",
        user_prompt_template=candidate.get("stage2_user_prompt_template") or "",
        stage_kwargs={"intermediate_json": json.dumps(stage1_graph, ensure_ascii=True)},
    )

    if stage2.get("error"):
        return {
            **chunk,
            "graph": _empty_graph(),
            "extraction": _empty_graph(),
            "intermediate_graph": stage1_graph,
            "structures": [],
            "render_graph": {"nodes": [], "edges": []},
            "llm_payload_keys": stage1.get("payload_keys", []),
            **(
                {
                    "raw_llm_response": stage2.get("raw", ""),
                    "stage1_raw_llm_response": stage1.get("raw", ""),
                    "stage2_raw_llm_response": stage2.get("raw", ""),
                }
                if include_raw_llm
                else {}
            ),
            "parse_error": f"stage2: {stage2['error']}",
            "parse_retries_used": int(stage1.get("parse_retries_used", 0) or 0) + int(stage2.get("parse_retries_used", 0) or 0),
            "empty_retries_used": int(stage1.get("empty_retries_used", 0) or 0) + int(stage2.get("empty_retries_used", 0) or 0),
            "stage1_parse_retries_used": int(stage1.get("parse_retries_used", 0) or 0),
            "stage1_empty_retries_used": int(stage1.get("empty_retries_used", 0) or 0),
            "stage2_parse_retries_used": int(stage2.get("parse_retries_used", 0) or 0),
            "stage2_empty_retries_used": int(stage2.get("empty_retries_used", 0) or 0),
            "_trace_meta": {
                "stage1_calls": stage1.get("call_records", []),
                "stage2_calls": stage2.get("call_records", []),
                "extraction_calls": list(stage1.get("call_records", [])) + list(stage2.get("call_records", [])),
                "final_status": "stage2_error",
            },
        }

    final_graph = _merge_stage_graphs(stage1_graph, stage2.get("graph") or _empty_graph())
    coverage_enforcement = None
    if bool(config.get("enforce_verb_sentence_coverage_postprocess", False)):
        final_graph, coverage_enforcement = enforce_verb_sentence_coverage(
            final_graph,
            chunk.get("text") or "",
            fallback_event_type=str(config.get("fallback_event_type") or "SUPPORTING_FACT"),
            fallback_confidence=float(config.get("fallback_event_confidence", 0.35) or 0.35),
        )
    structures = _build_structures(final_graph)
    render_graph = _build_render_graph(final_graph)
    llm_payload_keys = {
        "stage1": stage1.get("payload_keys", []),
        "stage2": stage2.get("payload_keys", []),
    }
    extraction_warning = stage2.get("warning")
    return {
        **chunk,
        "graph": final_graph,
        "extraction": copy.deepcopy(final_graph),
        "intermediate_graph": stage1_graph,
        "structures": structures,
        "render_graph": render_graph,
        "llm_payload_keys": llm_payload_keys,
        **(
            {
                "raw_llm_response": stage2.get("raw", ""),
                "stage1_raw_llm_response": stage1.get("raw", ""),
                "stage2_raw_llm_response": stage2.get("raw", ""),
            }
            if include_raw_llm
            else {}
        ),
        **({"extraction_warning": extraction_warning} if extraction_warning else {}),
        **({"coverage_enforcement": coverage_enforcement} if coverage_enforcement else {}),
        "parse_retries_used": int(stage1.get("parse_retries_used", 0) or 0) + int(stage2.get("parse_retries_used", 0) or 0),
        "empty_retries_used": int(stage1.get("empty_retries_used", 0) or 0) + int(stage2.get("empty_retries_used", 0) or 0),
        "stage1_parse_retries_used": int(stage1.get("parse_retries_used", 0) or 0),
        "stage1_empty_retries_used": int(stage1.get("empty_retries_used", 0) or 0),
        "stage2_parse_retries_used": int(stage2.get("parse_retries_used", 0) or 0),
        "stage2_empty_retries_used": int(stage2.get("empty_retries_used", 0) or 0),
        "_trace_meta": {
            "stage1_calls": stage1.get("call_records", []),
            "stage2_calls": stage2.get("call_records", []),
            "extraction_calls": list(stage1.get("call_records", [])) + list(stage2.get("call_records", [])),
            "final_status": "ok",
        },
    }


def _graph_summary_for_judge(chunk: dict[str, Any]) -> str:
    graph = chunk.get("graph") or {}
    summary = {
        "entities": graph.get("entities") or [],
        "events": graph.get("events") or [],
        "temporal_edges": graph.get("temporal_edges") or [],
        "causal_edges": graph.get("causal_edges") or [],
    }
    text = json.dumps(summary, ensure_ascii=True)
    return text[:6000]


def _parse_judge_response(raw: str) -> tuple[dict[str, Any], str | None]:
    payload, _, error = _extract_payload(raw)
    if error:
        return {}, error
    if not payload:
        return {}, "Empty response"
    payload = _unwrap_single_nested_object(payload)
    required = ["factual_accuracy", "completeness", "relevance", "faithfulness", "coherence", "overall"]
    if not all(k in payload for k in required):
        return {}, "Missing required judge keys"
    scores = [int(payload[k]) for k in required]
    payload = copy.deepcopy(payload)
    reason = payload.get("reason", "")
    if isinstance(reason, list):
        reason = " ".join(str(item) for item in reason)
    elif isinstance(reason, dict):
        reason = json.dumps(reason, ensure_ascii=True)
    elif reason is None:
        reason = ""
    else:
        reason = str(reason)

    major_issues = payload.get("major_issues", [])
    if isinstance(major_issues, list):
        major_issues = [str(item) for item in major_issues if str(item).strip()]
    elif isinstance(major_issues, str):
        major_issues = [major_issues] if major_issues.strip() else []
    elif major_issues in (None, ""):
        major_issues = []
    else:
        major_issues = [str(major_issues)]

    payload["reason"] = reason
    payload["major_issues"] = major_issues
    payload["mean_score"] = sum(scores) / len(scores)
    payload["normalized_mean_score"] = payload["mean_score"] / 5.0
    return payload, None


def _judge_chunk(chunk: dict[str, Any], doc_id: str, model_name: str) -> dict[str, Any]:
    client, model, temperature = _make_client(model_name)
    judge_max_completion_tokens = int(((chunk.get("_candidate_config") or {}).get("judge_max_completion_tokens", 1200)) or 1200)
    messages = build_judge_messages(
        doc_id=doc_id,
        chunk_id=chunk.get("chunk_id") or "chunk_0",
        text=chunk.get("text") or "",
        graph_json=_graph_summary_for_judge(chunk),
    )
    try:
        raw, call_stats = _request_json_with_stats(
            client=client,
            model=model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=judge_max_completion_tokens,
            enforce_json_response_format=True,
            request_kind="judge",
        )
        payload, error = _parse_judge_response(raw)
        if error:
            return {
                "judge_error": error,
                "judge_raw_response": raw,
                "doc_id": doc_id,
                "chunk_id": chunk.get("chunk_id"),
                "_trace_meta": call_stats,
            }
        payload["judge_raw_response"] = raw
        payload["doc_id"] = doc_id
        payload["chunk_id"] = chunk.get("chunk_id")
        payload["_trace_meta"] = call_stats
        return payload
    except Exception as exc:
        return {
            "judge_error": str(exc),
            "judge_raw_response": "",
            "doc_id": doc_id,
            "chunk_id": chunk.get("chunk_id"),
            "_trace_meta": {
                "request_kind": "judge",
                "latency_seconds": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }


def _build_edges_for_cycle(graph_edges: list[dict[str, Any]], *, temporal: bool) -> list[tuple[str, str]]:
    pairs = []
    for edge in graph_edges:
        src = edge.get("from_event")
        dst = edge.get("to_event")
        rel = (edge.get("relation") or "").upper()
        if not src or not dst:
            continue
        if temporal:
            if rel == "BEFORE":
                pairs.append((src, dst))
            elif rel == "AFTER":
                pairs.append((dst, src))
        else:
            pairs.append((src, dst))
    return pairs


def _has_cycle(edges: list[tuple[str, str]]) -> bool:
    graph = defaultdict(list)
    nodes = set()
    for src, dst in edges:
        graph[src].append(dst)
        nodes.add(src)
        nodes.add(dst)
    temp = set()
    perm = set()

    def visit(node: str) -> bool:
        if node in perm:
            return False
        if node in temp:
            return True
        temp.add(node)
        for nxt in graph.get(node, []):
            if visit(nxt):
                return True
        temp.remove(node)
        perm.add(node)
        return False

    return any(visit(node) for node in list(nodes))


def _normalized_time_key(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        for item in value:
            normalized = _normalized_time_key(item)
            if normalized:
                return normalized
        return ""
    if isinstance(value, dict):
        for key in ("normalized", "value", "date", "start", "raw_span", "raw"):
            normalized = _normalized_time_key(value.get(key))
            if normalized:
                return normalized
        return ""
    return ""


def _temporal_date_consistent(chunk: dict[str, Any]) -> bool:
    events = {e.get("event_id"): e for e in (chunk.get("graph") or {}).get("events") or []}
    for edge in (chunk.get("graph") or {}).get("temporal_edges") or []:
        rel = (edge.get("relation") or "").upper()
        if rel not in {"BEFORE", "AFTER"}:
            continue
        a = events.get(edge.get("from_event"))
        b = events.get(edge.get("to_event"))
        if not a or not b:
            continue
        ta = _normalized_time_key((a.get("time") or {}).get("normalized"))
        tb = _normalized_time_key((b.get("time") or {}).get("normalized"))
        if not ta or not tb:
            continue
        if rel == "BEFORE" and ta > tb:
            return False
        if rel == "AFTER" and ta < tb:
            return False
    return True


def _sum_call_stats(records: list[dict[str, Any]]) -> dict[str, float]:
    prompt_tokens = sum(int(record.get("prompt_tokens", 0) or 0) for record in records)
    completion_tokens = sum(int(record.get("completion_tokens", 0) or 0) for record in records)
    total_tokens = sum(int(record.get("total_tokens", 0) or 0) for record in records)
    latency_seconds = sum(float(record.get("latency_seconds", 0.0) or 0.0) for record in records)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "latency_seconds": latency_seconds,
        "call_count": len(records),
    }


def _participant_ref_integrity_for_chunk(chunk: dict[str, Any]) -> float:
    graph = chunk.get("graph") or {}
    entity_ids = {e.get("entity_id") for e in graph.get("entities") or [] if e.get("entity_id")}
    total = 0
    valid = 0
    for event in graph.get("events") or []:
        for participant in event.get("participants") or []:
            entity_id = participant.get("entity_id")
            if not entity_id:
                continue
            total += 1
            if entity_id in entity_ids:
                valid += 1
    return 1.0 if total == 0 else valid / total


def _build_trace_bundle(
    predictions_docs: list[dict[str, Any]],
    judge_details: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    judge_map = {(row.get("doc_id"), row.get("chunk_id")): row for row in judge_details}
    traces = []
    for doc in predictions_docs:
        doc_id = doc.get("doc_id")
        for chunk in doc.get("chunks") or []:
            graph = chunk.get("graph") or {}
            intermediate_graph = chunk.get("intermediate_graph") or {}
            judge = judge_map.get((doc_id, chunk.get("chunk_id")))
            trace = {
                "doc_id": doc_id,
                "chunk_id": chunk.get("chunk_id"),
                "theme": chunk.get("theme"),
                "text_excerpt": (chunk.get("text") or "")[:700],
                "parse_error": chunk.get("parse_error"),
                "extraction_warning": chunk.get("extraction_warning"),
                "intermediate_entity_count": len(intermediate_graph.get("entities") or []),
                "intermediate_event_count": len(intermediate_graph.get("events") or []),
                "entity_count": len(graph.get("entities") or []),
                "event_count": len(graph.get("events") or []),
                "temporal_edge_count": len(graph.get("temporal_edges") or []),
                "causal_edge_count": len(graph.get("causal_edges") or []),
                "participant_ref_integrity": _participant_ref_integrity_for_chunk(chunk),
                "parse_retries_used": chunk.get("parse_retries_used", 0),
                "empty_retries_used": chunk.get("empty_retries_used", 0),
                "extraction_trace": copy.deepcopy((chunk.get("_trace_meta") or {}).get("extraction_calls") or []),
                "stage1_trace": copy.deepcopy((chunk.get("_trace_meta") or {}).get("stage1_calls") or []),
                "stage2_trace": copy.deepcopy((chunk.get("_trace_meta") or {}).get("stage2_calls") or []),
                "stage1_raw_llm_excerpt": (chunk.get("stage1_raw_llm_response") or "")[:1200],
                "stage2_raw_llm_excerpt": (chunk.get("stage2_raw_llm_response") or "")[:1200],
            }
            if judge:
                trace["judge"] = {
                    "judge_error": judge.get("judge_error"),
                    "overall": judge.get("overall"),
                    "normalized_mean_score": judge.get("normalized_mean_score"),
                    "reason": judge.get("reason"),
                    "major_issues": (judge.get("major_issues") or [])[:5],
                    "trace": copy.deepcopy(judge.get("_trace_meta") or {}),
                }
            priority = 3
            if chunk.get("parse_error"):
                priority = 0
            elif not graph.get("events"):
                priority = 1
            elif judge and judge.get("judge_error"):
                priority = 1
            elif judge and judge.get("normalized_mean_score") is not None and float(judge.get("normalized_mean_score")) < 0.6:
                priority = 2
            trace["_priority"] = priority
            traces.append(trace)

    traces.sort(key=lambda row: (row.get("_priority", 99), row.get("event_count", 0)))
    representative = [{k: v for k, v in trace.items() if k != "_priority"} for trace in traces[:6]]
    successes = [
        {k: v for k, v in trace.items() if k != "_priority"}
        for trace in traces
        if trace.get("_priority") >= 2 and not trace.get("parse_error") and trace.get("event_count", 0) > 0
    ][:2]
    return {
        "summary": {
            "run_name": metrics.get("run_name"),
            "candidate_name": metrics.get("candidate_name"),
            "schema_validity": metrics.get("schema_validity"),
            "non_empty_rate": metrics.get("non_empty_rate"),
            "structural_score": metrics.get("structural_score"),
            "judge_score_mean": metrics.get("judge_score_mean"),
            "quality_score": metrics.get("quality_score"),
            "cost_proxy_total": metrics.get("cost_proxy_total"),
        },
        "representative_traces": representative,
        "representative_successes": successes,
    }


def _truncate_text(text: Any, limit: int) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)].rstrip() + "..."


def _compact_asi_text(asi_text: str, *, max_lines: int = 14, max_chars: int = 2400) -> str:
    lines = [line.strip() for line in (asi_text or "").splitlines() if line.strip()]
    if not lines:
        return "No actionable side information."
    compact = "\n".join(lines[:max_lines])
    return _truncate_text(compact, max_chars)


def _compact_trace_bundle_for_reflection(trace_bundle: dict[str, Any]) -> dict[str, Any]:
    compact = {
        "summary": copy.deepcopy((trace_bundle or {}).get("summary") or {}),
        "representative_traces": [],
        "representative_successes": [],
    }
    for key in ("representative_traces", "representative_successes"):
        for trace in (trace_bundle or {}).get(key) or []:
            if not isinstance(trace, dict):
                continue
            compact_trace = {
                "doc_id": trace.get("doc_id"),
                "chunk_id": trace.get("chunk_id"),
                "theme": trace.get("theme"),
                "text_excerpt": _truncate_text(trace.get("text_excerpt"), 240),
                "parse_error": trace.get("parse_error"),
                "extraction_warning": trace.get("extraction_warning"),
                "intermediate_event_count": trace.get("intermediate_event_count"),
                "event_count": trace.get("event_count"),
                "temporal_edge_count": trace.get("temporal_edge_count"),
                "causal_edge_count": trace.get("causal_edge_count"),
                "participant_ref_integrity": trace.get("participant_ref_integrity"),
                "parse_retries_used": trace.get("parse_retries_used"),
                "empty_retries_used": trace.get("empty_retries_used"),
            }
            judge = trace.get("judge")
            if isinstance(judge, dict):
                compact_trace["judge"] = {
                    "judge_error": judge.get("judge_error"),
                    "overall": judge.get("overall"),
                    "normalized_mean_score": judge.get("normalized_mean_score"),
                    "reason": _truncate_text(judge.get("reason"), 220),
                    "major_issues": [
                        _truncate_text(issue, 120) for issue in (judge.get("major_issues") or [])[:3]
                    ],
                }
            compact[key].append(compact_trace)
    return compact


def _fallback_reflection_mutations(
    *,
    candidate: dict[str, Any],
    metrics: dict[str, Any],
    asi_text: str,
    num_mutations: int,
) -> list[dict[str, Any]]:
    base = sanitize_candidate(candidate, default_name=candidate.get("name") or "two_stage_event_graph_candidate")
    issues_text = " ".join(
        [
            _truncate_text(asi_text, 4000),
            json.dumps(metrics, ensure_ascii=True),
        ]
    ).lower()

    mutations: list[dict[str, Any]] = []

    completeness_child = copy.deepcopy(base)
    completeness_child["name"] = f"{base.get('name')}_fallback_completeness"
    completeness_cfg = completeness_child.setdefault("config", {})
    completeness_cfg["max_events_per_chunk"] = min(10, int(completeness_cfg.get("max_events_per_chunk", 6) or 6) + 2)
    completeness_cfg["stage1_max_completion_tokens"] = min(
        3200,
        int(completeness_cfg.get("stage1_max_completion_tokens", 2200) or 2200) + 400,
    )
    completeness_cfg["stage2_max_completion_tokens"] = min(
        3400,
        int(completeness_cfg.get("stage2_max_completion_tokens", 2400) or 2400) + 400,
    )
    completeness_child["stage1_system_prompt"] = (
        (completeness_child.get("stage1_system_prompt") or "").rstrip()
        + "\n- Capture distinct meetings, emails, travel or logistics changes, performance feedback, and explicit state changes as separate events when they are clearly stated."
        + "\n- Do not collapse multiple concrete events into one summary event."
    )
    completeness_child["stage2_system_prompt"] = (
        (completeness_child.get("stage2_system_prompt") or "").rstrip()
        + "\n- Preserve distinct stage-1 events unless they are true duplicates."
    )
    mutations.append(sanitize_candidate(completeness_child))

    if num_mutations > 1 or "temporal edge" in issues_text or "causal" in issues_text:
        relation_child = copy.deepcopy(base)
        relation_child["name"] = f"{base.get('name')}_fallback_relations"
        relation_cfg = relation_child.setdefault("config", {})
        relation_cfg["stage2_max_completion_tokens"] = min(
            3400,
            int(relation_cfg.get("stage2_max_completion_tokens", 2400) or 2400) + 500,
        )
        relation_child["stage2_system_prompt"] = (
            (relation_child.get("stage2_system_prompt") or "").rstrip()
            + "\n- Use explicit event IDs consistently in every temporal and causal edge."
            + "\n- Prefer conservative, source-grounded temporal and causal edges over speculative ones."
        )
        mutations.append(sanitize_candidate(relation_child))

    return mutations[: max(1, num_mutations)]


def _compute_metrics(
    predictions_docs: list[dict[str, Any]],
    *,
    input_path: str,
    run_name: str,
    candidate: dict[str, Any],
    candidate_model: str,
    judge_details: list[dict[str, Any]],
) -> dict[str, Any]:
    chunks = [chunk for doc in predictions_docs for chunk in doc.get("chunks") or []]
    num_chunks = len(chunks)
    if num_chunks == 0:
        raise ValueError("No chunks evaluated")

    valid_chunks = 0
    non_empty_chunks = 0
    total_events = 0
    total_temporal = 0
    total_causal = 0
    valid_participant_refs = 0
    total_participant_refs = 0
    temporal_acyclic_hits = 0
    temporal_date_hits = 0
    causal_acyclic_hits = 0
    causal_self_loop_free_hits = 0
    extraction_prompt_tokens_total = 0
    extraction_completion_tokens_total = 0
    extraction_total_tokens_total = 0
    extraction_latency_seconds_total = 0.0
    extraction_attempts_total = 0
    total_verb_sentences = 0
    total_covered_verb_sentences_before = 0
    total_added_fallback_events = 0

    for chunk in chunks:
        graph = chunk.get("graph") or {}
        events = graph.get("events") or []
        temporal_edges = graph.get("temporal_edges") or []
        causal_edges = graph.get("causal_edges") or []
        extraction_stats = _sum_call_stats((chunk.get("_trace_meta") or {}).get("extraction_calls") or [])
        extraction_prompt_tokens_total += int(extraction_stats["prompt_tokens"])
        extraction_completion_tokens_total += int(extraction_stats["completion_tokens"])
        extraction_total_tokens_total += int(extraction_stats["total_tokens"])
        extraction_latency_seconds_total += float(extraction_stats["latency_seconds"])
        extraction_attempts_total += int(extraction_stats["call_count"])
        coverage_meta = chunk.get("coverage_enforcement") or {}
        total_verb_sentences += int(coverage_meta.get("num_verb_sentences", 0) or 0)
        total_covered_verb_sentences_before += int(coverage_meta.get("num_covered_verb_sentences_before", 0) or 0)
        total_added_fallback_events += int(coverage_meta.get("num_added_fallback_events", 0) or 0)
        if chunk.get("parse_error") in (None, "") and isinstance(graph, dict):
            valid_chunks += 1
        if events:
            non_empty_chunks += 1
        total_events += len(events)
        total_temporal += len(temporal_edges)
        total_causal += len(causal_edges)

        entity_ids = {e.get("entity_id") for e in graph.get("entities") or [] if e.get("entity_id")}
        for event in events:
            for participant in event.get("participants") or []:
                entity_id = participant.get("entity_id")
                if not entity_id:
                    continue
                total_participant_refs += 1
                if entity_id in entity_ids:
                    valid_participant_refs += 1

        temporal_pairs = _build_edges_for_cycle(temporal_edges, temporal=True)
        if not _has_cycle(temporal_pairs):
            temporal_acyclic_hits += 1
        if _temporal_date_consistent(chunk):
            temporal_date_hits += 1

        causal_pairs = _build_edges_for_cycle(causal_edges, temporal=False)
        if not _has_cycle(causal_pairs):
            causal_acyclic_hits += 1
        if all(edge.get("from_event") != edge.get("to_event") for edge in causal_edges):
            causal_self_loop_free_hits += 1

    schema_validity = valid_chunks / num_chunks
    non_empty_rate = non_empty_chunks / num_chunks
    mean_events_per_chunk = total_events / num_chunks
    mean_temporal_edges_per_chunk = total_temporal / num_chunks
    mean_causal_edges_per_chunk = total_causal / num_chunks
    participant_ref_integrity = 1.0 if total_participant_refs == 0 else valid_participant_refs / total_participant_refs
    temporal_acyclic_rate = temporal_acyclic_hits / num_chunks
    temporal_date_consistency_rate = temporal_date_hits / num_chunks
    causal_acyclic_rate = causal_acyclic_hits / num_chunks
    causal_self_loop_free_rate = causal_self_loop_free_hits / num_chunks
    coverage_event_score = min(mean_events_per_chunk / 4.0, 1.0)
    coverage_relation_score = min((mean_temporal_edges_per_chunk + mean_causal_edges_per_chunk) / 2.0, 1.0)
    verb_sentence_coverage_before = 1.0 if total_verb_sentences == 0 else (total_covered_verb_sentences_before / total_verb_sentences)

    judge_success = [row for row in judge_details if row.get("judge_error") in (None, "")]
    judge_calls_total = len(judge_details)
    judge_score_mean = None
    judge_criteria_means: dict[str, float] = {}
    judge_prompt_tokens_total = 0
    judge_completion_tokens_total = 0
    judge_total_tokens_total = 0
    judge_latency_seconds_total = 0.0
    judge_attempts_total = 0
    for row in judge_details:
        trace = row.get("_trace_meta") or {}
        judge_prompt_tokens_total += int(trace.get("prompt_tokens", 0) or 0)
        judge_completion_tokens_total += int(trace.get("completion_tokens", 0) or 0)
        judge_total_tokens_total += int(trace.get("total_tokens", 0) or 0)
        judge_latency_seconds_total += float(trace.get("latency_seconds", 0.0) or 0.0)
        judge_attempts_total += 1 if trace else 0
    if judge_success:
        judge_score_mean = sum(row.get("normalized_mean_score", 0.0) for row in judge_success) / len(judge_success)
        for key in ["factual_accuracy", "completeness", "relevance", "faithfulness", "coherence", "overall"]:
            judge_criteria_means[key] = sum(row.get(key, 0.0) for row in judge_success) / len(judge_success)

    structural_terms = [
        schema_validity,
        non_empty_rate,
        participant_ref_integrity,
        temporal_acyclic_rate,
        temporal_date_consistency_rate,
        causal_acyclic_rate,
        causal_self_loop_free_rate,
        coverage_event_score,
        coverage_relation_score,
    ]
    structural_score = sum(structural_terms) / len(structural_terms)
    quality_score = structural_score if judge_score_mean is None else ((structural_score + judge_score_mean) / 2.0)
    extraction_calls_total = num_chunks
    cost_calls_total = extraction_calls_total + judge_calls_total
    prompt_tokens_total = extraction_prompt_tokens_total + judge_prompt_tokens_total
    completion_tokens_total = extraction_completion_tokens_total + judge_completion_tokens_total
    total_tokens_total = extraction_total_tokens_total + judge_total_tokens_total
    total_latency_seconds = extraction_latency_seconds_total + judge_latency_seconds_total
    cost_proxy_total = total_tokens_total if total_tokens_total > 0 else cost_calls_total

    return {
        "run_name": run_name,
        "input_path": input_path,
        "candidate_name": candidate.get("name"),
        "candidate_hash": candidate_hash(candidate),
        "candidate_model": candidate_model,
        "stage1_model": candidate.get("stage1_model"),
        "stage2_model": candidate.get("stage2_model"),
        "num_chunks": num_chunks,
        "schema_validity": schema_validity,
        "non_empty_rate": non_empty_rate,
        "mean_events_per_chunk": mean_events_per_chunk,
        "mean_temporal_edges_per_chunk": mean_temporal_edges_per_chunk,
        "mean_causal_edges_per_chunk": mean_causal_edges_per_chunk,
        "participant_ref_integrity": participant_ref_integrity,
        "temporal_acyclic_rate": temporal_acyclic_rate,
        "temporal_date_consistency_rate": temporal_date_consistency_rate,
        "causal_acyclic_rate": causal_acyclic_rate,
        "causal_self_loop_free_rate": causal_self_loop_free_rate,
        "coverage_event_score": coverage_event_score,
        "coverage_relation_score": coverage_relation_score,
        "verb_sentence_coverage_before_postprocess": verb_sentence_coverage_before,
        "added_fallback_events_total": total_added_fallback_events,
        "structural_score": structural_score,
        "judge_calls_total": judge_calls_total,
        "judge_score_mean": judge_score_mean,
        "judge_criteria_means": judge_criteria_means,
        "quality_score": quality_score,
        "extraction_calls_total": extraction_calls_total,
        "cost_calls_total": cost_calls_total,
        "cost_calls_per_chunk": cost_calls_total / num_chunks,
        "extraction_attempts_total": extraction_attempts_total,
        "extraction_prompt_tokens_total": extraction_prompt_tokens_total,
        "extraction_completion_tokens_total": extraction_completion_tokens_total,
        "extraction_total_tokens_total": extraction_total_tokens_total,
        "extraction_latency_seconds_total": extraction_latency_seconds_total,
        "extraction_latency_seconds_per_chunk": extraction_latency_seconds_total / num_chunks,
        "judge_attempts_total": judge_attempts_total,
        "judge_prompt_tokens_total": judge_prompt_tokens_total,
        "judge_completion_tokens_total": judge_completion_tokens_total,
        "judge_total_tokens_total": judge_total_tokens_total,
        "judge_latency_seconds_total": judge_latency_seconds_total,
        "judge_latency_seconds_per_call": 0.0 if judge_calls_total == 0 else judge_latency_seconds_total / judge_calls_total,
        "prompt_tokens_total": prompt_tokens_total,
        "completion_tokens_total": completion_tokens_total,
        "total_tokens_total": total_tokens_total,
        "total_latency_seconds": total_latency_seconds,
        "cost_proxy_total": cost_proxy_total,
        "cost_proxy_per_chunk": cost_proxy_total / num_chunks,
    }


def _build_asi(predictions_docs: list[dict[str, Any]], judge_details: list[dict[str, Any]]) -> str:
    lines = []
    empty_chunks = []
    parse_failures = []
    for doc in predictions_docs:
        for chunk in doc.get("chunks") or []:
            graph = chunk.get("graph") or {}
            if chunk.get("parse_error"):
                parse_failures.append((doc.get("doc_id"), chunk.get("chunk_id"), chunk.get("parse_error")))
            elif not any(graph.get(name) for name in ("events", "temporal_edges", "causal_edges")):
                empty_chunks.append((doc.get("doc_id"), chunk.get("chunk_id")))
    if parse_failures:
        lines.append("Parse failures:")
        for doc_id, chunk_id, err in parse_failures[:5]:
            lines.append(f"- {doc_id} / {chunk_id}: {err}")
    if empty_chunks:
        lines.append("Empty chunks:")
        for doc_id, chunk_id in empty_chunks[:5]:
            lines.append(f"- {doc_id} / {chunk_id}")

    issues = []
    for row in judge_details:
        if row.get("judge_error"):
            issues.append(f"- {row.get('doc_id')} / {row.get('chunk_id')}: judge_error={row.get('judge_error')}")
        else:
            raw_issues = row.get("major_issues")
            if isinstance(raw_issues, list):
                issue_items = [str(item) for item in raw_issues if str(item).strip()]
            elif isinstance(raw_issues, str):
                issue_items = [raw_issues] if raw_issues.strip() else []
            elif raw_issues in (None, ""):
                issue_items = []
            else:
                issue_items = [str(raw_issues)]
            for item in issue_items:
                issues.append(f"- {row.get('doc_id')} / {row.get('chunk_id')}: {item}")
    if issues:
        lines.append("Judge issues:")
        lines.extend(issues[:10])
    return "\n".join(lines).strip() + ("\n" if lines else "")


def evaluate_candidate(
    *,
    input_path: str,
    output_dir: str,
    run_name: str,
    candidate: dict[str, Any],
    max_docs: int | None = None,
    max_total_chunks: int | None = None,
    judge_model: str | None = None,
    judge_chunk_limit: int | None = None,
    include_raw_llm: bool = True,
) -> EvalArtifacts:
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    candidate = sanitize_candidate(candidate)
    stage1_model = candidate.get("stage1_model") or "claude-sonnet-4-5"
    stage2_model = candidate.get("stage2_model") or stage1_model
    candidate_model = f"{stage1_model} -> {stage2_model}"
    docs = _iter_docs(input_path, max_docs=max_docs, max_total_chunks=max_total_chunks)
    generated_at = datetime.now(timezone.utc).isoformat()

    predictions_docs = []
    for doc in docs:
        out_chunks = []
        for chunk in doc.get("chunks") or []:
            result = _extract_chunk_with_candidate(
                chunk=chunk,
                doc=doc,
                candidate=candidate,
                include_raw_llm=include_raw_llm,
            )
            result["_candidate_config"] = copy.deepcopy(candidate.get("config") or {})
            out_chunks.append(result)
        predictions_docs.append(
            {
                "doc_id": doc.get("doc_id") or doc.get("case_id") or doc.get("title") or "doc",
                "model": candidate_model,
                "stage1_model": stage1_model,
                "stage2_model": stage2_model,
                "candidate_name": candidate.get("name"),
                "generated_at_utc": generated_at,
                "num_chunks": len(out_chunks),
                "chunks": out_chunks,
            }
        )

    judge_details = []
    if judge_model:
        judged = 0
        for doc in predictions_docs:
            for chunk in doc.get("chunks") or []:
                if judge_chunk_limit is not None and judged >= judge_chunk_limit:
                    break
                judge_details.append(_judge_chunk(chunk, doc.get("doc_id"), judge_model))
                judged += 1
            if judge_chunk_limit is not None and judged >= judge_chunk_limit:
                break

    metrics = _compute_metrics(
        predictions_docs,
        input_path=input_path,
        run_name=run_name,
        candidate=candidate,
        candidate_model=candidate_model,
        judge_details=judge_details,
    )
    asi_text = _build_asi(predictions_docs, judge_details)
    trace_bundle = _build_trace_bundle(predictions_docs, judge_details, metrics)
    persisted_predictions_docs = _strip_internal_fields(predictions_docs)
    persisted_judge_details = _strip_internal_fields(judge_details)

    prefix = output_dir_path / run_name
    predictions_path = prefix.with_name(f"{run_name}_predictions.jsonl")
    metrics_path = prefix.with_name(f"{run_name}_metrics.json")
    asi_path = prefix.with_name(f"{run_name}_asi.txt")
    judge_details_path = prefix.with_name(f"{run_name}_judge_details.jsonl")

    with open(predictions_path, "w") as fh:
        for doc in persisted_predictions_docs:
            fh.write(json.dumps(doc, default=str) + "\n")
    metrics_path.write_text(json.dumps(metrics, indent=2))
    asi_path.write_text(asi_text)
    with open(judge_details_path, "w") as fh:
        for row in persisted_judge_details:
            fh.write(json.dumps(row, default=str) + "\n")

    return EvalArtifacts(
        metrics=metrics,
        predictions_docs=persisted_predictions_docs,
        asi_text=asi_text,
        judge_details=persisted_judge_details,
        trace_bundle=trace_bundle,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        asi_path=asi_path,
        judge_details_path=judge_details_path,
    )


def _eligible_for_best(metrics: dict[str, Any], *, judge_required: bool, min_schema: float, min_non_empty: float, min_judge: float) -> bool:
    if metrics.get("schema_validity", 0.0) < min_schema:
        return False
    if metrics.get("non_empty_rate", 0.0) < min_non_empty:
        return False
    if judge_required:
        score = metrics.get("judge_score_mean")
        if score is None or score < min_judge:
            return False
    return True


def _dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
    aq, ac = a.get("quality_score", -math.inf), a.get("cost_proxy_total", a.get("cost_calls_total", math.inf))
    bq, bc = b.get("quality_score", -math.inf), b.get("cost_proxy_total", b.get("cost_calls_total", math.inf))
    return (aq >= bq and ac <= bc) and (aq > bq or ac < bc)


def build_pareto_front(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    front = []
    for row in rows:
        if any(_dominates(other, row) for other in rows if other is not row):
            continue
        front.append(row)
    front.sort(key=lambda r: (-float(r.get("quality_score", -1)), float(r.get("cost_proxy_total", r.get("cost_calls_total", 1e9)))))
    return front


def _entry_matches_row(entry: dict[str, Any], row: dict[str, Any]) -> bool:
    return (
        entry["row"].get("candidate_hash") == row.get("candidate_hash")
        and entry["row"].get("candidate_name") == row.get("candidate_name")
    )


def _select_frontier_parent_entries(
    evaluated_rows: list[dict[str, Any]],
    *,
    judge_required: bool,
    min_schema: float,
    min_non_empty: float,
    min_judge: float,
    frontier_parent_limit: int,
) -> list[dict[str, Any]]:
    eligible_entries = [
        entry
        for entry in evaluated_rows
        if _eligible_for_best(
            entry["artifacts"].metrics,
            judge_required=judge_required,
            min_schema=min_schema,
            min_non_empty=min_non_empty,
            min_judge=min_judge,
        )
    ]
    candidate_pool = eligible_entries if eligible_entries else evaluated_rows
    pareto_rows = build_pareto_front([entry["row"] for entry in candidate_pool])
    selected = []
    for row in pareto_rows:
        match = next((entry for entry in candidate_pool if _entry_matches_row(entry, row)), None)
        if match is not None:
            selected.append(match)
        if len(selected) >= frontier_parent_limit:
            break
    return selected


def reflect_mutations(
    *,
    candidate: dict[str, Any],
    metrics: dict[str, Any],
    asi_text: str,
    trace_bundle: dict[str, Any],
    reflection_model: str,
    num_mutations: int,
) -> tuple[list[dict[str, Any]], str | None]:
    client, model, temperature = _make_client(reflection_model)
    full_candidate_json = json.dumps(candidate, ensure_ascii=True, indent=2)
    compact_candidate_json = json.dumps(candidate, ensure_ascii=True, separators=(",", ":"))
    metrics_json = json.dumps(metrics, ensure_ascii=True, separators=(",", ":"))
    full_trace_json = json.dumps(trace_bundle, ensure_ascii=True, separators=(",", ":"))
    compact_trace_json = json.dumps(
        _compact_trace_bundle_for_reflection(trace_bundle),
        ensure_ascii=True,
        separators=(",", ":"),
    )
    attempts = [
        {
            "candidate_json": full_candidate_json,
            "metrics_json": metrics_json,
            "asi_text": _compact_asi_text(asi_text, max_lines=18, max_chars=3200),
            "trace_json": _truncate_text(full_trace_json, 9000),
            "num_mutations": num_mutations,
            "max_completion_tokens": 2600,
            "enforce_json_response_format": True,
            "retry_note": "",
        },
        {
            "candidate_json": compact_candidate_json,
            "metrics_json": metrics_json,
            "asi_text": _compact_asi_text(asi_text, max_lines=10, max_chars=1600),
            "trace_json": _truncate_text(compact_trace_json, 3500),
            "num_mutations": 1,
            "max_completion_tokens": 1600,
            "enforce_json_response_format": False,
            "retry_note": (
                "\n\nRETRY INSTRUCTION:\n"
                "Your previous response was empty or unusable. Return exactly one mutation. "
                "Keep the same candidate schema. Make targeted edits only."
            ),
        },
    ]

    last_error: str | None = None
    for attempt in attempts:
        messages = build_reflection_messages(
            candidate_json=attempt["candidate_json"],
            metrics_json=attempt["metrics_json"],
            asi_text=attempt["asi_text"],
            trace_json=attempt["trace_json"],
            num_mutations=int(attempt["num_mutations"]),
        )
        if attempt["retry_note"]:
            messages[-1]["content"] = str(messages[-1]["content"]) + str(attempt["retry_note"])
        try:
            raw, _ = _request_json_with_stats(
                client=client,
                model=model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=int(attempt["max_completion_tokens"]),
                enforce_json_response_format=bool(attempt["enforce_json_response_format"]),
                request_kind="reflect",
            )
        except Exception as exc:
            last_error = str(exc)
            continue

        payload, _, error = _extract_payload(raw)
        if error:
            last_error = error
            continue
        if not payload:
            last_error = "Empty response"
            continue
        payload = _unwrap_single_nested_object(payload) if isinstance(payload, dict) else payload
        muts = payload.get("mutations") if isinstance(payload, dict) else None
        if isinstance(muts, dict):
            muts = [muts]
        if not isinstance(muts, list) or not muts:
            last_error = "Empty response"
            continue
        out = []
        for idx, mut in enumerate(muts, start=1):
            if not isinstance(mut, dict):
                continue
            child = sanitize_candidate(mut, default_name=f"{candidate.get('name')}_m{idx}")
            out.append(child)
        if out:
            return out[:num_mutations], None
        last_error = "Empty response"

    fallback_mutations = _fallback_reflection_mutations(
        candidate=candidate,
        metrics=metrics,
        asi_text=asi_text,
        num_mutations=num_mutations,
    )
    if fallback_mutations:
        return fallback_mutations, None
    return [], last_error or "Empty response"


def optimize_candidates(
    *,
    input_path: str,
    seed_candidate: dict[str, Any],
    output_dir: str,
    run_name: str,
    max_evals: int,
    max_docs: int | None,
    max_total_chunks: int | None,
    judge_model: str | None,
    judge_chunk_limit: int | None,
    reflection_model: str,
    mutations_per_generation: int = 2,
    frontier_parent_limit: int = DEFAULT_FRONTIER_PARENT_LIMIT,
    plateau_patience: int = DEFAULT_PLATEAU_PATIENCE,
    min_quality_improvement: float = DEFAULT_MIN_QUALITY_IMPROVEMENT,
    include_raw_llm: bool = True,
    min_best_schema_validity: float = DEFAULT_MIN_SCHEMA_VALIDITY,
    min_best_non_empty_rate: float = DEFAULT_MIN_NON_EMPTY_RATE,
    min_best_judge_score_mean: float = DEFAULT_MIN_JUDGE_SCORE_MEAN,
) -> dict[str, Any]:
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    candidates_dir = output_dir_path / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    history_path = output_dir_path / f"{run_name}_history.jsonl"
    best_candidate_path = output_dir_path / f"{run_name}_best_candidate.json"
    best_metrics_path = output_dir_path / f"{run_name}_best_metrics.json"
    best_predictions_path = output_dir_path / f"{run_name}_best_predictions.jsonl"
    pareto_path = output_dir_path / f"{run_name}_pareto_front.json"

    evaluated_rows = []
    history_rows = []
    seed_candidate = sanitize_candidate(seed_candidate)
    save_candidate_json(seed_candidate, candidates_dir)

    seed_art = evaluate_candidate(
        input_path=input_path,
        output_dir=output_dir,
        run_name=f"{run_name}_seed",
        candidate=seed_candidate,
        max_docs=max_docs,
        max_total_chunks=max_total_chunks,
        judge_model=judge_model,
        judge_chunk_limit=judge_chunk_limit,
        include_raw_llm=include_raw_llm,
    )
    seed_row = {
        "candidate_name": seed_candidate.get("name"),
        "parent_candidate_name": None,
        "generation": 0,
        **seed_art.metrics,
        "reflection_error": None,
        "candidate_path": str(save_candidate_json(seed_candidate, candidates_dir)),
        "metrics_path": str(seed_art.metrics_path),
        "predictions_path": str(seed_art.predictions_path),
    }
    evaluated_rows.append({"candidate": seed_candidate, "row": seed_row, "artifacts": seed_art})
    history_rows.append(seed_row)

    eval_count = 1
    generation = 1
    best_quality_seen = float(seed_art.metrics.get("quality_score", -1))
    plateau_generations = 0
    judge_required = judge_model is not None
    while eval_count < max_evals:
        parent_entries = _select_frontier_parent_entries(
            evaluated_rows,
            judge_required=judge_required,
            min_schema=min_best_schema_validity,
            min_non_empty=min_best_non_empty_rate,
            min_judge=min_best_judge_score_mean,
            frontier_parent_limit=frontier_parent_limit,
        )
        if not parent_entries:
            break

        generation_improved = False
        evaluated_this_generation = 0
        for parent_entry in parent_entries:
            if eval_count >= max_evals:
                break
            parent_candidate = parent_entry["candidate"]
            mutations, reflection_error = reflect_mutations(
                candidate=parent_candidate,
                metrics=parent_entry["artifacts"].metrics,
                asi_text=parent_entry["artifacts"].asi_text,
                trace_bundle=parent_entry["artifacts"].trace_bundle,
                reflection_model=reflection_model,
                num_mutations=mutations_per_generation,
            )
            if reflection_error:
                history_rows.append(
                    {
                        "candidate_name": f"{run_name}_g{generation}_p{safe_name(parent_candidate.get('name'))}_m0",
                        "parent_candidate_name": parent_candidate.get("name"),
                        "generation": generation,
                        "structural_score": None,
                        "judge_score_mean": None,
                        "quality_score": None,
                        "reflection_error": reflection_error,
                    }
                )
                continue

            for idx, child in enumerate(mutations):
                if eval_count >= max_evals:
                    break
                child = sanitize_candidate(child)
                child.setdefault("name", f"{run_name}_g{generation}_p{safe_name(parent_candidate.get('name'))}_m{idx}")
                if child.get("name") == parent_candidate.get("name"):
                    child["name"] = f"{run_name}_g{generation}_p{safe_name(parent_candidate.get('name'))}_m{idx}"
                candidate_path = save_candidate_json(child, candidates_dir)
                child_run_name = child.get("name")
                art = evaluate_candidate(
                    input_path=input_path,
                    output_dir=output_dir,
                    run_name=child_run_name,
                    candidate=child,
                    max_docs=max_docs,
                    max_total_chunks=max_total_chunks,
                    judge_model=judge_model,
                    judge_chunk_limit=judge_chunk_limit,
                    include_raw_llm=include_raw_llm,
                )
                row = {
                    "candidate_name": child.get("name"),
                    "parent_candidate_name": parent_candidate.get("name"),
                    "generation": generation,
                    **art.metrics,
                    "reflection_error": None,
                    "candidate_path": str(candidate_path),
                    "metrics_path": str(art.metrics_path),
                    "predictions_path": str(art.predictions_path),
                }
                evaluated_rows.append({"candidate": child, "row": row, "artifacts": art})
                history_rows.append(row)
                eval_count += 1
                evaluated_this_generation += 1
                child_quality = float(art.metrics.get("quality_score", -1))
                if child_quality > best_quality_seen + min_quality_improvement:
                    best_quality_seen = child_quality
                    generation_improved = True
        generation += 1
        if evaluated_this_generation == 0:
            plateau_generations += 1
        elif generation_improved:
            plateau_generations = 0
        else:
            plateau_generations += 1
        if plateau_patience is not None and plateau_generations >= plateau_patience:
            break

    with open(history_path, "w") as fh:
        for row in history_rows:
            fh.write(json.dumps(row, default=str) + "\n")

    eligible_rows = [
        entry
        for entry in evaluated_rows
        if _eligible_for_best(
            entry["artifacts"].metrics,
            judge_required=judge_required,
            min_schema=min_best_schema_validity,
            min_non_empty=min_best_non_empty_rate,
            min_judge=min_best_judge_score_mean,
        )
    ]
    selected_pool = eligible_rows if eligible_rows else evaluated_rows
    best_entry = max(selected_pool, key=lambda entry: float(entry["artifacts"].metrics.get("quality_score", -1)))

    best_candidate_path.write_text(json.dumps(best_entry["candidate"], indent=2))
    best_metrics_path.write_text(json.dumps(best_entry["artifacts"].metrics, indent=2))
    with open(best_predictions_path, "w") as fh:
        for doc in best_entry["artifacts"].predictions_docs:
            fh.write(json.dumps(doc, default=str) + "\n")

    pareto_input = [entry["row"] for entry in (eligible_rows if eligible_rows else evaluated_rows)]
    pareto_front = build_pareto_front(pareto_input)
    pareto_path.write_text(json.dumps(pareto_front, indent=2))

    return {
        "history_path": str(history_path),
        "best_candidate_path": str(best_candidate_path),
        "best_metrics_path": str(best_metrics_path),
        "best_predictions_path": str(best_predictions_path),
        "pareto_front_path": str(pareto_path),
    }
