"""
LLM pass: normalize proper names in a chunk mini-graph to generic role-style labels
(Plaintiff, Defendant, Supervisor_1, Court, …) using chunk text for grounding.

Preserves entity_id, event_id, edge endpoints, and sentence_ids so deterministic merge still works.
"""

from __future__ import annotations

import json
import re
from typing import Any

MINIGRAPH_MASK_SYSTEM = """You are normalizing a legal event mini-graph for cross-chunk merging.

You receive:
1) The source chunk text (sentence-labeled if present).
2) A JSON mini-graph: entities, events, temporal_edges, causal_edges.

Task: rewrite **surface strings only** so proper names and org-specific strings become **generic, role-based labels** wherever the chunk text supports it. Work across **complaints, motions, opinions, orders** — do not assume a single template.

Rules:
- **Preserve exactly** every `entity_id`, `event_id`, `from_event`, `to_event`, numeric `sentence_id` fields, and graph topology (same number of entities, events, edges; same endpoints).
- Set each entity `name` to a stable generic label, e.g. `PLAINTIFF`, `DEFENDANT`, `SUPERVISOR_1`, `SUPERVISOR_2`, `COURT`, `WITNESS_1`, `ORG_EMPLOYER`, `OTHER_PARTY_1`, `UNKNOWN_PERSON_1`. Use the **same** label for the same referent every time it appears (mentions, snippets, triggers).
- Set `canonical_role` when the text clearly supports it; otherwise use `UNKNOWN`. Allowed examples: PLAINTIFF, DEFENDANT, EMPLOYER, SUPERVISOR, COURT, JUDGE, COUNSEL, MOVING_PARTY, RESPONDENT, PETITIONER, WITNESS, UNKNOWN.
- Rewrite `aliases`, participant `mention.span_text`, `trigger.span_text`, `evidence.snippets[].text`, and `time.raw_span` only when they contain **identifying proper names** — keep dates and neutral phrases when they contain no names.
- Do **not** invent events, entities, or edges. Do **not** change `relation` strings unless fixing obvious copy errors (prefer leave as-is).
- If unsure whether two strings are the same person, use distinct labels (`OTHER_PARTY_1` vs `OTHER_PARTY_2`).
- Output **only** valid JSON with exactly these top-level keys: entities, events, temporal_edges, causal_edges.
"""

MINIGRAPH_MASK_USER = """Chunk id: {chunk_id}
Case (if any): {case_name}

CHUNK TEXT:
{text}

MINI_GRAPH JSON (normalize surface strings as instructed; keep all IDs and sentence_ids):
{graph_json}
"""


def _strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()
    return text


def _extract_graph_json(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    text = _strip_code_fences(raw)
    if not text:
        return None, "empty response"
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, str(exc)
    if not isinstance(data, dict):
        return None, "not an object"
    return data, None


def _ids_entities(graph: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for e in graph.get("entities") or []:
        if isinstance(e, dict) and e.get("entity_id"):
            out.add(str(e["entity_id"]))
    return out


def _ids_events(graph: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for e in graph.get("events") or []:
        if isinstance(e, dict) and e.get("event_id"):
            out.add(str(e["event_id"]))
    return out


def _edge_pairs(graph: dict[str, Any], key: str) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for edge in graph.get(key) or []:
        if not isinstance(edge, dict):
            continue
        a, b = edge.get("from_event"), edge.get("to_event")
        if a is not None and b is not None:
            out.add((str(a), str(b)))
    return out


def masked_graph_is_compatible(original: dict[str, Any], masked: dict[str, Any]) -> bool:
    """Structural check: IDs and edge endpoints must match."""
    if not isinstance(masked, dict):
        return False
    for k in ("entities", "events", "temporal_edges", "causal_edges"):
        if k not in masked:
            return False
    if _ids_entities(original) != _ids_entities(masked):
        return False
    if _ids_events(original) != _ids_events(masked):
        return False
    for ek in ("temporal_edges", "causal_edges"):
        if _edge_pairs(original, ek) != _edge_pairs(masked, ek):
            return False
    return True


def mask_minigraph_with_llm(
    *,
    graph: dict[str, Any],
    chunk_text: str,
    chunk_id: str,
    case_name: str,
    client: Any,
    model: str,
    temperature: float,
    max_completion_tokens: int | None = 8192,
) -> tuple[dict[str, Any], str | None]:
    """
    Returns (masked_graph, error). On error, caller should fall back to the input graph.
    """
    graph_json = json.dumps(graph, ensure_ascii=False, indent=2)
    user = MINIGRAPH_MASK_USER.format(
        chunk_id=chunk_id or "chunk",
        case_name=case_name or "",
        text=chunk_text or "(no chunk text provided)",
        graph_json=graph_json,
    )
    messages = [
        {"role": "system", "content": MINIGRAPH_MASK_SYSTEM},
        {"role": "user", "content": user},
    ]
    request: dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature}
    if max_completion_tokens:
        request["max_completion_tokens"] = int(max_completion_tokens)
    raw = ""
    try:
        request["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(**request)
        raw = (response.choices[0].message.content or "").strip()
    except TypeError:
        request.pop("response_format", None)
        try:
            response = client.chat.completions.create(**request)
            raw = (response.choices[0].message.content or "").strip()
        except Exception as exc:
            return graph, str(exc)
    except Exception as exc:
        err_text = str(exc).lower()
        if "response_format" in err_text or "json_object" in err_text:
            request.pop("response_format", None)
            try:
                response = client.chat.completions.create(**request)
                raw = (response.choices[0].message.content or "").strip()
            except Exception as exc2:
                return graph, str(exc2)
        else:
            return graph, str(exc)

    parsed, err = _extract_graph_json(raw)
    if err or parsed is None:
        return graph, err or "parse failed"

    if not masked_graph_is_compatible(graph, parsed):
        return graph, "masked graph failed structural validation (IDs or edges)"

    return parsed, None
