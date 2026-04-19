#!/usr/bin/env python3
"""Baseline overall event-graph generation from fact-oriented document input."""

from __future__ import annotations

import argparse
import html
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
    _root = Path(__file__).resolve().parents[2]
    _src = _root / "src"
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from agents.config_loader import get_llm_config  # noqa: E402


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_project_dotenv() -> None:
    """Load repo `.env` so AGENT_API_KEY is set even when cwd is not the project root."""
    try:
        from dotenv import load_dotenv

        env_file = _project_root() / ".env"
        if env_file.is_file():
            load_dotenv(env_file, override=True)
    except ImportError:
        pass


def _baseline_api_key() -> str:
    """Baseline uses only AGENT_API_KEY (from env after loading project .env). No config.yaml key."""
    _load_project_dotenv()
    key = (os.environ.get("AGENT_API_KEY") or "").strip()
    if not key:
        env_path = _project_root() / ".env"
        raise ValueError(
            "Baseline requires AGENT_API_KEY. Set it in "
            f"{env_path} (e.g. AGENT_API_KEY=...) or export it in the shell. "
            "This script does not use llm.api_key from config.yaml."
        )
    return key


def _slugify(text: str) -> str:
    safe = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_"):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "graph"


def _short(s: Any, n: int = 120) -> str:
    t = "" if s is None else str(s).strip()
    if len(t) <= n:
        return t
    return t[: max(0, n - 3)].rstrip() + "..."


def _graph_to_vis_from_json(graph: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build vis-network nodes/edges directly from canonical graph JSON (not render_graph)."""
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    entity_map = {e.get("entity_id"): e for e in (graph.get("entities") or []) if isinstance(e, dict) and e.get("entity_id")}
    event_ids = {e.get("event_id") for e in (graph.get("events") or []) if isinstance(e, dict) and e.get("event_id")}

    # Participant-referenced entities missing from entities[]
    for ev in graph.get("events") or []:
        if not isinstance(ev, dict):
            continue
        for p in ev.get("participants") or []:
            if not isinstance(p, dict):
                continue
            eid = p.get("entity_id")
            if eid and eid not in entity_map:
                entity_map[eid] = {"entity_id": eid, "name": str(eid), "kind": "UNKNOWN", "canonical_role": ""}

    for eid, ent in entity_map.items():
        name = ent.get("name") or eid
        kind = ent.get("kind") or ""
        role = ent.get("canonical_role") or ""
        label = name if not kind else f"{name}\n({kind})"
        tip_lines = [f"entity_id: {eid}", f"name: {name}", f"kind: {kind}", f"role: {role}"]
        nodes.append(
            {
                "id": f"ent:{eid}",
                "label": label,
                "group": "entity",
                "shape": "ellipse",
                "color": {"background": "#dbeafe", "border": "#2563eb", "highlight": {"background": "#bfdbfe", "border": "#1d4ed8"}},
                "font": {"size": 13, "multi": True},
                "margin": 10,
                "title": html.escape("\n".join(tip_lines)),
            }
        )

    for ev in graph.get("events") or []:
        if not isinstance(ev, dict):
            continue
        eid = ev.get("event_id")
        if not eid:
            continue
        et = ev.get("event_type") or "EVENT"
        verb = ev.get("main_verb") or ""
        trig = (ev.get("trigger") or {}).get("span_text") or ""
        ev_ids = ev.get("evidence", {}).get("sentence_ids") or []
        label_parts = [str(et)]
        if verb:
            label_parts.append(f"verb: {_short(verb, 40)}")
        if trig:
            label_parts.append(f"trigger: {_short(trig, 50)}")
        label = "\n".join(label_parts)
        tip = [
            f"event_id: {eid}",
            f"event_type: {et}",
            f"main_verb: {verb}",
            f"trigger: {trig}",
            f"evidence_sentence_ids: {ev_ids}",
        ]
        nodes.append(
            {
                "id": f"evt:{eid}",
                "label": label,
                "group": "event",
                "shape": "box",
                "color": {"background": "#fef3c7", "border": "#b45309", "highlight": {"background": "#fde68a", "border": "#92400e"}},
                "font": {"size": 12, "multi": True, "face": "system-ui, sans-serif"},
                "margin": 12,
                "title": html.escape("\n".join(tip)),
            }
        )

    for ev in graph.get("events") or []:
        if not isinstance(ev, dict):
            continue
        eid = ev.get("event_id")
        if not eid:
            continue
        for p in ev.get("participants") or []:
            if not isinstance(p, dict):
                continue
            peid = p.get("entity_id")
            if not peid:
                continue
            role = str(p.get("role") or "participant")
            edges.append(
                {
                    "from": f"evt:{eid}",
                    "to": f"ent:{peid}",
                    "label": _short(role, 24),
                    "arrows": "to",
                    "color": {"color": "#64748b"},
                    "font": {"size": 10, "align": "middle"},
                    "smooth": {"type": "cubicBezier", "forceDirection": "horizontal", "roundness": 0.4},
                }
            )

    for edge in graph.get("temporal_edges") or []:
        if not isinstance(edge, dict):
            continue
        a, b = edge.get("from_event"), edge.get("to_event")
        if not a or not b or a not in event_ids or b not in event_ids:
            continue
        rel = str(edge.get("relation") or "TEMPORAL")
        edges.append(
            {
                "from": f"evt:{a}",
                "to": f"evt:{b}",
                "label": _short(rel, 20),
                "arrows": "to",
                "dashes": True,
                "color": {"color": "#0284c7"},
                "font": {"size": 11, "align": "middle"},
                "smooth": {"type": "cubicBezier", "roundness": 0.2},
            }
        )

    for edge in graph.get("causal_edges") or []:
        if not isinstance(edge, dict):
            continue
        a, b = edge.get("from_event"), edge.get("to_event")
        if not a or not b or a not in event_ids or b not in event_ids:
            continue
        rel = str(edge.get("relation") or "CAUSAL")
        edges.append(
            {
                "from": f"evt:{a}",
                "to": f"evt:{b}",
                "label": _short(rel, 20),
                "arrows": "to",
                "width": 2,
                "color": {"color": "#c2410c"},
                "font": {"size": 11, "align": "middle"},
                "smooth": {"type": "cubicBezier", "roundness": 0.35},
            }
        )

    return nodes, edges


def _write_baseline_graph_html(
    path: Path,
    *,
    doc_id: str,
    graph: dict[str, Any],
    parse_error: str | None = None,
) -> None:
    """Single-page HTML: vis-network from canonical graph JSON + stats + optional raw JSON."""
    ents = graph.get("entities") or []
    evs = graph.get("events") or []
    te = graph.get("temporal_edges") or []
    ce = graph.get("causal_edges") or []
    vis_nodes, vis_edges = _graph_to_vis_from_json(graph)
    nodes_payload = json.dumps(vis_nodes, ensure_ascii=False)
    edges_payload = json.dumps(vis_edges, ensure_ascii=False)
    graph_json_pp = json.dumps(graph, indent=2, ensure_ascii=False)

    vis_css = "https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/dist/vis-network.min.css"
    vis_js = "https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js"

    err_block = ""
    if parse_error:
        err_block = (
            f'<div class="banner error"><strong>Parse / extraction error</strong><pre>{html.escape(parse_error)}</pre></div>'
        )
    empty_note = ""
    if not evs and not ents and not parse_error:
        empty_note = '<div class="banner warn">No entities or events in graph JSON.</div>'

    title_esc = html.escape(doc_id)
    json_esc = html.escape(graph_json_pp)

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title_esc} — baseline graph</title>
  <link rel="stylesheet" href="{vis_css}" />
  <style>
    :root {{
      --bg: #f8fafc;
      --card: #ffffff;
      --border: #e2e8f0;
      --text: #0f172a;
      --muted: #64748b;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }}
    header {{
      background: var(--card);
      border-bottom: 1px solid var(--border);
      padding: 12px 20px;
      flex-shrink: 0;
    }}
    header h1 {{
      margin: 0 0 8px 0;
      font-size: 1.05rem;
      font-weight: 600;
      word-break: break-all;
    }}
    .stats {{
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      font-size: 0.875rem;
      color: var(--muted);
    }}
    .stats span strong {{ color: var(--text); }}
    .banner {{
      margin: 0 20px;
      padding: 12px 16px;
      border-radius: 8px;
      font-size: 0.875rem;
    }}
    .banner.error {{ background: #fef2f2; border: 1px solid #fecaca; color: #991b1b; }}
    .banner.error pre {{ margin: 8px 0 0 0; white-space: pre-wrap; word-break: break-word; font-size: 0.8rem; }}
    .banner.warn {{ background: #fffbeb; border: 1px solid #fde68a; color: #92400e; margin-bottom: 0; }}
    #graph {{
      flex: 1;
      min-height: 480px;
      width: 100%;
      border-top: 1px solid var(--border);
      background: var(--card);
    }}
    details.raw {{
      margin: 0;
      border-top: 1px solid var(--border);
      background: var(--card);
    }}
    details.raw summary {{
      cursor: pointer;
      padding: 10px 20px;
      font-size: 0.85rem;
      color: var(--muted);
      user-select: none;
    }}
    details.raw pre {{
      margin: 0;
      padding: 0 20px 16px;
      max-height: 320px;
      overflow: auto;
      font-size: 11px;
      line-height: 1.4;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .legend {{
      padding: 8px 20px;
      font-size: 0.8rem;
      color: var(--muted);
      border-top: 1px solid var(--border);
      background: var(--card);
    }}
    .legend span {{ margin-right: 16px; }}
    .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 4px; vertical-align: middle; }}
  </style>
</head>
<body>
  <header>
    <h1>{title_esc}</h1>
    <div class="stats">
      <span><strong>Entities</strong> {len(ents)}</span>
      <span><strong>Events</strong> {len(evs)}</span>
      <span><strong>Temporal edges</strong> {len(te)}</span>
      <span><strong>Causal edges</strong> {len(ce)}</span>
      <span><strong>Vis nodes</strong> {len(vis_nodes)}</span>
      <span><strong>Vis edges</strong> {len(vis_edges)}</span>
    </div>
  </header>
  {err_block}
  {empty_note}
  <div id="graph"></div>
  <div class="legend">
    <span><span class="dot" style="background:#fef3c7;border:1px solid #b45309"></span>Event</span>
    <span><span class="dot" style="background:#dbeafe;border:1px solid #2563eb"></span>Entity</span>
    <span>— dashed blue: temporal</span>
    <span>— solid orange: causal</span>
    <span>— gray: participant</span>
  </div>
  <details class="raw">
    <summary>Source graph JSON (this document)</summary>
    <pre>{json_esc}</pre>
  </details>
  <script src="{vis_js}"></script>
  <script>
    const visNodes = new vis.DataSet({nodes_payload});
    const visEdges = new vis.DataSet({edges_payload});
    const container = document.getElementById("graph");
    const data = {{ nodes: visNodes, edges: visEdges }};
    const options = {{
      physics: {{
        enabled: true,
        solver: "forceAtlas2Based",
        forceAtlas2Based: {{
          gravitationalConstant: -38,
          centralGravity: 0.008,
          springLength: 180,
          springConstant: 0.06,
          damping: 0.5,
          avoidOverlap: 0.9
        }},
        stabilization: {{ iterations: 200, updateInterval: 25 }}
      }},
      interaction: {{ hover: true, tooltipDelay: 120, multiselect: true, navigationButtons: true }},
      layout: {{ improvedLayout: true }},
      nodes: {{ font: {{ size: 12 }} }},
      edges: {{ font: {{ size: 10, strokeWidth: 0 }}, smooth: {{ type: "dynamic" }} }}
    }};
    const network = new vis.Network(container, data, options);
    network.once("stabilizationIterationsDone", function () {{
      network.setOptions({{ physics: {{ enabled: false }} }});
    }});
  </script>
</body>
</html>
"""
    path.write_text(html_doc, encoding="utf-8")

TRUNCATION_PARSE_MARKERS = (
    "expecting value",
    "unterminated string",
    "expecting ',' delimiter",
    "expecting property name enclosed in double quotes",
    "unterminated object",
    "unterminated array",
)

DOC_SYSTEM_PROMPT = """You are a legal event extraction analyst.

You are given one full legal document with sentence-labeled facts.
Extract ONE overall document-level event graph, just extract what you think are events.

Return strict JSON only. No markdown. No explanation.

Required top-level keys:
- entities
- events
- temporal_edges
- causal_edges

Requirements:
- Use only information stated in the provided text.
- Use concrete event types.
- Use unique IDs: entities E1..En, events EV1..EVn.
- Preserve sentence IDs from [S#] in trigger/mention/evidence fields.
- Keep IDs consistent:
  - participants[].entity_id must exist in entities[].entity_id
  - temporal_edges/causal_edges endpoints must exist in events[].event_id
- If no temporal/causal relation is supported, return empty arrays.
- ONLY THE EVENTS ARE THE NODES (ENTITIES ARE ATTACHED TO EVENTS WITH IT)

Field requirements for clean graph rendering:
- entities[] items: entity_id, name, kind, canonical_role, aliases
- events[] items: event_id, event_type, main_verb, trigger, participants, time, evidence, confidence
- temporal_edges[] items: from_event, to_event, relation, evidence, confidence
- causal_edges[] items: from_event, to_event, relation, evidence, confidence

Time handling:
- If time is unknown: {"kind":"UNKNOWN","raw_span":"","normalized":null,"confidence":0.0}

Relation labels:
- temporal: BEFORE,  SAME_TIME
- causal: CAUSES
"""

DOC_USER_PROMPT = """Extract an overall document-level legal event graph from the text below.

Document id: {doc_id}
Case/title: {case_name}
Max events to extract: {max_events}

Return strict JSON only with top-level keys:
- entities
- events
- temporal_edges
- causal_edges

TEXT:
{text}
"""

PARSE_RETRY_SUFFIX = (
    "Your previous response was not valid JSON for the required schema. "
    "Return strict JSON only. No markdown and no commentary."
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
        "confidence": float(time_value.get("confidence", 0.0) or 0.0),
    }


def _normalize_evidence(evidence: Any) -> dict[str, Any]:
    evidence = _as_dict(evidence)
    sentence_ids = [_normalize_sentence_id(sid) for sid in _as_list(evidence.get("sentence_ids")) if sid is not None]
    snippets = _normalize_snippets(evidence.get("snippets"))
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
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    seen_nodes = set()
    seen_entity_ids = set()

    for entity in graph.get("entities") or []:
        entity_id = entity.get("entity_id")
        if not entity_id:
            continue
        seen_entity_ids.add(entity_id)
        node_id = f"entity:{entity_id}"
        if node_id in seen_nodes:
            continue
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
            entity_id = participant.get("entity_id")
            if not entity_id:
                continue
            if entity_id not in seen_entity_ids:
                # Ensure participant-referenced entities still render even if entities[] is incomplete.
                seen_entity_ids.add(entity_id)
                entity_node_id = f"entity:{entity_id}"
                if entity_node_id not in seen_nodes:
                    seen_nodes.add(entity_node_id)
                    nodes.append(
                        {
                            "id": entity_node_id,
                            "node_type": "entity",
                            "label": str(entity_id),
                            "properties": {
                                "entity_id": entity_id,
                                "kind": "UNKNOWN",
                                "canonical_role": "UNKNOWN",
                                "aliases": [],
                            },
                        }
                    )
            edges.append(
                {
                    "source": event_node_id,
                    "target": f"entity:{entity_id}",
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
                "properties": {"relation": edge.get("relation")},
            }
        )
    for edge in graph.get("causal_edges") or []:
        edges.append(
            {
                "source": f"event:{edge.get('from_event')}",
                "target": f"event:{edge.get('to_event')}",
                "edge_type": "CAUSAL",
                "label": edge.get("relation") or "CAUSAL",
                "properties": {"relation": edge.get("relation")},
            }
        )
    return {"nodes": nodes, "edges": edges}


def _make_client(model_name: str | None):
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("openai package is required to run this script.") from exc

    cfg = get_llm_config()
    api_key = _baseline_api_key()
    api_base = cfg.get("api_base")
    model = model_name or cfg.get("model") or "gpt4o"
    model_lower = model.lower()
    temperature = 1.0 if "gpt-5" in model_lower or model_lower == "gpt5" else float(cfg.get("temperature", 0.0) or 0.0)
    return OpenAI(api_key=api_key, base_url=api_base, timeout=120.0), model, temperature


def _is_transient_transport_error(exc: Exception) -> bool:
    text = str(exc).lower()
    markers = ("504", "gateway timeout", "timed out", "timeout", "connection error", "502", "503", "rate limit")
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
    *,
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    max_completion_tokens: int | None,
    enforce_json_response_format: bool,
) -> str:
    # Full-document graphs need a large completion budget; do not cap at 4.5k (causes truncated JSON).
    token_budget = int(max_completion_tokens) if max_completion_tokens else None
    if token_budget:
        token_budget = min(token_budget, 32000)
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
            text = str(exc).lower()
            if "only temperature=1 is supported" in text:
                request["temperature"] = 1.0
                response = client.chat.completions.create(**request)
                return (response.choices[0].message.content or "").strip()
            if enforce_json_response_format and "response_format" in text:
                request.pop("response_format", None)
                response = client.chat.completions.create(**request)
                return (response.choices[0].message.content or "").strip()
            if token_budget and ("max_completion_tokens" in text or "unknown parameter" in text):
                request.pop("max_completion_tokens", None)
                response = client.chat.completions.create(**request)
                return (response.choices[0].message.content or "").strip()
            if _is_transient_transport_error(exc) and transport_attempt < 2:
                time.sleep(2 ** transport_attempt)
                if token_budget:
                    token_budget = max(1200, int(token_budget * 0.7))
                continue
            raise
    if last_exc:
        raise last_exc
    raise RuntimeError("LLM request failed without response")


def _iter_jsonl(path: Path):
    with open(path) as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError:
                continue


def _sentence_line(item: dict[str, Any], fallback_sid: int) -> str:
    sid = item.get("id") or item.get("sentence_id") or fallback_sid
    text = str(item.get("text") or "").strip()
    return f"[S{sid}] {text}" if text else ""


def _extract_fact_text(rec: dict[str, Any], *, sentences_field: str, fact_only: bool, facts_field: str | None) -> str:
    sentences = rec.get(sentences_field)
    if isinstance(sentences, list):
        lines = []
        for i, s in enumerate(sentences, start=1):
            if not isinstance(s, dict):
                continue
            if fact_only and s.get("pred_label") != 1:
                continue
            line = _sentence_line(s, i)
            if line:
                lines.append(line)
        if lines:
            return "\n".join(lines)

    if facts_field:
        facts_value = rec.get(facts_field)
        if isinstance(facts_value, str) and facts_value.strip():
            return facts_value.strip()
        if isinstance(facts_value, list):
            kept = []
            for item in facts_value:
                text = str(item or "").strip()
                if not text or text == "O":
                    continue
                kept.append(text)
            if kept:
                return "\n".join(kept)

    text_value = rec.get("document_text") or rec.get("extracted_facts") or ""
    return str(text_value or "").strip()


def _doc_messages(doc_id: str, case_name: str, text: str, max_events: int, retry_suffix: str = "") -> list[dict[str, str]]:
    user = DOC_USER_PROMPT.format(
        doc_id=doc_id,
        case_name=case_name or "Unknown",
        max_events=max_events,
        text=text,
    )
    if retry_suffix:
        user = user.rstrip() + "\n\n" + retry_suffix + "\n"
    return [
        {"role": "system", "content": DOC_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _extract_doc_graph(
    *,
    client: Any,
    model: str,
    temperature: float,
    doc_id: str,
    case_name: str,
    text: str,
    max_events: int,
    max_completion_tokens: int | None,
    max_parse_retries: int,
    include_raw_llm: bool,
) -> dict[str, Any]:
    parse_attempt = 0
    raw = ""
    payload_keys: list[str] = []
    parse_error: str | None = None

    while True:
        retry_suffix = PARSE_RETRY_SUFFIX if parse_attempt > 0 else ""
        messages = _doc_messages(
            doc_id=doc_id,
            case_name=case_name,
            text=text,
            max_events=max_events,
            retry_suffix=retry_suffix,
        )
        try:
            raw = _request_json(
                client=client,
                model=model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
                enforce_json_response_format=(parse_attempt == 0),
            )
        except Exception as exc:
            return {
                "doc_id": doc_id,
                "graph": {"entities": [], "events": [], "temporal_edges": [], "causal_edges": []},
                "structures": [],
                "render_graph": {"nodes": [], "edges": []},
                "parse_error": str(exc),
                **({"raw_llm_response": raw} if include_raw_llm else {}),
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
                "doc_id": doc_id,
                "graph": {"entities": [], "events": [], "temporal_edges": [], "causal_edges": []},
                "structures": [],
                "render_graph": {"nodes": [], "edges": []},
                "llm_payload_keys": payload_keys,
                "parse_error": parse_error,
                **({"raw_llm_response": raw} if include_raw_llm else {}),
            }

        graph = _normalize_payload(payload)
        return {
            "doc_id": doc_id,
            "graph": graph,
            "structures": _build_structures(graph),
            "render_graph": _build_render_graph(graph),
            "llm_payload_keys": payload_keys,
            "parse_retries_used": parse_attempt,
            **({"raw_llm_response": raw} if include_raw_llm else {}),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline overall event graph extraction with LLM.")
    parser.add_argument("--input", default=None, help="Input JSONL with facts/sentences (not used with --html-only).")
    parser.add_argument("--from-doc-index", type=int, required=True, help="1-based inclusive start doc index.")
    parser.add_argument("--to-doc-index", type=int, required=True, help="1-based inclusive end doc index.")
    parser.add_argument("--model", default=None, help="Optional model override.")
    parser.add_argument(
        "--output-json",
        default="/work/pi_dagarwal_umass_edu/project_1/agentic-legal-event-extraction/data/outputs/baseline/graph/llm.json",
    )
    parser.add_argument(
        "--output-html-dir",
        default="/work/pi_dagarwal_umass_edu/project_1/agentic-legal-event-extraction/data/outputs/baseline/graph/llm-html",
    )
    parser.add_argument("--sentences-field", default="sentences", help="Array field with sentence dicts.")
    parser.add_argument("--facts-field", default="predicted_extracted_facts", help="Fallback field for fact text.")
    parser.add_argument("--fact-only", dest="fact_only", action="store_true", help="Keep only pred_label==1 in --sentences-field.")
    parser.add_argument("--no-fact-only", dest="fact_only", action="store_false", help="Use all sentences in --sentences-field.")
    parser.set_defaults(fact_only=True)
    parser.add_argument("--max-events-per-doc", type=int, default=40)
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=16000,
        help="Output token budget (raised default so full-doc JSON is not truncated mid-string).",
    )
    parser.add_argument("--max-parse-retries", type=int, default=3)
    parser.add_argument("--include-raw-llm", action="store_true")
    parser.add_argument(
        "--html-only",
        action="store_true",
        help="Regenerate HTML from existing --output-json only (no LLM calls).",
    )
    args = parser.parse_args()

    output_json = Path(args.output_json)
    output_html_dir = Path(args.output_html_dir)
    output_html_dir.mkdir(parents=True, exist_ok=True)

    if args.html_only:
        if not output_json.is_file():
            raise SystemExit(f"--html-only: file not found: {output_json}")
        bundle = json.loads(output_json.read_text(encoding="utf-8"))
        manifest: list[dict[str, Any]] = []
        for doc in bundle.get("docs") or []:
            doc_id = str(doc.get("doc_id") or "doc")
            graph = doc.get("graph") or {"entities": [], "events": [], "temporal_edges": [], "causal_edges": []}
            parse_error = doc.get("parse_error")
            html_path = output_html_dir / f"{_slugify(doc_id)}__doc_graph.html"
            _write_baseline_graph_html(html_path, doc_id=doc_id, graph=graph, parse_error=parse_error)
            manifest.append({"doc_id": doc_id, "html": str(html_path)})
        (output_html_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Wrote HTML (from JSON) to {output_html_dir}")
        return

    if not args.input:
        raise SystemExit("--input is required unless using --html-only")

    if args.from_doc_index < 1 or args.to_doc_index < args.from_doc_index:
        raise SystemExit("--from-doc-index must be >=1 and --to-doc-index must be >= --from-doc-index")

    input_path = Path(args.input)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    client, model, temperature = _make_client(args.model)
    generated_at = datetime.now(timezone.utc).isoformat()

    docs_out: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    from_idx = args.from_doc_index
    to_idx = args.to_doc_index

    total_to_process = to_idx - from_idx + 1
    doc_num = 0
    for idx, (_, rec) in enumerate(_iter_jsonl(input_path), start=1):
        if idx < from_idx:
            continue
        if idx > to_idx:
            break
        doc_num += 1
        doc_id = str(rec.get("doc_id") or rec.get("case_id") or rec.get("title") or f"doc_{idx}")
        case_name = str(rec.get("case_name") or rec.get("title") or doc_id)
        print(f"[{doc_num}/{total_to_process}] Processing doc {idx}: {doc_id[:80]}...", flush=True)
        fact_text = _extract_fact_text(
            rec,
            sentences_field=args.sentences_field,
            fact_only=args.fact_only,
            facts_field=args.facts_field,
        )
        if not fact_text:
            doc_graph = {
                "doc_id": doc_id,
                "graph": {"entities": [], "events": [], "temporal_edges": [], "causal_edges": []},
                "structures": [],
                "render_graph": {"nodes": [], "edges": []},
                "parse_error": "No usable text/facts found in record.",
            }
            print(f"  -> No usable text/facts, skipping LLM call.", flush=True)
        else:
            try:
                doc_graph = _extract_doc_graph(
                    client=client,
                    model=model,
                    temperature=temperature,
                    doc_id=doc_id,
                    case_name=case_name,
                    text=fact_text,
                    max_events=args.max_events_per_doc,
                    max_completion_tokens=args.max_completion_tokens,
                    max_parse_retries=args.max_parse_retries,
                    include_raw_llm=args.include_raw_llm,
                )
                n_ev = len((doc_graph.get("graph") or {}).get("events") or [])
                print(f"  -> OK: {n_ev} events extracted.", flush=True)
            except Exception as exc:
                print(f"  -> ERROR: {exc}", flush=True)
                doc_graph = {
                    "doc_id": doc_id,
                    "graph": {"entities": [], "events": [], "temporal_edges": [], "causal_edges": []},
                    "structures": [],
                    "render_graph": {"nodes": [], "edges": []},
                    "parse_error": f"LLM call failed: {exc}",
                }

        graph = doc_graph.get("graph") or {"entities": [], "events": [], "temporal_edges": [], "causal_edges": []}
        render_graph = doc_graph.get("render_graph") or {"nodes": [], "edges": []}
        structures = doc_graph.get("structures") or []
        row = {
            "doc_id": doc_id,
            "source_doc_index": idx,
            "model": model,
            "generated_at_utc": generated_at,
            "graph": graph,
            "structures": structures,
            "render_graph": render_graph,
            "num_entities": len(graph.get("entities") or []),
            "num_events": len(graph.get("events") or []),
            "num_temporal_edges": len(graph.get("temporal_edges") or []),
            "num_causal_edges": len(graph.get("causal_edges") or []),
        }
        if doc_graph.get("llm_payload_keys") is not None:
            row["llm_payload_keys"] = doc_graph.get("llm_payload_keys")
        if doc_graph.get("parse_error"):
            row["parse_error"] = doc_graph.get("parse_error")
        if doc_graph.get("parse_retries_used") is not None:
            row["parse_retries_used"] = doc_graph.get("parse_retries_used")
        if args.include_raw_llm and doc_graph.get("raw_llm_response") is not None:
            row["raw_llm_response"] = doc_graph.get("raw_llm_response")
        docs_out.append(row)

        html_path = output_html_dir / f"{_slugify(doc_id)}__doc_graph.html"
        _write_baseline_graph_html(
            html_path,
            doc_id=doc_id,
            graph=graph,
            parse_error=doc_graph.get("parse_error"),
        )
        manifest.append({"doc_id": doc_id, "source_doc_index": idx, "html": str(html_path)})

    payload = {
        "input": str(input_path),
        "from_doc_index": from_idx,
        "to_doc_index": to_idx,
        "model": model,
        "generated_at_utc": generated_at,
        "num_docs": len(docs_out),
        "docs": docs_out,
    }
    output_json.write_text(json.dumps(payload, indent=2))
    (output_html_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Wrote JSON: {output_json}")
    print(f"Wrote HTML directory: {output_html_dir}")
    print(f"Docs processed: {len(docs_out)}")


if __name__ == "__main__":
    main()
