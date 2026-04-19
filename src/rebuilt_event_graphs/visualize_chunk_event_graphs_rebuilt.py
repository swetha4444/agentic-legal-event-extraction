#!/usr/bin/env python3
"""Rebuilt HTML visualizer for chunk-level or doc-level legal event graphs."""
import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

VIS_CSS = "https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/dist/vis-network.min.css"
VIS_JS = "https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js"
BOOTSTRAP_CSS = "https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta3/dist/css/bootstrap.min.css"
BOOTSTRAP_JS = "https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta3/dist/js/bootstrap.bundle.min.js"
# Bootstrap .table ensures borders/grid render in all browsers (avoids “missing table” with CDN CSS alone).
AUDIT_TABLE_CLASSES = "audit-table table table-bordered table-sm bg-white mb-0"


def _slugify(text: str) -> str:
    safe = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_"):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "graph"


def _read_docs(path: str) -> list[dict[str, Any]]:
    docs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def _html_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _tooltip_from_event(chunk: dict[str, Any], event: dict[str, Any], graph: dict[str, Any]) -> str:
    entity_map = {e.get("entity_id"): e for e in graph.get("entities") or []}
    participants = []
    for participant in event.get("participants") or []:
        entity = entity_map.get(participant.get("entity_id"), {})
        label = entity.get("name") or participant.get("entity_id") or "UNKNOWN"
        participants.append(f"{participant.get('role')}: {label}")
    time_value = dict(event.get("time") or {})
    time_value.pop("confidence", None)
    lines = [
        f"chunk: {chunk.get('chunk_id')}",
        f"event_id: {event.get('event_id')}",
        f"event_type: {event.get('event_type')}",
        f"main_verb: {event.get('main_verb')}",
        f"trigger: {event.get('trigger', {}).get('span_text')}",
        f"time: {json.dumps(time_value, ensure_ascii=True)}",
        f"participants: {', '.join(participants) if participants else 'none'}",
        f"evidence_sentence_ids: {event.get('evidence', {}).get('sentence_ids') or []}",
    ]
    return "<br>".join(_html_escape(line) for line in lines)


def _short_text(value: Any, limit: int = 80) -> str:
    text = "" if value is None else str(value).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _compact_role_label(value: Any) -> str:
    text = str(value or "").strip()
    lowered = text.lower()
    if not lowered:
        return ""
    if "plaintiff" in lowered:
        return "Plaintiff"
    if "defendant" in lowered:
        return "Defendant"
    if any(token in lowered for token in ("institution", "organization", "company", "employer", "corporation", "corp", "inc", "llc")):
        return "Institution"
    if any(token in lowered for token in ("supervisor", "manager", "cto", "coo", "ceo", "executive", "boss")):
        return "Supervisor"
    if any(token in lowered for token in ("employee", "director", "contractor", "worker", "staff")):
        return "Employee"
    if "recruit" in lowered:
        return "Recruiter"
    if "human resources" in lowered or lowered == "hr":
        return "HR"
    return ""


def _participant_compact_label(participant: dict[str, Any], entity: dict[str, Any]) -> str:
    role_candidates = [
        participant.get("role"),
        *(entity.get("aliases") or []),
        entity.get("canonical_role"),
    ]
    for candidate in role_candidates:
        label = _compact_role_label(candidate)
        if label:
            return label
    if (entity.get("kind") or "").upper() == "ORGANIZATION":
        return "Institution"
    return str(entity.get("name") or participant.get("entity_id") or "Participant")


def _event_only_node_label(event: dict[str, Any], graph: dict[str, Any]) -> str:
    entity_map = {e.get("entity_id"): e for e in graph.get("entities") or []}
    event_title = str(event.get("event_type") or event.get("event_id") or "EVENT")
    participant_labels = []
    seen = set()
    for participant in event.get("participants") or []:
        entity = entity_map.get(participant.get("entity_id"), {})
        label = _participant_compact_label(participant, entity)
        if not label or label in seen:
            continue
        seen.add(label)
        participant_labels.append(label)
    lines = [event_title]
    if participant_labels:
        compact = " | ".join(participant_labels[:4])
        if len(participant_labels) > 4:
            compact += " | ..."
        lines.append(compact)

    main_verb = _short_text(event.get("main_verb"), 40)
    trigger = _short_text((event.get("trigger") or {}).get("span_text"), 60)
    evidence = (event.get("evidence") or {}).get("snippets") or []
    evidence_text = _short_text(evidence[0], 70) if evidence else ""

    if main_verb:
        lines.append(f"verb: {main_verb}")
    if trigger:
        lines.append(f"trigger: {trigger}")
    if evidence_text:
        lines.append(f"evidence: {evidence_text}")
    return "\n".join(lines)


def _tooltip_from_properties(chunk_id: str, label: str, props: dict[str, Any]) -> str:
    lines = [f"chunk: {chunk_id}", f"label: {label}"]
    for key, value in props.items():
        lines.append(f"{key}: {json.dumps(value, ensure_ascii=True)}")
    return "<br>".join(_html_escape(line) for line in lines)


def _event_only_graph(doc: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes = []
    edges = []
    index = 0
    for chunk in doc.get("chunks") or []:
        graph = chunk.get("graph") or {}
        event_map = {event.get("event_id"): event for event in graph.get("events") or []}
        temporal_pairs = {
            (edge.get("from_event"), edge.get("to_event"))
            for edge in graph.get("temporal_edges") or []
            if edge.get("from_event") and edge.get("to_event")
        }
        causal_pairs = {
            (edge.get("from_event"), edge.get("to_event"))
            for edge in graph.get("causal_edges") or []
            if edge.get("from_event") and edge.get("to_event")
        }
        dual_pairs = temporal_pairs & causal_pairs
        for event in graph.get("events") or []:
            node_id = f"{chunk.get('chunk_id')}::event::{event.get('event_id')}"
            nodes.append(
                {
                    "id": node_id,
                    "label": _event_only_node_label(event, graph),
                    "shape": "box",
                    "color": {"background": "#f4efe2", "border": "#9b7d3e"},
                    "title": _tooltip_from_event(chunk, event, graph),
                    "chunk_id": chunk.get("chunk_id"),
                    "index": index,
                }
            )
            index += 1
        for edge in graph.get("temporal_edges") or []:
            src = f"{chunk.get('chunk_id')}::event::{edge.get('from_event')}"
            dst = f"{chunk.get('chunk_id')}::event::{edge.get('to_event')}"
            if edge.get("from_event") in event_map and edge.get("to_event") in event_map:
                smooth = {"type": "continuous"}
                if (edge.get("from_event"), edge.get("to_event")) in dual_pairs:
                    smooth = {"type": "curvedCW", "roundness": 0.25}
                edges.append(
                    {
                        "from": src,
                        "to": dst,
                        "label": edge.get("relation") or "TEMPORAL",
                        "font": {"align": "middle"},
                        "color": {"color": "#2c7fb8"},
                        "dashes": True,
                        "arrows": "to",
                        "smooth": smooth,
                    }
                )
        for edge in graph.get("causal_edges") or []:
            src = f"{chunk.get('chunk_id')}::event::{edge.get('from_event')}"
            dst = f"{chunk.get('chunk_id')}::event::{edge.get('to_event')}"
            if edge.get("from_event") in event_map and edge.get("to_event") in event_map:
                smooth = {"type": "continuous"}
                if (edge.get("from_event"), edge.get("to_event")) in dual_pairs:
                    smooth = {"type": "curvedCCW", "roundness": 0.25}
                edges.append(
                    {
                        "from": src,
                        "to": dst,
                        "label": edge.get("relation") or "CAUSAL",
                        "font": {"align": "middle"},
                        "color": {"color": "#d95f0e"},
                        "width": 2,
                        "arrows": "to",
                        "smooth": smooth,
                    }
                )
    return nodes, edges


def _full_graph(doc: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes = []
    edges = []
    seen = set()
    index = 0
    for chunk in doc.get("chunks") or []:
        render = chunk.get("render_graph") or {}
        for node in render.get("nodes") or []:
            scoped_id = f"{chunk.get('chunk_id')}::{node.get('id')}"
            if scoped_id in seen:
                continue
            seen.add(scoped_id)
            node_type = node.get("node_type")
            color = {
                "event": {"background": "#f4efe2", "border": "#9b7d3e"},
                "entity": {"background": "#e6f2ff", "border": "#3b6ea5"},
                "verb": {"background": "#f2e6ff", "border": "#7a4ea3"},
            }.get(node_type, {"background": "#ffffff", "border": "#555555"})
            shape = "box" if node_type == "event" else "ellipse"
            nodes.append(
                {
                    "id": scoped_id,
                    "label": node.get("label") or node.get("id"),
                    "shape": shape,
                    "color": color,
                    "title": _tooltip_from_properties(chunk.get("chunk_id") or "", node.get("label") or "", node.get("properties") or {}),
                    "chunk_id": chunk.get("chunk_id"),
                    "index": index,
                }
            )
            index += 1
        for edge in render.get("edges") or []:
            edges.append(
                {
                    "from": f"{chunk.get('chunk_id')}::{edge.get('source')}",
                    "to": f"{chunk.get('chunk_id')}::{edge.get('target')}",
                    "label": edge.get("label") or edge.get("edge_type") or "",
                    "font": {"align": "middle"},
                    "color": {
                        "color": {
                            "TEMPORAL": "#2c7fb8",
                            "CAUSAL": "#d95f0e",
                            "PARTICIPATES_IN": "#666666",
                            "HAS_VERB": "#999999",
                        }.get(edge.get("edge_type"), "#888888")
                    },
                    "dashes": edge.get("edge_type") == "TEMPORAL",
                    "arrows": "to",
                }
            )
    return nodes, edges


def _assign_positions(nodes: list[dict[str, Any]]) -> None:
    if not nodes:
        return
    n = len(nodes)
    # Fewer nodes per row + larger gaps so wide event labels do not overlap.
    cols = max(1, min(n, int(math.ceil(math.sqrt(n) * 0.85))))
    spacing_x = 520
    spacing_y = 280
    for idx, node in enumerate(nodes):
        row = idx // cols
        col = idx % cols
        node["x"] = col * spacing_x
        node["y"] = row * spacing_y


def _build_options(layout_mode: str) -> dict[str, Any]:
    physics = False
    node_fixed = False
    if layout_mode == "rigid":
        physics = False
        node_fixed = True
    elif layout_mode == "draggable":
        physics = False
        node_fixed = False
    else:
        physics = {
            "enabled": True,
            "solver": "barnesHut",
            "barnesHut": {"gravitationalConstant": -4200, "springLength": 320, "springConstant": 0.035, "damping": 0.3},
            "stabilization": {"enabled": True, "iterations": 250},
        }
        node_fixed = False
    return {
        "interaction": {"hover": True, "multiselect": True, "navigationButtons": True},
        "physics": physics,
        "layout": {"improvedLayout": True},
        "nodes": {"shape": "box", "margin": 10, "font": {"size": 16}, "fixed": node_fixed},
        "edges": {"font": {"align": "middle"}, "smooth": {"type": "continuous"}},
    }


def _sum_chunk_graph_metrics(doc: dict[str, Any]) -> tuple[int, int, int, int]:
    """Return (event_rows, temporal_edges, causal_edges, entity_rows) summed over chunk graphs."""
    total_e = total_t = total_c = total_ent = 0
    for ch in doc.get("chunks") or []:
        g = ch.get("graph") or ch.get("mini_graph") or {}
        total_e += len(g.get("events") or [])
        total_t += len(g.get("temporal_edges") or [])
        total_c += len(g.get("causal_edges") or [])
        total_ent += len(g.get("entities") or [])
    return total_e, total_t, total_c, total_ent


def _merged_graph_counts_panel_html(g: dict[str, Any]) -> str:
    """Always-visible counts from the merged graph when no pre-merge source file is available."""
    rows = [
        ("Entities", str(len(g.get("entities") or []))),
        ("Events", str(len(g.get("events") or []))),
        ("Temporal edges", str(len(g.get("temporal_edges") or []))),
        ("Causal edges", str(len(g.get("causal_edges") or []))),
    ]
    trs: list[str] = []
    for label, val in rows:
        trs.append(
            "        <tr><td>"
            + _html_escape(label)
            + "</td><td style=\"text-align:right\">"
            + _html_escape(val)
            + "</td></tr>\n"
        )
    note = (
        "Counts above are from the merged graph on this page. "
        "For chunk-sum vs merged (dedupe) rows, pass --source-predictions with the full path to that doc's chunk-level JSONL (same doc_id)."
    )
    return (
        "<div class=\"audit\" id=\"merge-dedupe-audit\">\n"
        "  <div class=\"audit-title\">Merged graph statistics</div>\n"
        f"  <table class=\"{AUDIT_TABLE_CLASSES}\" role=\"grid\" aria-label=\"Merged graph counts\">\n"
        "    <thead><tr><th scope=\"col\">Metric</th><th scope=\"col\" style=\"text-align:right\">Value</th></tr></thead>\n"
        "    <tbody>\n"
        + "".join(trs)
        + "    </tbody>\n"
        "  </table>\n"
        f"  <div class=\"audit-note\">{_html_escape(note)}</div>\n"
        "</div>\n"
    )


def _merge_audit_panel_html(
    *,
    pre_events: int,
    pre_temporal: int,
    pre_causal: int,
    pre_entities: int,
    post_events: int,
    post_temporal: int,
    post_causal: int,
    post_entities: int,
) -> str:
    def _collapse_str(pre: int, post: int) -> str:
        if pre < post:
            return "N/A (pre-sum &lt; merged — source-predictions likely incomplete vs merge input)"
        return str(pre - post)

    rows = [
        ("Sum of chunk-level entity rows (pre-merge)", str(pre_entities)),
        ("Unique entities in merged graph", str(post_entities)),
        ("Entity rows collapsed by name dedupe (pre − merged)", _collapse_str(pre_entities, post_entities)),
        ("Sum of chunk-level event rows (pre-merge)", str(pre_events)),
        ("Unique events in merged graph", str(post_events)),
        ("Event rows collapsed by merge-key dedupe (pre − merged)", _collapse_str(pre_events, post_events)),
        ("Sum of chunk temporal edges (pre-merge)", str(pre_temporal)),
        ("Temporal edges in merged graph", str(post_temporal)),
        ("Sum of chunk causal edges (pre-merge)", str(pre_causal)),
        ("Causal edges in merged graph", str(post_causal)),
    ]
    trs: list[str] = []
    for label, val in rows:
        trs.append(
            "        <tr><td>"
            + _html_escape(label)
            + "</td><td style=\"text-align:right\">"
            + (val if val.startswith("N/A") else _html_escape(val))
            + "</td></tr>\n"
        )
    note = (
        "Chunk columns must be summed from the same doc you merged (all chunks, non-empty graphs). "
        "If pre-merge sums are smaller than merged counts, your --source-predictions file does not match the merge input — "
        "edge rows are still shown, but entity/event dedupe rows show N/A. "
        "Events collapse when the merge key matches across chunks; entities collapse on normalized name. "
        "Merge appends edges across chunks (no edge dedupe). "
        "Edges drop when endpoints cannot be remapped after event dedupe."
    )
    return (
        "<div class=\"audit\" id=\"merge-dedupe-audit\">\n"
        "  <div class=\"audit-title\">Merge audit &amp; dedupe (chunk sums vs merged graph)</div>\n"
        f"  <table class=\"{AUDIT_TABLE_CLASSES}\" role=\"grid\" aria-label=\"Merge statistics\">\n"
        "    <thead><tr><th scope=\"col\">Metric</th><th scope=\"col\" style=\"text-align:right\">Value</th></tr></thead>\n"
        "    <tbody>\n"
        + "".join(trs)
        + "    </tbody>\n"
        "  </table>\n"
        f"  <div class=\"audit-note\">{_html_escape(note)}</div>\n"
        "</div>\n"
    )


def _merge_audit_from_user_dict(data: dict[str, Any]) -> str:
    rows_html = []
    for k, v in sorted(data.items(), key=lambda kv: str(kv[0])):
        disp = v if isinstance(v, str) else json.dumps(v, ensure_ascii=True)
        rows_html.append(
            f"<tr><td>{_html_escape(str(k))}</td><td>{_html_escape(disp)}</td></tr>"
        )
    return (
        "<div class=\"audit\" id=\"merge-dedupe-audit\"><div class=\"audit-title\">merge_audit (from JSON)</div>"
        f"<table class=\"{AUDIT_TABLE_CLASSES}\"><thead><tr><th>Key</th><th>Value</th></tr></thead><tbody>"
        + "".join(rows_html)
        + "</tbody></table></div>"
    )


def _default_legend_html(
    graph_view: str, layout_mode: str, *, extra_bullets: list[str] | None = None
) -> str:
    drag = "Drag nodes when layout mode is draggable." if layout_mode == "draggable" else ""
    temporal = (
        "Temporal edges: blue, dashed, arrow. Label on the edge is the relation in JSON "
        "(commonly BEFORE, AFTER, OVERLAP, SAME_TIME). BEFORE and AFTER are opposite precedence orientations."
    )
    causal = "Causal edges: orange, thicker, solid arrow. Labels often CAUSES, ENABLES, PREVENTS."
    nodes_ev = "Event-only view: tan boxes are events; hover for event_id, time, participants, evidence."
    nodes_full = (
        "Full view: event boxes (tan), entity ellipses (blue tint), verb nodes (purple tint) when render_graph is present."
    )
    parts = [temporal, causal, nodes_ev if graph_view == "event-only" else nodes_full, drag]
    parts = [p for p in parts if p]
    if extra_bullets:
        parts.extend(extra_bullets)
    return "<ul class=\"legend-list\"><li>" + "</li><li>".join(_html_escape(p) for p in parts) + "</li></ul>"


def _write_html(
    path: Path,
    title: str,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    layout_mode: str,
    *,
    graph_view: str = "event-only",
    audit_panel_html: str = "",
    legend_extras: list[str] | None = None,
) -> None:
    _assign_positions(nodes)
    options = _build_options(layout_mode)
    payload_nodes = json.dumps(nodes)
    payload_edges = json.dumps(edges)
    payload_options = json.dumps(options)
    legend_body = _default_legend_html(graph_view, layout_mode, extra_bullets=legend_extras)
    html = f"""<html>
  <head>
    <meta charset=\"utf-8\" />
    <script src=\"{VIS_JS}\"></script>
    <link rel=\"stylesheet\" href=\"{VIS_CSS}\" />
    <link rel=\"stylesheet\" href=\"{BOOTSTRAP_CSS}\" />
    <script src=\"{BOOTSTRAP_JS}\"></script>
    <style>
      body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; background: #faf8f2; }}
      .wrap {{ width: 100%; }}
      .title {{ padding: 16px 20px 0 20px; font-size: 20px; font-weight: 600; }}
      #mynetwork {{ width: 100%; height: min(96vh, 1200px); min-height: 880px; background: #ffffff; border-top: 1px solid #ddd; }}
      .legend {{ padding: 8px 20px 12px 20px; color: #444; font-size: 13px; }}
      .legend-list {{ margin: 6px 0 0 18px; padding: 0; }}
      .legend-list li {{ margin: 4px 0; }}
      .audit {{ padding: 12px 20px 16px 20px; background: #ede8dc; border-bottom: 2px solid #b8a882; font-size: 13px; color: #222; border-left: 5px solid #8b7355; }}
      .audit-title {{ font-weight: 700; margin-bottom: 10px; font-size: 15px; }}
      .audit-table {{ display: table !important; width: 100%; max-width: 900px; border-collapse: collapse; background: #fff; box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-top: 8px; table-layout: auto; }}
      .audit-table thead {{ display: table-header-group; }}
      .audit-table tbody {{ display: table-row-group; }}
      .audit-table tr {{ display: table-row; }}
      .audit-table th, .audit-table td {{ display: table-cell; border: 1px solid #888; padding: 8px 12px; vertical-align: top; }}
      .audit-table th {{ background: #d8cfc0; text-align: left; font-weight: 600; }}
      .audit-note {{ margin-top: 10px; font-size: 12px; color: #555; max-width: 900px; }}
    </style>
  </head>
  <body>
    <div class=\"wrap\">
      <div class=\"title\">{_html_escape(title)}</div>
      {audit_panel_html}
      <div class=\"legend\"><strong>Legend</strong>{legend_body}</div>
      <div id=\"mynetwork\"></div>
    </div>
    <script>
      const nodes = new vis.DataSet({payload_nodes});
      const edges = new vis.DataSet({payload_edges});
      const container = document.getElementById('mynetwork');
      const data = {{ nodes, edges }};
      const options = {payload_options};
      const network = new vis.Network(container, data, options);
      if (options.physics && options.physics.enabled) {{
        network.once('stabilizationIterationsDone', function () {{
          network.setOptions({{physics: false}});
        }});
      }}
    </script>
  </body>
</html>
"""
    path.write_text(html)


def _manifest_record(doc_id: str, html_path: Path, view: str, layout_mode: str) -> dict[str, Any]:
    return {
        "doc_id": doc_id,
        "html": str(html_path),
        "graph_view": view,
        "html_layout": layout_mode,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuilt HTML visualization for chunk event graphs.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--graph-view", choices=("event-only", "full"), default="event-only")
    parser.add_argument("--html-layout", choices=("draggable", "rigid", "physics"), default="draggable")
    parser.add_argument("--doc-level", action="store_true", help="Write one HTML per document instead of one HTML per chunk")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument(
        "--source-predictions",
        default=None,
        help="Optional chunk-level predictions JSONL: for doc-level input with a single chunk_id=merged, "
        "compare pre-merge sums to merged graph and render a merge audit table.",
    )
    args = parser.parse_args()

    docs = _read_docs(args.input)
    source_docs: list[dict[str, Any]] | None = None
    if args.source_predictions:
        sp = Path(args.source_predictions)
        if not sp.is_file():
            print(
                f"Warning: --source-predictions file not found: {sp}\n"
                "  (Using merged-graph-only statistics table instead. Fix the path for pre-merge vs merged comparison.)",
                file=sys.stderr,
            )
        else:
            source_docs = _read_docs(str(sp))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    written = 0
    with open(manifest_path, "w") as manifest:
        for doc in docs:
            if args.max_docs is not None and written >= args.max_docs:
                break
            doc_id = doc.get("doc_id") or f"doc_{written}"
            slug = _slugify(doc_id)
            if args.doc_level:
                if args.graph_view == "event-only":
                    nodes, edges = _event_only_graph(doc)
                else:
                    nodes, edges = _full_graph(doc)
                html_path = out_dir / f"{slug}__doc_graph.html"
                audit_html = ""
                merge_audit = doc.get("merge_audit")
                if isinstance(merge_audit, dict):
                    audit_html = _merge_audit_from_user_dict(merge_audit)
                elif source_docs is not None:
                    chunks = doc.get("chunks") or []
                    if len(chunks) == 1 and chunks[0].get("chunk_id") == "merged":
                        src = next((d for d in source_docs if d.get("doc_id") == doc_id), None)
                        if src:
                            pe, pt, pc, pent = _sum_chunk_graph_metrics(src)
                            g = chunks[0].get("graph") or {}
                            ge = len(g.get("events") or [])
                            gent = len(g.get("entities") or [])
                            gt = len(g.get("temporal_edges") or [])
                            gc = len(g.get("causal_edges") or [])
                            audit_html = _merge_audit_panel_html(
                                pre_events=pe,
                                pre_temporal=pt,
                                pre_causal=pc,
                                pre_entities=pent,
                                post_events=ge,
                                post_temporal=gt,
                                post_causal=gc,
                                post_entities=gent,
                            )
                        else:
                            print(
                                f"Warning: doc_id {doc_id!r} not found in --source-predictions; "
                                "using merged-graph-only statistics.",
                                file=sys.stderr,
                            )
                if not audit_html:
                    chunks = doc.get("chunks") or []
                    if len(chunks) == 1 and chunks[0].get("chunk_id") == "merged":
                        g = chunks[0].get("graph") or chunks[0].get("mini_graph") or {}
                        if g:
                            audit_html = _merged_graph_counts_panel_html(g)
                legend_extras: list[str] | None = None
                if audit_html:
                    legend_extras = [
                        "Training loss, GEPA objective, and judge scores are not shown here — "
                        "use run logs, GEPA history JSONL, or evaluation outputs for those metrics."
                    ]
                _write_html(
                    html_path,
                    f"{doc_id} ({args.graph_view})",
                    nodes,
                    edges,
                    args.html_layout,
                    graph_view=args.graph_view,
                    audit_panel_html=audit_html,
                    legend_extras=legend_extras,
                )
                manifest.write(json.dumps(_manifest_record(doc_id, html_path, args.graph_view, args.html_layout)) + "\n")
            else:
                for chunk in doc.get("chunks") or []:
                    chunk_doc = {"doc_id": doc_id, "chunks": [chunk]}
                    if args.graph_view == "event-only":
                        nodes, edges = _event_only_graph(chunk_doc)
                    else:
                        nodes, edges = _full_graph(chunk_doc)
                    html_path = out_dir / f"{slug}__{_slugify(chunk.get('chunk_id') or 'chunk')}__graph.html"
                    title = f"{doc_id} / {chunk.get('chunk_id')} ({args.graph_view})"
                    _write_html(
                        html_path,
                        title,
                        nodes,
                        edges,
                        args.html_layout,
                        graph_view=args.graph_view,
                        audit_panel_html="",
                    )
                    manifest.write(json.dumps(_manifest_record(doc_id, html_path, args.graph_view, args.html_layout)) + "\n")
            written += 1


if __name__ == "__main__":
    main()
