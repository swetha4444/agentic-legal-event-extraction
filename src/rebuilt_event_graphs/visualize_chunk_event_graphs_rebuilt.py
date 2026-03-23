#!/usr/bin/env python3
"""Rebuilt HTML visualizer for chunk-level or doc-level legal event graphs."""
import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

VIS_CSS = "https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/dist/vis-network.min.css"
VIS_JS = "https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js"
BOOTSTRAP_CSS = "https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta3/dist/css/bootstrap.min.css"
BOOTSTRAP_JS = "https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta3/dist/js/bootstrap.bundle.min.js"


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
    lines = [
        f"chunk: {chunk.get('chunk_id')}",
        f"event_id: {event.get('event_id')}",
        f"event_type: {event.get('event_type')}",
        f"main_verb: {event.get('main_verb')}",
        f"trigger: {event.get('trigger', {}).get('span_text')}",
        f"time: {json.dumps(event.get('time') or {}, ensure_ascii=True)}",
        f"participants: {', '.join(participants) if participants else 'none'}",
        f"evidence_sentence_ids: {event.get('evidence', {}).get('sentence_ids') or []}",
    ]
    return "<br>".join(_html_escape(line) for line in lines)


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
        for event in graph.get("events") or []:
            node_id = f"{chunk.get('chunk_id')}::event::{event.get('event_id')}"
            nodes.append(
                {
                    "id": node_id,
                    "label": event.get("event_type") or event.get("event_id") or "EVENT",
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
                edges.append(
                    {
                        "from": src,
                        "to": dst,
                        "label": edge.get("relation") or "TEMPORAL",
                        "color": {"color": "#2c7fb8"},
                        "dashes": True,
                        "arrows": "to",
                    }
                )
        for edge in graph.get("causal_edges") or []:
            src = f"{chunk.get('chunk_id')}::event::{edge.get('from_event')}"
            dst = f"{chunk.get('chunk_id')}::event::{edge.get('to_event')}"
            if edge.get("from_event") in event_map and edge.get("to_event") in event_map:
                edges.append(
                    {
                        "from": src,
                        "to": dst,
                        "label": edge.get("relation") or "CAUSAL",
                        "color": {"color": "#d95f0e"},
                        "width": 2,
                        "arrows": "to",
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
    cols = max(1, int(math.ceil(math.sqrt(len(nodes)))))
    spacing_x = 280
    spacing_y = 180
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
            "barnesHut": {"gravitationalConstant": -3500, "springLength": 180, "springConstant": 0.04, "damping": 0.28},
            "stabilization": {"enabled": True, "iterations": 250},
        }
        node_fixed = False
    return {
        "interaction": {"hover": True, "multiselect": True, "navigationButtons": True},
        "physics": physics,
        "layout": {"improvedLayout": True},
        "nodes": {"shape": "box", "margin": 10, "font": {"size": 16}, "fixed": node_fixed},
        "edges": {"font": {"align": "top"}, "smooth": {"type": "continuous"}},
    }


def _write_html(path: Path, title: str, nodes: list[dict[str, Any]], edges: list[dict[str, Any]], layout_mode: str) -> None:
    _assign_positions(nodes)
    options = _build_options(layout_mode)
    payload_nodes = json.dumps(nodes)
    payload_edges = json.dumps(edges)
    payload_options = json.dumps(options)
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
      #mynetwork {{ width: 100%; height: 880px; background: #ffffff; border-top: 1px solid #ddd; }}
      .legend {{ padding: 8px 20px 12px 20px; color: #444; font-size: 13px; }}
    </style>
  </head>
  <body>
    <div class=\"wrap\">
      <div class=\"title\">{_html_escape(title)}</div>
      <div class=\"legend\">Blue dashed edges = temporal. Orange edges = causal. Drag nodes when layout mode is draggable.</div>
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
    args = parser.parse_args()

    docs = _read_docs(args.input)
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
                _write_html(html_path, f"{doc_id} ({args.graph_view})", nodes, edges, args.html_layout)
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
                    _write_html(html_path, title, nodes, edges, args.html_layout)
                    manifest.write(json.dumps(_manifest_record(doc_id, html_path, args.graph_view, args.html_layout)) + "\n")
            written += 1


if __name__ == "__main__":
    main()
