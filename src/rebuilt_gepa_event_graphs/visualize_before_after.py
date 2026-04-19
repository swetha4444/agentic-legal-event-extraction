#!/usr/bin/env python3
"""Side-by-side HTML comparison for before/after chunk event-graph predictions."""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

if __name__ == "__main__":
    _src = Path(__file__).resolve().parents[1]
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from rebuilt_event_graphs.visualize_chunk_event_graphs_rebuilt import (  # noqa: E402
    _event_only_graph,
    _full_graph,
    _html_escape,
    _slugify,
    _write_html,
)


def _read_all_docs(path: str) -> list[dict[str, Any]]:
    docs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    if not docs:
        raise ValueError(f"No JSONL rows found in {path}")
    return docs


def _normalize_temporal_edges_in_docs(docs: list[dict[str, Any]]) -> None:
    """Canonicalize temporal edges for visualization: BEFORE -> AFTER (swap), SIMULTANEOUS -> OVERLAP."""
    overlap_aliases = {"OVERLAP", "SIMULTANEOUS", "SAME_TIME", "CONCURRENT"}
    for doc in docs:
        for chunk in doc.get("chunks") or []:
            graph = chunk.get("graph") or {}
            temporal = graph.get("temporal_edges") or []
            for edge in temporal:
                if not isinstance(edge, dict):
                    continue
                relation = str(edge.get("relation") or "").strip().upper()
                if relation == "BEFORE":
                    edge["relation"] = "NEXT"
                elif relation == "AFTER":
                    edge["from_event"], edge["to_event"] = edge.get("to_event"), edge.get("from_event")
                    edge["relation"] = "NEXT"
                elif relation in overlap_aliases:
                    edge["relation"] = "OVERLAP"


def _related_artifact_path(predictions_path: str, suffix: str) -> Path | None:
    path = Path(predictions_path)
    name = path.name
    if not name.endswith("_predictions.jsonl"):
        return None
    candidate = path.with_name(name.replace("_predictions.jsonl", suffix))
    return candidate if candidate.exists() else None


def _load_related_metrics(predictions_path: str) -> dict[str, Any] | None:
    path = _related_artifact_path(predictions_path, "_metrics.json")
    if not path:
        return None
    return json.loads(path.read_text())


def _load_related_judge_map(predictions_path: str) -> dict[tuple[str, str], dict[str, Any]]:
    path = _related_artifact_path(predictions_path, "_judge_details.jsonl")
    if not path:
        return {}
    out = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[(row.get("doc_id"), row.get("chunk_id"))] = row
    return out


def _chunk_summary(chunk: dict[str, Any]) -> dict[str, Any]:
    graph = chunk.get("graph") or {}
    events = graph.get("events") or []
    temporal = graph.get("temporal_edges") or []
    causal = graph.get("causal_edges") or []
    entities = graph.get("entities") or []
    return {
        "parse_error": chunk.get("parse_error"),
        "parse_ok": chunk.get("parse_error") in (None, ""),
        "entity_count": len(entities),
        "event_count": len(events),
        "temporal_count": len(temporal),
        "causal_count": len(causal),
        "parse_retries": chunk.get("parse_retries_used", 0),
        "empty_retries": chunk.get("empty_retries_used", 0),
        "non_empty": bool(events),
    }


def _doc_summary(doc: dict[str, Any]) -> dict[str, Any]:
    chunks = doc.get("chunks") or []
    if not chunks:
        return {
            "num_chunks": 0,
            "valid_chunks": 0,
            "non_empty_chunks": 0,
            "mean_events": 0.0,
            "mean_temporal": 0.0,
            "mean_causal": 0.0,
        }
    chunk_summaries = [_chunk_summary(chunk) for chunk in chunks]
    return {
        "num_chunks": len(chunks),
        "valid_chunks": sum(1 for row in chunk_summaries if row["parse_ok"]),
        "non_empty_chunks": sum(1 for row in chunk_summaries if row["non_empty"]),
        "mean_events": sum(row["event_count"] for row in chunk_summaries) / len(chunk_summaries),
        "mean_temporal": sum(row["temporal_count"] for row in chunk_summaries) / len(chunk_summaries),
        "mean_causal": sum(row["causal_count"] for row in chunk_summaries) / len(chunk_summaries),
    }


def _write_placeholder_html(path: Path, title: str, message: str) -> None:
    html = f"""<html>
  <head>
    <meta charset="utf-8" />
    <style>
      body {{ font-family: Arial, sans-serif; background: #faf8f2; margin: 0; padding: 0; }}
      .wrap {{ padding: 18px 20px; }}
      .title {{ font-size: 18px; font-weight: 600; margin-bottom: 10px; }}
      .msg {{ color: #444; white-space: pre-wrap; }}
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="title">{_html_escape(title)}</div>
      <div class="msg">{_html_escape(message)}</div>
    </div>
  </body>
</html>
"""
    path.write_text(html)


def _write_state_graphs(
    *,
    docs: list[dict[str, Any]],
    out_dir: Path,
    state_name: str,
    graph_view: str,
    html_layout: str,
) -> dict[str, Path]:
    state_dir = out_dir / "graphs" / state_name
    state_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, Path] = {}
    for doc in docs:
        doc_id = doc.get("doc_id") or state_name
        slug = _slugify(doc_id)
        for chunk in doc.get("chunks") or []:
            chunk_id = chunk.get("chunk_id") or "chunk"
            html_path = state_dir / f"{slug}__{_slugify(chunk_id)}__{graph_view}.html"
            title = f"{state_name} / {doc_id} / {chunk_id} ({graph_view})"
            if chunk.get("parse_error"):
                _write_placeholder_html(html_path, title, f"Parse error:\n{chunk.get('parse_error')}")
            else:
                chunk_doc = {"doc_id": doc_id, "chunks": [chunk]}
                if graph_view == "event-only":
                    nodes, edges = _event_only_graph(chunk_doc)
                else:
                    nodes, edges = _full_graph(chunk_doc)
                _write_html(html_path, title, nodes, edges, html_layout)
            mapping[chunk_id] = html_path
    return mapping


def _chunk_order(before_doc: dict[str, Any], after_doc: dict[str, Any]) -> list[str]:
    order = []
    seen = set()
    for doc in (before_doc, after_doc):
        for chunk in doc.get("chunks") or []:
            chunk_id = chunk.get("chunk_id")
            if chunk_id and chunk_id not in seen:
                seen.add(chunk_id)
                order.append(chunk_id)
    return order


def _chunk_map(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {chunk.get("chunk_id"): chunk for chunk in (doc.get("chunks") or []) if chunk.get("chunk_id")}


def _judge_html(judge_row: dict[str, Any] | None) -> str:
    if not judge_row:
        return "<div class=\"judge none\">No judge data</div>"
    if judge_row.get("judge_error"):
        return f"<div class=\"judge bad\">Judge error: {_html_escape(judge_row.get('judge_error'))}</div>"
    score = judge_row.get("normalized_mean_score")
    reason = judge_row.get("reason") or ""
    return (
        f"<div class=\"judge good\">Judge score: {score:.3f}</div>"
        f"<div class=\"judge-reason\">{_html_escape(reason)}</div>"
    )


def _state_card_html(
    *,
    label: str,
    chunk: dict[str, Any] | None,
    graph_path: Path | None,
    report_dir: Path,
    judge_row: dict[str, Any] | None,
) -> str:
    if not chunk:
        return f"<div class=\"state-card\"><div class=\"state-title\">{_html_escape(label)}</div><div class=\"missing\">Missing chunk</div></div>"
    summary = _chunk_summary(chunk)
    graph_rel = os.path.relpath(graph_path, report_dir) if graph_path else ""
    parse_html = (
        "<span class=\"ok\">OK</span>"
        if summary["parse_ok"]
        else f"<span class=\"bad\">{_html_escape(summary['parse_error'])}</span>"
    )
    iframe_html = (
        f"<iframe src=\"{_html_escape(graph_rel)}\" loading=\"lazy\"></iframe>"
        if graph_rel
        else "<div class=\"missing\">No graph HTML</div>"
    )
    return f"""
    <div class="state-card">
      <div class="state-title">{_html_escape(label)}</div>
      <div class="meta-grid">
        <div><strong>Parse</strong>: {parse_html}</div>
        <div><strong>Entities</strong>: {summary['entity_count']}</div>
        <div><strong>Events</strong>: {summary['event_count']}</div>
        <div><strong>Temporal</strong>: {summary['temporal_count']}</div>
        <div><strong>Causal</strong>: {summary['causal_count']}</div>
        <div><strong>Retries</strong>: parse={summary['parse_retries']}, empty={summary['empty_retries']}</div>
      </div>
      {_judge_html(judge_row)}
      <div class="graph-link"><a href="{_html_escape(graph_rel)}" target="_blank" rel="noopener noreferrer">Open graph</a></div>
      {iframe_html}
    </div>
    """


def _summary_table_html(label: str, summary: dict[str, Any], metrics: dict[str, Any] | None) -> str:
    judge_score = metrics.get("judge_score_mean") if metrics else None
    quality_score = metrics.get("quality_score") if metrics else None
    return f"""
    <div class="summary-card">
      <div class="summary-title">{_html_escape(label)}</div>
      <table>
        <tr><td>Chunks</td><td>{summary['num_chunks']}</td></tr>
        <tr><td>Valid chunks</td><td>{summary['valid_chunks']}</td></tr>
        <tr><td>Non-empty chunks</td><td>{summary['non_empty_chunks']}</td></tr>
        <tr><td>Mean events</td><td>{summary['mean_events']:.2f}</td></tr>
        <tr><td>Mean temporal edges</td><td>{summary['mean_temporal']:.2f}</td></tr>
        <tr><td>Mean causal edges</td><td>{summary['mean_causal']:.2f}</td></tr>
        <tr><td>Judge mean</td><td>{'' if judge_score is None else f'{judge_score:.3f}'}</td></tr>
        <tr><td>Quality score</td><td>{'' if quality_score is None else f'{quality_score:.3f}'}</td></tr>
      </table>
    </div>
    """


def _write_report(
    *,
    before_path: str,
    after_path: str,
    output_dir: Path,
    before_doc: dict[str, Any],
    after_doc: dict[str, Any],
    before_metrics: dict[str, Any] | None,
    after_metrics: dict[str, Any] | None,
    before_graphs: dict[str, Path],
    after_graphs: dict[str, Path],
    before_judges: dict[tuple[str, str], dict[str, Any]],
    after_judges: dict[tuple[str, str], dict[str, Any]],
) -> Path:
    before_map = _chunk_map(before_doc)
    after_map = _chunk_map(after_doc)
    chunk_ids = _chunk_order(before_doc, after_doc)
    doc_id = after_doc.get("doc_id") or before_doc.get("doc_id") or "doc"
    before_summary = _doc_summary(before_doc)
    after_summary = _doc_summary(after_doc)

    sections = []
    for chunk_id in chunk_ids:
        before_chunk = before_map.get(chunk_id)
        after_chunk = after_map.get(chunk_id)
        before_judge = before_judges.get((before_doc.get("doc_id"), chunk_id))
        after_judge = after_judges.get((after_doc.get("doc_id"), chunk_id))
        sections.append(
            f"""
            <section class="chunk-section">
              <div class="chunk-header">
                <div class="chunk-title">{_html_escape(chunk_id)}</div>
                <div class="chunk-theme">{_html_escape((after_chunk or before_chunk or {{}}).get('theme') or '')}</div>
              </div>
              <div class="state-grid">
                {_state_card_html(label="Current sequence", chunk=before_chunk, graph_path=before_graphs.get(chunk_id), report_dir=output_dir, judge_row=before_judge)}
                {_state_card_html(label="Next sequence", chunk=after_chunk, graph_path=after_graphs.get(chunk_id), report_dir=output_dir, judge_row=after_judge)}
              </div>
            </section>
            """
        )

    html = f"""<html>
  <head>
    <meta charset="utf-8" />
    <title>{_html_escape(doc_id)} before/after GEPA</title>
    <style>
      body {{ font-family: Arial, sans-serif; margin: 0; background: #f7f4ed; color: #222; }}
      .wrap {{ max-width: 1600px; margin: 0 auto; padding: 20px; }}
      .top {{ display: grid; grid-template-columns: 1.1fr 1fr 1fr; gap: 18px; margin-bottom: 20px; }}
      .hero, .summary-card {{ background: #fff; border: 1px solid #ddd; border-radius: 10px; padding: 16px; }}
      .hero-title {{ font-size: 24px; font-weight: 700; margin-bottom: 8px; }}
      .hero-meta {{ font-size: 14px; color: #555; line-height: 1.5; }}
      .summary-title {{ font-size: 18px; font-weight: 600; margin-bottom: 10px; }}
      table {{ width: 100%; border-collapse: collapse; }}
      td {{ padding: 6px 0; border-bottom: 1px solid #eee; font-size: 14px; }}
      td:first-child {{ color: #555; }}
      .chunk-section {{ background: #fff; border: 1px solid #ddd; border-radius: 10px; padding: 14px; margin-bottom: 20px; }}
      .chunk-header {{ display: flex; justify-content: space-between; align-items: baseline; gap: 12px; margin-bottom: 12px; }}
      .chunk-title {{ font-size: 20px; font-weight: 700; }}
      .chunk-theme {{ color: #666; font-size: 14px; }}
      .state-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
      .state-card {{ border: 1px solid #e5e5e5; border-radius: 8px; padding: 12px; background: #fcfcfc; }}
      .state-title {{ font-size: 17px; font-weight: 600; margin-bottom: 10px; }}
      .meta-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px 12px; font-size: 13px; margin-bottom: 10px; }}
      .ok {{ color: #0a7a34; font-weight: 600; }}
      .bad {{ color: #a01f1f; font-weight: 600; }}
      .judge {{ margin: 6px 0; font-size: 13px; }}
      .judge.good {{ color: #0a7a34; }}
      .judge.bad {{ color: #a01f1f; }}
      .judge.none {{ color: #777; }}
      .judge-reason {{ font-size: 12px; color: #555; margin-bottom: 10px; }}
      .graph-link {{ margin-bottom: 8px; }}
      iframe {{ width: 100%; height: 760px; border: 1px solid #ddd; border-radius: 6px; background: white; }}
      .missing {{ color: #777; font-size: 14px; padding: 10px 0; }}
      a {{ color: #1f5aa6; text-decoration: none; }}
      a:hover {{ text-decoration: underline; }}
      @media (max-width: 1100px) {{
        .top {{ grid-template-columns: 1fr; }}
        .state-grid {{ grid-template-columns: 1fr; }}
      }}
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="top">
        <div class="hero">
          <div class="hero-title">Sequence Comparison</div>
          <div class="hero-meta">
            <div><strong>Document</strong>: {_html_escape(doc_id)}</div>
            <div><strong>Current</strong>: {_html_escape(before_path)}</div>
            <div><strong>Next</strong>: {_html_escape(after_path)}</div>
          </div>
        </div>
        {_summary_table_html("Current sequence", before_summary, before_metrics)}
        {_summary_table_html("Next sequence", after_summary, after_metrics)}
      </div>
      {''.join(sections)}
    </div>
  </body>
</html>
"""
    report_path = output_dir / "index.html"
    report_path.write_text(html)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare before/after GEPA predictions with HTML graph visualizations.")
    parser.add_argument("--before", required=True, help="Path to before-GEPA predictions JSONL")
    parser.add_argument("--after", required=True, help="Path to after-GEPA predictions JSONL")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--graph-view", choices=("event-only", "full"), default="full")
    parser.add_argument("--html-layout", choices=("draggable", "rigid", "physics"), default="draggable")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing output dir: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    before_docs = _read_all_docs(args.before)
    after_docs = _read_all_docs(args.after)
    _normalize_temporal_edges_in_docs(before_docs)
    _normalize_temporal_edges_in_docs(after_docs)
    before_doc = before_docs[0]
    after_doc = after_docs[0]
    before_metrics = _load_related_metrics(args.before)
    after_metrics = _load_related_metrics(args.after)
    before_judges = _load_related_judge_map(args.before)
    after_judges = _load_related_judge_map(args.after)

    before_graphs = _write_state_graphs(
        docs=before_docs,
        out_dir=output_dir,
        state_name="before",
        graph_view=args.graph_view,
        html_layout=args.html_layout,
    )
    after_graphs = _write_state_graphs(
        docs=after_docs,
        out_dir=output_dir,
        state_name="after",
        graph_view=args.graph_view,
        html_layout=args.html_layout,
    )

    report_path = _write_report(
        before_path=args.before,
        after_path=args.after,
        output_dir=output_dir,
        before_doc=before_doc,
        after_doc=after_doc,
        before_metrics=before_metrics,
        after_metrics=after_metrics,
        before_graphs=before_graphs,
        after_graphs=after_graphs,
        before_judges=before_judges,
        after_judges=after_judges,
    )
    print(report_path)


if __name__ == "__main__":
    main()
