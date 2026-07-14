#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from graphrag.config import get_llm_config
from graphrag.llm import generate_answer
from graphrag.prompting import build_prompt
from graphrag.retrieval import RetrievalConfig, retrieve
from graphrag.store import GraphStore, build_database


def _parse_args() -> argparse.Namespace:
    llm_cfg = get_llm_config()
    parser = argparse.ArgumentParser(
        description="Run graph-RAG FAQ inference over a dataset JSON file."
    )
    parser.add_argument(
        "--input-json",
        required=True,
        help="Path to FAQ dataset JSON, e.g. ekg_faq_case_legal_10docs_300.json",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL") or llm_cfg["model"],
        help="OpenAI-compatible model name.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL") or llm_cfg["api_base"],
        help="Optional OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--output-dir",
        default=".artifacts",
        help="Directory for local outputs. Must stay inside graph-rag.",
    )
    parser.add_argument("--top-k", type=int, default=32, help="Retriever top-k candidates.")
    parser.add_argument("--seed-limit", type=int, default=24, help="Retriever seed limit.")
    parser.add_argument("--graph-hops", type=int, default=2, help="Retriever graph hops.")
    parser.add_argument(
        "--final-events",
        type=int,
        default=8,
        help="Final per-question event count after doc filtering.",
    )
    parser.add_argument(
        "--max-faqs",
        type=int,
        default=None,
        help="Optional limit for quick debugging.",
    )
    parser.add_argument(
        "--doc-id",
        default=None,
        help="Optional single document id filter for quick debugging.",
    )
    parser.add_argument(
        "--chunk-jsonl",
        default=None,
        help="Optional chunk graph JSONL. If omitted, chunk provenance is not indexed.",
    )
    return parser.parse_args()


def _ensure_inside(root: Path, candidate: Path) -> Path:
    resolved_root = root.resolve()
    resolved_candidate = candidate.resolve()
    if resolved_candidate != resolved_root and resolved_root not in resolved_candidate.parents:
        raise ValueError(f"Refusing to write outside graph-rag: {resolved_candidate}")
    return resolved_candidate


def _load_dataset(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "docs" not in data:
        raise ValueError("Expected dataset JSON with top-level 'docs'.")
    return data


def _resolve_merged_jsonl(project_root: Path, dataset: dict[str, Any]) -> Path:
    docs = dataset.get("docs") or []
    if not docs:
        raise ValueError("Dataset contains no docs.")
    source_rel = docs[0].get("source")
    if not source_rel:
        raise ValueError("Dataset docs[0].source is missing.")
    merged_jsonl = Path(source_rel)
    if not merged_jsonl.is_absolute():
        merged_jsonl = project_root / merged_jsonl
    return merged_jsonl.resolve()


def main() -> int:
    args = _parse_args()
    llm_cfg = get_llm_config()

    script_path = Path(__file__).resolve()
    graph_rag_dir = script_path.parent
    project_root = graph_rag_dir.parent

    output_dir = _ensure_inside(graph_rag_dir, (graph_rag_dir / args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    input_json = Path(args.input_json).resolve()
    dataset = _load_dataset(input_json)
    merged_jsonl = _resolve_merged_jsonl(project_root, dataset)
    chunk_jsonl = Path(args.chunk_jsonl).resolve() if args.chunk_jsonl else None

    model_tag = args.model.replace("/", "_")
    input_stem = input_json.stem
    db_path = _ensure_inside(graph_rag_dir, output_dir / f"faq_{input_stem}.sqlite")
    pred_path = _ensure_inside(
        graph_rag_dir, output_dir / f"faq_{input_stem}_{model_tag}_predictions.jsonl"
    )
    summary_path = _ensure_inside(
        graph_rag_dir, output_dir / f"faq_{input_stem}_{model_tag}_summary.json"
    )

    stats = build_database(
        db_path=db_path,
        merged_jsonl=merged_jsonl,
        chunk_jsonl=chunk_jsonl,
        reset=True,
    )

    store = GraphStore(db_path)
    cfg = RetrievalConfig(
        seed_limit=args.seed_limit,
        top_k=args.top_k,
        graph_hops=args.graph_hops,
    )

    rows: list[dict[str, Any]] = []
    processed = 0

    with pred_path.open("w", encoding="utf-8") as fout:
        for doc in dataset["docs"]:
            target_doc_id = str(doc.get("doc_id") or "")
            if args.doc_id and target_doc_id != args.doc_id:
                continue

            for faq in doc.get("faqs") or []:
                if args.max_faqs is not None and processed >= args.max_faqs:
                    break

                question = str(faq.get("question") or "").strip()
                if not question:
                    continue

                bundle = retrieve(store, question, config=cfg)
                bundle.events = [e for e in bundle.events if e.doc_id == target_doc_id][: args.final_events]
                keep_event_ids = {e.event_id for e in bundle.events}
                bundle.entities = [e for e in bundle.entities if e.get("doc_id") == target_doc_id]
                bundle.edges = [
                    e
                    for e in bundle.edges
                    if e.get("doc_id") == target_doc_id
                    and (e.get("from_event") in keep_event_ids or e.get("to_event") in keep_event_ids)
                ]
                bundle.chunks = [c for c in bundle.chunks if c.get("doc_id") == target_doc_id]
                bundle.summary["target_doc_id"] = target_doc_id
                bundle.summary["faq_id"] = faq.get("faq_id")

                messages = build_prompt(bundle)
                answer = generate_answer(
                    messages,
                    model=args.model,
                    api_key=llm_cfg["api_key"],
                    base_url=args.base_url,
                    temperature=0.0,
                )

                row = {
                    "doc_id": target_doc_id,
                    "faq_id": faq.get("faq_id"),
                    "question": question,
                    "gold_answer": faq.get("answer"),
                    "required_graph_ops": faq.get("required_graph_ops"),
                    "why_ekg_edge": faq.get("why_ekg_edge"),
                    "retrieved_event_ids": [e.event_id for e in bundle.events],
                    "predicted_answer": answer,
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append(row)
                processed += 1
                print(f"[{processed}] {faq.get('faq_id')}", flush=True)

            if args.max_faqs is not None and processed >= args.max_faqs:
                break

    summary = {
        "input_json": str(input_json),
        "merged_jsonl": str(merged_jsonl),
        "chunk_jsonl": str(chunk_jsonl) if chunk_jsonl else None,
        "db_path": str(db_path),
        "predictions_path": str(pred_path),
        "summary_path": str(summary_path),
        "model": args.model,
        "index_stats": stats,
        "num_docs": len(dataset["docs"]),
        "num_predictions": len(rows),
        "doc_filter": args.doc_id,
        "max_faqs": args.max_faqs,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
