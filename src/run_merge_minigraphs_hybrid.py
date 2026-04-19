#!/usr/bin/env python3
"""
Deterministic merge + hybrid refinement (embeddings retrieval + LLM linking).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if __name__ == "__main__":
    _src = Path(__file__).resolve().parent
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from openai import OpenAI

from agents.config_loader import get_llm_config
from agents.events import MiniGraphMergeAgent
from agents.events.hybrid_merge import HybridMergeRefiner


def _make_client() -> OpenAI:
    cfg = get_llm_config()
    api_key = cfg.get("api_key")
    api_base = cfg.get("api_base")
    if not api_key:
        raise ValueError("Missing AGENT_API_KEY / llm.api_key in config")
    return OpenAI(api_key=api_key, base_url=api_base, timeout=180.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge mini-graphs and add hybrid cross-chunk links/identity via embeddings+LLM."
    )
    parser.add_argument("--input", required=True, help="Input JSONL with chunks[].graph or chunks[].mini_graph.")
    parser.add_argument("--output", required=True, help="Output JSONL merged docs with hybrid refinement.")
    parser.add_argument("--max-docs", type=int, default=None, help="Optional cap on documents.")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--llm-model", default=None, help="Defaults to config llm.model.")
    parser.add_argument("--sim-threshold", type=float, default=0.78, help="Candidate pair cosine cutoff.")
    parser.add_argument("--top-k", type=int, default=4, help="Max retained candidate pairs per event.")
    parser.add_argument("--max-candidates", type=int, default=120, help="Global candidate cap per doc.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-completion-tokens", type=int, default=700)
    parser.add_argument(
        "--dry-run-llm",
        action="store_true",
        help="Skip LLM linking calls (runs deterministic merge + candidate retrieval only).",
    )
    args = parser.parse_args()

    cfg = get_llm_config()
    llm_model = args.llm_model or cfg.get("model") or "gpt-4o"
    client = _make_client()
    merger = MiniGraphMergeAgent()
    refiner = HybridMergeRefiner(
        embedding_model=args.embedding_model,
        llm_model=llm_model,
        similarity_threshold=args.sim_threshold,
        top_k=args.top_k,
        max_candidates=args.max_candidates,
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    written = 0
    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            if args.max_docs is not None and written >= args.max_docs:
                break
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue

            base = merger.merge_document(doc)
            refined_graph, dedupe_stats = refiner.refine(
                doc_id=str(base.get("doc_id") or "doc"),
                merged_graph=base.get("merged_graph") or {},
                client=client,
                apply_llm=not args.dry_run_llm,
            )
            base["merged_graph"] = refined_graph
            base["hybrid_stats"] = {
                **dedupe_stats,
                "num_events_after_hybrid": len((refined_graph.get("events") or [])),
                "num_temporal_edges_after_hybrid": len((refined_graph.get("temporal_edges") or [])),
                "num_causal_edges_after_hybrid": len((refined_graph.get("causal_edges") or [])),
            }

            fout.write(json.dumps(base, default=str) + "\n")
            written += 1

    print(f"Done. Wrote {written} merged+hybrid docs to {args.output}")


if __name__ == "__main__":
    main()
