#!/usr/bin/env python3
"""
LLM-normalize (mask) each chunk mini-graph for role-style entity labels, then write JSONL.

Output is suitable as input to run_merge_minigraphs.py: each chunk keeps the same graph field
(`graph` or `mini_graph`) but with normalized names/mentions; `masked_graph` duplicates the
masked graph for audit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __name__ == "__main__":
    _src = Path(__file__).resolve().parent
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from openai import OpenAI

from agents.config_loader import get_llm_config
from agents.events.minigraph_mask import mask_minigraph_with_llm


def _pick_graph(chunk: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    if chunk.get("graph"):
        return "graph", chunk["graph"]
    if chunk.get("mini_graph"):
        return "mini_graph", chunk["mini_graph"]
    return None, {}


def _make_client(model_name: str | None) -> tuple[OpenAI, str, float]:
    cfg = get_llm_config()
    api_key = cfg.get("api_key")
    api_base = cfg.get("api_base")
    if not api_key:
        raise ValueError("Missing AGENT_API_KEY / llm.api_key in config")
    model = model_name or cfg.get("model") or "gpt-4o"
    model_lower = model.lower()
    temperature = 1.0 if "gpt-5" in model_lower or model_lower == "gpt5" else float(cfg.get("temperature", 0.0) or 0.0)
    return OpenAI(api_key=api_key, base_url=api_base, timeout=180.0), model, temperature


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-normalize chunk mini-graphs (role-style masking) and write JSONL for merge."
    )
    parser.add_argument("--input", required=True, help="Input JSONL with doc.chunks[].graph or mini_graph.")
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL (e.g. masked.jsonl). Pass this file to run_merge_minigraphs --input.",
    )
    parser.add_argument("--model", default=None, help="Override model from config.")
    parser.add_argument("--max-docs", type=int, default=None, help="Max documents to process.")
    parser.add_argument("--max-completion-tokens", type=int, default=8192, help="LLM completion token budget.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Copy input to output without calling the LLM (no masking).",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    client, model, temperature = _make_client(args.model)
    generated_at = datetime.now(timezone.utc).isoformat()

    written = 0
    chunks_masked = 0
    chunks_failed = 0

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

            out_chunks: list[dict[str, Any]] = []
            for chunk in doc.get("chunks") or []:
                chunk = dict(chunk)
                field, graph = _pick_graph(chunk)
                if not graph or not isinstance(graph, dict):
                    out_chunks.append(chunk)
                    continue
                if not (graph.get("entities") or graph.get("events")):
                    chunk["masked_graph"] = deepcopy(graph)
                    if field:
                        chunk[field] = graph
                    out_chunks.append(chunk)
                    continue

                if args.dry_run:
                    masked = deepcopy(graph)
                    err = None
                else:
                    masked, err = mask_minigraph_with_llm(
                        graph=graph,
                        chunk_text=str(chunk.get("text") or ""),
                        chunk_id=str(chunk.get("chunk_id") or ""),
                        case_name=str(doc.get("case_name") or doc.get("title") or doc.get("doc_id") or ""),
                        client=client,
                        model=model,
                        temperature=temperature,
                        max_completion_tokens=args.max_completion_tokens,
                    )

                chunk["masked_graph"] = masked
                if field:
                    chunk[field] = masked
                if err:
                    chunk["mask_error"] = err
                    chunks_failed += 1
                else:
                    chunks_masked += 1
                out_chunks.append(chunk)

            out_doc = {
                **{k: v for k, v in doc.items() if k != "chunks"},
                "chunks": out_chunks,
                "mask_model": model,
                "masked_at_utc": generated_at,
            }
            fout.write(json.dumps(out_doc, default=str) + "\n")
            written += 1

    print(
        f"Done. Wrote {written} docs to {args.output} "
        f"({chunks_masked} chunks masked ok, {chunks_failed} chunks fell back or errored)."
    )


if __name__ == "__main__":
    main()
