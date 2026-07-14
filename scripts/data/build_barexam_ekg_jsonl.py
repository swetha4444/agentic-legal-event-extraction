"""
Join bar-exam per-doc EKG JSON (chunks1_doc_ekg_llm/*.json) with chunks.jsonl
into clustering-compatible JSONL: doc_id, chunks, merged_graph.

Usage:
  python scripts/data/build_barexam_ekg_jsonl.py \\
      --ekg-dir data/outputs/legal_rag_benchmark/chunks1_doc_ekg_llm \\
      --chunks-jsonl data/outputs/legal_rag_benchmark/chunks.jsonl \\
      --output data/outputs/legal_rag_benchmark/barexam_doc_graphs.jsonl
"""
import argparse
import json
from pathlib import Path


def _merged_graph(ekg: dict) -> dict:
    causal = list(ekg.get("causal_edges") or [])
    if not causal:
        causal = list(ekg.get("expansion_edges_llm") or ekg.get("expansion_edges") or [])
    return {
        "entities": ekg.get("entities") or [],
        "events": ekg.get("events") or [],
        "temporal_edges": ekg.get("temporal_edges") or [],
        "causal_edges": causal,
        "role_edges": ekg.get("role_edges") or [],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ekg-dir", default="data/outputs/legal_rag_benchmark/chunks1_doc_ekg_llm")
    ap.add_argument("--chunks-jsonl", default="data/outputs/legal_rag_benchmark/chunks.jsonl")
    ap.add_argument("--output", default="data/outputs/legal_rag_benchmark/barexam_doc_graphs.jsonl")
    args = ap.parse_args()

    chunks_by = {}
    with open(args.chunks_jsonl, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            chunks_by[row["doc_id"]] = row

    ekg_dir = Path(args.ekg_dir)
    paths = sorted(ekg_dir.glob("*_doc_ekg_llm.json"))
    if not paths:
        raise SystemExit(f"No EKG files in {ekg_dir}")

    written = 0
    skipped = 0
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        for p in paths:
            ekg = json.loads(p.read_text(encoding="utf-8"))
            doc_id = ekg.get("doc_id") or p.stem.replace("_doc_ekg_llm", "")
            chunk_row = chunks_by.get(doc_id)
            if not chunk_row:
                skipped += 1
                continue
            rec = {
                "doc_id": doc_id,
                "num_chunks": chunk_row.get("num_chunks", len(chunk_row.get("chunks") or [])),
                "chunks": chunk_row.get("chunks") or [],
                "merged_graph": _merged_graph(ekg),
                "merge_stats": ekg.get("metadata") or {},
            }
            out.write(json.dumps(rec, default=str) + "\n")
            written += 1

    print(f"Wrote {written} rows -> {out_path} (skipped {skipped} without chunks)")


if __name__ == "__main__":
    main()
