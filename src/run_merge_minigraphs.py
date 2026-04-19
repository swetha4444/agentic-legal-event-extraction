#!/usr/bin/env python3
"""
Merge chunk mini-graphs into doc-level graphs.

Input: JSONL where each line is one doc with `chunks` and per-chunk `graph` or `mini_graph`.
Output: JSONL where each line is one merged doc graph.
"""

import argparse
import json
import os
import sys

if __name__ == "__main__":
    _src = os.path.dirname(os.path.abspath(__file__))
    if _src not in sys.path:
        sys.path.insert(0, _src)

from agents.events import MiniGraphMergeAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge chunk-level mini-graphs into one doc-level graph.")
    parser.add_argument("--input", required=True, help="Input JSONL from chunk graph extraction.")
    parser.add_argument("--output", required=True, help="Output JSONL with merged doc-level graphs.")
    parser.add_argument("--max-docs", type=int, default=None, help="Optional cap on number of documents.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    merger = MiniGraphMergeAgent()

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
            merged = merger.merge_document(doc)
            fout.write(json.dumps(merged, default=str) + "\n")
            written += 1

    print(f"Done. Wrote {written} merged docs to {args.output}")


if __name__ == "__main__":
    main()

