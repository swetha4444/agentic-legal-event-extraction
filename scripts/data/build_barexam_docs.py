"""
Build document-level JSONL from the reglab/barexam_qa passage pool
(part of "A Reasoning-Focused Legal Retrieval Benchmark", https://reglab.github.io/legal-rag-benchmarks).

The passage pool (data/passages/passages.tsv) is paragraph-level. We reconstruct
full case "documents" by grouping `caselaw` paragraphs on `case_id` and ordering
them by `absolute_paragraph_id`, then concatenating into `document_text`.

Output records match the pipeline's expectations (run_facts_agent / run_events_agent):
  {case_id, case_name, source, num_paragraphs, document_text}

Usage:
  python scripts/data/build_barexam_docs.py --num-docs 100 \
      --output data/outputs/legal_rag_benchmark/docs.jsonl
"""
import argparse
import csv
import json
import os
import sys
import urllib.request

PASSAGES_URL = (
    "https://huggingface.co/datasets/reglab/barexam_qa/resolve/main/"
    "data/passages/passages.tsv"
)
DEFAULT_CACHE = "/tmp/barexam_passages.tsv"


def _download(url: str, dest: str) -> None:
    print(f"Downloading passage pool to {dest} (~550MB, one-time)...")
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    print("Download complete.")


def _to_int(s: str):
    """Parse float-like ids ('12574997.0') into int; fall back to original string."""
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--passages-tsv", default=DEFAULT_CACHE,
                    help=f"Path to passages.tsv (default {DEFAULT_CACHE}; downloaded if missing)")
    ap.add_argument("--source", default="caselaw",
                    help="Passage source to keep (caselaw|wex|mbe). Default caselaw.")
    ap.add_argument("--num-docs", type=int, default=100, help="Number of case documents to build")
    ap.add_argument("--output", default="data/outputs/legal_rag_benchmark/docs.jsonl")
    args = ap.parse_args()

    if not os.path.exists(args.passages_tsv):
        _download(PASSAGES_URL, args.passages_tsv)

    # First 100 distinct case_ids (by appearance). Gather ALL their paragraphs (robust to
    # non-contiguous ordering), keeping only the chosen cases in memory.
    order: list = []
    cases: dict = {}  # case_id -> list[(abs_para_id, text)]
    csv.field_size_limit(sys.maxsize)

    with open(args.passages_tsv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if (row.get("source") or "").strip() != args.source:
                continue
            cid = (row.get("case_id") or "").strip()
            text = (row.get("text") or "").strip()
            if not cid or not text:
                continue
            if cid not in cases:
                if len(cases) >= args.num_docs:
                    continue  # already have enough distinct cases
                cases[cid] = []
                order.append(cid)
            abs_para = _to_int(row.get("absolute_paragraph_id"))
            cases[cid].append((abs_para if abs_para is not None else 0, text))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    written = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for cid in order:
            paras = sorted(cases[cid], key=lambda x: x[0])
            document_text = "\n\n".join(t for _, t in paras)
            if not document_text.strip():
                continue
            cid_int = _to_int(cid)
            rec = {
                "case_id": f"barexam_{args.source}_{cid_int if cid_int is not None else cid}",
                "case_name": "",
                "docket_number": "",
                "source": f"reglab/barexam_qa:{args.source}",
                "orig_case_id": cid,
                "num_paragraphs": len(paras),
                "document_text": document_text,
            }
            out.write(json.dumps(rec) + "\n")
            written += 1

    print(f"Wrote {written} docs to {args.output}")


if __name__ == "__main__":
    main()
