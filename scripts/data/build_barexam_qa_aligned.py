"""
Build QA-aligned docs + qa.jsonl from reglab/barexam_qa.

Each doc is the gold passage for one benchmark question (gold_idx -> passages.tsv).
This makes retrieval eval possible: question.gold_idx == doc.passage_idx.

Usage:
  python scripts/data/build_barexam_qa_aligned.py --num-qa 100 \\
      --qa-split test \\
      --output-dir data/outputs/legal_rag_benchmark
"""
import argparse
import csv
import json
import os
import sys
import urllib.request

HF = "https://huggingface.co/datasets/reglab/barexam_qa/resolve/main"
PASSAGES_URL = f"{HF}/data/passages/passages.tsv"
QA_URLS = {
    "qa": f"{HF}/data/qa/qa.csv",
    "test": f"{HF}/data/qa/test.csv",
    "train": f"{HF}/data/qa/train.csv",
    "validation": f"{HF}/data/qa/validation.csv",
}
DEFAULT_PASSAGES = "/tmp/barexam_passages.tsv"


def _download(url: str, dest: str) -> None:
    if os.path.exists(dest):
        return
    print(f"Downloading {url} -> {dest}")
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num-qa", type=int, default=100, help="Number of QA rows (docs) to build")
    ap.add_argument("--qa-split", choices=tuple(QA_URLS), default="test",
                    help="QA file split (default: test)")
    ap.add_argument("--passages-tsv", default=DEFAULT_PASSAGES)
    ap.add_argument("--output-dir", default="data/outputs/legal_rag_benchmark")
    args = ap.parse_args()

    csv.field_size_limit(sys.maxsize)
    qa_cache = f"/tmp/barexam_{args.qa_split}.csv"
    _download(QA_URLS[args.qa_split], qa_cache)
    if not os.path.exists(args.passages_tsv):
        _download(PASSAGES_URL, args.passages_tsv)

    qa_rows = []
    with open(qa_cache, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gold = (row.get("gold_idx") or "").strip()
            if not gold:
                continue
            qa_rows.append(row)
            if len(qa_rows) >= args.num_qa:
                break

    if not qa_rows:
        raise SystemExit(f"No QA rows loaded from {qa_cache}")

    need = {r["gold_idx"].strip() for r in qa_rows}
    passages: dict[str, dict] = {}
    with open(args.passages_tsv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            idx = (row.get("idx") or "").strip()
            if idx in need and idx not in passages:
                passages[idx] = row
            if len(passages) == len(need):
                break

    missing = need - set(passages)
    if missing:
        print(f"Warning: {len(missing)} gold passages not found in passages.tsv (first: {next(iter(missing))})")

    os.makedirs(args.output_dir, exist_ok=True)
    docs_path = os.path.join(args.output_dir, "docs.jsonl")
    qa_path = os.path.join(args.output_dir, "qa.jsonl")

    docs_written = 0
    qa_written = 0
    with open(docs_path, "w", encoding="utf-8") as f_docs, open(qa_path, "w", encoding="utf-8") as f_qa:
        for row in qa_rows:
            gold_idx = row["gold_idx"].strip()
            passage = passages.get(gold_idx)
            if not passage:
                continue
            text = (passage.get("text") or "").strip()
            if not text:
                continue

            doc_id = f"barexam_passage_{gold_idx}"
            doc_rec = {
                "case_id": doc_id,
                "case_name": "",
                "docket_number": "",
                "source": f"reglab/barexam_qa:passage:{passage.get('source', '')}",
                "passage_idx": gold_idx,
                "passage_source": passage.get("source") or "",
                "orig_case_id": (passage.get("case_id") or "").strip(),
                "document_text": text,
            }
            f_docs.write(json.dumps(doc_rec) + "\n")
            docs_written += 1

            qa_rec = {
                "qa_idx": row.get("idx") or "",
                "doc_id": doc_id,
                "gold_idx": gold_idx,
                "prompt": row.get("prompt") or "",
                "question": row.get("question") or "",
                "choice_a": row.get("choice_a") or "",
                "choice_b": row.get("choice_b") or "",
                "choice_c": row.get("choice_c") or "",
                "choice_d": row.get("choice_d") or "",
                "answer": row.get("answer") or "",
                "gold_passage": row.get("gold_passage") or text,
                "dataset_split": args.qa_split,
            }
            f_qa.write(json.dumps(qa_rec) + "\n")
            qa_written += 1

    print(f"Wrote {docs_written} docs -> {docs_path}")
    print(f"Wrote {qa_written} QA rows -> {qa_path} (each linked via doc_id / gold_idx)")


if __name__ == "__main__":
    main()
