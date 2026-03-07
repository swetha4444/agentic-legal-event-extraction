#!/usr/bin/env python3
"""
Split sentence-level JSONL into train/holdout by document title.

Why by title:
- Avoids sentence leakage from the same source document across splits.

Input JSONL format (one object per line):
  {"sentence": "...", "label": 0|1|"fact"|"non_fact", "title": "...", ...}

Outputs:
- train JSONL
- holdout JSONL
- stats JSON
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


def normalize_label(value) -> int:
    if value in (0, 1):
        return int(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "fact", "facts"}:
            return 1
        if lowered in {"0", "non_fact", "non-fact", "nonfact"}:
            return 0
    raise ValueError(f"Unsupported label value: {value!r}")


@dataclass
class DocInfo:
    title: str
    line_indexes: List[int]
    fact_count: int
    nonfact_count: int

    @property
    def sentence_count(self) -> int:
        return self.fact_count + self.nonfact_count


def read_jsonl(path: Path) -> Tuple[List[str], Dict[str, DocInfo]]:
    raw_lines: List[str] = []
    docs: Dict[str, DocInfo] = {}

    with path.open("r", encoding="utf-8") as infile:
        for idx, line in enumerate(infile):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            label = normalize_label(obj.get("label"))
            title = str(obj.get("title") or f"__MISSING_TITLE_DOC_{idx}")

            if title not in docs:
                docs[title] = DocInfo(
                    title=title,
                    line_indexes=[],
                    fact_count=0,
                    nonfact_count=0,
                )

            docs[title].line_indexes.append(idx)
            if label == 1:
                docs[title].fact_count += 1
            else:
                docs[title].nonfact_count += 1

            raw_lines.append(line)

    return raw_lines, docs


def objective(
    train_docs_count: int,
    train_fact: int,
    train_nonfact: int,
    target_docs: int,
    target_fact: float,
    target_nonfact: float,
) -> float:
    docs_gap = abs(train_docs_count - target_docs) / max(1, target_docs)
    fact_gap = abs(train_fact - target_fact) / max(1.0, target_fact)
    nonfact_gap = abs(train_nonfact - target_nonfact) / max(1.0, target_nonfact)
    return docs_gap + fact_gap + nonfact_gap


def split_docs(
    docs: List[DocInfo],
    train_ratio: float,
    seed: int,
) -> Tuple[List[DocInfo], List[DocInfo]]:
    total_docs = len(docs)
    target_docs = round(total_docs * train_ratio)
    target_docs = min(max(target_docs, 1), max(1, total_docs - 1))

    total_fact = sum(d.fact_count for d in docs)
    total_nonfact = sum(d.nonfact_count for d in docs)
    target_fact = total_fact * train_ratio
    target_nonfact = total_nonfact * train_ratio

    rng = random.Random(seed)
    shuffled = docs[:]
    rng.shuffle(shuffled)

    train_docs: List[DocInfo] = []
    holdout_docs: List[DocInfo] = []
    train_fact = 0
    train_nonfact = 0

    for i, doc in enumerate(shuffled):
        remaining = total_docs - i
        slots_left = target_docs - len(train_docs)

        if slots_left <= 0:
            holdout_docs.append(doc)
            continue
        if slots_left >= remaining:
            train_docs.append(doc)
            train_fact += doc.fact_count
            train_nonfact += doc.nonfact_count
            continue

        score_if_train = objective(
            train_docs_count=len(train_docs) + 1,
            train_fact=train_fact + doc.fact_count,
            train_nonfact=train_nonfact + doc.nonfact_count,
            target_docs=target_docs,
            target_fact=target_fact,
            target_nonfact=target_nonfact,
        )
        score_if_hold = objective(
            train_docs_count=len(train_docs),
            train_fact=train_fact,
            train_nonfact=train_nonfact,
            target_docs=target_docs,
            target_fact=target_fact,
            target_nonfact=target_nonfact,
        )

        if score_if_train <= score_if_hold:
            train_docs.append(doc)
            train_fact += doc.fact_count
            train_nonfact += doc.nonfact_count
        else:
            holdout_docs.append(doc)

    if len(train_docs) != target_docs:
        raise RuntimeError(
            f"Split bug: expected {target_docs} train docs, got {len(train_docs)}."
        )

    return train_docs, holdout_docs


def write_split(
    raw_lines: List[str],
    docs: List[DocInfo],
    output_path: Path,
) -> None:
    allowed = {idx for doc in docs for idx in doc.line_indexes}
    with output_path.open("w", encoding="utf-8") as outfile:
        for i, line in enumerate(raw_lines):
            if i in allowed:
                outfile.write(line)
                outfile.write("\n")


def summarize(docs: List[DocInfo]) -> Dict[str, float]:
    facts = sum(d.fact_count for d in docs)
    nonfacts = sum(d.nonfact_count for d in docs)
    total = facts + nonfacts
    return {
        "docs": len(docs),
        "sentences": total,
        "facts": facts,
        "non_facts": nonfacts,
        "fact_pct": (facts / total * 100.0) if total else 0.0,
        "non_fact_pct": (nonfacts / total * 100.0) if total else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Split JSONL by title into train and holdout sets.")
    parser.add_argument("--input", required=True, help="Input JSONL path.")
    parser.add_argument("--train_out", required=True, help="Output JSONL for training set.")
    parser.add_argument("--holdout_out", required=True, help="Output JSONL for holdout set.")
    parser.add_argument("--stats_out", required=True, help="Output JSON with split summary.")
    parser.add_argument("--train_ratio", type=float, default=0.6, help="Train split ratio (default: 0.6).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    input_path = Path(args.input)
    train_out = Path(args.train_out)
    holdout_out = Path(args.holdout_out)
    stats_out = Path(args.stats_out)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train_ratio must be between 0 and 1.")

    raw_lines, docs_map = read_jsonl(input_path)
    docs = list(docs_map.values())
    if len(docs) < 2:
        raise ValueError("Need at least 2 documents (by title) to split into train/holdout.")

    train_docs, holdout_docs = split_docs(docs, train_ratio=args.train_ratio, seed=args.seed)

    train_out.parent.mkdir(parents=True, exist_ok=True)
    holdout_out.parent.mkdir(parents=True, exist_ok=True)
    stats_out.parent.mkdir(parents=True, exist_ok=True)

    write_split(raw_lines, train_docs, train_out)
    write_split(raw_lines, holdout_docs, holdout_out)

    stats = {
        "input_path": str(input_path),
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "total": summarize(docs),
        "train": summarize(train_docs),
        "holdout": summarize(holdout_docs),
        "train_titles": sorted(d.title for d in train_docs),
        "holdout_titles": sorted(d.title for d in holdout_docs),
    }
    with stats_out.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
        f.write("\n")

    print(f"Wrote train split:   {train_out}")
    print(f"Wrote holdout split: {holdout_out}")
    print(f"Wrote stats:         {stats_out}")
    print(
        "Train docs/sentences: "
        f"{stats['train']['docs']}/{stats['train']['sentences']} "
        f"(fact {stats['train']['fact_pct']:.2f}%, non-fact {stats['train']['non_fact_pct']:.2f}%)"
    )
    print(
        "Holdout docs/sentences: "
        f"{stats['holdout']['docs']}/{stats['holdout']['sentences']} "
        f"(fact {stats['holdout']['fact_pct']:.2f}%, non-fact {stats['holdout']['non_fact_pct']:.2f}%)"
    )


if __name__ == "__main__":
    main()
