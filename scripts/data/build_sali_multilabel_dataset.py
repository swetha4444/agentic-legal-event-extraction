#!/usr/bin/env python3
"""
Build a multi-label SALI/LMSS training dataset (raw text + linearized EKG).

Design goals (see project discussion):
- Schema-stable labels from data/sali/lmss_claim_subset_v1.json
- Identical document list for raw vs EKG runs when using --fair_ekg_subset
- Stratified split by exact label-set signature (pure Python, no extra deps)

Outputs JSONL lines:
  {"document_id","text","ekg_text","labels":[...], "split":"train|val|test"}

Corpus snapshot (this repo snapshot):
- ~141 compiled complaint docs (deduped by court+case_id in prior pipeline)
- ~138 silver-labeled without API error
- ~133 EKG docs; ~130 overlap with labeled set
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from sali.ekg_linearize import linearize_ekg_row  # noqa: E402


DEFAULT_LABELS = ROOT / "data/sali/lmss_claim_subset_v1.json"
DEFAULT_SILVER = ROOT / "data/outputs/sali_labels/full_run/documents_lmss_labels.jsonl"
DEFAULT_EKG = ROOT / "data/outputs/qwen122b_hybrid_doc_graphs_final.jsonl"


def load_label_vocab(path: Path) -> Tuple[List[str], Dict[str, int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    names = [str(x["pref_label"]) for x in data["labels"]]
    name_to_idx = {n: i for i, n in enumerate(names)}
    return names, name_to_idx


def discover_compiled_paths(scan_dir: Path) -> List[Path]:
    by_parent: Dict[Path, Path] = {}
    for p in scan_dir.rglob("*_compiled.jsonl"):
        if not p.is_file():
            continue
        parent = p.parent
        prev = by_parent.get(parent)
        if prev is None or p.stat().st_mtime > prev.stat().st_mtime:
            by_parent[parent] = p
    return sorted(by_parent.values())


def load_doc_texts(compiled_paths: Iterable[Path]) -> Dict[str, str]:
    """doc_id -> longest document_text seen (dedupe)."""
    out: Dict[str, str] = {}
    for path in compiled_paths:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                did = str(obj.get("case_id") or obj.get("title") or "").strip()
                txt = (obj.get("document_text") or "").strip()
                if not did or not txt:
                    continue
                if len(txt) > len(out.get(did, "")):
                    out[did] = txt
    return out


def load_silver_labels(path: Path, iri_to_name: Dict[str, str]) -> Dict[str, List[str]]:
    """document_id -> pref_label names (from IRIs via whitelist)."""
    out: Dict[str, List[str]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("error"):
                continue
            did = str(row.get("document_id") or "").strip()
            if not did:
                continue
            names: List[str] = []
            # lmss_iris only in this file
            for iri in row.get("lmss_iris") or []:
                nm = iri_to_name.get(str(iri).strip())
                if nm and nm not in names:
                    names.append(nm)
            out[did] = names
    return out


def load_manual_accept_labels(path: Path) -> Dict[str, List[str]]:
    """document_id -> label_names for items with verdict == accept."""
    data = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, List[str]] = {}
    for it in data.get("items", []):
        if it.get("verdict") != "accept":
            continue
        did = str(it.get("document_id") or "").strip()
        if not did:
            continue
        names = [str(x) for x in (it.get("label_names") or []) if x]
        out[did] = sorted(set(names))
    return out


def load_ekg_map(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            did = str(row.get("doc_id") or "").strip()
            if did:
                out[did] = row
    return out


def stratified_split(
    rows: List[Dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> None:
    rng = random.Random(seed)
    groups: Dict[Tuple[str, ...], List[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        key = tuple(sorted(r.get("labels") or []))
        groups[key].append(i)

    train_idx: Set[int] = set()
    val_idx: Set[int] = set()
    test_idx: Set[int] = set()

    for _key, idxs in groups.items():
        rng.shuffle(idxs)
        n = len(idxs)
        nt = int(round(n * train_ratio))
        nv = int(round(n * val_ratio))
        nt = max(0, min(nt, n))
        nv = max(0, min(nv, n - nt))
        for j, pos in enumerate(idxs):
            if j < nt:
                train_idx.add(pos)
            elif j < nt + nv:
                val_idx.add(pos)
            else:
                test_idx.add(pos)

    for i, r in enumerate(rows):
        if i in train_idx:
            r["split"] = "train"
        elif i in val_idx:
            r["split"] = "val"
        else:
            r["split"] = "test"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SALI multi-label dataset (raw + EKG text)")
    parser.add_argument("--label_json", type=Path, default=DEFAULT_LABELS)
    parser.add_argument(
        "--scan_dir",
        type=Path,
        default=ROOT / "data/outputs/fact_extraction_hybrid",
        help="Root to discover *_compiled.jsonl (newest per folder)",
    )
    parser.add_argument(
        "--labels_from",
        choices=("silver", "manual_accept"),
        default="silver",
        help="silver: documents_lmss_labels.jsonl (no error). manual_accept: manual_review.json verdict accept.",
    )
    parser.add_argument("--silver_jsonl", type=Path, default=DEFAULT_SILVER)
    parser.add_argument(
        "--manual_review_json",
        type=Path,
        default=ROOT / "data/outputs/sali_labels/full_run/manual_review.json",
    )
    parser.add_argument("--ekg_jsonl", type=Path, default=DEFAULT_EKG)
    parser.add_argument("--out_jsonl", type=Path, default=ROOT / "data/outputs/sali_ml/dataset.jsonl")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fair_ekg_subset",
        action="store_true",
        help="Keep only documents that have silver labels AND a non-empty EKG row",
    )
    args = parser.parse_args()

    label_names, _ = load_label_vocab(args.label_json)
    subset = json.loads(args.label_json.read_text(encoding="utf-8"))
    iri_to_name = {x["iri"]: x["pref_label"] for x in subset["labels"]}

    compiled_paths = discover_compiled_paths(args.scan_dir.resolve())
    texts = load_doc_texts(compiled_paths)
    if args.labels_from == "silver":
        label_map = load_silver_labels(args.silver_jsonl, iri_to_name)
    else:
        label_map = load_manual_accept_labels(args.manual_review_json)
    ekg_map = load_ekg_map(args.ekg_jsonl)

    rows: List[Dict[str, Any]] = []
    for did, labs in label_map.items():
        txt = texts.get(did, "").strip()
        if not txt:
            continue
        ekg_row = ekg_map.get(did)
        ekg_txt = linearize_ekg_row(ekg_row) if ekg_row else ""
        if args.fair_ekg_subset and not ekg_row:
            continue
        rows.append(
            {
                "document_id": did,
                "text": txt,
                "ekg_text": ekg_txt,
                "labels": sorted(set(labs)),
            }
        )

    stratified_split(rows, args.train_ratio, args.val_ratio, args.seed)

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as sink:
        for r in rows:
            sink.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "label_json": str(args.label_json),
        "labels_from": args.labels_from,
        "num_label_classes": len(label_names),
        "compiled_files": len(compiled_paths),
        "rows": len(rows),
        "fair_ekg_subset": args.fair_ekg_subset,
        "split_counts": {
            "train": sum(1 for r in rows if r["split"] == "train"),
            "val": sum(1 for r in rows if r["split"] == "val"),
            "test": sum(1 for r in rows if r["split"] == "test"),
        },
        "label_names": label_names,
    }
    (args.out_jsonl.with_suffix(".manifest.json")).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out_jsonl} ({len(rows)} rows)")
    print(json.dumps(manifest["split_counts"], indent=2))


if __name__ == "__main__":
    main()
