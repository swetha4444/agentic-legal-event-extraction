#!/usr/bin/env python3
"""
Compare SALI labeling quality for RAW vs EKG predictions.

Gold source:
  manual_review.json
Uses docs where verdict == "accept" as evaluation set.

Predictions:
  - raw: documents_lmss_labels.jsonl
  - ekg: documents_lmss_labels_ekg.jsonl

Fair comparison: pass `--fair_doc_intersection` so scores use only docs present in **both**
pred files and in gold (EKG is not penalized for docs raw labeled alone).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set

ROOT = Path(__file__).resolve().parents[2]


def load_gold(path: Path) -> Dict[str, Set[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, Set[str]] = {}
    for item in data.get("items", []):
        if item.get("verdict") != "accept":
            continue
        did = str(item.get("document_id") or "").strip()
        if not did:
            continue
        out[did] = set(item.get("label_names") or [])
    return out


def load_preds(path: Path, iri_to_name: Dict[str, str]) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            did = str(row.get("document_id") or "").strip()
            if not did:
                continue
            if row.get("error"):
                # treat API/runtime failures as missing predictions
                continue
            labels = set()
            for iri in row.get("lmss_iris") or []:
                nm = iri_to_name.get(iri)
                if nm:
                    labels.add(nm)
            out[did] = labels
    return out


def score_on_docs(
    gold: Dict[str, Set[str]], pred: Dict[str, Set[str]], doc_ids: List[str]
) -> Dict[str, float]:
    exact = 0
    tp = fp = fn = 0
    for did in doc_ids:
        g = gold.get(did, set())
        p = pred.get(did, set())
        if p == g:
            exact += 1
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    exact_match = exact / len(doc_ids) if doc_ids else 0.0
    return {
        "docs": float(len(doc_ids)),
        "exact_match": exact_match,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


def score(gold: Dict[str, Set[str]], pred: Dict[str, Set[str]]) -> Dict[str, float]:
    doc_ids = sorted(gold.keys())
    exact = 0
    tp = fp = fn = 0
    for did in doc_ids:
        g = gold.get(did, set())
        p = pred.get(did, set())
        if p == g:
            exact += 1
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    exact_match = exact / len(doc_ids) if doc_ids else 0.0
    return {
        "docs": float(len(doc_ids)),
        "exact_match": exact_match,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


def per_label_f1_on_docs(
    gold: Dict[str, Set[str]], pred: Dict[str, Set[str]], doc_ids: List[str]
) -> Dict[str, Dict[str, float]]:
    labels: Set[str] = set()
    for did in doc_ids:
        labels |= gold.get(did, set())
        labels |= pred.get(did, set())
    out: Dict[str, Dict[str, float]] = {}
    for lab in sorted(labels):
        tp = fp = fn = 0
        for did in doc_ids:
            g = lab in gold.get(did, set())
            p = lab in pred.get(did, set())
            tp += int(g and p)
            fp += int((not g) and p)
            fn += int(g and (not p))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        sup = sum(1 for did in doc_ids if lab in gold.get(did, set()))
        out[lab] = {"precision": prec, "recall": rec, "f1": f1, "support": float(sup)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare SALI RAW vs EKG predictions")
    parser.add_argument(
        "--manual_review_json",
        type=Path,
        default=ROOT / "data/outputs/sali_labels/full_run/manual_review.json",
    )
    parser.add_argument(
        "--raw_preds_jsonl",
        type=Path,
        default=ROOT / "data/outputs/sali_labels/full_run/documents_lmss_labels.jsonl",
    )
    parser.add_argument("--ekg_preds_jsonl", type=Path, required=True)
    parser.add_argument(
        "--label_json",
        type=Path,
        default=ROOT / "data/sali/lmss_claim_subset_v1.json",
    )
    parser.add_argument(
        "--out_json",
        type=Path,
        default=ROOT / "data/outputs/sali_labels/full_run/raw_vs_ekg_metrics.json",
    )
    parser.add_argument(
        "--fair_doc_intersection",
        action="store_true",
        help="Score only document_ids that appear in BOTH pred files (and in gold). "
        "Avoids raw being evaluated on docs EKG never labeled.",
    )
    args = parser.parse_args()

    labels = json.loads(args.label_json.read_text(encoding="utf-8")).get("labels", [])
    iri_to_name = {x["iri"]: x.get("pref_label", x["iri"]) for x in labels}

    gold = load_gold(args.manual_review_json)
    raw = load_preds(args.raw_preds_jsonl, iri_to_name)
    ekg = load_preds(args.ekg_preds_jsonl, iri_to_name)

    doc_ids_full = sorted(gold.keys())
    if args.fair_doc_intersection:
        fair_ids = sorted(set(raw.keys()) & set(ekg.keys()) & set(gold.keys()))
        raw_scores = score_on_docs(gold, raw, fair_ids)
        ekg_scores = score_on_docs(gold, ekg, fair_ids)
        raw_per_label = per_label_f1_on_docs(gold, raw, fair_ids)
        ekg_per_label = per_label_f1_on_docs(gold, ekg, fair_ids)
        fair_note = {
            "mode": "intersection_raw_ekg_gold",
            "n_docs_scored": float(len(fair_ids)),
            "n_gold_accept_total": float(len(doc_ids_full)),
        }
    else:
        all_ids = sorted(gold.keys())
        raw_scores = score(gold, raw)
        ekg_scores = score(gold, ekg)
        raw_per_label = per_label_f1_on_docs(gold, raw, all_ids)
        ekg_per_label = per_label_f1_on_docs(gold, ekg, all_ids)
        fair_note = {"mode": "all_gold_accept_docs", "n_docs_scored": float(len(doc_ids_full))}

    result = {
        "gold_docs": len(gold),
        "fairness": fair_note,
        "raw_scores": raw_scores,
        "ekg_scores": ekg_scores,
        "delta_ekg_minus_raw": {
            "exact_match": ekg_scores["exact_match"] - raw_scores["exact_match"],
            "micro_f1": ekg_scores["micro_f1"] - raw_scores["micro_f1"],
            "micro_precision": ekg_scores["micro_precision"] - raw_scores["micro_precision"],
            "micro_recall": ekg_scores["micro_recall"] - raw_scores["micro_recall"],
        },
        "raw_per_label": raw_per_label,
        "ekg_per_label": ekg_per_label,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("gold_docs (accept):", len(gold), "| scored:", int(result["fairness"]["n_docs_scored"]), f"({result['fairness']['mode']})")
    print("raw micro_f1:", round(raw_scores["micro_f1"], 4), "| exact:", round(raw_scores["exact_match"], 4))
    print("ekg micro_f1:", round(ekg_scores["micro_f1"], 4), "| exact:", round(ekg_scores["exact_match"], 4))
    print("delta micro_f1:", round(result["delta_ekg_minus_raw"]["micro_f1"], 4))
    print("wrote:", args.out_json)


if __name__ == "__main__":
    main()
