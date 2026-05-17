#!/usr/bin/env python3
"""
Merge SALI **supervised** eval JSONs into one fair comparison table.

Fairness rules (enforced):
- Same underlying `dataset_jsonl` / test split → every JSON must report the same `n_test` (and usually `n_val`).
- Same gold labels (silver in JSONL) for all rows; only the **model / input** differs.

Typical inputs (paths are examples; pass yours):
  --eval_raw_json data/outputs/sali_ml/eval_raw_fair.json
  --eval_ekg_linear_json data/outputs/sali_ml/eval_ekg_linear_fair.json
  --eval_gnn_json data/outputs/sali_ml/eval_gnn_fair_ftbert.json

Does not re-run models; only reads eval outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _pick_multilabel(m: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "micro_f1_0p5": m.get("test_default_0p5", {}).get("micro_f1"),
        "micro_f1_tuned": m.get("test_tuned", {}).get("micro_f1"),
        "macro_f1_0p5": m.get("test_default_0p5", {}).get("macro_f1"),
        "macro_f1_tuned": m.get("test_tuned", {}).get("macro_f1"),
    }


def _load(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    p = argparse.ArgumentParser(description="Fair table: raw vs EKG (linear) vs GNN from eval JSONs")
    p.add_argument("--eval_raw_json", type=Path, required=True)
    p.add_argument("--eval_ekg_linear_json", type=Path, required=True)
    p.add_argument("--eval_gnn_json", type=Path, default=None)
    p.add_argument("--out_json", type=Path, default=None)
    args = p.parse_args()

    raw_j = _load(args.eval_raw_json)
    ekg_j = _load(args.eval_ekg_linear_json)
    gnn_j = _load(args.eval_gnn_json)

    for name, j in ("raw", raw_j), ("ekg_linear", ekg_j):
        if j is None:
            raise SystemExit(f"missing {name}")
        if j.get("classification_mode", "multi") != "multi":
            raise SystemExit(f"{name}: expected classification_mode multi, got {j.get('classification_mode')}")

    n_tests = [int(raw_j["n_test"]), int(ekg_j["n_test"])]
    if gnn_j is not None:
        n_tests.append(int(gnn_j["n_test"]))
    if len(set(n_tests)) != 1:
        raise SystemExit(f"Unfair comparison: n_test differs across JSONs: {n_tests}")

    rows: List[Dict[str, Any]] = [
        {
            "model": "Legal-BERT multi-label",
            "input": "raw complaint (text_field=text)",
            "checkpoint": raw_j.get("checkpoint_dir"),
            **_pick_multilabel(raw_j),
        },
        {
            "model": "Legal-BERT multi-label",
            "input": "linearized EKG (text_field=ekg_text)",
            "checkpoint": ekg_j.get("checkpoint_dir"),
            **_pick_multilabel(ekg_j),
        },
    ]
    if gnn_j is not None:
        rows.append(
            {
                "model": "BERT node CLS + 2-layer GCN (graph)",
                "input": "merged_graph from ekg_jsonl",
                "checkpoint": gnn_j.get("checkpoint_dir"),
                **_pick_multilabel(gnn_j),
            }
        )

    out: Dict[str, Any] = {
        "n_test": n_tests[0],
        "n_val": int(raw_j["n_val"]),
        "note": "Same dataset_fair_ekg test split and silver multi-hot labels; micro_f1 is headline for sparse multi-label.",
        "rows": rows,
    }

    path_out = args.out_json or Path("data/outputs/sali_ml/compare_ml_fair.json")
    path_out.parent.mkdir(parents=True, exist_ok=True)
    path_out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(out, indent=2))
    print("wrote:", path_out)


if __name__ == "__main__":
    main()
