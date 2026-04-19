#!/usr/bin/env python3
"""
Compute Fleiss' kappa and Krippendorff's alpha (nominal) from ekg_judge_report.json.

Requires multiple raters per item:
- Inter: same doc judged by ≥2 models (stack inter.results).
- Intra: same doc judged by ≥2 temperatures (stack intra.results).

With a single model / single temperature, inter-rater κ and α are not defined; the script
prints category counts and mode agreement instead.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from krippendorff_nominal import krippendorff_alpha_nominal  # noqa: E402

DIMS = [
    "EVENT_COMPLETENESS",
    "EVIDENCE_GROUNDING",
    "EDGE_QUALITY",
    "OVERALL_WINNER",
]


def enc(v: str) -> int:
    if "HYBRID" in v:
        return 2
    if "TIE" in v:
        return 1
    return 0


def fleiss_kappa(matrix: np.ndarray) -> float:
    """matrix: (n_items, n_categories) counts."""
    n_items, K = matrix.shape
    n = int(matrix.sum(axis=1)[0])
    if n <= 1 or n_items == 0:
        return float("nan")
    P_i = (np.sum(matrix**2, axis=1) - n) / (n * (n - 1))
    P_bar = float(np.mean(P_i))
    p_j = np.sum(matrix, axis=0) / (n_items * n)
    P_e = float(np.sum(p_j**2))
    if P_e >= 1.0 - 1e-15:
        return 1.0
    return (P_bar - P_e) / (1.0 - P_e)


def build_count_matrix(
    rows: list[dict[str, Any]],
    *,
    group_key: str,
    group_values: list[str],
    doc_ids: list[str],
) -> dict[str, np.ndarray]:
    """
    rows: each has doc_id, model or temperature, verdict dict.
    group_values: string keys matching how we index (model id or str(temperature)).
    Returns dim -> matrix (len(doc_ids), 3) counts per category.
    """
    lookup: dict[tuple[str, str], dict[str, str]] = {}
    for r in rows:
        v = r.get("verdict")
        if not isinstance(v, dict):
            continue
        did = str(r.get("doc_id") or "")
        if group_key == "temperature":
            g = str(float(r.get("temperature"))) if r.get("temperature") is not None else ""
        else:
            g = str(r.get(group_key) or "")
        lookup[(did, g)] = {k: str(v[k]) for k in DIMS if k in v}

    out: dict[str, np.ndarray] = {}
    n_cats = 3
    for dim in DIMS:
        mat = np.zeros((len(doc_ids), n_cats))
        for i, did in enumerate(doc_ids):
            for gv in group_values:
                key = (did, gv)
                if key not in lookup:
                    continue
                lab = lookup[key].get(dim, "")
                mat[i, enc(lab)] += 1
        out[dim] = mat
    return out


def interpret(k: float) -> str:
    if np.isnan(k):
        return "n/a"
    if k > 0.8:
        return "Almost perfect"
    if k > 0.6:
        return "Substantial"
    if k > 0.4:
        return "Moderate"
    if k > 0.2:
        return "Fair"
    if k > 0.0:
        return "Slight"
    return "Poor / negative"


def _intra_models_list(intra: dict[str, Any], intra_results: list[dict[str, Any]]) -> list[str]:
    raw = intra.get("models")
    if isinstance(raw, list) and raw:
        return list(dict.fromkeys(str(x) for x in raw if x))
    one = intra.get("model")
    if one:
        return [str(one)]
    return list(dict.fromkeys(str(r.get("model")) for r in intra_results if r.get("model")))


def summarize_single_rater(rows: list[dict[str, Any]], doc_ids: list[str]) -> None:
    """Descriptive counts when only one judgment per doc."""
    print("Single rater per doc (one model or one temperature): Fleiss κ and Krippendorff α are not defined.")
    print("(They need ≥2 independent ratings on the same doc × dimension.)\n")
    for dim in DIMS:
        c = Counter()
        for r in rows:
            v = r.get("verdict") or {}
            if dim in v:
                c[v[dim]] += 1
        print(f"  {dim}: {dict(c)}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("report_json", type=Path, nargs="?", default=Path("data/outputs/eval/ekg_judge_report.json"))
    args = ap.parse_args()
    data = json.loads(args.report_json.read_text(encoding="utf-8"))
    doc_ids = list(data.get("doc_ids") or [])

    inter = data.get("inter") or {}
    intra = data.get("intra") or {}
    inter_results = list(inter.get("results") or [])
    intra_results = list(intra.get("results") or [])

    print(f"Report: {args.report_json}")
    print(f"Docs: {len(doc_ids)}\n")

    # --- Inter ---
    models = list(dict.fromkeys(str(r.get("model")) for r in inter_results if r.get("model")))
    print("=== INTER (by model) ===")
    if len(models) < 2:
        summarize_single_rater(inter_results, doc_ids)
    else:
        mat_by_dim = build_count_matrix(inter_results, group_key="model", group_values=models, doc_ids=doc_ids)
        for dim in DIMS:
            mat = mat_by_dim[dim]
            fk = fleiss_kappa(mat)
            # Krippendorff: raters × units
            raters = len(models)
            kd_mat = np.full((raters, len(doc_ids)), np.nan)
            for j, did in enumerate(doc_ids):
                for i, m in enumerate(models):
                    for r in inter_results:
                        if str(r.get("doc_id")) == did and str(r.get("model")) == m and r.get("verdict"):
                            kd_mat[i, j] = enc(str(r["verdict"].get(dim, "")))
            ka = krippendorff_alpha_nominal(kd_mat)
            print(f"  {dim:30s}  Fleiss κ = {fk:.4f} ({interpret(fk)})  |  Krippendorff α = {ka:.4f} ({interpret(ka)})")
        print()

    # --- Intra (per model: temperatures = raters) ---
    print("=== INTRA (by temperature, per model) ===")
    intra_models = _intra_models_list(intra, intra_results)
    if not intra_results:
        print("  No intra results (empty).\n")
    else:
        for im in intra_models:
            rows_m = [r for r in intra_results if str(r.get("model")) == im]
            temps = sorted({float(r.get("temperature")) for r in rows_m if r.get("temperature") is not None})
            print(f"  --- model: {im} ---")
            if len(temps) < 2:
                summarize_single_rater(rows_m, doc_ids)
                continue
            temp_strs = [str(float(t)) for t in temps]
            mat_by_dim = build_count_matrix(
                rows_m,
                group_key="temperature",
                group_values=temp_strs,
                doc_ids=doc_ids,
            )
            for dim in DIMS:
                mat = mat_by_dim[dim]
                fk = fleiss_kappa(mat)
                raters = len(temps)
                kd_mat = np.full((raters, len(doc_ids)), np.nan)
                for j, did in enumerate(doc_ids):
                    for i, t in enumerate(temps):
                        for r in rows_m:
                            if str(r.get("doc_id")) == did and float(r.get("temperature", -999)) == t and r.get("verdict"):
                                kd_mat[i, j] = enc(str(r["verdict"].get(dim, "")))
                ka = krippendorff_alpha_nominal(kd_mat)
                print(f"  {dim:30s}  Fleiss κ = {fk:.4f} ({interpret(fk)})  |  Krippendorff α = {ka:.4f} ({interpret(ka)})")
            print()


if __name__ == "__main__":
    main()
