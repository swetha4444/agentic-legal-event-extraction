#!/usr/bin/env python3
"""Print per-cluster doc counts, %, and optional LLM labels from cluster_llm_eval_ekg.json."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--clustering-results",
        type=Path,
        default=None,
        help="clustering_detailed_results.json (default: repo clustering/results/...)",
    )
    p.add_argument(
        "--llm-eval",
        type=Path,
        default=None,
        help="cluster_llm_eval_ekg.json for cluster_name (default: alongside clustering results)",
    )
    p.add_argument("--repo-root", type=Path, default=None)
    ns = p.parse_args()

    root = ns.repo_root.expanduser().resolve() if ns.repo_root else Path(__file__).resolve().parents[2]
    cr = ns.clustering_results or (root / "clustering/results/clustering_detailed_results.json")
    llm_path = ns.llm_eval or (root / "clustering/results/cluster_validation/cluster_llm_eval_ekg.json")

    data = json.loads(cr.read_text(encoding="utf-8"))
    labels = data["visualization_data"]["ekg"]["labels"]
    counts = Counter(int(x) for x in labels)
    n_total = len(labels)

    meta = data.get("metadata") or {}
    ekg_h = meta.get("ekg_hyperparameters") or {}

    llm_names: dict[int, str] = {}
    if llm_path.is_file():
        ev = json.loads(llm_path.read_text(encoding="utf-8"))
        for k, v in (ev.get("cluster_meta") or {}).items():
            llm_names[int(k)] = str(v.get("cluster_name") or f"Cluster {k}")

    print(f"Source: {cr}")
    print(f"Total documents: {n_total}")
    print(f"EKG embedding: {ekg_h.get('winning_ekg_embedding', '?')}, k={ekg_h.get('winning_ekg_k', '?')}")
    if ekg_h.get("balanced_selection"):
        print(
            f"Balanced selection: min_cluster>={ekg_h.get('ekg_min_cluster_size_fair_phase')} "
            f"(divisor={ekg_h.get('ekg_balance_divisor')})"
        )
    sil = (data.get("ekg") or {}).get("metrics", {}).get("silhouette_score")
    if sil is not None:
        print(f"EKG silhouette (sampled): {float(sil):.4f}")
    print()

    cids = sorted(counts.keys())
    print(f"{'ID':>3}  {'n_docs':>6}  {'%':>6}  LLM cluster name")
    print("-" * 90)
    for cid in cids:
        n = counts[cid]
        pct = 100.0 * n / n_total
        name = llm_names.get(cid, "(run eval_cluster_llm_validation.py for names)")
        print(f"{cid:3d}  {n:6d}  {pct:5.1f}%  {name}")
    print("-" * 90)
    print(f"{'ALL':>3}  {n_total:6d}  100.0%")


if __name__ == "__main__":
    main()
