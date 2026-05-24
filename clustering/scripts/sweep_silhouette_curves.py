#!/usr/bin/env python3
"""
Sweep k for TF-IDF and one fixed EKG embedding; record silhouette and plot both curves.

Scores are within-space only (not comparable across TF-IDF vs EKG); the plot is for shape/trend.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _load_comparison_module():
    here = Path(__file__).resolve().parent / "run_clustering_comparison.py"
    spec = importlib.util.spec_from_file_location("clustering_compare", here)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {here}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _silhouette_for(cc, X: np.ndarray, labels: np.ndarray, mode: str) -> float | None:
    lab = np.asarray(labels).astype(int)
    if len(np.unique(lab)) < 2:
        return None
    if mode == "sklearn":
        s = cc.silhouette_metric_sklearn(X, lab)
        if s is None:
            return float(cc.compute_silhouette_score(X, lab))
        return float(s)
    return float(cc.compute_silhouette_score(X, lab))


def best_silhouette_at_k(
    cc,
    X: np.ndarray,
    k: int,
    restarts: int,
    base_seed: int,
    sil_mode: str,
    min_cluster_size: int,
) -> float | None:
    """Best silhouette over random KMeans restarts at fixed k."""
    best: float | None = None
    for t in range(max(1, restarts)):
        rs = base_seed + t * 9973 + k * 17
        lab, _ = cc.simple_kmeans(X, n_clusters=k, random_state=rs)
        bc = np.bincount(lab.astype(int), minlength=k)
        if int(bc.min()) < max(1, min_cluster_size):
            continue
        s = _silhouette_for(cc, X, lab, sil_mode)
        if s is None:
            continue
        if best is None or s > best:
            best = s
    return best


def main() -> None:
    cc = _load_comparison_module()
    script_here = Path(__file__).resolve()
    env_root = Path(os.environ.get("CLUSTERING_REPO_ROOT", "").strip()).expanduser().resolve() if os.environ.get("CLUSTERING_REPO_ROOT") else None  # noqa: E501
    inferred = env_root if env_root and env_root.is_dir() else cc._repo_root(script_here)

    p = argparse.ArgumentParser(description="Silhouette vs k sweep for TF-IDF and EKG.")
    p.add_argument("--repo-root", type=Path, default=None)
    p.add_argument("--ekg-jsonl", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--k-min", type=int, default=2)
    p.add_argument("--k-max", type=int, default=30)
    p.add_argument("--restarts-per-k", type=int, default=12)
    p.add_argument("--seed", type=int, default=cc.RANDOM_SEED)
    p.add_argument(
        "--silhouette",
        choices=("sampled", "sklearn"),
        default="sampled",
        help="Silhouette metric (sampled matches clustering script default)",
    )
    p.add_argument("--tfidf-max-vocab", type=int, default=2000)
    p.add_argument(
        "--ekg-embedding",
        type=str,
        default="enhanced_standardize",
        help="Name from build_ekg_embedding_candidates (see --list-ekg-embeddings)",
    )
    p.add_argument("--ekg-top-entity-types", type=int, default=40)
    p.add_argument("--ekg-top-event-types", type=int, default=40)
    p.add_argument("--ekg-top-roles", type=int, default=20)
    p.add_argument("--min-cluster-size", type=int, default=1)
    p.add_argument(
        "--list-ekg-embeddings",
        action="store_true",
        help="Print candidate embedding names (needs JSONL to build vocabs) and exit",
    )
    ns = p.parse_args()

    root = ns.repo_root.expanduser().resolve() if ns.repo_root else inferred
    dr_ekg, dr_out = cc._defaults(root)
    ekg_path = ns.ekg_jsonl.expanduser().resolve() if ns.ekg_jsonl else dr_ekg
    out_dir = ns.output_dir.expanduser().resolve() if ns.output_dir else dr_out
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ekg_path.is_file():
        print(f"ERROR: JSONL not found: {ekg_path}", file=sys.stderr)
        sys.exit(1)

    docs = cc.load_documents_and_graphs(ekg_path, limit=ns.max_docs)
    doc_ids = list(docs.keys())
    if len(doc_ids) < 4:
        sys.exit("Need at least ~4 documents for a meaningful sweep.")

    graphs_ordered = [docs[d]["merged_graph"] for d in doc_ids]
    candidates = cc.build_ekg_embedding_candidates(
        graphs_ordered,
        ns.ekg_top_entity_types,
        ns.ekg_top_event_types,
        ns.ekg_top_roles,
    )
    names = [n for n, _ in candidates]
    if ns.list_ekg_embeddings:
        print("EKG embedding candidate names:")
        for n in names:
            print(f"  {n}")
        sys.exit(0)

    ekg_X = None
    for n, X in candidates:
        if n == ns.ekg_embedding:
            ekg_X = X
            break
    if ekg_X is None:
        print(f"Unknown --ekg-embedding {ns.ekg_embedding!r}. Choices:", file=sys.stderr)
        for n in names:
            print(f"  {n}", file=sys.stderr)
        sys.exit(1)

    texts = [docs[doc_id]["text"] for doc_id in doc_ids]
    tfidf = cc.SimpleTFIDF(min_freq=1, max_vocab_size=max(100, int(ns.tfidf_max_vocab)))
    tfidf.fit(texts)
    tfidf_X = np.array([tfidf.transform(t) for t in texts])
    tfidf_X = cc.normalize_embeddings(tfidf_X)

    n = len(doc_ids)
    k_hi = min(int(ns.k_max), n - 1)
    k_lo = max(2, int(ns.k_min))
    ks: List[int] = []
    tfidf_curve: List[float | None] = []
    ekg_curve: List[float | None] = []

    print(f"Documents: {n}, k ∈ [{k_lo}, {k_hi}], restarts/k={ns.restarts_per_k}, sil={ns.silhouette}")
    print(f"EKG matrix: {ns.ekg_embedding}, shape={ekg_X.shape}; TF-IDF shape={tfidf_X.shape}")

    for k in range(k_lo, k_hi + 1):
        ks.append(k)
        ts = best_silhouette_at_k(
            cc,
            tfidf_X,
            k,
            ns.restarts_per_k,
            ns.seed,
            ns.silhouette,
            ns.min_cluster_size,
        )
        es = best_silhouette_at_k(
            cc,
            ekg_X,
            k,
            ns.restarts_per_k,
            ns.seed + 404,
            ns.silhouette,
            ns.min_cluster_size,
        )
        tfidf_curve.append(ts)
        ekg_curve.append(es)
        ts_p = f"{ts:.4f}" if ts is not None else "—"
        es_p = f"{es:.4f}" if es is not None else "—"
        print(f"  k={k:3d}  TF-IDF sil={ts_p}  EKG sil={es_p}")

    payload: Dict[str, Any] = {
        "caveat": (
            "Silhouette values are not comparable across TF-IDF vs EKG spaces; compare curves "
            "within each series (trend vs k), not absolute gap between lines."
        ),
        "metadata": {
            "num_documents": n,
            "k_min": k_lo,
            "k_max": k_hi,
            "restarts_per_k": int(ns.restarts_per_k),
            "silhouette_metric": ns.silhouette,
            "min_cluster_size": int(ns.min_cluster_size),
            "ekg_embedding_name": ns.ekg_embedding,
            "ekg_shape": list(ekg_X.shape),
            "tfidf_shape": list(tfidf_X.shape),
            "tfidf_max_vocab": int(ns.tfidf_max_vocab),
            "random_seed": int(ns.seed),
            "ekg_jsonl": str(ekg_path),
        },
        "k": ks,
        "tfidf_silhouette": [float(x) if x is not None else None for x in tfidf_curve],
        "ekg_silhouette": [float(x) if x is not None else None for x in ekg_curve],
    }

    json_path = out_dir / "silhouette_sweep.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {json_path}")

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5))
        tf_y = [np.nan if v is None else v for v in tfidf_curve]
        ek_y = [np.nan if v is None else v for v in ekg_curve]
        ax.plot(ks, tf_y, marker="o", label="TF-IDF (within TF-IDF space)")
        ax.plot(ks, ek_y, marker="s", label=f"EKG: {ns.ekg_embedding} (within EKG space)")
        ax.set_xlabel("k (number of clusters)")
        ax.set_ylabel(f"Silhouette ({ns.silhouette})")
        ax.set_title("Silhouette vs k (two separate representations — do not compare y-values across lines)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        png_path = out_dir / "silhouette_sweep.png"
        fig.tight_layout()
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        print(f"Wrote {png_path}")
    except ImportError:
        print("matplotlib not installed; skipped PNG (install matplotlib for silhouette_sweep.png)")


if __name__ == "__main__":
    main()
