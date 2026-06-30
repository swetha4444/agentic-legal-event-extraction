#!/usr/bin/env python3
"""
Cluster complaints using **SALI LMSS claim tags (LLM-assigned from EKG)** + **Core6D** structure.

Pipeline:
  1. Load EKG JSONL (merged_graph per doc).
  2. SALI labels: load --sali-labels-jsonl OR run --label-sali (LLM on EKG summary,
     same whitelist as scripts/data/label_documents_sali_from_ekg.py).
  3. Features: multi-hot over LMSS claim IRIs || Core6D (optional + hybrid context block).
  4. KMeans with balanced restarts; report silhouette in the clustering space.

Examples:
  # Use existing SALI labels, cluster with SALI + Core6D
  python3 clustering/scripts/run_sali_core6d_clustering.py --repo-root . --k 4 \\
    --sali-labels-jsonl data/outputs/sali_labels/full_run/documents_lmss_labels_ekg.jsonl

  # Label then cluster (needs AGENT_API_KEY / OPENAI_API_KEY)
  python3 clustering/scripts/run_sali_core6d_clustering.py --repo-root . --k 4 --label-sali

  # SALI + Core6D + hybrid EKG context (PCA block)
  python3 clustering/scripts/run_sali_core6d_clustering.py --repo-root . --k 4 --label-sali \\
    --include-hybrid

  # Full stack: SALI tags + Core6D + hybrid + corpus graph neighbor context
  python3 clustering/scripts/run_sali_core6d_clustering.py --repo-root . --k 4 \\
    --sali-labels-jsonl clustering/results/sali_labels_cache_ekg.jsonl \\
    --with-graph-context --graph-knn 12 --fair-trials 48
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


def _load_comparison_module():
    here = Path(__file__).resolve().parent / "run_clustering_comparison.py"
    spec = importlib.util.spec_from_file_location("clustering_compare", here)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {here}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_context_graph_module():
    here = Path(__file__).resolve().parent / "run_ekg_context_graph_clustering.py"
    spec = importlib.util.spec_from_file_location("ekg_ctx_cluster", here)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {here}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_sali_label_module():
    here = Path(__file__).resolve().parents[2] / "scripts" / "data" / "label_documents_sali_from_ekg.py"
    spec = importlib.util.spec_from_file_location("sali_ekg_label", here)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {here}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _repo_root(script_path: Path) -> Path:
    cc = _load_comparison_module()
    return cc._repo_root(script_path)  # type: ignore[attr-defined]


def load_sali_vocab(label_json: Path) -> Tuple[List[str], Dict[str, int], Dict[str, str]]:
    data = json.loads(label_json.read_text(encoding="utf-8"))
    iris = [str(x["iri"]).strip() for x in data["labels"]]
    iri_to_pref = {str(x["iri"]).strip(): str(x.get("pref_label") or "") for x in data["labels"]}
    return iris, {iri: i for i, iri in enumerate(iris)}, iri_to_pref


def load_sali_labels_jsonl(path: Path) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            doc_id = str(row.get("document_id") or row.get("doc_id") or "").strip()
            if not doc_id:
                continue
            iris = row.get("lmss_iris") or row.get("applicable_lmss_iris") or []
            if not isinstance(iris, list):
                iris = []
            out[doc_id] = [str(x).strip() for x in iris if x]
    return out


def build_sali_multihot(
    doc_ids: List[str],
    labels_by_doc: Dict[str, List[str]],
    iri_to_idx: Dict[str, int],
    n_labels: int,
) -> np.ndarray:
    X = np.zeros((len(doc_ids), n_labels), dtype=np.float64)
    for i, did in enumerate(doc_ids):
        for iri in labels_by_doc.get(did, []):
            j = iri_to_idx.get(iri)
            if j is not None:
                X[i, j] = 1.0
    return X


def label_sali_from_ekg(
    rows: List[Dict[str, Any]],
    *,
    label_json: Path,
    config_path: Path,
    model: str | None,
    temperature: float,
    sleep_s: float,
    cache_path: Path | None,
) -> Dict[str, List[str]]:
    sali_mod = _load_sali_label_module()
    runtime = sali_mod.load_runtime_llm_config(config_path)
    api_key = runtime["api_key"]
    api_base = runtime["api_base"]
    model_name = model or runtime["model"] or "gpt4o"
    if not api_key:
        raise SystemExit("Missing API key for --label-sali (AGENT_API_KEY or OPENAI_API_KEY)")

    try:
        from openai import OpenAI
    except ImportError as e:
        raise SystemExit("pip install openai") from e

    whitelist, allowed = sali_mod.load_whitelist(label_json)
    catalog = sali_mod.build_label_catalog_snippet(whitelist)
    client = OpenAI(api_key=api_key, base_url=api_base, timeout=180.0)

    labels_by_doc: Dict[str, List[str]] = {}
    cache_f = cache_path.open("w", encoding="utf-8") if cache_path else None
    for i, row in enumerate(rows, start=1):
        doc_id = str(row.get("doc_id") or "")
        ekg_text = sali_mod.ekg_to_prompt_text(row)
        try:
            iris, note = sali_mod.call_llm(
                client, model_name, doc_id, ekg_text, catalog, allowed, temperature
            )
            rec = {
                "document_id": doc_id,
                "lmss_iris": iris,
                "label_source": "ekg",
                "model": model_name,
                "note": note,
            }
        except Exception as exc:
            iris = []
            rec = {
                "document_id": doc_id,
                "lmss_iris": [],
                "label_source": "ekg",
                "model": model_name,
                "error": str(exc),
            }
        labels_by_doc[doc_id] = iris
        if cache_f is not None:
            cache_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if sleep_s > 0:
            import time

            time.sleep(sleep_s)
        if i % 10 == 0:
            print(f"  SALI labeled {i}/{len(rows)}...", flush=True)
    if cache_f is not None:
        cache_f.close()
        print(f"Wrote SALI cache {cache_path}", flush=True)
    return labels_by_doc


def column_standardize(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0)
    sig = X.std(axis=0) + 1e-8
    return ((X - mu) / sig).astype(np.float64)


def build_fused_embedding(
    cc,
    graphs: List[Dict],
    doc_ids: List[str],
    labels_by_doc: Dict[str, List[str]],
    iri_vocab: List[str],
    iri_to_idx: Dict[str, int],
    *,
    include_hybrid: bool,
    with_graph_context: bool,
    graph_knn: int,
    w_entity: float,
    w_event: float,
    w_role: float,
    top_ent: int,
    top_evt: int,
    top_rl: int,
    hybrid_pca_dim: int,
    sali_weight: float,
    core6d_weight: float,
    hybrid_weight: float,
    graph_ctx_weight: float,
) -> Tuple[np.ndarray, str]:
    X6 = cc.build_core6d_embedding(graphs)
    Xsali = build_sali_multihot(doc_ids, labels_by_doc, iri_to_idx, len(iri_vocab))
    Xh = cc.build_core6d_plus_ekg_context_embedding(
        graphs, top_ent, top_evt, top_rl, context_pca_dim=int(hybrid_pca_dim)
    )
    blocks: List[np.ndarray] = []
    names: List[str] = []

    if sali_weight > 0:
        blocks.append(column_standardize(Xsali) * float(sali_weight))
        names.append(f"sali_mhot_{len(iri_vocab)}")
    if core6d_weight > 0:
        blocks.append(column_standardize(X6.astype(np.float64)) * float(core6d_weight))
        names.append("core6d")
    if (include_hybrid or with_graph_context) and hybrid_weight > 0:
        blocks.append(column_standardize(Xh.astype(np.float64)) * float(hybrid_weight))
        names.append(f"hybrid_pca{int(hybrid_pca_dim)}")
    if with_graph_context and graph_ctx_weight > 0:
        ctx_mod = _load_context_graph_module()
        S = ctx_mod.build_ekg_type_similarity_matrix(
            cc, graphs, w_entity=w_entity, w_event=w_event, w_role=w_role
        )
        knn_idx = ctx_mod.topk_neighbors_from_S(S, int(graph_knn))
        Xctx = ctx_mod.neighbor_context_embedding(Xh.astype(np.float64), knn_idx)
        blocks.append(column_standardize(Xctx) * float(graph_ctx_weight))
        names.append(f"graph_neighbor_knn{int(graph_knn)}")

    if not blocks:
        raise SystemExit("All feature weights are zero")
    Z = np.hstack(blocks).astype(np.float64)
    Z, _, _ = cc.standardize_columns(Z)
    return Z, "+".join(names)


def cluster_balanced(
    cc,
    Z: np.ndarray,
    k: int,
    seed: int,
    fair_trials: int,
    balance_divisor: float,
    max_cluster_frac: float,
) -> Tuple[np.ndarray, float, float | None]:
    n = Z.shape[0]
    min_sz = cc.ekg_balance_min_cluster_size(n, k, float(balance_divisor))
    max_sz = int(max(1, float(max_cluster_frac) * n)) if max_cluster_frac > 0 else n
    best_lab: np.ndarray | None = None
    best_sil = -1e18
    for trial in range(max(1, int(fair_trials))):
        rs = cc.poster_core6d_restart_seed(seed, trial, k)
        lab, _ = cc.kmeans_ekg(Z, n_clusters=k, random_state=rs)
        sizes = cc.cluster_size_dict(lab)
        if min(sizes.values()) < min_sz:
            continue
        if max(sizes.values()) > max_sz:
            continue
        np.random.seed(seed)
        s = float(cc.compute_silhouette_score(Z, lab))
        if s > best_sil:
            best_sil = s
            best_lab = lab
    if best_lab is None:
        for trial in range(max(1, int(fair_trials))):
            rs = cc.poster_core6d_restart_seed(seed, trial, k)
            lab, _ = cc.kmeans_ekg(Z, n_clusters=k, random_state=rs)
            np.random.seed(seed)
            s = float(cc.compute_silhouette_score(Z, lab))
            if s > best_sil:
                best_sil = s
                best_lab = lab
    assert best_lab is not None
    np.random.seed(seed)
    sil_s = float(cc.compute_silhouette_score(Z, best_lab))
    sil_k = cc.silhouette_metric_sklearn(Z, best_lab)
    return best_lab, sil_s, sil_k


def main() -> None:
    script_here = Path(__file__).resolve()
    root_default = _repo_root(script_here)
    p = argparse.ArgumentParser(description="SALI (LLM) tags + Core6D clustering.")
    p.add_argument("--repo-root", type=Path, default=None)
    p.add_argument("--ekg-jsonl", type=Path, default=None)
    p.add_argument("--sali-label-json", type=Path, default=None, help="LMSS whitelist JSON")
    p.add_argument("--sali-labels-jsonl", type=Path, default=None, help="Precomputed document_id -> lmss_iris")
    p.add_argument("--label-sali", action="store_true", help="LLM-label missing docs from EKG before clustering")
    p.add_argument("--sali-model", type=str, default=None)
    p.add_argument("--config", type=Path, default=None, help="config.yaml for SALI labeling API")
    p.add_argument("--label-sleep-s", type=float, default=0.0)
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--include-hybrid", action="store_true", help="Add Core6D+PCA(enhanced EKG) block")
    p.add_argument(
        "--with-graph-context",
        action="store_true",
        help="Add corpus kNN graph on EKG types + mean hybrid of neighbors (TAG-style context)",
    )
    p.add_argument("--graph-knn", type=int, default=12)
    p.add_argument("--w-entity", type=float, default=1.0)
    p.add_argument("--w-event", type=float, default=1.0)
    p.add_argument("--w-role", type=float, default=0.5)
    p.add_argument("--graph-ctx-weight", type=float, default=1.0)
    p.add_argument(
        "--max-cluster-frac",
        type=float,
        default=0.55,
        help="Reject KMeans splits where any cluster has more than this fraction of n (0=off)",
    )
    p.add_argument("--hybrid-pca-dim", type=int, default=16)
    p.add_argument("--ekg-top-entity-types", type=int, default=40)
    p.add_argument("--ekg-top-event-types", type=int, default=40)
    p.add_argument("--ekg-top-roles", type=int, default=20)
    p.add_argument("--sali-weight", type=float, default=1.0)
    p.add_argument("--core6d-weight", type=float, default=1.0)
    p.add_argument("--hybrid-weight", type=float, default=1.0)
    p.add_argument("--fair-trials", type=int, default=24)
    p.add_argument("--balance-divisor", type=float, default=3.0)
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Default: clustering/results/sali_core6d_clustering.json",
    )
    ns = p.parse_args()

    cc = _load_comparison_module()
    root = ns.repo_root.expanduser().resolve() if ns.repo_root else root_default
    cc._load_repo_dotenv(root)
    dr_ekg, dr_out = cc._defaults(root)
    ekg_path = ns.ekg_jsonl.expanduser().resolve() if ns.ekg_jsonl else dr_ekg
    label_json = (
        ns.sali_label_json.expanduser().resolve()
        if ns.sali_label_json
        else (root / "data" / "sali" / "lmss_claim_subset_v1.json")
    )
    config_path = ns.config.expanduser().resolve() if ns.config else (root / "config" / "config.yaml")
    default_out = (
        "sali_core6d_graph_context_clustering.json"
        if ns.with_graph_context
        else "sali_core6d_clustering.json"
    )
    out_path = (
        ns.output_json.expanduser().resolve()
        if ns.output_json
        else (root / "clustering" / "results" / default_out)
    )

    iri_vocab, iri_to_idx, iri_to_pref = load_sali_vocab(label_json)
    docs = cc.load_documents_and_graphs(ekg_path, limit=ns.max_docs)
    doc_ids = sorted(docs.keys())
    graphs = [docs[d].get("merged_graph") or {} for d in doc_ids]
    rows = [docs[d] for d in doc_ids]
    n = len(doc_ids)
    k = max(2, int(ns.k))
    if n < k + 1:
        raise SystemExit(f"Need at least k+1 documents; got {n}")

    labels_by_doc: Dict[str, List[str]] = {}
    if ns.sali_labels_jsonl and ns.sali_labels_jsonl.expanduser().resolve().is_file():
        labels_by_doc = load_sali_labels_jsonl(ns.sali_labels_jsonl.expanduser().resolve())
        print(f"Loaded SALI labels for {len(labels_by_doc)} docs from {ns.sali_labels_jsonl}", flush=True)

    if ns.label_sali:
        cache = out_path.parent / "sali_labels_cache_ekg.jsonl"
        print(f"LLM SALI labeling for {len(rows)} documents...", flush=True)
        new_labels = label_sali_from_ekg(
            rows,
            label_json=label_json,
            config_path=config_path,
            model=ns.sali_model,
            temperature=0.0,
            sleep_s=float(ns.label_sleep_s),
            cache_path=cache,
        )
        labels_by_doc.update(new_labels)
    elif not labels_by_doc:
        print(
            "No SALI labels loaded; using zero SALI vectors. "
            "Pass --sali-labels-jsonl or --label-sali.",
            flush=True,
        )
    missing = [d for d in doc_ids if d not in labels_by_doc]
    if missing:
        print(f"Warning: {len(missing)} docs missing from SALI file (zero vector)", flush=True)

    label_counts = Counter()
    for d in doc_ids:
        label_counts[len(labels_by_doc.get(d, []))] += 1
    print(f"SALI labels per doc (count -> #docs): {dict(sorted(label_counts.items()))}", flush=True)

    with_graph = bool(ns.with_graph_context)
    Z, emb_name = build_fused_embedding(
        cc,
        graphs,
        doc_ids,
        labels_by_doc,
        iri_vocab,
        iri_to_idx,
        include_hybrid=bool(ns.include_hybrid),
        with_graph_context=with_graph,
        graph_knn=int(ns.graph_knn),
        w_entity=float(ns.w_entity),
        w_event=float(ns.w_event),
        w_role=float(ns.w_role),
        top_ent=int(ns.ekg_top_entity_types),
        top_evt=int(ns.ekg_top_event_types),
        top_rl=int(ns.ekg_top_roles),
        hybrid_pca_dim=int(ns.hybrid_pca_dim),
        sali_weight=float(ns.sali_weight),
        core6d_weight=float(ns.core6d_weight),
        hybrid_weight=float(ns.hybrid_weight),
        graph_ctx_weight=float(ns.graph_ctx_weight),
    )

    labels, sil_s, sil_k = cluster_balanced(
        cc,
        Z,
        k,
        int(ns.seed),
        int(ns.fair_trials),
        float(ns.balance_divisor),
        float(ns.max_cluster_frac),
    )
    sizes = dict(sorted(Counter(int(x) for x in labels.tolist()).items()))

    # Per-cluster dominant SALI tags for interpretation
    cluster_sali: Dict[str, List[Dict[str, Any]]] = {}
    for c in range(k):
        mask = labels == c
        cnt: Counter = Counter()
        for i, did in enumerate(doc_ids):
            if not mask[i]:
                continue
            for iri in labels_by_doc.get(did, []):
                cnt[iri] += 1
        top = [
            {"iri": iri, "pref_label": iri_to_pref.get(iri, iri), "count": int(v)}
            for iri, v in cnt.most_common(5)
        ]
        cluster_sali[str(c)] = top

    method_name = (
        "sali_core6d_graph_context"
        if with_graph
        else ("sali_llm_multihot_plus_core6d_hybrid" if ns.include_hybrid else "sali_llm_multihot_plus_core6d")
    )
    payload = {
        "method": method_name,
        "embedding_name": emb_name,
        "with_graph_context": with_graph,
        "graph_knn": int(ns.graph_knn) if with_graph else None,
        "similarity_weights": {"entity": ns.w_entity, "event": ns.w_event, "role": ns.w_role}
        if with_graph
        else None,
        "k": k,
        "seed": int(ns.seed),
        "n_docs": n,
        "sali_vocab_size": len(iri_vocab),
        "sali_label_json": str(label_json),
        "ekg_jsonl": str(ekg_path),
        "silhouette_sampled": sil_s,
        "silhouette_sklearn_full": sil_k,
        "cluster_sizes": sizes,
        "cluster_top_sali_tags": cluster_sali,
        "doc_ids": doc_ids,
        "labels": labels.astype(int).tolist(),
        "assignments": {doc_ids[i]: int(labels[i]) for i in range(n)},
        "sali_iris_by_doc": {d: labels_by_doc.get(d, []) for d in doc_ids},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nCluster sizes: {sizes}", flush=True)
    print(f"Silhouette (sampled): {sil_s:.4f}", flush=True)
    if sil_k is not None:
        print(f"Silhouette (sklearn): {sil_k:.4f}", flush=True)
    print(f"Wrote {out_path}", flush=True)
    eval_method = "sali_core6d_graph" if with_graph else "sali_core6d"
    print("\nLLM eval on these clusters:", flush=True)
    print(
        f"  python3 clustering/scripts/eval_cluster_llm_validation.py --repo-root . "
        f"--method {eval_method} --labels-json {out_path} --membership-mode classification "
        f"--validate-samples-per-cluster 5",
        flush=True,
    )


if __name__ == "__main__":
    main()
