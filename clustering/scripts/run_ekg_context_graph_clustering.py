#!/usr/bin/env python3
"""
EKG-aware clustering with **inter-document context** (corpus-level graph), inspired by
text-attributed graph (TAG) and shallow-GNN practice: topology links documents whose
extracted EKGs overlap in entity/event/role *types*, then clustering uses either
**neighbor-smoothed features** (1-hop mean aggregation, cf. GNN readout) or
**spectral clustering** on a sparse similarity graph (classic graph partitioning).

This addresses the limitation of per-document-only vectors (no cross-doc signal).

Background (for paper text / further reading):
  - Text-attributed graphs (TAGs): NeurIPS 2023 CS-TAG benchmark overview —
    https://neurips.cc/virtual/2023/poster/73479
  - Joint graph–text representation (contrastive): ConGraT (ACL TextGraphs 2024) —
    https://aclanthology.org/2024.textgraphs-1.2.pdf
  - LLM + graph learning survey: arXiv:2412.12456 — https://arxiv.org/abs/2412.12456

Examples:
  python3 clustering/scripts/run_ekg_context_graph_clustering.py --repo-root . --k 4 \\
    --method neighbor_agg --graph-knn 12

  python3 clustering/scripts/run_ekg_context_graph_clustering.py --repo-root . --k 4 \\
    --method spectral --graph-knn 20

  # Optional cluster names (needs AGENT_API_KEY or ANTHROPIC_API_KEY):
  python3 clustering/scripts/run_ekg_context_graph_clustering.py --repo-root . --k 4 \\
    --method neighbor_agg --llm-describe-clusters
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _load_comparison_module():
    here = Path(__file__).resolve().parent / "run_clustering_comparison.py"
    spec = importlib.util.spec_from_file_location("clustering_compare", here)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {here}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _repo_root(script_path: Path) -> Path:
    cc = _load_comparison_module()
    return cc._repo_root(script_path)  # type: ignore[attr-defined]


def _type_label(cc, x: Any) -> str:
    return cc._type_label(x)  # type: ignore[attr-defined]


def _multiset_jaccard(ca: Counter, cb: Counter) -> float:
    if not ca and not cb:
        return 1.0
    keys = set(ca.keys()) | set(cb.keys())
    num = sum(min(ca.get(k, 0), cb.get(k, 0)) for k in keys)
    den = sum(max(ca.get(k, 0), cb.get(k, 0)) for k in keys)
    return float(num) / float(den) if den > 0 else 0.0


def _per_doc_ekg_type_counters(cc, graph: Dict) -> Tuple[Counter, Counter, Counter]:
    entities = graph.get("entities") or []
    events = graph.get("events") or []
    ent_c = Counter(_type_label(cc, e.get("type")) for e in entities)
    evt_c = Counter()
    role_c = Counter()
    for ev in events:
        evt_c[_type_label(cc, ev.get("event_type") or ev.get("type"))] += 1
        for p in ev.get("participants") or []:
            if isinstance(p, dict):
                role_c[_type_label(cc, p.get("role"))] += 1
    return ent_c, evt_c, role_c


def build_ekg_type_similarity_matrix(
    cc,
    graphs: List[Dict],
    *,
    w_entity: float,
    w_event: float,
    w_role: float,
) -> np.ndarray:
    """Dense pairwise similarity in [0,1] from multiset Jaccard on EKG types."""
    n = len(graphs)
    triples = [_per_doc_ekg_type_counters(cc, g) for g in graphs]
    S = np.zeros((n, n), dtype=np.float64)
    wsum = w_entity + w_event + w_role
    if wsum <= 0:
        w_entity = w_event = w_role = 1.0 / 3.0
        wsum = 1.0
    for i in range(n):
        S[i, i] = 1.0
        ei, evi, ri = triples[i]
        for j in range(i + 1, n):
            ej, evj, rj = triples[j]
            s = (
                w_entity * _multiset_jaccard(ei, ej)
                + w_event * _multiset_jaccard(evi, evj)
                + w_role * _multiset_jaccard(ri, rj)
            ) / wsum
            S[i, j] = S[j, i] = s
    return S


def knn_symmetric_adjacency(S: np.ndarray, k: int, *, mutual: bool) -> np.ndarray:
    """Nonnegative affinity from top-k EKG-type similarity (symmetrized OR-kNN, or mutual kNN)."""
    n = S.shape[0]
    k = max(1, min(int(k), n - 1))
    forward = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        row = S[i].copy()
        row[i] = -1.0
        idx = np.argpartition(-row, k - 1)[:k]
        for j in idx:
            if row[j] < 0:
                continue
            forward[i, j] = S[i, j]
    if mutual:
        mask = (forward > 0) & (forward.T > 0)
        W = np.where(mask, S, 0.0)
    else:
        mask = ((forward > 0) | (forward.T > 0)).astype(np.float64)
        W = mask * S
    np.fill_diagonal(W, 0.0)
    deg = W.sum(axis=1)
    iso = deg < 1e-12
    if np.any(iso):
        W[np.diag_indices(n)] = np.where(iso, 1.0, W[np.diag_indices(n)])
    return W


def neighbor_context_embedding(X: np.ndarray, knn_idx: np.ndarray) -> np.ndarray:
    """
    X: (n, d) row features. knn_idx: (n, k) indices of neighbors (excl. self).
    Returns (n, d) mean neighbor X (broadcast GNN mean pool).
    """
    n, d = X.shape
    out = np.zeros((n, d), dtype=np.float64)
    for i in range(n):
        jj = knn_idx[i]
        out[i] = X[jj].mean(axis=0)
    return out


def topk_neighbors_from_S(S: np.ndarray, k: int) -> np.ndarray:
    n = S.shape[0]
    k = max(1, min(int(k), n - 1))
    knn = np.zeros((n, k), dtype=np.int64)
    for i in range(n):
        row = S[i].copy()
        row[i] = -1.0
        part = np.argpartition(-row, k - 1)[:k]
        order = part[np.argsort(-row[part])]
        knn[i] = order
    return knn


def column_standardize(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0)
    sig = X.std(axis=0) + 1e-8
    return ((X - mu) / sig).astype(np.float64)


def _parse_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        return json.loads(text[start:end])
    raise json.JSONDecodeError("no JSON object", text, 0)


def _llm_cluster_blurb(cc, repo_root: Path, cluster_id: int, member_blocks: List[str]) -> Dict[str, Any]:
    joined = "\n\n---\n\n".join(f"Member {i + 1}:\n{b}" for i, b in enumerate(member_blocks))
    prompt = f"""You name a cluster of legal complaints after graph-assisted grouping (EKG overlap + structure).

Cluster ID: {cluster_id}

{joined}

Respond ONLY with JSON:
{{"cluster_name": "<short title>", "cluster_description": "<2 sentences>"}}"""
    raw = cc._llm_coherence_completion(prompt, repo_root, max_tokens=512)
    try:
        return _parse_json_object(raw)
    except json.JSONDecodeError:
        return {"cluster_name": f"Cluster {cluster_id}", "cluster_description": "parse error"}


def run_pipeline(
    *,
    repo_root: Path,
    ekg_jsonl: Path,
    max_docs: int | None,
    k: int,
    seed: int,
    method: str,
    graph_knn: int,
    mutual_knn: bool,
    w_entity: float,
    w_event: float,
    w_role: float,
    top_ent: int,
    top_evt: int,
    top_rl: int,
    hybrid_pca_dim: int,
    fair_trials: int,
    balance_divisor: float,
    spectral_diag_eps: float,
    llm_describe: bool,
) -> Dict[str, Any]:
    cc = _load_comparison_module()
    cc._load_repo_dotenv(repo_root)
    docs = cc.load_documents_and_graphs(ekg_jsonl, limit=max_docs)
    doc_ids = sorted(docs.keys())
    graphs = [docs[d].get("merged_graph") or {} for d in doc_ids]
    texts = [str(docs[d].get("text") or "") for d in doc_ids]
    n = len(graphs)
    if n < k + 1:
        raise SystemExit(f"Need at least k+1={k+1} documents; got {n}")

    print(f"Building EKG type similarity ({n}x{n})...", flush=True)
    S = build_ekg_type_similarity_matrix(
        cc, graphs, w_entity=w_entity, w_event=w_event, w_role=w_role
    )
    knn_idx = topk_neighbors_from_S(S, graph_knn)
    W = knn_symmetric_adjacency(S, graph_knn, mutual=mutual_knn)

    X_hybrid = cc.build_core6d_plus_ekg_context_embedding(
        graphs, top_ent, top_evt, top_rl, context_pca_dim=int(hybrid_pca_dim)
    )
    emb_name = f"core6d_plus_ekgctx_pca{int(hybrid_pca_dim)}"

    results: Dict[str, Any] = {
        "ekg_jsonl": str(ekg_jsonl),
        "n_docs": n,
        "k": k,
        "seed": seed,
        "base_embedding": emb_name,
        "graph_knn": int(graph_knn),
        "mutual_knn": bool(mutual_knn),
        "similarity_weights": {"entity": w_entity, "event": w_event, "role": w_role},
        "balance_divisor": float(balance_divisor),
        "spectral_diag_eps": float(spectral_diag_eps),
        "methods": {},
    }

    def cluster_stats(labels: np.ndarray) -> Dict[int, int]:
        from collections import Counter as C

        return dict(sorted(C(int(x) for x in labels.tolist()).items()))

    if method in ("neighbor_agg", "both"):
        Xctx = neighbor_context_embedding(X_hybrid.astype(np.float64), knn_idx)
        Z = np.hstack([X_hybrid, Xctx.astype(np.float64)])
        best_lbl: np.ndarray | None = None
        best_sil = -1e18
        min_sz = cc.ekg_balance_min_cluster_size(n, k, float(balance_divisor))
        for trial in range(max(1, int(fair_trials))):
            rs = cc.poster_core6d_restart_seed(seed, trial, k)
            lab, _ = cc.kmeans_ekg(Z, n_clusters=k, random_state=rs)
            if min(cc.cluster_size_dict(lab).values()) < min_sz:
                continue
            np.random.seed(seed)
            s = float(cc.compute_silhouette_score(Z, lab))
            if s > best_sil:
                best_sil = s
                best_lbl = lab
        if best_lbl is None:
            for trial in range(max(1, int(fair_trials))):
                rs = cc.poster_core6d_restart_seed(seed, trial, k)
                lab, _ = cc.kmeans_ekg(Z, n_clusters=k, random_state=rs)
                np.random.seed(seed)
                s = float(cc.compute_silhouette_score(Z, lab))
                if s > best_sil:
                    best_sil = s
                    best_lbl = lab
        assert best_lbl is not None
        labels = best_lbl
        sil_s = float(cc.compute_silhouette_score(Z, labels))
        sil_k = cc.silhouette_metric_sklearn(Z, labels)
        results["methods"]["neighbor_agg"] = {
            "description": "concat(base_hybrid, mean_knn_base_hybrid) column-standardized + KMeans",
            "silhouette_sampled_on_Z": sil_s,
            "silhouette_sklearn_full_on_Z": sil_k,
            "cluster_sizes": cluster_stats(labels),
            "labels": labels.astype(int).tolist(),
        }
        labels_na = labels
        Z_na = Z
    else:
        labels_na = None
        Z_na = None

    if method in ("spectral", "both"):
        try:
            from sklearn.cluster import SpectralClustering
        except ImportError as e:
            raise SystemExit("spectral method requires sklearn: pip install scikit-learn") from e
        Wsp = W + float(spectral_diag_eps) * np.eye(n)
        sc = SpectralClustering(
            n_clusters=k,
            affinity="precomputed",
            random_state=seed,
            assign_labels="kmeans",
        )
        labels_sp = sc.fit_predict(Wsp)
        np.random.seed(seed)
        sil_on_hybrid = float(cc.compute_silhouette_score(X_hybrid, labels_sp))
        sil_on_hybrid_sk = cc.silhouette_metric_sklearn(X_hybrid, labels_sp)
        results["methods"]["spectral"] = {
            "description": "SpectralClustering on symmetrized EKG-type kNN affinity W; "
            "silhouette reported on base hybrid embedding for interpretability",
            "silhouette_sampled_on_hybrid": sil_on_hybrid,
            "silhouette_sklearn_full_on_hybrid": sil_on_hybrid_sk,
            "cluster_sizes": cluster_stats(labels_sp),
            "labels": labels_sp.astype(int).tolist(),
        }
        labels_sp_final = labels_sp
    else:
        labels_sp_final = None

    # Primary labels for optional LLM: prefer neighbor_agg if present
    primary = labels_na if labels_na is not None else labels_sp_final
    primary_Z = Z_na if Z_na is not None else X_hybrid

    cluster_meta: Dict[str, Any] = {}
    if llm_describe and primary is not None:
        agent_key = (os.environ.get("AGENT_API_KEY") or "").strip()
        anth_key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
        if not agent_key and not anth_key:
            print("Skipping --llm-describe-clusters: no AGENT_API_KEY / ANTHROPIC_API_KEY", flush=True)
        else:
            centroids = np.zeros((k, primary_Z.shape[1]), dtype=np.float64)
            for c in range(k):
                m = primary == c
                if m.any():
                    centroids[c] = primary_Z[m].mean(axis=0)
            for c in range(k):
                member_ids = [doc_ids[i] for i in range(n) if int(primary[i]) == c]
                dists = []
                for i in range(n):
                    if int(primary[i]) != c:
                        continue
                    dists.append((float(np.linalg.norm(primary_Z[i] - centroids[c])), i))
                dists.sort(key=lambda t: t[0])
                pick_idx = [i for _, i in dists[: min(3, len(dists))]]
                blocks = []
                for i in pick_idx:
                    t = re.sub(r"\s+", " ", texts[i].strip())
                    blocks.append(t[:900] + ("..." if len(t) > 900 else ""))
                print(f"  LLM describe cluster {c} ({len(member_ids)} members)...", flush=True)
                cluster_meta[str(c)] = _llm_cluster_blurb(cc, repo_root, c, blocks)

    results["doc_ids"] = doc_ids
    results["cluster_llm_meta"] = cluster_meta
    return results


def main() -> None:
    script_here = Path(__file__).resolve()
    p = argparse.ArgumentParser(
        description="EKG corpus graph + context aggregation or spectral clustering (TAG-style)."
    )
    p.add_argument("--repo-root", type=Path, default=None)
    p.add_argument("--ekg-jsonl", type=Path, default=None)
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--method",
        choices=("neighbor_agg", "spectral", "both"),
        default="neighbor_agg",
        help="neighbor_agg: GNN-like mean pool of neighbors in hybrid space; spectral: graph cuts on W",
    )
    p.add_argument("--graph-knn", type=int, default=12, help="Top-k EKG-type neighbors per document")
    p.add_argument(
        "--mutual-knn",
        action="store_true",
        help="Use mutual kNN graph (sparser, often cleaner) instead of max symmetrization",
    )
    p.add_argument("--w-entity", type=float, default=1.0)
    p.add_argument("--w-event", type=float, default=1.0)
    p.add_argument("--w-role", type=float, default=0.5)
    p.add_argument("--ekg-top-entity-types", type=int, default=40)
    p.add_argument("--ekg-top-event-types", type=int, default=40)
    p.add_argument("--ekg-top-roles", type=int, default=20)
    p.add_argument("--hybrid-pca-dim", type=int, default=16)
    p.add_argument("--fair-trials", type=int, default=24, help="KMeans restarts for neighbor_agg")
    p.add_argument(
        "--balance-divisor",
        type=float,
        default=3.0,
        help="Min cluster size ~ n/(k*divisor) when picking neighbor_agg KMeans restarts",
    )
    p.add_argument(
        "--spectral-diag-eps",
        type=float,
        default=1e-3,
        help="Tiny identity added to W before spectral embedding (connectivity / numerical stability)",
    )
    p.add_argument(
        "--llm-describe-clusters",
        action="store_true",
        help="After clustering, call LLM once per cluster for name/description (optional keys)",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Default: <repo>/clustering/results/ekg_context_graph_clustering.json",
    )
    ns = p.parse_args()

    cc = _load_comparison_module()
    root = ns.repo_root.expanduser().resolve() if ns.repo_root else _repo_root(script_here)
    dr_ekg, dr_out = cc._defaults(root)
    ekg_path = ns.ekg_jsonl.expanduser().resolve() if ns.ekg_jsonl else dr_ekg
    out_path = (
        ns.output_json.expanduser().resolve()
        if ns.output_json
        else (root / "clustering" / "results" / "ekg_context_graph_clustering.json")
    )

    payload = run_pipeline(
        repo_root=root,
        ekg_jsonl=ekg_path,
        max_docs=ns.max_docs,
        k=max(2, int(ns.k)),
        seed=int(ns.seed),
        method=str(ns.method),
        graph_knn=int(ns.graph_knn),
        mutual_knn=bool(ns.mutual_knn),
        w_entity=float(ns.w_entity),
        w_event=float(ns.w_event),
        w_role=float(ns.w_role),
        top_ent=int(ns.ekg_top_entity_types),
        top_evt=int(ns.ekg_top_event_types),
        top_rl=int(ns.ekg_top_roles),
        hybrid_pca_dim=int(ns.hybrid_pca_dim),
        fair_trials=int(ns.fair_trials),
        balance_divisor=float(ns.balance_divisor),
        spectral_diag_eps=float(ns.spectral_diag_eps),
        llm_describe=bool(ns.llm_describe_clusters),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}", flush=True)
    for name, block in (payload.get("methods") or {}).items():
        ss = block.get("silhouette_sampled_on_Z") or block.get("silhouette_sampled_on_hybrid")
        sk = block.get("silhouette_sklearn_full_on_Z") or block.get("silhouette_sklearn_full_on_hybrid")
        sk_txt = f" sil_sklearn={sk:.4f}" if sk is not None else ""
        print(f"  [{name}] sizes={block.get('cluster_sizes')} sil_sampled={ss:.4f}{sk_txt}", flush=True)


if __name__ == "__main__":
    main()
