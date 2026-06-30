#!/usr/bin/env python3
"""
Semi-automatic cluster evaluation (Steps A–D).

Step A — Describe each cluster (top-k nearest centroid → LLM theme).
Step B — Membership: classification and/or entailment; optional cross-cluster ranking.
Step C — Centroid distance vs confidence correlation + scatter plot data.
Step D — Human borderline queue + optional agreement aggregation.

Example:
  python3 clustering/scripts/eval_cluster_llm_validation.py \\
    --repo-root /path/to/repo --method ekg \\
    --membership-mode both --score-all-clusters
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _load_comparison_module():
    here = Path(__file__).resolve().parent / "run_clustering_comparison.py"
    spec = importlib.util.spec_from_file_location("clustering_compare", here)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {here}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cluster_fields_from_loose_json(text: str) -> Optional[Dict[str, Any]]:
    """Recover cluster_name/description when the model returns truncated JSON."""
    name_m = re.search(r'"cluster_name"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
    desc_m = re.search(r'"cluster_description"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
    if not name_m and not desc_m:
        return None
    out: Dict[str, Any] = {
        "cluster_name": name_m.group(1) if name_m else "",
        "cluster_description": desc_m.group(1) if desc_m else "",
        "common_themes": [],
        "key_indicators": [],
    }
    themes_m = re.search(r'"common_themes"\s*:\s*\[(.*?)\]', text, re.DOTALL)
    if themes_m:
        out["common_themes"] = re.findall(r'"((?:[^"\\]|\\.)*)"', themes_m.group(1))
    return out


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


def _pearson(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3 or float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    def _rank(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(v), dtype=np.float64)
        return ranks

    return _pearson(_rank(x), _rank(y))


def _compute_centroids(embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
    k = int(labels.max()) + 1
    centroids = np.zeros((k, embeddings.shape[1]), dtype=np.float64)
    for c in range(k):
        mask = labels == c
        if int(mask.sum()) > 0:
            centroids[c] = embeddings[mask].mean(axis=0)
    return centroids


def _centroid_distances(embeddings: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    return np.array(
        [float(np.linalg.norm(embeddings[i] - centroids[int(lab)])) for i, lab in enumerate(labels)],
        dtype=np.float64,
    )


def _nearest_to_centroid(idxs: List[int], embeddings: np.ndarray, centroids: np.ndarray, cid: int, k: int) -> List[int]:
    c = centroids[int(cid)]
    return sorted(idxs, key=lambda i: float(np.linalg.norm(embeddings[i] - c)))[:k]


def _load_assignments_from_results(results_path: Path, method: str) -> Tuple[List[str], np.ndarray]:
    data = json.loads(results_path.read_text(encoding="utf-8"))
    block = (data.get("visualization_data") or {}).get(method)
    if not block:
        raise SystemExit(f"No visualization_data.{method} in {results_path}")
    doc_ids = [str(x) for x in block["doc_ids"]]
    labels = np.asarray(block["labels"], dtype=int)
    if len(doc_ids) != len(labels):
        raise SystemExit("doc_ids / labels length mismatch")
    return doc_ids, labels


def _parse_label_assignments(obj: Any, *, submethod: str | None = None) -> Tuple[List[str], np.ndarray]:
    """Load doc_ids + cluster labels from llm_clustering_labels.json, ekg_context_graph_clustering.json, etc."""
    if not isinstance(obj, dict):
        raise SystemExit(f"Unsupported labels format (expected object): {type(obj)}")
    if isinstance(obj.get("doc_ids"), list) and isinstance(obj.get("labels"), list):
        ids = [str(x) for x in obj["doc_ids"]]
        lab = np.asarray(obj["labels"], dtype=int)
        if len(ids) != len(lab):
            raise SystemExit("doc_ids / labels length mismatch in labels JSON")
        return ids, lab
    if "methods" in obj and submethod:
        m = (obj.get("methods") or {}).get(submethod)
        if not m or not isinstance(m.get("labels"), list):
            raise SystemExit(
                f"Labels JSON has no methods.{submethod}.labels; "
                f"try --labels-json-method spectral or neighbor_agg"
            )
        if not isinstance(obj.get("doc_ids"), list):
            raise SystemExit("Labels JSON with methods.* requires top-level doc_ids list")
        ids = [str(x) for x in obj["doc_ids"]]
        lab = np.asarray(m["labels"], dtype=int)
        if len(ids) != len(lab):
            raise SystemExit("doc_ids / labels length mismatch (nested methods.*)")
        return ids, lab
    if "assignments" in obj:
        pairs = sorted(obj["assignments"].items(), key=lambda kv: kv[0])
        return [p[0] for p in pairs], np.asarray([p[1] for p in pairs], dtype=int)
    if isinstance(obj, list):
        pairs = sorted((str(r["doc_id"]), int(r["cluster_id"])) for r in obj)
        return [p[0] for p in pairs], np.asarray([p[1] for p in pairs], dtype=int)
    raise SystemExit("Unsupported labels JSON (need doc_ids+labels, assignments, methods.*.labels, or list)")


def _load_context_graph_module():
    here = Path(__file__).resolve().parent / "run_ekg_context_graph_clustering.py"
    spec = importlib.util.spec_from_file_location("ekg_ctx_cluster", here)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {here}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rebuild_neighbor_agg_embedding_Z(
    cc,
    ctx_mod,
    graphs: List[Dict],
    labels_blob: Dict[str, Any],
    top_ent: int,
    top_evt: int,
    top_rl: int,
) -> Tuple[np.ndarray, str]:
    """Same Z as run_ekg_context_graph_clustering neighbor_agg (hybrid || mean kNN hybrid)."""
    sw = labels_blob.get("similarity_weights") or {}
    w_ent = float(sw.get("entity", 1.0))
    w_evt = float(sw.get("event", 1.0))
    w_role = float(sw.get("role", 0.5))
    graph_knn = int(labels_blob.get("graph_knn", 12))
    base = str(labels_blob.get("base_embedding") or "core6d_plus_ekgctx_pca16")
    hdim = 16
    if "pca" in base:
        suf = base.split("pca")[-1]
        if suf.isdigit():
            hdim = int(suf)
    X_hybrid = cc.build_core6d_plus_ekg_context_embedding(
        graphs, top_ent, top_evt, top_rl, context_pca_dim=hdim
    )
    S = ctx_mod.build_ekg_type_similarity_matrix(
        cc, graphs, w_entity=w_ent, w_event=w_evt, w_role=w_role
    )
    knn_idx = ctx_mod.topk_neighbors_from_S(S, graph_knn)
    Xctx = ctx_mod.neighbor_context_embedding(X_hybrid.astype(np.float64), knn_idx)
    Z = np.hstack([X_hybrid.astype(np.float64), Xctx.astype(np.float64)])
    Z = ctx_mod.column_standardize(Z)
    name = f"{base}+neighbor_knn{graph_knn}"
    return Z.astype(np.float64), name


def _should_use_context_graph_embedding(labels_blob: Optional[Dict[str, Any]], submethod: str) -> bool:
    if not labels_blob:
        return False
    if labels_blob.get("graph_knn") is None:
        return False
    if submethod != "neighbor_agg":
        return False
    return bool((labels_blob.get("methods") or {}).get("neighbor_agg"))


def _load_sali_core6d_module():
    here = Path(__file__).resolve().parent / "run_sali_core6d_clustering.py"
    spec = importlib.util.spec_from_file_location("sali_core6d_cluster", here)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {here}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _should_use_sali_core6d_embedding(
    labels_blob: Optional[Dict[str, Any]], method: str
) -> bool:
    if not labels_blob:
        return False
    if method in ("sali_core6d", "sali_core6d_graph"):
        return True
    m = str(labels_blob.get("method") or "")
    if m.startswith("sali_"):
        return True
    if "sali_iris_by_doc" in labels_blob:
        return True
    en = str(labels_blob.get("embedding_name") or "")
    return en.startswith("sali_mhot")


def _rebuild_sali_core6d_embedding_Z(
    cc,
    sali_mod,
    graphs: List[Dict],
    doc_ids: List[str],
    labels_blob: Dict[str, Any],
    top_ent: int,
    top_evt: int,
    top_rl: int,
    *,
    force_graph_context: bool = False,
) -> Tuple[np.ndarray, str]:
    label_json = Path(str(labels_blob.get("sali_label_json") or ""))
    if not label_json.is_file():
        raise SystemExit(f"SALI label JSON not found: {label_json}")
    iri_vocab, iri_to_idx, _ = sali_mod.load_sali_vocab(label_json)
    labels_by_doc = labels_blob.get("sali_iris_by_doc")
    if not isinstance(labels_by_doc, dict):
        raise SystemExit(
            "SALI+Core6D labels JSON must include sali_iris_by_doc (from run_sali_core6d_clustering.py)"
        )
    labels_by_doc = {str(k): (v if isinstance(v, list) else []) for k, v in labels_by_doc.items()}
    en = str(labels_blob.get("embedding_name") or "")
    with_graph = bool(labels_blob.get("with_graph_context")) or force_graph_context
    if "graph_neighbor" in en:
        with_graph = True
    include_hybrid = "hybrid_pca" in en or with_graph or bool(
        labels_blob.get("method", "").endswith("hybrid")
    )
    hdim = int(labels_blob.get("hybrid_pca_dim") or 16)
    if "hybrid_pca" in en:
        suf = en.split("hybrid_pca")[-1].split("+")[0]
        if suf.isdigit():
            hdim = int(suf)
    graph_knn = int(labels_blob.get("graph_knn") or 12)
    if "graph_neighbor_knn" in en:
        suf = en.split("graph_neighbor_knn")[-1].split("+")[0]
        if suf.isdigit():
            graph_knn = int(suf)
    sw = labels_blob.get("similarity_weights") or {}
    Z, emb_name = sali_mod.build_fused_embedding(
        cc,
        graphs,
        doc_ids,
        labels_by_doc,
        iri_vocab,
        iri_to_idx,
        include_hybrid=include_hybrid,
        with_graph_context=with_graph,
        graph_knn=graph_knn,
        w_entity=float(sw.get("entity", 1.0)),
        w_event=float(sw.get("event", 1.0)),
        w_role=float(sw.get("role", 0.5)),
        top_ent=top_ent,
        top_evt=top_evt,
        top_rl=top_rl,
        hybrid_pca_dim=hdim,
        sali_weight=1.0,
        core6d_weight=1.0,
        hybrid_weight=1.0 if include_hybrid else 0.0,
        graph_ctx_weight=1.0 if with_graph else 0.0,
    )
    return Z.astype(np.float64), emb_name


def _embedding_hints_from_labels_blob(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Merge into clustering metadata so hybrid / winning embedding matches the labeling run."""
    hyp: Dict[str, Any] = {}
    en = obj.get("embedding_name") or obj.get("base_embedding")
    if en and str(en).startswith(("core6d", "enhanced")):
        hyp["winning_ekg_embedding"] = str(en)
    hdim = obj.get("hybrid_pca_dim") or obj.get("ekg_hybrid_context_pca_dim")
    if hdim is None and isinstance(en, str) and "pca" in en:
        suf = en.split("pca")[-1]
        if suf.isdigit():
            hdim = int(suf)
    if hdim is not None:
        hyp["ekg_hybrid_context_pca_dim"] = int(hdim)
    if hyp:
        return {"ekg_hyperparameters": hyp}
    return {}


def _load_assignments_from_labels(path: Path, submethod: str | None = None) -> Tuple[List[str], np.ndarray]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    return _parse_label_assignments(obj, submethod=submethod)


def _rebuild_ekg_embeddings(
    cc,
    graphs: List[Dict],
    emb_name: str,
    top_ent: int,
    top_evt: int,
    top_rl: int,
    hybrid_pca_dim: int | None = None,
) -> np.ndarray:
    if emb_name == "core6d_row_l2":
        return cc.build_core6d_embedding(graphs)
    if emb_name.startswith("core6d_plus_ekgctx_pca"):
        dim = int(hybrid_pca_dim or 16)
        suf = emb_name.split("pca")[-1]
        if suf.isdigit():
            dim = int(suf)
        return cc.build_core6d_plus_ekg_context_embedding(
            graphs, top_ent, top_evt, top_rl, context_pca_dim=dim
        )
    for name, X in cc.build_ekg_embedding_candidates(
        graphs, top_ent, top_evt, top_rl, candidates="full"
    ):
        if name == emb_name:
            return np.asarray(X, dtype=np.float64)
    raise SystemExit(f"Unknown embedding {emb_name!r}")


def _rebuild_tfidf_embeddings(cc, texts: List[str], max_vocab: int) -> np.ndarray:
    tfidf = cc.SimpleTFIDF(min_freq=1, max_vocab_size=max(100, int(max_vocab)))
    tfidf.fit(texts)
    X = np.array([tfidf.transform(t) for t in texts], dtype=np.float32)
    return cc.normalize_embeddings(X).astype(np.float64)


def _excerpt(text: str, max_chars: int) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    return t if len(t) <= max_chars else t[: max_chars - 3].rstrip() + "..."


def _ekg_facts_snippet(cc, graph: Dict) -> str:
    fe = cc.extract_ekg_features(graph)
    et, ev = Counter(), Counter()
    for e in graph.get("entities") or []:
        et[str(e.get("type") or "Unknown")] += 1
    for ev_obj in graph.get("events") or []:
        ev[str(ev_obj.get("event_type") or ev_obj.get("type") or "Unknown")] += 1
    top_ent = ", ".join(f"{k}({v})" for k, v in et.most_common(4))
    top_evt = ", ".join(f"{k}({v})" for k, v in ev.most_common(4))
    return (
        f"EKG: entities={fe['num_entities']}, events={fe['num_events']}, "
        f"temporal={fe['num_temporal_edges']}, causal={fe['num_causal_edges']}, "
        f"density={fe['graph_density']:.3f}; entity types: {top_ent or 'n/a'}; event types: {top_evt or 'n/a'}"
    )


def _doc_block(text: str, graph: Dict, cc, text_chars: int, include_ekg: bool) -> str:
    parts = [_excerpt(text, text_chars)]
    if include_ekg:
        parts.append(_ekg_facts_snippet(cc, graph))
    return "\n".join(parts)


def _signed_confidence(belongs: Optional[bool], confidence: Optional[float]) -> Optional[float]:
    if belongs is None or confidence is None:
        return None
    return float(confidence if belongs else 1.0 - confidence)


def _llm_cluster_description(cc, repo_root: Path, cluster_id: int, member_blocks: List[str]) -> Dict[str, Any]:
    joined = "\n\n---\n\n".join(f"Member {i + 1}:\n{b}" for i, b in enumerate(member_blocks))
    prompt = f"""Interpret a cluster of legal complaints (representative docs nearest centroid).

Cluster ID: {cluster_id}

{joined}

Respond ONLY with JSON:
{{"cluster_name": "<short title>", "cluster_description": "<2-3 sentences>",
  "common_themes": ["..."], "key_indicators": ["..."]}}"""
    raw = cc._llm_coherence_completion(prompt, repo_root, max_tokens=1024)
    try:
        parsed = _parse_json_object(raw)
    except json.JSONDecodeError:
        parsed = _cluster_fields_from_loose_json(raw)
        if parsed is None:
            return {
                "cluster_name": f"Cluster {cluster_id}",
                "cluster_description": raw[:500],
                "common_themes": [],
                "key_indicators": [],
            }
    return {
        "cluster_name": str(parsed.get("cluster_name") or f"Cluster {cluster_id}"),
        "cluster_description": str(parsed.get("cluster_description") or ""),
        "common_themes": parsed.get("common_themes") or [],
        "key_indicators": parsed.get("key_indicators") or [],
    }


def _llm_classification(cc, repo_root: Path, cid: int, name: str, desc: str, doc_id: str, block: str) -> Dict[str, Any]:
    prompt = f"""Cluster {cid}: {name}
Description: {desc}

Document ({doc_id}):
{block}

Does this document belong? JSON only:
{{"belongs": true/false, "confidence": 0.0-1.0, "reasoning": "..."}}"""
    raw = cc._llm_coherence_completion(prompt, repo_root)
    try:
        o = _parse_json_object(raw)
        return {
            "belongs": bool(o.get("belongs")),
            "confidence": float(np.clip(float(o.get("confidence", 0.5)), 0.0, 1.0)),
            "reasoning": o.get("reasoning", ""),
            "method": "classification",
        }
    except (json.JSONDecodeError, TypeError, ValueError):
        return {"belongs": None, "confidence": None, "reasoning": "parse error", "method": "classification"}


def _llm_entailment(cc, repo_root: Path, cid: int, name: str, desc: str, doc_id: str, block: str) -> Dict[str, Any]:
    hypothesis = f"This document belongs to cluster '{name}': {desc}"
    prompt = f"""Natural-language inference for cluster membership.

Hypothesis: {hypothesis}

Premise (document {doc_id}):
{block}

Label entailment of the hypothesis given the premise.
JSON only:
{{"entailment": "entail"|"neutral"|"contradict", "confidence": 0.0-1.0, "reasoning": "..."}}"""
    raw = cc._llm_coherence_completion(prompt, repo_root)
    try:
        o = _parse_json_object(raw)
        ent = str(o.get("entailment", "neutral")).lower()
        conf = float(np.clip(float(o.get("confidence", 0.5)), 0.0, 1.0))
        belongs = ent == "entail"
        if ent == "contradict":
            signed = 1.0 - conf
        elif ent == "entail":
            signed = conf
        else:
            signed = 0.5
        return {
            "entailment": ent,
            "belongs": belongs,
            "confidence": conf,
            "signed_confidence": signed,
            "reasoning": o.get("reasoning", ""),
            "method": "entailment",
        }
    except (json.JSONDecodeError, TypeError, ValueError):
        return {"entailment": None, "belongs": None, "confidence": None, "method": "entailment"}


def _merge_membership(classification: Optional[Dict], entailment: Optional[Dict]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if classification:
        out.update({f"cls_{k}": v for k, v in classification.items() if k != "method"})
    if entailment:
        out.update({f"nli_{k}": v for k, v in entailment.items() if k != "method"})
    cls_s = _signed_confidence(
        classification.get("belongs") if classification else None,
        classification.get("confidence") if classification else None,
    )
    nli_s = entailment.get("signed_confidence") if entailment else None
    if cls_s is not None and nli_s is not None:
        out["membership_score"] = 0.5 * cls_s + 0.5 * float(nli_s)
    elif cls_s is not None:
        out["membership_score"] = cls_s
    elif nli_s is not None:
        out["membership_score"] = float(nli_s)
    else:
        out["membership_score"] = None
    return out


def _short_cluster_label(row: Dict[str, Any], max_len: int = 28) -> str:
    cid = int(row.get("cluster_id", 0))
    name = str(row.get("cluster_name") or f"Cluster {cid}")
    short = name if len(name) <= max_len else name[: max_len - 3].rstrip() + "..."
    return f"C{cid}: {short}"


def _scatter_correlation_stats(rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, Optional[float], Optional[float]]:
    scored = [r for r in rows if r.get("membership_score") is not None]
    if not scored:
        return np.array([]), np.array([]), None, None
    dist = np.array([float(r["centroid_distance"]) for r in scored], dtype=np.float64)
    score = np.array([float(r["membership_score"]) for r in scored], dtype=np.float64)
    return dist, score, _pearson(dist, score), _spearman(dist, score)


def _scatter_distance_axis_scale(dist: np.ndarray) -> Tuple[float, float]:
    """
    Return (x_axis_right, near_centroid_cutoff) for scatter plots.

    Core6D row-L2 distances are typically ~0.05–0.22; hybrid / high-dim Euclidean
    can be much larger. A fixed xlim(0, 0.22) hides all points off the right edge.
    """
    if len(dist) == 0:
        return 0.25, 0.22
    dmax = float(np.max(dist))
    if dmax < 0.6:
        return max(0.28, dmax * 1.15), 0.22
    near = float(np.percentile(dist, 25))
    x_right = max(dmax * 1.12, near * 3.5, 1.0)
    near_cut = max(near, 1e-9)
    return x_right, near_cut


def _write_scatter_plots(
    out_dir: Path,
    method: str,
    rows: List[Dict[str, Any]],
) -> List[str]:
    """Write main + faceted scatter plots; returns paths written."""
    scored = [r for r in rows if r.get("membership_score") is not None]
    if len(scored) < 2:
        return []
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        return []

    dist, score, pearson_r, spearman_r = _scatter_correlation_stats(scored)
    x_right, near_cut = _scatter_distance_axis_scale(dist)

    def _style_axis(ax: Any, title: str) -> None:
        ax.axhline(0.5, color="#999999", linestyle="--", linewidth=0.8, alpha=0.7, label="score = 0.5")
        span_hi = min(near_cut, x_right * 0.35)
        ax.axvspan(0.0, span_hi, color="#e8f5e9", alpha=0.35, zorder=0)
        ax.axhspan(0.7, 1.0, color="#e8f5e9", alpha=0.25, zorder=0)
        ax.axvspan(0.0, span_hi, ymin=0.0, ymax=0.45, color="#ffebee", alpha=0.45, zorder=0)
        ax.set_xlabel("Distance to assigned cluster centroid (embedding space)")
        ax.set_ylabel("LLM membership score (1 = strong fit)")
        ax.set_xlim(0.0, x_right)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)

    written: List[str] = []
    cluster_ids = sorted({int(r["cluster_id"]) for r in scored})
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    # --- Main plot: color by cluster, fit line, mismatch markers ---
    fig, ax = plt.subplots(figsize=(9, 6))
    _style_axis(ax, "Cluster validation: centroid distance vs LLM membership (per document)")
    for i, cid in enumerate(cluster_ids):
        sub = [r for r in scored if int(r["cluster_id"]) == cid]
        xs = [float(r["centroid_distance"]) for r in sub]
        ys = [float(r["membership_score"]) for r in sub]
        color = palette[i % len(palette)]
        lbl = _short_cluster_label(sub[0])
        ax.scatter(xs, ys, c=color, s=90, alpha=0.85, edgecolors="white", linewidths=0.6, label=lbl, zorder=3)
        mismatch = [
            r
            for r in sub
            if float(r["centroid_distance"]) < near_cut and float(r["membership_score"]) < 0.45
        ]
        if mismatch:
            ax.scatter(
                [float(r["centroid_distance"]) for r in mismatch],
                [float(r["membership_score"]) for r in mismatch],
                s=160,
                facecolors="none",
                edgecolors="#111111",
                linewidths=2.0,
                zorder=4,
            )

    if len(dist) >= 2 and float(np.std(dist)) > 1e-12:
        coef = np.polyfit(dist, score, 1)
        x_line = np.linspace(float(dist.min()), float(max(dist.max(), 0.05)), 100)
        ax.plot(x_line, coef[0] * x_line + coef[1], color="#333333", linewidth=1.5, linestyle="-", zorder=2)

    stat_lines = [f"n = {len(scored)} docs (5 sampled per cluster)"]
    if pearson_r is not None:
        stat_lines.append(f"Pearson r = {pearson_r:.3f}")
    if spearman_r is not None:
        stat_lines.append(f"Spearman ρ = {spearman_r:.3f}")
    ax.text(
        0.98,
        0.02,
        "\n".join(stat_lines),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    h, lab = ax.get_legend_handles_labels()
    h.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="none",
            markeredgecolor="#111111",
            markeredgewidth=2,
            markersize=10,
            label="near centroid, low score",
        )
    )
    lab.append("near centroid, low score")
    ax.legend(h, lab, loc="upper right", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    main_path = out_dir / f"scatter_distance_vs_confidence_{method}.png"
    fig.savefig(main_path, dpi=160)
    plt.close(fig)
    written.append(str(main_path))

    # --- Faceted: one panel per cluster ---
    n_c = len(cluster_ids)
    ncols = 2
    nrows = int(np.ceil(n_c / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4 * nrows), squeeze=False)
    for idx, cid in enumerate(cluster_ids):
        ax = axes[idx // ncols][idx % ncols]
        sub = [r for r in scored if int(r["cluster_id"]) == cid]
        xs = np.array([float(r["centroid_distance"]) for r in sub])
        ys = np.array([float(r["membership_score"]) for r in sub])
        color = palette[idx % len(palette)]
        _style_axis(ax, _short_cluster_label(sub[0], max_len=40))
        ax.scatter(xs, ys, c=color, s=100, alpha=0.9, edgecolors="white", linewidths=0.6, zorder=3)
        for r in sub:
            if float(r["centroid_distance"]) < near_cut and float(r["membership_score"]) < 0.45:
                ax.scatter(
                    [float(r["centroid_distance"])],
                    [float(r["membership_score"])],
                    s=180,
                    facecolors="none",
                    edgecolors="#111111",
                    linewidths=2.0,
                    zorder=4,
                )
        if len(xs) >= 2 and float(np.std(xs)) > 1e-12:
            c = np.polyfit(xs, ys, 1)
            xl = np.linspace(float(xs.min()), float(xs.max()), 50)
            ax.plot(xl, c[0] * xl + c[1], color="#333333", linewidth=1.2)
        ax.text(0.03, 0.97, f"k={len(sub)} samples", transform=ax.transAxes, ha="left", va="top", fontsize=8)
    for j in range(n_c, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle("Per-cluster: distance vs LLM membership", fontsize=12, y=1.01)
    fig.tight_layout()
    facet_path = out_dir / f"scatter_distance_vs_confidence_{method}_by_cluster.png"
    fig.savefig(facet_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    written.append(str(facet_path))

    # --- Belongs vs not (shape), color by cluster ---
    fig, ax = plt.subplots(figsize=(9, 6))
    _style_axis(ax, "LLM belongs? (marker) vs distance (color = cluster)")
    for i, cid in enumerate(cluster_ids):
        sub = [r for r in scored if int(r["cluster_id"]) == cid]
        for r in sub:
            belongs = r.get("llm_belongs")
            marker = "o" if belongs is True else "X" if belongs is False else "s"
            ax.scatter(
                float(r["centroid_distance"]),
                float(r["membership_score"]),
                c=palette[i % len(palette)],
                marker=marker,
                s=100 if marker != "X" else 120,
                alpha=0.9,
                edgecolors="white" if marker == "o" else "#333333",
                linewidths=0.6,
                zorder=3,
            )
    ax.scatter([], [], c="gray", marker="o", label="LLM: belongs")
    ax.scatter([], [], c="gray", marker="X", label="LLM: does not belong")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    belongs_path = out_dir / f"scatter_distance_vs_confidence_{method}_belongs.png"
    fig.savefig(belongs_path, dpi=160)
    plt.close(fig)
    written.append(str(belongs_path))

    return written


def _write_scatter_plot(path: Path, rows: List[Dict[str, Any]]) -> bool:
    paths = _write_scatter_plots(path.parent, path.stem.replace("scatter_distance_vs_confidence_", ""), rows)
    return bool(paths)


def _borderline_reason(row: Dict[str, Any], dist_q75: float, dist_q25: float) -> str:
    reasons = []
    conf = row.get("llm_confidence") or row.get("cls_confidence")
    score = row.get("membership_score")
    dist = row.get("centroid_distance", 0.0)
    if conf is not None and abs(float(conf) - 0.5) < 0.15:
        reasons.append("low_confidence_near_boundary")
    if score is not None and score < 0.55:
        reasons.append("low_membership_score")
    if dist >= dist_q75 and score is not None and score >= 0.65:
        reasons.append("far_from_centroid_but_high_confidence")
    if dist <= dist_q25 and score is not None and score < 0.45:
        reasons.append("near_centroid_but_low_confidence")
    if row.get("assigned_cluster_is_top") is False:
        reasons.append("cross_cluster_mismatch")
    return "; ".join(reasons) if reasons else "borderline_sample"


def _aggregate_human_labels(csv_path: Path) -> Dict[str, Any]:
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    filled = [r for r in rows if (r.get("human_belongs") or "").strip().lower() in ("yes", "no", "unsure")]
    if not filled:
        return {"error": "no human_belongs labels found (use yes/no/unsure)"}
    agree = 0
    comparable = 0
    for r in filled:
        human = r["human_belongs"].strip().lower()
        auto = r.get("llm_belongs") or r.get("cls_belongs")
        if auto is None or str(auto).strip() == "":
            continue
        auto_yes = str(auto).lower() in ("true", "1", "yes")
        comparable += 1
        if human == "unsure":
            continue
        if (human == "yes") == auto_yes:
            agree += 1
    return {
        "human_labeled": len(filled),
        "comparable_to_auto": comparable,
        "agreement_count": agree,
        "agreement_rate": (agree / comparable) if comparable else None,
    }


def _write_rubric(path: Path, method: str) -> None:
    path.write_text(
        f"""Human cluster validation ({method})

Fill human_validation_queue_{method}.csv:
  human_belongs: yes | no | unsure
  human_notes: optional rationale

Borderline types in borderline_reason column:
  - low_confidence_near_boundary
  - far_from_centroid_but_high_confidence
  - near_centroid_but_low_confidence
  - cross_cluster_mismatch

After labeling, run:
  python3 eval_cluster_llm_validation.py --aggregate-human path/to/human_validation_queue_{method}.csv
""",
        encoding="utf-8",
    )


def main() -> None:
    cc = _load_comparison_module()
    script_here = Path(__file__).resolve()
    env_root = os.environ.get("CLUSTERING_REPO_ROOT", "").strip()
    inferred = Path(env_root).expanduser().resolve() if env_root else cc._repo_root(script_here)

    p = argparse.ArgumentParser(description="Steps A–D: LLM cluster eval + human borderline queue")
    p.add_argument("--repo-root", type=Path, default=None)
    p.add_argument("--ekg-jsonl", type=Path, default=None)
    p.add_argument("--clustering-results", type=Path, default=None)
    p.add_argument("--labels-json", type=Path, default=None)
    p.add_argument(
        "--labels-json-method",
        type=str,
        default="neighbor_agg",
        help="When labels JSON has methods.<name>.labels (e.g. ekg_context_graph_clustering.json), pick this key",
    )
    p.add_argument(
        "--method",
        choices=("ekg", "tfidf", "llm", "ekg_graph", "sali_core6d", "sali_core6d_graph"),
        default="ekg",
        help="Output tag; sali_core6d* rebuilds the same fused Z as run_sali_core6d_clustering.py",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--text-chars", type=int, default=600)
    p.add_argument("--description-samples", type=int, default=5, help="Step A: top-k nearest centroid")
    p.add_argument("--validate-samples-per-cluster", type=int, default=8)
    p.add_argument("--borderline-per-cluster", type=int, default=5)
    p.add_argument("--random-controls-per-cluster", type=int, default=2, help="Step D: random in-cluster controls")
    p.add_argument(
        "--membership-mode",
        choices=("classification", "entailment", "both"),
        default="both",
    )
    p.add_argument("--score-all-clusters", action="store_true", help="Step B: rank doc against all cluster descriptions")
    p.add_argument("--include-ekg-facts", action="store_true", default=True)
    p.add_argument("--no-ekg-facts", action="store_false", dest="include_ekg_facts")
    p.add_argument("--skip-llm", action="store_true")
    p.add_argument("--aggregate-human", type=Path, default=None, help="Score agreement on filled human CSV")
    p.add_argument(
        "--plots-only",
        action="store_true",
        help="Regenerate scatter plots from existing cluster_membership_eval_*.jsonl (no LLM)",
    )
    p.add_argument("--seed", type=int, default=cc.RANDOM_SEED)
    p.add_argument("--tfidf-max-vocab", type=int, default=2000)
    p.add_argument("--ekg-top-entity-types", type=int, default=40)
    p.add_argument("--ekg-top-event-types", type=int, default=40)
    p.add_argument("--ekg-top-roles", type=int, default=20)
    ns = p.parse_args()

    if ns.aggregate_human:
        stats = _aggregate_human_labels(ns.aggregate_human.expanduser().resolve())
        print(json.dumps(stats, indent=2))
        return

    root = ns.repo_root.expanduser().resolve() if ns.repo_root else inferred
    dr_ekg, dr_out = cc._defaults(root)
    out_dir = ns.output_dir.expanduser().resolve() if ns.output_dir else dr_out / "cluster_validation"

    if ns.plots_only:
        jsonl_path = out_dir / f"cluster_membership_eval_{ns.method}.jsonl"
        if not jsonl_path.is_file():
            raise SystemExit(f"Missing {jsonl_path}; run full eval first.")
        rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        paths = _write_scatter_plots(out_dir, ns.method, rows)
        if not paths:
            raise SystemExit("Could not write plots (need matplotlib and >=2 scored rows).")
        for pth in paths:
            print(f"Wrote {pth}")
        return

    cc._load_repo_dotenv(root)
    ekg_path = ns.ekg_jsonl.expanduser().resolve() if ns.ekg_jsonl else dr_ekg
    results_path = (
        ns.clustering_results.expanduser().resolve()
        if ns.clustering_results
        else dr_out / "clustering_detailed_results.json"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ekg_path.is_file():
        raise SystemExit(f"EKG JSONL not found: {ekg_path}")

    rng = random.Random(ns.seed)
    docs = cc.load_documents_and_graphs(ekg_path, limit=ns.max_docs)

    if ns.labels_json:
        labels_path = ns.labels_json.expanduser().resolve()
        labels_blob = json.loads(labels_path.read_text(encoding="utf-8"))
        doc_ids, labels = _parse_label_assignments(
            labels_blob, submethod=str(ns.labels_json_method) if ns.labels_json_method else None
        )
    else:
        labels_blob = None
        if not results_path.is_file():
            raise SystemExit(f"Clustering results not found: {results_path}")
        doc_ids, labels = _load_assignments_from_results(results_path, ns.method)

    keep = [i for i, d in enumerate(doc_ids) if d in docs]
    if len(keep) < len(doc_ids):
        print(f"Warning: dropped {len(doc_ids) - len(keep)} missing doc_ids")
    doc_ids = [doc_ids[i] for i in keep]
    labels = labels[keep]
    if len(doc_ids) < 4:
        raise SystemExit("Need >= 4 documents")

    texts = [docs[d]["text"] for d in doc_ids]
    graphs = [docs[d]["merged_graph"] for d in doc_ids]

    meta = {}
    if results_path.is_file():
        meta = json.loads(results_path.read_text(encoding="utf-8")).get("metadata") or {}
    if labels_blob is not None:
        hints = _embedding_hints_from_labels_blob(labels_blob)
        if hints.get("ekg_hyperparameters"):
            e0 = dict(meta.get("ekg_hyperparameters") or {})
            e0.update(hints["ekg_hyperparameters"])
            meta = {**meta, "ekg_hyperparameters": e0}

    use_ctx_z = _should_use_context_graph_embedding(labels_blob, str(ns.labels_json_method))
    use_sali_z = _should_use_sali_core6d_embedding(labels_blob, str(ns.method))
    if use_ctx_z:
        ctx_mod = _load_context_graph_module()
        embeddings, emb_name = _rebuild_neighbor_agg_embedding_Z(
            cc,
            ctx_mod,
            graphs,
            labels_blob,  # type: ignore[arg-type]
            ns.ekg_top_entity_types,
            ns.ekg_top_event_types,
            ns.ekg_top_roles,
        )
        embedding_info = {"type": "ekg_context_graph", "name": emb_name}
        print(f"Centroid distances in context-graph space: {emb_name}", flush=True)
    elif use_sali_z:
        sali_mod = _load_sali_core6d_module()
        force_graph = str(ns.method) == "sali_core6d_graph"
        embeddings, emb_name = _rebuild_sali_core6d_embedding_Z(
            cc,
            sali_mod,
            graphs,
            doc_ids,
            labels_blob,  # type: ignore[arg-type]
            ns.ekg_top_entity_types,
            ns.ekg_top_event_types,
            ns.ekg_top_roles,
            force_graph_context=force_graph,
        )
        etype = "sali_core6d_graph" if force_graph or "graph_neighbor" in emb_name else "sali_core6d"
        embedding_info = {"type": etype, "name": emb_name}
        print(f"Centroid distances in fused space: {emb_name}", flush=True)
    elif ns.method == "ekg" or ns.method in ("llm", "ekg_graph"):
        hyp = meta.get("ekg_hyperparameters") or {}
        emb_name = str(hyp.get("winning_ekg_embedding") or "core6d_row_l2")
        hdim = hyp.get("ekg_hybrid_context_pca_dim")
        hybrid_pca = int(hdim) if hdim is not None else None
        embeddings = _rebuild_ekg_embeddings(
            cc,
            graphs,
            emb_name,
            ns.ekg_top_entity_types,
            ns.ekg_top_event_types,
            ns.ekg_top_roles,
            hybrid_pca_dim=hybrid_pca,
        )
        embedding_info = {"type": "ekg", "name": emb_name}
    else:
        embeddings = _rebuild_tfidf_embeddings(cc, texts, ns.tfidf_max_vocab)
        embedding_info = {"type": "tfidf"}

    centroids = _compute_centroids(embeddings, labels)
    distances = _centroid_distances(embeddings, labels, centroids)
    clusters: Dict[int, List[int]] = {}
    for i, lab in enumerate(labels.tolist()):
        clusters.setdefault(int(lab), []).append(i)

    skip_llm = ns.skip_llm or os.environ.get("CLUSTERING_SKIP_LLM", "").lower() in ("1", "true", "yes")
    can_llm = not skip_llm and bool(
        (os.environ.get("AGENT_API_KEY") or "").strip() or (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    )
    if not skip_llm and not can_llm:
        print("No LLM key; distance-only mode.")

    # Step A
    cluster_meta: Dict[int, Dict[str, Any]] = {}
    for cid in sorted(clusters):
        nearest = _nearest_to_centroid(clusters[cid], embeddings, centroids, cid, ns.description_samples)
        blocks = [
            _doc_block(texts[i], graphs[i], cc, ns.text_chars, ns.include_ekg_facts) for i in nearest
        ]
        if can_llm:
            print(f"[A] Cluster {cid}: LLM description from {len(blocks)} centroid-nearest docs...")
            cluster_meta[cid] = _llm_cluster_description(cc, root, cid, blocks)
        else:
            cluster_meta[cid] = {
                "cluster_name": f"Cluster {cid}",
                "cluster_description": "(LLM skipped)",
                "common_themes": [],
                "key_indicators": [],
            }
        cluster_meta[cid]["description_doc_ids"] = [doc_ids[i] for i in nearest]

    # Step B
    membership_rows: List[Dict[str, Any]] = []
    for cid in sorted(clusters):
        name = str(cluster_meta[cid].get("cluster_name") or f"Cluster {cid}")
        desc = str(cluster_meta[cid].get("cluster_description") or "")
        idxs = list(clusters[cid])
        rng.shuffle(idxs)
        val_idxs = idxs[: min(len(idxs), ns.validate_samples_per_cluster)]
        cluster_dists = [float(distances[i]) for i in clusters[cid]]
        dist_q75 = float(np.percentile(cluster_dists, 75)) if cluster_dists else 0.0
        dist_q25 = float(np.percentile(cluster_dists, 25)) if cluster_dists else 0.0

        for i in val_idxs:
            doc_id = doc_ids[i]
            block = _doc_block(texts[i], graphs[i], cc, ns.text_chars, ns.include_ekg_facts)
            dist = float(distances[i])

            cls_res = ent_res = None
            if can_llm:
                if ns.membership_mode in ("classification", "both"):
                    cls_res = _llm_classification(cc, root, cid, name, desc, doc_id, block)
                if ns.membership_mode in ("entailment", "both"):
                    ent_res = _llm_entailment(cc, root, cid, name, desc, doc_id, block)
            merged = _merge_membership(cls_res, ent_res)

            cross_scores: Dict[str, float] = {}
            top_cid = cid
            if can_llm and ns.score_all_clusters:
                for oc in sorted(clusters):
                    oname = str(cluster_meta[oc].get("cluster_name") or f"Cluster {oc}")
                    odesc = str(cluster_meta[oc].get("cluster_description") or "")
                    oc_cls = _llm_classification(cc, root, oc, oname, odesc, doc_id, block)
                    sc = _signed_confidence(oc_cls.get("belongs"), oc_cls.get("confidence"))
                    if sc is not None:
                        cross_scores[str(oc)] = sc
                if cross_scores:
                    top_cid = int(max(cross_scores.items(), key=lambda kv: kv[1])[0])

            row = {
                "cluster_id": cid,
                "doc_id": doc_id,
                "cluster_name": name,
                "cluster_description": desc,
                "doc_excerpt": _excerpt(texts[i], ns.text_chars),
                "centroid_distance": dist,
                "membership_score": merged.get("membership_score"),
                "llm_belongs": (cls_res or ent_res or {}).get("belongs"),
                "llm_confidence": (cls_res or ent_res or {}).get("confidence"),
                "assigned_cluster_is_top": (top_cid == cid) if cross_scores else None,
                "top_scoring_cluster": top_cid if cross_scores else None,
                "cross_cluster_scores": cross_scores or None,
                **merged,
            }
            row["borderline_reason"] = _borderline_reason(row, dist_q75, dist_q25)
            membership_rows.append(row)

    # Step C
    scored = [r for r in membership_rows if r.get("membership_score") is not None]
    dist_arr = np.array([r["centroid_distance"] for r in scored], dtype=np.float64)
    conf_arr = np.array([r["membership_score"] for r in scored], dtype=np.float64)
    correlation = {
        "n": len(scored),
        "pearson_distance_vs_membership_score": _pearson(dist_arr, conf_arr),
        "spearman_distance_vs_membership_score": _spearman(dist_arr, conf_arr),
        "expected": "Negative correlation: closer to centroid → higher membership score",
    }

    scatter_csv = out_dir / f"scatter_distance_vs_confidence_{ns.method}.csv"
    with scatter_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cluster_id", "doc_id", "centroid_distance", "membership_score"])
        w.writeheader()
        for r in scored:
            w.writerow(
                {
                    "cluster_id": r["cluster_id"],
                    "doc_id": r["doc_id"],
                    "centroid_distance": r["centroid_distance"],
                    "membership_score": r["membership_score"],
                }
            )
    plot_paths = _write_scatter_plots(out_dir, ns.method, scored)
    scatter_png = out_dir / f"scatter_distance_vs_confidence_{ns.method}.png"
    plotted = bool(plot_paths)

    # Step D
    borderline_rows: List[Dict[str, Any]] = []
    for cid in sorted(clusters):
        cluster_eval = [r for r in membership_rows if r["cluster_id"] == cid]
        cluster_eval.sort(
            key=lambda r: (
                0 if r.get("assigned_cluster_is_top") is False else 1,
                -(1.0 - abs((r.get("llm_confidence") or 0.5) - 0.5) * 2),
                r["centroid_distance"],
            )
        )
        for r in cluster_eval[: ns.borderline_per_cluster]:
            borderline_rows.append({**r, "human_belongs": "", "human_notes": "", "sample_type": "borderline"})
        pool = [i for i in clusters[cid] if doc_ids[i] not in {x["doc_id"] for x in borderline_rows}]
        for i in rng.sample(pool, min(len(pool), ns.random_controls_per_cluster)):
            borderline_rows.append(
                {
                    "cluster_id": cid,
                    "doc_id": doc_ids[i],
                    "cluster_name": cluster_meta[cid].get("cluster_name"),
                    "cluster_description": cluster_meta[cid].get("cluster_description"),
                    "doc_excerpt": _excerpt(texts[i], ns.text_chars),
                    "centroid_distance": float(distances[i]),
                    "membership_score": None,
                    "sample_type": "random_control",
                    "borderline_reason": "random_in_cluster_control",
                    "human_belongs": "",
                    "human_notes": "",
                }
            )

    prefix = out_dir / f"cluster_eval_{ns.method}"
    summary = {
        "method": ns.method,
        "steps": ["A_cluster_descriptions", "B_membership", "C_correlation", "D_human_queue"],
        "num_documents": len(doc_ids),
        "num_clusters": len(clusters),
        "embedding": embedding_info,
        "membership_mode": ns.membership_mode,
        "score_all_clusters": ns.score_all_clusters,
        "correlation": correlation,
        "cluster_meta": {str(k): v for k, v in cluster_meta.items()},
        "llm_enabled": can_llm,
        "scatter_plot": str(scatter_png) if plotted else None,
        "scatter_plots": plot_paths,
        "scatter_csv": str(scatter_csv),
    }

    summary_path = out_dir / f"cluster_llm_eval_{ns.method}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (out_dir / f"cluster_membership_eval_{ns.method}.jsonl").open("w", encoding="utf-8") as f:
        for row in membership_rows:
            f.write(json.dumps(row) + "\n")

    csv_fields = [
        "sample_type",
        "cluster_id",
        "cluster_name",
        "doc_id",
        "cluster_description",
        "doc_excerpt",
        "centroid_distance",
        "membership_score",
        "llm_belongs",
        "llm_confidence",
        "assigned_cluster_is_top",
        "borderline_reason",
        "human_belongs",
        "human_notes",
    ]
    human_csv = out_dir / f"human_validation_queue_{ns.method}.csv"
    with human_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        w.writeheader()
        for row in borderline_rows:
            w.writerow(row)
    _write_rubric(out_dir / f"human_validation_rubric_{ns.method}.txt", ns.method)

    print("\n=== Cluster evaluation (A–D) complete ===")
    print(f"Documents: {len(doc_ids)}, clusters: {len(clusters)}, membership checks: {len(membership_rows)}")
    print(f"Borderline + controls for humans: {len(borderline_rows)}")
    if correlation.get("pearson_distance_vs_membership_score") is not None:
        print(f"Pearson(distance, score): {correlation['pearson_distance_vs_membership_score']:.4f}")
    print(f"Summary: {summary_path}")
    print(f"Human sheet: {human_csv}")
    if plotted:
        print("Scatter plots:")
        for pth in plot_paths:
            print(f"  {pth}")


if __name__ == "__main__":
    main()
