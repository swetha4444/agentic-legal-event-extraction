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


def _load_assignments_from_labels(path: Path) -> Tuple[List[str], np.ndarray]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict) and "assignments" in obj:
        pairs = sorted(obj["assignments"].items(), key=lambda kv: kv[0])
    elif isinstance(obj, list):
        pairs = sorted((str(r["doc_id"]), int(r["cluster_id"])) for r in obj)
    else:
        raise SystemExit(f"Unsupported labels format: {path}")
    return [p[0] for p in pairs], np.asarray([p[1] for p in pairs], dtype=int)


def _rebuild_ekg_embeddings(cc, graphs: List[Dict], emb_name: str, top_ent: int, top_evt: int, top_rl: int) -> np.ndarray:
    if emb_name == "core6d_row_l2":
        return cc.build_core6d_embedding(graphs)
    for name, X in cc.build_ekg_embedding_candidates(graphs, top_ent, top_evt, top_rl, candidates="full"):
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


def _write_scatter_plot(path: Path, rows: List[Dict[str, Any]]) -> bool:
    pts = [(r["centroid_distance"], r["membership_score"]) for r in rows if r.get("membership_score") is not None]
    if len(pts) < 2:
        return False
    try:
        import matplotlib.pyplot as plt

        xs, ys = zip(*pts)
        plt.figure(figsize=(6, 4))
        plt.scatter(xs, ys, alpha=0.6, s=24)
        plt.xlabel("Distance to cluster centroid")
        plt.ylabel("Membership score (higher = better fit)")
        plt.title("Centroid distance vs automatic membership confidence")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        return True
    except ImportError:
        return False


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
    p.add_argument("--method", choices=("ekg", "tfidf"), default="ekg")
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
    cc._load_repo_dotenv(root)
    dr_ekg, dr_out = cc._defaults(root)
    ekg_path = ns.ekg_jsonl.expanduser().resolve() if ns.ekg_jsonl else dr_ekg
    results_path = (
        ns.clustering_results.expanduser().resolve()
        if ns.clustering_results
        else dr_out / "clustering_detailed_results.json"
    )
    out_dir = ns.output_dir.expanduser().resolve() if ns.output_dir else dr_out / "cluster_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ekg_path.is_file():
        raise SystemExit(f"EKG JSONL not found: {ekg_path}")

    rng = random.Random(ns.seed)
    docs = cc.load_documents_and_graphs(ekg_path, limit=ns.max_docs)

    if ns.labels_json:
        doc_ids, labels = _load_assignments_from_labels(ns.labels_json.expanduser().resolve())
    else:
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
    if results_path.is_file() and not ns.labels_json:
        meta = json.loads(results_path.read_text(encoding="utf-8")).get("metadata") or {}

    if ns.method == "ekg":
        emb_name = str((meta.get("ekg_hyperparameters") or {}).get("winning_ekg_embedding") or "core6d_row_l2")
        embeddings = _rebuild_ekg_embeddings(
            cc, graphs, emb_name, ns.ekg_top_entity_types, ns.ekg_top_event_types, ns.ekg_top_roles
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
    scatter_png = out_dir / f"scatter_distance_vs_confidence_{ns.method}.png"
    plotted = _write_scatter_plot(scatter_png, scored)

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
        print(f"Scatter plot: {scatter_png}")


if __name__ == "__main__":
    main()
