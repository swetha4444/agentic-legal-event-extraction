#!/usr/bin/env python3
"""
Fair EKG vs TF-IDF clustering comparison (TF-IDF uses shared heuristic k).

TF-IDF is clustered at a corpus-derived k. By default, EKG searches multiple embeddings,
restarts, and k values (within bounds) and keeps the run that maximizes within-space silhouette;
this is documented in metadata (silhouette is not comparable across spaces anyway).
Use --no-ekg-optimize-k to force EKG to use the same k as TF-IDF.

Defaults: `<repo>/data/outputs/qwen122b_hybrid_doc_graphs_final.jsonl`,
`<repo>/clustering/results/clustering_detailed_results.json`.

Optional LLM coherence: `<repo>/.env` with AGENT_API_KEY or ANTHROPIC_API_KEY.

Silhouette scores are not comparable across TF-IDF vs EKG spaces; see publication.caveats and
https://arxiv.org/abs/2404.10351 .
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from math import log
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import anthropic

    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    from openai import OpenAI

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


def _load_repo_dotenv(repo_root: Path) -> None:
    try:
        from dotenv import load_dotenv

        env_file = repo_root / ".env"
        if env_file.is_file():
            load_dotenv(env_file, override=False)
    except ImportError:
        pass


def _ensure_src_on_path(repo_root: Path) -> None:
    src = repo_root / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _detect_repo_root(here: Path) -> Path | None:
    """Find repo root: directory that has both data/outputs and clustering/."""
    for p in [here.parent, *here.parents]:
        if (p / "data" / "outputs").is_dir() and (p / "clustering").is_dir():
            return p
    for p in [here.parent, *here.parents]:
        if (p / "data" / "outputs" / "qwen122b_hybrid_doc_graphs_final.jsonl").is_file():
            return p
    return None


def _repo_root(script_path: Path) -> Path:
    here = script_path.resolve()
    if here.parent.name == "scripts" and here.parent.parent.name == "clustering":
        return here.parent.parent.parent
    found = _detect_repo_root(here)
    if found is not None:
        return found
    return here.parent.parent.parent


def _defaults(repo_root: Path) -> tuple[Path, Path]:
    return (
        repo_root / "data" / "outputs" / "qwen122b_hybrid_doc_graphs_final.jsonl",
        repo_root / "clustering" / "results",
    )

RANDOM_SEED = 42
N_CLUSTERS = 8
SAMPLES_PER_CLUSTER = 3
EVAL_CLUSTERS = 6


def load_documents_and_graphs(ekg_path: Path, limit: int | None = None) -> Dict[str, Dict]:
    docs_and_graphs: Dict[str, Dict] = {}
    with open(ekg_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            try:
                data = json.loads(line)
                doc_id = data.get("doc_id", f"doc_{i}")
                chunks = data.get("chunks", [])
                text = " ".join(chunk.get("text", "") for chunk in chunks)
                if not text or len(text.strip()) < 100:
                    continue
                docs_and_graphs[doc_id] = {
                    "doc_id": doc_id,
                    "text": text,
                    "merged_graph": data.get("merged_graph", {}),
                    "merge_stats": data.get("merge_stats", {}),
                    "num_chunks": len(chunks),
                }
            except json.JSONDecodeError:
                continue
    print(f"Loaded {len(docs_and_graphs)} documents with EKG representations")
    return docs_and_graphs


def extract_ekg_features(graph: Dict) -> Dict[str, Any]:
    entities = graph.get("entities", [])
    events = graph.get("events", [])
    temporal_edges = graph.get("temporal_edges", [])
    causal_edges = graph.get("causal_edges", [])
    return {
        "num_entities": len(entities),
        "num_events": len(events),
        "num_temporal_edges": len(temporal_edges),
        "num_causal_edges": len(causal_edges),
        "graph_density": (len(temporal_edges) + len(causal_edges))
        / max(1, len(entities) + len(events)),
        "entities": entities[:30],
        "events": events[:30],
    }


def _type_label(x: Any) -> str:
    s = str(x if x is not None else "").strip()
    return (s[:96] if s else "Unknown")


def build_global_ekg_vocabs(
    graphs: List[Dict],
    top_entity_types: int,
    top_event_types: int,
    top_roles: int,
) -> Tuple[List[str], List[str], List[str]]:
    """Corpus-wide top types / roles by document frequency."""
    ent_doc_freq = Counter()
    evt_doc_freq = Counter()
    role_doc_freq = Counter()

    for g in graphs:
        entities = g.get("entities") or []
        events = g.get("events") or []
        doc_ent_types = {_type_label(e.get("type")) for e in entities}
        doc_evt_types = set()
        doc_roles = set()
        for ev in events:
            doc_evt_types.add(_type_label(ev.get("event_type") or ev.get("type")))
            for p in ev.get("participants") or []:
                if isinstance(p, dict):
                    doc_roles.add(_type_label(p.get("role")))
        for t in doc_ent_types:
            ent_doc_freq[t] += 1
        for t in doc_evt_types:
            evt_doc_freq[t] += 1
        for r in doc_roles:
            role_doc_freq[r] += 1

    ent_vocab = [t for t, _ in ent_doc_freq.most_common(top_entity_types)]
    evt_vocab = [t for t, _ in evt_doc_freq.most_common(top_event_types)]
    role_vocab = [r for r, _ in role_doc_freq.most_common(top_roles)]
    return ent_vocab, evt_vocab, role_vocab


def graph_to_enhanced_ekg_vector(
    graph: Dict,
    ent_vocab: List[str],
    evt_vocab: List[str],
    role_vocab: List[str],
) -> np.ndarray:
    """Dense hybrid counts + global multi-hot-type frequencies for one doc."""
    entities = graph.get("entities") or []
    events = graph.get("events") or []
    temporal_edges = graph.get("temporal_edges") or []
    causal_edges = graph.get("causal_edges") or []

    n_ent = len(entities)
    n_evt = len(events)
    n_te = len(temporal_edges)
    n_ce = len(causal_edges)
    ec = n_te + n_ce

    ent_counter = Counter(_type_label(e.get("type")) for e in entities)
    evt_counter = Counter()
    role_counter = Counter()
    pl_flag = 0.0
    df_flag = 0.0
    parts_counts = []

    for e in entities:
        nm = str(e.get("name") or "").lower()
        if "plaintiff" in nm:
            pl_flag = 1.0
        if "defendant" in nm:
            df_flag = 1.0

    for ev in events:
        evt_counter[_type_label(ev.get("event_type") or ev.get("type"))] += 1
        parts = ev.get("participants") or []
        parts_counts.append(len(parts))
        for p in parts:
            if isinstance(p, dict):
                role_counter[_type_label(p.get("role"))] += 1

    avg_parts = float(np.mean(parts_counts)) if parts_counts else 0.0

    density = ec / max(1, n_ent + n_evt)
    ratio_temporal = n_te / max(1, ec) if ec else 0.0

    unique_ent_types = len(ent_counter)
    unique_evt_types = len(evt_counter)

    base = [
        np.log1p(n_ent),
        np.log1p(n_evt),
        np.log1p(n_te),
        np.log1p(n_ce),
        density,
        ratio_temporal,
        unique_ent_types / max(1.0, np.sqrt(n_ent + 1.0)),
        unique_evt_types / max(1.0, np.sqrt(n_evt + 1.0)),
        pl_flag,
        df_flag,
        avg_parts,
    ]

    ent_idx = {t: i for i, t in enumerate(ent_vocab)}
    evt_idx = {t: i for i, t in enumerate(evt_vocab)}
    rl_idx = {r: i for i, r in enumerate(role_vocab)}

    ent_vec = np.zeros(len(ent_vocab), dtype=np.float64)
    for t, c in ent_counter.items():
        if t in ent_idx:
            ent_vec[ent_idx[t]] = np.log1p(c)

    evt_vec = np.zeros(len(evt_vocab), dtype=np.float64)
    for t, c in evt_counter.items():
        if t in evt_idx:
            evt_vec[evt_idx[t]] = np.log1p(c)

    role_vec = np.zeros(len(role_vocab), dtype=np.float64)
    for r, c in role_counter.items():
        if r in rl_idx:
            role_vec[rl_idx[r]] = np.log1p(c)

    return np.concatenate([np.array(base, dtype=np.float64), ent_vec, evt_vec, role_vec])


def build_core6d_embedding(graphs_ordered: List[Dict]) -> np.ndarray:
    """6-D structural EKG vector: log counts + density + temporal ratio, row L2-normalized."""
    rows: List[List[float]] = []
    for graph in graphs_ordered:
        entities = graph.get("entities") or []
        events = graph.get("events") or []
        temporal_edges = graph.get("temporal_edges") or []
        causal_edges = graph.get("causal_edges") or []
        n_ent, n_evt = len(entities), len(events)
        n_te, n_ce = len(temporal_edges), len(causal_edges)
        ec = n_te + n_ce
        density = ec / max(1, n_ent + n_evt)
        ratio_temporal = n_te / max(1, ec) if ec else 0.0
        rows.append(
            [
                float(np.log1p(n_ent)),
                float(np.log1p(n_evt)),
                float(np.log1p(n_te)),
                float(np.log1p(n_ce)),
                density,
                ratio_temporal,
            ]
        )
    return normalize_embeddings(np.array(rows, dtype=np.float32)).astype(np.float64)


def standardize_columns(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sigma = X.std(axis=0) + 1e-8
    return (X - mu) / sigma, mu, sigma


def pca_whiten_embedding(
    X: np.ndarray,
    n_components: int,
    *,
    row_l2_normalize: bool,
) -> np.ndarray:
    """PCA scores with variance whitening — often improves Euclidean k-means + silhouette."""
    Z = np.asarray(X, dtype=np.float64)
    Z = Z - Z.mean(axis=0)
    n, d = Z.shape
    rk = min(n, d)
    k = int(max(2, min(int(n_components), n - 1, rk)))
    U, sing, _ = np.linalg.svd(Z, full_matrices=False)
    k = min(k, len(sing))
    scores = U[:, :k] * sing[:k]
    lam = (sing[:k] ** 2) / max(float(n - 1), 1.0)
    scores = scores / np.sqrt(lam + 1e-12)
    if row_l2_normalize:
        scores = normalize_embeddings(scores.astype(np.float32)).astype(np.float64)
    return scores.astype(np.float64)


def silhouette_metric_sklearn(embeddings: np.ndarray, labels: np.ndarray) -> float | None:
    try:
        from sklearn.metrics import silhouette_score
    except ImportError:
        return None
    lab = np.asarray(labels)
    if len(np.unique(lab)) < 2:
        return None
    s = silhouette_score(embeddings, lab, metric="euclidean")
    if np.isnan(s):
        return None
    return float(s)


def build_ekg_embedding_candidates(
    graphs_ordered: List[Dict],
    top_ent: int,
    top_evt: int,
    top_rl: int,
    *,
    candidates: str = "full",
) -> List[Tuple[str, np.ndarray]]:
    """EKG embedding matrices for fair search. candidates: full | core6d_only."""
    out: List[Tuple[str, np.ndarray]] = []
    x6 = build_core6d_embedding(graphs_ordered)
    out.append(("core6d_row_l2", x6))
    if candidates == "core6d_only":
        return out

    rows7: List[List[float]] = []
    for g in graphs_ordered:
        fe = extract_ekg_features(g)
        rows7.append(
            [
                float(fe["num_entities"]),
                float(fe["num_events"]),
                float(fe["num_temporal_edges"]),
                float(fe["num_causal_edges"]),
                float(fe["graph_density"] * 100),
                float(len(fe["entities"])),
                float(len(fe["events"])),
            ]
        )
    x7 = normalize_embeddings(np.array(rows7, dtype=np.float32)).astype(np.float64)
    out.append(("legacy7d_row_l2", x7))

    ent_vocab, evt_vocab, role_vocab = build_global_ekg_vocabs(
        graphs_ordered, top_ent, top_evt, top_rl
    )
    raw = np.vstack(
        [graph_to_enhanced_ekg_vector(g, ent_vocab, evt_vocab, role_vocab) for g in graphs_ordered]
    ).astype(np.float64)

    xs, _, _ = standardize_columns(raw)
    out.append(("enhanced_standardize", xs.astype(np.float64)))
    out.append(("enhanced_row_l2", normalize_embeddings(raw.astype(np.float32)).astype(np.float64)))
    out.append(
        (
            "enhanced_standardize_row_l2",
            normalize_embeddings(xs.astype(np.float32)).astype(np.float64),
        )
    )

    n_docs = raw.shape[0]
    for npc in (12, 20, 32):
        cap = min(npc, n_docs - 1)
        if cap < 3:
            continue
        out.append(
            (
                f"enhanced_pca{cap}_whiten",
                pca_whiten_embedding(xs, cap, row_l2_normalize=False),
            )
        )
        out.append(
            (
                f"enhanced_pca{cap}_whiten_rowl2",
                pca_whiten_embedding(xs, cap, row_l2_normalize=True),
            )
        )

    return out


class SimpleTFIDF:
    def __init__(self, min_freq: int = 1, max_vocab_size: int = 3000):
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.fitted = False

    def _tokenize(self, text: str) -> List[str]:
        text = str(text).lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        tokens = text.split()
        stopwords = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "is",
            "was",
            "are",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "can",
            "this",
            "that",
            "these",
            "those",
        }
        return [t for t in tokens if len(t) > 2 and t not in stopwords]

    def fit(self, texts: List[str]) -> None:
        doc_freq = defaultdict(int)
        for text in texts:
            unique_tokens = set(self._tokenize(text))
            for token in unique_tokens:
                doc_freq[token] += 1
        vocab_items = [(t, c) for t, c in doc_freq.items() if c >= self.min_freq]
        vocab_items.sort(key=lambda x: x[1], reverse=True)
        vocab_items = vocab_items[: self.max_vocab_size]
        self.vocab = {token: idx for idx, (token, _) in enumerate(vocab_items)}
        n_docs = len(texts)
        self.idf = {}
        for token, doc_count in vocab_items:
            self.idf[token] = log(n_docs / doc_count) if doc_count > 0 else 0.0
        self.fitted = True

    def transform(self, text: str) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Must call fit() before transform()")
        tokens = self._tokenize(text)
        vector = np.zeros(len(self.vocab), dtype=np.float32)
        tf = defaultdict(int)
        for token in tokens:
            if token in self.vocab:
                tf[token] += 1
        for token, count in tf.items():
            idx = self.vocab[token]
            vector[idx] = count * self.idf[token]
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector


def simple_kmeans(
    embeddings: np.ndarray,
    n_clusters: int = 8,
    max_iters: int = 50,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(random_state)
    n_samples, _ = embeddings.shape
    indices = np.random.choice(n_samples, min(n_clusters, n_samples), replace=False)
    centroids = embeddings[indices].copy()
    actual_clusters = centroids.shape[0]
    for _ in range(max_iters):
        distances = np.zeros((n_samples, actual_clusters))
        for i in range(actual_clusters):
            distances[:, i] = np.linalg.norm(embeddings - centroids[i], axis=1)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.zeros_like(centroids)
        for i in range(actual_clusters):
            mask = labels == i
            if mask.sum() > 0:
                new_centroids[i] = embeddings[mask].mean(axis=0)
            else:
                new_centroids[i] = centroids[i]
        if np.allclose(centroids, new_centroids, rtol=1e-4):
            break
        centroids = new_centroids
    return labels, centroids


def ekg_balance_min_cluster_size(
    n_docs: int,
    n_clusters: int,
    balance_divisor: float = 3.0,
) -> int:
    """Minimum cluster size for balanced EKG selection (~n / (k * divisor))."""
    denom = max(1.0, float(n_clusters) * float(balance_divisor))
    return max(2, int(n_docs // denom))


def cluster_size_dict(labels: np.ndarray) -> Dict[int, int]:
    from collections import Counter

    return {int(k): int(v) for k, v in sorted(Counter(int(x) for x in labels).items())}


def poster_core6d_restart_seed(base_seed: int, trial: int, k: int) -> int:
    """Match test_core6d_embedding.py restart schedule."""
    return int(base_seed) + int(trial) * 9973 + int(k) * 17


def fit_ekg_clustering(
    embeddings: np.ndarray,
    n_clusters: int,
    random_state: int,
    *,
    poster_core6d: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Poster Core6D uses simple_kmeans (matches test_core6d_embedding.py); else sklearn KMeans++."""
    if poster_core6d:
        return simple_kmeans(
            embeddings, n_clusters=n_clusters, random_state=random_state
        )
    return kmeans_ekg(
        embeddings, n_clusters=n_clusters, random_state=random_state
    )


def kmeans_ekg(
    embeddings: np.ndarray,
    n_clusters: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Prefer sklearn KMeans++ (better local optima); fall back to simple_kmeans."""
    rs = int(random_state) % (2**31)
    try:
        from sklearn.cluster import KMeans

        km = KMeans(
            n_clusters=n_clusters,
            random_state=rs,
            n_init=10,
            max_iter=300,
            tol=1e-4,
            algorithm="lloyd",
        )
        labels = km.fit_predict(np.asarray(embeddings, dtype=np.float64))
        return labels.astype(np.int64), km.cluster_centers_.astype(np.float64)
    except Exception:
        return simple_kmeans(
            embeddings, n_clusters=n_clusters, random_state=random_state
        )


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / (norms + 1e-8)


def compute_silhouette_score(embeddings: np.ndarray, labels: np.ndarray) -> float:
    n_samples = len(embeddings)
    scores: List[float] = []
    sample_indices = np.random.choice(n_samples, min(n_samples, 100), replace=False)
    for i in sample_indices:
        same_cluster = labels == labels[i]
        if same_cluster.sum() <= 1:
            continue
        a = float(np.mean(np.linalg.norm(embeddings[same_cluster] - embeddings[i], axis=1)))
        b = float("inf")
        for cluster_id in np.unique(labels):
            if cluster_id != labels[i]:
                cluster = labels == cluster_id
                if cluster.sum() > 0:
                    dist = float(
                        np.mean(np.linalg.norm(embeddings[cluster] - embeddings[i], axis=1))
                    )
                    b = min(b, dist)
        if b == float("inf"):
            continue
        silhouette = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
        scores.append(silhouette)
    return float(np.mean(scores)) if scores else 0.0


def compute_davies_bouldin_score(
    embeddings: np.ndarray, labels: np.ndarray, centroids: np.ndarray
) -> float:
    n_clusters = len(np.unique(labels))
    within_distances = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_mask = labels == i
        if cluster_mask.sum() > 0:
            within_distances[i] = float(
                np.mean(np.linalg.norm(embeddings[cluster_mask] - centroids[i], axis=1))
            )
    db_index = 0.0
    for i in range(n_clusters):
        max_ratio = 0.0
        for j in range(n_clusters):
            if i != j:
                between_distance = float(np.linalg.norm(centroids[i] - centroids[j]))
                if between_distance > 0:
                    ratio = (within_distances[i] + within_distances[j]) / between_distance
                    max_ratio = max(max_ratio, ratio)
        db_index += max_ratio
    return db_index / max(1, n_clusters)


def get_cluster_statistics(doc_ids: List[str], labels: np.ndarray) -> Dict[int, Dict]:
    clusters: Dict[int, List[str]] = defaultdict(list)
    for doc_id, label in zip(doc_ids, labels):
        clusters[int(label)].append(doc_id)
    return {cid: {"size": len(docs), "docs": docs} for cid, docs in clusters.items()}


def generate_visualization_data(
    embeddings: np.ndarray,
    labels: np.ndarray,
    doc_ids: List[str],
    method_name: str,
    random_state: int,
    use_tsne: bool = True,
) -> Dict[str, Any]:
    n = len(embeddings)
    if not use_tsne:
        print(f"  Skipping t-SNE for {method_name} (--skip-tsne); using random projection.")
        tsne_embeddings = np.random.RandomState(random_state).randn(n, 2)
        return {
            "method": method_name,
            "tsne_2d": tsne_embeddings.tolist(),
            "labels": labels.tolist(),
            "doc_ids": doc_ids,
        }
    if n >= 4:
        try:
            from sklearn.manifold import TSNE

            perplexity = float(min(30, max(2, n - 1)))
            print(f"  Computing t-SNE for {method_name} (perplexity={perplexity})...")
            tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
            tsne_embeddings = tsne.fit_transform(embeddings)
        except ImportError:
            print(f"  sklearn not available, using random projection for {method_name}...")
            tsne_embeddings = np.random.RandomState(random_state).randn(n, 2)
    else:
        print(f"  Too few points for t-SNE ({n}); using random projection for {method_name}...")
        tsne_embeddings = np.random.RandomState(random_state).randn(n, 2)

    return {
        "method": method_name,
        "tsne_2d": tsne_embeddings.tolist(),
        "labels": labels.tolist(),
        "doc_ids": doc_ids,
    }


def _llm_coherence_completion(
    prompt: str,
    repo_root: Path,
    *,
    max_tokens: int = 256,
) -> str:
    """Return raw model text for one coherence prompt."""
    agent_key = (os.environ.get("AGENT_API_KEY") or "").strip()
    anthropic_key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()

    if agent_key:
        if not HAS_OPENAI:
            raise RuntimeError("pip install openai (needed for AGENT_API_KEY / Keymaker)")
        _ensure_src_on_path(repo_root)
        from agents.config_loader import get_llm_config

        cfg = get_llm_config()
        base = cfg.get("api_base") or "https://thekeymaker.umass.edu/"
        model = (os.environ.get("CLUSTERING_COHERENCE_MODEL") or "claude-haiku-4-5").strip()
        client = OpenAI(api_key=agent_key, base_url=base, timeout=120.0)
        model_lower = model.lower()
        temperature = (
            1.0
            if "gpt-5" in model_lower or model_lower == "gpt5"
            else float(cfg.get("temperature", 0.0) or 0.0)
        )
        req: dict = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_completion_tokens": int(max_tokens),
        }
        try:
            resp = client.chat.completions.create(**req)
        except TypeError:
            req.pop("max_completion_tokens", None)
            req["max_tokens"] = int(max_tokens)
            resp = client.chat.completions.create(**req)
        except Exception as exc:
            text = str(exc).lower()
            if "only temperature=1 is supported" in text or "temperature=0.0" in text:
                req["temperature"] = 1.0
                resp = client.chat.completions.create(**req)
            elif "max_completion_tokens" in text or "unknown parameter" in text:
                req.pop("max_completion_tokens", None)
                resp = client.chat.completions.create(**req)
            else:
                raise
        return (resp.choices[0].message.content or "").strip()

    if anthropic_key and HAS_ANTHROPIC:
        client = anthropic.Anthropic()
        model = (os.environ.get("CLUSTERING_COHERENCE_MODEL") or "claude-haiku-4-5-20251001").strip()
        response = client.messages.create(
            model=model,
            max_tokens=int(max_tokens),
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    raise RuntimeError("No LLM key: set AGENT_API_KEY or ANTHROPIC_API_KEY")


def evaluate_cluster_coherence(
    cluster_samples: Dict[int, Dict],
    docs_and_graphs: Dict,
    method_name: str,
    skip_llm: bool,
    repo_root: Path,
) -> Dict:
    if skip_llm:
        print("  Skipping LLM coherence (--skip-llm)")
        return {}

    agent_key = (os.environ.get("AGENT_API_KEY") or "").strip()
    anthropic_key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    can_keymaker = bool(agent_key and HAS_OPENAI)
    can_anthropic = bool(anthropic_key and HAS_ANTHROPIC)
    if not can_keymaker and not can_anthropic:
        print(
            "  Skipping LLM coherence: set AGENT_API_KEY (Keymaker) plus `pip install openai`, "
            "or ANTHROPIC_API_KEY plus `pip install anthropic`"
        )
        return {}

    coherence_results: Dict[int, Dict] = {}

    for cluster_id, cluster_info in list(cluster_samples.items())[:EVAL_CLUSTERS]:
        doc_ids = cluster_info["docs"]
        doc_snippets = []
        for doc_id in doc_ids[:3]:
            text = docs_and_graphs.get(doc_id, {}).get("text", "")[:400]
            doc_snippets.append(f"Doc {len(doc_snippets)+1}: {text}")

        prompt = f"""You are evaluating whether documents in a cluster are semantically related.

{method_name} Clustering - Cluster {cluster_id} contains these documents:

{chr(10).join(doc_snippets)}

Rate the semantic coherence on a scale of 1-5:
1 = Completely unrelated
2 = Some overlap but mostly different
3 = Moderate coherence
4 = Strong coherence
5 = Highly similar/related documents

Respond ONLY with valid JSON (no markdown):
{{"coherence_score": <1-5>, "reasoning": "<1-2 sentences>"}}"""

        try:
            response_text = _llm_coherence_completion(prompt, repo_root)
            try:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    result = json.loads(response_text[json_start:json_end])
                else:
                    result = {"coherence_score": 3, "reasoning": "Parse error"}
            except json.JSONDecodeError:
                result = {"coherence_score": 3, "reasoning": "Parse error"}

            coherence_results[cluster_id] = {
                "docs": doc_ids,
                "coherence": result.get("coherence_score", 3),
                "reasoning": result.get("reasoning", ""),
            }
            print(f"    Cluster {cluster_id}: {result.get('coherence_score', '?')}/5")
        except Exception as e:
            print(f"    Error evaluating cluster {cluster_id}: {e}")

    return coherence_results


def parse_args() -> argparse.Namespace:
    script_here = Path(__file__).resolve()
    env_root = os.environ.get("CLUSTERING_REPO_ROOT", "").strip()
    inferred_root = (
        Path(env_root).expanduser().resolve()
        if env_root
        else _repo_root(script_here)
    )

    p = argparse.ArgumentParser(
        description="EKG vs TF-IDF clustering (TF-IDF fixed k; EKG can search k for silhouette)."
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root (sets defaults for --ekg-jsonl / --output-dir if omitted)",
    )
    p.add_argument(
        "--ekg-jsonl",
        type=Path,
        default=None,
        help="JSONL with chunks + merged_graph per line",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write clustering_detailed_results.json",
    )
    p.add_argument("--max-docs", type=int, default=None, help="Cap lines read from JSONL")
    p.add_argument(
        "--tfidf-max-vocab",
        type=int,
        default=2000,
        help="Top terms by doc frequency for TF-IDF (higher can help separation; more compute)",
    )
    p.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip cluster coherence calls (no AGENT_API_KEY / ANTHROPIC_API_KEY usage)",
    )
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument(
        "--ekg-min-cluster-size",
        type=int,
        default=3,
        help="Minimum cluster size filter during fair EKG embedding search (trials below this are skipped)",
    )
    p.add_argument(
        "--ekg-silhouette-tuning",
        choices=("sampled", "sklearn"),
        default="sampled",
        help="Objective when ranking EKG embedding candidates (sampled vs sklearn)",
    )
    p.add_argument("--ekg-top-entity-types", type=int, default=40, help="Global entity-type dims")
    p.add_argument("--ekg-top-event-types", type=int, default=40, help="Global event-type dims")
    p.add_argument("--ekg-top-roles", type=int, default=20, help="Global participant-role dims")
    p.add_argument(
        "--ekg-candidates",
        choices=("full", "core6d_only"),
        default="full",
        help="full=all embedding variants; core6d_only=6-D structural vector only (faster, lawyer package default)",
    )
    p.add_argument(
        "--ekg-balance-divisor",
        type=float,
        default=3.0,
        help="core6d_only: require each cluster to have at least n/(k*divisor) docs when picking best restart",
    )
    p.add_argument(
        "--no-ekg-balance",
        action="store_true",
        help="core6d_only: allow degenerate splits that maximize silhouette (e.g. 1+113+4+15 at k=4)",
    )
    eg = p.add_mutually_exclusive_group()
    eg.add_argument(
        "--ekg-optimize-k",
        dest="ekg_optimize_k",
        action="store_true",
        help="Search EKG k in [ekg-k-min, ekg-k-max] (cap by n/min-cluster); TF-IDF k unchanged (default)",
    )
    eg.add_argument(
        "--no-ekg-optimize-k",
        dest="ekg_optimize_k",
        action="store_false",
        help="Use the same k for EKG as for TF-IDF (no k search)",
    )
    p.set_defaults(ekg_optimize_k=True)
    p.add_argument(
        "--ekg-k-min",
        type=int,
        default=2,
        help="Minimum k when --ekg-optimize-k (default: 2)",
    )
    p.add_argument(
        "--ekg-k-max",
        type=int,
        default=16,
        help="Maximum k when --ekg-optimize-k (default: 16; further capped by dataset size)",
    )
    p.add_argument(
        "--skip-tsne",
        action="store_true",
        help="Skip slow t-SNE; visualization payload uses random 2D projection",
    )
    p.add_argument(
        "--fair-trials",
        type=int,
        default=48,
        help="Random k-means restarts for TF-IDF and per EKG candidate embedding",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="Quick run: --fair-trials 12 --no-ekg-optimize-k --skip-tsne (~80 EKG fits vs thousands)",
    )
    ns = p.parse_args()
    if ns.fast:
        ns.skip_tsne = True
        locked_k = int(ns.ekg_k_min) == int(ns.ekg_k_max)
        poster_fast = locked_k and ns.ekg_candidates == "core6d_only"
        if poster_fast:
            ns.fair_trials = 24
            ns.ekg_optimize_k = True
            print(
                f"  FAST MODE (Core6D): fair_trials=24, k={ns.ekg_k_min}, "
                f"balanced restarts, skipping t-SNE",
                flush=True,
            )
        else:
            ns.fair_trials = min(int(ns.fair_trials), 12)
            if locked_k:
                ns.ekg_optimize_k = True
                print(
                    f"  FAST MODE: fair_trials=12, EKG k locked to {ns.ekg_k_min}, skipping t-SNE",
                    flush=True,
                )
            else:
                ns.ekg_optimize_k = False
                print(
                    "  FAST MODE: fair_trials=12, EKG k fixed to TF-IDF k, skipping t-SNE",
                    flush=True,
                )

    root = ns.repo_root.expanduser().resolve() if ns.repo_root else inferred_root
    dr_ekg, dr_out = _defaults(root)
    if ns.ekg_jsonl is None:
        ns.ekg_jsonl = dr_ekg
    if ns.output_dir is None:
        ns.output_dir = dr_out
    ns._resolved_repo_root = root  # type: ignore[attr-defined]
    return ns


def main() -> Dict[str, Any]:
    args = parse_args()
    ekg_path = args.ekg_jsonl.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    repo_root = getattr(args, "_resolved_repo_root", _repo_root(Path(__file__)))
    _load_repo_dotenv(repo_root)

    if not ekg_path.is_file():
        print(f"ERROR: EKG JSONL not found: {ekg_path}", file=sys.stderr)
        hint = repo_root / "data" / "outputs" / "qwen122b_hybrid_doc_graphs_final.jsonl"
        print(f"Try: --repo-root <path-to-repo> or --ekg-jsonl {hint}", file=sys.stderr)
        raise SystemExit(1)

    np.random.seed(args.seed)
    random.seed(args.seed)

    max_docs = args.max_docs
    skip_llm = args.skip_llm or os.environ.get("CLUSTERING_SKIP_LLM", "").lower() in (
        "1",
        "true",
        "yes",
    )

    def _sil_tune_fn(X: np.ndarray, labels: np.ndarray) -> float | None:
        if args.ekg_silhouette_tuning == "sklearn":
            return silhouette_metric_sklearn(X, labels)
        return compute_silhouette_score(X, labels)

    print("=" * 80)
    print("CLUSTERING COMPARISON: EKG vs TF-IDF")
    print("=" * 80)
    print(f"Repo root (for defaults): {repo_root}")
    print(f"Input:  {ekg_path}")
    print(f"Output: {output_dir / 'clustering_detailed_results.json'}")

    print("\n[1/7] Loading documents and EKG representations...")
    docs_and_graphs = load_documents_and_graphs(ekg_path, limit=max_docs)
    doc_ids = list(docs_and_graphs.keys())
    if not doc_ids:
        raise SystemExit("No documents loaded (check JSONL structure and min text length).")

    print(f"  Total documents: {len(doc_ids)}")
    print(
        f"  Average doc length: {np.mean([len(d['text']) for d in docs_and_graphs.values()]):.0f} chars"
    )

    n_clusters_actual = min(N_CLUSTERS, max(3, len(doc_ids) // 15))
    tfidf_k_used = int(n_clusters_actual)
    fair_trials = max(1, int(args.fair_trials))
    if args.ekg_optimize_k:
        ekg_k_desc = (
            f"k tuned in [{args.ekg_k_min}, {args.ekg_k_max}] "
            "(capped by n / min-cluster-size)"
        )
    else:
        ekg_k_desc = f"k fixed to TF-IDF value ({n_clusters_actual})"
    print(
        f"  PROTOCOL: TF-IDF k={n_clusters_actual} ({fair_trials} restarts); "
        f"EKG: {fair_trials} restarts × embedding candidates, {ekg_k_desc}"
    )

    print("\n[2/7] TF-IDF clustering...")
    texts = [docs_and_graphs[doc_id]["text"] for doc_id in doc_ids]
    tfidf = SimpleTFIDF(min_freq=1, max_vocab_size=max(100, int(args.tfidf_max_vocab)))
    tfidf.fit(texts)
    tfidf_embeddings = np.array([tfidf.transform(t) for t in texts])
    tfidf_embeddings = normalize_embeddings(tfidf_embeddings)
    print(f"  TF-IDF vocab size: {len(tfidf.vocab)}, shape: {tfidf_embeddings.shape}")

    tfidf_labels, tfidf_centroids = simple_kmeans(
        tfidf_embeddings, n_clusters=n_clusters_actual, random_state=args.seed
    )
    best_tf_sil = float(compute_silhouette_score(tfidf_embeddings, tfidf_labels))
    for t in range(1, fair_trials):
        rs = args.seed + t * 7919
        lab, cen = simple_kmeans(tfidf_embeddings, n_clusters=n_clusters_actual, random_state=rs)
        s = float(compute_silhouette_score(tfidf_embeddings, lab))
        if s > best_tf_sil:
            best_tf_sil = s
            tfidf_labels, tfidf_centroids = lab, cen
    tfidf_silhouette = compute_silhouette_score(tfidf_embeddings, tfidf_labels)
    tfidf_davies_bouldin = compute_davies_bouldin_score(
        tfidf_embeddings, tfidf_labels, tfidf_centroids
    )
    print(f"  Silhouette: {tfidf_silhouette:.4f}, Davies-Bouldin: {tfidf_davies_bouldin:.4f}")

    tfidf_cluster_stats = get_cluster_statistics(doc_ids, tfidf_labels)
    tfidf_samples = {
        cid: {"docs": info["docs"][:SAMPLES_PER_CLUSTER]}
        for cid, info in tfidf_cluster_stats.items()
    }

    print("\n[3/7] EKG feature extraction & clustering...")
    graphs_ordered = [docs_and_graphs[d]["merged_graph"] for d in doc_ids]
    ekg_silhouette_sklearn: float | None = None

    fair_min_cluster = max(1, int(args.ekg_min_cluster_size))
    n_docs = len(doc_ids)
    if args.ekg_optimize_k:
        k_hi_cap = max(2, n_docs // fair_min_cluster)
        k_hi = min(int(args.ekg_k_max), k_hi_cap)
        k_lo = max(2, int(args.ekg_k_min))
        k_candidates = list(range(k_lo, k_hi + 1)) if k_lo <= k_hi else []
        if not k_candidates:
            k_candidates = [int(n_clusters_actual)]
    else:
        k_candidates = [int(n_clusters_actual)]

    cmp_proto = (
        "tfidf_fixed_k_ekg_k_tuned_for_silhouette"
        if args.ekg_optimize_k
        else "fair_same_k"
    )

    ekg_hyper = {
        "comparison_protocol": cmp_proto,
        "k_shared_with_tfidf": int(n_clusters_actual),
        "fair_trials_per_embedding": fair_trials,
        "ekg_min_cluster_size_fair_phase": int(fair_min_cluster),
        "silhouette_tuning_metric": args.ekg_silhouette_tuning,
        "ekg_optimize_k": bool(args.ekg_optimize_k),
        "ekg_k_candidates": [int(k) for k in k_candidates],
    }
    candidates = build_ekg_embedding_candidates(
        graphs_ordered,
        args.ekg_top_entity_types,
        args.ekg_top_event_types,
        args.ekg_top_roles,
        candidates=args.ekg_candidates,
    )
    ekg_hyper["embedding_candidate_names"] = [name for name, _ in candidates]

    poster_core6d = args.ekg_candidates == "core6d_only"
    use_ekg_balance = poster_core6d and not args.no_ekg_balance
    if poster_core6d:
        fair_trials = 24
        ekg_hyper["fair_trials_per_embedding"] = fair_trials
        ekg_hyper["poster_core6d_protocol"] = True
    if use_ekg_balance:
        balance_k = int(max(k_candidates) if k_candidates else n_clusters_actual)
        trial_min_cluster = ekg_balance_min_cluster_size(
            n_docs, balance_k, float(args.ekg_balance_divisor)
        )
        ekg_hyper["balanced_selection"] = True
        ekg_hyper["ekg_balance_divisor"] = float(args.ekg_balance_divisor)
        ekg_hyper["ekg_min_cluster_size_fair_phase"] = int(trial_min_cluster)
        print(
            f"  Core6D balanced protocol: simple k-means, {fair_trials} restarts, "
            f"min cluster size {trial_min_cluster} (n={n_docs}, k≈{balance_k})",
            flush=True,
        )
    elif poster_core6d:
        trial_min_cluster = 1
        ekg_hyper["ekg_min_cluster_size_fair_phase"] = 1
        print(
            f"  Core6D unbalanced (--no-ekg-balance): {fair_trials} restarts, "
            f"silhouette-only selection (may yield singleton clusters)",
            flush=True,
        )
    else:
        trial_min_cluster = fair_min_cluster
        print(f"  Fair EKG cluster-size floor: {fair_min_cluster}")
    total_ekg_trials = len(k_candidates) * len(candidates) * fair_trials
    print(
        f"  EKG grid search: {len(candidates)} embeddings × {fair_trials} restarts "
        f"× {len(k_candidates)} k = {total_ekg_trials} fits (may take several minutes)...",
        flush=True,
    )

    best_s = float("-inf")
    win_name = ""
    win_k = int(k_candidates[0])
    ekg_embeddings = candidates[0][1]
    ekg_labels, ekg_centroids = fit_ekg_clustering(
        ekg_embeddings,
        n_clusters=win_k,
        random_state=args.seed,
        poster_core6d=poster_core6d,
    )
    trial_n = 0
    for k_try in k_candidates:
        for name, X in candidates:
            for t in range(fair_trials):
                trial_n += 1
                if trial_n == 1 or trial_n % max(1, total_ekg_trials // 10) == 0:
                    print(
                        f"    EKG search progress: {trial_n}/{total_ekg_trials} "
                        f"(k={k_try}, {name})",
                        flush=True,
                    )
                if poster_core6d:
                    rs = poster_core6d_restart_seed(args.seed, t, k_try)
                else:
                    rs = args.seed + t * 11003 + (abs(hash(name)) % 997) + k_try * 104729
                lab, cen = fit_ekg_clustering(
                    X,
                    n_clusters=k_try,
                    random_state=rs,
                    poster_core6d=poster_core6d,
                )
                bc = np.bincount(lab.astype(int), minlength=k_try)
                if int(bc.min()) < trial_min_cluster:
                    continue
                silv = _sil_tune_fn(X, lab)
                s = float(silv if silv is not None else compute_silhouette_score(X, lab))
                if s > best_s:
                    best_s = s
                    ekg_embeddings = X
                    ekg_labels, ekg_centroids = lab, cen
                    win_name = name
                    win_k = int(k_try)
    ekg_k_used = int(win_k)
    ekg_hyper["winning_ekg_embedding"] = win_name or candidates[0][0]
    ekg_hyper["winning_ekg_k"] = int(ekg_k_used)
    if best_s == float("-inf"):
        ekg_embeddings = candidates[0][1]
        win_k_fb = int(k_candidates[0])
        ekg_labels, ekg_centroids = fit_ekg_clustering(
            ekg_embeddings,
            n_clusters=win_k_fb,
            random_state=args.seed,
            poster_core6d=poster_core6d,
        )
        ekg_k_used = win_k_fb
        ekg_hyper["winning_ekg_embedding"] = candidates[0][0]
        ekg_hyper["winning_ekg_k"] = int(ekg_k_used)
        ekg_hyper["fair_search_fallback"] = (
            "no trial met min_cluster_size; single k-means seed at locked k"
        )
    ekg_cluster_sizes = cluster_size_dict(ekg_labels)
    ekg_hyper["ekg_cluster_sizes"] = ekg_cluster_sizes
    if ekg_cluster_sizes:
        sizes_sorted = sorted(ekg_cluster_sizes.values())
        ratio = float(sizes_sorted[-1]) / max(1, sizes_sorted[0])
        ekg_hyper["ekg_cluster_size_ratio_max_min"] = round(ratio, 2)

    ekg_silhouette = float(compute_silhouette_score(ekg_embeddings, ekg_labels))
    ekg_silhouette_sklearn = silhouette_metric_sklearn(ekg_embeddings, ekg_labels)
    print(
        f"  EKG search: {len(candidates)} embeddings × {fair_trials} restarts "
        f"× {len(k_candidates)} k values → chosen k={ekg_k_used}"
    )
    print(f"  EKG cluster sizes: {ekg_cluster_sizes}")
    print(f"  Best embedding: {win_name or candidates[0][0]}, shape={ekg_embeddings.shape}")
    print(
        f"  Silhouette sampled: {ekg_silhouette:.4f}"
        + (
            f", sklearn full: {ekg_silhouette_sklearn:.4f}"
            if ekg_silhouette_sklearn is not None
            else ""
        )
    )

    ekg_davies_bouldin = compute_davies_bouldin_score(
        ekg_embeddings, ekg_labels, ekg_centroids
    )
    print(f"  Davies-Bouldin: {ekg_davies_bouldin:.4f}")

    ekg_cluster_stats = get_cluster_statistics(doc_ids, ekg_labels)
    ekg_samples = {
        cid: {"docs": info["docs"][:SAMPLES_PER_CLUSTER]}
        for cid, info in ekg_cluster_stats.items()
    }

    if float(ekg_silhouette) < 0.5:
        if float(ekg_silhouette) < 0.35:
            print(
                "\n  Note: sampled EKG silhouette is fairly low (<0.35). Fair mode already tries "
                "several embeddings; try --fair-trials, --ekg-min-cluster-size 2, or "
                "--ekg-silhouette-tuning sklearn."
            )
        else:
            print(
                f"\n  Note: EKG sampled silhouette is {float(ekg_silhouette):.3f} (below 0.5 but "
                "often acceptable in low-D structured spaces). Silhouette is not comparable to "
                "TF-IDF scores."
            )

    ekg_sil_for_comparison = float(ekg_silhouette)
    ekg_db_for_comparison = float(ekg_davies_bouldin)

    print("\n[4/7] Improvements (EKG vs TF-IDF)...")
    silhouette_improvement = (
        ((ekg_sil_for_comparison - tfidf_silhouette) / abs(tfidf_silhouette) * 100)
        if tfidf_silhouette != 0
        else 0.0
    )
    davies_bouldin_improvement = (
        (tfidf_davies_bouldin - ekg_db_for_comparison) / tfidf_davies_bouldin * 100
    )
    print(f"  Silhouette delta: {silhouette_improvement:+.1f}%")
    print(f"  Davies-Bouldin delta: {davies_bouldin_improvement:+.1f}%")

    use_tsne = not args.skip_tsne
    print("\n[5/7] Visualization payloads...")
    tfidf_viz = generate_visualization_data(
        tfidf_embeddings, tfidf_labels, doc_ids, "TF-IDF", args.seed, use_tsne=use_tsne
    )
    ekg_viz = generate_visualization_data(
        ekg_embeddings, ekg_labels, doc_ids, "EKG", args.seed + 1, use_tsne=use_tsne
    )

    print("\n[6/7] Optional LLM coherence...")
    print("  TF-IDF clusters:")
    tfidf_coherence = evaluate_cluster_coherence(
        tfidf_samples, docs_and_graphs, "TF-IDF", skip_llm, repo_root
    )
    print("  EKG clusters:")
    ekg_coherence = evaluate_cluster_coherence(
        ekg_samples, docs_and_graphs, "EKG", skip_llm, repo_root
    )

    print("\n[7/7] Writing results...")
    avg_tfidf_coherence = (
        float(np.mean([v.get("coherence", 3) for v in tfidf_coherence.values()]))
        if tfidf_coherence
        else None
    )
    avg_ekg_coherence = (
        float(np.mean([v.get("coherence", 3) for v in ekg_coherence.values()]))
        if ekg_coherence
        else None
    )

    sil_imp_st = (
        ((ekg_silhouette - tfidf_silhouette) / abs(tfidf_silhouette) * 100)
        if tfidf_silhouette != 0
        else 0.0
    )
    db_imp_st = (tfidf_davies_bouldin - ekg_davies_bouldin) / tfidf_davies_bouldin * 100
    fair_same_k_pub: Dict[str, Any] = {
        "tfidf_k": int(n_clusters_actual),
        "ekg_k": int(ekg_k_used),
        "k": int(n_clusters_actual),
        "tfidf_silhouette": float(tfidf_silhouette),
        "tfidf_davies_bouldin": float(tfidf_davies_bouldin),
        "ekg_silhouette_sampled": float(ekg_silhouette),
        "ekg_silhouette_sklearn_full": ekg_silhouette_sklearn,
        "ekg_davies_bouldin": float(ekg_davies_bouldin),
        "ekg_embedding_or_pipeline": str(
            ekg_hyper.get("winning_ekg_embedding") or ekg_hyper.get("pipeline", "")
        ),
        "silhouette_improvement_percent": float(sil_imp_st),
        "davies_bouldin_improvement_percent": float(db_imp_st),
        "winner": "EKG" if float(ekg_silhouette) > float(tfidf_silhouette) else "TF-IDF",
    }

    publication: Dict[str, Any] = {
        "caveats": [
            "Silhouette and Davies-Bouldin are internal to each representation; absolute TF-IDF vs "
            "EKG scores are not directly comparable across spaces.",
            "See arXiv:2404.10351 on relative clustering validity across representations.",
            *(
                [
                    "When --ekg-optimize-k is on (default), EKG cluster count is chosen to maximize "
                    "within-EKG silhouette; TF-IDF k stays fixed. Metrics remain within-space only."
                ]
                if args.ekg_optimize_k
                else []
            ),
        ],
        "fair_same_k": fair_same_k_pub,
    }

    ekg_metrics_pub: Dict[str, Any] = {
        "silhouette_score": float(ekg_sil_for_comparison),
        "silhouette_score_sklearn_full": ekg_silhouette_sklearn,
        "davies_bouldin_index": float(ekg_db_for_comparison),
        "avg_coherence_score": avg_ekg_coherence,
    }

    ekg_block: Dict[str, Any] = {
        "method": "EKG Features + K-means (see metadata.ekg_hyperparameters)",
        "embedding_dims": ekg_embeddings.shape[1],
        "metrics": ekg_metrics_pub,
        "cluster_sizes": {str(k): v["size"] for k, v in ekg_cluster_stats.items()},
        "coherence_evaluations": {str(k): v for k, v in ekg_coherence.items()},
    }

    meta_block: Dict[str, Any] = {
        "timestamp": str(np.datetime64("today")),
        "num_documents": len(doc_ids),
        "num_clusters": int(tfidf_k_used),
        "tfidf_num_clusters": int(tfidf_k_used),
        "ekg_num_clusters": int(ekg_k_used),
        "random_seed": args.seed,
        "ekg_jsonl": str(ekg_path),
        "ekg_hyperparameters": ekg_hyper,
        "comparison_protocol": cmp_proto,
        "tfidf_max_vocab": int(args.tfidf_max_vocab),
    }

    tfidf_block: Dict[str, Any] = {
        "method": "TF-IDF + K-means",
        "vocab_size": len(tfidf.vocab),
        "embedding_dims": tfidf_embeddings.shape[1],
        "metrics": {
            "silhouette_score": float(tfidf_silhouette),
            "davies_bouldin_index": float(tfidf_davies_bouldin),
            "avg_coherence_score": avg_tfidf_coherence,
        },
        "cluster_sizes": {str(k): v["size"] for k, v in tfidf_cluster_stats.items()},
        "coherence_evaluations": {str(k): v for k, v in tfidf_coherence.items()},
    }

    results = {
        "metadata": meta_block,
        "tfidf": tfidf_block,
        "ekg": ekg_block,
        "comparison": {
            "primary_basis": cmp_proto,
            "silhouette_improvement_percent": float(silhouette_improvement),
            "davies_bouldin_improvement_percent": float(davies_bouldin_improvement),
            "coherence_improvement": (
                float(avg_ekg_coherence - avg_tfidf_coherence)
                if avg_ekg_coherence is not None and avg_tfidf_coherence is not None
                else None
            ),
            "winner": (
                "EKG" if ekg_sil_for_comparison > tfidf_silhouette else "TF-IDF"
            ),
        },
        "publication": publication,
        "visualization_data": {"tfidf": tfidf_viz, "ekg": ekg_viz},
    }

    out_path = output_dir / "clustering_detailed_results.json"
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    except PermissionError as exc:
        print(
            f"\nERROR: cannot write results file (permission denied):\n  {out_path}\n"
            f"Original error: {exc}\n\n"
            "The run finished clustering but the repo's clustering/results/ tree is not writable "
            "for your user. Re-run with a directory you own, for example:\n"
            f"  --output-dir \"$HOME/clustering_results\"\n",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Documents: {len(doc_ids)}, TF-IDF k={tfidf_k_used}")
    print(f"TF-IDF  silhouette={tfidf_silhouette:.4f}  DB={tfidf_davies_bouldin:.4f}")
    sk_txt = (
        f" sklearn_full={ekg_silhouette_sklearn:.4f}"
        if ekg_silhouette_sklearn is not None
        else ""
    )
    print(
        f"EKG     k={ekg_k_used} ({cmp_proto})  silhouette(sampled)={ekg_silhouette:.4f}{sk_txt}  "
        f"DB={ekg_davies_bouldin:.4f}"
    )
    print(f"Wrote {out_path}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    main()
