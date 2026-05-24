#!/usr/bin/env python3
"""
Standalone script to test Core 6D embedding.

Runs silhouette sweep for k=2-40 using only these features:
1. log1p(num_entities)
2. log1p(num_events)
3. log1p(num_temporal_edges)
4. log1p(num_causal_edges)
5. density
6. ratio_temporal

Usage:
    python3 test_core6d_embedding.py \
        --ekg-jsonl /path/to/qwen122b_hybrid_doc_graphs_final.jsonl \
        --output-dir /path/to/output
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


def load_documents_and_graphs(ekg_path: str, limit: int = None) -> Dict[str, Dict]:
    """Load documents with EKG representations."""
    docs = {}
    with open(ekg_path, 'r') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                data = json.loads(line)
                doc_id = data.get('doc_id', f'doc_{i}')
                chunks = data.get('chunks', [])
                text = ' '.join([chunk.get('text', '') for chunk in chunks])

                if not text or len(text.strip()) < 100:
                    continue

                docs[doc_id] = {
                    'doc_id': doc_id,
                    'merged_graph': data.get('merged_graph', {}),
                }
            except json.JSONDecodeError:
                continue

    return docs


def build_core6d_embedding(graphs: List[Dict]) -> np.ndarray:
    """
    Build 6D embedding using core structural features only.

    Features:
    1. log1p(num_entities) - Actor count
    2. log1p(num_events) - Incident count
    3. log1p(num_temporal_edges) - Timeline complexity
    4. log1p(num_causal_edges) - Causal chains
    5. density - Connectivity
    6. ratio_temporal - Event flow (temporal vs causal)
    """
    rows = []

    for graph in graphs:
        entities = graph.get('entities') or []
        events = graph.get('events') or []
        temporal_edges = graph.get('temporal_edges') or []
        causal_edges = graph.get('causal_edges') or []

        n_ent = len(entities)
        n_evt = len(events)
        n_te = len(temporal_edges)
        n_ce = len(causal_edges)
        ec = n_te + n_ce

        density = ec / max(1, n_ent + n_evt)
        ratio_temporal = n_te / max(1, ec) if ec else 0.0

        feature_vector = [
            np.log1p(n_ent),      # Feature 1
            np.log1p(n_evt),      # Feature 2
            np.log1p(n_te),       # Feature 3
            np.log1p(n_ce),       # Feature 4
            density,              # Feature 5
            ratio_temporal,       # Feature 6
        ]
        rows.append(feature_vector)

    # Convert to array
    X = np.array(rows, dtype=np.float32)

    # L2 normalize rows
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_normalized = X / (norms + 1e-8)

    return X_normalized.astype(np.float64)


def simple_kmeans(
    embeddings: np.ndarray,
    n_clusters: int = 8,
    max_iters: int = 50,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Basic K-means clustering."""
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


def compute_silhouette_score(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Compute silhouette score (sampled)."""
    n_samples = len(embeddings)
    scores = []

    # Sample for efficiency
    sample_indices = np.random.choice(n_samples, min(n_samples, 100), replace=False)

    for i in sample_indices:
        same_cluster = labels == labels[i]
        if same_cluster.sum() <= 1:
            continue

        a = np.mean(np.linalg.norm(embeddings[same_cluster] - embeddings[i], axis=1))

        b = float('inf')
        for cluster_id in np.unique(labels):
            if cluster_id != labels[i]:
                cluster = labels == cluster_id
                if cluster.sum() > 0:
                    dist = np.mean(np.linalg.norm(embeddings[cluster] - embeddings[i], axis=1))
                    b = min(b, dist)

        if b == float('inf'):
            continue

        silhouette = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
        scores.append(silhouette)

    return float(np.mean(scores)) if scores else 0.0


def main():
    parser = argparse.ArgumentParser(description='Test Core 6D Embedding')
    parser.add_argument('--ekg-jsonl', type=Path, required=True, help='Path to JSONL file')
    parser.add_argument('--output-dir', type=Path, required=True, help='Output directory')
    parser.add_argument('--k-min', type=int, default=2, help='Min k')
    parser.add_argument('--k-max', type=int, default=40, help='Max k')
    parser.add_argument('--restarts-per-k', type=int, default=24, help='Restarts per k')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print("CORE 6D EMBEDDING TEST")
    print(f"{'='*80}\n")

    # Load documents
    print("[1/3] Loading documents...")
    docs = load_documents_and_graphs(str(args.ekg_jsonl))
    doc_ids = list(docs.keys())
    graphs = [docs[doc_id]['merged_graph'] for doc_id in doc_ids]
    print(f"  Loaded {len(docs)} documents\n")

    # Build embedding
    print("[2/3] Building Core 6D embedding...")
    embeddings = build_core6d_embedding(graphs)
    print(f"  Embedding shape: {embeddings.shape}\n")

    # K-sweep
    print("[3/3] Running silhouette sweep (k={} to {})...\n".format(args.k_min, args.k_max))

    results = {
        'metadata': {
            'num_documents': len(doc_ids),
            'embedding': 'core6d_row_l2',
            'k_range': [args.k_min, args.k_max],
            'restarts_per_k': args.restarts_per_k,
        },
        'results': {}
    }

    best_k = None
    best_silhouette = -1

    for k in range(args.k_min, args.k_max + 1):
        best_sil_at_k = -1

        for restart in range(args.restarts_per_k):
            rs = args.seed + restart * 9973 + k * 17
            labels, _ = simple_kmeans(embeddings, n_clusters=k, random_state=rs)

            if len(np.unique(labels)) < 2:
                continue

            sil = compute_silhouette_score(embeddings, labels)
            best_sil_at_k = max(best_sil_at_k, sil)

        if best_sil_at_k > best_silhouette:
            best_silhouette = best_sil_at_k
            best_k = k

        results['results'][k] = {
            'silhouette': float(best_sil_at_k),
            'best_k': int(best_k) if best_k else None,
            'best_silhouette': float(best_silhouette),
        }

        print(f"k={k:2d}  Silhouette: {best_sil_at_k:.4f}  (Best so far: k={best_k} with {best_silhouette:.4f})")

    # Save results
    output_file = args.output_dir / 'core6d_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Best k: {best_k}")
    print(f"Best silhouette: {best_silhouette:.4f}")
    print(f"\nResults saved: {output_file}\n")


if __name__ == '__main__':
    main()
