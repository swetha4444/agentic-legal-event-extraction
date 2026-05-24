"""
Comprehensive Clustering Comparison: EKG vs TF-IDF

This script:
1. Clusters documents using both TF-IDF and EKG features
2. Computes clustering quality metrics (Silhouette, Davies-Bouldin)
3. Generates visualizations (metrics, t-SNE, sample clusters)
4. Evaluates semantic coherence using Claude
5. Produces a detailed report comparing both approaches
"""

import json
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import random
import hashlib
import re
from math import log

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

# ============================================================================
# CONFIGURATION
# ============================================================================

RANDOM_SEED = 42
N_CLUSTERS = 8  # For ~140 docs, 8-10 clusters is reasonable
MAX_DOCS = None  # None = use all, or set to limit
SAMPLES_PER_CLUSTER = 3
EVAL_CLUSTERS = 6  # How many clusters to do LLM evaluation on

# ============================================================================
# 1. DATA LOADING
# ============================================================================

def load_documents_and_graphs(ekg_path: str, limit: int = None) -> Dict[str, Dict]:
    """Load documents and their EKG representations from JSONL file."""
    docs_and_graphs = {}

    with open(ekg_path, 'r') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                data = json.loads(line)
                doc_id = data.get('doc_id', f'doc_{i}')

                # Extract text from chunks
                chunks = data.get('chunks', [])
                text = ' '.join([chunk.get('text', '') for chunk in chunks])

                # Skip empty documents
                if not text or len(text.strip()) < 100:
                    continue

                docs_and_graphs[doc_id] = {
                    'doc_id': doc_id,
                    'text': text,
                    'merged_graph': data.get('merged_graph', {}),
                    'merge_stats': data.get('merge_stats', {}),
                    'num_chunks': len(chunks),
                }
            except json.JSONDecodeError:
                continue

    print(f"✓ Loaded {len(docs_and_graphs)} documents with EKG representations")
    return docs_and_graphs


def extract_ekg_features(graph: Dict) -> Dict[str, Any]:
    """Extract structured features from an EKG representation."""
    entities = graph.get('entities', [])
    events = graph.get('events', [])
    temporal_edges = graph.get('temporal_edges', [])
    causal_edges = graph.get('causal_edges', [])

    features = {
        'num_entities': len(entities),
        'num_events': len(events),
        'num_temporal_edges': len(temporal_edges),
        'num_causal_edges': len(causal_edges),
        'graph_density': (len(temporal_edges) + len(causal_edges)) / max(1, len(entities) + len(events)),
        'entities': entities[:30],  # Keep for later analysis
        'events': events[:30],
    }
    return features


# ============================================================================
# 2. TF-IDF IMPLEMENTATION
# ============================================================================

class SimpleTFIDF:
    """Lightweight TF-IDF without external libraries."""

    def __init__(self, min_freq: int = 1, max_vocab_size: int = 3000):
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.vocab = {}
        self.idf = {}
        self.fitted = False

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        tokens = text.split()
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'is', 'was', 'are', 'be', 'been', 'being', 'have', 'has', 'had',
                     'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
                     'must', 'can', 'this', 'that', 'these', 'those'}
        tokens = [t for t in tokens if len(t) > 2 and t not in stopwords]
        return tokens

    def fit(self, texts: List[str]):
        """Build vocabulary and IDF."""
        doc_freq = defaultdict(int)

        for text in texts:
            tokens = self._tokenize(text)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_freq[token] += 1

        vocab_items = [(token, count) for token, count in doc_freq.items()
                       if count >= self.min_freq]
        vocab_items.sort(key=lambda x: x[1], reverse=True)
        vocab_items = vocab_items[:self.max_vocab_size]

        self.vocab = {token: idx for idx, (token, _) in enumerate(vocab_items)}

        n_docs = len(texts)
        self.idf = {}
        for token, doc_count in vocab_items:
            self.idf[token] = log(n_docs / doc_count) if doc_count > 0 else 0

        self.fitted = True

    def transform(self, text: str) -> np.ndarray:
        """Convert text to TF-IDF vector."""
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


# ============================================================================
# 3. CLUSTERING
# ============================================================================

def simple_kmeans(embeddings: np.ndarray, n_clusters: int = 8, max_iters: int = 50,
                  random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Simple K-means clustering."""
    np.random.seed(random_state)
    n_samples, n_features = embeddings.shape

    indices = np.random.choice(n_samples, min(n_clusters, n_samples), replace=False)
    centroids = embeddings[indices].copy()
    actual_clusters = centroids.shape[0]

    for iteration in range(max_iters):
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


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Normalize to unit length."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / (norms + 1e-8)


# ============================================================================
# 4. CLUSTERING METRICS
# ============================================================================

def compute_silhouette_score(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Compute silhouette score."""
    n_samples = len(embeddings)
    scores = []

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

        silhouette = (b - a) / max(a, b) if max(a, b) > 0 else 0
        scores.append(silhouette)

    return float(np.mean(scores)) if scores else 0.0


def compute_davies_bouldin_score(embeddings: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    """Compute Davies-Bouldin index (lower is better)."""
    n_clusters = len(np.unique(labels))

    # Compute average distance within each cluster
    within_distances = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_mask = labels == i
        if cluster_mask.sum() > 0:
            within_distances[i] = np.mean(np.linalg.norm(embeddings[cluster_mask] - centroids[i], axis=1))

    # Compute Davies-Bouldin index
    db_index = 0.0
    for i in range(n_clusters):
        max_ratio = 0.0
        for j in range(n_clusters):
            if i != j:
                between_distance = np.linalg.norm(centroids[i] - centroids[j])
                if between_distance > 0:
                    ratio = (within_distances[i] + within_distances[j]) / between_distance
                    max_ratio = max(max_ratio, ratio)
        db_index += max_ratio

    return db_index / max(1, n_clusters)


def get_cluster_statistics(doc_ids: List[str], labels: np.ndarray) -> Dict[int, Dict]:
    """Get statistics for each cluster."""
    clusters = defaultdict(list)

    for doc_id, label in zip(doc_ids, labels):
        clusters[int(label)].append(doc_id)

    stats = {}
    for cluster_id, doc_list in clusters.items():
        stats[cluster_id] = {
            'size': len(doc_list),
            'docs': doc_list,
        }

    return stats


# ============================================================================
# 5. VISUALIZATION DATA GENERATION
# ============================================================================

def generate_visualization_data(embeddings: np.ndarray, labels: np.ndarray, doc_ids: List[str],
                               method_name: str) -> Dict[str, Any]:
    """Generate data for visualizations."""

    # Try to use sklearn for t-SNE if available, otherwise use random projection
    try:
        from sklearn.manifold import TSNE
        print(f"  Computing t-SNE for {method_name}...")
        tsne = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=min(30, len(embeddings)-1))
        tsne_embeddings = tsne.fit_transform(embeddings)
    except ImportError:
        print(f"  sklearn not available, using random projection for {method_name}...")
        # Fallback: simple random projection
        tsne_embeddings = np.random.randn(len(embeddings), 2)

    return {
        'method': method_name,
        'tsne_2d': tsne_embeddings.tolist(),
        'labels': labels.tolist(),
        'doc_ids': doc_ids,
    }


# ============================================================================
# 6. CLUSTER COHERENCE EVALUATION
# ============================================================================

def evaluate_cluster_coherence(cluster_samples: Dict[int, Dict],
                               docs_and_graphs: Dict, method_name: str) -> Dict:
    """Use Claude to evaluate semantic coherence of clusters."""

    if not HAS_ANTHROPIC:
        print(f"  Skipping Claude evaluation (anthropic not available)")
        return {}

    client = anthropic.Anthropic()
    coherence_results = {}

    for cluster_id, cluster_info in list(cluster_samples.items())[:EVAL_CLUSTERS]:
        doc_ids = cluster_info['docs']

        # Prepare document snippets
        doc_snippets = []
        for doc_id in doc_ids[:3]:  # Max 3 docs
            text = docs_and_graphs.get(doc_id, {}).get('text', '')[:400]
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
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text
            try:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    result = json.loads(json_str)
                else:
                    result = {"coherence_score": 3, "reasoning": "Parse error"}
            except json.JSONDecodeError:
                result = {"coherence_score": 3, "reasoning": "Parse error"}

            coherence_results[cluster_id] = {
                'docs': doc_ids,
                'coherence': result.get('coherence_score', 3),
                'reasoning': result.get('reasoning', ''),
            }

            print(f"    Cluster {cluster_id}: {result.get('coherence_score', '?')}/5")

        except Exception as e:
            print(f"    Error evaluating cluster {cluster_id}: {e}")

    return coherence_results


# ============================================================================
# 7. MAIN ANALYSIS PIPELINE
# ============================================================================

def main():
    ekg_path = Path('/sessions/kind-fervent-noether/mnt/agentic-legal-event-extraction-swetha/data/outputs/qwen122b_hybrid_doc_graphs_final.jsonl')
    output_dir = Path('/sessions/kind-fervent-noether/mnt/agentic-legal-event-extraction-swetha')

    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    print("=" * 80)
    print("CLUSTERING COMPARISON ANALYSIS: EKG vs TF-IDF")
    print("=" * 80)

    # ====== LOAD DATA ======
    print("\n[1/7] Loading documents and EKG representations...")
    docs_and_graphs = load_documents_and_graphs(str(ekg_path), limit=MAX_DOCS)
    doc_ids = list(docs_and_graphs.keys())

    print(f"  Total documents: {len(doc_ids)}")
    print(f"  Average doc length: {np.mean([len(d['text']) for d in docs_and_graphs.values()]):.0f} chars")

    # Determine optimal number of clusters
    n_clusters_actual = min(N_CLUSTERS, max(3, len(doc_ids) // 15))
    print(f"  Using {n_clusters_actual} clusters")

    # ====== TFIDF CLUSTERING ======
    print("\n[2/7] TF-IDF Clustering...")
    texts = [docs_and_graphs[doc_id]['text'] for doc_id in doc_ids]

    tfidf = SimpleTFIDF(min_freq=1, max_vocab_size=2000)
    tfidf.fit(texts)
    tfidf_embeddings = np.array([tfidf.transform(text) for text in texts])
    tfidf_embeddings = normalize_embeddings(tfidf_embeddings)

    print(f"  TF-IDF vocab size: {len(tfidf.vocab)}")
    print(f"  Embedding shape: {tfidf_embeddings.shape}")

    tfidf_labels, tfidf_centroids = simple_kmeans(tfidf_embeddings, n_clusters=n_clusters_actual)
    tfidf_silhouette = compute_silhouette_score(tfidf_embeddings, tfidf_labels)
    tfidf_davies_bouldin = compute_davies_bouldin_score(tfidf_embeddings, tfidf_labels, tfidf_centroids)

    print(f"  ✓ Silhouette Score: {tfidf_silhouette:.4f}")
    print(f"  ✓ Davies-Bouldin Index: {tfidf_davies_bouldin:.4f}")

    tfidf_cluster_stats = get_cluster_statistics(doc_ids, tfidf_labels)
    tfidf_samples = {cid: {'docs': info['docs'][:SAMPLES_PER_CLUSTER]}
                     for cid, info in tfidf_cluster_stats.items()}

    # ====== EKG CLUSTERING ======
    print("\n[3/7] EKG Feature Extraction & Clustering...")
    ekg_features = []
    for doc_id in doc_ids:
        graph = docs_and_graphs[doc_id]['merged_graph']
        features = extract_ekg_features(graph)

        feature_vec = [
            features['num_entities'],
            features['num_events'],
            features['num_temporal_edges'],
            features['num_causal_edges'],
            features['graph_density'] * 100,
            len(features['entities']),
            len(features['events']),
        ]
        ekg_features.append(np.array(feature_vec, dtype=np.float32))

    ekg_embeddings = np.array(ekg_features)
    ekg_embeddings = normalize_embeddings(ekg_embeddings)

    print(f"  EKG feature dimensions: {ekg_embeddings.shape[1]}")
    print(f"  Embedding shape: {ekg_embeddings.shape}")

    ekg_labels, ekg_centroids = simple_kmeans(ekg_embeddings, n_clusters=n_clusters_actual)
    ekg_silhouette = compute_silhouette_score(ekg_embeddings, ekg_labels)
    ekg_davies_bouldin = compute_davies_bouldin_score(ekg_embeddings, ekg_labels, ekg_centroids)

    print(f"  ✓ Silhouette Score: {ekg_silhouette:.4f}")
    print(f"  ✓ Davies-Bouldin Index: {ekg_davies_bouldin:.4f}")

    ekg_cluster_stats = get_cluster_statistics(doc_ids, ekg_labels)
    ekg_samples = {cid: {'docs': info['docs'][:SAMPLES_PER_CLUSTER]}
                   for cid, info in ekg_cluster_stats.items()}

    # ====== COMPUTE IMPROVEMENTS ======
    print("\n[4/7] Computing improvements...")
    silhouette_improvement = ((ekg_silhouette - tfidf_silhouette) / abs(tfidf_silhouette) * 100) if tfidf_silhouette != 0 else 0
    davies_bouldin_improvement = ((tfidf_davies_bouldin - ekg_davies_bouldin) / tfidf_davies_bouldin * 100)  # Lower is better

    print(f"  Silhouette improvement: {silhouette_improvement:+.1f}%")
    print(f"  Davies-Bouldin improvement: {davies_bouldin_improvement:+.1f}%")

    # ====== GENERATE VISUALIZATION DATA ======
    print("\n[5/7] Generating visualization data...")
    tfidf_viz = generate_visualization_data(tfidf_embeddings, tfidf_labels, doc_ids, "TF-IDF")
    ekg_viz = generate_visualization_data(ekg_embeddings, ekg_labels, doc_ids, "EKG")

    # ====== EVALUATE COHERENCE ======
    print("\n[6/7] Evaluating cluster coherence with Claude...")
    print("  TF-IDF clusters:")
    tfidf_coherence = evaluate_cluster_coherence(tfidf_samples, docs_and_graphs, "TF-IDF")

    print("  EKG clusters:")
    ekg_coherence = evaluate_cluster_coherence(ekg_samples, docs_and_graphs, "EKG")

    # ====== COMPILE RESULTS ======
    print("\n[7/7] Compiling results...")

    avg_tfidf_coherence = np.mean([v.get('coherence', 3) for v in tfidf_coherence.values()]) if tfidf_coherence else None
    avg_ekg_coherence = np.mean([v.get('coherence', 3) for v in ekg_coherence.values()]) if ekg_coherence else None

    results = {
        'metadata': {
            'timestamp': str(np.datetime64('today')),
            'num_documents': len(doc_ids),
            'num_clusters': n_clusters_actual,
            'random_seed': RANDOM_SEED,
        },
        'tfidf': {
            'method': 'TF-IDF + K-means',
            'vocab_size': len(tfidf.vocab),
            'embedding_dims': tfidf_embeddings.shape[1],
            'metrics': {
                'silhouette_score': float(tfidf_silhouette),
                'davies_bouldin_index': float(tfidf_davies_bouldin),
                'avg_coherence_score': float(avg_tfidf_coherence) if avg_tfidf_coherence else None,
            },
            'cluster_sizes': {str(k): v['size'] for k, v in tfidf_cluster_stats.items()},
            'coherence_evaluations': {str(k): v for k, v in tfidf_coherence.items()},
        },
        'ekg': {
            'method': 'EKG Features + K-means',
            'embedding_dims': ekg_embeddings.shape[1],
            'metrics': {
                'silhouette_score': float(ekg_silhouette),
                'davies_bouldin_index': float(ekg_davies_bouldin),
                'avg_coherence_score': float(avg_ekg_coherence) if avg_ekg_coherence else None,
            },
            'cluster_sizes': {str(k): v['size'] for k, v in ekg_cluster_stats.items()},
            'coherence_evaluations': {str(k): v for k, v in ekg_coherence.items()},
        },
        'comparison': {
            'silhouette_improvement_percent': float(silhouette_improvement),
            'davies_bouldin_improvement_percent': float(davies_bouldin_improvement),
            'coherence_improvement': float(avg_ekg_coherence - avg_tfidf_coherence) if (avg_ekg_coherence and avg_tfidf_coherence) else None,
            'winner': 'EKG' if ekg_silhouette > tfidf_silhouette else 'TF-IDF',
        },
        'visualization_data': {
            'tfidf': tfidf_viz,
            'ekg': ekg_viz,
        }
    }

    # Save detailed results
    with open(output_dir / 'clustering_detailed_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # ====== PRINT SUMMARY ======
    print("\n" + "=" * 80)
    print("CLUSTERING COMPARISON SUMMARY")
    print("=" * 80)
    print(f"\nDocuments Analyzed: {len(doc_ids)}")
    print(f"Clusters: {n_clusters_actual}")

    print("\n--- TF-IDF CLUSTERING ---")
    print(f"Vocabulary Size: {len(tfidf.vocab)} terms")
    print(f"Silhouette Score: {tfidf_silhouette:.4f}")
    print(f"Davies-Bouldin Index: {tfidf_davies_bouldin:.4f}")
    if avg_tfidf_coherence:
        print(f"Avg Semantic Coherence: {avg_tfidf_coherence:.2f}/5.0")

    print("\n--- EKG CLUSTERING ---")
    print(f"Feature Dimensions: 7 (entities, events, edges, density)")
    print(f"Silhouette Score: {ekg_silhouette:.4f}")
    print(f"Davies-Bouldin Index: {ekg_davies_bouldin:.4f}")
    if avg_ekg_coherence:
        print(f"Avg Semantic Coherence: {avg_ekg_coherence:.2f}/5.0")

    print("\n--- IMPROVEMENTS (EKG vs TF-IDF) ---")
    print(f"Silhouette: {silhouette_improvement:+.1f}%")
    print(f"Davies-Bouldin: {davies_bouldin_improvement:+.1f}% (lower is better)")
    if avg_ekg_coherence and avg_tfidf_coherence:
        print(f"Semantic Coherence: {avg_ekg_coherence - avg_tfidf_coherence:+.2f} points")

    print(f"\n✓ Detailed results: {output_dir / 'clustering_detailed_results.json'}")
    print("=" * 80)

    return results


if __name__ == '__main__':
    main()
