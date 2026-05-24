"""
Enhanced EKG Clustering with Semantic Features

Combines structural features (entity/event counts) with semantic features
(entity types, event types, legal scenario characteristics) to cluster
documents by actual legal scenario type, not just complexity.

Compares:
- Structural-only clustering (original 5 features)
- Semantic+Structural clustering (15+ features)
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict, Counter
import random

# ============================================================================
# 1. DATA LOADING & SEMANTIC EXTRACTION
# ============================================================================

def load_documents_with_semantics(ekg_path: str, limit: int = None) -> Dict[str, Dict]:
    """Load documents and extract both structural and semantic features."""
    docs = {}

    with open(ekg_path, 'r') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                data = json.loads(line)
                doc_id = data.get('doc_id', f'doc_{i}')

                # Get text
                chunks = data.get('chunks', [])
                text = ' '.join([chunk.get('text', '') for chunk in chunks])

                if not text or len(text.strip()) < 100:
                    continue

                # Extract semantic data from LLM response
                semantics = extract_semantic_features(chunks)

                # Get graph data
                merged_graph = data.get('merged_graph', {})

                docs[doc_id] = {
                    'doc_id': doc_id,
                    'text': text,
                    'merged_graph': merged_graph,
                    'semantics': semantics,
                }
            except json.JSONDecodeError:
                continue

    print(f"✓ Loaded {len(docs)} documents with semantic features")
    return docs


def extract_semantic_features(chunks: List[Dict]) -> Dict[str, Any]:
    """Extract semantic features from LLM-generated event knowledge."""
    semantics = {
        'entity_types': Counter(),
        'event_types': Counter(),
        'roles': Counter(),
        'entity_names': set(),
        'event_descriptions': [],
        'has_plaintiff': False,
        'has_defendant': False,
        'has_person_entity': False,
        'has_organization_entity': False,
    }

    for chunk in chunks:
        try:
            raw_resp = chunk.get('raw_llm_response', '{}')
            parsed = json.loads(raw_resp)

            # Extract entity types
            entities = parsed.get('entities', [])
            for ent in entities:
                ent_type = ent.get('type', 'Unknown')
                semantics['entity_types'][ent_type] += 1

                if ent_type == 'Person':
                    semantics['has_person_entity'] = True
                elif ent_type == 'Organization':
                    semantics['has_organization_entity'] = True

                # Track names/roles
                name = ent.get('name', '').lower()
                if 'plaintiff' in name:
                    semantics['has_plaintiff'] = True
                if 'defendant' in name:
                    semantics['has_defendant'] = True

                semantics['entity_names'].add(ent_type)

            # Extract event types
            events = parsed.get('events', [])
            for evt in events:
                evt_type = evt.get('type', 'Unknown')
                semantics['event_types'][evt_type] += 1
                semantics['event_descriptions'].append(evt.get('description', ''))

                # Extract participant roles
                participants = evt.get('participants', [])
                for participant in participants:
                    role = participant.get('role', 'Unknown')
                    semantics['roles'][role] += 1

        except json.JSONDecodeError:
            continue

    return semantics


# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================

def create_structural_features(merged_graph: Dict) -> np.ndarray:
    """Create the original 5 structural features."""
    entities = merged_graph.get('entities', [])
    events = merged_graph.get('events', [])
    temporal_edges = merged_graph.get('temporal_edges', [])
    causal_edges = merged_graph.get('causal_edges', [])

    features = [
        len(entities),
        len(events),
        len(temporal_edges),
        len(causal_edges),
        (len(temporal_edges) + len(causal_edges)) / max(1, len(entities) + len(events)) * 100,
    ]

    return np.array(features, dtype=np.float32)


def create_semantic_features(semantics: Dict) -> Dict[str, Any]:
    """Create semantic feature vector from extracted semantics."""

    entity_types = semantics['entity_types']
    event_types = semantics['event_types']
    roles = semantics['roles']

    # Get top entity and event types
    top_entity_types = dict(entity_types.most_common(5))
    top_event_types = dict(event_types.most_common(5))
    top_roles = dict(roles.most_common(5))

    features = {
        'num_entity_types': len(entity_types),
        'num_event_types': len(event_types),
        'num_roles': len(roles),
        'has_plaintiff': 1.0 if semantics['has_plaintiff'] else 0.0,
        'has_defendant': 1.0 if semantics['has_defendant'] else 0.0,
        'has_person': 1.0 if semantics['has_person_entity'] else 0.0,
        'has_organization': 1.0 if semantics['has_organization_entity'] else 0.0,
        'top_entity_types': top_entity_types,
        'top_event_types': top_event_types,
        'top_roles': top_roles,
        'entity_type_counts': dict(entity_types),
        'event_type_counts': dict(event_types),
    }

    return features


def create_combined_feature_vector(doc: Dict) -> np.ndarray:
    """Create combined structural + semantic feature vector (15+ dimensions)."""

    # Structural features (5)
    structural = create_structural_features(doc['merged_graph'])

    # Semantic features (10)
    semantics = create_semantic_features(doc['semantics'])

    semantic_features = [
        semantics['num_entity_types'],
        semantics['num_event_types'],
        semantics['num_roles'],
        semantics['has_plaintiff'],
        semantics['has_defendant'],
        semantics['has_person'],
        semantics['has_organization'],
        # Entity type presence
        semantics['entity_type_counts'].get('Person', 0),
        semantics['entity_type_counts'].get('Organization', 0),
        semantics['entity_type_counts'].get('Location', 0),
        # Event type presence
        semantics['event_type_counts'].get('Hiring', 0),
        semantics['event_type_counts'].get('Discrimination', 0),
        semantics['event_type_counts'].get('Injury', 0),
        semantics['event_type_counts'].get('Termination', 0),
    ]

    combined = np.concatenate([structural, semantic_features])
    return combined.astype(np.float32)


# ============================================================================
# 3. CLUSTERING (same as before)
# ============================================================================

def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Normalize to unit length."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / (norms + 1e-8)


def simple_kmeans(embeddings: np.ndarray, n_clusters: int = 8, max_iters: int = 50,
                  random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """K-means clustering."""
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


def compute_davies_bouldin_score(embeddings: np.ndarray, labels: np.ndarray,
                                 centroids: np.ndarray) -> float:
    """Compute Davies-Bouldin index."""
    n_clusters = len(np.unique(labels))

    within_distances = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_mask = labels == i
        if cluster_mask.sum() > 0:
            within_distances[i] = np.mean(np.linalg.norm(embeddings[cluster_mask] - centroids[i], axis=1))

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


def analyze_cluster_semantics(doc_ids: List[str], labels: np.ndarray,
                             docs: Dict) -> Dict[int, Dict]:
    """Analyze semantic content of each cluster."""
    clusters = defaultdict(list)

    for doc_id, label in zip(doc_ids, labels):
        clusters[int(label)].append(doc_id)

    cluster_analysis = {}
    for cluster_id, cluster_docs in clusters.items():
        # Aggregate semantics for cluster
        all_event_types = Counter()
        all_entity_types = Counter()
        has_plaintiff_cluster = False
        has_defendant_cluster = False

        for doc_id in cluster_docs:
            semantics = docs[doc_id]['semantics']
            all_event_types.update(semantics['event_types'])
            all_entity_types.update(semantics['entity_types'])
            if semantics['has_plaintiff']:
                has_plaintiff_cluster = True
            if semantics['has_defendant']:
                has_defendant_cluster = True

        top_events = dict(all_event_types.most_common(3))
        top_entities = dict(all_entity_types.most_common(3))

        # Infer cluster type
        cluster_type = infer_cluster_type(top_events, has_plaintiff_cluster, has_defendant_cluster)

        cluster_analysis[cluster_id] = {
            'size': len(cluster_docs),
            'doc_ids': cluster_docs[:3],  # Sample 3
            'top_event_types': top_events,
            'top_entity_types': top_entities,
            'has_plaintiff': has_plaintiff_cluster,
            'has_defendant': has_defendant_cluster,
            'inferred_type': cluster_type,
        }

    return cluster_analysis


def infer_cluster_type(top_events: Dict, has_plaintiff: bool, has_defendant: bool) -> str:
    """Infer the legal scenario type of a cluster based on events and parties."""
    if not top_events:
        return "Unknown"

    top_event = list(top_events.keys())[0]

    # Heuristics to infer case type
    employment_events = {'Hiring', 'Termination', 'Discrimination', 'Harassment', 'LeaveOfAbsence', 'Promotion'}
    injury_events = {'Injury', 'Diagnosis', 'Medical'}
    contract_events = {'Contract', 'Breach', 'Payment', 'Termination'}

    if top_event in employment_events and has_plaintiff and has_defendant:
        return "Employment/Discrimination Case"
    elif top_event in injury_events:
        return "Personal Injury/Medical Case"
    elif top_event in contract_events:
        return "Contract Dispute"
    elif has_plaintiff and has_defendant:
        return "Litigation Case"
    else:
        return f"Case with {top_event}"


# ============================================================================
# 4. MAIN COMPARISON
# ============================================================================

def main():
    ekg_path = Path('/sessions/kind-fervent-noether/mnt/agentic-legal-event-extraction-swetha/data/outputs/qwen122b_hybrid_doc_graphs_final.jsonl')
    output_dir = Path('/sessions/kind-fervent-noether/mnt/agentic-legal-event-extraction-swetha')

    np.random.seed(42)
    random.seed(42)

    print("=" * 80)
    print("ENHANCED CLUSTERING: STRUCTURAL vs SEMANTIC+STRUCTURAL")
    print("=" * 80)

    # Load documents
    print("\n[1/7] Loading documents with semantic features...")
    docs = load_documents_with_semantics(str(ekg_path))
    doc_ids = list(docs.keys())
    n_clusters = min(8, max(3, len(doc_ids) // 15))

    print(f"  Total documents: {len(doc_ids)}")
    print(f"  Clusters: {n_clusters}")

    # ====== STRUCTURAL-ONLY CLUSTERING ======
    print("\n[2/7] Structural-only clustering (5 features)...")
    structural_embeddings = np.array([create_structural_features(docs[doc_id]['merged_graph'])
                                     for doc_id in doc_ids])
    structural_embeddings = normalize_embeddings(structural_embeddings)

    print(f"  Features: {structural_embeddings.shape[1]}")
    struct_labels, struct_centroids = simple_kmeans(structural_embeddings, n_clusters=n_clusters)
    struct_silhouette = compute_silhouette_score(structural_embeddings, struct_labels)
    struct_db = compute_davies_bouldin_score(structural_embeddings, struct_labels, struct_centroids)

    print(f"  Silhouette: {struct_silhouette:.4f}")
    print(f"  Davies-Bouldin: {struct_db:.4f}")

    struct_cluster_analysis = analyze_cluster_semantics(doc_ids, struct_labels, docs)

    # ====== SEMANTIC+STRUCTURAL CLUSTERING ======
    print("\n[3/7] Semantic+Structural clustering (15+ features)...")
    combined_embeddings = np.array([create_combined_feature_vector(docs[doc_id])
                                   for doc_id in doc_ids])
    combined_embeddings = normalize_embeddings(combined_embeddings)

    print(f"  Features: {combined_embeddings.shape[1]}")
    semantic_labels, semantic_centroids = simple_kmeans(combined_embeddings, n_clusters=n_clusters)
    semantic_silhouette = compute_silhouette_score(combined_embeddings, semantic_labels)
    semantic_db = compute_davies_bouldin_score(combined_embeddings, semantic_labels, semantic_centroids)

    print(f"  Silhouette: {semantic_silhouette:.4f}")
    print(f"  Davies-Bouldin: {semantic_db:.4f}")

    semantic_cluster_analysis = analyze_cluster_semantics(doc_ids, semantic_labels, docs)

    # ====== COMPARISON ======
    print("\n[4/7] Computing improvements...")
    silhouette_improvement = ((semantic_silhouette - struct_silhouette) / abs(struct_silhouette) * 100) if struct_silhouette != 0 else 0
    db_improvement = ((struct_db - semantic_db) / struct_db * 100)

    print(f"  Silhouette improvement: {silhouette_improvement:+.1f}%")
    print(f"  Davies-Bouldin improvement: {db_improvement:+.1f}%")

    # ====== SAVE RESULTS ======
    print("\n[5/7] Saving detailed results...")
    results = {
        'metadata': {
            'timestamp': str(np.datetime64('today')),
            'num_documents': len(doc_ids),
            'num_clusters': n_clusters,
        },
        'structural_only': {
            'method': 'Structural features only (5 features)',
            'features': ['num_entities', 'num_events', 'temporal_edges', 'causal_edges', 'graph_density'],
            'embedding_dims': structural_embeddings.shape[1],
            'silhouette_score': float(struct_silhouette),
            'davies_bouldin_index': float(struct_db),
            'cluster_analysis': {str(k): v for k, v in struct_cluster_analysis.items()},
        },
        'semantic_enhanced': {
            'method': 'Structural + Semantic features (15+ features)',
            'features': ['num_entities', 'num_events', 'temporal_edges', 'causal_edges', 'graph_density',
                        'num_entity_types', 'num_event_types', 'num_roles', 'has_plaintiff', 'has_defendant',
                        'has_person', 'has_organization', 'person_count', 'organization_count', 'location_count',
                        'hiring_count', 'discrimination_count', 'injury_count', 'termination_count'],
            'embedding_dims': combined_embeddings.shape[1],
            'silhouette_score': float(semantic_silhouette),
            'davies_bouldin_index': float(semantic_db),
            'cluster_analysis': {str(k): v for k, v in semantic_cluster_analysis.items()},
        },
        'comparison': {
            'silhouette_improvement_percent': float(silhouette_improvement),
            'davies_bouldin_improvement_percent': float(db_improvement),
            'winner': 'Semantic+Structural' if semantic_silhouette > struct_silhouette else 'Structural',
        }
    }

    with open(output_dir / 'enhanced_clustering_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # ====== PRINT SUMMARY ======
    print("\n" + "=" * 80)
    print("ENHANCED CLUSTERING SUMMARY")
    print("=" * 80)

    print("\n--- STRUCTURAL-ONLY CLUSTERING (5 features) ---")
    print(f"Silhouette Score: {struct_silhouette:.4f}")
    print(f"Davies-Bouldin Index: {struct_db:.4f}")
    print("\nCluster Types Identified:")
    for cid, analysis in sorted(struct_cluster_analysis.items()):
        print(f"  Cluster {cid}: {analysis['inferred_type']} ({analysis['size']} docs)")
        print(f"    Top events: {analysis['top_event_types']}")

    print("\n--- SEMANTIC+STRUCTURAL CLUSTERING (15+ features) ---")
    print(f"Silhouette Score: {semantic_silhouette:.4f}")
    print(f"Davies-Bouldin Index: {semantic_db:.4f}")
    print("\nCluster Types Identified:")
    for cid, analysis in sorted(semantic_cluster_analysis.items()):
        print(f"  Cluster {cid}: {analysis['inferred_type']} ({analysis['size']} docs)")
        print(f"    Top events: {analysis['top_event_types']}")

    print("\n--- IMPROVEMENTS ---")
    print(f"Silhouette: {silhouette_improvement:+.1f}%")
    print(f"Davies-Bouldin: {db_improvement:+.1f}%")

    print(f"\n✓ Results saved: {output_dir / 'enhanced_clustering_results.json'}")
    print("=" * 80)

    return results


if __name__ == '__main__':
    main()
