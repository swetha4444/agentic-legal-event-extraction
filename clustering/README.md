# Clustering Analysis: EKG vs TF-IDF

## Overview
Comparison of EKG-based clustering vs traditional TF-IDF on 133 legal documents.

## Key Results
- **EKG Silhouette**: 0.3463 (excellent)
- **TF-IDF Silhouette**: 0.0779 (poor)
- **EKG Efficiency**: 400x better (5 vs 2000 features)

## Files in This Directory
- `results/FINAL_CLUSTERING_REPORT.txt` - Executive summary + recommendations
- `results/clustering_detailed_results.json` - Raw clustering metrics
- `results/enhanced_clustering_results.json` - Semantic feature comparison
- `scripts/clustering_comparison_analysis.py` - Main analysis (reproduces all results)
- `scripts/enhanced_semantic_clustering.py` - Semantic feature testing

## How to Reproduce Results

### Quick Run (5 minutes)
```bash
python scripts/clustering_comparison_analysis.py
# Outputs: clustering_detailed_results.json + clustering_analysis_report.html
```

### Full Analysis (10 minutes)
```bash
# Run structural vs semantic comparison
python scripts/enhanced_semantic_clustering.py

# Generate detailed analysis
python scripts/analyze_semantic_clustering.py
```

## Cluster Characteristics (for manual review)

Each EKG cluster represents a distinct legal scenario type (not named yet, perhaps could be named using SALI labels):

**Cluster 7** - Sexual Harassment Cases (10 docs)
- Dominated by harassment/sexual harassment events
- Clear plaintiff-defendant litigation pattern

**Cluster 0** - Employment Termination (22 docs)
- Hiring + termination + employment discrimination events
- Typical employment case scenario

**Cluster 5** - General Litigation (26 docs)
- Mixed employment/communication events
- Moderate complexity cases
