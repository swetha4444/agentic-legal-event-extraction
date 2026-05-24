# LLM Evaluation of Clusters for Lawyer Review

## Strategy
Use LLM to:
1. Evaluate if documents in each cluster are semantically related
2. Generate cluster summaries for lawyers
3. Flag anomalies (documents that don't fit the cluster theme)

## Suggested Evaluation Script
```python
# For each cluster:
# 1. Sample 3-5 documents
# 2. Ask LLM: "Are these legal cases similar? Why?"
# 3. Ask LLM: "What legal scenario do these represent?"
# 4. Generate cluster summary for lawyer review
```

## Cluster Context for Lawyers
When you review, pay attention to:
- **Consistency**: Do documents in cluster share similar legal claims?
- **Coherence**: Do they describe similar factual scenarios?
- **Scenario Type**: Employment? Contract? Injury? Mixed?

## Expected Outcomes
- Clusters should show strong thematic coherence
- EKG clustering should create meaningful legal scenario groups
- Validates that structured event representation beats word frequency

See `results/clustering_detailed_results.json` for cluster compositions.