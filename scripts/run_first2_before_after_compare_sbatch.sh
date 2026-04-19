#!/usr/bin/env bash
#SBATCH --job-name=event_graph_first2_compare
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/work/pi_dagarwal_umass_edu/project_1/agentic-legal-event-extraction/scripts/slurm_event_graph_first2_compare_%j.out

set -euo pipefail

PROJECT_ROOT="/work/pi_dagarwal_umass_edu/project_1/agentic-legal-event-extraction"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"

INPUT_JSON="$PROJECT_ROOT/data/outputs/chunks_only_1-5.jsonl"

BEFORE_JSON="$PROJECT_ROOT/data/outputs/chunk_event_graphs_rebuilt_first2_${RUN_STAMP}.jsonl"
AFTER_DIR="$PROJECT_ROOT/data/outputs/rebuilt_gepa_event_graph/eval_transport_safe_sonnet45_judged_sonnet45_2chunks_${RUN_STAMP}"
AFTER_RUN="chunk_event_gepa_rebuilt_eval_transport_safe_sonnet45_judged_sonnet45_2chunks_${RUN_STAMP}"
AFTER_JSON="$AFTER_DIR/${AFTER_RUN}_predictions.jsonl"
COMPARE_DIR="$PROJECT_ROOT/data/outputs/rebuilt_gepa_event_graph_compare/compare_first2_${RUN_STAMP}"

if [[ -f "$PROJECT_ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/.venv/bin/activate"
fi

export PYTHONPATH="$PROJECT_ROOT/src"

python "$PROJECT_ROOT/src/rebuilt_event_graphs/extract_chunk_event_graphs_rebuilt.py" \
  --input "$INPUT_JSON" \
  --output "$BEFORE_JSON" \
  --model "claude-sonnet-4-5" \
  --max-chunks-total 2 \
  --max-events-per-chunk 6 \
  --max-completion-tokens 2400 \
  --max-parse-retries 2 \
  --max-empty-retries 2 \
  --include-raw-llm

python "$PROJECT_ROOT/src/rebuilt_gepa_event_graphs/eval_candidate_rebuilt.py" \
  --input "$INPUT_JSON" \
  --candidate-json "$PROJECT_ROOT/src/rebuilt_gepa_event_graphs/seed_candidate_transport_safe_sonnet45_v1.json" \
  --output-dir "$AFTER_DIR" \
  --run-name "$AFTER_RUN" \
  --max-docs 1 \
  --max-total-chunks 2 \
  --judge-model "claude-sonnet-4-5" \
  --judge-chunk-limit 2

python "$PROJECT_ROOT/src/rebuilt_gepa_event_graphs/visualize_before_after.py" \
  --before "$BEFORE_JSON" \
  --after "$AFTER_JSON" \
  --output-dir "$COMPARE_DIR" \
  --graph-view "event-only" \
  --html-layout "draggable"

echo "Before JSON: $BEFORE_JSON"
echo "After JSON:  $AFTER_JSON"
echo "Report:      $COMPARE_DIR/index.html"
