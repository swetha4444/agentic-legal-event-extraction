#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/work/pi_dagarwal_umass_edu/project_1/agentic-legal-event-extraction"
INPUT_JSON="${INPUT_JSON:-$PROJECT_ROOT/data/outputs/chunks_only_1-5.jsonl}"
SEED_CANDIDATE_JSON="${SEED_CANDIDATE_JSON:-$PROJECT_ROOT/src/rebuilt_gepa_event_graphs/seed_candidate.json}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTDIR="${OUTDIR:-$PROJECT_ROOT/data/outputs/rebuilt_gepa_event_graph/opt_$RUN_STAMP}"
RUN_NAME="${RUN_NAME:-chunk_event_gepa_rebuilt_opt_$RUN_STAMP}"
MAX_EVALS="${MAX_EVALS:-12}"
MAX_DOCS="${MAX_DOCS:-1}"
MAX_TOTAL_CHUNKS="${MAX_TOTAL_CHUNKS:-10}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5-mini}"
JUDGE_CHUNK_LIMIT="${JUDGE_CHUNK_LIMIT:-5}"
REFLECTION_MODEL="${REFLECTION_MODEL:-gpt-5-mini}"
MUTATIONS_PER_GENERATION="${MUTATIONS_PER_GENERATION:-2}"
MIN_BEST_SCHEMA_VALIDITY="${MIN_BEST_SCHEMA_VALIDITY:-0.8}"
MIN_BEST_NON_EMPTY_RATE="${MIN_BEST_NON_EMPTY_RATE:-0.8}"
MIN_BEST_JUDGE_SCORE_MEAN="${MIN_BEST_JUDGE_SCORE_MEAN:-0.5}"

if [[ -e "$OUTDIR" ]]; then
  echo "Refusing to overwrite existing output dir: $OUTDIR" >&2
  exit 1
fi
mkdir -p "$OUTDIR"

cd "$PROJECT_ROOT"
if [[ -f "$PROJECT_ROOT/.venv/bin/activate" ]]; then
  source "$PROJECT_ROOT/.venv/bin/activate"
fi
export PYTHONPATH="$PROJECT_ROOT/src"

python "$PROJECT_ROOT/src/rebuilt_gepa_event_graphs/optimize_rebuilt.py" \
  --input "$INPUT_JSON" \
  --seed-candidate-json "$SEED_CANDIDATE_JSON" \
  --output-dir "$OUTDIR" \
  --run-name "$RUN_NAME" \
  --max-evals "$MAX_EVALS" \
  --max-docs "$MAX_DOCS" \
  --max-total-chunks "$MAX_TOTAL_CHUNKS" \
  --judge-model "$JUDGE_MODEL" \
  --judge-chunk-limit "$JUDGE_CHUNK_LIMIT" \
  --reflection-model "$REFLECTION_MODEL" \
  --mutations-per-generation "$MUTATIONS_PER_GENERATION" \
  --min-best-schema-validity "$MIN_BEST_SCHEMA_VALIDITY" \
  --min-best-non-empty-rate "$MIN_BEST_NON_EMPTY_RATE" \
  --min-best-judge-score-mean "$MIN_BEST_JUDGE_SCORE_MEAN"
