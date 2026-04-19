#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/work/pi_dagarwal_umass_edu/project_1/agentic-legal-event-extraction"
INPUT_JSON="${INPUT_JSON:-$PROJECT_ROOT/data/outputs/chunks_only_1-5.jsonl}"
CANDIDATE_JSON="${CANDIDATE_JSON:-$PROJECT_ROOT/src/rebuilt_two_stage_gepa_event_graphs/seed_candidate_two_stage_transport_safe_sonnet45_v1.json}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTDIR="${OUTDIR:-$PROJECT_ROOT/data/outputs/rebuilt_two_stage_gepa_event_graph/eval_$RUN_STAMP}"
RUN_NAME="${RUN_NAME:-chunk_event_two_stage_gepa_rebuilt_eval_$RUN_STAMP}"
MAX_DOCS="${MAX_DOCS:-1}"
MAX_TOTAL_CHUNKS="${MAX_TOTAL_CHUNKS:-5}"
JUDGE_MODEL="${JUDGE_MODEL:-claude-sonnet-4-5}"
JUDGE_CHUNK_LIMIT="${JUDGE_CHUNK_LIMIT:-5}"

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

python "$PROJECT_ROOT/src/rebuilt_two_stage_gepa_event_graphs/eval_candidate_rebuilt.py" \
  --input "$INPUT_JSON" \
  --candidate-json "$CANDIDATE_JSON" \
  --output-dir "$OUTDIR" \
  --run-name "$RUN_NAME" \
  --max-docs "$MAX_DOCS" \
  --max-total-chunks "$MAX_TOTAL_CHUNKS" \
  --judge-model "$JUDGE_MODEL" \
  --judge-chunk-limit "$JUDGE_CHUNK_LIMIT" \
  "$@"
