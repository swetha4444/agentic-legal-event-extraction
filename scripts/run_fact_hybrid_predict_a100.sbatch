#!/usr/bin/env bash
# Submit with:
#   sbatch scripts/run_fact_hybrid_predict_a100.sbatch
# Override model if needed:
#   MODEL=gpt4o sbatch scripts/run_fact_hybrid_predict_a100.sbatch
# Override input source if needed:
#   INPUT_JSONL=/path/to/one.jsonl sbatch scripts/run_fact_hybrid_predict_a100.sbatch

#SBATCH --job-name=fact_hybrid_predict
#SBATCH --partition=gpu-preempt
#SBATCH --qos=long
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

PROJECT_ROOT="/work/pi_dagarwal_umass_edu/project_1/agentic-legal-event-extraction"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

INPUT_DIR="${INPUT_DIR:-$PROJECT_ROOT/data/outputs/facts/input_by_court_facts_only}"
INPUT_JSONL="${INPUT_JSONL:-}"
OUTDIR="${OUTDIR:-$PROJECT_ROOT/data/outputs/fact_extraction_hybrid}"
CHECKPOINT="${CHECKPOINT:-$PROJECT_ROOT/data/models/02-24-2:04PM_legalbert_train60}"
MODE="${MODE:-hybrid}"
MODEL="${MODEL:-claude-sonnet-4-5}"
MODEL_TAG="${MODEL_TAG:-${MODEL//[^[:alnum:]]/_}}"
RUN_TAG="${RUN_TAG:-${MODEL_TAG}_${RUN_STAMP}}"
CLASSIFIER_BATCH_SIZE="${CLASSIFIER_BATCH_SIZE:-128}"
TOLERANCE="${TOLERANCE:-0.80}"
LLM_CHUNK_SIZE="${LLM_CHUNK_SIZE:-120}"
MAX_DOCS="${MAX_DOCS:-}"
MAX_SENTENCES_PER_DOC="${MAX_SENTENCES_PER_DOC:-}"
TEXT_FIELD="${TEXT_FIELD:-document_text}"
TITLE_FIELD="${TITLE_FIELD:-case_id}"

mkdir -p "$OUTDIR"

cd "$PROJECT_ROOT"
if [[ -f "$PROJECT_ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/.venv/bin/activate"
fi
export PYTHONPATH="$PROJECT_ROOT/src"
export TOKENIZERS_PARALLELISM=false

declare -a INPUT_FILES=()
if [[ -n "$INPUT_JSONL" ]]; then
  INPUT_FILES=("$INPUT_JSONL")
else
  while IFS= read -r path; do
    INPUT_FILES+=("$path")
  done < <(find "$INPUT_DIR" -maxdepth 1 -type f -name '*.jsonl' | sort)
fi

if [[ ${#INPUT_FILES[@]} -eq 0 ]]; then
  echo "No input JSONL files found."
  echo "INPUT_DIR=$INPUT_DIR"
  exit 1
fi

echo "Found ${#INPUT_FILES[@]} input file(s)."
echo "LLM model: $MODEL"
echo "Checkpoint: $CHECKPOINT"

for input_jsonl in "${INPUT_FILES[@]}"; do
  input_stem="$(basename "$input_jsonl" .jsonl)"
  run_name="${input_stem}_hybrid_${RUN_TAG}"
  file_outdir="$OUTDIR/$input_stem"
  compiled_json="$file_outdir/${run_name}_compiled.jsonl"

  mkdir -p "$file_outdir"

  CONVERT_CMD=(
    python "$PROJECT_ROOT/scripts/data/convert_raw_fact_input_to_compiled_json.py"
    --input "$input_jsonl"
    --output "$compiled_json"
    --text-field "$TEXT_FIELD"
    --title-field "$TITLE_FIELD"
  )

  PREDICT_CMD=(
    python "$PROJECT_ROOT/scripts/eval/predict_fact_modes_on_json.py"
    --input-json "$compiled_json"
    --mode "$MODE"
    --checkpoint "$CHECKPOINT"
    --llm-model "$MODEL"
    --classifier-batch-size "$CLASSIFIER_BATCH_SIZE"
    --tolerance "$TOLERANCE"
    --llm-chunk-size "$LLM_CHUNK_SIZE"
    --text-field document_text
    --title-field title
    --output-dir "$file_outdir"
    --run-name "$run_name"
  )

  if [[ -n "$MAX_DOCS" ]]; then
    PREDICT_CMD+=(--max-docs "$MAX_DOCS")
  fi

  if [[ -n "$MAX_SENTENCES_PER_DOC" ]]; then
    PREDICT_CMD+=(--max-sentences-per-doc "$MAX_SENTENCES_PER_DOC")
  fi

  echo "============================================================"
  echo "Input: $input_jsonl"
  echo "Run name: $run_name"
  echo "Output dir: $file_outdir"
  echo "Running converter: ${CONVERT_CMD[*]}"
  "${CONVERT_CMD[@]}"

  echo "Running predictor: ${PREDICT_CMD[*]}"
  "${PREDICT_CMD[@]}"

  echo "Completed: $input_stem"
  echo "Compiled JSON: $compiled_json"
  echo "Predictions JSONL: $file_outdir/${run_name}_predictions.jsonl"
  echo "Summary JSON: $file_outdir/${run_name}_summary.json"
done

echo "Done processing ${#INPUT_FILES[@]} input file(s)."
