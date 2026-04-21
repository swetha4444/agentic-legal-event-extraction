#!/usr/bin/env bash
# Submit with:
#   sbatch scripts/run_facts_hybrid_4courts_sbatch.sh

#SBATCH --job-name=facts_hybrid_4courts
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/work/pi_dagarwal_umass_edu/project_1/agentic-legal-event-extraction/scripts/train/output/slurm_facts_hybrid_4courts_%j.out
#SBATCH --error=/work/pi_dagarwal_umass_edu/project_1/agentic-legal-event-extraction/scripts/train/output/slurm_facts_hybrid_4courts_%j.err
#SBATCH --qos=short
#SBATCH --partition=gpu-preempt

set -euo pipefail

PROJECT_ROOT="/work/pi_dagarwal_umass_edu/project_1/agentic-legal-event-extraction"
CHECKPOINT="$PROJECT_ROOT/data/models/lexlm_facts_us_courtlistener"
INPUT_BASE="$PROJECT_ROOT/data/outputs/facts/input_by_court_facts_only"
OUTPUT_BASE="$PROJECT_ROOT/data/outputs/facts/by_court"
LLM_MODEL="${LLM_MODEL:-gpt4o}"
TOLERANCE="${TOLERANCE:-0.80}"
LOG_EVERY="${LOG_EVERY:-1}"

COURTS=(nysd pawd rid kyed)

cd "$PROJECT_ROOT"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export PYTHONPATH=src

echo "Starting hybrid facts run for courts: ${COURTS[*]}"
echo "Checkpoint: $CHECKPOINT"
echo "LLM model: $LLM_MODEL"

for court in "${COURTS[@]}"; do
  input_json="$INPUT_BASE/${court}.jsonl"
  out_dir="$OUTPUT_BASE/${court}/hybrid"
  mkdir -p "$out_dir"

  if [[ ! -f "$input_json" ]]; then
    echo "[skip] missing input: $input_json"
    continue
  fi

  echo "============================================================"
  echo "[run] court=$court input=$input_json output=$out_dir"
  PYTHONPATH=src .venv/bin/python scripts/eval/predict_fact_modes_on_json.py \
    --mode hybrid \
    --checkpoint "$CHECKPOINT" \
    --input-json "$input_json" \
    --output-dir "$out_dir" \
    --llm-model "$LLM_MODEL" \
    --tolerance "$TOLERANCE" \
    --run-name "${court}_hybrid_facts_only" \
    --log-every "$LOG_EVERY" \
    2>&1 | tee "$out_dir/run.log"
done

echo "Done hybrid facts run."
