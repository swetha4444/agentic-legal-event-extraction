#!/usr/bin/env bash
# Submit with:
#   sbatch scripts/run_eval_holdout40_3modes_sbatch.sh

#SBATCH --job-name=us_holdout40_eval_3modes
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/work/pi_dagarwal_umass_edu/project_1/agentic-legal-event-extraction/scripts/train/output/slurm_eval_holdout40_%j.out
#SBATCH --error=/work/pi_dagarwal_umass_edu/project_1/agentic-legal-event-extraction/scripts/train/output/slurm_eval_holdout40_%j.err
#SBATCH --qos=short

# Optional cluster-specific setting:
#SBATCH --partition=gpu-preempt

set -euo pipefail

PROJECT_ROOT="/work/pi_dagarwal_umass_edu/project_1/agentic-legal-event-extraction"
INPUT_JSON="/work/pi_dagarwal_umass_edu/project_1/sriram/outputs/processed_json/courtlistener_fact_jurisdiction_compiled_holdout40_for_eval.json"
DATASET_DIR="/work/pi_dagarwal_umass_edu/project_1/sriram/outputs/processed_json"
CHECKPOINT="/work/pi_dagarwal_umass_edu/project_1/agentic-legal-event-extraction/data/models/02-24-2:04PM_legalbert_train60"
OUTDIR="/work/pi_dagarwal_umass_edu/project_1/sriram/outputs/fact_extraction/US Dataset/holdout40_eval_train60_legalbert"

# Runtime knobs
CLASSIFIER_BATCH_SIZE=128
TOLERANCE=0.80
LLM_CHUNK_SIZE=120

cd "$PROJECT_ROOT"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

mkdir -p "$OUTDIR"
export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=false

STAMP="$(date +%Y%m%d_%H%M%S)"

echo "[$(date)] Starting classifier mode..."
python scripts/eval/eval_fact_modes_on_marro_json.py \
  --input-json "$INPUT_JSON" \
  --dataset-dir "$DATASET_DIR" \
  --mode classifier \
  --checkpoint "$CHECKPOINT" \
  --classifier-batch-size "$CLASSIFIER_BATCH_SIZE" \
  --output-dir "$OUTDIR" \
  --run-name "us_holdout40_classifier_legalbert_train60_${STAMP}"

echo "[$(date)] Starting llm mode..."
python scripts/eval/eval_fact_modes_on_marro_json.py \
  --input-json "$INPUT_JSON" \
  --dataset-dir "$DATASET_DIR" \
  --mode llm \
  --llm-chunk-size "$LLM_CHUNK_SIZE" \
  --output-dir "$OUTDIR" \
  --run-name "us_holdout40_llm_train60_${STAMP}"

echo "[$(date)] Starting hybrid mode..."
python scripts/eval/eval_fact_modes_on_marro_json.py \
  --input-json "$INPUT_JSON" \
  --dataset-dir "$DATASET_DIR" \
  --mode hybrid \
  --checkpoint "$CHECKPOINT" \
  --classifier-batch-size "$CLASSIFIER_BATCH_SIZE" \
  --tolerance "$TOLERANCE" \
  --llm-chunk-size "$LLM_CHUNK_SIZE" \
  --output-dir "$OUTDIR" \
  --run-name "us_holdout40_hybrid_legalbert_train60_${STAMP}"

echo "[$(date)] All 3 modes finished."
echo "Metrics files:"
ls -1 "$OUTDIR"/*"${STAMP}"*_metrics.json

