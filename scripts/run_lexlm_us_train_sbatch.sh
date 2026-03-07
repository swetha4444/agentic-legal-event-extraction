#!/usr/bin/env bash
# Submit with:
#   sbatch scripts/run_lexlm_us_train_sbatch.sh

#SBATCH --job-name=lexlm_us_train
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/work/pi_dagarwal_umass_edu/project_1/agentic-legal-event-extraction/scripts/train/output/slurm_lexlm_us_%j.out
#SBATCH --error=/work/pi_dagarwal_umass_edu/project_1/agentic-legal-event-extraction/scripts/train/output/slurm_lexlm_us_%j.err
#SBATCH --qos=short

# Optional cluster-specific settings:
#SBATCH --partition=gpu-preempt

set -euo pipefail

PROJECT_ROOT="/work/pi_dagarwal_umass_edu/project_1/agentic-legal-event-extraction"
DATA_JSONL="/work/pi_dagarwal_umass_edu/project_1/sriram/outputs/processed_json/courtlistener_fact_jurisdiction_compiled.json"
OUT_DIR="$PROJECT_ROOT/data/models/lexlm_facts_us_courtlistener"

cd "$PROJECT_ROOT"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export PYTHONPATH=src

python scripts/train/train_legal_bert_facts.py \
  --data "$DATA_JSONL" \
  --output_dir "$OUT_DIR" \
  --model_name lexlms/legal-roberta-base \
  --epochs 3 \
  --batch_size 16 \
  --lr 2e-5 \
  --val_ratio 0.2 \
  --max_length 256 \
  --seed 42
