#!/usr/bin/env bash
# Train LegalBERT on full MARRO. From project root: ./scripts/train/run_legal_bert_train.sh
set -e
cd "$(dirname "$0")/../.."
[[ -z "$VIRTUAL_ENV" ]] && { echo "Activate venv: source .venv/bin/activate"; exit 1; }
echo "Running LegalBERT training (MARRO)..."
PYTHONPATH=src .venv/bin/python scripts/train/train_legal_bert_facts.py \
  --data data/labeled/marro_fact_nonfact.jsonl \
  --output_dir data/models/legal_bert_facts_marro \
  --epochs 2 --batch_size 16 --val_ratio 0.05
echo "Done. Checkpoint: data/models/legal_bert_facts_marro"
