#!/usr/bin/env bash
# Train LexLM on hold-out MARRO (uksc-2011-0183 excluded). From project root: ./scripts/train/run_lexlm_train_holdout.sh
set -e
cd "$(dirname "$0")/../.."
[[ -z "$VIRTUAL_ENV" ]] && { echo "Activate venv: source .venv/bin/activate"; exit 1; }
[[ ! -f data/labeled/marro_fact_nonfact_holdout.jsonl ]] && { echo "Create hold-out data first (see scripts/README): --exclude uksc-2011-0183.txt --output data/labeled/marro_fact_nonfact_holdout.jsonl"; exit 1; }
echo "Running LexLM training (hold-out: uksc-2011-0183.txt)..."
PYTHONPATH=src .venv/bin/python scripts/train/train_legal_bert_facts.py \
  --data data/labeled/marro_fact_nonfact_holdout.jsonl \
  --output_dir data/models/lexlm_facts_marro_holdout \
  --model_name lexlms/legal-roberta-base \
  --epochs 2 --batch_size 16 --val_ratio 0.05
echo "Done. Eval: PYTHONPATH=src .venv/bin/python scripts/eval/eval_legal_bert_on_marro_dir.py --checkpoint data/models/lexlm_facts_marro_holdout --docs data/MARRO/UK-dataset/uksc-2011-0183.txt"
