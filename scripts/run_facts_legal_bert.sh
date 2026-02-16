#!/usr/bin/env bash
# Convenience wrapper: run facts extraction with LegalBERT agent.
# Usage: ./scripts/run_facts_legal_bert.sh --checkpoint data/models/legal_bert_facts [--limit N] [--first-contiguous-only]
# Same as: ./scripts/run_facts_agent.sh --agent bert --checkpoint data/models/legal_bert_facts ...
set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ ! -f .venv/bin/python ]; then
    echo "Run setup first: ./scripts/setup.sh"
    exit 1
fi
source .venv/bin/activate
pip install -q -r requirements.txt

python src/run_facts_agent.py --agent bert "$@"
