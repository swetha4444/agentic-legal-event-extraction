#!/usr/bin/env bash
# Run facts extraction. Swap agent with --agent llm (default) or --agent bert.
# LLM: uses config/config.yaml and .env (AGENT_API_KEY). Optional: --model gpt4o
# BERT: requires --checkpoint <path>. Optional: --first-contiguous-only
set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ ! -f .venv/bin/python ]; then
    echo "Run setup first: ./scripts/setup.sh"
    exit 1
fi
source .venv/bin/activate
pip install -q -r requirements.txt

python src/run_facts_agent.py "$@"
