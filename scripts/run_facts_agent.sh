#!/usr/bin/env bash
# Run LLM facts extraction. Uses .venv, config/config.yaml, .env (AGENT_API_KEY).
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
