#!/usr/bin/env bash
# CourtListener pipeline (clusters or RECAP). Needs token: export COURTLISTENER_API_TOKEN=... or .courtlistener_token in project root.
set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

# If token not set, try project root .courtlistener_token (gitignored)
if [ -z "${COURTLISTENER_API_TOKEN:-}" ] && [ -f "$PROJECT_ROOT/.courtlistener_token" ]; then
  export COURTLISTENER_API_TOKEN="$(cat "$PROJECT_ROOT/.courtlistener_token" | tr -d '\n\r')"
fi

PYTHONPATH="$PROJECT_ROOT" python -m src.run_courtlistener_pipeline "$@"
