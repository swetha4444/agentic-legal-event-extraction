#!/usr/bin/env bash
# Setup: venv, deps, data dirs. Run from project root.
set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

mkdir -p data/raw data/processed data/outputs
echo "Setup done. Activate with: source .venv/bin/activate"
