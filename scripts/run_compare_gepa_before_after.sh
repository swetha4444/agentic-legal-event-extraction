#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${PROJECT_ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/.venv/bin/activate"
fi

BEFORE_JSON="${1:-${PROJECT_ROOT}/data/outputs/chunk_event_graphs_rebuilt_1-5.jsonl}"
AFTER_JSON="${2:-}"
OUT_DIR="${3:-${PROJECT_ROOT}/data/outputs/rebuilt_gepa_event_graph_compare/compare_$(date +%Y%m%d_%H%M%S)}"
GRAPH_VIEW="${GRAPH_VIEW:-full}"
HTML_LAYOUT="${HTML_LAYOUT:-draggable}"

if [[ -z "${AFTER_JSON}" ]]; then
  AFTER_JSON="$(ls -1t "${PROJECT_ROOT}"/data/outputs/rebuilt_gepa_event_graph/*/*_predictions.jsonl 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "${AFTER_JSON}" ]]; then
  echo "No after-GEPA predictions JSONL found. Pass it explicitly as arg 2." >&2
  exit 1
fi

if [[ -e "${OUT_DIR}" ]]; then
  echo "Refusing to overwrite existing output dir: ${OUT_DIR}" >&2
  exit 1
fi

python "${PROJECT_ROOT}/src/rebuilt_gepa_event_graphs/visualize_before_after.py" \
  --before "${BEFORE_JSON}" \
  --after "${AFTER_JSON}" \
  --output-dir "${OUT_DIR}" \
  --graph-view "${GRAPH_VIEW}" \
  --html-layout "${HTML_LAYOUT}"

echo "Wrote comparison report to: ${OUT_DIR}/index.html"
