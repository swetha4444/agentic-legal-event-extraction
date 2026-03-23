#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/work/pi_dagarwal_umass_edu/project_1/agentic-legal-event-extraction"
MODEL="${MODEL:-gpt-5-mini}"
MAX_CHUNKS_TOTAL="${MAX_CHUNKS_TOTAL:-}"
MAX_CHUNKS_PER_DOC="${MAX_CHUNKS_PER_DOC:-}"
MAX_DOCS="${MAX_DOCS:-}"
GRAPH_VIEW="${GRAPH_VIEW:-event-only}"
HTML_LAYOUT="${HTML_LAYOUT:-draggable}"
DOC_LEVEL="${DOC_LEVEL:-1}"
INCLUDE_RAW_LLM="${INCLUDE_RAW_LLM:-1}"
MAX_EVENTS_PER_CHUNK="${MAX_EVENTS_PER_CHUNK:-8}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-7000}"
MAX_PARSE_RETRIES="${MAX_PARSE_RETRIES:-1}"
MAX_EMPTY_RETRIES="${MAX_EMPTY_RETRIES:-1}"

INPUTS=(
  "${INPUT_JSON:-$PROJECT_ROOT/data/outputs/chunks_only_1-5.jsonl}"
  "$PROJECT_ROOT/data/outputs/chunks_only_6-10.jsonl"
  "$PROJECT_ROOT/data/outputs/chunks_only_11-15.jsonl"
)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input) INPUT_JSON="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --max-chunks-total) MAX_CHUNKS_TOTAL="$2"; shift 2 ;;
    --max-chunks-per-doc) MAX_CHUNKS_PER_DOC="$2"; shift 2 ;;
    --max-docs) MAX_DOCS="$2"; shift 2 ;;
    --graph-view) GRAPH_VIEW="$2"; shift 2 ;;
    --html-layout) HTML_LAYOUT="$2"; shift 2 ;;
    --doc-level) DOC_LEVEL=1; shift ;;
    --per-chunk) DOC_LEVEL=0; shift ;;
    --include-raw-llm) INCLUDE_RAW_LLM=1; shift ;;
    --no-include-raw-llm) INCLUDE_RAW_LLM=0; shift ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

cd "$PROJECT_ROOT"
if [[ -f "$PROJECT_ROOT/.venv/bin/activate" ]]; then
  source "$PROJECT_ROOT/.venv/bin/activate"
fi
export PYTHONPATH="$PROJECT_ROOT/src"

for input_json in "${INPUTS[@]}"; do
  if [[ ! -f "$input_json" ]]; then
    echo "Missing input: $input_json" >&2
    exit 1
  fi

  base_name="$(basename "$input_json")"
  range_suffix="${base_name#chunks_only_}"
  range_suffix="${range_suffix%.jsonl}"

  output_json="$PROJECT_ROOT/data/outputs/chunk_event_graphs_rebuilt_${range_suffix}.jsonl"
  html_outdir="$PROJECT_ROOT/data/outputs/doc_graph_viz_rebuilt_${range_suffix}"

  if [[ -e "$output_json" ]]; then
    echo "Refusing to overwrite existing output file: $output_json" >&2
    exit 1
  fi
  if [[ -e "$html_outdir" ]]; then
    echo "Refusing to overwrite existing html output dir: $html_outdir" >&2
    exit 1
  fi

  extract_args=(
    --input "$input_json"
    --output "$output_json"
    --model "$MODEL"
    --max-events-per-chunk "$MAX_EVENTS_PER_CHUNK"
    --max-completion-tokens "$MAX_COMPLETION_TOKENS"
    --max-parse-retries "$MAX_PARSE_RETRIES"
    --max-empty-retries "$MAX_EMPTY_RETRIES"
  )

  if [[ -n "$MAX_DOCS" ]]; then
    extract_args+=(--max-docs "$MAX_DOCS")
  fi
  if [[ -n "$MAX_CHUNKS_TOTAL" ]]; then
    extract_args+=(--max-chunks-total "$MAX_CHUNKS_TOTAL")
  fi
  if [[ -n "$MAX_CHUNKS_PER_DOC" ]]; then
    extract_args+=(--max-chunks-per-doc "$MAX_CHUNKS_PER_DOC")
  fi
  if [[ "$INCLUDE_RAW_LLM" == "1" ]]; then
    extract_args+=(--include-raw-llm)
  fi

  python "$PROJECT_ROOT/src/rebuilt_event_graphs/extract_chunk_event_graphs_rebuilt.py" "${extract_args[@]}"

  viz_args=(
    --input "$output_json"
    --output-dir "$html_outdir"
    --graph-view "$GRAPH_VIEW"
    --html-layout "$HTML_LAYOUT"
  )
  if [[ -n "$MAX_DOCS" ]]; then
    viz_args+=(--max-docs "$MAX_DOCS")
  fi
  if [[ "$DOC_LEVEL" == "1" ]]; then
    viz_args+=(--doc-level)
  fi

  python "$PROJECT_ROOT/src/rebuilt_event_graphs/visualize_chunk_event_graphs_rebuilt.py" "${viz_args[@]}"

  echo "Finished $input_json"
  echo "  JSONL: $output_json"
  echo "  HTML:  $html_outdir"
done
