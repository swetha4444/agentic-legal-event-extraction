#!/usr/bin/env bash
#SBATCH --job-name=hybrid_api_1doc
#SBATCH --partition=gpu-preempt
#SBATCH --qos=short
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

PROJECT_ROOT="/work/pi_dagarwal_umass_edu/project_1/agentic-legal-event-extraction"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

INPUT_JSON="${INPUT_JSON:-$PROJECT_ROOT/data/outputs/rebuilt_two_stage_gepa_event_graph/eval_best_5docs_20260331_162752/two_stage_best_5docs_20260331_162752_predictions.jsonl}"
OUTDIR="${OUTDIR:-$PROJECT_ROOT/data/outputs/hybrid_merge_api}"
mkdir -p "$OUTDIR"
OUTPUT_JSON="${OUTPUT_JSON:-$OUTDIR/hybrid_api_1doc_${RUN_STAMP}.jsonl}"

MODEL="${MODEL:-claude-sonnet-4-6}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-local_lexical_cosine}"
SEMANTIC_RESCUE_MODEL="${SEMANTIC_RESCUE_MODEL:-huggingface/BAAI/bge-small-en-v1.5}"
MAX_DOCS="${MAX_DOCS:-1}"
TOP_K="${TOP_K:-6}"
MIN_SIMILARITY="${MIN_SIMILARITY:-0.60}"
MAX_PAIRS="${MAX_PAIRS:-56}"
MAX_RELATION_PAIRS="${MAX_RELATION_PAIRS:-36}"
MAX_DUPLICATE_PAIRS="${MAX_DUPLICATE_PAIRS:-20}"
SEMANTIC_RESCUE_CANDIDATES="${SEMANTIC_RESCUE_CANDIDATES:-28}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-500}"

cd "$PROJECT_ROOT"
if [[ -f "$PROJECT_ROOT/.venv/bin/activate" ]]; then
  source "$PROJECT_ROOT/.venv/bin/activate"
fi
export PYTHONPATH="$PROJECT_ROOT/src"

CMD=(
  python "$PROJECT_ROOT/src/run_hybrid_merge_minigraphs.py"
  --input "$INPUT_JSON"
  --output "$OUTPUT_JSON"
  --embedding-model "$EMBEDDING_MODEL"
  --semantic-rescue-model "$SEMANTIC_RESCUE_MODEL"
  --max-docs "$MAX_DOCS"
  --top-k "$TOP_K"
  --min-similarity "$MIN_SIMILARITY"
  --max-pairs "$MAX_PAIRS"
  --max-relation-pairs "$MAX_RELATION_PAIRS"
  --max-duplicate-pairs "$MAX_DUPLICATE_PAIRS"
  --semantic-rescue-candidates "$SEMANTIC_RESCUE_CANDIDATES"
  --max-completion-tokens "$MAX_COMPLETION_TOKENS"
)

if [[ -n "$MODEL" ]]; then
  CMD+=(--model "$MODEL")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo "Done. Output JSONL: $OUTPUT_JSON"
