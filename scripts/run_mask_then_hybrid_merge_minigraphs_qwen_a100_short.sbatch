#!/usr/bin/env bash
#SBATCH --job-name=mask_hybrid_qwen
#SBATCH --partition=gpu-preempt
#SBATCH --qos=short
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

PROJECT_ROOT="/work/pi_dagarwal_umass_edu/project_1/agentic-legal-event-extraction"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

# Input JSONL with doc.chunks[].graph or doc.chunks[].mini_graph
INPUT_JSON="${INPUT_JSON:-$PROJECT_ROOT/data/outputs/rebuilt_two_stage_gepa_event_graph/eval_force_sentence_events_46_24chunks_20260407_044915/chunk_event_two_stage_force_sentence_events_46_24chunks_20260407_044915_predictions.jsonl}"

# Output JSONL (timestamped to avoid overwrite)
OUTDIR="${OUTDIR:-$PROJECT_ROOT/data/outputs/hybrid_merge_local_qwen}"
mkdir -p "$OUTDIR"
OUTPUT_BASENAME="${OUTPUT_BASENAME:-mask_then_hybrid_merged_${RUN_STAMP}.jsonl}"
OUTPUT_JSON="$OUTDIR/$OUTPUT_BASENAME"

# Local Qwen model path
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-/datasets/ai/qwen3/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}"
QWEN_DEVICE="${QWEN_DEVICE:-cuda}"

# Runtime knobs. These defaults are intentionally conservative so at least
# one document can finish and flush to disk before walltime.
MAX_DOCS="${MAX_DOCS:-1}"
QWEN_MAX_NEW_TOKENS="${QWEN_MAX_NEW_TOKENS:-384}"
MASK_MAX_COMPLETION_TOKENS="${MASK_MAX_COMPLETION_TOKENS:-768}"
LINK_MAX_COMPLETION_TOKENS="${LINK_MAX_COMPLETION_TOKENS:-192}"
TOP_K="${TOP_K:-2}"
MIN_SIMILARITY="${MIN_SIMILARITY:-0.35}"
MAX_PAIRS="${MAX_PAIRS:-12}"
DRY_RUN_MASK="${DRY_RUN_MASK:-0}"

cd "$PROJECT_ROOT"
if [[ -f "$PROJECT_ROOT/.venv/bin/activate" ]]; then
  source "$PROJECT_ROOT/.venv/bin/activate"
fi
export PYTHONPATH="$PROJECT_ROOT/src"

CMD=(
  python "$PROJECT_ROOT/src/run_mask_then_hybrid_merge_minigraphs.py"
  --input "$INPUT_JSON"
  --output "$OUTPUT_JSON"
  --qwen-model-path "$QWEN_MODEL_PATH"
  --qwen-device "$QWEN_DEVICE"
  --qwen-max-new-tokens "$QWEN_MAX_NEW_TOKENS"
  --mask-max-completion-tokens "$MASK_MAX_COMPLETION_TOKENS"
  --link-max-completion-tokens "$LINK_MAX_COMPLETION_TOKENS"
  --top-k "$TOP_K"
  --min-similarity "$MIN_SIMILARITY"
  --max-pairs "$MAX_PAIRS"
)

if [[ -n "${MAX_DOCS}" ]]; then
  CMD+=(--max-docs "$MAX_DOCS")
fi

if [[ "$DRY_RUN_MASK" == "1" ]]; then
  CMD+=(--dry-run-mask)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo "Done. Output JSONL: $OUTPUT_JSON"
