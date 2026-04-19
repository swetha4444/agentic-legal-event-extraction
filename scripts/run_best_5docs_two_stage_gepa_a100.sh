#!/usr/bin/env bash
set -euo pipefail

RUN_STAMP=$(date +%Y%m%d_%H%M%S)
PROJECT=/work/pi_dagarwal_umass_edu/project_1/agentic-legal-event-extraction
export CANDIDATE_JSON=$PROJECT/data/outputs/rebuilt_two_stage_gepa_event_graph/opt_claude_gepa_20260330_174837/chunk_event_two_stage_gepa_rebuilt_opt_claude_20260330_174837_best_candidate.json
export OUTDIR=$PROJECT/data/outputs/rebuilt_two_stage_gepa_event_graph/eval_best_5docs_${RUN_STAMP}
export RUN_NAME=two_stage_best_5docs_${RUN_STAMP}
export MAX_DOCS=5
export MAX_TOTAL_CHUNKS=9999
export JUDGE_MODEL=claude-sonnet-4-5
export JUDGE_CHUNK_LIMIT=9999

PRED=$OUTDIR/${RUN_NAME}_predictions.jsonl
RENDER_OUT=$PROJECT/data/outputs/rebuilt_two_stage_gepa_event_graph_render/eval_best_5docs_${RUN_STAMP}

bash $PROJECT/scripts/run_chunk_event_two_stage_gepa_rebuilt_eval.sh

python $PROJECT/src/rebuilt_gepa_event_graphs/visualize_before_after.py \
  --before $PRED \
  --after $PRED \
  --output-dir $RENDER_OUT

echo "Output folder: $OUTDIR"
echo "Render output: $RENDER_OUT/index.html"
