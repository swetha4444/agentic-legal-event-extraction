#!/usr/bin/env bash
#SBATCH --job-name=two_stage_claude_eval
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --qos=short
#SBATCH --partition=gpu-preempt
#SBATCH --output=/work/pi_dagarwal_umass_edu/project_1/agentic-legal-event-extraction/scripts/slurm_two_stage_claude_eval_%j.out
#SBATCH --error=/work/pi_dagarwal_umass_edu/project_1/agentic-legal-event-extraction/scripts/slurm_two_stage_claude_eval_%j.err

set -euo pipefail

PROJECT_ROOT="/work/pi_dagarwal_umass_edu/project_1/agentic-legal-event-extraction"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

export INPUT_JSON="${INPUT_JSON:-$PROJECT_ROOT/data/outputs/chunks_only_1-5.jsonl}"
export CANDIDATE_JSON="${CANDIDATE_JSON:-$PROJECT_ROOT/src/rebuilt_two_stage_gepa_event_graphs/seed_candidate_two_stage_transport_safe_sonnet45_v1.json}"
export OUTDIR="${OUTDIR:-$PROJECT_ROOT/data/outputs/rebuilt_two_stage_gepa_event_graph/eval_claude_seed_${RUN_STAMP}}"
export RUN_NAME="${RUN_NAME:-chunk_event_two_stage_gepa_rebuilt_eval_claude_seed_${RUN_STAMP}}"
export MAX_DOCS="${MAX_DOCS:-1}"
export MAX_TOTAL_CHUNKS="${MAX_TOTAL_CHUNKS:-5}"
export JUDGE_MODEL="${JUDGE_MODEL:-claude-sonnet-4-5}"
export JUDGE_CHUNK_LIMIT="${JUDGE_CHUNK_LIMIT:-5}"

bash "$PROJECT_ROOT/scripts/run_chunk_event_two_stage_gepa_rebuilt_eval.sh"
