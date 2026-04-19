#!/usr/bin/env python3
"""CLI for optimizing rebuilt two-stage GEPA event-graph candidates."""
import argparse
import sys
from pathlib import Path

if __name__ == "__main__":
    _src = Path(__file__).resolve().parents[1]
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from rebuilt_two_stage_gepa_event_graphs.core import (
    DEFAULT_FRONTIER_PARENT_LIMIT,
    DEFAULT_MIN_JUDGE_SCORE_MEAN,
    DEFAULT_MIN_NON_EMPTY_RATE,
    DEFAULT_MIN_QUALITY_IMPROVEMENT,
    DEFAULT_MIN_SCHEMA_VALIDITY,
    DEFAULT_PLATEAU_PATIENCE,
    load_candidate,
    optimize_candidates,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize rebuilt two-stage GEPA event-graph candidates.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--seed-candidate-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--max-evals", type=int, default=12)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--max-total-chunks", type=int, default=None)
    parser.add_argument("--judge-model", default="claude-sonnet-4-5")
    parser.add_argument("--judge-chunk-limit", type=int, default=5)
    parser.add_argument("--reflection-model", default="claude-sonnet-4-5")
    parser.add_argument("--mutations-per-generation", type=int, default=2)
    parser.add_argument("--frontier-parent-limit", type=int, default=DEFAULT_FRONTIER_PARENT_LIMIT)
    parser.add_argument("--plateau-patience", type=int, default=DEFAULT_PLATEAU_PATIENCE)
    parser.add_argument("--min-quality-improvement", type=float, default=DEFAULT_MIN_QUALITY_IMPROVEMENT)
    parser.add_argument("--no-include-raw-llm", action="store_true")
    parser.add_argument("--min-best-schema-validity", type=float, default=DEFAULT_MIN_SCHEMA_VALIDITY)
    parser.add_argument("--min-best-non-empty-rate", type=float, default=DEFAULT_MIN_NON_EMPTY_RATE)
    parser.add_argument("--min-best-judge-score-mean", type=float, default=DEFAULT_MIN_JUDGE_SCORE_MEAN)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    seed_candidate = load_candidate(args.seed_candidate_json)
    optimize_candidates(
        input_path=args.input,
        seed_candidate=seed_candidate,
        output_dir=args.output_dir,
        run_name=args.run_name,
        max_evals=args.max_evals,
        max_docs=args.max_docs,
        max_total_chunks=args.max_total_chunks,
        judge_model=args.judge_model,
        judge_chunk_limit=args.judge_chunk_limit,
        reflection_model=args.reflection_model,
        mutations_per_generation=args.mutations_per_generation,
        frontier_parent_limit=args.frontier_parent_limit,
        plateau_patience=args.plateau_patience,
        min_quality_improvement=args.min_quality_improvement,
        include_raw_llm=not args.no_include_raw_llm,
        min_best_schema_validity=args.min_best_schema_validity,
        min_best_non_empty_rate=args.min_best_non_empty_rate,
        min_best_judge_score_mean=args.min_best_judge_score_mean,
    )


if __name__ == "__main__":
    main()
