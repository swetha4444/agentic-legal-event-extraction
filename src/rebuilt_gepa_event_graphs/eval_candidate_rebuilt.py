#!/usr/bin/env python3
"""CLI for evaluating one rebuilt GEPA candidate."""
import argparse
from pathlib import Path

from rebuilt_gepa_event_graphs.core import evaluate_candidate, load_candidate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one rebuilt GEPA event-graph candidate.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--candidate-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--max-total-chunks", type=int, default=None)
    parser.add_argument("--judge-model", default="gpt-5-mini")
    parser.add_argument("--judge-chunk-limit", type=int, default=5)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--no-include-raw-llm", action="store_true")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    candidate = load_candidate(args.candidate_json)
    evaluate_candidate(
        input_path=args.input,
        output_dir=args.output_dir,
        run_name=args.run_name,
        candidate=candidate,
        max_docs=args.max_docs,
        max_total_chunks=args.max_total_chunks,
        judge_model=None if args.no_judge else args.judge_model,
        judge_chunk_limit=None if args.no_judge else args.judge_chunk_limit,
        include_raw_llm=not args.no_include_raw_llm,
    )


if __name__ == "__main__":
    main()
