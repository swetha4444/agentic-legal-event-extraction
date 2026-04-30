"""
Dataset Preprocessing CLI
Reads raw CourtListener JSONL, cleans text, writes structured output.
Scales to any dataset size — processes one case at a time (streaming).

Usage:
    python src/run_preprocess.py --input data/processed/courtlistener_recap.jsonl --output data/processed/clean_dataset.jsonl
    python src/run_preprocess.py --input data/processed/courtlistener_recap.jsonl --output data/processed/clean_dataset.jsonl --compare
"""

import json
import argparse
import os
import sys
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, os.path.dirname(__file__))
from processing.preprocess import preprocess_case, TextCleaner


def run_preprocess(input_file: str, output_file: str, limit: int = None, 
                   show_compare: bool = False):
    """
    Main preprocessing function. Reads JSONL line by line for scalability.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    count = 0
    total_orig = 0
    total_clean = 0

    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        lines = infile.readlines()
        if limit:
            lines = lines[:limit]

        print(f"Preprocessing {len(lines)} cases...")

        for line in tqdm(lines):
            try:
                case_data = json.loads(line)
            except json.JSONDecodeError:
                continue

            result = preprocess_case(case_data)
            outfile.write(json.dumps(result) + "\n")
            count += 1

            total_orig += result["metadata"]["original_length"]
            total_clean += result["metadata"]["cleaned_length"]

            # Show before/after for first 2 cases if requested
            if show_compare and count <= 2:
                print_comparison(case_data, result)

    reduction = round((1 - total_clean / total_orig) * 100) if total_orig else 0
    print(f"\nDone! Processed {count} cases.")
    print(f"  Original: {total_orig:,} chars")
    print(f"  Cleaned:  {total_clean:,} chars")
    print(f"  Reduced:  {reduction}%")
    print(f"  Output:   {output_file}")


def print_comparison(raw: dict, cleaned: dict):
    """Print a side-by-side before/after comparison for one case."""
    raw_text = raw.get("document_text", "")
    clean_text = cleaned.get("clean_text", "")

    print("\n" + "=" * 70)
    print(f"CASE: {cleaned['case_name']}")
    print("=" * 70)

    print("\n--- BEFORE (first 500 chars) ---")
    print(raw_text[:500])

    print("\n--- AFTER (first 500 chars) ---")
    print(clean_text[:500])

    print(f"\n--- METADATA ---")
    meta = cleaned["metadata"]
    print(f"  Original length:  {meta['original_length']:,} chars")
    print(f"  Cleaned length:   {meta['cleaned_length']:,} chars")
    print(f"  Reduction:        {meta['reduction_pct']}")
    print(f"  Sections found:   {meta['sections_found']}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess raw legal dataset")
    parser.add_argument("--input", required=True, help="Path to raw JSONL file")
    parser.add_argument("--output", required=True, help="Path to cleaned output JSONL")
    parser.add_argument("--limit", type=int, help="Limit number of cases (for testing)")
    parser.add_argument("--compare", action="store_true", 
                        help="Show before/after comparison for first 2 cases")

    args = parser.parse_args()
    run_preprocess(args.input, args.output, args.limit, args.compare)
