#!/usr/bin/env python3
"""
Compile MARRO UK-dataset .txt files into a single JSON file.

Each output item has:
- title
- extracted_facts (all FAC sentences concatenated)
- extracted_nonfacts (all non-FAC sentences concatenated)
"""

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "MARRO" / "UK-dataset"


def parse_labeled_line(line: str):
    line = line.strip()
    if not line or "\t" not in line:
        return None
    sentence, label = line.rsplit("\t", 1)
    sentence = sentence.strip()
    label = label.strip().upper()
    if not sentence:
        return None
    return sentence, label


def compile_document(path: Path) -> dict:
    facts = []
    nonfacts = []

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            parsed = parse_labeled_line(raw_line)
            if parsed is None:
                continue
            sentence, label = parsed
            if label == "FAC":
                facts.append(sentence)
            else:
                nonfacts.append(sentence)

    return {
        "title": path.stem,
        "extracted_facts": " ".join(facts),
        "extracted_nonfacts": " ".join(nonfacts),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compile MARRO UK-dataset text files into one JSON file with concatenated facts/non-facts."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing MARRO UK .txt files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Default: <input_dir>/marro_uk_compiled.json",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation spaces.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_path = args.output.resolve() if args.output else input_dir / "marro_uk_compiled.json"
    doc_paths = sorted(input_dir.glob("*.txt"))
    if not doc_paths:
        raise FileNotFoundError(f"No .txt files found in: {input_dir}")

    compiled = [compile_document(doc_path) for doc_path in doc_paths]

    try:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(compiled, f, ensure_ascii=False, indent=args.indent)
            f.write("\n")
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot write output file: {output_path}. "
            f"Use --output to choose a writable path."
        ) from exc

    print(f"Wrote {len(compiled)} documents to {output_path}")


if __name__ == "__main__":
    main()
