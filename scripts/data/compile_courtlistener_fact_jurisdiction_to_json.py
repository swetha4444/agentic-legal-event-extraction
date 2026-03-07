#!/usr/bin/env python3
"""
Build a training-ready sentence-label dataset from CourtListener processed JSON files.

Default behavior:
- Keep docs that contain jurisdiction/venue section names (the 108-doc analysis set).
- Label FACT (1): section names matching "fact" (case-insensitive).
- Label NON-FACT (0): section names matching "jurisdiction|venue" OR selected
  extra non-fact section names from coverage analysis.
- Write output as line-delimited JSON objects (JSONL content) to the same default
  path used earlier:
  /work/pi_dagarwal_umass_edu/project_1/sriram/outputs/processed_json/courtlistener_fact_jurisdiction_compiled.json

Each output line has:
{"sentence": "...", "label": 0|1, "title": "...", "section_name": "..."}
"""

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "courtlistener-pipeline" / "courtlistener_processed"
DEFAULT_OUTPUT = (
    Path("/work/pi_dagarwal_umass_edu/project_1/sriram/outputs/processed_json")
    / "courtlistener_fact_jurisdiction_compiled.json"
)
DEFAULT_FACT_REGEX = r"fact"
DEFAULT_BASE_NONFACT_REGEX = r"jurisdiction|venue"

# Selected from 108-doc coverage analysis to push NON-FACT into ~40-50% range.
DEFAULT_EXTRA_NONFACT_SECTION_NAMES: Sequence[str] = (
    "PARTIES",
    "PRAYER FOR RELIEF",
    "COUNT I",
    "INTRODUCTION",
    "NOTE: IN LAND CONDEMNATION CASES, USE THE LOCATION OF",
    "COUNT II",
    "JURY DEMAND",
    "IF ANY JUDGE DOCKET NUMBER",
    "I. (A) PLAINTIFFS DEFENDANTS",
    "I. INTRODUCTION",
    "CIVIL ACTION COMPLAINT",
    "II. PARTIES",
    "COUNT III",
    "EXHAUSTION OF ADMINISTRATIVE REMEDIES",
    "RELIEF",
    "THE PARTIES",
    "COUNT IV",
    "COVER1SHEET",
    "EASTERN DISTRICT OF PENNSYLVANIA",
    "JURY TRIAL DEMAND",
)


def _normalize_name(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip()).upper()


def _to_sentence_list(value) -> List[str]:
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    text = str(value or "").strip()
    return [text] if text else []


def _iter_json_files(input_dir: Path) -> Iterable[Path]:
    return sorted(p for p in input_dir.glob("*.json") if p.is_file())


def _resolve_title(payload: dict, source_path: Path) -> str:
    raw_file = str(payload.get("file") or "").strip()
    if raw_file:
        return Path(raw_file).stem
    return source_path.stem.replace("_processed", "")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compile CourtListener processed JSON into training-ready sentence labels "
            "(line-delimited JSON objects)."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing CourtListener processed .json files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Output dataset path (JSONL content). "
            "Default: /work/pi_dagarwal_umass_edu/project_1/sriram/outputs/processed_json/"
            "courtlistener_fact_jurisdiction_compiled.json"
        ),
    )
    parser.add_argument(
        "--fact_regex",
        type=str,
        default=DEFAULT_FACT_REGEX,
        help='Regex for section names mapped to label=1 (FACT). Default: "fact"',
    )
    parser.add_argument(
        "--base_nonfact_regex",
        type=str,
        default=DEFAULT_BASE_NONFACT_REGEX,
        help='Regex for base non-fact sections. Default: "jurisdiction|venue"',
    )
    parser.add_argument(
        "--use_default_extra_nonfact_names",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use curated extra non-fact section names from analysis (default: enabled). "
            "Pass --no-use_default_extra_nonfact_names to disable."
        ),
    )
    parser.add_argument(
        "--require_base_nonfact_doc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep only docs containing base_nonfact_regex (default: enabled). "
            "Pass --no-require_base_nonfact_doc to disable."
        ),
    )
    parser.add_argument(
        "--drop_docs_without_any_labels",
        action="store_true",
        help="Skip docs with no extracted fact/non-fact labeled sentences.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    fact_pattern = re.compile(args.fact_regex, flags=re.IGNORECASE)
    base_nonfact_pattern = re.compile(args.base_nonfact_regex, flags=re.IGNORECASE)

    extra_nonfact_names = set()
    if args.use_default_extra_nonfact_names:
        extra_nonfact_names = {_normalize_name(x) for x in DEFAULT_EXTRA_NONFACT_SECTION_NAMES}

    json_files = list(_iter_json_files(input_dir))
    if not json_files:
        raise FileNotFoundError(f"No .json files found in: {input_dir}")

    output_rows = []
    docs_seen = 0
    docs_kept = 0
    docs_with_fact = 0
    docs_with_nonfact = 0
    fact_count = 0
    nonfact_count = 0

    for path in json_files:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            continue

        sections = payload.get("sections") or []
        if not isinstance(sections, list):
            sections = []
        docs_seen += 1

        if args.require_base_nonfact_doc:
            if not any(
                isinstance(sec, dict)
                and base_nonfact_pattern.search(str(sec.get("name") or ""))
                for sec in sections
            ):
                continue

        title = _resolve_title(payload, path)
        doc_fact = 0
        doc_nonfact = 0
        doc_rows = []

        for section in sections:
            if not isinstance(section, dict):
                continue
            raw_name = str(section.get("name") or "").strip()
            if not raw_name:
                continue
            name_norm = _normalize_name(raw_name)
            sentences = _to_sentence_list(section.get("sentences"))
            if not sentences:
                continue

            label = None
            # Precedence: FACT first, then NON-FACT.
            if fact_pattern.search(raw_name):
                label = 1
            elif base_nonfact_pattern.search(raw_name) or name_norm in extra_nonfact_names:
                label = 0

            if label is None:
                continue

            for sent in sentences:
                doc_rows.append(
                    {
                        "sentence": sent,
                        "label": label,
                        "title": title,
                        "section_name": raw_name,
                    }
                )
                if label == 1:
                    doc_fact += 1
                else:
                    doc_nonfact += 1

        if args.drop_docs_without_any_labels and not doc_rows:
            continue

        docs_kept += 1
        if doc_fact > 0:
            docs_with_fact += 1
        if doc_nonfact > 0:
            docs_with_nonfact += 1
        fact_count += doc_fact
        nonfact_count += doc_nonfact
        output_rows.extend(doc_rows)

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    names_used_path = output_path.with_name(output_path.stem + "_nonfact_names_used.txt")
    with names_used_path.open("w", encoding="utf-8") as f:
        for name in DEFAULT_EXTRA_NONFACT_SECTION_NAMES:
            f.write(name + "\n")

    total = fact_count + nonfact_count
    nonfact_pct = (100.0 * nonfact_count / total) if total else 0.0

    print(f"Input docs scanned: {docs_seen}")
    print(f"Docs kept: {docs_kept}")
    print(f"Docs with FACT labels: {docs_with_fact}")
    print(f"Docs with NON-FACT labels: {docs_with_nonfact}")
    print(f"Fact sentences (label=1): {fact_count}")
    print(f"Non-fact sentences (label=0): {nonfact_count}")
    print(f"Non-fact percentage: {nonfact_pct:.2f}%")
    print(f"Rows written: {len(output_rows)}")
    print(f"Wrote training dataset: {output_path}")
    print(f"Wrote non-fact names list: {names_used_path}")


if __name__ == "__main__":
    main()
