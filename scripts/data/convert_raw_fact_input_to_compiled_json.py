#!/usr/bin/env python3
"""
Convert raw JSONL/JSON document records into a compact JSON/JSONL format suitable
for prediction-time fact extraction scripts.

This is intentionally lightweight:
- input records may have raw text in fields like document_text/opinion_text/complaint_text
- output records preserve stable identifiers plus one normalized document_text field
- the output can be either JSON array or JSONL, based on the output suffix
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_TEXT_FIELD_CANDIDATES = (
    "document_text",
    "opinion_text",
    "complaint_text",
    "text",
    "content",
)

DEFAULT_TITLE_FIELD_CANDIDATES = (
    "title",
    "case_name",
    "case_id",
    "id",
    "docket_number",
)

DEFAULT_PASSTHROUGH_FIELDS = (
    "case_id",
    "court",
    "case_name",
    "title",
    "docket_number",
    "source",
    "jurisdiction",
)


def _load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict):
                    records.append(payload)
        return records

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON array in {path}, got {type(payload).__name__}")
    return [rec for rec in payload if isinstance(rec, dict)]


def _first_nonempty(record: dict[str, Any], candidates: tuple[str, ...]) -> tuple[str | None, str]:
    for key in candidates:
        value = record.get(key)
        text = str(value or "").strip()
        if text:
            return key, text
    return None, ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert raw JSONL/JSON records into a compact compiled JSON for fact prediction.",
    )
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL or JSON file.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON or JSONL file. Use .jsonl for large raw corpora.",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default=None,
        help="Explicit text field to use. If omitted, common raw-text fields are tried automatically.",
    )
    parser.add_argument(
        "--title-field",
        type=str,
        default=None,
        help="Explicit title field to use. If omitted, common id/title fields are tried automatically.",
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    records = _load_records(input_path)
    compiled: list[dict[str, Any]] = []
    skipped = 0

    for idx, record in enumerate(records, start=1):
        text_key, text = _first_nonempty(
            record,
            (args.text_field,) if args.text_field else DEFAULT_TEXT_FIELD_CANDIDATES,
        )
        if not text:
            skipped += 1
            continue

        title_key, title = _first_nonempty(
            record,
            (args.title_field,) if args.title_field else DEFAULT_TITLE_FIELD_CANDIDATES,
        )
        if not title:
            title = f"doc_{idx:05d}"

        out: dict[str, Any] = {
            "title": title,
            "document_text": text,
            "source_text_field": text_key,
            "source_title_field": title_key,
            "source_record_index": idx,
        }

        for field in DEFAULT_PASSTHROUGH_FIELDS:
            value = record.get(field)
            if value not in (None, "", [], {}):
                out[field] = value

        compiled.append(out)

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        if output_path.suffix.lower() == ".jsonl":
            for record in compiled:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        else:
            json.dump(compiled, f, ensure_ascii=False, indent=2)
            f.write("\n")

    print(
        json.dumps(
            {
                "input": str(input_path),
                "output": str(output_path),
                "records_read": len(records),
                "records_written": len(compiled),
                "records_skipped": skipped,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
