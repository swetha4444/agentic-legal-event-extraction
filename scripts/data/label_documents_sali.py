#!/usr/bin/env python3
"""
Assign SALI LMSS claim labels to documents using OpenAI Chat Completions and a fixed whitelist.

Reads hybrid fact-extraction compiled JSONL (one object per line with document_text,
case_id, court, title). For each document, calls LiteLLM with the whitelist from
data/sali/lmss_claim_subset_v1.json (or --label_json), validates IRIs, writes JSONL:

{"document_id","court","case_id","title","lmss_iris":[...],"model","error"?}

Example:
  python scripts/data/label_documents_sali.py \\
    --scan_dir data/outputs/fact_extraction_hybrid \\
    --model gpt-4o-mini \\
    --out_dir data/outputs/sali_labels/run01

Uses the repo's standard auth/config pattern:
- API key precedence: AGENT_API_KEY -> OPENAI_API_KEY -> llm.api_key in config
- base_url: llm.api_base from config/config.yaml (defaults to Keymaker URL)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABELS = ROOT / "data/sali/lmss_claim_subset_v1.json"
DEFAULT_SCAN = ROOT / "data/outputs/fact_extraction_hybrid"
DEFAULT_CONFIG = ROOT / "config" / "config.yaml"


def load_runtime_llm_config(path: Path = DEFAULT_CONFIG) -> Dict[str, Any]:
    """
    Match existing repo behavior:
    - key: AGENT_API_KEY -> OPENAI_API_KEY -> llm.api_key
    - base_url: llm.api_base -> Keymaker default
    - model: llm.model (optional)
    """
    data: Dict[str, Any] = {}
    if path.exists():
        try:
            import yaml  # type: ignore
        except ImportError:
            yaml = None
        if yaml is not None:
            with path.open(encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
    llm = data.get("llm") or {}
    return {
        "api_key": (os.environ.get("AGENT_API_KEY") or os.environ.get("OPENAI_API_KEY") or llm.get("api_key") or "").strip(),
        "api_base": (llm.get("api_base") or "https://thekeymaker.umass.edu/").strip(),
        "model": (llm.get("model") or "").strip(),
    }


def load_whitelist(path: Path) -> Tuple[List[Dict[str, str]], Set[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    labels = list(data["labels"])
    iris = {str(x["iri"]).strip() for x in labels}
    return labels, iris


def discover_compiled(scan_dir: Path) -> List[Path]:
    """One newest *_compiled.jsonl per immediate parent folder under scan_dir."""
    by_parent: Dict[Path, Path] = {}
    for p in scan_dir.rglob("*_compiled.jsonl"):
        if not p.is_file():
            continue
        parent = p.parent
        prev = by_parent.get(parent)
        if prev is None or p.stat().st_mtime > prev.stat().st_mtime:
            by_parent[parent] = p
    return sorted(by_parent.values())


def iter_documents(paths: Iterable[Path]) -> Iterable[Dict[str, Any]]:
    seen: Set[Tuple[str, str]] = set()
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = (obj.get("document_text") or "").strip()
                if not text:
                    continue
                case_id = str(obj.get("case_id") or obj.get("title") or "").strip()
                court = str(obj.get("court") or "").strip()
                key = (court, case_id)
                if key in seen:
                    continue
                seen.add(key)
                yield {
                    "document_id": case_id,
                    "case_id": case_id,
                    "court": court,
                    "title": str(obj.get("title") or case_id),
                    "document_text": text,
                    "source_compiled": str(path),
                }


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head - 80
    return text[:head] + "\n\n[... omitted middle ...]\n\n" + text[-tail:]


def build_label_catalog_snippet(labels: List[Dict[str, str]], max_def: int = 220) -> str:
    lines = []
    for x in labels:
        iri = x["iri"]
        name = x.get("pref_label") or ""
        desc = (x.get("definition") or "").replace("\n", " ").strip()
        if len(desc) > max_def:
            desc = desc[: max_def - 3] + "..."
        lines.append(f"- {iri}\n  Name: {name}\n  Summary: {desc}")
    return "\n".join(lines)


def extract_json_object(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    m = re.search(r"\{[\s\S]*\}\s*$", raw)
    if not m:
        m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        raise ValueError("no JSON object in model output")
    return json.loads(m.group(0))


def call_llm(
    client: Any,
    model: str,
    document_id: str,
    court: str,
    title: str,
    body: str,
    catalog: str,
    allowed_iris: Set[str],
    temperature: float,
) -> Tuple[List[str], Optional[str]]:
    if client is None:
        raise RuntimeError("OpenAI client is not initialized")

    system = """You are a legal taxonomy assistant. You label civil/legal documents using ONLY
the LMSS claim concepts provided in the user message. Output strict JSON only."""

    user = f"""Document metadata:
- document_id: {document_id}
- court: {court}
- title: {title}

Applicable LMSS concepts (you may assign ZERO OR MORE; only use IRIs from this list):
{catalog}

Instructions:
- Choose every concept that plausibly applies to the substantive claims or disputes described in the document.
- Use only IRIs copied exactly from the list above. Do not invent IRIs or free-text labels.
- If none apply, return an empty list.

DOCUMENT TEXT:
\"\"\"
{body}
\"\"\"

Return exactly this JSON shape (no markdown fences):
{{"document_id": "{document_id}", "applicable_lmss_iris": ["http://lmss.sali.org/..."], "note": "optional short rationale"}}"""

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    content = (resp.choices[0].message.content or "").strip()
    parsed = extract_json_object(content)
    iris = parsed.get("applicable_lmss_iris") or parsed.get("lmss_iris") or []
    if not isinstance(iris, list):
        raise ValueError("applicable_lmss_iris must be a list")
    out: List[str] = []
    bad: List[str] = []
    for iri in iris:
        if not isinstance(iri, str):
            bad.append(str(iri))
            continue
        iri = iri.strip()
        if iri in allowed_iris:
            if iri not in out:
                out.append(iri)
        else:
            bad.append(iri)
    note = parsed.get("note")
    if bad:
        note = (note + "; " if note else "") + "dropped_unknown_iris=" + ",".join(bad[:5])
    return out, note


def main() -> None:
    parser = argparse.ArgumentParser(description="Label documents with SALI LMSS subset via LLM")
    parser.add_argument(
        "--scan_dir",
        type=Path,
        default=DEFAULT_SCAN,
        help="Directory tree containing *_compiled.jsonl (default: hybrid outputs)",
    )
    parser.add_argument(
        "--compiled_jsonl",
        type=Path,
        action="append",
        default=None,
        help="Explicit compiled JSONL path (repeatable). Overrides discovery when provided.",
    )
    parser.add_argument(
        "--label_json",
        type=Path,
        default=DEFAULT_LABELS,
        help="Whitelist JSON (default: data/sali/lmss_claim_subset_v1.json)",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML config path (default: config/config.yaml)")
    parser.add_argument("--model", type=str, default=None, help="Model id (default: llm.model from config, else gpt4o)")
    parser.add_argument("--max_chars", type=int, default=24000, help="Truncate document_text for prompting")
    parser.add_argument("--max_docs", type=int, default=None, help="Stop after N documents (debug)")
    parser.add_argument("--sleep_s", type=float, default=0.0, help="Pause between API calls")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output directory (default: data/outputs/sali_labels/<utc_timestamp>)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Discover inputs and write manifest only; no LLM calls",
    )
    args = parser.parse_args()
    runtime_cfg = load_runtime_llm_config(args.config)

    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("openai is required; pip install openai") from e

    api_key = runtime_cfg["api_key"]
    api_base = runtime_cfg["api_base"]
    model = args.model or runtime_cfg["model"] or "gpt4o"
    if not api_key:
        raise RuntimeError("Missing API key: set AGENT_API_KEY (preferred) or OPENAI_API_KEY")
    client = OpenAI(api_key=api_key, base_url=api_base, timeout=180.0)

    labels, allowed_iris = load_whitelist(args.label_json)
    catalog = build_label_catalog_snippet(labels)

    if args.compiled_jsonl:
        compiled_paths = [p.resolve() for p in args.compiled_jsonl]
    else:
        if not args.scan_dir.is_dir():
            print(f"Error: scan_dir not found: {args.scan_dir}", file=sys.stderr)
            sys.exit(1)
        compiled_paths = discover_compiled(args.scan_dir.resolve())

    if not compiled_paths:
        print("No *_compiled.jsonl files found.", file=sys.stderr)
        sys.exit(1)

    out_dir = args.out_dir
    if out_dir is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        out_dir = ROOT / "data/outputs/sali_labels" / ts
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "documents_lmss_labels.jsonl"
    manifest_path = out_dir / "manifest.json"

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "api_base": api_base,
        "config_path": str(args.config),
        "label_json": str(args.label_json),
        "whitelist_size": len(allowed_iris),
        "compiled_inputs": [str(p) for p in compiled_paths],
        "dry_run": args.dry_run,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    docs = list(iter_documents(compiled_paths))
    if args.max_docs is not None:
        docs = docs[: args.max_docs]

    print(f"Whitelist: {len(allowed_iris)} LMSS IRIs from {args.label_json}")
    print(f"Model: {model}")
    print(f"API base: {api_base}")
    print(f"Compiled files: {len(compiled_paths)}")
    print(f"Documents (deduped by court+case_id): {len(docs)}")
    print(f"Output dir: {out_dir}")

    if args.dry_run:
        (out_dir / "dry_run_doc_ids.txt").write_text(
            "\n".join(d["document_id"] for d in docs) + "\n", encoding="utf-8"
        )
        print("Dry run: wrote manifest and dry_run_doc_ids.txt")
        return

    n_ok = 0
    n_err = 0
    with out_jsonl.open("w", encoding="utf-8") as sink:
        for i, row in enumerate(docs):
            doc_id = row["document_id"]
            body = truncate(row["document_text"], args.max_chars)
            try:
                iris, note = call_llm(
                    client,
                    model,
                    doc_id,
                    row["court"],
                    row["title"],
                    body,
                    catalog,
                    allowed_iris,
                    args.temperature,
                )
                rec = {
                    "document_id": doc_id,
                    "case_id": row["case_id"],
                    "court": row["court"],
                    "title": row["title"],
                    "lmss_iris": iris,
                    "model": model,
                    "source_compiled": row["source_compiled"],
                    "note": note,
                }
                n_ok += 1
            except Exception as e:
                rec = {
                    "document_id": doc_id,
                    "case_id": row["case_id"],
                    "court": row["court"],
                    "title": row["title"],
                    "lmss_iris": [],
                    "model": model,
                    "source_compiled": row["source_compiled"],
                    "error": str(e),
                }
                n_err += 1
            sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if args.sleep_s > 0:
                time.sleep(args.sleep_s)
            if (i + 1) % 10 == 0:
                print(f"  labeled {i + 1}/{len(docs)} …")

    manifest["documents_total"] = len(docs)
    manifest["labeled_ok"] = n_ok
    manifest["labeled_errors"] = n_err
    manifest["output_jsonl"] = str(out_jsonl)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Done. Wrote {out_jsonl} ({n_ok} ok, {n_err} errors)")


if __name__ == "__main__":
    main()
