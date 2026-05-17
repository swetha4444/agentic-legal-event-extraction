#!/usr/bin/env python3
"""
Label documents with SALI LMSS claim labels from EKG JSONL input.

Input (default):
  data/outputs/qwen122b_hybrid_doc_graphs_final.jsonl

Output JSONL rows:
  {
    "document_id": "...",
    "lmss_iris": ["http://lmss.sali.org/..."],
    "label_source": "ekg",
    "model": "...",
    "note": "...",
    "error": "..."?
  }
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EKG = ROOT / "data/outputs/qwen122b_hybrid_doc_graphs_final.jsonl"
DEFAULT_LABELS = ROOT / "data/sali/lmss_claim_subset_v1.json"
DEFAULT_CONFIG = ROOT / "config/config.yaml"


def load_runtime_llm_config(path: Path = DEFAULT_CONFIG) -> Dict[str, Any]:
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


def _safe_trigger(event: Dict[str, Any]) -> str:
    t = event.get("trigger") or {}
    if isinstance(t, dict):
        return str(t.get("span_text") or "").strip()
    return ""


def ekg_to_prompt_text(row: Dict[str, Any], max_events: int = 80, max_entities: int = 60) -> str:
    graph = row.get("merged_graph") or {}
    events = list(graph.get("events") or [])[:max_events]
    entities = list(graph.get("entities") or [])[:max_entities]

    ent_lines = []
    for ent in entities:
        nm = str(ent.get("name") or "").strip()
        role = str(ent.get("canonical_role") or "UNKNOWN").strip()
        if nm:
            ent_lines.append(f"- {nm} (role={role})")

    ev_lines = []
    for ev in events:
        et = str(ev.get("event_type") or "UNKNOWN").strip()
        trig = _safe_trigger(ev)
        sent_ids = ev.get("sentence_ids") or []
        sid_text = ",".join(str(x) for x in sent_ids[:5]) if isinstance(sent_ids, list) else ""
        if trig:
            ev_lines.append(f"- [{et}] {trig} (sent_ids={sid_text})")
        else:
            ev_lines.append(f"- [{et}] (sent_ids={sid_text})")

    return (
        f"DOC_ID: {row.get('doc_id')}\n"
        f"NUM_CHUNKS: {row.get('num_chunks')}\n\n"
        "KEY ENTITIES:\n"
        + ("\n".join(ent_lines) if ent_lines else "- (none)")
        + "\n\nKEY EVENTS:\n"
        + ("\n".join(ev_lines) if ev_lines else "- (none)")
    )


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
    ekg_text: str,
    catalog: str,
    allowed_iris: Set[str],
    temperature: float,
) -> Tuple[List[str], Optional[str]]:
    system = (
        "You are a legal taxonomy assistant. "
        "Assign SALI LMSS claim concepts using ONLY the provided whitelist. "
        "Use the event graph summary as evidence. Output strict JSON."
    )
    user = f"""Document id: {document_id}

Applicable LMSS concepts (use only IRIs from this list):
{catalog}

Event Graph Summary:
\"\"\"
{ekg_text}
\"\"\"

Return strict JSON only:
{{"document_id":"{document_id}","applicable_lmss_iris":["http://lmss.sali.org/..."],"note":"optional short rationale"}}
"""
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


def iter_ekg_docs(path: Path) -> Iterable[Dict[str, Any]]:
    seen: Set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            doc_id = str(row.get("doc_id") or "").strip()
            if not doc_id or doc_id in seen:
                continue
            seen.add(doc_id)
            yield row


def main() -> None:
    parser = argparse.ArgumentParser(description="Label SALI claims from EKG doc graphs")
    parser.add_argument("--ekg_jsonl", type=Path, default=DEFAULT_EKG)
    parser.add_argument("--label_json", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", type=str, default=None, help="Default: llm.model from config")
    parser.add_argument("--max_docs", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--sleep_s", type=float, default=0.0)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    runtime_cfg = load_runtime_llm_config(args.config)
    api_key = runtime_cfg["api_key"]
    api_base = runtime_cfg["api_base"]
    model = args.model or runtime_cfg["model"] or "gpt4o"

    labels, allowed_iris = load_whitelist(args.label_json)
    catalog = build_label_catalog_snippet(labels)
    docs = list(iter_ekg_docs(args.ekg_jsonl.resolve()))
    if args.max_docs is not None:
        docs = docs[: args.max_docs]

    out_dir = args.out_dir
    if out_dir is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        out_dir = ROOT / "data/outputs/sali_labels" / f"ekg_{ts}"
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "documents_lmss_labels_ekg.jsonl"
    manifest_path = out_dir / "manifest_ekg.json"

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "ekg",
        "ekg_jsonl": str(args.ekg_jsonl.resolve()),
        "label_json": str(args.label_json.resolve()),
        "model": model,
        "api_base": api_base,
        "documents_total": len(docs),
        "dry_run": args.dry_run,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"EKG docs: {len(docs)} | model={model}")
    print(f"Output dir: {out_dir}")
    if args.dry_run:
        return

    if not api_key:
        raise RuntimeError("Missing API key: set AGENT_API_KEY (preferred) or OPENAI_API_KEY")
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("openai is required; pip install openai") from e
    client = OpenAI(api_key=api_key, base_url=api_base, timeout=180.0)

    n_ok = 0
    n_err = 0
    with out_jsonl.open("w", encoding="utf-8") as sink:
        for i, doc in enumerate(docs, start=1):
            doc_id = str(doc.get("doc_id") or "")
            ekg_text = ekg_to_prompt_text(doc)
            try:
                iris, note = call_llm(client, model, doc_id, ekg_text, catalog, allowed_iris, args.temperature)
                rec = {
                    "document_id": doc_id,
                    "lmss_iris": iris,
                    "label_source": "ekg",
                    "model": model,
                    "note": note,
                }
                n_ok += 1
            except Exception as e:
                rec = {
                    "document_id": doc_id,
                    "lmss_iris": [],
                    "label_source": "ekg",
                    "model": model,
                    "error": str(e),
                }
                n_err += 1
            sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if args.sleep_s > 0:
                time.sleep(args.sleep_s)
            if i % 10 == 0:
                print(f"  labeled {i}/{len(docs)} ...")

    manifest["labeled_ok"] = n_ok
    manifest["labeled_errors"] = n_err
    manifest["output_jsonl"] = str(out_jsonl)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Done. Wrote {out_jsonl} ({n_ok} ok, {n_err} errors)")


if __name__ == "__main__":
    main()
