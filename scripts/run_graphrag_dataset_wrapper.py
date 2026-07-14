#!/usr/bin/env python3
"""
Prepare graph-RAG-ready artifacts from an EKG folder, chunks JSONL, and QA JSONL.

This wrapper keeps the original inputs unchanged. It normalizes per-document EKG JSON
into the merged graph format expected by `graph-rag/graphrag/loader.py`, builds a
dataset JSON with one `docs[]` entry per `doc_id`, and can optionally hand the result
to the existing graph-RAG inference script.

Typical usage:
  python scripts/run_graphrag_dataset_wrapper.py \
      --ekg-dir legal_bench/step5_doc_ekg_llm \
      --chunks-jsonl legal_bench/chunks.jsonl \
      --qa-jsonl legal_bench/qa.jsonl \
      --output-dir legal_bench/graphrag_ready

Optional inference:
  python scripts/run_graphrag_dataset_wrapper.py \
      --ekg-dir legal_bench/step5_doc_ekg_llm \
      --chunks-jsonl legal_bench/chunks.jsonl \
      --qa-jsonl legal_bench/qa.jsonl \
      --output-dir legal_bench/graphrag_ready \
      --run-inference \
      --model gpt-4.1-mini \
      --inference-output-dir .artifacts/legal_bench
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(row)
    return rows


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _normalize_sentence_ids(values: Any) -> list[int]:
    out: list[int] = []
    if not isinstance(values, list):
        return out
    for item in values:
        if isinstance(item, int):
            out.append(item)
            continue
        text = _safe_text(item)
        if not text:
            continue
        match = re.search(r"\d+", text)
        if match:
            out.append(int(match.group()))
    return sorted(set(out))


def _normalize_snippets(value: Any, fallback_text: str = "") -> list[dict[str, str]]:
    snippets = value if isinstance(value, list) else []
    out: list[dict[str, str]] = []
    for snippet in snippets:
        if isinstance(snippet, dict):
            text = _safe_text(snippet.get("text"))
        else:
            text = _safe_text(snippet)
        if text:
            out.append({"text": text})
    if not out and fallback_text:
        out.append({"text": fallback_text})
    return out


def _normalize_evidence(raw: Any, *, fallback_text: str = "", fallback_sentence_ids: Any = None) -> dict[str, Any]:
    if isinstance(raw, dict):
        sentence_ids = _normalize_sentence_ids(raw.get("sentence_ids"))
        snippets = _normalize_snippets(raw.get("snippets"), fallback_text=fallback_text)
        if not snippets and fallback_text:
            snippets = [{"text": fallback_text}]
        return {"sentence_ids": sentence_ids, "snippets": snippets}
    return {
        "sentence_ids": _normalize_sentence_ids(fallback_sentence_ids),
        "snippets": _normalize_snippets([], fallback_text=fallback_text),
    }


def _normalize_entities(raw_entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    for entity in raw_entities:
        entity_id = _safe_text(entity.get("entity_id") or entity.get("id"))
        if not entity_id:
            continue
        aliases = entity.get("aliases") or []
        entities.append(
            {
                "entity_id": entity_id,
                "name": _safe_text(entity.get("name")),
                "kind": _safe_text(entity.get("kind")) or "UNKNOWN",
                "canonical_role": _safe_text(entity.get("canonical_role")) or "UNKNOWN",
                "aliases": [_safe_text(alias) for alias in aliases if _safe_text(alias)],
            }
        )
    return entities


def _normalize_participants(raw_participants: list[dict[str, Any]]) -> list[dict[str, Any]]:
    participants: list[dict[str, Any]] = []
    for participant in raw_participants:
        if not isinstance(participant, dict):
            continue
        mention_text = ""
        mention = participant.get("mention")
        if isinstance(mention, dict):
            mention_text = _safe_text(mention.get("span_text"))
        else:
            mention_text = _safe_text(mention)
        if not mention_text:
            mention_text = _safe_text(participant.get("evidence"))
        participants.append(
            {
                "entity_id": _safe_text(participant.get("entity_id")),
                "role": _safe_text(participant.get("role")),
                "mention": {"span_text": mention_text},
            }
        )
    return participants


def _normalize_source_refs(event: dict[str, Any]) -> list[dict[str, str]]:
    refs = event.get("source_refs")
    if isinstance(refs, list) and refs:
        out: list[dict[str, str]] = []
        for ref in refs:
            if isinstance(ref, dict):
                ref_type = _safe_text(ref.get("type")) or "source_ref"
                ref_value = _safe_text(ref.get("value") or ref.get("id") or ref.get("ref"))
            else:
                ref_type = "source_ref"
                ref_value = _safe_text(ref)
            if ref_value:
                out.append({"type": ref_type, "value": ref_value})
        if out:
            return out

    out: list[dict[str, str]] = []
    for chunk_id in event.get("chunk_ids") or []:
        chunk_text = _safe_text(chunk_id)
        if chunk_text:
            out.append({"type": "chunk_id", "value": chunk_text})
    for source_event_id in event.get("source_event_ids") or []:
        source_text = _safe_text(source_event_id)
        if source_text:
            out.append({"type": "source_event_id", "value": source_text})
    return out


def _normalize_events(raw_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for event in raw_events:
        event_id = _safe_text(event.get("event_id") or event.get("id"))
        if not event_id:
            continue
        trigger = event.get("trigger") or {}
        trigger_text = ""
        if isinstance(trigger, dict):
            trigger_text = _safe_text(trigger.get("span_text"))
        else:
            trigger_text = _safe_text(trigger)
        evidence = _normalize_evidence(event.get("evidence"))
        events.append(
            {
                "event_id": event_id,
                "event_type": _safe_text(event.get("event_type")) or "UNKNOWN",
                "main_verb": _safe_text(event.get("main_verb")) or trigger_text,
                "trigger": {"span_text": trigger_text},
                "time": event.get("time") or {"kind": "UNKNOWN"},
                "confidence": float(event.get("confidence") or 0.0),
                "evidence": evidence,
                "participants": _normalize_participants(event.get("participants") or []),
                "source_refs": _normalize_source_refs(event),
            }
        )
    return events


def _edge_key(edge_kind: str, from_event: str, to_event: str, relation: str) -> tuple[str, str, str, str]:
    return (edge_kind, from_event, to_event, relation)


def _normalize_edge(
    raw_edge: dict[str, Any],
    *,
    default_relation: str = "",
) -> dict[str, Any] | None:
    from_event = _safe_text(
        raw_edge.get("from_event") or raw_edge.get("source_event_id") or raw_edge.get("source_id")
    )
    to_event = _safe_text(
        raw_edge.get("to_event") or raw_edge.get("target_event_id") or raw_edge.get("target_id")
    )
    relation = _safe_text(raw_edge.get("relation") or raw_edge.get("edge_type") or default_relation)
    if not from_event or not to_event:
        return None
    return {
        "from_event": from_event,
        "to_event": to_event,
        "relation": relation or "RELATED",
        "confidence": float(raw_edge.get("confidence") or 0.0),
        "evidence": _normalize_evidence(
            raw_edge.get("evidence"),
            fallback_text=_safe_text(raw_edge.get("rationale") or raw_edge.get("reason")),
            fallback_sentence_ids=raw_edge.get("evidence_sentence_ids"),
        ),
    }


def _normalize_graph(ekg: dict[str, Any]) -> dict[str, Any]:
    temporal_edges: list[dict[str, Any]] = []
    causal_edges: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()

    for raw_edge in ekg.get("temporal_edges") or []:
        edge = _normalize_edge(raw_edge)
        if not edge:
            continue
        key = _edge_key("temporal", edge["from_event"], edge["to_event"], edge["relation"])
        if key not in seen:
            temporal_edges.append(edge)
            seen.add(key)

    for raw_edge in ekg.get("causal_edges") or []:
        edge = _normalize_edge(raw_edge)
        if not edge:
            continue
        key = _edge_key("causal", edge["from_event"], edge["to_event"], edge["relation"])
        if key not in seen:
            causal_edges.append(edge)
            seen.add(key)

    for raw_edge in (ekg.get("expansion_edges_llm") or []) + (ekg.get("expansion_edges") or []):
        edge_type = _safe_text(raw_edge.get("edge_type") or raw_edge.get("relation")).lower()
        if edge_type.startswith("causal"):
            edge_kind = "causal"
        elif edge_type.startswith("temporal"):
            edge_kind = "temporal"
        else:
            continue
        edge = _normalize_edge(raw_edge, default_relation=edge_type)
        if not edge:
            continue
        key = _edge_key(edge_kind, edge["from_event"], edge["to_event"], edge["relation"])
        if key in seen:
            continue
        if edge_kind == "causal":
            causal_edges.append(edge)
        else:
            temporal_edges.append(edge)
        seen.add(key)

    return {
        "entities": _normalize_entities(ekg.get("entities") or []),
        "events": _normalize_events(ekg.get("events") or []),
        "temporal_edges": temporal_edges,
        "causal_edges": causal_edges,
        "role_edges": ekg.get("role_edges") or [],
    }


def _load_chunk_lookup(path: Path) -> dict[str, dict[str, Any]]:
    rows = _iter_jsonl(path)
    if not rows:
        return {}

    if any("chunks" in row for row in rows):
        out: dict[str, dict[str, Any]] = {}
        for row in rows:
            doc_id = _safe_text(row.get("doc_id") or row.get("case_id"))
            if not doc_id:
                continue
            chunks = row.get("chunks") or []
            out[doc_id] = {
                "doc_id": doc_id,
                "num_chunks": row.get("num_chunks", len(chunks)),
                "chunks": chunks,
            }
        return out

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        doc_id = _safe_text(row.get("doc_id") or row.get("case_id"))
        if not doc_id:
            continue
        grouped[doc_id].append(row)

    out: dict[str, dict[str, Any]] = {}
    for doc_id, chunk_rows in grouped.items():
        out[doc_id] = {
            "doc_id": doc_id,
            "num_chunks": len(chunk_rows),
            "chunks": chunk_rows,
        }
    return out


def _format_question(
    row: dict[str, Any],
    *,
    include_prompt: bool,
    include_choices: bool,
    question_field: str,
) -> str:
    sections: list[str] = []
    prompt = _safe_text(row.get("prompt"))
    question = _safe_text(row.get(question_field))
    if include_prompt and prompt:
        sections.append(f"Context:\n{prompt}")
    if question:
        sections.append(f"Question:\n{question}")
    if include_choices:
        choice_lines: list[str] = []
        for label in ("a", "b", "c", "d", "e", "f"):
            value = _safe_text(row.get(f"choice_{label}"))
            if value:
                choice_lines.append(f"{label.upper()}. {value}")
        if choice_lines:
            sections.append("Choices:\n" + "\n".join(choice_lines))
    return "\n\n".join(section for section in sections if section).strip()


def _faq_id(row: dict[str, Any], doc_id: str, ordinal: int) -> str:
    for key in ("faq_id", "qa_idx", "id", "question_id"):
        value = _safe_text(row.get(key))
        if value:
            return value
    return f"{doc_id}__q{ordinal:03d}"


def _build_faq_entry(
    row: dict[str, Any],
    *,
    doc_id: str,
    ordinal: int,
    include_prompt: bool,
    include_choices: bool,
    question_field: str,
) -> dict[str, Any]:
    question_text = _format_question(
        row,
        include_prompt=include_prompt,
        include_choices=include_choices,
        question_field=question_field,
    )
    known_keys = {
        "faq_id",
        "qa_idx",
        "id",
        "question_id",
        "doc_id",
        "prompt",
        question_field,
        "answer",
        "gold_answer",
        "gold_passage",
        "required_graph_ops",
        "why_ekg_edge",
        "dataset_split",
        "choice_a",
        "choice_b",
        "choice_c",
        "choice_d",
        "choice_e",
        "choice_f",
    }
    extra = {
        key: value
        for key, value in row.items()
        if key not in known_keys and value not in ("", None, [], {})
    }
    return {
        "faq_id": _faq_id(row, doc_id, ordinal),
        "question": question_text,
        "raw_question": _safe_text(row.get(question_field)),
        "prompt": _safe_text(row.get("prompt")),
        "choices": {
            label.upper(): _safe_text(row.get(f"choice_{label}"))
            for label in ("a", "b", "c", "d", "e", "f")
            if _safe_text(row.get(f"choice_{label}"))
        },
        "answer": _safe_text(row.get("answer") or row.get("gold_answer")),
        "gold_passage": _safe_text(row.get("gold_passage")),
        "required_graph_ops": row.get("required_graph_ops"),
        "why_ekg_edge": row.get("why_ekg_edge"),
        "dataset_split": _safe_text(row.get("dataset_split")),
        "metadata": extra,
    }


def _load_qa_groups(
    path: Path,
    *,
    include_prompt: bool,
    include_choices: bool,
    question_field: str,
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _iter_jsonl(path):
        doc_id = _safe_text(row.get("doc_id") or row.get("case_id"))
        if not doc_id:
            continue
        faq = _build_faq_entry(
            row,
            doc_id=doc_id,
            ordinal=len(groups[doc_id]) + 1,
            include_prompt=include_prompt,
            include_choices=include_choices,
            question_field=question_field,
        )
        if faq["question"]:
            groups[doc_id].append(faq)
    return groups


def _write_merged_graphs(
    *,
    ekg_dir: Path,
    ekg_glob: str,
    chunk_lookup: dict[str, dict[str, Any]],
    output_path: Path,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped_missing_chunks = 0
    doc_ids: list[str] = []
    paths = sorted(ekg_dir.glob(ekg_glob))
    if not paths:
        raise SystemExit(f"No EKG files matched {ekg_glob} under {ekg_dir}")

    with output_path.open("w", encoding="utf-8") as fh:
        for ekg_path in paths:
            ekg = _read_json(ekg_path)
            doc_id = _safe_text(ekg.get("doc_id") or ekg_path.stem.replace("_doc_ekg_llm", ""))
            chunk_row = chunk_lookup.get(doc_id)
            if not chunk_row:
                skipped_missing_chunks += 1
                continue
            row = {
                "doc_id": doc_id,
                "num_source_chunks": chunk_row.get("num_chunks", len(chunk_row.get("chunks") or [])),
                "chunks": chunk_row.get("chunks") or [],
                "merged_graph": _normalize_graph(ekg),
                "merge_stats": ekg.get("metadata") or {},
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            doc_ids.append(doc_id)

    return {
        "written_docs": written,
        "skipped_missing_chunks": skipped_missing_chunks,
        "merged_doc_ids": sorted(doc_ids),
        "ekg_files_seen": len(paths),
    }


def _write_dataset_json(
    *,
    qa_groups: dict[str, list[dict[str, Any]]],
    merged_doc_ids: set[str],
    merged_jsonl_path: Path,
    output_path: Path,
    dataset_name: str,
    project_root: Path,
) -> dict[str, Any]:
    docs: list[dict[str, Any]] = []
    skipped_missing_graph = 0
    source_path = _relative_or_absolute(merged_jsonl_path, project_root)

    for doc_id in sorted(qa_groups):
        if doc_id not in merged_doc_ids:
            skipped_missing_graph += 1
            continue
        faqs = qa_groups[doc_id]
        docs.append(
            {
                "doc_id": doc_id,
                "source": source_path,
                "faqs": faqs,
                "num_faqs": len(faqs),
            }
        )

    payload = {
        "dataset_name": dataset_name,
        "created_at_utc": _utc_now(),
        "docs": docs,
        "metadata": {
            "num_docs": len(docs),
            "num_faqs": sum(len(doc["faqs"]) for doc in docs),
            "source_merged_jsonl": source_path,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "written_docs": len(docs),
        "written_faqs": sum(len(doc["faqs"]) for doc in docs),
        "skipped_missing_graph": skipped_missing_graph,
    }


def _run_inference(
    *,
    project_root: Path,
    dataset_json: Path,
    chunks_jsonl: Path,
    args: argparse.Namespace,
) -> None:
    cmd = [
        sys.executable,
        str(project_root / "graph-rag" / "run_faq_inference.py"),
        "--input-json",
        str(dataset_json.resolve()),
        "--chunk-jsonl",
        str(chunks_jsonl.resolve()),
        "--output-dir",
        args.inference_output_dir,
        "--top-k",
        str(args.top_k),
        "--seed-limit",
        str(args.seed_limit),
        "--graph-hops",
        str(args.graph_hops),
        "--final-events",
        str(args.final_events),
    ]
    if args.model:
        cmd.extend(["--model", args.model])
    if args.base_url:
        cmd.extend(["--base-url", args.base_url])
    if args.max_faqs is not None:
        cmd.extend(["--max-faqs", str(args.max_faqs)])
    if args.doc_id:
        cmd.extend(["--doc-id", args.doc_id])

    subprocess.run(cmd, cwd=project_root, check=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ekg-dir", required=True, help="Directory containing per-doc EKG JSON files.")
    parser.add_argument("--chunks-jsonl", required=True, help="JSONL with chunk records grouped by doc_id.")
    parser.add_argument("--qa-jsonl", required=True, help="JSONL with QA rows keyed by doc_id.")
    parser.add_argument("--output-dir", required=True, help="Directory for normalized graph-RAG-ready outputs.")
    parser.add_argument(
        "--dataset-name",
        default="graphrag_dataset",
        help="Name stored in the output dataset JSON metadata.",
    )
    parser.add_argument(
        "--ekg-glob",
        default="*_doc_ekg_llm.json",
        help="Glob used to discover per-doc EKG files inside --ekg-dir.",
    )
    parser.add_argument(
        "--merged-jsonl-name",
        default="merged_graphs.jsonl",
        help="Filename for the normalized merged-graph JSONL output.",
    )
    parser.add_argument(
        "--dataset-json-name",
        default="dataset.json",
        help="Filename for the graph-RAG dataset JSON output.",
    )
    parser.add_argument(
        "--summary-json-name",
        default="summary.json",
        help="Filename for the preparation summary JSON output.",
    )
    parser.add_argument(
        "--question-field",
        default="question",
        help="Field in QA rows that contains the question text.",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Do not prepend the QA row's prompt/context when building the graph-RAG question.",
    )
    parser.add_argument(
        "--no-choices",
        action="store_true",
        help="Do not append multiple-choice options when building the graph-RAG question.",
    )
    parser.add_argument(
        "--run-inference",
        action="store_true",
        help="After preparing artifacts, run graph-rag/run_faq_inference.py on the dataset.",
    )
    parser.add_argument("--model", default=None, help="Model name for optional inference.")
    parser.add_argument("--base-url", default=None, help="Base URL override for optional inference.")
    parser.add_argument(
        "--inference-output-dir",
        default=".artifacts",
        help="Output dir argument passed to graph-rag/run_faq_inference.py.",
    )
    parser.add_argument("--top-k", type=int, default=32, help="Retriever top-k for optional inference.")
    parser.add_argument("--seed-limit", type=int, default=24, help="Retriever seed limit for optional inference.")
    parser.add_argument("--graph-hops", type=int, default=2, help="Retriever graph hops for optional inference.")
    parser.add_argument("--final-events", type=int, default=8, help="Final event count for optional inference.")
    parser.add_argument("--max-faqs", type=int, default=None, help="Optional FAQ cap for inference/debugging.")
    parser.add_argument("--doc-id", default=None, help="Optional single-doc filter for inference/debugging.")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]

    ekg_dir = Path(args.ekg_dir).resolve()
    chunks_jsonl = Path(args.chunks_jsonl).resolve()
    qa_jsonl = Path(args.qa_jsonl).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_jsonl = output_dir / args.merged_jsonl_name
    dataset_json = output_dir / args.dataset_json_name
    summary_json = output_dir / args.summary_json_name

    chunk_lookup = _load_chunk_lookup(chunks_jsonl)
    qa_groups = _load_qa_groups(
        qa_jsonl,
        include_prompt=not args.no_prompt,
        include_choices=not args.no_choices,
        question_field=args.question_field,
    )
    merged_stats = _write_merged_graphs(
        ekg_dir=ekg_dir,
        ekg_glob=args.ekg_glob,
        chunk_lookup=chunk_lookup,
        output_path=merged_jsonl,
    )
    dataset_stats = _write_dataset_json(
        qa_groups=qa_groups,
        merged_doc_ids=set(merged_stats["merged_doc_ids"]),
        merged_jsonl_path=merged_jsonl,
        output_path=dataset_json,
        dataset_name=args.dataset_name,
        project_root=project_root,
    )

    summary = {
        "created_at_utc": _utc_now(),
        "inputs": {
            "ekg_dir": str(ekg_dir),
            "chunks_jsonl": str(chunks_jsonl),
            "qa_jsonl": str(qa_jsonl),
        },
        "outputs": {
            "output_dir": str(output_dir),
            "merged_jsonl": str(merged_jsonl),
            "dataset_json": str(dataset_json),
            "summary_json": str(summary_json),
        },
        "settings": {
            "dataset_name": args.dataset_name,
            "ekg_glob": args.ekg_glob,
            "include_prompt": not args.no_prompt,
            "include_choices": not args.no_choices,
            "question_field": args.question_field,
        },
        "counts": {
            "chunk_docs": len(chunk_lookup),
            "qa_docs": len(qa_groups),
            "qa_rows": sum(len(rows) for rows in qa_groups.values()),
            "merged_docs_written": merged_stats["written_docs"],
            "merged_docs_skipped_missing_chunks": merged_stats["skipped_missing_chunks"],
            "ekg_files_seen": merged_stats["ekg_files_seen"],
            "dataset_docs_written": dataset_stats["written_docs"],
            "dataset_faqs_written": dataset_stats["written_faqs"],
            "dataset_docs_skipped_missing_graph": dataset_stats["skipped_missing_graph"],
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.run_inference:
        _run_inference(
            project_root=project_root,
            dataset_json=dataset_json,
            chunks_jsonl=chunks_jsonl,
            args=args,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
