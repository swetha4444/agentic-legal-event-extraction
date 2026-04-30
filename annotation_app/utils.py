"""
utils.py — Data loading and atomic JSONL updates for the simplified 
human annotation app.

This version supports direct updates to the JSONL files, making it 
ideal for shared hosting (e.g., Hostinger).
"""

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Field-name aliases
# ---------------------------------------------------------------------------
CHUNK_TEXT_CANDIDATES = ["text", "chunk_text", "chunk", "content"]
CHUNK_ID_CANDIDATES = ["chunk_id", "id", "chunk_index"]
EVENTS_CANDIDATES = ["extracted_events", "events", "mini_graph", "minigraph",
                      "event_graph", "graph"]
ENTITIES_CANDIDATES = ["extracted_entities", "entities"]
TITLE_CANDIDATES = ["case_id", "case_name", "title", "doc_id", "document_id", "filename", "source"]

ERROR_TAGS = [
    "hallucination",
    "missing_event",
    "unsupported_relation",
    "wrong_causality",
    "wrong_temporal_order",
    "entity_confusion",
    "too_vague",
    "other",
]

ALIGNMENT_LABELS = ["aligned", "partially_aligned", "misaligned"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_field(record: dict, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in record and record[c] is not None:
            return c
    return None

def get_record_title(record: dict, line_no: int) -> str:
    field = detect_field(record, TITLE_CANDIDATES)
    if field:
        return str(record[field])
    return f"doc_{line_no}"


def _make_chunk_key(doc_title: str, chunk_id: str) -> str:
    return f"{doc_title}::{chunk_id}"


def get_output_path(input_path: str) -> Path:
    """Return the path for the annotated version of the input file."""
    p = Path(input_path)
    return p.parent / f"human_annotated_{p.name}"


# ---------------------------------------------------------------------------
# Loading data
# ---------------------------------------------------------------------------

def load_data(input_path: str) -> list[dict]:
    """
    Load data from the output path if it exists, otherwise from the input path.
    Flattens records into a list of annotatable chunks.
    """
    out_path = get_output_path(input_path)
    source_path = out_path if out_path.exists() else Path(input_path)
    
    chunks_out: list[dict] = []
    if not source_path.exists():
        return []

    with open(source_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            doc_title = get_record_title(record, line_no)
            events_field = detect_field(record, EVENTS_CANDIDATES)
            entities_field = detect_field(record, ENTITIES_CANDIDATES)
            all_events = record.get(events_field, []) if events_field else []
            all_entities = record.get(entities_field, []) if entities_field else []

            events_by_chunk: dict[str, list] = {}
            for ev in all_events:
                cid = ev.get("_chunk_id", "unknown")
                events_by_chunk.setdefault(cid, []).append(ev)

            raw_chunks = record.get("chunks", [])
            for chunk in raw_chunks:
                cid_field = detect_field(chunk, CHUNK_ID_CANDIDATES) or "chunk_id"
                text_field = detect_field(chunk, CHUNK_TEXT_CANDIDATES) or "text"
                chunk_id = chunk.get(cid_field, f"chunk_{len(chunks_out)}")
                
                chunk_key = _make_chunk_key(doc_title, chunk_id)
                
                chunks_out.append({
                    "chunk_key": chunk_key,
                    "doc_title": doc_title,
                    "chunk_id": chunk_id,
                    "chunk_theme": chunk.get("theme", ""),
                    "chunk_text": chunk.get(text_field, ""),
                    "events": events_by_chunk.get(chunk_id, []),
                    "entities": all_entities,
                    "annotation": chunk.get("annotation", {}), # Preload if exists
                    "metadata": {
                        "start_sentence_id": chunk.get("start_sentence_id"),
                        "end_sentence_id": chunk.get("end_sentence_id"),
                    },
                })

    return chunks_out


# ---------------------------------------------------------------------------
# Atomic JSONL Update
# ---------------------------------------------------------------------------

def update_jsonl_with_annotation(input_path: str, chunk_key: str, annotation: dict):
    """
    Update the output JSONL file with a new annotation for a specific chunk.
    This reads the whole file, updates the matching record, and writes to a temp file
    before replacing the original (safely).
    """
    in_path = Path(input_path)
    out_path = get_output_path(input_path)
    
    # If output doesn't exist yet, start from input
    src_path = out_path if out_path.exists() else in_path
    if not src_path.exists():
        raise FileNotFoundError(f"Source file not found: {src_path}")

    fd, temp_path = tempfile.mkstemp(dir=out_path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as tmp_f:
            with open(src_path, 'r', encoding='utf-8') as src_f:
                for line_no, line in enumerate(src_f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    
                    doc_title = get_record_title(record, line_no)
                    
                    # Check if this record contains the chunk we are updating
                    chunks = record.get("chunks", [])
                    updated = False
                    for chunk in chunks:
                        cid_field = detect_field(chunk, CHUNK_ID_CANDIDATES) or "chunk_id"
                        chunk_id = chunk.get(cid_field, "unknown")
                        if _make_chunk_key(doc_title, chunk_id) == chunk_key:
                            chunk["annotation"] = annotation
                            updated = True
                    
                    tmp_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        # Atomically replace
        shutil.move(temp_path, out_path)
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e


def make_annotation_dict(annotator: str, summary: str, label: str, 
                         reasoning: str, tags: list[str]) -> dict:
    return {
        "human_summary": summary,
        "alignment_label": label,
        "annotation_reasoning": reasoning,
        "error_tags": tags,
        "annotation_completed": True,
        "annotated_by": annotator,
        "annotation_timestamp": datetime.now(timezone.utc).isoformat(),
        "annotation_status": "annotated",
    }


def get_progress_stats(chunks: list[dict]) -> dict:
    total = len(chunks)
    completed_list = [c for c in chunks if c.get("annotation", {}).get("annotation_completed")]
    completed = len(completed_list)
    
    by_label = {}
    by_annotator = {}
    for c in completed_list:
        ann = c["annotation"]
        l = ann.get("alignment_label", "unknown")
        by_label[l] = by_label.get(l, 0) + 1
        w = ann.get("annotated_by", "unknown")
        by_annotator[w] = by_annotator.get(w, 0) + 1
        
    return {
        "total": total,
        "completed": completed,
        "remaining": total - completed,
        "pct": round(100 * completed / total, 1) if total else 0,
        "by_label": by_label,
        "by_annotator": by_annotator,
    }
