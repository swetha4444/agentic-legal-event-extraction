from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .models import (
    ChunkRecord,
    EdgeRecord,
    EntityRecord,
    EventRecord,
    GraphDocument,
    ParticipantRecord,
)


def _iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _join_snippets(evidence: dict[str, Any] | None) -> str:
    if not isinstance(evidence, dict):
        return ""
    snippets = evidence.get("snippets") or []
    parts: list[str] = []
    for snippet in snippets:
        if isinstance(snippet, dict):
            text = _safe_text(snippet.get("text"))
        else:
            text = _safe_text(snippet)
        if text:
            parts.append(text)
    return " ".join(parts)


def _sentence_ids(evidence: dict[str, Any] | None) -> list[int]:
    if not isinstance(evidence, dict):
        return []
    out: list[int] = []
    for sid in evidence.get("sentence_ids") or []:
        try:
            out.append(int(sid))
        except (TypeError, ValueError):
            continue
    return sorted(set(out))


def _event_supporting_flag(
    event: dict[str, Any],
    edge_counts: dict[str, int],
) -> bool:
    event_id = _safe_text(event.get("event_id"))
    event_type = _safe_text(event.get("event_type")).upper()
    if event_type == "SUPPORTING_FACT":
        return True
    participants = event.get("participants") or []
    if participants:
        return False
    return edge_counts.get(event_id, 0) == 0


def _chunk_map(path: str | Path | None) -> dict[str, dict[str, ChunkRecord]]:
    if not path:
        return {}
    out: dict[str, dict[str, ChunkRecord]] = {}
    for row in _iter_jsonl(path):
        doc_id = _safe_text(row.get("doc_id"))
        if not doc_id:
            continue
        chunks: dict[str, ChunkRecord] = {}
        for chunk in row.get("chunks") or []:
            chunk_id = _safe_text(chunk.get("chunk_id"))
            if not chunk_id:
                continue
            chunks[chunk_id] = ChunkRecord(
                chunk_id=chunk_id,
                text=_safe_text(chunk.get("text")),
                theme=_safe_text(chunk.get("theme")),
                start_sentence_id=chunk.get("start_sentence_id"),
                end_sentence_id=chunk.get("end_sentence_id"),
            )
        out[doc_id] = chunks
    return out


def load_graph_documents(
    merged_jsonl: str | Path,
    chunk_jsonl: str | Path | None = None,
) -> list[GraphDocument]:
    chunk_lookup = _chunk_map(chunk_jsonl)
    docs: list[GraphDocument] = []

    for row in _iter_jsonl(merged_jsonl):
        doc_id = _safe_text(row.get("doc_id"))
        merged_graph = row.get("merged_graph") or {}
        entities = merged_graph.get("entities") or []
        events = merged_graph.get("events") or []
        temporal_edges = merged_graph.get("temporal_edges") or []
        causal_edges = merged_graph.get("causal_edges") or []

        edge_counts: dict[str, int] = {}
        for edge in list(temporal_edges) + list(causal_edges):
            a = _safe_text(edge.get("from_event"))
            b = _safe_text(edge.get("to_event"))
            if a:
                edge_counts[a] = edge_counts.get(a, 0) + 1
            if b:
                edge_counts[b] = edge_counts.get(b, 0) + 1

        doc = GraphDocument(
            doc_id=doc_id,
            chunks=chunk_lookup.get(doc_id, {}),
            metadata={
                "num_source_chunks": row.get("num_source_chunks"),
                "merge_stats": row.get("merge_stats") or {},
                "hybrid_merge": row.get("hybrid_merge") or {},
            },
        )

        for entity in entities:
            entity_id = _safe_text(entity.get("entity_id"))
            if not entity_id:
                continue
            doc.entities[entity_id] = EntityRecord(
                entity_id=entity_id,
                name=_safe_text(entity.get("name")),
                kind=_safe_text(entity.get("kind")) or "UNKNOWN",
                canonical_role=_safe_text(entity.get("canonical_role")) or "UNKNOWN",
                aliases=[_safe_text(alias) for alias in entity.get("aliases") or [] if _safe_text(alias)],
            )

        for event in events:
            event_id = _safe_text(event.get("event_id"))
            if not event_id:
                continue
            trigger = event.get("trigger") or {}
            time_obj = event.get("time") or {}
            participants: list[ParticipantRecord] = []
            for participant in event.get("participants") or []:
                if not isinstance(participant, dict):
                    continue
                mention = participant.get("mention") or {}
                mention_text = _safe_text(
                    mention.get("span_text") if isinstance(mention, dict) else mention
                )
                participants.append(
                    ParticipantRecord(
                        entity_id=_safe_text(participant.get("entity_id")),
                        role=_safe_text(participant.get("role")),
                        mention_text=mention_text,
                    )
                )
            doc.events[event_id] = EventRecord(
                event_id=event_id,
                event_type=_safe_text(event.get("event_type")) or "UNKNOWN",
                main_verb=_safe_text(event.get("main_verb")),
                trigger_text=_safe_text(trigger.get("span_text")),
                time_kind=_safe_text(time_obj.get("kind")) or "UNKNOWN",
                time_raw=_safe_text(time_obj.get("raw_span")),
                time_normalized=json.dumps(time_obj.get("normalized"), ensure_ascii=False)
                if isinstance(time_obj.get("normalized"), (list, dict))
                else _safe_text(time_obj.get("normalized")),
                confidence=float(event.get("confidence") or 0.0),
                evidence_text=_join_snippets(event.get("evidence")),
                evidence_sentence_ids=_sentence_ids(event.get("evidence")),
                participants=participants,
                source_refs=list(event.get("source_refs") or []),
                is_supporting_fact=_event_supporting_flag(event, edge_counts=edge_counts),
            )

        for raw_edge in temporal_edges:
            doc.temporal_edges.append(
                EdgeRecord(
                    edge_kind="temporal",
                    from_event=_safe_text(raw_edge.get("from_event")),
                    to_event=_safe_text(raw_edge.get("to_event")),
                    relation=_safe_text(raw_edge.get("relation")),
                    confidence=float(raw_edge.get("confidence") or 0.0),
                    evidence_text=_join_snippets(raw_edge.get("evidence")),
                    evidence_sentence_ids=_sentence_ids(raw_edge.get("evidence")),
                )
            )
        for raw_edge in causal_edges:
            doc.causal_edges.append(
                EdgeRecord(
                    edge_kind="causal",
                    from_event=_safe_text(raw_edge.get("from_event")),
                    to_event=_safe_text(raw_edge.get("to_event")),
                    relation=_safe_text(raw_edge.get("relation")),
                    confidence=float(raw_edge.get("confidence") or 0.0),
                    evidence_text=_join_snippets(raw_edge.get("evidence")),
                    evidence_sentence_ids=_sentence_ids(raw_edge.get("evidence")),
                )
            )

        docs.append(doc)

    return docs
