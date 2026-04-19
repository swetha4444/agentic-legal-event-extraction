"""
Deterministic mini-graph merger.

Merges chunk-level graphs (from `graph` or `mini_graph`) into one doc-level graph:
- dedupes entities by normalized name
- dedupes events by type/verb/time/evidence/participants
- remaps participant and edge references to merged IDs
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json
import re


def _norm_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _ensure_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _clean_entity_kind(kind: Any) -> str:
    allowed = {"PERSON", "ORG", "AGENCY", "COURT", "UNKNOWN"}
    text = str(kind or "").upper()
    return text if text in allowed else "UNKNOWN"


def _clean_role(role: Any) -> str:
    allowed = {"PLAINTIFF", "DEFENDANT", "EMPLOYER", "SUPERVISOR", "AGENCY", "COURT", "UNKNOWN"}
    text = str(role or "").upper()
    return text if text in allowed else "UNKNOWN"


def _event_sentence_ids(event: dict[str, Any]) -> tuple[int, ...]:
    evidence_ids = []
    evidence = event.get("evidence") or {}
    for sid in _ensure_list(evidence.get("sentence_ids")):
        try:
            evidence_ids.append(int(sid))
        except (TypeError, ValueError):
            continue
    trigger = event.get("trigger") or {}
    trig_sid = trigger.get("sentence_id")
    if trig_sid is not None:
        try:
            evidence_ids.append(int(trig_sid))
        except (TypeError, ValueError):
            pass
    return tuple(sorted(set(evidence_ids)))


def _time_key(event: dict[str, Any]) -> tuple[str, str]:
    time_obj = event.get("time") or {}
    kind = str(time_obj.get("kind") or "UNKNOWN").upper()
    normalized = str(time_obj.get("normalized") or time_obj.get("value") or time_obj.get("raw_span") or "")
    return kind, _norm_text(normalized)


@dataclass
class MiniGraphMergeAgent:
    entity_prefix: str = "E"
    event_prefix: str = "EV"

    def merge_document(self, doc: dict[str, Any]) -> dict[str, Any]:
        merged_entities: list[dict[str, Any]] = []
        merged_events: list[dict[str, Any]] = []
        merged_temporal: list[dict[str, Any]] = []
        merged_causal: list[dict[str, Any]] = []
        source_event_rows = 0

        entity_key_to_id: dict[str, str] = {}
        event_key_to_id: dict[str, str] = {}

        def upsert_entity(entity: dict[str, Any], fallback_name: str) -> str:
            name = str(entity.get("name") or fallback_name or "").strip() or "UNKNOWN_ENTITY"
            key = _norm_text(name)
            if key in entity_key_to_id:
                return entity_key_to_id[key]
            eid = f"{self.entity_prefix}{len(merged_entities) + 1}"
            merged_entities.append(
                {
                    "entity_id": eid,
                    "name": name,
                    "kind": _clean_entity_kind(entity.get("kind")),
                    "canonical_role": _clean_role(entity.get("canonical_role")),
                    "aliases": [str(a) for a in _ensure_list(entity.get("aliases")) if str(a).strip()],
                }
            )
            entity_key_to_id[key] = eid
            return eid

        for chunk in _ensure_list(doc.get("chunks")):
            graph = chunk.get("graph") or chunk.get("mini_graph") or {}
            chunk_entity_map: dict[str, str] = {}
            chunk_event_map: dict[str, str] = {}

            # 1) Entities
            for raw_entity in _ensure_list(graph.get("entities")):
                if not isinstance(raw_entity, dict):
                    continue
                source_eid = str(raw_entity.get("entity_id") or "")
                fallback_name = source_eid or "UNKNOWN_ENTITY"
                merged_eid = upsert_entity(raw_entity, fallback_name=fallback_name)
                if source_eid:
                    chunk_entity_map[source_eid] = merged_eid

            # 2) Events
            for raw_event in _ensure_list(graph.get("events")):
                if not isinstance(raw_event, dict):
                    continue
                source_event_rows += 1
                participants = []
                participant_keys = []
                for p in _ensure_list(raw_event.get("participants")):
                    if not isinstance(p, dict):
                        continue
                    src_eid = str(p.get("entity_id") or "")
                    mention = p.get("mention") or {}
                    mention_text = ""
                    if isinstance(mention, dict):
                        mention_text = str(mention.get("span_text") or mention.get("text") or "")
                    merged_eid = chunk_entity_map.get(src_eid)
                    if not merged_eid and mention_text:
                        merged_eid = upsert_entity({"name": mention_text}, fallback_name=mention_text)
                    if not merged_eid:
                        continue
                    role = str(p.get("role") or "CONTEXT").upper()
                    participants.append(
                        {
                            "entity_id": merged_eid,
                            "role": role,
                            "mention": mention if isinstance(mention, dict) else {"span_text": str(mention or "")},
                        }
                    )
                    participant_keys.append((merged_eid, role))

                ev_type = str(raw_event.get("event_type") or "UNKNOWN_EVENT").upper()
                main_verb = str(raw_event.get("main_verb") or (raw_event.get("trigger") or {}).get("span_text") or "")
                sent_ids = _event_sentence_ids(raw_event)
                time_key = _time_key(raw_event)
                dedupe_key = json.dumps(
                    {
                        "event_type": ev_type,
                        "main_verb": _norm_text(main_verb),
                        "time": time_key,
                        "participants": sorted(participant_keys),
                        "sentence_ids": sent_ids,
                    },
                    sort_keys=True,
                )

                if dedupe_key in event_key_to_id:
                    merged_evid = event_key_to_id[dedupe_key]
                    existing = next((ev for ev in merged_events if ev.get("event_id") == merged_evid), None)
                    if isinstance(existing, dict):
                        refs = _ensure_list(existing.get("source_refs"))
                        refs.append(
                            {
                                "chunk_id": chunk.get("chunk_id"),
                                "source_event_id": str(raw_event.get("event_id") or ""),
                                "sentence_ids": list(sent_ids),
                            }
                        )
                        existing["source_refs"] = refs
                else:
                    merged_evid = f"{self.event_prefix}{len(merged_events) + 1}"
                    merged_event = {
                        "event_id": merged_evid,
                        "event_type": ev_type,
                        "main_verb": main_verb,
                        "trigger": raw_event.get("trigger") if isinstance(raw_event.get("trigger"), dict) else {},
                        "participants": participants,
                        "time": raw_event.get("time") if isinstance(raw_event.get("time"), dict) else {"kind": "UNKNOWN"},
                        "evidence": raw_event.get("evidence") if isinstance(raw_event.get("evidence"), dict) else {"sentence_ids": [], "snippets": []},
                        "confidence": float(raw_event.get("confidence", 0.0) or 0.0),
                        "source_refs": [
                            {
                                "chunk_id": chunk.get("chunk_id"),
                                "source_event_id": str(raw_event.get("event_id") or ""),
                                "sentence_ids": list(sent_ids),
                            }
                        ],
                    }
                    merged_events.append(merged_event)
                    event_key_to_id[dedupe_key] = merged_evid

                src_event_id = str(raw_event.get("event_id") or "")
                if src_event_id:
                    chunk_event_map[src_event_id] = merged_evid

            # 3) Edges (only if endpoints resolve)
            for edge_name, sink in (("temporal_edges", merged_temporal), ("causal_edges", merged_causal)):
                for edge in _ensure_list(graph.get(edge_name)):
                    if not isinstance(edge, dict):
                        continue
                    src = chunk_event_map.get(str(edge.get("from_event") or ""))
                    dst = chunk_event_map.get(str(edge.get("to_event") or ""))
                    if not src or not dst:
                        continue
                    sink.append(
                        {
                            "from_event": src,
                            "to_event": dst,
                            "relation": str(edge.get("relation") or ""),
                            "evidence": edge.get("evidence") if isinstance(edge.get("evidence"), dict) else {"sentence_ids": [], "snippets": []},
                            "confidence": float(edge.get("confidence", 0.0) or 0.0),
                        }
                    )

        return {
            "doc_id": doc.get("doc_id") or doc.get("case_id") or "doc",
            "num_source_chunks": len(_ensure_list(doc.get("chunks"))),
            "merged_graph": {
                "entities": merged_entities,
                "events": merged_events,
                "temporal_edges": merged_temporal,
                "causal_edges": merged_causal,
            },
            "merge_stats": {
                "num_entities": len(merged_entities),
                "num_events": len(merged_events),
                "num_temporal_edges": len(merged_temporal),
                "num_causal_edges": len(merged_causal),
                "source_event_rows": source_event_rows,
                "event_rows_collapsed": max(0, source_event_rows - len(merged_events)),
            },
        }

