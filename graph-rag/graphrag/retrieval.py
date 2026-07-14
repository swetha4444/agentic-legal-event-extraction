from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .models import RetrievalBundle, RetrievalHit
from .store import GraphStore


TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


@dataclass
class RetrievalConfig:
    seed_limit: int = 12
    top_k: int = 8
    graph_hops: int = 2
    lexical_weight: float = 0.35
    graph_weight: float = 0.25
    entity_weight: float = 0.15
    temporal_weight: float = 0.10
    evidence_weight: float = 0.10
    supporting_fact_weight: float = 0.03
    structure_weight: float = 0.12
    intent_weight: float = 0.18


def _tokenize(text: str) -> set[str]:
    return {tok.lower() for tok in TOKEN_RE.findall(text or "")}


def _token_overlap(a: str, b: str) -> float:
    aa = _tokenize(a)
    bb = _tokenize(b)
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / len(aa | bb)


def _temporal_match(question: str, time_raw: str, time_normalized: str) -> float:
    q = question.lower()
    if any(marker in q for marker in ("when", "before", "after", "timeline", "date", "time")):
        hay = f"{time_raw} {time_normalized}".strip().lower()
        return 1.0 if hay else 0.15
    return 0.0


def _evidence_richness(row: Any) -> float:
    evidence = str(row["evidence_text"] or "")
    confidence = float(row["confidence"] or 0.0)
    snippet_bonus = min(len(evidence) / 240.0, 1.0)
    return 0.5 * snippet_bonus + 0.5 * confidence


def _structure_score(
    participant_rows: list[Any],
    neighbor_rows: list[Any],
    is_supporting_fact: bool,
) -> float:
    participant_bonus = min(len(participant_rows) / 3.0, 1.0)
    neighbor_bonus = min(len(neighbor_rows) / 4.0, 1.0)
    support_penalty = 0.2 if is_supporting_fact and not participant_rows and not neighbor_rows else 0.0
    return max(0.0, 0.55 * participant_bonus + 0.45 * neighbor_bonus - support_penalty)


def _intent_score(question: str, row: Any) -> float:
    q = question.lower()
    event_type = str(row["event_type"] or "").upper()
    score = 0.0

    if any(term in q for term in ("terminate", "terminated", "termination", "fired", "discharged", "let go")):
        if event_type in {"TERMINATION", "DISCHARGE"}:
            score += 1.0
        elif event_type in {"PREGNANCYDISCLOSURE", "PARENTALLEAVEDISCUSSION", "SUPERVISORCHANGE", "MEETINGRESCHEDULE"}:
            score += 0.45
        elif event_type == "SUPPORTING_FACT":
            score += 0.2

    if any(term in q for term in ("why", "cause", "because", "led to", "lead to")):
        if event_type in {"PREGNANCYDISCLOSURE", "PARENTALLEAVEDISCUSSION", "SUPERVISORCHANGE", "TASKREQUEST", "TASKASSIGNMENT", "TOPICABANDONMENT"}:
            score += 0.75
        elif event_type == "SUPPORTING_FACT":
            score += 0.2

    if any(term in q for term in ("pregnan", "maternity", "parental leave", "leave")):
        if event_type in {"PREGNANCYDISCLOSURE", "PARENTALLEAVEDISCUSSION", "SHOCKREACTION", "TOPICABANDONMENT"}:
            score += 0.8

    return min(score, 1.0)


def _rank_norm(index: int) -> float:
    return 1.0 / (1.0 + index)


def retrieve(
    store: GraphStore,
    question: str,
    *,
    config: RetrievalConfig | None = None,
) -> RetrievalBundle:
    cfg = config or RetrievalConfig()
    query = question.strip()
    with store.connect() as conn:
        seed_event_rows = store.search_events(conn, query, limit=cfg.seed_limit)
        seed_chunk_rows = store.search_chunks(conn, query, limit=max(4, cfg.seed_limit // 2))
        seed_entity_rows = store.search_entities(conn, query, limit=max(4, cfg.seed_limit // 2))

        event_scores: dict[tuple[str, str], dict[str, Any]] = {}
        entity_names_by_doc: dict[str, set[str]] = defaultdict(set)
        for row in seed_entity_rows:
            entity_names_by_doc[str(row["doc_id"])].add(str(row["name"]))

        for idx, row in enumerate(seed_event_rows):
            key = (str(row["doc_id"]), str(row["event_id"]))
            lexical = max(_rank_norm(idx), _token_overlap(query, str(row["lexical_text"] or "")))
            event_scores[key] = {
                "row": row,
                "lexical_score": lexical,
                "graph_score": 1.0,
                "entity_score": 0.0,
                "temporal_score": _temporal_match(question, str(row["time_raw"]), str(row["time_normalized"])),
                "evidence_score": _evidence_richness(row),
                "supporting_bonus": 1.0 if int(row["is_supporting_fact"]) else 0.0,
                "intent_score": _intent_score(question, row),
                "seed": True,
            }

        for row in seed_chunk_rows:
            doc_id = str(row["doc_id"])
            text = str(row["text"] or "")
            overlap = _token_overlap(question, text)
            if overlap <= 0:
                continue
            for event in store.search_events(conn, text[:200], limit=6):
                if str(event["doc_id"]) != doc_id:
                    continue
                key = (doc_id, str(event["event_id"]))
                current = event_scores.get(key)
                if current is None:
                    current = {
                        "row": event,
                        "lexical_score": overlap * 0.7,
                        "graph_score": 0.3,
                        "entity_score": 0.0,
                        "temporal_score": _temporal_match(question, str(event["time_raw"]), str(event["time_normalized"])),
                        "evidence_score": _evidence_richness(event),
                        "supporting_bonus": 1.0 if int(event["is_supporting_fact"]) else 0.0,
                        "intent_score": _intent_score(question, event),
                        "seed": False,
                    }
                    event_scores[key] = current
                else:
                    current["lexical_score"] = max(current["lexical_score"], overlap * 0.7)

        frontier_by_doc: dict[str, set[str]] = defaultdict(set)
        for (doc_id, event_id), feature_row in event_scores.items():
            frontier_by_doc[doc_id].add(event_id)
            participant_rows = store.get_event_participants(conn, doc_id, event_id)
            neighbor_rows = store.get_neighbor_events(conn, doc_id, [event_id])
            q_tokens = _tokenize(question)
            participant_name_tokens: set[str] = set()
            for participant in participant_rows:
                participant_name_tokens |= _tokenize(str(participant["mention_text"] or ""))
            entity_name_tokens: set[str] = set()
            for entity_name in entity_names_by_doc.get(doc_id, set()):
                entity_name_tokens |= _tokenize(entity_name)
            feature_row["entity_score"] = (
                len((participant_name_tokens | entity_name_tokens) & q_tokens) / max(len(q_tokens), 1)
                if q_tokens
                else 0.0
            )
            feature_row["structure_score"] = _structure_score(
                participant_rows=participant_rows,
                neighbor_rows=neighbor_rows,
                is_supporting_fact=bool(int(feature_row["row"]["is_supporting_fact"])),
            )

        visited_by_doc: dict[str, dict[str, int]] = defaultdict(dict)
        for doc_id, seeds in frontier_by_doc.items():
            for seed in seeds:
                visited_by_doc[doc_id][seed] = 0

        for _ in range(cfg.graph_hops):
            next_frontier: dict[str, set[str]] = defaultdict(set)
            for doc_id, event_ids in frontier_by_doc.items():
                neighbors = store.get_neighbor_events(conn, doc_id, sorted(event_ids))
                for neighbor in neighbors:
                    event_id = str(neighbor["event_id"])
                    if event_id in visited_by_doc[doc_id]:
                        continue
                    depth = min(visited_by_doc[doc_id].values()) + 1 if visited_by_doc[doc_id] else 1
                    visited_by_doc[doc_id][event_id] = depth
                    key = (doc_id, event_id)
                    graph_score = 1.0 / (1.0 + depth)
                    event_scores.setdefault(
                        key,
                        {
                            "row": neighbor,
                            "lexical_score": _token_overlap(question, str(neighbor["lexical_text"] or "")),
                            "graph_score": graph_score,
                            "entity_score": 0.0,
                            "temporal_score": _temporal_match(question, str(neighbor["time_raw"]), str(neighbor["time_normalized"])),
                            "evidence_score": _evidence_richness(neighbor),
                            "supporting_bonus": 1.0 if int(neighbor["is_supporting_fact"]) else 0.0,
                            "structure_score": 0.0,
                            "intent_score": _intent_score(question, neighbor),
                            "seed": False,
                        },
                    )
                    event_scores[key]["graph_score"] = max(event_scores[key]["graph_score"], graph_score)
                    next_frontier[doc_id].add(event_id)
            frontier_by_doc = next_frontier

        hits: list[RetrievalHit] = []
        for (doc_id, event_id), features in event_scores.items():
            row = features["row"]
            score = (
                cfg.lexical_weight * float(features["lexical_score"])
                + cfg.graph_weight * float(features["graph_score"])
                + cfg.entity_weight * float(features["entity_score"])
                + cfg.temporal_weight * float(features["temporal_score"])
                + cfg.evidence_weight * float(features["evidence_score"])
                + cfg.structure_weight * float(features.get("structure_score", 0.0))
                + cfg.intent_weight * float(features.get("intent_score", 0.0))
                + cfg.supporting_fact_weight * float(features["supporting_bonus"])
            )
            chunk_rows = store.get_chunks_for_events(conn, doc_id, [event_id])
            neighbor_rows = store.get_neighbor_events(conn, doc_id, [event_id])
            hits.append(
                RetrievalHit(
                    doc_id=doc_id,
                    event_id=event_id,
                    event_type=str(row["event_type"]),
                    text=str(row["evidence_text"] or row["lexical_text"] or ""),
                    score=score,
                    lexical_score=float(features["lexical_score"]),
                    graph_score=float(features["graph_score"]),
                    entity_score=float(features["entity_score"]),
                    temporal_score=float(features["temporal_score"]),
                    evidence_score=float(features["evidence_score"]),
                    is_supporting_fact=bool(int(row["is_supporting_fact"])),
                    neighbors=[str(n["event_id"]) for n in neighbor_rows[:6]],
                    chunk_ids=[str(ch["chunk_id"]) for ch in chunk_rows],
                )
            )

        hits.sort(key=lambda item: item.score, reverse=True)
        top_hits = hits[: cfg.top_k]
        top_doc_ids = {hit.doc_id for hit in top_hits}
        top_event_ids_by_doc: dict[str, list[str]] = defaultdict(list)
        for hit in top_hits:
            top_event_ids_by_doc[hit.doc_id].append(hit.event_id)

        entities: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        chunks: list[dict[str, Any]] = []
        with store.connect() as conn2:
            for doc_id in sorted(top_doc_ids):
                event_ids = top_event_ids_by_doc[doc_id]
                entities.extend(dict(row) for row in store.get_entities_for_events(conn2, doc_id, event_ids))
                edges.extend(dict(row) for row in store.get_edges_for_events(conn2, doc_id, event_ids))
                chunks.extend(dict(row) for row in store.get_chunks_for_events(conn2, doc_id, event_ids))

        return RetrievalBundle(
            question=question,
            events=top_hits,
            entities=entities,
            edges=edges,
            chunks=chunks,
            summary={
                "question": question,
                "seed_events": len(seed_event_rows),
                "seed_chunks": len(seed_chunk_rows),
                "seed_entities": len(seed_entity_rows),
                "candidate_events": len(hits),
                "returned_events": len(top_hits),
            },
        )
