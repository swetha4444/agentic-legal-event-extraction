from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChunkRecord:
    chunk_id: str
    text: str
    theme: str = ""
    start_sentence_id: int | None = None
    end_sentence_id: int | None = None


@dataclass
class EntityRecord:
    entity_id: str
    name: str
    kind: str = "UNKNOWN"
    canonical_role: str = "UNKNOWN"
    aliases: list[str] = field(default_factory=list)


@dataclass
class ParticipantRecord:
    entity_id: str
    role: str
    mention_text: str = ""


@dataclass
class EventRecord:
    event_id: str
    event_type: str
    main_verb: str
    trigger_text: str = ""
    time_kind: str = "UNKNOWN"
    time_raw: str = ""
    time_normalized: str = ""
    confidence: float = 0.0
    evidence_text: str = ""
    evidence_sentence_ids: list[int] = field(default_factory=list)
    participants: list[ParticipantRecord] = field(default_factory=list)
    source_refs: list[dict[str, Any]] = field(default_factory=list)
    is_supporting_fact: bool = False


@dataclass
class EdgeRecord:
    edge_kind: str
    from_event: str
    to_event: str
    relation: str
    confidence: float = 0.0
    evidence_text: str = ""
    evidence_sentence_ids: list[int] = field(default_factory=list)


@dataclass
class GraphDocument:
    doc_id: str
    entities: dict[str, EntityRecord] = field(default_factory=dict)
    events: dict[str, EventRecord] = field(default_factory=dict)
    chunks: dict[str, ChunkRecord] = field(default_factory=dict)
    temporal_edges: list[EdgeRecord] = field(default_factory=list)
    causal_edges: list[EdgeRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalHit:
    doc_id: str
    event_id: str
    event_type: str
    text: str
    score: float
    lexical_score: float
    graph_score: float
    entity_score: float
    temporal_score: float
    evidence_score: float
    is_supporting_fact: bool
    neighbors: list[str] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)


@dataclass
class RetrievalBundle:
    question: str
    events: list[RetrievalHit]
    entities: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    chunks: list[dict[str, Any]]
    summary: dict[str, Any]
