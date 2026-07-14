from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from .loader import load_graph_documents
from .models import GraphDocument


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=OFF;

CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    metadata_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    doc_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    theme TEXT NOT NULL,
    text TEXT NOT NULL,
    start_sentence_id INTEGER,
    end_sentence_id INTEGER,
    PRIMARY KEY (doc_id, chunk_id)
);

CREATE TABLE IF NOT EXISTS entities (
    doc_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    canonical_role TEXT NOT NULL,
    aliases_json TEXT NOT NULL,
    PRIMARY KEY (doc_id, entity_id)
);

CREATE TABLE IF NOT EXISTS events (
    doc_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    main_verb TEXT NOT NULL,
    trigger_text TEXT NOT NULL,
    time_kind TEXT NOT NULL,
    time_raw TEXT NOT NULL,
    time_normalized TEXT NOT NULL,
    confidence REAL NOT NULL,
    evidence_text TEXT NOT NULL,
    evidence_sentence_ids_json TEXT NOT NULL,
    source_refs_json TEXT NOT NULL,
    is_supporting_fact INTEGER NOT NULL,
    lexical_text TEXT NOT NULL,
    PRIMARY KEY (doc_id, event_id)
);

CREATE TABLE IF NOT EXISTS event_participants (
    doc_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    role TEXT NOT NULL,
    mention_text TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS event_edges (
    doc_id TEXT NOT NULL,
    edge_kind TEXT NOT NULL,
    from_event TEXT NOT NULL,
    to_event TEXT NOT NULL,
    relation TEXT NOT NULL,
    confidence REAL NOT NULL,
    evidence_text TEXT NOT NULL,
    evidence_sentence_ids_json TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS event_fts USING fts5(
    doc_id UNINDEXED,
    event_id UNINDEXED,
    lexical_text
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
    doc_id UNINDEXED,
    chunk_id UNINDEXED,
    text,
    theme
);

CREATE VIRTUAL TABLE IF NOT EXISTS entity_fts USING fts5(
    doc_id UNINDEXED,
    entity_id UNINDEXED,
    name,
    aliases
);
"""

TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")

EVENT_TYPE_SYNONYMS = {
    "TERMINATION": ["terminated", "termination", "fired", "dismissed", "let go", "discharged"],
    "DISCHARGE": ["terminated", "termination", "fired", "dismissed", "let go", "discharged"],
    "PREGNANCYDISCLOSURE": ["pregnancy", "pregnant", "disclosed pregnancy", "told supervisors"],
    "PARENTALLEAVEDISCUSSION": ["parental leave", "maternity leave", "leave policy", "pregnancy leave"],
    "SUPERVISORCHANGE": ["new boss", "new supervisor", "manager change"],
    "TASKREQUEST": ["request", "project list", "work update"],
    "TASKASSIGNMENT": ["assigned", "task", "project", "responsibility"],
    "QUESTION": ["asked", "questioned"],
    "STATEMENT": ["said", "stated", "claimed", "responded"],
}


def _connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _event_lexical_text(doc: GraphDocument, event_id: str) -> str:
    event = doc.events[event_id]
    participant_bits: list[str] = []
    for participant in event.participants:
        entity = doc.entities.get(participant.entity_id)
        if entity:
            participant_bits.append(entity.name)
        if participant.role:
            participant_bits.append(participant.role)
        if participant.mention_text:
            participant_bits.append(participant.mention_text)
    bits = [
        event.event_type,
        event.main_verb,
        event.trigger_text,
        event.time_kind,
        event.time_raw,
        event.time_normalized,
        event.evidence_text,
        " ".join(participant_bits),
        " ".join(EVENT_TYPE_SYNONYMS.get(event.event_type.upper(), [])),
    ]
    return " | ".join(bit for bit in bits if bit)


def _fts_query(text: str) -> str:
    tokens = [tok.lower() for tok in TOKEN_RE.findall(text or "") if len(tok) >= 3]
    if not tokens:
        tokens = [tok.lower() for tok in TOKEN_RE.findall(text or "")]
    if not tokens:
        return ""
    unique_tokens: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        unique_tokens.append(f"{token}*")
    return " OR ".join(unique_tokens)


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def reset_database(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DELETE FROM documents;
        DELETE FROM chunks;
        DELETE FROM entities;
        DELETE FROM events;
        DELETE FROM event_participants;
        DELETE FROM event_edges;
        DELETE FROM event_fts;
        DELETE FROM chunk_fts;
        DELETE FROM entity_fts;
        """
    )
    conn.commit()


def index_documents(
    conn: sqlite3.Connection,
    docs: Iterable[GraphDocument],
) -> dict[str, int]:
    doc_count = 0
    chunk_count = 0
    entity_count = 0
    event_count = 0
    edge_count = 0

    for doc in docs:
        doc_count += 1
        conn.execute(
            "INSERT INTO documents(doc_id, metadata_json) VALUES (?, ?)",
            (doc.doc_id, json.dumps(doc.metadata, ensure_ascii=False)),
        )

        for chunk in doc.chunks.values():
            chunk_count += 1
            conn.execute(
                """
                INSERT INTO chunks(doc_id, chunk_id, theme, text, start_sentence_id, end_sentence_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    doc.doc_id,
                    chunk.chunk_id,
                    chunk.theme,
                    chunk.text,
                    chunk.start_sentence_id,
                    chunk.end_sentence_id,
                ),
            )
            conn.execute(
                "INSERT INTO chunk_fts(doc_id, chunk_id, text, theme) VALUES (?, ?, ?, ?)",
                (doc.doc_id, chunk.chunk_id, chunk.text, chunk.theme),
            )

        for entity in doc.entities.values():
            entity_count += 1
            aliases_json = json.dumps(entity.aliases, ensure_ascii=False)
            conn.execute(
                """
                INSERT INTO entities(doc_id, entity_id, name, kind, canonical_role, aliases_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    doc.doc_id,
                    entity.entity_id,
                    entity.name,
                    entity.kind,
                    entity.canonical_role,
                    aliases_json,
                ),
            )
            conn.execute(
                "INSERT INTO entity_fts(doc_id, entity_id, name, aliases) VALUES (?, ?, ?, ?)",
                (doc.doc_id, entity.entity_id, entity.name, " ".join(entity.aliases)),
            )

        for event in doc.events.values():
            event_count += 1
            lexical_text = _event_lexical_text(doc, event.event_id)
            conn.execute(
                """
                INSERT INTO events(
                    doc_id, event_id, event_type, main_verb, trigger_text,
                    time_kind, time_raw, time_normalized, confidence,
                    evidence_text, evidence_sentence_ids_json, source_refs_json,
                    is_supporting_fact, lexical_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc.doc_id,
                    event.event_id,
                    event.event_type,
                    event.main_verb,
                    event.trigger_text,
                    event.time_kind,
                    event.time_raw,
                    event.time_normalized,
                    event.confidence,
                    event.evidence_text,
                    json.dumps(event.evidence_sentence_ids),
                    json.dumps(event.source_refs, ensure_ascii=False),
                    int(event.is_supporting_fact),
                    lexical_text,
                ),
            )
            conn.execute(
                "INSERT INTO event_fts(doc_id, event_id, lexical_text) VALUES (?, ?, ?)",
                (doc.doc_id, event.event_id, lexical_text),
            )
            for participant in event.participants:
                conn.execute(
                    """
                    INSERT INTO event_participants(doc_id, event_id, entity_id, role, mention_text)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        doc.doc_id,
                        event.event_id,
                        participant.entity_id,
                        participant.role,
                        participant.mention_text,
                    ),
                )

        for edge in list(doc.temporal_edges) + list(doc.causal_edges):
            edge_count += 1
            conn.execute(
                """
                INSERT INTO event_edges(
                    doc_id, edge_kind, from_event, to_event, relation,
                    confidence, evidence_text, evidence_sentence_ids_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc.doc_id,
                    edge.edge_kind,
                    edge.from_event,
                    edge.to_event,
                    edge.relation,
                    edge.confidence,
                    edge.evidence_text,
                    json.dumps(edge.evidence_sentence_ids),
                ),
            )

    conn.commit()
    return {
        "documents": doc_count,
        "chunks": chunk_count,
        "entities": entity_count,
        "events": event_count,
        "edges": edge_count,
    }


def build_database(
    db_path: str | Path,
    merged_jsonl: str | Path,
    chunk_jsonl: str | Path | None = None,
    *,
    reset: bool = True,
) -> dict[str, int]:
    docs = load_graph_documents(merged_jsonl=merged_jsonl, chunk_jsonl=chunk_jsonl)
    conn = _connect(db_path)
    try:
        create_schema(conn)
        if reset:
            reset_database(conn)
        return index_documents(conn, docs)
    finally:
        conn.close()


class GraphStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)

    def connect(self) -> sqlite3.Connection:
        return _connect(self.db_path)

    def search_events(self, conn: sqlite3.Connection, query: str, limit: int = 20) -> list[sqlite3.Row]:
        fts_query = _fts_query(query)
        if not fts_query:
            return []
        sql = """
        SELECT e.*, bm25(event_fts) AS rank
        FROM event_fts
        JOIN events e
          ON e.doc_id = event_fts.doc_id AND e.event_id = event_fts.event_id
        WHERE event_fts MATCH ?
        ORDER BY bm25(event_fts)
        LIMIT ?
        """
        try:
            return list(conn.execute(sql, (fts_query, limit)))
        except sqlite3.OperationalError:
            return []

    def search_chunks(self, conn: sqlite3.Connection, query: str, limit: int = 10) -> list[sqlite3.Row]:
        fts_query = _fts_query(query)
        if not fts_query:
            return []
        sql = """
        SELECT c.*, bm25(chunk_fts) AS rank
        FROM chunk_fts
        JOIN chunks c
          ON c.doc_id = chunk_fts.doc_id AND c.chunk_id = chunk_fts.chunk_id
        WHERE chunk_fts MATCH ?
        ORDER BY bm25(chunk_fts)
        LIMIT ?
        """
        try:
            return list(conn.execute(sql, (fts_query, limit)))
        except sqlite3.OperationalError:
            return []

    def search_entities(self, conn: sqlite3.Connection, query: str, limit: int = 10) -> list[sqlite3.Row]:
        fts_query = _fts_query(query)
        if not fts_query:
            return []
        sql = """
        SELECT e.*, bm25(entity_fts) AS rank
        FROM entity_fts
        JOIN entities e
          ON e.doc_id = entity_fts.doc_id AND e.entity_id = entity_fts.entity_id
        WHERE entity_fts MATCH ?
        ORDER BY bm25(entity_fts)
        LIMIT ?
        """
        try:
            return list(conn.execute(sql, (fts_query, limit)))
        except sqlite3.OperationalError:
            return []

    def get_event(self, conn: sqlite3.Connection, doc_id: str, event_id: str) -> sqlite3.Row | None:
        return conn.execute(
            "SELECT * FROM events WHERE doc_id = ? AND event_id = ?",
            (doc_id, event_id),
        ).fetchone()

    def get_event_participants(self, conn: sqlite3.Connection, doc_id: str, event_id: str) -> list[sqlite3.Row]:
        return list(
            conn.execute(
                "SELECT * FROM event_participants WHERE doc_id = ? AND event_id = ?",
                (doc_id, event_id),
            )
        )

    def get_edges_for_events(
        self,
        conn: sqlite3.Connection,
        doc_id: str,
        event_ids: list[str],
    ) -> list[sqlite3.Row]:
        if not event_ids:
            return []
        placeholders = ", ".join("?" for _ in event_ids)
        sql = f"""
        SELECT * FROM event_edges
        WHERE doc_id = ?
          AND (from_event IN ({placeholders}) OR to_event IN ({placeholders}))
        """
        return list(conn.execute(sql, [doc_id, *event_ids, *event_ids]))

    def get_neighbor_events(
        self,
        conn: sqlite3.Connection,
        doc_id: str,
        event_ids: list[str],
    ) -> list[sqlite3.Row]:
        edges = self.get_edges_for_events(conn, doc_id, event_ids)
        neighbor_ids: set[str] = set()
        for edge in edges:
            neighbor_ids.add(str(edge["from_event"]))
            neighbor_ids.add(str(edge["to_event"]))
        neighbor_ids.difference_update(event_ids)
        if not neighbor_ids:
            return []
        placeholders = ", ".join("?" for _ in neighbor_ids)
        sql = f"SELECT * FROM events WHERE doc_id = ? AND event_id IN ({placeholders})"
        return list(conn.execute(sql, [doc_id, *sorted(neighbor_ids)]))

    def get_chunks_for_events(
        self,
        conn: sqlite3.Connection,
        doc_id: str,
        event_ids: list[str],
    ) -> list[sqlite3.Row]:
        if not event_ids:
            return []
        placeholders = ", ".join("?" for _ in event_ids)
        sql = f"""
        SELECT DISTINCT c.*
        FROM events e
        JOIN json_each(e.source_refs_json) src
        JOIN chunks c
          ON c.doc_id = e.doc_id
         AND c.chunk_id = json_extract(src.value, '$.chunk_id')
        WHERE e.doc_id = ?
          AND e.event_id IN ({placeholders})
        """
        try:
            return list(conn.execute(sql, [doc_id, *event_ids]))
        except sqlite3.OperationalError:
            return []

    def get_entities_for_events(
        self,
        conn: sqlite3.Connection,
        doc_id: str,
        event_ids: list[str],
    ) -> list[sqlite3.Row]:
        if not event_ids:
            return []
        placeholders = ", ".join("?" for _ in event_ids)
        sql = f"""
        SELECT DISTINCT e.*
        FROM entities e
        JOIN event_participants p
          ON p.doc_id = e.doc_id AND p.entity_id = e.entity_id
        WHERE p.doc_id = ?
          AND p.event_id IN ({placeholders})
        """
        return list(conn.execute(sql, [doc_id, *event_ids]))
