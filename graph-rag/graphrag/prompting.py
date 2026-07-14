from __future__ import annotations

import json
from typing import Any

from .models import RetrievalBundle


SYSTEM_PROMPT = """You answer legal-document questions using only the provided graph evidence.
Do not invent facts.
Do not treat temporal order as causation unless a causal edge or explicit evidence supports it.
If the graph evidence is insufficient, say so clearly.
Prefer explicit event evidence over inference.
When useful, distinguish:
- explicit graph fact
- temporal relation
- causal relation
- inference"""


def _format_event_line(hit: Any) -> str:
    support = "supporting_fact" if hit.is_supporting_fact else "event"
    return (
        f"- [{hit.event_id}] {hit.event_type} ({support}) | "
        f"score={hit.score:.3f} | lexical={hit.lexical_score:.3f} | "
        f"graph={hit.graph_score:.3f} | text={hit.text}"
    )


def _format_edge_line(edge: dict[str, Any]) -> str:
    return (
        f"- [{edge['edge_kind']}] {edge['from_event']} -> {edge['to_event']} | "
        f"relation={edge['relation']} | confidence={float(edge['confidence'] or 0.0):.2f} | "
        f"evidence={edge['evidence_text']}"
    )


def _format_entity_line(entity: dict[str, Any]) -> str:
    aliases = json.loads(entity["aliases_json"]) if entity.get("aliases_json") else []
    alias_suffix = f" | aliases={', '.join(aliases)}" if aliases else ""
    return (
        f"- [{entity['entity_id']}] {entity['name']} | kind={entity['kind']} | "
        f"role={entity['canonical_role']}{alias_suffix}"
    )


def _format_chunk_line(chunk: dict[str, Any]) -> str:
    span = []
    if chunk.get("start_sentence_id") is not None:
        span.append(f"start={chunk['start_sentence_id']}")
    if chunk.get("end_sentence_id") is not None:
        span.append(f"end={chunk['end_sentence_id']}")
    span_suffix = f" | {' '.join(span)}" if span else ""
    return f"- [{chunk['chunk_id']}] theme={chunk['theme']}{span_suffix} | text={chunk['text']}"


def build_prompt(bundle: RetrievalBundle) -> list[dict[str, str]]:
    event_block = "\n".join(_format_event_line(hit) for hit in bundle.events) or "- none"
    edge_block = "\n".join(_format_edge_line(edge) for edge in bundle.edges[:24]) or "- none"
    entity_block = "\n".join(_format_entity_line(entity) for entity in bundle.entities[:20]) or "- none"
    chunk_block = "\n".join(_format_chunk_line(chunk) for chunk in bundle.chunks[:12]) or "- none"
    summary_json = json.dumps(bundle.summary, indent=2, ensure_ascii=False)

    user_prompt = f"""Question:
{bundle.question}

Retrieval summary:
{summary_json}

Retrieved events:
{event_block}

Retrieved edges:
{edge_block}

Retrieved entities:
{entity_block}

Retrieved provenance chunks:
{chunk_block}

Instructions:
1. Answer directly.
2. Cite event ids inline when you rely on them.
3. If you use a causal relation, name the causal edge or say it is an inference.
4. If the evidence only supports temporal ordering, say "temporal evidence only".
5. If the graph is insufficient, say what is missing."""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
