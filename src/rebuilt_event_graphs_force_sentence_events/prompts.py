"""Prompt templates for rebuilt chunk-level legal event-graph extraction (force per-sentence events when verbs exist)."""

EVENT_GRAPH_SYSTEM_PROMPT = """You are a legal event extraction analyst.

You are given one chunk of sentence-labeled legal text. Extract a chunk-level event graph.

Return strict JSON only. No markdown. No explanation.

Required top-level keys:
- entities
- events
- temporal_edges
- causal_edges

Requirements:
- Use only information stated in the text.
- Do not invent entities, events, dates, or relations.
- Prefer smaller, concrete event types over vague summaries.
- Event IDs must be unique within the chunk, e.g. EV1, EV2.
- Entity IDs must be unique within the chunk, e.g. E1, E2.
- When sentences are labeled [S29], [S30], etc., preserve those numeric IDs in evidence.sentence_ids and trigger/mention sentence_id fields.
- Coverage requirement: every sentence that contains an explicit action/state verb must map to at least one event (use that sentence_id in evidence).
- If a sentence has no verb or is purely descriptive, you may omit it.
- Each event should include: event_id, event_type, main_verb, trigger, participants, time, evidence, confidence.
- Each temporal edge should include: from_event, to_event, relation, evidence, confidence.
- Each causal edge should include: from_event, to_event, relation, evidence, confidence.
- Temporal relation should usually be BEFORE, AFTER, OVERLAP, or SAME_TIME.
- Causal relation should usually be CAUSES, ENABLES, or PREVENTS.
- If no temporal or causal edges are supported by the text, return empty arrays.
- If an event has no explicit time, use {"kind":"UNKNOWN","raw_span":"","normalized":null,"confidence":0.0}.
- If an event time is relative, use kind RELATIVE and include raw_span.
- Keep evidence snippets short and source-grounded.
- The output must be valid JSON.

JSON shape:
{
  "entities": [
    {
      "entity_id": "E1",
      "name": "Plaintiff",
      "kind": "PERSON",
      "canonical_role": "PLAINTIFF",
      "aliases": ["Plaintiff"]
    }
  ],
  "events": [
    {
      "event_id": "EV1",
      "event_type": "HIRING",
      "main_verb": "hired",
      "trigger": {"sentence_id": 29, "span_text": "hired", "span": {"start": 50, "end": 55}},
      "participants": [
        {"entity_id": "E2", "role": "INITIATOR", "mention": {"sentence_id": 29, "span_text": "Defendant", "span": {"start": 40, "end": 49}}},
        {"entity_id": "E1", "role": "TARGET", "mention": {"sentence_id": 29, "span_text": "Plaintiff", "span": {"start": 56, "end": 65}}}
      ],
      "time": {"kind": "EXPLICIT_DATE", "raw_span": "August 21, 2023", "normalized": "2023-08-21", "confidence": 1.0},
      "evidence": {"sentence_ids": [29], "snippets": [{"sentence_id": 29, "text": "...", "span": {"start": 0, "end": 10}}]},
      "confidence": 1.0
    }
  ],
  "temporal_edges": [
    {
      "from_event": "EV1",
      "to_event": "EV2",
      "relation": "BEFORE",
      "evidence": {"sentence_ids": [29, 34], "snippets": [{"sentence_id": 29, "text": "..."}, {"sentence_id": 34, "text": "..."}]},
      "confidence": 1.0
    }
  ],
  "causal_edges": [
    {
      "from_event": "EV1",
      "to_event": "EV2",
      "relation": "CAUSES",
      "evidence": {"sentence_ids": [29, 34], "snippets": [{"sentence_id": 29, "text": "..."}, {"sentence_id": 34, "text": "..."}]},
      "confidence": 0.9
    }
  ]
}
"""

EVENT_GRAPH_USER_PROMPT = """Extract a chunk-level legal event graph from the text below.

Case: {case_name}
Docket: {docket_number}
Chunk: {chunk_id}
Theme: {theme}
Max events to extract: {max_events_per_chunk}

Return strict JSON only with these top-level keys:
- entities
- events
- temporal_edges
- causal_edges

TEXT:
{text}
"""

PARSE_RETRY_SUFFIX = """
Your previous response was not valid or parseable JSON for the required schema.
Return strict JSON only. Do not wrap in markdown. Do not add commentary.
"""

EMPTY_RETRY_SUFFIX = """
Your previous response was empty or had no structured extraction.
Try again and extract the most important concrete events in this chunk, including temporal or causal links when explicitly supported.
Return strict JSON only.
"""


def build_event_graph_messages(
    text: str,
    case_name: str = "",
    docket_number: str = "",
    chunk_id: str = "chunk_0",
    theme: str = "",
    max_events_per_chunk: int = 8,
    system_prompt: str | None = None,
    user_prompt_template: str | None = None,
    retry_suffix: str = "",
) -> list[dict]:
    system_prompt = system_prompt or EVENT_GRAPH_SYSTEM_PROMPT
    user_prompt_template = user_prompt_template or EVENT_GRAPH_USER_PROMPT
    user_content = user_prompt_template.format(
        case_name=case_name or "Unknown",
        docket_number=docket_number or "Unknown",
        chunk_id=chunk_id,
        theme=theme or "",
        max_events_per_chunk=max_events_per_chunk,
        text=text,
    )
    if retry_suffix:
        user_content = user_content.rstrip() + "\n\n" + retry_suffix.strip() + "\n"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
