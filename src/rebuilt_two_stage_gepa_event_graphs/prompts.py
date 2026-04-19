"""Prompts for rebuilt two-stage GEPA legal event-graph extraction."""

STAGE1_SYSTEM_PROMPT = """You are a legal event extraction analyst.

You are given one chunk of sentence-labeled legal text.

Stage 1 goal:
- extract grounded entities
- extract grounded event records
- do not infer temporal edges
- do not infer causal edges

Return strict JSON only. No markdown. No explanation.

Required top-level keys:
- entities
- events

Requirements:
- Use only information explicitly stated in the text.
- Do not invent entities, events, dates, or relations.
- Prefer smaller, concrete event types over vague summaries.
- Event IDs must be unique within the chunk, e.g. EV1, EV2.
- Entity IDs must be unique within the chunk, e.g. E1, E2.
- When sentences are labeled [S29], [S30], etc., preserve those numeric IDs in evidence.sentence_ids and trigger/mention sentence_id fields.
- Each event should include: event_id, event_type, main_verb, trigger, participants, time, evidence, confidence.
- Do not include temporal_edges or causal_edges in stage 1.
- If an event has no explicit time, use {"kind":"UNKNOWN","raw_span":"","normalized":null,"confidence":0.0}.
- If an event time is relative, use kind RELATIVE and include raw_span.
- Keep evidence snippets short and source-grounded.
- The output must be valid JSON.
"""

STAGE1_USER_PROMPT = """Extract grounded entities and event records from the text below.

Case: {case_name}
Docket: {docket_number}
Chunk: {chunk_id}
Theme: {theme}
Max events to extract: {max_events_per_chunk}

Return strict JSON only with these top-level keys:
- entities
- events

Do not include temporal_edges or causal_edges in this stage.

TEXT:
{text}
"""

STAGE2_SYSTEM_PROMPT = """You are a legal event graph assembly analyst.

You are given:
- the original chunk text
- a grounded stage-1 extraction containing entities and event records

Stage 2 goal:
- preserve grounded entities and events from stage 1
- add only source-supported temporal_edges and causal_edges
- return the final chunk-level event graph

Return strict JSON only. No markdown. No explanation.

Required top-level keys:
- entities
- events
- temporal_edges
- causal_edges

Requirements:
- Use only the source text and the stage-1 extraction.
- Do not invent entities, events, dates, or relations.
- Keep stage-1 entities and events unless they are clearly invalid duplicates.
- Temporal relation should usually be BEFORE, AFTER, OVERLAP, or SAME_TIME.
- Causal relation should usually be CAUSES, ENABLES, or PREVENTS.
- If no temporal or causal edges are supported by the text, return empty arrays.
- Keep evidence snippets short and source-grounded.
- The output must be valid JSON.
"""

STAGE2_USER_PROMPT = """Convert the grounded stage-1 extraction into a final chunk-level legal event graph.

Case: {case_name}
Docket: {docket_number}
Chunk: {chunk_id}
Theme: {theme}

SOURCE TEXT:
{text}

STAGE-1 EXTRACTION JSON:
{intermediate_json}

Return strict JSON only with these top-level keys:
- entities
- events
- temporal_edges
- causal_edges
"""

JUDGE_SYSTEM_PROMPT = """You are a strict evaluator for legal event-graph extraction.

You are given:
- the source chunk text
- the extracted graph JSON

Score the extraction against the source text only.
Return strict JSON only with this schema:
- factual_accuracy
- completeness
- relevance
- faithfulness
- coherence
- overall
- reason
- major_issues

Field requirements:
- factual_accuracy, completeness, relevance, faithfulness, coherence, overall: integers from 1 to 5
- reason: short string explanation
- major_issues: array of short strings; use [] if there are no major issues
"""

JUDGE_USER_PROMPT = """Evaluate this chunk-level legal event graph.

DOC_ID: {doc_id}
CHUNK_ID: {chunk_id}

SOURCE CHUNK:
{text}

EXTRACTED GRAPH JSON:
{graph_json}
"""

REFLECTION_SYSTEM_PROMPT = """You improve two-stage prompt/config candidates for legal event-graph extraction.

You will receive:
- the current candidate JSON
- evaluation metrics
- actionable side information summarizing failures
- representative execution traces from the evaluation run

Return strict JSON only in this form:
{{
  "mutations": [
    {{
      "name": "candidate_name",
      "stage1_model": "claude-sonnet-4-5",
      "stage2_model": "claude-sonnet-4-5",
      "stage1_system_prompt": "...",
      "stage1_user_prompt_template": "...",
      "stage2_system_prompt": "...",
      "stage2_user_prompt_template": "...",
      "retry_suffix_templates": {{
        "stage1_parse_retry": "...",
        "stage1_empty_retry": "...",
        "stage2_parse_retry": "...",
        "stage2_empty_retry": "..."
      }},
      "config": {{
        "max_events_per_chunk": 8,
        "stage1_max_completion_tokens": 2600,
        "stage2_max_completion_tokens": 2600,
        "enforce_json_response_format": true,
        "max_stage1_parse_retries": 2,
        "max_stage1_empty_retries": 1,
        "max_stage2_parse_retries": 2,
        "max_stage2_empty_retries": 1
      }}
    }}
  ]
}}

Rules:
- Preserve the same overall candidate schema.
- Make targeted edits rather than rewriting everything blindly.
- Prefer changes that improve schema validity, non-empty extraction, structural consistency, and judge score.
- Use the trace bundle to diagnose concrete failure patterns, not just aggregate scores.
- Keep the two-stage design intact: stage 1 extracts grounded entities/events, stage 2 builds the final graph.
- Do not remove required keys.
- Return 1 to {num_mutations} mutations.
"""

REFLECTION_USER_PROMPT = """Current candidate JSON:
{candidate_json}

Evaluation metrics:
{metrics_json}

Actionable side information:
{asi_text}

Representative execution traces:
{trace_json}
"""


def build_stage1_messages(
    *,
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
    system_prompt = system_prompt or STAGE1_SYSTEM_PROMPT
    user_prompt_template = user_prompt_template or STAGE1_USER_PROMPT
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


def build_stage2_messages(
    *,
    text: str,
    intermediate_json: str,
    case_name: str = "",
    docket_number: str = "",
    chunk_id: str = "chunk_0",
    theme: str = "",
    max_events_per_chunk: int = 8,
    system_prompt: str | None = None,
    user_prompt_template: str | None = None,
    retry_suffix: str = "",
) -> list[dict]:
    system_prompt = system_prompt or STAGE2_SYSTEM_PROMPT
    user_prompt_template = user_prompt_template or STAGE2_USER_PROMPT
    user_content = user_prompt_template.format(
        case_name=case_name or "Unknown",
        docket_number=docket_number or "Unknown",
        chunk_id=chunk_id,
        theme=theme or "",
        text=text,
        intermediate_json=intermediate_json,
    )
    if retry_suffix:
        user_content = user_content.rstrip() + "\n\n" + retry_suffix.strip() + "\n"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def build_judge_messages(doc_id: str, chunk_id: str, text: str, graph_json: str) -> list[dict]:
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": JUDGE_USER_PROMPT.format(
                doc_id=doc_id,
                chunk_id=chunk_id,
                text=text,
                graph_json=graph_json,
            ),
        },
    ]


def build_reflection_messages(
    candidate_json: str,
    metrics_json: str,
    asi_text: str,
    trace_json: str,
    num_mutations: int,
) -> list[dict]:
    return [
        {"role": "system", "content": REFLECTION_SYSTEM_PROMPT.format(num_mutations=num_mutations)},
        {
            "role": "user",
            "content": REFLECTION_USER_PROMPT.format(
                candidate_json=candidate_json,
                metrics_json=metrics_json,
                asi_text=asi_text,
                trace_json=trace_json,
            ),
        },
    ]
