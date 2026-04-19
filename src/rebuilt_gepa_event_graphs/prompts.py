"""Judge and reflection prompts for rebuilt GEPA event-graph optimization."""

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

Guidance:
- factual_accuracy: are extracted facts correct
- completeness: are major events/actors/relations missing
- relevance: is extracted content focused and non-redundant
- faithfulness: no hallucinations; all claims traceable to text
- coherence: entities/events/relations form a consistent story
- overall: holistic assessment
- reason: one short source-grounded explanation
- major_issues: short bullet-style strings
"""

JUDGE_USER_PROMPT = """Evaluate this chunk-level legal event graph.

DOC_ID: {doc_id}
CHUNK_ID: {chunk_id}

SOURCE CHUNK:
{text}

EXTRACTED GRAPH JSON:
{graph_json}
"""

REFLECTION_SYSTEM_PROMPT = """You improve prompt/config candidates for legal event-graph extraction.

You will receive:
- the current candidate JSON
- evaluation metrics
- actionable side information summarizing failures
- representative execution traces from the evaluation run

Return strict JSON only in this form:
{
  "mutations": [
    {
      "name": "candidate_name",
      "model": "gpt-5-mini",
      "system_prompt": "...",
      "user_prompt_template": "...",
      "retry_suffix_templates": {
        "parse_retry": "...",
        "empty_retry": "..."
      },
      "config": {
        "max_events_per_chunk": 8,
        "max_completion_tokens": 7000,
        "enforce_json_response_format": true,
        "max_parse_retries": 1,
        "max_empty_retries": 1
      }
    }
  ]
}

Rules:
- Preserve the same overall candidate schema.
- Make targeted edits rather than rewriting everything blindly.
- Prefer changes that improve schema validity, non-empty extraction, structural consistency, and judge score.
- Use the trace bundle to diagnose concrete failure patterns, not just aggregate scores.
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
