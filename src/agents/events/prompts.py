"""
Prompts for event extraction from legal text (5W1H-style: who, what, when, where, why, how).
Output: structured JSON with entities and events (trigger, participants, time, evidence).
Also: LLM-based semantic chunking (meaningful chunks with overlap).
"""

# ---- LLM chunking: ask model to split doc into meaningful chunks ----
CHUNK_DOC_SYSTEM_PROMPT = """You are a legal document analyst. Your task is to split the given factual narrative into **meaningful chunks**.

Each chunk should be a **coherent unit**: one event sequence, one topic, or one phase of the story (e.g. "hiring and role", "complaint and investigation", "termination", "aftermath"). Do not split in the middle of an event or a thought. Prefer natural boundaries (new theme, new time period, new actor).

Use **overlap** for context: the last 2-3 sentences of chunk N should also appear as the first 2-3 sentences of chunk N+1, so the next chunk has continuity. So start_sentence_id of chunk 2 might be (end_sentence_id of chunk 1 minus 2).

The text is labeled with sentence IDs: [S1], [S2], ... Use those numbers in your output.

Output **only** valid JSON (no markdown, no explanation):
{"chunks": [{"chunk_id": "chunk_0", "start_sentence_id": 1, "end_sentence_id": 15, "theme": "brief description"}, {"chunk_id": "chunk_1", "start_sentence_id": 13, "end_sentence_id": 28, "theme": "brief description"}, ...]}

Rules: start_sentence_id and end_sentence_id are 1-based (match [S1], [S2]). Chunks must cover every sentence. Overlap: start of next chunk <= end of previous chunk + 1 (typically end - 2 or end - 1)."""

CHUNK_DOC_USER_PROMPT = """Split this factual narrative into meaningful chunks (coherent units, with 2-3 sentence overlap between consecutive chunks). Output only JSON with "chunks" array as described.

Case: {case_name}

---
TEXT (sentence IDs [S1], [S2], ...):
---
{text}
---
"""


def build_chunk_doc_messages(
    text: str,
    case_name: str = "",
    system_prompt: str = None,
    user_prompt_template: str = None,
) -> list:
    """Build messages for LLM chunking call. Returns list of dicts for chat completion."""
    system_prompt = system_prompt or CHUNK_DOC_SYSTEM_PROMPT
    user_prompt_template = user_prompt_template or CHUNK_DOC_USER_PROMPT
    user_content = user_prompt_template.format(
        case_name=case_name or "Unknown",
        text=text,
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


EVENTS_EXTRACTION_SYSTEM_PROMPT = """You are a legal document analyst. Your task is to extract **events** and **entities** from the given legal text (e.g., court opinion or complaint facts).

For each **event**, extract:
- **event_type**: e.g. InternalComplaintFiled, InvestigationStarted, Termination, EEOCChargeFiled, Filing, Decision, etc.
- **trigger**: the exact span of text that denotes the event (phrase or clause).
- **participants**: who was involved. For each participant give: entity reference (name or ID), **role** (INITIATOR = who did it, TARGET = who was affected, CONTEXT = org/agency/court providing context).
- **time**: explicit date if present (normalize to YYYY-MM-DD), or "relative" (e.g. "after the complaint"), or "unknown".
- **evidence**: the sentence(s) or snippet that support this event. When the text is labeled with [S1], [S2], ... use **sentence_ids** as those numbers (e.g. [1, 2]).

For **entities**, list people, organizations, agencies, courts with: name, kind (PERSON, ORG, AGENCY, COURT), canonical_role (PLAINTIFF, DEFENDANT, EMPLOYER, SUPERVISOR, AGENCY, COURT, UNKNOWN).

Output **only** valid JSON in this shape (no markdown, no explanation):
{"entities": [{"entity_id": "E1", "kind": "PERSON", "canonical_role": "PLAINTIFF", "name": "...", "aliases": []}], "events": [{"event_id": "EV1", "event_type": "...", "trigger": {"span_text": "..."}, "participants": [{"entity_id": "E1", "role": "INITIATOR", "mention": "..."}], "time": {"kind": "EXPLICIT_DATE"|"RELATIVE"|"UNKNOWN", "raw_span": "...", "normalized": "YYYY-MM-DD or null"}, "evidence": {"sentence_ids": [], "snippets": [{"text": "..."}]}}]}"""

EVENTS_EXTRACTION_USER_PROMPT = """Extract entities and events from the following legal text. Output only valid JSON (entities + events as described).

Case context (optional): {case_name}
Docket: {docket_number}
Chunk: {chunk_id}

---
TEXT:
---
{text}
---
"""


def build_events_extraction_messages(
    text: str,
    case_name: str = "",
    docket_number: str = "",
    chunk_id: str = "chunk_0",
    system_prompt: str = None,
    user_prompt_template: str = None,
) -> list:
    """Build messages for LLM event-extraction call. Returns list of dicts for chat completion."""
    system_prompt = system_prompt or EVENTS_EXTRACTION_SYSTEM_PROMPT
    user_prompt_template = user_prompt_template or EVENTS_EXTRACTION_USER_PROMPT
    user_content = user_prompt_template.format(
        case_name=case_name or "Unknown",
        docket_number=docket_number or "Unknown",
        chunk_id=chunk_id,
        text=text,
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
