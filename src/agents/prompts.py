"""
Prompts for facts extraction from court opinions.
Centralized so we can explore different prompt strategies.
"""

FACTS_EXTRACTION_SYSTEM_PROMPT = """You are a legal document analyst. Your task is to extract the court-established facts from U.S. court opinions.

Court-established facts are the factual narrative that the court accepts as true: what happened, who did what, and the key events that led to the case. Exclude:
- Legal arguments and legal conclusions
- Citations to statutes or precedent
- The court's reasoning or holding
- Procedural history unless it states concrete events

Output only the extracted factual statements, as plain text. Preserve paragraph structure if the input has it. Do not add headings or commentary."""

FACTS_EXTRACTION_USER_PROMPT = """Extract the court-established facts from the following court opinion.

Case context (optional): {case_name}
Docket: {docket_number}

---
OPINION TEXT:
---
{text}
---
"""


def build_facts_extraction_messages(
    text: str,
    case_name: str = "",
    docket_number: str = "",
    system_prompt: str = None,
    user_prompt_template: str = None,
) -> list:
    """
    Build messages for LLM facts-extraction call.
    Returns list of dicts suitable for litellm completion(messages=...).
    """
    system_prompt = system_prompt or FACTS_EXTRACTION_SYSTEM_PROMPT
    user_prompt_template = user_prompt_template or FACTS_EXTRACTION_USER_PROMPT
    user_content = user_prompt_template.format(
        case_name=case_name or "Unknown",
        docket_number=docket_number or "Unknown",
        text=text,
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]