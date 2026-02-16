# Facts extraction agents: LLM and LegalBERT
from .prompts import (
    FACTS_EXTRACTION_SYSTEM_PROMPT,
    FACTS_EXTRACTION_USER_PROMPT,
    build_facts_extraction_messages,
)
from .llm import LLMFactsExtractor

# LegalBERT optional (requires torch, transformers); allows LLM-only use without them
try:
    from .legal_bert import LegalBERTFactsExtractor
except ImportError:
    LegalBERTFactsExtractor = None  # type: ignore[misc, assignment]

# Agent type identifiers (use with run_facts_agent.py --agent)
FACTS_AGENT_LLM = "llm"
FACTS_AGENT_BERT = "bert"
FACTS_AGENT_TYPES = (FACTS_AGENT_LLM, FACTS_AGENT_BERT)

__all__ = [
    "FACTS_EXTRACTION_SYSTEM_PROMPT",
    "FACTS_EXTRACTION_USER_PROMPT",
    "build_facts_extraction_messages",
    "LLMFactsExtractor",
    "LegalBERTFactsExtractor",
    "FACTS_AGENT_LLM",
    "FACTS_AGENT_BERT",
    "FACTS_AGENT_TYPES",
]
