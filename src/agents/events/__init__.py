# Events extraction agent: entities + events (5W1H-style) from legal text
from .prompts import (
    EVENTS_EXTRACTION_SYSTEM_PROMPT,
    EVENTS_EXTRACTION_USER_PROMPT,
    build_events_extraction_messages,
)
from .llm import LLMEventsExtractor
from .merge import MiniGraphMergeAgent
from .minigraph_mask import mask_minigraph_with_llm, masked_graph_is_compatible
from .hybrid_merge import HybridMergeRefiner

EVENTS_AGENT_LLM = "llm"
EVENTS_AGENT_TYPES = (EVENTS_AGENT_LLM,)

__all__ = [
    "EVENTS_EXTRACTION_SYSTEM_PROMPT",
    "EVENTS_EXTRACTION_USER_PROMPT",
    "build_events_extraction_messages",
    "LLMEventsExtractor",
    "MiniGraphMergeAgent",
    "HybridMergeRefiner",
    "mask_minigraph_with_llm",
    "masked_graph_is_compatible",
    "EVENTS_AGENT_LLM",
    "EVENTS_AGENT_TYPES",
]
