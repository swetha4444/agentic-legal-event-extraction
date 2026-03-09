"""
LLM-based event extraction agent.
Consumes chunked text and returns structured entities + events (JSON).
Uses OpenAI-compatible API; enforces call budget.
"""
import json
import re
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from .prompts import build_events_extraction_messages, build_chunk_doc_messages
from ..budget import CallBudgetChecker, BudgetExceededError


def _get_config() -> dict:
    try:
        from ..config_loader import get_llm_config
        return get_llm_config()
    except Exception:
        return {}


def _parse_events_json(raw: str) -> tuple[Dict[str, Any], Optional[str]]:
    """
    Parse LLM response into entities + events dict.
    Strips markdown code fences if present. Returns (parsed_dict, error_message).
    """
    if not (raw or "").strip():
        return {"entities": [], "events": []}, None
    text = raw.strip()
    # Remove optional markdown code block
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        text = m.group(1).strip()
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            return {"entities": [], "events": []}, "Response is not a JSON object"
        entities = data.get("entities")
        events = data.get("events")
        if entities is None:
            entities = []
        if events is None:
            events = []
        if not isinstance(entities, list):
            entities = []
        if not isinstance(events, list):
            events = []
        return {"entities": entities, "events": events}, None
    except json.JSONDecodeError as e:
        return {"entities": [], "events": []}, str(e)


def _parse_chunk_doc_json(raw: str) -> tuple[List[Dict[str, Any]], Optional[str]]:
    """Parse LLM chunking response: { chunks: [ {chunk_id, start_sentence_id, end_sentence_id, theme}, ... ] }. Returns (list, error)."""
    if not (raw or "").strip():
        return [], None
    text = raw.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        text = m.group(1).strip()
    try:
        data = json.loads(text)
        chunks = data.get("chunks") if isinstance(data, dict) else None
        if not isinstance(chunks, list):
            return [], "Missing or invalid 'chunks' array"
        out = []
        for c in chunks:
            if not isinstance(c, dict):
                continue
            sid = c.get("start_sentence_id")
            eid = c.get("end_sentence_id")
            if sid is None or eid is None:
                continue
            out.append({
                "chunk_id": c.get("chunk_id") or f"chunk_{len(out)}",
                "start_sentence_id": int(sid),
                "end_sentence_id": int(eid),
                "theme": c.get("theme") or "",
            })
        return out, None
    except json.JSONDecodeError as e:
        return [], str(e)


class LLMEventsExtractor:
    """Extract entities and events from legal text (per chunk) using an LLM. Returns structured dict."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        max_calls: Optional[int] = None,
    ):
        cfg = _get_config()
        self.model_name = model_name or cfg.get("model") or "gpt4o"
        self.temperature = (
            temperature if temperature is not None else cfg.get("temperature", 0.0)
        )
        self.api_base = api_base or cfg.get("api_base") or "https://thekeymaker.umass.edu/"
        self.api_key = api_key or cfg.get("api_key")
        self.max_calls = max_calls if max_calls is not None else cfg.get("max_calls")
        self._budget = CallBudgetChecker(max_calls=self.max_calls)

        if OpenAI is None:
            raise ImportError(
                "openai is required for LLMEventsExtractor. Install with: pip install openai"
            )
        if not self.api_key:
            raise ValueError(
                "LLM api_key required. Set AGENT_API_KEY in .env (your Keymaker key, e.g. sk-...)."
            )
        if self.api_key.startswith("http://") or self.api_key.startswith("https://"):
            raise ValueError(
                "AGENT_API_KEY must be your Keymaker API key (starts with sk-), not the base URL."
            )

        self._client = OpenAI(api_key=self.api_key, base_url=self.api_base)

    def _effective_temperature(self) -> float:
        name = (self.model_name or "").lower()
        if "gpt-5" in name or name == "gpt5":
            return 1.0
        return self.temperature

    def extract(
        self,
        text: str,
        case_name: str = "",
        docket_number: str = "",
        chunk_id: str = "chunk_0",
    ) -> Dict[str, Any]:
        """
        Extract entities and events from one chunk. Returns {"entities": [...], "events": [...]}.
        Raises BudgetExceededError if max_calls reached.
        """
        return self.extract_with_prompt(
            text=text,
            case_name=case_name,
            docket_number=docket_number,
            chunk_id=chunk_id,
        )

    def extract_with_prompt(
        self,
        text: str,
        case_name: str = "",
        docket_number: str = "",
        chunk_id: str = "chunk_0",
        system_prompt: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract using optional custom prompts. Returns dict with entities and events."""
        if not (text or "").strip():
            return {"entities": [], "events": []}

        self._budget.check()
        messages = build_events_extraction_messages(
            text=text,
            case_name=case_name,
            docket_number=docket_number,
            chunk_id=chunk_id,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
        )
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self._effective_temperature(),
            )
            self._budget.record_call()
            raw = (response.choices[0].message.content or "").strip()
            data, err = _parse_events_json(raw)
            if err:
                return {"entities": [], "events": [], "_parse_error": err}
            return data
        except BudgetExceededError:
            raise
        except Exception as e:
            return {"entities": [], "events": [], "_error": str(e)}

    def get_meaningful_chunks(
        self,
        text: str,
        case_name: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Ask LLM to split the document into meaningful chunks (coherent units, with overlap).
        text should be formatted with [S1], [S2], ... sentence IDs.
        Returns list of {chunk_id, start_sentence_id, end_sentence_id, theme}. 1-based IDs.
        """
        if not (text or "").strip():
            return []
        self._budget.check()
        messages = build_chunk_doc_messages(text=text, case_name=case_name)
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self._effective_temperature(),
            )
            self._budget.record_call()
            raw = (response.choices[0].message.content or "").strip()
            chunks, err = _parse_chunk_doc_json(raw)
            if err:
                return []
            return chunks
        except BudgetExceededError:
            raise
        except Exception:
            return []
