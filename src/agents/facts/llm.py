"""
LLM-based facts extraction agent.
Uses OpenAI-compatible API (Keymaker/LiteLLM proxy): api_key, base_url, model.
Enforces call budget.
"""
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from .prompts import build_facts_extraction_messages
from ..budget import CallBudgetChecker, BudgetExceededError


def _get_config() -> dict:
    try:
        from ..config_loader import get_llm_config
        return get_llm_config()
    except Exception:
        return {}


class LLMFactsExtractor:
    """Extract court-established facts from opinion text using an LLM (OpenAI-compatible API)."""

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
                "openai is required for LLMFactsExtractor. Install with: pip install openai"
            )
        if not self.api_key:
            raise ValueError(
                "LLM api_key required. Set AGENT_API_KEY in .env (your Keymaker key, e.g. sk-...)."
            )
        if self.api_key.startswith("http://") or self.api_key.startswith("https://"):
            raise ValueError(
                "AGENT_API_KEY must be your Keymaker API key (starts with sk-), not the base URL. "
                "Set AGENT_API_KEY=sk-your-key in .env and api_base is already in config."
            )

        self._client = OpenAI(api_key=self.api_key, base_url=self.api_base)

    def _effective_temperature(self) -> float:
        """gpt-5 family only supports temperature=1; use that for those models."""
        name = (self.model_name or "").lower()
        if "gpt-5" in name or name == "gpt5":
            return 1.0
        return self.temperature

    def extract(self, text: str, case_name: str = "", docket_number: str = "") -> str:
        """Extract court-established facts. Raises BudgetExceededError if max_calls reached."""
        return self.extract_with_prompt(
            text=text,
            case_name=case_name,
            docket_number=docket_number,
        )

    def extract_with_prompt(
        self,
        text: str,
        case_name: str = "",
        docket_number: str = "",
        system_prompt: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
    ) -> str:
        """Extract facts using optional custom prompts (for eval pipelines)."""
        if not (text or "").strip():
            return ""

        self._budget.check()
        messages = build_facts_extraction_messages(
            text=text,
            case_name=case_name,
            docket_number=docket_number,
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
            return (response.choices[0].message.content or "").strip()
        except BudgetExceededError:
            raise
        except Exception as e:
            print(f"LLM facts extraction error: {e}")
            return ""
