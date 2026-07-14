from __future__ import annotations

from .config import get_llm_config


def generate_answer(
    messages: list[dict[str, str]],
    *,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.0,
) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The 'openai' package is required for answer generation.") from exc

    llm_cfg = get_llm_config()
    resolved_api_key = api_key or llm_cfg["api_key"]
    resolved_base_url = base_url or llm_cfg["api_base"]
    if not resolved_api_key:
        raise RuntimeError(
            "Missing API key. Set AGENT_API_KEY in the project .env or environment."
        )

    client = OpenAI(
        api_key=resolved_api_key,
        base_url=resolved_base_url,
        timeout=120.0,
    )
    response = client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore[arg-type]
        temperature=temperature,
    )
    if not response.choices:
        return ""
    message = response.choices[0].message
    return message.content or ""
