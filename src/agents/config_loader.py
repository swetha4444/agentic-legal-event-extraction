"""
Load LLM/agent config from config/config.yaml and env.
API key: use .env AGENT_API_KEY (recommended). Other settings: config.yaml.
Values from .env are loaded into os.environ when this module is first used.
"""
import os
from pathlib import Path

import yaml

# Load .env into os.environ so AGENT_API_KEY is available (optional dependency)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _load_yaml() -> dict:
    config_path = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def get_llm_config() -> dict:
    """LLM section from config; api_key overridden by env AGENT_API_KEY."""
    data = _load_yaml()
    llm = data.get("llm") or {}
    api_key = os.environ.get("AGENT_API_KEY") or llm.get("api_key")
    api_base = llm.get("api_base") or "https://thekeymaker.umass.edu/"
    model = llm.get("model") or "gpt4o"
    return {
        "api_key": api_key,
        "api_base": api_base,
        "model": model,
        "temperature": llm.get("temperature", 0.0),
        "max_calls": llm.get("max_calls"),
        "max_docs": llm.get("max_docs"),
    }
