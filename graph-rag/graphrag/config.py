from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def graphrag_root() -> Path:
    return Path(__file__).resolve().parent.parent


def project_root() -> Path:
    return graphrag_root().parent


def load_project_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    env_file = project_root() / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=False)


def _load_yaml() -> dict[str, Any]:
    config_path = project_root() / "config" / "config.yaml"
    if not config_path.exists():
        return {}
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_llm_config() -> dict[str, Any]:
    load_project_dotenv()
    data = _load_yaml()
    llm = data.get("llm") or {}

    api_key = (
        os.environ.get("AGENT_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or llm.get("api_key")
        or ""
    ).strip()
    api_base = (
        os.environ.get("OPENAI_BASE_URL")
        or llm.get("api_base")
        or "https://thekeymaker.umass.edu/"
    )
    model = os.environ.get("MODEL") or llm.get("model") or "gpt4o"

    return {
        "api_key": api_key,
        "api_base": api_base,
        "model": model,
        "temperature": llm.get("temperature", 0.0),
        "max_calls": llm.get("max_calls"),
        "max_docs": llm.get("max_docs"),
    }
