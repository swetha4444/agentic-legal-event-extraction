#!/usr/bin/env python3
"""One-shot chat completion via Keymaker (same pattern as baseline: OpenAI client + AGENT_API_KEY)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env", override=True)
except ImportError:
    pass

from agents.config_loader import get_llm_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Ping Keymaker with a minimal Claude chat request.")
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-5",
        help="Model id as Keymaker expects (e.g. claude-sonnet-4-5, Claude-azure-models).",
    )
    parser.add_argument("--prompt", default="Reply with exactly one word: hello", help="User message.")
    args = parser.parse_args()

    key = (os.environ.get("AGENT_API_KEY") or "").strip()
    if not key:
        raise SystemExit("Set AGENT_API_KEY in .env or the environment.")

    cfg = get_llm_config()
    base = cfg.get("api_base") or "https://thekeymaker.umass.edu/"

    from openai import OpenAI

    client = OpenAI(api_key=key, base_url=base, timeout=120.0)
    model_lower = args.model.lower()
    temperature = 1.0 if "gpt-5" in model_lower or model_lower == "gpt5" else float(cfg.get("temperature", 0.0) or 0.0)

    req: dict = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": args.prompt},
        ],
        "temperature": temperature,
        "max_completion_tokens": 256,
    }
    try:
        resp = client.chat.completions.create(**req)
    except TypeError:
        req.pop("max_completion_tokens", None)
        resp = client.chat.completions.create(**req)
    except Exception as exc:
        text = str(exc).lower()
        if "only temperature=1 is supported" in text or "temperature=0.0" in text:
            req["temperature"] = 1.0
            resp = client.chat.completions.create(**req)
        elif "max_completion_tokens" in text or "unknown parameter" in text:
            req.pop("max_completion_tokens", None)
            resp = client.chat.completions.create(**req)
        else:
            raise

    content = (resp.choices[0].message.content or "").strip()
    print("model:", args.model)
    print("base_url:", base)
    print("reply:", content)


if __name__ == "__main__":
    main()
