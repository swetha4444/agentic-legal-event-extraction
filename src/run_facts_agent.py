"""
Run LLM-based facts extraction on JSONL (CourtListener output).
Uses config (config.yaml) and .env (AGENT_API_KEY). Respects max_calls and max_docs.
"""
import argparse
import json
import os
import sys

# Ensure src is on path when run as python src/run_facts_agent.py
if __name__ == "__main__":
    _src = os.path.dirname(os.path.abspath(__file__))
    if _src not in sys.path:
        sys.path.insert(0, _src)

from agents.llm_facts_agent import LLMFactsExtractor
from agents.budget import BudgetExceededError
from agents.config_loader import get_llm_config


def _output_path_for_model(model: str) -> str:
    """Sanitize model name for filename: data/outputs/facts_{model}.jsonl"""
    safe = (model or "default").replace("/", "-").strip()
    if not safe:
        safe = "default"
    return os.path.join("data", "outputs", f"facts_{safe}.jsonl")


def main():
    parser = argparse.ArgumentParser(description="Extract facts from court opinions using LLM agent")
    parser.add_argument("--input", default="data/processed/courtlistener_recap.jsonl", help="Input JSONL")
    parser.add_argument("--output", default=None, help="Output JSONL (default: data/outputs/facts_{model}.jsonl)")
    parser.add_argument("--limit", type=int, default=None, help="Max documents (overrides config max_docs if set)")
    parser.add_argument("--model", type=str, default=None, help="Override model (e.g. gpt4o, gpt-5-mini, Phi-4-mini-reasoning, gemma-3-4b-it)")
    args = parser.parse_args()

    cfg = get_llm_config()
    limit = args.limit if args.limit is not None else cfg.get("max_docs")
    model = args.model or cfg.get("model")
    output_path = args.output or _output_path_for_model(model)

    extractor = LLMFactsExtractor(model_name=model)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    count = 0

    with open(args.input) as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            if limit is not None and count >= limit:
                print(f"Stopped at max_docs={limit}")
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = rec.get("document_text") or ""
            if not text.strip():
                continue
            try:
                facts = extractor.extract(
                    text=text,
                    case_name=rec.get("case_name") or "",
                    docket_number=rec.get("docket_number") or "",
                )
            except BudgetExceededError as e:
                print(f"Budget exceeded: {e}")
                break
            # Same record shape as input (courtlistener_recap.jsonl), but document_text → extracted_facts
            out = {k: v for k, v in rec.items() if k != "document_text"}
            out["extracted_facts"] = facts
            f_out.write(json.dumps(out, default=str) + "\n")
            count += 1
            if count % 10 == 0:
                print(f"Processed {count} cases...")

    print(f"Done. Wrote {count} records to {output_path}")


if __name__ == "__main__":
    main()
