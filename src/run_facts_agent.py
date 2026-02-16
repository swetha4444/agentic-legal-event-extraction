"""
Run facts extraction on JSONL (CourtListener output).
Swap agent type: LLM or LegalBERT (--agent llm | bert).
Same output format for both: input records with document_text replaced by extracted_facts.
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

from agents import (
    LLMFactsExtractor,
    LegalBERTFactsExtractor,
    BudgetExceededError,
    FACTS_AGENT_LLM,
    FACTS_AGENT_BERT,
    FACTS_AGENT_TYPES,
)
from agents.config_loader import get_llm_config


def _output_path_for_agent(agent: str, model: str = None) -> str:
    """Default output path: facts_{agent_suffix}.jsonl"""
    if agent == FACTS_AGENT_BERT:
        return os.path.join("data", "outputs", "facts_legal_bert.jsonl")
    # LLM: use model name in path
    safe = (model or "default").replace("/", "-").strip() or "default"
    return os.path.join("data", "outputs", f"facts_{safe}.jsonl")


def main():
    parser = argparse.ArgumentParser(
        description="Extract facts from court opinions. Choose agent type: LLM or LegalBERT.",
    )
    parser.add_argument("--agent", choices=FACTS_AGENT_TYPES, default=FACTS_AGENT_LLM,
                        help="Facts extraction agent: llm or bert (default: llm)")
    parser.add_argument("--input", default="data/processed/courtlistener_recap.jsonl", help="Input JSONL")
    parser.add_argument("--output", default=None,
                        help="Output JSONL (default: data/outputs/facts_{model|legal_bert}.jsonl)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max documents (overrides config max_docs when agent=llm)")
    # LLM-specific
    parser.add_argument("--model", type=str, default=None,
                        help="LLM model name (e.g. gpt4o, gpt-5-mini). Used only when --agent llm")
    # LegalBERT-specific
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained LegalBERT checkpoint. Required when --agent bert")
    parser.add_argument("--first-contiguous-only", action="store_true",
                        help="Keep only first contiguous block of fact sentences (LegalBERT only)")
    parser.add_argument("--device", default=None, help="Device for LegalBERT: cuda or cpu (default: auto)")
    args = parser.parse_args()

    if args.agent == FACTS_AGENT_BERT:
        if LegalBERTFactsExtractor is None:
            print("Error: LegalBERT agent requires torch and transformers. Install with: pip install torch transformers")
            sys.exit(1)
        if not args.checkpoint or not os.path.isdir(args.checkpoint):
            print("Error: --agent bert requires --checkpoint <path to trained model dir>")
            print("Train with: python scripts/train_legal_bert_facts.py --data <labeled.jsonl> --output_dir <dir>")
            sys.exit(1)
        extractor = LegalBERTFactsExtractor(
            checkpoint_path=args.checkpoint,
            device=args.device,
            first_contiguous_only=args.first_contiguous_only,
        )
        output_path = args.output or _output_path_for_agent(FACTS_AGENT_BERT)
    else:
        cfg = get_llm_config()
        limit = args.limit if args.limit is not None else cfg.get("max_docs")
        model = args.model or cfg.get("model")
        extractor = LLMFactsExtractor(model_name=model)
        output_path = args.output or _output_path_for_agent(FACTS_AGENT_LLM, model)

    limit = args.limit
    if limit is None and args.agent == FACTS_AGENT_LLM:
        limit = get_llm_config().get("max_docs")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    count = 0

    with open(args.input) as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            if limit is not None and count >= limit:
                print(f"Stopped at limit={limit}")
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
            out = {k: v for k, v in rec.items() if k != "document_text"}
            out["extracted_facts"] = facts
            f_out.write(json.dumps(out, default=str) + "\n")
            count += 1
            if count % 10 == 0:
                print(f"Processed {count} cases...")

    print(f"Done. Wrote {count} records to {output_path}")


if __name__ == "__main__":
    main()
