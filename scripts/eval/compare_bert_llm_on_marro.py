#!/usr/bin/env python3
"""
Run LegalBERT and LLM on the same MARRO doc(s); report comparable metrics (Acc, P, R, F1).
"""
import argparse
import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path

# Project root (script lives in scripts/eval/)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

COUNTRY_TO_DIR = {
    "uk": ROOT / "data" / "MARRO" / "UK-dataset",
    "india": ROOT / "data" / "MARRO" / "IN-dataset",
    "in": ROOT / "data" / "MARRO" / "IN-dataset",
}

LLM_COMPARE_SYSTEM_PROMPT = """You are a legal annotation assistant.
Classify numbered sentences as FACT vs NON-FACT for judicial case documents (any jurisdiction, including UK/India).
Return only JSON with fact sentence ids, no prose."""

LLM_COMPARE_USER_PROMPT = """Given the numbered sentences below, identify which sentence ids are FACT.

FACT means concrete events or accepted factual narrative (who did what, what happened, when/where).
NON-FACT includes legal arguments, legal reasoning, holdings, citations, procedural boilerplate, and metadata.

Return strict JSON only:
{{"fact_sentence_ids": [1, 4, 9]}}

Rules:
- Use only ids from the list.
- Do not paraphrase or quote sentences.
- If no facts, return: {{"fact_sentence_ids": []}}

SENTENCES:
{text}
"""


def load_marro_doc(path: Path):
    """Return (sentences, gold_labels). Gold: 1 = FAC, 0 = rest."""
    sentences, gold = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            sent, label = line.split("\t", 1)
            sent = sent.strip()
            if not sent:
                continue
            sentences.append(sent)
            gold.append(1 if label.strip().upper() == "FAC" else 0)
    return sentences, gold


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def metrics(gold: list, pred: list):
    if len(pred) != len(gold):
        return None
    correct = sum(1 for p, g in zip(pred, gold) if p == g)
    acc = correct / len(gold)
    tp = sum(1 for p, g in zip(pred, gold) if p == 1 and g == 1)
    fp = sum(1 for p, g in zip(pred, gold) if p == 1 and g == 0)
    fn = sum(1 for p, g in zip(pred, gold) if p == 0 and g == 1)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"acc": acc, "p": prec, "r": rec, "f1": f1, "n": len(gold), "n_fact_gold": sum(gold), "n_fact_pred": sum(pred)}


def resolve_doc_paths(docs_path: Path):
    docs_path = docs_path.resolve()
    if docs_path.is_file():
        return [docs_path]
    if docs_path.is_dir():
        return sorted(docs_path.glob("*.txt"))
    raise FileNotFoundError(f"--docs must be a file or dir: {docs_path}")


def prompt_country():
    while True:
        raw = input("Choose country [uk/india]: ").strip().lower()
        if raw in COUNTRY_TO_DIR:
            return raw
        print("Invalid choice. Enter 'uk' or 'india'.")


def prompt_limit():
    while True:
        raw = input("Enter sentence limit (e.g. 100, or blank for all): ").strip()
        if raw == "":
            return None
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
        print("Invalid limit. Enter a positive integer or blank.")


def balanced_sample_counts(lengths, limit, rng):
    """Allocate up to `limit` examples across docs with near-equal per-doc counts."""
    total_available = sum(lengths)
    if limit is None or limit >= total_available:
        return list(lengths)
    if limit <= 0:
        return [0] * len(lengths)

    non_empty = [idx for idx, n in enumerate(lengths) if n > 0]
    if not non_empty:
        return [0] * len(lengths)

    counts = [0] * len(lengths)
    base = limit // len(non_empty)
    for idx in non_empty:
        counts[idx] = min(base, lengths[idx])

    remaining = limit - sum(counts)
    while remaining > 0:
        candidates = [idx for idx in non_empty if counts[idx] < lengths[idx]]
        if not candidates:
            break
        rng.shuffle(candidates)
        for idx in candidates:
            if remaining == 0:
                break
            if counts[idx] < lengths[idx]:
                counts[idx] += 1
                remaining -= 1

    return counts


def parse_fact_ids(output: str, n_sentences: int):
    """Parse fact ids from LLM JSON output. Returns (ids_set, parsed_ok)."""
    text = (output or "").strip()
    if not text:
        return set(), False

    # Handle fenced output.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    candidates = [text]
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        candidates.insert(0, match.group(0))

    keys = ("fact_sentence_ids", "fact_ids", "facts", "ids")
    for cand in candidates:
        try:
            payload = json.loads(cand)
        except Exception:
            continue

        raw_ids = []
        if isinstance(payload, dict):
            for key in keys:
                if key in payload:
                    raw_ids = payload[key]
                    break
        elif isinstance(payload, list):
            raw_ids = payload

        if not isinstance(raw_ids, list):
            return set(), True

        parsed = set()
        for item in raw_ids:
            try:
                idx = int(item)
            except (TypeError, ValueError):
                continue
            if 1 <= idx <= n_sentences:
                parsed.add(idx)
        return parsed, True

    return set(), False


def save_run_output(lines, prefix: str):
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{prefix}_{stamp}.txt"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def main():
    p = argparse.ArgumentParser(description="Compare LegalBERT vs LLM on MARRO doc(s). Same test set, same metrics.")
    p.add_argument("--doc", type=Path, nargs="*", default=None, help="One or more MARRO .txt files")
    p.add_argument("--docs", type=Path, default=None, help="Directory of MARRO .txt files")
    p.add_argument(
        "--country",
        type=str,
        choices=["uk", "india", "in"],
        default=None,
        help="Shortcut for MARRO country directory (data/MARRO/UK-dataset or IN-dataset)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Total sentence limit sampled across selected .txt files (balanced per file).",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed for balanced sampling.")
    p.add_argument("--bert", type=Path, required=True, help="LegalBERT/LexLM checkpoint dir")
    p.add_argument("--llm", type=str, default=None, help="LLM model name (e.g. gpt4o). If omitted, only BERT.")
    p.add_argument("--device", default=None)
    p.add_argument("--no-save-output", action="store_true", help="Do not save run output to scripts/eval/outputs.")
    args = p.parse_args()
    run_lines = []

    def emit(line=""):
        print(line)
        run_lines.append(str(line))

    if args.docs is None and not args.doc:
        if args.country is None:
            if sys.stdin.isatty():
                args.country = prompt_country()
            else:
                emit("Error: provide --country {uk|india} when running non-interactively.")
                sys.exit(1)
        if args.limit is None:
            if sys.stdin.isatty():
                args.limit = prompt_limit()
            else:
                emit("Error: provide --limit when running non-interactively with --country.")
                sys.exit(1)

    if args.doc:
        doc_paths = [Path(d).resolve() for d in args.doc]
    elif args.docs is not None:
        try:
            doc_paths = resolve_doc_paths(args.docs)
        except FileNotFoundError as exc:
            emit(f"Error: {exc}")
            sys.exit(1)
    else:
        country_key = args.country or "uk"
        country_dir = COUNTRY_TO_DIR[country_key]
        doc_paths = sorted(country_dir.glob("*.txt"))
    if not doc_paths:
        emit("No documents to evaluate.")
        sys.exit(1)
    if args.limit is not None and args.limit <= 0:
        emit("Error: --limit must be a positive integer.")
        sys.exit(1)
    if not args.bert.resolve().is_dir():
        emit(f"Checkpoint not found: {args.bert}")
        sys.exit(1)

    from processing.legal_bert_facts import LegalBERTFactExtractor
    bert_extractor = LegalBERTFactExtractor(checkpoint_path=str(args.bert), device=args.device)

    llm_extractor = None
    if args.llm:
        try:
            from agents.facts import LLMFactsExtractor
            llm_extractor = LLMFactsExtractor(model_name=args.llm)
        except Exception as e:
            emit(f"LLM not available: {e}. Run without --llm for BERT only.")
            args.llm = None

    rng = random.Random(args.seed)
    loaded_docs = []
    for path in doc_paths:
        if not path.is_file():
            continue
        sentences, gold = load_marro_doc(path)
        if sentences:
            loaded_docs.append((path, sentences, gold))
    if not loaded_docs:
        emit("No documents to evaluate.")
        sys.exit(1)

    per_doc_lengths = [len(sentences) for _, sentences, _ in loaded_docs]
    sample_counts = balanced_sample_counts(per_doc_lengths, args.limit, rng)
    sampled_total = 0
    full_total = sum(per_doc_lengths)

    all_gold, all_bert_pred, all_llm_pred = [], [], []

    for (doc_path, sentences, gold), sample_n in zip(loaded_docs, sample_counts):
        if sample_n <= 0:
            continue
        if sample_n < len(sentences):
            picked = sorted(rng.sample(range(len(sentences)), sample_n))
            sampled_sentences = [sentences[i] for i in picked]
            sampled_gold = [gold[i] for i in picked]
        else:
            sampled_sentences = sentences
            sampled_gold = gold

        sampled_total += len(sampled_sentences)
        pred_bert = bert_extractor.predict_labels(sampled_sentences)
        all_gold.extend(sampled_gold)
        all_bert_pred.extend(pred_bert)
        if llm_extractor is not None:
            doc_text = "\n".join(
                f"{idx}\t{sentence}" for idx, sentence in enumerate(sampled_sentences, start=1)
            )
            try:
                out = llm_extractor.extract_with_prompt(
                    text=doc_text,
                    case_name="",
                    docket_number="",
                    system_prompt=LLM_COMPARE_SYSTEM_PROMPT,
                    user_prompt_template=LLM_COMPARE_USER_PROMPT,
                )
            except Exception as e:
                emit(f"LLM failed on {doc_path.name}: {e}")
                out = ""
            fact_ids, parsed_ok = parse_fact_ids(out, len(sampled_sentences))
            if parsed_ok:
                all_llm_pred.extend(
                    [1 if (idx in fact_ids) else 0 for idx in range(1, len(sampled_sentences) + 1)]
                )
            else:
                # Fallback to old containment matching if JSON parsing fails.
                out_norm = _norm(out)
                all_llm_pred.extend([1 if _norm(s) in out_norm else 0 for s in sampled_sentences])

    m_bert = metrics(all_gold, all_bert_pred)
    if m_bert is None:
        sys.exit(1)
    emit("=" * 60)
    emit(f"Test docs: {len(doc_paths)}  |  Sentences: {m_bert['n']}  |  Gold facts: {m_bert['n_fact_gold']}")
    sampled_msg = f"sampled {sampled_total}/{full_total} sentence(s)"
    if args.limit is None:
        sampled_msg = f"using all {sampled_total} sentence(s)"
    emit(f"Selection: {sampled_msg} (seed={args.seed})")
    emit("=" * 60)
    emit("LegalBERT/LexLM")
    emit(f"  Accuracy: {m_bert['acc']:.4f}  |  Fact P: {m_bert['p']:.4f}  R: {m_bert['r']:.4f}  F1: {m_bert['f1']:.4f}  (pred fact: {m_bert['n_fact_pred']})")
    if llm_extractor is not None and all_llm_pred:
        m_llm = metrics(all_gold, all_llm_pred)
        if m_llm:
            emit(f"LLM ({args.llm})")
            emit(f"  Accuracy: {m_llm['acc']:.4f}  |  Fact P: {m_llm['p']:.4f}  R: {m_llm['r']:.4f}  F1: {m_llm['f1']:.4f}  (pred fact: {m_llm['n_fact_pred']})")
    emit("=" * 60)

    if not args.no_save_output:
        out_path = save_run_output(run_lines, prefix="compare_out")
        emit(f"Saved output: {out_path}")


if __name__ == "__main__":
    main()
