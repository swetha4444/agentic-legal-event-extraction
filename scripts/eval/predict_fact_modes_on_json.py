#!/usr/bin/env python3
"""
Prediction-only FACT vs NON-FACT runner for raw/compiled JSON inputs.

This mirrors the useful parts of eval_fact_modes_on_marro_json.py but removes the
requirement for gold labels. It is intended for new inference-time document sets.

Output:
- <run_name>_predictions.jsonl
- <run_name>_summary.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import torch
except ImportError as exc:
    raise ImportError("torch is required for classifier/hybrid mode. Install with: pip install torch") from exc

from agents.config_loader import get_llm_config
from agents.facts import LLMFactsExtractor
from processing.legal_bert_facts import LegalBERTFactExtractor, split_sentences


DEFAULT_OUTPUT_DIR = ROOT / "data" / "outputs" / "fact_extraction_predictions"
DEFAULT_CHECKPOINT = ROOT / "data" / "models" / "02-24-2:04PM_legalbert_train60"
DEFAULT_TEXT_FIELD_CANDIDATES = (
    "document_text",
    "opinion_text",
    "complaint_text",
    "text",
    "content",
    "extracted_facts",
)
DEFAULT_TITLE_FIELD_CANDIDATES = (
    "title",
    "case_name",
    "case_id",
    "id",
    "docket_number",
)

LLM_SYSTEM_PROMPT = """You are a legal annotation assistant.
Classify numbered sentences as FACT vs NON-FACT for judicial documents.
Return strict JSON only, no prose."""

LLM_USER_PROMPT = """Given the numbered sentences below, identify which sentence ids are FACT.

FACT means a concrete, observable event:
- an action taken by a person or entity
- a communication (said, called, warned, asked)
- a decision or outcome (fired, hired, assigned)
- a physical act (grabbed, drove, entered)
- what happened, who did what, when/where, who felt what

NON-FACT includes:
- opinions, emotions, or internal states
- general descriptions or background information

Return strict JSON only:
{{"fact_sentence_ids": [1, 4, 9]}}

Rules:
- Use only ids from the list.
- Do not paraphrase or quote sentences.
- If no facts, return: {{"fact_sentence_ids": []}}

SENTENCES:
{text}
"""


def _load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict):
                    records.append(payload)
        return records

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON array in {path}, got {type(payload).__name__}")
    return [rec for rec in payload if isinstance(rec, dict)]


def _first_nonempty(record: dict[str, Any], candidates: Sequence[str]) -> tuple[str | None, str]:
    for key in candidates:
        value = record.get(key)
        text = str(value or "").strip()
        if text:
            return key, text
    return None, ""


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _predict_with_classifier(
    extractor: LegalBERTFactExtractor,
    sentences: Sequence[str],
) -> tuple[list[int], list[float]]:
    preds: list[int] = []
    confs: list[float] = []
    batch_size = extractor.batch_size
    tokenizer = extractor.tokenizer
    model = extractor.model
    max_length = extractor.max_length
    device = extractor.device

    for i in range(0, len(sentences), batch_size):
        batch = list(sentences[i : i + batch_size])
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        batch_conf, batch_pred = probs.max(dim=-1)
        preds.extend(batch_pred.detach().cpu().tolist())
        confs.extend(batch_conf.detach().cpu().tolist())

    return preds, confs


def _parse_fact_ids(output: str, n_sentences: int) -> tuple[set[int], bool]:
    text = (output or "").strip()
    if not text:
        return set(), False

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

        raw_ids: Any = []
        if isinstance(payload, dict):
            for key in keys:
                if key in payload:
                    raw_ids = payload[key]
                    break
        elif isinstance(payload, list):
            raw_ids = payload

        if not isinstance(raw_ids, list):
            return set(), True

        parsed: set[int] = set()
        for item in raw_ids:
            try:
                idx = int(item)
            except (TypeError, ValueError):
                continue
            if 1 <= idx <= n_sentences:
                parsed.add(idx)
        return parsed, True

    return set(), False


def _predict_with_llm(
    llm_extractor: LLMFactsExtractor,
    sentences: Sequence[str],
    llm_chunk_size: int,
) -> tuple[list[int], int]:
    preds = [0] * len(sentences)
    parse_failures = 0

    for i in range(0, len(sentences), llm_chunk_size):
        chunk = list(sentences[i : i + llm_chunk_size])
        numbered_text = "\n".join(f"{idx}\t{sent}" for idx, sent in enumerate(chunk, start=1))
        raw = llm_extractor.extract_with_prompt(
            text=numbered_text,
            case_name="",
            docket_number="",
            system_prompt=LLM_SYSTEM_PROMPT,
            user_prompt_template=LLM_USER_PROMPT,
        )
        fact_ids, parsed_ok = _parse_fact_ids(raw, len(chunk))
        if parsed_ok:
            for local_idx in fact_ids:
                preds[i + (local_idx - 1)] = 1
            continue

        parse_failures += 1
        raw_norm = _norm(raw)
        for local_idx, sent in enumerate(chunk):
            if _norm(sent) in raw_norm:
                preds[i + local_idx] = 1

    return preds, parse_failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prediction-only classifier|llm|hybrid FACT tagging for JSON/JSONL documents.",
    )
    parser.add_argument("--input-json", type=Path, required=True, help="Input JSON or JSONL file.")
    parser.add_argument(
        "--mode",
        choices=["classifier", "llm", "hybrid"],
        default="hybrid",
        help="Prediction mode.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Classifier checkpoint path for classifier/hybrid mode.",
    )
    parser.add_argument("--device", type=str, default=None, help="Classifier device: cuda|cpu (default: auto).")
    parser.add_argument(
        "--classifier-batch-size",
        type=int,
        default=None,
        help="Override classifier batch size for classifier/hybrid modes.",
    )
    parser.add_argument("--llm-model", type=str, default=None, help="LLM model name for llm/hybrid mode.")
    parser.add_argument("--llm-chunk-size", type=int, default=120, help="Sentences per LLM request.")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.80,
        help="Hybrid mode: classifier confidence threshold below which LLM refinement is applied.",
    )
    parser.add_argument("--max-docs", type=int, default=None, help="Optional cap on number of documents.")
    parser.add_argument(
        "--max-sentences-per-doc",
        type=int,
        default=None,
        help="Optional cap on number of sentences per document.",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default=None,
        help="Explicit text field to use. If omitted, common raw-text fields are tried automatically.",
    )
    parser.add_argument(
        "--title-field",
        type=str,
        default=None,
        help="Explicit title field to use. If omitted, common id/title fields are tried automatically.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for prediction JSONL and summary JSON.",
    )
    parser.add_argument("--run-name", type=str, default=None, help="Optional run-name prefix.")
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Print progress every N evaluated docs (default: 10).",
    )
    args = parser.parse_args()

    input_path = args.input_json.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input JSON not found: {input_path}")
    if args.mode in {"classifier", "hybrid"} and not args.checkpoint.is_dir():
        raise FileNotFoundError(f"Classifier checkpoint directory not found: {args.checkpoint}")
    if args.llm_chunk_size <= 0:
        raise ValueError("--llm-chunk-size must be > 0")
    if args.mode == "hybrid" and not (0.0 <= args.tolerance <= 1.0):
        raise ValueError("--tolerance must be in [0, 1]")
    if args.classifier_batch_size is not None and args.classifier_batch_size <= 0:
        raise ValueError("--classifier-batch-size must be > 0 when provided")

    records = _load_records(input_path)
    if args.max_docs is not None:
        records = records[: args.max_docs]
    if not records:
        raise ValueError("No records found in input.")

    cfg = get_llm_config()
    llm_model = args.llm_model or cfg.get("model")
    run_name = args.run_name or f"fact_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    classifier: Optional[LegalBERTFactExtractor] = None
    llm_extractor: Optional[LLMFactsExtractor] = None

    if args.mode in {"classifier", "hybrid"}:
        classifier = LegalBERTFactExtractor(checkpoint_path=str(args.checkpoint), device=args.device)
        if args.classifier_batch_size is not None:
            classifier.batch_size = int(args.classifier_batch_size)
    if args.mode in {"llm", "hybrid"}:
        llm_extractor = LLMFactsExtractor(model_name=llm_model)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / f"{run_name}_predictions.jsonl"
    summary_path = output_dir / f"{run_name}_summary.json"
    debug_log_path = output_dir / f"{run_name}_debug.log"

    debug_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        debug_lines.append(msg)

    docs_total = 0
    docs_skipped = 0
    skip_reasons: Counter[str] = Counter()
    parse_failures_total = 0
    hybrid_refined_sentences_total = 0
    total_sentences = 0
    total_fact_sentences = 0
    per_doc_summary: list[dict[str, Any]] = []

    log(f"[start] mode={args.mode} input={input_path} records={len(records)}")
    if args.mode in {"classifier", "hybrid"}:
        log(f"[start] checkpoint={args.checkpoint.resolve()}")
    if args.mode in {"llm", "hybrid"}:
        log(f"[start] llm_model={llm_model} tolerance={args.tolerance if args.mode == 'hybrid' else 'n/a'}")

    with predictions_path.open("w", encoding="utf-8") as pred_writer:
        for idx, record in enumerate(records, start=1):
            text_key, text = _first_nonempty(
                record,
                (args.text_field,) if args.text_field else DEFAULT_TEXT_FIELD_CANDIDATES,
            )
            if not text:
                docs_skipped += 1
                skip_reasons["missing_text_field"] += 1
                if skip_reasons["missing_text_field"] <= 5:
                    log(
                        f"[skip] record_index={idx} reason=missing_text_field "
                        f"tried={((args.text_field,) if args.text_field else DEFAULT_TEXT_FIELD_CANDIDATES)}"
                    )
                continue

            _, title = _first_nonempty(
                record,
                (args.title_field,) if args.title_field else DEFAULT_TITLE_FIELD_CANDIDATES,
            )
            if not title:
                title = f"doc_{idx:05d}"

            sentences = split_sentences(text)
            if args.max_sentences_per_doc is not None:
                sentences = sentences[: args.max_sentences_per_doc]
            if not sentences:
                docs_skipped += 1
                skip_reasons["no_sentences_after_split"] += 1
                if skip_reasons["no_sentences_after_split"] <= 5:
                    log(f"[skip] record_index={idx} reason=no_sentences_after_split text_field={text_key}")
                continue

            if args.mode == "classifier":
                assert classifier is not None
                pred, classifier_conf = _predict_with_classifier(classifier, sentences)
                pred_source = ["classifier"] * len(sentences)
            elif args.mode == "llm":
                assert llm_extractor is not None
                pred, parse_fails = _predict_with_llm(llm_extractor, sentences, args.llm_chunk_size)
                classifier_conf = [None] * len(sentences)
                pred_source = ["llm"] * len(sentences)
                parse_failures_total += parse_fails
            else:
                assert classifier is not None and llm_extractor is not None
                classifier_pred, classifier_conf = _predict_with_classifier(classifier, sentences)
                pred = list(classifier_pred)
                pred_source = ["classifier"] * len(sentences)
                low_conf_indices = [i for i, c in enumerate(classifier_conf) if c < args.tolerance]
                if low_conf_indices:
                    low_conf_sents = [sentences[i] for i in low_conf_indices]
                    llm_pred, parse_fails = _predict_with_llm(llm_extractor, low_conf_sents, args.llm_chunk_size)
                    parse_failures_total += parse_fails
                    for local_i, sent_i in enumerate(low_conf_indices):
                        pred[sent_i] = llm_pred[local_i]
                        pred_source[sent_i] = "llm_refine"
                    hybrid_refined_sentences_total += len(low_conf_indices)

            pred_fact_count = int(sum(pred))
            docs_total += 1
            total_sentences += len(sentences)
            total_fact_sentences += pred_fact_count
            if args.log_every > 0 and docs_total % args.log_every == 0:
                log(
                    f"[progress] evaluated={docs_total} skipped={docs_skipped} "
                    f"sentences={total_sentences} facts={total_fact_sentences}"
                )

            pred_writer.write(
                json.dumps(
                    {
                        "title": title,
                        "case_id": record.get("case_id"),
                        "court": record.get("court"),
                        "source_record_index": idx,
                        "source_text_field": text_key,
                        "n_sentences": len(sentences),
                        "pred_fact_count": pred_fact_count,
                        "pred_nonfact_count": len(sentences) - pred_fact_count,
                        "predicted_extracted_facts": " ".join(
                            s for s, p in zip(sentences, pred) if p == 1
                        ),
                        "predicted_extracted_nonfacts": " ".join(
                            s for s, p in zip(sentences, pred) if p == 0
                        ),
                        "sentences": [
                            {
                                "id": i + 1,
                                "text": sent,
                                "pred_label": int(p),
                                "pred_source": src,
                                "classifier_confidence": None if c is None else float(c),
                            }
                            for i, (sent, p, src, c) in enumerate(
                                zip(sentences, pred, pred_source, classifier_conf)
                            )
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            per_doc_summary.append(
                {
                    "title": title,
                    "case_id": record.get("case_id"),
                    "court": record.get("court"),
                    "n_sentences": len(sentences),
                    "pred_fact_count": pred_fact_count,
                }
            )

    summary = {
        "run_name": run_name,
        "mode": args.mode,
        "input_json": str(input_path),
        "checkpoint": str(args.checkpoint.resolve()) if args.mode in {"classifier", "hybrid"} else None,
        "llm_model": llm_model if args.mode in {"llm", "hybrid"} else None,
        "classifier_batch_size": args.classifier_batch_size if args.mode in {"classifier", "hybrid"} else None,
        "tolerance": args.tolerance if args.mode == "hybrid" else None,
        "llm_chunk_size": args.llm_chunk_size if args.mode in {"llm", "hybrid"} else None,
        "docs_total": docs_total,
        "docs_skipped": docs_skipped,
        "skip_reasons": dict(skip_reasons),
        "total_sentences": total_sentences,
        "total_fact_sentences": total_fact_sentences,
        "parse_failures_total": parse_failures_total,
        "hybrid_refined_sentences_total": hybrid_refined_sentences_total,
        "avg_sentences_per_doc": (total_sentences / docs_total) if docs_total else 0.0,
        "avg_fact_sentences_per_doc": (total_fact_sentences / docs_total) if docs_total else 0.0,
        "outputs": {
            "predictions_jsonl": str(predictions_path),
            "summary_json": str(summary_path),
            "debug_log": str(debug_log_path),
        },
        "per_doc_summary": per_doc_summary,
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

    with debug_log_path.open("w", encoding="utf-8") as f:
        for line in debug_lines:
            f.write(line + "\n")

    log(
        f"[done] docs_total={docs_total} docs_skipped={docs_skipped} "
        f"predictions={predictions_path} summary={summary_path} debug={debug_log_path}"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
