#!/usr/bin/env python3
"""
Evaluate FACT vs NON-FACT sentence classification on MARRO UK docs listed in a compiled JSON.

Modes:
1) classifier: legal-domain classifier only (default: LexLM MARRO checkpoint if available)
2) llm: LLM-only classification with JSON id outputs
3) hybrid: classifier first, then LLM only on low-confidence sentences

Input:
- Compiled JSON from scripts/data/compile_marro_uk_to_json.py
  (must contain at least `title`; sentence-level gold labels are loaded from <dataset_dir>/<title>.txt)

Output:
- Metrics JSON and predictions JSONL in output directory.

Default classifier preference follows legal-domain PLM literature:
- LEGAL-BERT (Findings EMNLP 2020): https://aclanthology.org/2020.findings-emnlp.261/
- LeXFiles/LegalLAMA (ACL 2023, LexLM models): https://aclanthology.org/2023.acl-long.865/
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import torch
except ImportError as exc:
    raise ImportError("torch is required for classifier/hybrid mode. Install with: pip install torch") from exc

from agents.facts import LLMFactsExtractor
from processing.legal_bert_facts import LegalBERTFactExtractor, split_sentences


DEFAULT_INPUT_JSON = ROOT / "data" / "processed_json" / "marro_uk_compiled.json"
DEFAULT_DATASET_DIR = ROOT / "data" / "MARRO" / "UK-dataset"
DEFAULT_OUTPUT_DIR = Path("/work/pi_dagarwal_umass_edu/project_1/sriram/outputs/fact_extraction")
DEFAULT_CHECKPOINT_CANDIDATES = [
    ROOT / "data" / "models" / "lexlm_facts_marro",
    ROOT / "data" / "models" / "legal_bert_facts_marro",
    ROOT / "data" / "models" / "legal_bert_facts_base",
]

LLM_SYSTEM_PROMPT = """You are a legal annotation assistant.
Classify numbered sentences as FACT vs NON-FACT for judicial documents.
Return strict JSON only, no prose."""

LLM_USER_PROMPT = """Given the numbered sentences below, identify which sentence ids are FACT.

FACT means concrete events or accepted factual narrative (who did what, what happened, when/where).
NON-FACT includes legal argument, legal reasoning, holdings, citations, and procedural boilerplate.

Return strict JSON only:
{{"fact_sentence_ids": [1, 4, 9]}}

Rules:
- Use only ids from the list.
- Do not paraphrase or quote sentences.
- If no facts, return: {{"fact_sentence_ids": []}}

SENTENCES:
{text}
"""


def resolve_default_checkpoint() -> Path:
    for candidate in DEFAULT_CHECKPOINT_CANDIDATES:
        if candidate.is_dir():
            return candidate
    return DEFAULT_CHECKPOINT_CANDIDATES[0]


def load_compiled_json(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON array in {path}, got {type(payload).__name__}")
    return payload


def load_marro_doc_with_labels(path: Path) -> Tuple[List[str], List[int]]:
    sentences: List[str] = []
    gold: List[int] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            sent, label = line.rsplit("\t", 1)
            sent = sent.strip()
            if not sent:
                continue
            sentences.append(sent)
            gold.append(1 if label.strip().upper() == "FAC" else 0)
    return sentences, gold


def _normalize_label(value) -> Optional[int]:
    if value in (0, 1):
        return int(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "fac", "fact", "facts"}:
            return 1
        if v in {"0", "nonfact", "non_fact", "non-fact", "other", "sta", "arg", "ratio", "rlc", "pre"}:
            return 0
    return None


def load_doc_from_record(record: dict, dataset_dir: Path) -> Tuple[List[str], List[int], str]:
    title = (record.get("title") or "").strip()
    if title:
        txt_path = dataset_dir / f"{title}.txt"
        if txt_path.is_file():
            sents, gold = load_marro_doc_with_labels(txt_path)
            if sents:
                return sents, gold, f"title_txt:{txt_path}"

    if isinstance(record.get("sentences"), list):
        sents: List[str] = []
        gold: List[int] = []
        for item in record["sentences"]:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or item.get("sentence") or "").strip()
            label = _normalize_label(item.get("label"))
            if text and label is not None:
                sents.append(text)
                gold.append(label)
        if sents:
            return sents, gold, "record.sentences"

    facts = split_sentences(str(record.get("extracted_facts") or ""))
    nonfacts = split_sentences(str(record.get("extracted_nonfacts") or ""))
    if facts or nonfacts:
        # Fallback if no labeled txt is available.
        # Order is facts then non-facts (document order is lost in this fallback).
        sents = facts + nonfacts
        gold = [1] * len(facts) + [0] * len(nonfacts)
        return sents, gold, "fallback_split_extracted_fields"

    return [], [], "empty"


def predict_with_classifier(
    extractor: LegalBERTFactExtractor, sentences: Sequence[str]
) -> Tuple[List[int], List[float]]:
    preds: List[int] = []
    confs: List[float] = []
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


def parse_fact_ids(output: str, n_sentences: int) -> Tuple[set, bool]:
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


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def predict_with_llm(
    llm_extractor: LLMFactsExtractor,
    sentences: Sequence[str],
    llm_chunk_size: int,
) -> Tuple[List[int], int]:
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
        fact_ids, parsed_ok = parse_fact_ids(raw, len(chunk))
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


def binary_metrics(gold: Sequence[int], pred: Sequence[int]) -> Dict[str, float]:
    if len(gold) != len(pred):
        raise ValueError(f"gold/pred length mismatch: {len(gold)} vs {len(pred)}")

    tp = sum(1 for g, p in zip(gold, pred) if g == 1 and p == 1)
    tn = sum(1 for g, p in zip(gold, pred) if g == 0 and p == 0)
    fp = sum(1 for g, p in zip(gold, pred) if g == 0 and p == 1)
    fn = sum(1 for g, p in zip(gold, pred) if g == 1 and p == 0)
    n = len(gold)
    acc = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "n": n,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": acc,
        "precision_fact": precision,
        "recall_fact": recall,
        "f1_fact": f1,
    }


def aggregate_metrics(per_doc: Sequence[dict]) -> dict:
    if not per_doc:
        return {
            "docs": 0,
            "sentences": 0,
            "micro": binary_metrics([], []),
            "macro": {"accuracy": 0.0, "precision_fact": 0.0, "recall_fact": 0.0, "f1_fact": 0.0},
        }

    tp = sum(d["tp"] for d in per_doc)
    tn = sum(d["tn"] for d in per_doc)
    fp = sum(d["fp"] for d in per_doc)
    fn = sum(d["fn"] for d in per_doc)
    n = sum(d["n"] for d in per_doc)
    micro_acc = (tp + tn) / n if n else 0.0
    micro_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    micro_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0
    micro = {
        "n": n,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": micro_acc,
        "precision_fact": micro_prec,
        "recall_fact": micro_rec,
        "f1_fact": micro_f1,
    }

    macro = {
        "accuracy": sum(d["accuracy"] for d in per_doc) / len(per_doc),
        "precision_fact": sum(d["precision_fact"] for d in per_doc) / len(per_doc),
        "recall_fact": sum(d["recall_fact"] for d in per_doc) / len(per_doc),
        "f1_fact": sum(d["f1_fact"] for d in per_doc) / len(per_doc),
    }
    return {"docs": len(per_doc), "sentences": n, "micro": micro, "macro": macro}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate FACT/NON-FACT classification in classifier|llm|hybrid mode using compiled MARRO JSON."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=DEFAULT_INPUT_JSON,
        help="Compiled JSON path (default: data/processed_json/marro_uk_compiled.json).",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Directory containing source labeled MARRO .txt files.",
    )
    parser.add_argument(
        "--mode",
        choices=["classifier", "llm", "hybrid"],
        default="classifier",
        help="Evaluation mode.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=resolve_default_checkpoint(),
        help="Classifier checkpoint path (default: best available local legal-domain checkpoint).",
    )
    parser.add_argument("--device", type=str, default=None, help="Classifier device: cuda|cpu (default: auto).")
    parser.add_argument(
        "--classifier-batch-size",
        type=int,
        default=None,
        help="Override classifier batch size for classifier/hybrid modes (default: checkpoint extractor default).",
    )
    parser.add_argument("--llm-model", type=str, default=None, help="LLM model name (for llm/hybrid modes).")
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
        help="Optional cap on number of sentences per document (useful for quick sanity runs).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for metrics and predictions.",
    )
    parser.add_argument("--run-name", type=str, default=None, help="Optional run-name prefix.")
    parser.add_argument("--no-predictions", action="store_true", help="Skip predictions JSONL output.")
    args = parser.parse_args()

    if not args.input_json.is_file():
        raise FileNotFoundError(f"Input JSON not found: {args.input_json}")
    if args.mode in {"classifier", "hybrid"} and not args.checkpoint.is_dir():
        raise FileNotFoundError(f"Classifier checkpoint directory not found: {args.checkpoint}")
    if args.llm_chunk_size <= 0:
        raise ValueError("--llm-chunk-size must be > 0")
    if args.mode == "hybrid" and not (0.0 <= args.tolerance <= 1.0):
        raise ValueError("--tolerance must be in [0, 1]")
    if args.classifier_batch_size is not None and args.classifier_batch_size <= 0:
        raise ValueError("--classifier-batch-size must be > 0 when provided")

    compiled_records = load_compiled_json(args.input_json)
    if args.max_docs is not None:
        compiled_records = compiled_records[: args.max_docs]
    if not compiled_records:
        raise ValueError("No records found in input JSON.")

    classifier: Optional[LegalBERTFactExtractor] = None
    llm_extractor: Optional[LLMFactsExtractor] = None

    if args.mode in {"classifier", "hybrid"}:
        classifier = LegalBERTFactExtractor(checkpoint_path=str(args.checkpoint), device=args.device)
        if args.classifier_batch_size is not None:
            classifier.batch_size = int(args.classifier_batch_size)
    if args.mode in {"llm", "hybrid"}:
        llm_extractor = LLMFactsExtractor(model_name=args.llm_model)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{args.mode}_{stamp}"
    metrics_path = args.output_dir / f"{run_name}_metrics.json"
    predictions_path = args.output_dir / f"{run_name}_predictions.jsonl"

    dataset_dir = args.dataset_dir.resolve()

    per_doc_metrics: List[dict] = []
    docs_total = 0
    docs_skipped = 0
    parse_failures_total = 0
    hybrid_refined_sentences_total = 0

    pred_writer = None
    if not args.no_predictions:
        pred_writer = predictions_path.open("w", encoding="utf-8")

    try:
        for idx, record in enumerate(compiled_records, start=1):
            title = (record.get("title") or f"doc_{idx:04d}").strip()
            sentences, gold, label_source = load_doc_from_record(record, dataset_dir=dataset_dir)
            if not sentences:
                docs_skipped += 1
                continue
            if args.max_sentences_per_doc is not None:
                if args.max_sentences_per_doc <= 0:
                    raise ValueError("--max-sentences-per-doc must be > 0 when provided")
                sentences = sentences[: args.max_sentences_per_doc]
                gold = gold[: args.max_sentences_per_doc]

            if args.mode == "classifier":
                assert classifier is not None
                pred, classifier_conf = predict_with_classifier(classifier, sentences)
                pred_source = ["classifier"] * len(sentences)
            elif args.mode == "llm":
                assert llm_extractor is not None
                pred, parse_fails = predict_with_llm(llm_extractor, sentences, args.llm_chunk_size)
                classifier_conf = [None] * len(sentences)
                pred_source = ["llm"] * len(sentences)
                parse_failures_total += parse_fails
            else:
                assert classifier is not None and llm_extractor is not None
                classifier_pred, classifier_conf = predict_with_classifier(classifier, sentences)
                pred = list(classifier_pred)
                pred_source = ["classifier"] * len(sentences)
                low_conf_indices = [i for i, c in enumerate(classifier_conf) if c < args.tolerance]
                if low_conf_indices:
                    low_conf_sents = [sentences[i] for i in low_conf_indices]
                    llm_pred, parse_fails = predict_with_llm(llm_extractor, low_conf_sents, args.llm_chunk_size)
                    parse_failures_total += parse_fails
                    for local_i, sent_i in enumerate(low_conf_indices):
                        pred[sent_i] = llm_pred[local_i]
                        pred_source[sent_i] = "llm_refine"
                    hybrid_refined_sentences_total += len(low_conf_indices)

            doc_metrics = binary_metrics(gold, pred)
            doc_metrics["title"] = title
            doc_metrics["label_source"] = label_source
            per_doc_metrics.append(doc_metrics)
            docs_total += 1

            if pred_writer is not None:
                pred_writer.write(
                    json.dumps(
                        {
                            "title": title,
                            "label_source": label_source,
                            "n_sentences": len(sentences),
                            "gold_fact_count": int(sum(gold)),
                            "pred_fact_count": int(sum(pred)),
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
                                    "gold_label": int(g),
                                    "pred_label": int(p),
                                    "pred_source": src,
                                    "classifier_confidence": None if c is None else float(c),
                                }
                                for i, (sent, g, p, src, c) in enumerate(
                                    zip(sentences, gold, pred, pred_source, classifier_conf)
                                )
                            ],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    finally:
        if pred_writer is not None:
            pred_writer.close()

    aggregate = aggregate_metrics(per_doc_metrics)
    run_summary = {
        "run_name": run_name,
        "mode": args.mode,
        "input_json": str(args.input_json.resolve()),
        "dataset_dir": str(dataset_dir),
        "checkpoint": str(args.checkpoint.resolve()) if args.mode in {"classifier", "hybrid"} else None,
        "llm_model": args.llm_model if args.mode in {"llm", "hybrid"} else None,
        "classifier_batch_size": args.classifier_batch_size if args.mode in {"classifier", "hybrid"} else None,
        "tolerance": args.tolerance if args.mode == "hybrid" else None,
        "llm_chunk_size": args.llm_chunk_size if args.mode in {"llm", "hybrid"} else None,
        "docs_total": docs_total,
        "docs_skipped": docs_skipped,
        "parse_failures_total": parse_failures_total,
        "hybrid_refined_sentences_total": hybrid_refined_sentences_total,
        "aggregate": aggregate,
        "per_doc_metrics": per_doc_metrics,
        "outputs": {
            "metrics_json": str(metrics_path),
            "predictions_jsonl": None if args.no_predictions else str(predictions_path),
        },
        "timestamp": datetime.now().isoformat(),
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

    micro = aggregate["micro"]
    macro = aggregate["macro"]
    print("=" * 72)
    print(f"Mode: {args.mode}")
    print(f"Docs evaluated: {docs_total} (skipped: {docs_skipped})")
    print(f"Sentences: {aggregate['sentences']}")
    print(
        "Micro  | "
        f"Acc={micro['accuracy']:.4f} "
        f"P={micro['precision_fact']:.4f} "
        f"R={micro['recall_fact']:.4f} "
        f"F1={micro['f1_fact']:.4f}"
    )
    print(
        "Macro  | "
        f"Acc={macro['accuracy']:.4f} "
        f"P={macro['precision_fact']:.4f} "
        f"R={macro['recall_fact']:.4f} "
        f"F1={macro['f1_fact']:.4f}"
    )
    if args.mode in {"llm", "hybrid"}:
        print(f"LLM parse fallback count: {parse_failures_total}")
    if args.mode == "hybrid":
        print(f"Hybrid refined sentences (confidence < {args.tolerance:.2f}): {hybrid_refined_sentences_total}")
    print(f"Saved metrics: {metrics_path}")
    if not args.no_predictions:
        print(f"Saved predictions: {predictions_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
