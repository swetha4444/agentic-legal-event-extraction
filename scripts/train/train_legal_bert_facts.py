#!/usr/bin/env python3
"""
Train LegalBERT for sentence-level fact vs non-fact classification.
Input: JSONL with one object per line: {"sentence": "...", "label": 0|1 or "fact"|"non_fact"}.
       Label 1 / "fact" = court-established fact; 0 / "non_fact" = procedure, reasoning, etc.
Output: Saved model + tokenizer in --output_dir (for use with LegalBERTFactExtractor).
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict

# Suppress verbose "UNEXPECTED/MISSING" load report when loading BERT for classification (expected)
for _name in ("transformers.modeling_utils", "transformers"):
    logging.getLogger(_name).setLevel(logging.WARNING)

# Project root (script lives in scripts/train/)
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, default_data_collator


def normalize_label(val) -> int:
    if val in (0, 1):
        return int(val)
    if isinstance(val, str):
        v = val.strip().lower()
        if v in ("1", "fact", "facts"):
            return 1
        if v in ("0", "non_fact", "non-fact", "nonfact"):
            return 0
    raise ValueError(f"Label must be 0/1 or fact/non_fact, got {val!r}")


class SentenceLabelDataset(Dataset):
    def __init__(self, sentences: list[str], labels: list[int], tokenizer, max_length: int = 256):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, i):
        enc = self.tokenizer(
            self.sentences[i],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[i], dtype=torch.long),
        }


def load_jsonl(path: str) -> tuple[list[str], list[int]]:
    sentences = []
    labels = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("sentence") or obj.get("text") or ""
            if not text.strip():
                continue
            lab = normalize_label(obj["label"])
            sentences.append(text.strip())
            labels.append(lab)
    return sentences, labels


def compute_binary_metrics(eval_pred) -> Dict[str, float]:
    """Compute binary classification metrics for FACT (label=1)."""
    logits, labels = eval_pred
    pred = np.argmax(logits, axis=-1)
    labels = np.asarray(labels)

    tp = int(np.sum((pred == 1) & (labels == 1)))
    tn = int(np.sum((pred == 0) & (labels == 0)))
    fp = int(np.sum((pred == 1) & (labels == 0)))
    fn = int(np.sum((pred == 0) & (labels == 1)))
    n = int(labels.shape[0])

    acc = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "accuracy": float(acc),
        "precision_fact": float(precision),
        "recall_fact": float(recall),
        "f1_fact": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main():
    parser = argparse.ArgumentParser(description="Train LegalBERT for fact vs non-fact sentence classification")
    parser.add_argument("--data", required=True, help="JSONL: each line {\"sentence\": \"...\", \"label\": 0|1 or \"fact\"|\"non_fact\"}")
    parser.add_argument("--output_dir", default="data/models/legal_bert_facts", help="Where to save model and tokenizer")
    parser.add_argument("--model_name", default="nlpaueb/legal-bert-base-uncased", help="Base LegalBERT model")
    parser.add_argument("--max_length", type=int, default=256, help="Max tokens per sentence")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Fraction of data for validation (0 = no val)")
    parser.add_argument("--max_samples", type=int, default=None, help="Cap training data size (for low-memory sanity runs, e.g. 500)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.isfile(args.data):
        print(f"Error: data file not found: {args.data}")
        sys.exit(1)

    sentences, labels = load_jsonl(args.data)
    if args.max_samples is not None and len(sentences) > args.max_samples:
        sentences, labels = sentences[: args.max_samples], labels[: args.max_samples]
        print(f"Using --max_samples={args.max_samples}: training on {len(sentences)} examples.")
    if len(sentences) < 10:
        print(f"Warning: only {len(sentences)} samples; consider adding more data (e.g. from rhetorical-role corpus).")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print(f"Loading {args.model_name} (classification head is new; training will learn it)...")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    torch.manual_seed(args.seed)
    n = len(sentences)
    idx = torch.randperm(n)
    val_size = max(0, int(n * args.val_ratio))
    train_idx = idx[val_size:]
    val_idx = idx[:val_size]

    train_sentences = [sentences[i] for i in train_idx.tolist()]
    train_labels = [labels[i] for i in train_idx.tolist()]
    train_ds = SentenceLabelDataset(train_sentences, train_labels, tokenizer, args.max_length)

    eval_ds = None
    if val_size > 0:
        eval_sentences = [sentences[i] for i in val_idx.tolist()]
        eval_labels = [labels[i] for i in val_idx.tolist()]
        eval_ds = SentenceLabelDataset(eval_sentences, eval_labels, tokenizer, args.max_length)

    use_cuda = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_ds else "no",
        load_best_model_at_end=bool(eval_ds),
        metric_for_best_model="eval_loss" if eval_ds else None,
        dataloader_pin_memory=use_cuda,
    )

    device = "cuda" if use_cuda else "cpu"
    print(f"Using device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if use_cuda else ""))
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_binary_metrics if eval_ds else None,
        data_collator=default_data_collator,
    )
    trainer.train()
    if eval_ds:
        eval_metrics = trainer.evaluate(eval_dataset=eval_ds)
        metrics_path = Path(args.output_dir) / "validation_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(eval_metrics, f, indent=2)
            f.write("\n")
        print(f"Saved validation metrics to {metrics_path}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved model and tokenizer to {args.output_dir}. Use this path as checkpoint_path in LegalBERTFactExtractor.")


if __name__ == "__main__":
    main()
