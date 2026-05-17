#!/usr/bin/env python3
"""
Fine-tune a transformer for multi-label SALI/LMSS classification.

Modes:
  --text_field text   : raw complaint body (baseline)
  --text_field ekg_text : linearized EKG (serialized graph)

Loss:
- multi: BCE-with-logits + optional pos_weight (default).
- single_singleton: rows with exactly one label; Cross-entropy (23-way).
- binary: --binary_target; positive if that label is in the row's label list; 2-way CE.

Requires: torch, transformers, numpy (project .venv).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from sali.multilabel_dataset import (  # noqa: E402
    JsonlBinaryLabelDataset,
    JsonlMultiLabelDataset,
    JsonlSingleLabelSingletonDataset,
    compute_pos_weight,
    filter_rows_for_classification,
    load_rows,
)

for _name in ("transformers.modeling_utils", "transformers"):
    logging.getLogger(_name).setLevel(logging.WARNING)


class WeightedBCETrainer(Trainer):
    def __init__(self, pos_weight: Optional[torch.Tensor], *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        pw = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pw)
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multi-label SALI classifier (raw or EKG text)")
    parser.add_argument("--dataset_jsonl", type=Path, required=True)
    parser.add_argument("--label_json", type=Path, default=ROOT / "data/sali/lmss_claim_subset_v1.json")
    parser.add_argument("--text_field", choices=("text", "ekg_text"), default="text")
    parser.add_argument("--model_name", type=str, default="nlpaueb/legal-bert-base-uncased")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--output_dir", type=Path, default=ROOT / "data/models/sali_multilabel_raw")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_pos_weight", action="store_true", help="Disable BCE pos_weight (multi only)")
    parser.add_argument(
        "--classification_mode",
        choices=("multi", "single_singleton", "binary"),
        default="multi",
        help="multi=multi-label BCE; single_singleton=exactly one label per kept row; binary=--binary_target yes/no",
    )
    parser.add_argument(
        "--binary_target",
        type=str,
        default=None,
        help="For mode=binary: pref_label string that defines the positive class",
    )
    args = parser.parse_args()

    label_meta = json.loads(args.label_json.read_text(encoding="utf-8"))
    label_names = [x["pref_label"] for x in label_meta["labels"]]
    name_to_idx = {n: i for i, n in enumerate(label_names)}
    num_labels = len(label_names)

    train_rows = load_rows(args.dataset_jsonl, "train")
    val_rows = load_rows(args.dataset_jsonl, "val")
    if len(train_rows) < 5 or len(val_rows) < 2:
        print("Need more train/val rows; rebuild dataset with more docs.", file=sys.stderr)
        sys.exit(1)

    mode = args.classification_mode
    if mode == "binary" and not args.binary_target:
        print("--binary_target is required for --classification_mode binary", file=sys.stderr)
        sys.exit(1)

    train_rows_f, sk_tr = filter_rows_for_classification(train_rows, mode, name_to_idx, args.binary_target)
    val_rows_f, sk_va = filter_rows_for_classification(val_rows, mode, name_to_idx, args.binary_target)
    if sk_tr or sk_va:
        print(f"Filtered rows: train skipped={sk_tr}, val skipped={sk_va} (mode={mode})", flush=True)

    if mode == "single_singleton" and (len(train_rows_f) < 5 or len(val_rows_f) < 2):
        print(
            f"Not enough singleton-label rows (train={len(train_rows_f)}, val={len(val_rows_f)}); "
            "use multi/binary or more data.",
            file=sys.stderr,
        )
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if mode == "multi":
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=num_labels,
            problem_type="multi_label_classification",
        )
        train_ds = JsonlMultiLabelDataset(
            train_rows_f, label_names, name_to_idx, tokenizer, args.text_field, args.max_length
        )
        val_ds = JsonlMultiLabelDataset(
            val_rows_f, label_names, name_to_idx, tokenizer, args.text_field, args.max_length
        )
        pos_w = None if args.no_pos_weight else compute_pos_weight(train_rows_f, num_labels, name_to_idx)
        out_num_labels = num_labels
    elif mode == "single_singleton":
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=num_labels,
            problem_type="single_label_classification",
        )
        train_ds = JsonlSingleLabelSingletonDataset(
            train_rows_f, label_names, name_to_idx, tokenizer, args.text_field, args.max_length
        )
        val_ds = JsonlSingleLabelSingletonDataset(
            val_rows_f, label_names, name_to_idx, tokenizer, args.text_field, args.max_length
        )
        out_num_labels = num_labels
        pos_w = None
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=2,
            problem_type="single_label_classification",
        )
        train_ds = JsonlBinaryLabelDataset(
            train_rows_f, name_to_idx, tokenizer, args.text_field, args.max_length, args.binary_target or ""
        )
        val_ds = JsonlBinaryLabelDataset(
            val_rows_f, name_to_idx, tokenizer, args.text_field, args.max_length, args.binary_target or ""
        )
        out_num_labels = 2
        pos_w = None

    use_cuda = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        seed=args.seed,
        dataloader_pin_memory=use_cuda,
        logging_steps=10,
    )

    if mode == "multi":
        trainer = WeightedBCETrainer(
            pos_w,
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=default_data_collator,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=default_data_collator,
        )
    trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    meta = {
        "model_name": args.model_name,
        "text_field": args.text_field,
        "max_length": args.max_length,
        "classification_mode": mode,
        "binary_target": args.binary_target,
        "num_labels": out_num_labels,
        "num_classes_vocab": num_labels,
        "label_names": label_names,
        "dataset_jsonl": str(args.dataset_jsonl),
        "pos_weight": pos_w.tolist() if pos_w is not None else None,
    }
    (args.output_dir / "sali_multilabel_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Saved model to {args.output_dir}")


if __name__ == "__main__":
    main()
