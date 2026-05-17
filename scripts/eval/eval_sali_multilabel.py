#!/usr/bin/env python3
"""
Evaluate a trained multi-label SALI classifier.

- Loads model + tokenizer from train_sali_multilabel output_dir
- Optionally tunes per-label thresholds on val split (greedy per label)
- Reports micro / macro precision, recall, F1 on test split

Requires: torch, transformers, numpy (.venv).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, default_data_collator

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

from sali.multilabel_dataset import (  # noqa: E402
    JsonlBinaryLabelDataset,
    JsonlMultiLabelDataset,
    JsonlSingleLabelSingletonDataset,
    filter_rows_for_classification,
    load_rows,
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def multilabel_prf(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp = np.sum((y_true == 1) & (y_pred == 1), axis=0, dtype=np.float64)
    fp = np.sum((y_true == 0) & (y_pred == 1), axis=0, dtype=np.float64)
    fn = np.sum((y_true == 1) & (y_pred == 0), axis=0, dtype=np.float64)

    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)

    micro_prec = float(np.sum(tp) / (np.sum(tp) + np.sum(fp) + 1e-12))
    micro_rec = float(np.sum(tp) / (np.sum(tp) + np.sum(fn) + 1e-12))
    micro_f1 = float(2 * micro_prec * micro_rec / (micro_prec + micro_rec + 1e-12))

    macro_prec = float(np.mean(prec))
    macro_rec = float(np.mean(rec))
    macro_f1 = float(np.mean(f1))
    return {
        "micro_precision": micro_prec,
        "micro_recall": micro_rec,
        "micro_f1": micro_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
    }


def tune_thresholds(logits: np.ndarray, y_true: np.ndarray, grid: np.ndarray) -> np.ndarray:
    probs = _sigmoid(logits)
    L = y_true.shape[1]
    thr = np.full(L, 0.5, dtype=np.float64)
    for j in range(L):
        best = (0.5, 0.0)
        for t in grid:
            pred = (probs[:, j] >= t).astype(np.float64)
            m = multilabel_prf(y_true[:, j : j + 1], pred[:, None])
            f1 = m["macro_f1"]
            if f1 > best[1]:
                best = (float(t), f1)
        thr[j] = best[0]
    return thr


@torch.no_grad()
def predict_logits(model, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    all_logits: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    for batch in loader:
        labels = batch["labels"].to(device)
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        logits = model(**batch).logits
        all_logits.append(logits.cpu().numpy())
        all_y.append(labels.cpu().numpy())
    xl = np.vstack(all_logits)
    if not all_y:
        return xl, np.array([])
    if all_y[0].ndim == 1:
        return xl, np.concatenate(all_y, axis=0)
    return xl, np.vstack(all_y)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SALI multi-label classifier")
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--dataset_jsonl", type=Path, required=True)
    parser.add_argument("--text_field", choices=("text", "ekg_text"), default="text")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--out_json", type=Path, default=None)
    args = parser.parse_args()

    meta_path = args.checkpoint_dir / "sali_multilabel_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    label_names: List[str] = meta["label_names"]
    name_to_idx = {n: i for i, n in enumerate(label_names)}
    max_len = int(meta.get("max_length", args.max_length))
    clf_mode = str(meta.get("classification_mode", "multi"))

    ckpt = str(args.checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    val_rows = load_rows(args.dataset_jsonl, "val")
    test_rows = load_rows(args.dataset_jsonl, "test")
    binary_target = meta.get("binary_target")

    if clf_mode == "multi":
        val_ds = JsonlMultiLabelDataset(val_rows, label_names, name_to_idx, tokenizer, args.text_field, max_len)
        test_ds = JsonlMultiLabelDataset(test_rows, label_names, name_to_idx, tokenizer, args.text_field, max_len)
    elif clf_mode == "single_singleton":
        val_rows, _ = filter_rows_for_classification(val_rows, "single_singleton", name_to_idx, None)
        test_rows, _ = filter_rows_for_classification(test_rows, "single_singleton", name_to_idx, None)
        val_ds = JsonlSingleLabelSingletonDataset(
            val_rows, label_names, name_to_idx, tokenizer, args.text_field, max_len
        )
        test_ds = JsonlSingleLabelSingletonDataset(
            test_rows, label_names, name_to_idx, tokenizer, args.text_field, max_len
        )
    else:
        val_ds = JsonlBinaryLabelDataset(
            val_rows, name_to_idx, tokenizer, args.text_field, max_len, str(binary_target or "")
        )
        test_ds = JsonlBinaryLabelDataset(
            test_rows, name_to_idx, tokenizer, args.text_field, max_len, str(binary_target or "")
        )

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=default_data_collator)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=default_data_collator)

    val_logits, val_y = predict_logits(model, val_loader, device)
    test_logits, test_y = predict_logits(model, test_loader, device)

    if clf_mode == "multi":
        grid = np.linspace(0.1, 0.9, 17)
        thr = tune_thresholds(val_logits, val_y, grid)
        probs = _sigmoid(test_logits)
        pred_default = (probs >= 0.5).astype(np.float64)
        pred_tuned = (probs >= thr[None, :]).astype(np.float64)
        metrics = {
            "classification_mode": clf_mode,
            "text_field": args.text_field,
            "checkpoint_dir": str(args.checkpoint_dir),
            "n_val": len(val_ds),
            "n_test": len(test_ds),
            "thresholds": thr.tolist(),
            "test_default_0p5": multilabel_prf(test_y, pred_default),
            "test_tuned": multilabel_prf(test_y, pred_tuned),
        }
    else:
        test_pred = np.argmax(test_logits, axis=1)
        metrics = {
            "classification_mode": clf_mode,
            "binary_target": binary_target,
            "text_field": args.text_field,
            "checkpoint_dir": str(args.checkpoint_dir),
            "n_val": len(val_ds),
            "n_test": len(test_ds),
            "test_accuracy": float(accuracy_score(test_y, test_pred)),
            "test_macro_f1": float(f1_score(test_y, test_pred, average="macro", zero_division=0)),
            "test_weighted_f1": float(f1_score(test_y, test_pred, average="weighted", zero_division=0)),
        }
        if clf_mode == "binary":
            metrics["test_n_positive_gold"] = int(np.sum(test_y == 1))
            metrics["test_n_positive_pred"] = int(np.sum(test_pred == 1))
            p, r, f, _ = precision_recall_fscore_support(
                test_y, test_pred, average="binary", pos_label=1, zero_division=0
            )
            metrics["test_precision_positive"] = float(p)
            metrics["test_recall_positive"] = float(r)
            metrics["test_f1_positive"] = float(f)

    out = args.out_json or (args.checkpoint_dir / "eval_metrics.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
