#!/usr/bin/env python3
"""Evaluate SALI GNN checkpoint (BERT nodes + GCN) on test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from sali.ekg_graph_model import EKGGNNClassifier, EkgGraphBatchItem, graphs_from_row, load_ekg_map  # noqa: E402
from sali.multilabel_dataset import load_rows  # noqa: E402


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


def collate_graphs(batch: List[EkgGraphBatchItem]) -> List[EkgGraphBatchItem]:
    return batch


@torch.no_grad()
def predict_split(
    model: EKGGNNClassifier,
    tokenizer: AutoTokenizer,
    rows: List[dict],
    ekg_map: dict,
    name_to_idx: dict,
    num_labels: int,
    device: torch.device,
    max_node_tokens: int,
) -> tuple[np.ndarray, np.ndarray]:
    cpu = torch.device("cpu")
    logits_out: List[np.ndarray] = []
    y_out: List[np.ndarray] = []
    model.eval()
    for r in rows:
        did = str(r.get("document_id") or "").strip()
        ek = ekg_map.get(did)
        if not ek:
            continue
        merged = {"document_id": did, "labels": r.get("labels") or [], "merged_graph": ek.get("merged_graph") or {}}
        g = graphs_from_row(merged, name_to_idx, num_labels, cpu)
        if g is None:
            continue
        enc = tokenizer(
            g.node_texts,
            padding=True,
            truncation=True,
            max_length=max_node_tokens,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        node_emb = model.encode_nodes(enc["input_ids"], enc["attention_mask"])
        ei = g.edge_index.to(device)
        logits = model.forward_logits_from_node_batch(node_emb, ei).cpu().numpy()
        logits_out.append(logits)
        y_out.append(g.labels.numpy())
    return np.stack(logits_out, axis=0), np.stack(y_out, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--dataset_jsonl", type=Path, required=True)
    parser.add_argument(
        "--ekg_jsonl",
        type=Path,
        default=None,
        help="Default: ekg_jsonl from sali_gnn_meta.json",
    )
    parser.add_argument("--out_json", type=Path, default=None)
    args = parser.parse_args()

    meta = json.loads((args.checkpoint_dir / "sali_gnn_meta.json").read_text(encoding="utf-8"))
    ekg_path = args.ekg_jsonl or Path(meta["ekg_jsonl"])
    label_names: List[str] = meta["label_names"]
    name_to_idx = {n: i for i, n in enumerate(label_names)}
    num_labels = int(meta["num_labels"])
    max_node_tokens = int(meta.get("max_node_tokens", 64))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EKGGNNClassifier(
        meta["model_name"],
        num_labels=num_labels,
        gcn_hidden=int(meta.get("gcn_hidden", 256)),
        bert_trainable=False,
    ).to(device)
    state = torch.load(args.checkpoint_dir / "pytorch_model.bin", map_location=device)
    model.load_state_dict(state)
    tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint_dir))

    ekg_map = load_ekg_map(ekg_path.resolve())
    val_rows = load_rows(args.dataset_jsonl, "val")
    test_rows = load_rows(args.dataset_jsonl, "test")

    val_logits, val_y = predict_split(model, tokenizer, val_rows, ekg_map, name_to_idx, num_labels, device, max_node_tokens)
    grid = np.linspace(0.1, 0.9, 17)
    thr = tune_thresholds(val_logits, val_y, grid)

    test_logits, test_y = predict_split(model, tokenizer, test_rows, ekg_map, name_to_idx, num_labels, device, max_node_tokens)
    probs = _sigmoid(test_logits)
    pred_default = (probs >= 0.5).astype(np.float64)
    pred_tuned = (probs >= thr[None, :]).astype(np.float64)

    metrics = {
        "architecture": meta.get("architecture", "gnn"),
        "checkpoint_dir": str(args.checkpoint_dir),
        "n_val": val_logits.shape[0],
        "n_test": test_logits.shape[0],
        "thresholds": thr.tolist(),
        "test_default_0p5": multilabel_prf(test_y, pred_default),
        "test_tuned": multilabel_prf(test_y, pred_tuned),
    }
    out = args.out_json or (args.checkpoint_dir / "eval_metrics.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
