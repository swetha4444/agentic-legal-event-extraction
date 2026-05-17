#!/usr/bin/env python3
"""
Train a graph model (BERT node encodings + 2-layer GCN) for multi-label SALI/LMSS.

Joins `dataset_jsonl` rows with `ekg_jsonl` on document_id == doc_id to recover merged_graph.

Requires: torch, transformers (same .venv as Legal-BERT baseline).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from sali.ekg_graph_model import EKGGNNClassifier, EkgGraphBatchItem, graphs_from_row, load_ekg_map  # noqa: E402
from sali.multilabel_dataset import compute_pos_weight, load_rows  # noqa: E402


class GraphDocDataset(Dataset):
    def __init__(
        self,
        rows: List[Dict[str, Any]],
        ekg_map: Dict[str, Dict[str, Any]],
        name_to_idx: Dict[str, int],
        num_labels: int,
    ):
        self.items: List[EkgGraphBatchItem] = []
        cpu = torch.device("cpu")
        for r in rows:
            did = str(r.get("document_id") or "").strip()
            ek = ekg_map.get(did)
            if not ek:
                continue
            merged = {"document_id": did, "labels": r.get("labels") or [], "merged_graph": ek.get("merged_graph") or {}}
            g = graphs_from_row(merged, name_to_idx, num_labels, cpu)
            if g is not None:
                self.items.append(g)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> EkgGraphBatchItem:
        return self.items[i]


def collate_graphs(batch: List[EkgGraphBatchItem]) -> List[EkgGraphBatchItem]:
    return batch


def train_one_epoch(
    model: EKGGNNClassifier,
    tokenizer: AutoTokenizer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    pos_weight: torch.Tensor | None,
    device: torch.device,
    max_node_tokens: int,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        logits_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []
        for g in batch:
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
            logits = model.forward_logits_from_node_batch(node_emb, ei)
            logits_list.append(logits)
            labels_list.append(g.labels.to(device))
        logits_b = torch.stack(logits_list, dim=0)
        labels_b = torch.stack(labels_list, dim=0)
        pw = pos_weight.to(device) if pos_weight is not None else None
        loss = F.binary_cross_entropy_with_logits(logits_b, labels_b, pos_weight=pw)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss.item())
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def eval_loss(
    model: EKGGNNClassifier,
    tokenizer: AutoTokenizer,
    loader: DataLoader,
    pos_weight: torch.Tensor | None,
    device: torch.device,
    max_node_tokens: int,
) -> float:
    model.eval()
    total = 0.0
    n = 0
    for batch in loader:
        for g in batch:
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
            logits = model.forward_logits_from_node_batch(node_emb, ei).unsqueeze(0)
            lab = g.labels.to(device).unsqueeze(0)
            pw = pos_weight.to(device) if pos_weight is not None else None
            loss = F.binary_cross_entropy_with_logits(logits, lab, pos_weight=pw)
            total += float(loss.item())
            n += 1
    return total / max(n, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GCN over EKG graphs for SALI multi-label")
    parser.add_argument("--dataset_jsonl", type=Path, required=True)
    parser.add_argument(
        "--ekg_jsonl",
        type=Path,
        default=ROOT / "data/outputs/qwen122b_hybrid_doc_graphs_final.jsonl",
        help="Doc-level merged_graph source (doc_id key)",
    )
    parser.add_argument("--label_json", type=Path, default=ROOT / "data/sali/lmss_claim_subset_v1.json")
    parser.add_argument("--model_name", type=str, default="nlpaueb/legal-bert-base-uncased")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "data/models/sali_gnn_ekg")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gcn_hidden", type=int, default=256)
    parser.add_argument("--max_node_tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_pos_weight", action="store_true")
    parser.add_argument("--finetune_bert", action="store_true", help="Train BERT weights (slower, more capacity)")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_meta = json.loads(args.label_json.read_text(encoding="utf-8"))
    label_names = [x["pref_label"] for x in label_meta["labels"]]
    name_to_idx = {n: i for i, n in enumerate(label_names)}
    num_labels = len(label_names)

    train_rows = load_rows(args.dataset_jsonl, "train")
    val_rows = load_rows(args.dataset_jsonl, "val")
    ekg_map = load_ekg_map(args.ekg_jsonl)

    train_ds = GraphDocDataset(train_rows, ekg_map, name_to_idx, num_labels)
    val_ds = GraphDocDataset(val_rows, ekg_map, name_to_idx, num_labels)
    if len(train_ds) < 4 or len(val_ds) < 2:
        print("Not enough graphs after EKG join; check paths and ids.", file=sys.stderr)
        sys.exit(1)

    pos_w = None if args.no_pos_weight else compute_pos_weight(train_rows, num_labels, name_to_idx)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = EKGGNNClassifier(
        args.model_name,
        num_labels=num_labels,
        gcn_hidden=args.gcn_hidden,
        bert_trainable=args.finetune_bert,
    ).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        print("No trainable parameters; use --finetune_bert or check module.", file=sys.stderr)
        sys.exit(1)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_graphs,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_graphs)

    best_val = float("inf")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, tokenizer, train_loader, optimizer, pos_w, device, args.max_node_tokens)
        va = eval_loss(model, tokenizer, val_loader, pos_w, device, args.max_node_tokens)
        print(f"epoch {epoch} train_loss={tr:.4f} val_loss={va:.4f}")
        if va < best_val:
            best_val = va
            torch.save(model.state_dict(), args.output_dir / "pytorch_model.bin")
            tokenizer.save_pretrained(str(args.output_dir))

    meta: Dict[str, Any] = {
        "model_name": args.model_name,
        "architecture": "bert_node_cls_gcn2_meanpool",
        "gcn_hidden": args.gcn_hidden,
        "num_labels": num_labels,
        "label_names": label_names,
        "dataset_jsonl": str(args.dataset_jsonl),
        "ekg_jsonl": str(args.ekg_jsonl),
        "max_node_tokens": args.max_node_tokens,
        "finetune_bert": args.finetune_bert,
        "pos_weight": pos_w.tolist() if pos_w is not None else None,
    }
    (args.output_dir / "sali_gnn_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Saved best checkpoint to {args.output_dir} (val_loss={best_val:.4f})")


if __name__ == "__main__":
    main()
