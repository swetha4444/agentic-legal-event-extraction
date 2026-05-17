"""Multi-label JSONL dataset for SALI/LMSS transformer training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

ClassificationMode = Literal["multi", "single_singleton", "binary"]


def load_rows(path: Path, split: Optional[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if split is None or obj.get("split") == split:
                rows.append(obj)
    return rows


class JsonlMultiLabelDataset(Dataset):
    def __init__(
        self,
        rows: List[Dict[str, Any]],
        label_names: List[str],
        name_to_idx: Dict[str, int],
        tokenizer,
        text_field: str,
        max_length: int,
    ):
        self.rows = rows
        self.label_names = label_names
        self.name_to_idx = name_to_idx
        self.tokenizer = tokenizer
        self.text_field = text_field
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        r = self.rows[i]
        text = (r.get(self.text_field) or "").strip()
        if not text:
            text = "[empty]"
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        y = np.zeros(len(self.label_names), dtype=np.float32)
        for lab in r.get("labels") or []:
            j = self.name_to_idx.get(lab)
            if j is not None:
                y[j] = 1.0
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(y, dtype=torch.float32),
        }


def filter_rows_for_classification(
    rows: List[Dict[str, Any]],
    mode: ClassificationMode,
    name_to_idx: Dict[str, int],
    binary_target: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Returns (filtered_rows, num_skipped).

    - single_singleton: keep rows with exactly one known label (that label is the class).
    - binary: keep all rows; binary label derived from whether binary_target is in labels.
    - multi: no filtering.
    """
    if mode == "multi":
        return rows, 0
    skipped = 0
    out: List[Dict[str, Any]] = []
    if mode == "single_singleton":
        for r in rows:
            labs = [str(l) for l in (r.get("labels") or []) if str(l) in name_to_idx]
            if len(labs) != 1:
                skipped += 1
                continue
            out.append(r)
        return out, skipped
    if mode == "binary":
        if not binary_target or binary_target not in name_to_idx:
            raise ValueError("binary mode requires --binary_target matching a pref_label in label_json")
        return rows, 0
    raise ValueError(f"unknown mode {mode}")


class JsonlSingleLabelSingletonDataset(Dataset):
    """One gold class per row (23-way softmax). Rows must already be singleton-filtered."""

    def __init__(
        self,
        rows: List[Dict[str, Any]],
        label_names: List[str],
        name_to_idx: Dict[str, int],
        tokenizer,
        text_field: str,
        max_length: int,
    ):
        self.rows = rows
        self.label_names = label_names
        self.name_to_idx = name_to_idx
        self.tokenizer = tokenizer
        self.text_field = text_field
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        r = self.rows[i]
        text = (r.get(self.text_field) or "").strip()
        if not text:
            text = "[empty]"
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        labs = [str(l) for l in (r.get("labels") or []) if str(l) in self.name_to_idx]
        if len(labs) != 1:
            raise ValueError("JsonlSingleLabelSingletonDataset expects exactly one known label per row")
        y = self.name_to_idx[labs[0]]
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(y, dtype=torch.long),
        }


class JsonlBinaryLabelDataset(Dataset):
    """Binary: 1 if binary_target appears in row labels else 0."""

    def __init__(
        self,
        rows: List[Dict[str, Any]],
        name_to_idx: Dict[str, int],
        tokenizer,
        text_field: str,
        max_length: int,
        binary_target: str,
    ):
        if binary_target not in name_to_idx:
            raise ValueError(f"binary_target {binary_target!r} not in vocabulary")
        self.rows = rows
        self.binary_target = binary_target
        self.tokenizer = tokenizer
        self.text_field = text_field
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        r = self.rows[i]
        text = (r.get(self.text_field) or "").strip()
        if not text:
            text = "[empty]"
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        names = {str(l) for l in (r.get("labels") or [])}
        y = 1 if self.binary_target in names else 0
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(y, dtype=torch.long),
        }


def compute_pos_weight(train_rows: List[Dict[str, Any]], num_labels: int, name_to_idx: Dict[str, int]) -> torch.Tensor:
    pos = np.zeros(num_labels, dtype=np.float64)
    n = len(train_rows)
    for r in train_rows:
        for lab in r.get("labels") or []:
            j = name_to_idx.get(lab)
            if j is not None:
                pos[j] += 1.0
    neg = n - pos
    w = np.ones(num_labels, dtype=np.float64)
    mask = pos > 0
    w[mask] = neg[mask] / pos[mask]
    w[~np.isfinite(w)] = 1.0
    w = np.clip(w, 1.0, 100.0)
    return torch.tensor(w, dtype=torch.float32)
