"""Homogeneous graph view of merged EKG + simple GCN stack for multi-label SALI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn


def _trunc(s: str, n: int) -> str:
    s = (s or "").strip().replace("\n", " ")
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."


def merged_graph_to_homogeneous(
    merged_graph: Dict[str, Any],
    max_entities: int = 80,
    max_events: int = 120,
    max_temporal: int = 60,
    max_causal: int = 60,
) -> Tuple[List[str], torch.Tensor, int]:
    """
    Build one undirected graph: entity nodes first, then event nodes.

    Returns (node_texts, edge_index [2, E] long CPU, num_entities).
    """
    entities = list((merged_graph or {}).get("entities") or [])[:max_entities]
    events = list((merged_graph or {}).get("events") or [])[:max_events]
    temporal = list((merged_graph or {}).get("temporal_edges") or [])[:max_temporal]
    causal = list((merged_graph or {}).get("causal_edges") or [])[:max_causal]

    node_texts: List[str] = []
    ent_id_to_idx: Dict[str, int] = {}
    ev_id_to_idx: Dict[str, int] = {}

    edges: List[Tuple[int, int]] = []

    for i, ent in enumerate(entities):
        eid = str(ent.get("entity_id") or "").strip()
        nm = str(ent.get("name") or "").strip() or "(unnamed)"
        typ = str(ent.get("type") or "").strip() or "UNKNOWN"
        role = str(ent.get("canonical_role") or "").strip()
        role_s = f" role={role}" if role else ""
        node_texts.append(_trunc(f"ENTITY | name={nm} | type={typ}{role_s}", 256))
        if eid:
            ent_id_to_idx[eid] = i

    n_ent = len(entities)
    for j, ev in enumerate(events):
        idx = n_ent + j
        ev_id = str(ev.get("event_id") or "").strip()
        if not ev_id:
            ev_id = f"EV{j + 1}"
        ev_id_to_idx[ev_id] = idx

        et = str(ev.get("event_type") or ev.get("type") or "UNKNOWN").strip()
        trig = ""
        t = ev.get("trigger") or {}
        if isinstance(t, dict):
            trig = str(t.get("span_text") or "").strip()
        desc = str(ev.get("description") or "").strip()
        if desc and desc != trig:
            desc = _trunc(desc, 120)
        else:
            desc = ""
        desc_s = f" | description={desc}" if desc else ""
        node_texts.append(_trunc(f"EVENT | type={et} | trigger={trig}{desc_s}", 320))

        for p in ev.get("participants") or []:
            if not isinstance(p, dict):
                continue
            eid = str(p.get("entity_id") or "").strip()
            ei = ent_id_to_idx.get(eid)
            if ei is None:
                continue
            edges.append((ei, idx))
            edges.append((idx, ei))

    def _wire_event_edge(ev_from: str, ev_to: str) -> None:
        a = ev_id_to_idx.get(str(ev_from).strip())
        b = ev_id_to_idx.get(str(ev_to).strip())
        if a is None or b is None or a == b:
            return
        edges.append((a, b))
        edges.append((b, a))

    for e in temporal:
        if not isinstance(e, dict):
            continue
        _wire_event_edge(str(e.get("from_event") or ""), str(e.get("to_event") or ""))

    for e in causal:
        if not isinstance(e, dict):
            continue
        _wire_event_edge(str(e.get("from_event") or ""), str(e.get("to_event") or ""))

    if not node_texts:
        node_texts = ["[EMPTY_GRAPH]"]
        return node_texts, torch.zeros(2, 0, dtype=torch.long), 0

    n = len(node_texts)
    for i in range(n):
        edges.append((i, i))

    if not edges:
        edges = [(0, 0)]

    row, col = zip(*edges)
    edge_index = torch.tensor([row, col], dtype=torch.long)
    return node_texts, edge_index, n_ent


@dataclass
class EkgGraphBatchItem:
    document_id: str
    node_texts: List[str]
    edge_index: torch.Tensor
    num_entities: int
    labels: torch.Tensor


class GraphConvLayer(nn.Module):
    """Mean aggregation over neighbors + self (undirected edges expected)."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.lin_n = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_s = nn.Linear(in_dim, out_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n, _ = x.shape
        device = x.device
        dtype = x.dtype
        neigh_sum = torch.zeros_like(x)
        deg = torch.zeros(n, device=device, dtype=dtype)
        if edge_index.numel() > 0:
            row, col = edge_index[0], edge_index[1]
            mask = row != col
            row, col = row[mask], col[mask]
            if row.numel() > 0:
                neigh_sum.index_add_(0, col, x[row])
                deg.index_add_(0, col, torch.ones(col.shape[0], device=device, dtype=dtype))
        deg = deg.clamp(min=1.0).unsqueeze(-1)
        agg = neigh_sum / deg
        out = torch.relu(self.lin_n(agg) + self.lin_s(x))
        return self.dropout(out)


class EKGGNNClassifier(nn.Module):
    """
    Per-node CLS embedding from Legal-BERT, two GCN layers, mean pool, multi-label logits.
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        gcn_hidden: int = 256,
        bert_trainable: bool = False,
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        self.bert = AutoModel.from_pretrained(model_name)
        d = self.bert.config.hidden_size
        if not bert_trainable:
            for p in self.bert.parameters():
                p.requires_grad = False
        self.proj = nn.Linear(d, gcn_hidden)
        self.gcn1 = GraphConvLayer(gcn_hidden, gcn_hidden)
        self.gcn2 = GraphConvLayer(gcn_hidden, gcn_hidden)
        self.head = nn.Linear(gcn_hidden, num_labels)

    def encode_nodes(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.proj(cls)

    def forward_graph(
        self,
        node_emb: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        h = self.gcn1(node_emb, edge_index)
        h = self.gcn2(h, edge_index)
        pooled = h.mean(dim=0)
        return self.head(pooled)

    def forward_logits_from_node_batch(
        self,
        node_emb: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_graph(node_emb, edge_index)


def load_ekg_map(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            did = str(row.get("doc_id") or "").strip()
            if did:
                out[did] = row
    return out


def graphs_from_row(
    row: Dict[str, Any],
    name_to_idx: Dict[str, int],
    num_labels: int,
    device: torch.device,
) -> EkgGraphBatchItem | None:
    mg = row.get("merged_graph")
    if not isinstance(mg, dict):
        return None
    node_texts, edge_index, n_ent = merged_graph_to_homogeneous(mg)
    y = torch.zeros(num_labels, dtype=torch.float32)
    for lab in row.get("labels") or []:
        j = name_to_idx.get(str(lab))
        if j is not None:
            y[j] = 1.0
    return EkgGraphBatchItem(
        document_id=str(row.get("document_id") or ""),
        node_texts=node_texts,
        edge_index=edge_index.to(device),
        num_entities=n_ent,
        labels=y.to(device),
    )
