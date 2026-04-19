#!/usr/bin/env python3
"""
Hybrid doc-level merge:
1) deterministic merge of chunk mini-graphs
2) embedding-based event candidate retrieval
3) LLM adjudication for same_event / temporal / causal links
4) optional event collapse from same_event decisions
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __name__ == "__main__":
    _src = Path(__file__).resolve().parent
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from openai import OpenAI

from agents.config_loader import get_llm_config
from agents.events import MiniGraphMergeAgent


_LOCAL_HF_EMBED_CACHE: dict[str, tuple[Any, Any, Any]] = {}


HYBRID_LINK_SYSTEM = """You adjudicate whether two legal events are the same, and whether a directed temporal/causal relation should be added.

Output ONLY valid JSON with this exact schema:
{
  "same_event": boolean,
  "temporal": {"add": boolean, "relation": string, "direction": "A_to_B"|"B_to_A"|"none"},
  "causal": {"add": boolean, "relation": string, "direction": "A_to_B"|"B_to_A"|"none"},
  "confidence": number,
  "rationale_short": string
}

Rules:
- Prefer precision over recall; avoid speculative links.
- `same_event=true` only when both records describe the same real-world occurrence.
- For temporal/causal, set `add=true` only when evidence is explicit in the provided context.
- Use `direction="none"` when uncertain.
- Use the provided source chunk context and event evidence to reason about chronology and causation.
- Keep relations concise (e.g., BEFORE, AFTER, OVERLAP, CAUSES, ENABLES, PREVENTS, RELATED_TO).
"""


def _strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()
    return text


def _ensure_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _clip_text(text: str, limit: int = 240) -> str:
    text = re.sub(r"\s+", " ", _safe_text(text))
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _normalized_label(value: Any) -> str:
    return re.sub(r"\s+", " ", _safe_text(value).lower()).strip()


def _event_trigger_text(event: dict[str, Any]) -> str:
    trigger = event.get("trigger") or {}
    if isinstance(trigger, dict):
        return _safe_text(trigger.get("span_text") or trigger.get("text"))
    return _safe_text(trigger)


def _event_time_anchor(event: dict[str, Any]) -> str:
    time_obj = event.get("time") or {}
    if not isinstance(time_obj, dict):
        return ""
    return _safe_text(time_obj.get("normalized") or time_obj.get("raw_span"))


def _build_chunk_lookup(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for chunk in _ensure_list(doc.get("chunks")):
        if not isinstance(chunk, dict):
            continue
        chunk_id = _safe_text(chunk.get("chunk_id"))
        if not chunk_id:
            continue
        lookup[chunk_id] = {
            "theme": _safe_text(chunk.get("theme")),
            "text": _safe_text(chunk.get("text")),
            "start_sentence_id": chunk.get("start_sentence_id"),
            "end_sentence_id": chunk.get("end_sentence_id"),
        }
    return lookup


def _event_source_contexts(event: dict[str, Any], chunk_lookup: dict[str, dict[str, Any]] | None, *, limit: int = 3) -> list[dict[str, Any]]:
    if not chunk_lookup:
        return []
    contexts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ref in _ensure_list(event.get("source_refs")):
        if not isinstance(ref, dict):
            continue
        chunk_id = _safe_text(ref.get("chunk_id"))
        if not chunk_id or chunk_id in seen:
            continue
        meta = chunk_lookup.get(chunk_id)
        if not meta:
            continue
        contexts.append(
            {
                "chunk_id": chunk_id,
                "theme": _safe_text(meta.get("theme")),
                "sentence_window": {
                    "start": meta.get("start_sentence_id"),
                    "end": meta.get("end_sentence_id"),
                },
                "text_excerpt": _clip_text(_safe_text(meta.get("text")), limit=520),
            }
        )
        seen.add(chunk_id)
        if len(contexts) >= limit:
            break
    return contexts


def _event_participant_rows(
    event: dict[str, Any],
    *,
    entity_name_by_id: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for p in _ensure_list(event.get("participants")):
        if not isinstance(p, dict):
            continue
        role = _safe_text(p.get("role")).upper() or "CONTEXT"
        ent_id = _safe_text(p.get("entity_id"))
        ent_name = _safe_text((entity_name_by_id or {}).get(ent_id))
        mention = p.get("mention") or {}
        mention_text = ""
        if isinstance(mention, dict):
            mention_text = _safe_text(mention.get("span_text") or mention.get("text"))
        rows.append(
            {
                "role": role,
                "entity_id": ent_id,
                "entity_name": ent_name,
                "mention_text": mention_text,
            }
        )
    return rows


def _event_sentence_ids(event: dict[str, Any]) -> set[int]:
    out: set[int] = set()
    evidence = event.get("evidence") or {}
    if isinstance(evidence, dict):
        out.update(int(x) for x in _ensure_list(evidence.get("sentence_ids")) if isinstance(x, int))
    trigger = event.get("trigger") or {}
    if isinstance(trigger, dict):
        sid = trigger.get("sentence_id")
        if isinstance(sid, int):
            out.add(int(sid))
    return out


def _event_sentence_bounds(event: dict[str, Any]) -> tuple[int | None, int | None]:
    sids = sorted(_event_sentence_ids(event))
    if not sids:
        return None, None
    return sids[0], sids[-1]


def _event_participant_ids(event: dict[str, Any]) -> set[str]:
    return {
        row["entity_id"]
        for row in _event_participant_rows(event)
        if row.get("entity_id")
    }


def _event_participant_names(
    event: dict[str, Any],
    *,
    entity_name_by_id: dict[str, str] | None = None,
) -> set[str]:
    out: set[str] = set()
    for row in _event_participant_rows(event, entity_name_by_id=entity_name_by_id):
        for value in (row.get("entity_name"), row.get("mention_text")):
            label = _normalized_label(value)
            if label:
                out.add(label)
    return out


def _event_source_chunk_ids(event: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for ref in _ensure_list(event.get("source_refs")):
        if isinstance(ref, dict):
            cid = _safe_text(ref.get("chunk_id"))
            if cid:
                out.append(cid)
    return sorted(set(out))


def _event_summary(
    event: dict[str, Any],
    *,
    entity_name_by_id: dict[str, str] | None = None,
    chunk_lookup: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    participants = _event_participant_rows(event, entity_name_by_id=entity_name_by_id)
    snippets: list[str] = []
    evidence = event.get("evidence") or {}
    if isinstance(evidence, dict):
        for s in _ensure_list(evidence.get("snippets"))[:2]:
            if isinstance(s, dict):
                snippets.append(_clip_text(_safe_text(s.get("text")), limit=200))
            else:
                snippets.append(_clip_text(_safe_text(s), limit=200))
    first_sid, last_sid = _event_sentence_bounds(event)
    return {
        "event_id": _safe_text(event.get("event_id")),
        "event_type": _safe_text(event.get("event_type")),
        "main_verb": _safe_text(event.get("main_verb")),
        "trigger": _event_trigger_text(event),
        "time_anchor": _event_time_anchor(event),
        "sentence_window": {"start": first_sid, "end": last_sid},
        "participants": [
            {
                "role": row.get("role"),
                "entity_id": row.get("entity_id"),
                "entity_name": row.get("entity_name"),
                "mention_text": row.get("mention_text"),
            }
            for row in participants[:4]
        ],
        "snippets": snippets,
        "source_chunks": _event_source_chunk_ids(event),
        "source_contexts": _event_source_contexts(event, chunk_lookup, limit=2),
    }


def _event_identity_snapshot(
    event: dict[str, Any],
    *,
    index: int,
    entity_name_by_id: dict[str, str] | None = None,
) -> dict[str, Any]:
    participants = _event_participant_rows(event, entity_name_by_id=entity_name_by_id)
    return {
        "event_id": _safe_text(event.get("event_id")),
        "event_index": int(index),
        "event_key": f"{_safe_text(event.get('event_id'))}@@idx:{index}",
        "event_type": _safe_text(event.get("event_type")),
        "trigger": _event_trigger_text(event),
        "time_anchor": _event_time_anchor(event),
        "sentence_ids": sorted(_event_sentence_ids(event)),
        "source_chunks": _event_source_chunk_ids(event),
        "participants": [
            {
                "role": row.get("role"),
                "entity_id": row.get("entity_id"),
                "entity_name": row.get("entity_name"),
                "mention_text": row.get("mention_text"),
            }
            for row in participants[:5]
        ],
    }


def _event_sort_key(event: dict[str, Any], index: int) -> tuple[int, str, int]:
    first_sid, _ = _event_sentence_bounds(event)
    return (
        first_sid if first_sid is not None else 10**9,
        _event_time_anchor(event) or "zzzz",
        index,
    )


def _event_neighborhood(
    events: list[dict[str, Any]],
    focus_index: int,
    *,
    entity_name_by_id: dict[str, str] | None = None,
    chunk_lookup: dict[str, dict[str, Any]] | None = None,
    radius: int = 2,
) -> list[dict[str, Any]]:
    if not (0 <= focus_index < len(events)):
        return []
    ordered = sorted(range(len(events)), key=lambda idx: _event_sort_key(events[idx], idx))
    try:
        pos = ordered.index(focus_index)
    except ValueError:
        return []
    window = ordered[max(0, pos - radius) : min(len(ordered), pos + radius + 1)]
    out = []
    for idx in window:
        summary = _event_summary(events[idx], entity_name_by_id=entity_name_by_id, chunk_lookup=chunk_lookup)
        summary["relative_position"] = (
            "focus" if idx == focus_index else ("before" if ordered.index(idx) < pos else "after")
        )
        out.append(summary)
    return out


def _event_text_bundle(
    event: dict[str, Any],
    *,
    entity_name_by_id: dict[str, str] | None = None,
    chunk_lookup: dict[str, dict[str, Any]] | None = None,
) -> str:
    participant_rows = _event_participant_rows(event, entity_name_by_id=entity_name_by_id)
    participants = [
        f"{row['role']}:{row['entity_id']}:{row['entity_name']}:{row['mention_text']}"
        for row in participant_rows
    ]
    evidence = event.get("evidence") or {}
    snippets = []
    if isinstance(evidence, dict):
        for s in _ensure_list(evidence.get("snippets"))[:3]:
            if isinstance(s, dict):
                snippets.append(_safe_text(s.get("text")))
            else:
                snippets.append(_safe_text(s))
    trigger_text = _event_trigger_text(event)
    source_contexts = _event_source_contexts(event, chunk_lookup)
    chunks_str = ",".join(_event_source_chunk_ids(event))
    participant_names = sorted(
        {
            label
            for label in (
                _normalized_label(row.get("entity_name")) or _normalized_label(row.get("mention_text"))
                for row in participant_rows
            )
            if label
        }
    )
    first_sid, last_sid = _event_sentence_bounds(event)
    return " | ".join(
        [
            f"event_id={_safe_text(event.get('event_id'))}",
            f"type={_safe_text(event.get('event_type')).upper()}",
            f"verb={_safe_text(event.get('main_verb'))}",
            f"trigger={trigger_text}",
            f"trigger_sentence_id={_safe_text((event.get('trigger') or {}).get('sentence_id') if isinstance(event.get('trigger'), dict) else '')}",
            f"time={json.dumps(event.get('time') or {}, ensure_ascii=False)}",
            f"time_anchor={_event_time_anchor(event)}",
            f"sentence_window={json.dumps({'start': first_sid, 'end': last_sid})}",
            f"participants={';'.join(participants)}",
            f"participant_names={';'.join(participant_names)}",
            f"evidence_sids={json.dumps(_ensure_list(evidence.get('sentence_ids')) if isinstance(evidence, dict) else [])}",
            f"evidence_context={';'.join(_clip_text(text, limit=220) for text in snippets)}",
            f"snippets={';'.join(snippets)}",
            f"source_chunks={chunks_str}",
            f"source_contexts={json.dumps(source_contexts, ensure_ascii=False)}",
        ]
    )


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", _safe_text(text).lower())


def _bow(text: str) -> Counter[str]:
    return Counter(_tokenize(text))


def _cosine_counter(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    keys = set(a.keys()) | set(b.keys())
    dot = sum(float(a[k]) * float(b[k]) for k in keys)
    na = math.sqrt(sum(float(v) * float(v) for v in a.values()))
    nb = math.sqrt(sum(float(v) * float(v) for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _is_local_hf_embedding_model(model_name: str) -> bool:
    name = _safe_text(model_name)
    return name.startswith("huggingface/")


def _hf_repo_id(model_name: str) -> str:
    name = _safe_text(model_name)
    if name.startswith("huggingface/"):
        return name.split("/", 1)[1]
    return name


def _get_local_hf_embedder(model_name: str) -> tuple[Any, Any, Any]:
    repo_id = _hf_repo_id(model_name)
    cached = _LOCAL_HF_EMBED_CACHE.get(repo_id)
    if cached is not None:
        return cached

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "Local Hugging Face embeddings require 'torch' and 'transformers' to be installed."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModel.from_pretrained(repo_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    _LOCAL_HF_EMBED_CACHE[repo_id] = (tokenizer, model, device)
    return tokenizer, model, device


def _embed_texts_local_hf(model_name: str, texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    tokenizer, model, device = _get_local_hf_embedder(model_name)

    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:
        raise RuntimeError(
            "Local Hugging Face embeddings require 'torch' to be installed."
        ) from exc

    out: list[list[float]] = []
    batch_size = 16
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            masked = token_embeddings * attention_mask
            summed = masked.sum(dim=1)
            counts = attention_mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / counts
            pooled = F.normalize(pooled, p=2, dim=1)
        out.extend(pooled.detach().cpu().tolist())
    return out


def _make_client(model_name: str | None) -> tuple[OpenAI, str, float]:
    cfg = get_llm_config()
    api_key = cfg.get("api_key")
    api_base = cfg.get("api_base")
    if not api_key:
        raise ValueError("Missing AGENT_API_KEY / llm.api_key in config")
    model = model_name or cfg.get("model") or "gpt-4o"
    model_lower = model.lower()
    temperature = 1.0 if "gpt-5" in model_lower or model_lower == "gpt5" else float(cfg.get("temperature", 0.0) or 0.0)
    return OpenAI(api_key=api_key, base_url=api_base, timeout=180.0), model, temperature


def _embed_texts(client: OpenAI, embedding_model: str, texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    if _is_local_hf_embedding_model(embedding_model):
        return _embed_texts_local_hf(embedding_model, texts)
    response = client.embeddings.create(model=embedding_model, input=texts)
    rows = sorted(response.data, key=lambda x: x.index)
    return [list(r.embedding) for r in rows]


def _event_chunk_set(event: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for ref in _ensure_list(event.get("source_refs")):
        if isinstance(ref, dict):
            cid = _safe_text(ref.get("chunk_id"))
            if cid:
                out.add(cid)
    return out


def _shared_time_token_count(time_a: str, time_b: str) -> int:
    if not time_a or not time_b:
        return 0
    toks_a = set(_tokenize(time_a))
    toks_b = set(_tokenize(time_b))
    return len(toks_a & toks_b)


def _candidate_feature_map(
    a: dict[str, Any],
    b: dict[str, Any],
    *,
    entity_name_by_id: dict[str, str] | None = None,
) -> dict[str, Any]:
    sids_a = _event_sentence_ids(a)
    sids_b = _event_sentence_ids(b)
    parts_a = _event_participant_ids(a)
    parts_b = _event_participant_ids(b)
    part_names_a = _event_participant_names(a, entity_name_by_id=entity_name_by_id)
    part_names_b = _event_participant_names(b, entity_name_by_id=entity_name_by_id)
    shared_sids = sorted(sids_a & sids_b)
    shared_participants = sorted(parts_a & parts_b)
    shared_participant_names = sorted(part_names_a & part_names_b)
    time_a = _event_time_anchor(a)
    time_b = _event_time_anchor(b)
    trigger_a = _normalized_label(_event_trigger_text(a))
    trigger_b = _normalized_label(_event_trigger_text(b))
    same_trigger = bool(trigger_a and trigger_a == trigger_b)
    sentence_gap: int | None = None
    if sids_a and sids_b:
        sentence_gap = min(abs(x - y) for x in sids_a for y in sids_b)
    time_token_overlap = _shared_time_token_count(time_a, time_b)
    return {
        "shared_sentence_ids": shared_sids,
        "shared_sentence": bool(shared_sids),
        "shared_participants": shared_participants,
        "shared_participant_count": len(shared_participants),
        "shared_participant_names": shared_participant_names,
        "shared_participant_name_count": len(shared_participant_names),
        "shared_time": bool(time_a and time_b and time_a == time_b),
        "shared_time_token_count": time_token_overlap,
        "time_a": time_a,
        "time_b": time_b,
        "same_trigger": same_trigger,
        "trigger_a": trigger_a,
        "trigger_b": trigger_b,
        "sentence_gap": sentence_gap,
        "nearby_sentence_window": sentence_gap is not None and sentence_gap <= 2,
        "same_main_verb": bool(
            _safe_text(a.get("main_verb")) and _safe_text(a.get("main_verb")).lower() == _safe_text(b.get("main_verb")).lower()
        ),
        "same_event_type": bool(
            _safe_text(a.get("event_type")) and _safe_text(a.get("event_type")).lower() == _safe_text(b.get("event_type")).lower()
        ),
    }


def _pair_candidates(
    events: list[dict[str, Any]],
    embeds: list[list[float]],
    top_k: int,
    min_similarity: float,
) -> list[tuple[int, int, float]]:
    if not events or len(events) != len(embeds):
        return []
    per_i: list[list[tuple[float, int]]] = [[] for _ in events]
    for i in range(len(events)):
        ci = _event_chunk_set(events[i])
        for j in range(i + 1, len(events)):
            cj = _event_chunk_set(events[j])
            # Prefer cross-chunk linking: skip if clearly same chunk sources.
            if ci and cj and not ci.isdisjoint(cj):
                continue
            sim = _cosine(embeds[i], embeds[j])
            if sim < min_similarity:
                continue
            per_i[i].append((sim, j))
            per_i[j].append((sim, i))
    candidate_pairs: set[tuple[int, int]] = set()
    for i, nbrs in enumerate(per_i):
        nbrs = sorted(nbrs, key=lambda x: x[0], reverse=True)[: max(1, top_k)]
        for _, j in nbrs:
            a, b = sorted((i, j))
            candidate_pairs.add((a, b))
    out = []
    for i, j in sorted(candidate_pairs):
        out.append((i, j, _cosine(embeds[i], embeds[j])))
    return out


def _pair_candidates_local_lexical(
    events: list[dict[str, Any]],
    top_k: int,
    min_similarity: float,
    *,
    entity_name_by_id: dict[str, str] | None = None,
    chunk_lookup: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if not events:
        return []
    bundles = [
        _event_text_bundle(ev, entity_name_by_id=entity_name_by_id, chunk_lookup=chunk_lookup)
        for ev in events
    ]
    bows = [_bow(bundle) for bundle in bundles]
    duplicate_quota = max(1, top_k // 2)
    relation_quota = max(1, top_k - duplicate_quota)
    rescue_similarity = max(0.42, min_similarity - 0.18)
    duplicate_neighbors: list[list[tuple[float, int, float, dict[str, Any]]]] = [[] for _ in events]
    relation_neighbors: list[list[tuple[float, int, float, dict[str, Any]]]] = [[] for _ in events]
    for i in range(len(events)):
        ci = _event_chunk_set(events[i])
        for j in range(i + 1, len(events)):
            cj = _event_chunk_set(events[j])
            if ci and cj and not ci.isdisjoint(cj):
                continue
            sim = _cosine_counter(bows[i], bows[j])
            features = _candidate_feature_map(events[i], events[j], entity_name_by_id=entity_name_by_id)
            duplicate_like = (
                features["shared_sentence"]
                or (
                    sim >= min_similarity
                    and (
                        features["same_main_verb"]
                        or features["same_event_type"]
                        or features["same_trigger"]
                        or features["shared_participant_count"] > 0
                        or features["shared_participant_name_count"] > 0
                    )
                )
                or (
                    sim >= rescue_similarity
                    and features["same_trigger"]
                    and (features["shared_participant_count"] > 0 or features["shared_participant_name_count"] > 0)
                )
            )
            relation_like = (
                sim >= rescue_similarity
                and (
                    features["shared_participant_count"] > 0
                    or features["shared_participant_name_count"] > 0
                    or features["shared_time"]
                    or features["shared_time_token_count"] > 0
                    or features["same_main_verb"]
                    or features["same_event_type"]
                    or features["same_trigger"]
                    or features["nearby_sentence_window"]
                )
                and not features["shared_sentence"]
            )
            anchor_biased_relation = (
                sim >= max(0.30, rescue_similarity - 0.08)
                and not features["shared_sentence"]
                and (
                    features["shared_time"]
                    or features["shared_time_token_count"] > 0
                    or features["shared_participant_count"] >= 2
                    or features["shared_participant_name_count"] >= 1
                    or features["nearby_sentence_window"]
                )
            )
            if duplicate_like:
                dup_priority = (
                    sim
                    + (0.40 if features["shared_sentence"] else 0.0)
                    + (0.10 if features["same_main_verb"] else 0.0)
                    + (0.08 if features["same_event_type"] else 0.0)
                    + (0.09 if features["same_trigger"] else 0.0)
                    + (0.06 if features["shared_participant_count"] > 0 else 0.0)
                    + (0.05 if features["shared_participant_name_count"] > 0 else 0.0)
                )
                duplicate_neighbors[i].append((dup_priority, j, sim, features))
                duplicate_neighbors[j].append((dup_priority, i, sim, features))
            if relation_like or anchor_biased_relation:
                rel_priority = (
                    sim
                    + (0.18 if features["shared_time"] else 0.0)
                    + (0.12 if features["shared_time_token_count"] > 0 else 0.0)
                    + (0.12 if features["shared_participant_count"] > 0 else 0.0)
                    + (0.08 if features["shared_participant_name_count"] > 0 else 0.0)
                    + (0.08 if features["shared_participant_count"] >= 2 else 0.0)
                    + (0.05 if features["same_main_verb"] else 0.0)
                    + (0.06 if features["same_trigger"] else 0.0)
                    + (0.10 if features["nearby_sentence_window"] else 0.0)
                )
                relation_neighbors[i].append((rel_priority, j, sim, features))
                relation_neighbors[j].append((rel_priority, i, sim, features))
    candidate_map: dict[tuple[int, int], dict[str, Any]] = {}
    for i, nbrs in enumerate(duplicate_neighbors):
        for priority, j, sim, features in sorted(nbrs, key=lambda x: x[0], reverse=True)[:duplicate_quota]:
            a, b = sorted((i, j))
            candidate_map[(a, b)] = {
                "i": a,
                "j": b,
                "score": sim,
                "priority": priority,
                "bucket": "duplicate",
                "rescue": False,
                "features": features,
            }
    for i, nbrs in enumerate(relation_neighbors):
        for priority, j, sim, features in sorted(nbrs, key=lambda x: x[0], reverse=True)[:relation_quota]:
            a, b = sorted((i, j))
            existing = candidate_map.get((a, b))
            row = {
                "i": a,
                "j": b,
                "score": sim,
                "priority": priority,
                "bucket": "relation",
                "rescue": sim < min_similarity,
                "features": features,
            }
            if existing is None or existing.get("bucket") != "relation" or priority > float(existing.get("priority", 0.0)):
                candidate_map[(a, b)] = row
    out = []
    for _, row in sorted(candidate_map.items(), key=lambda item: (item[1]["bucket"] != "relation", -item[1]["priority"], -item[1]["score"])):
        out.append(row)
    return out


def _semantic_rescue_candidates(
    *,
    client: OpenAI,
    semantic_model: str,
    events: list[dict[str, Any]],
    lexical_rows: list[dict[str, Any]],
    min_similarity: float,
    max_candidates: int,
    entity_name_by_id: dict[str, str] | None = None,
    chunk_lookup: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if not semantic_model or not events or max_candidates <= 0:
        return []

    existing = {(int(row["i"]), int(row["j"])) for row in lexical_rows}
    bundles = [
        _event_text_bundle(ev, entity_name_by_id=entity_name_by_id, chunk_lookup=chunk_lookup)
        for ev in events
    ]
    bows = [_bow(bundle) for bundle in bundles]
    shortlist: list[dict[str, Any]] = []
    shortlist_floor = max(0.24, min_similarity - 0.30)
    for i in range(len(events)):
        ci = _event_chunk_set(events[i])
        for j in range(i + 1, len(events)):
            pair = (i, j)
            if pair in existing:
                continue
            cj = _event_chunk_set(events[j])
            if ci and cj and not ci.isdisjoint(cj):
                continue
            lexical_sim = _cosine_counter(bows[i], bows[j])
            if lexical_sim < shortlist_floor:
                continue
            features = _candidate_feature_map(events[i], events[j], entity_name_by_id=entity_name_by_id)
            strong_anchor = (
                features["shared_time"]
                or features["shared_time_token_count"] > 0
                or features["shared_participant_count"] > 0
                or features["shared_participant_name_count"] > 0
                or features["same_trigger"]
                or features["nearby_sentence_window"]
            )
            if not strong_anchor:
                continue
            priority = (
                lexical_sim
                + (0.24 if features["shared_time"] else 0.0)
                + (0.16 if features["shared_time_token_count"] > 0 else 0.0)
                + (0.14 if features["shared_participant_count"] > 0 else 0.0)
                + (0.10 if features["shared_participant_name_count"] > 0 else 0.0)
                + (0.08 if features["same_trigger"] else 0.0)
                + (0.08 if features["nearby_sentence_window"] else 0.0)
            )
            shortlist.append(
                {
                    "i": i,
                    "j": j,
                    "lexical_score": lexical_sim,
                    "priority": priority,
                    "features": features,
                }
            )

    if not shortlist:
        return []

    shortlist = sorted(shortlist, key=lambda row: row["priority"], reverse=True)[:max_candidates]
    embeds = _embed_texts(client, semantic_model, bundles)
    semantic_threshold = max(0.56, min_similarity - 0.04)
    out: list[dict[str, Any]] = []
    for row in shortlist:
        i = int(row["i"])
        j = int(row["j"])
        sim = _cosine(embeds[i], embeds[j])
        if sim < semantic_threshold:
            continue
        features = row["features"]
        bucket = "relation_semantic_rescue"
        if features["shared_sentence"] or (
            features["same_trigger"]
            and (features["shared_participant_count"] > 0 or features["shared_participant_name_count"] > 0)
        ):
            bucket = "duplicate_semantic_rescue"
        out.append(
            {
                "i": i,
                "j": j,
                "score": sim,
                "lexical_score": float(row["lexical_score"]),
                "priority": sim + float(row["priority"]),
                "bucket": bucket,
                "rescue": True,
                "features": {
                    **features,
                    "semantic_rescue_model": semantic_model,
                    "semantic_score": sim,
                    "lexical_score": float(row["lexical_score"]),
                },
            }
        )
    return out


def _call_link_llm(
    *,
    client: OpenAI,
    model: str,
    temperature: float,
    doc_id: str,
    event_a: dict[str, Any],
    event_b: dict[str, Any],
    events: list[dict[str, Any]],
    event_a_index: int,
    event_b_index: int,
    entity_name_by_id: dict[str, str] | None,
    chunk_lookup: dict[str, dict[str, Any]] | None,
    candidate_bucket: str,
    candidate_features: dict[str, Any],
    max_completion_tokens: int,
) -> dict[str, Any] | None:
    user = {
        "doc_id": doc_id,
        "candidate_bucket": candidate_bucket,
        "candidate_features": candidate_features,
        "event_a": event_a,
        "event_b": event_b,
        "event_a_identity": _event_identity_snapshot(event_a, index=event_a_index, entity_name_by_id=entity_name_by_id),
        "event_b_identity": _event_identity_snapshot(event_b, index=event_b_index, entity_name_by_id=entity_name_by_id),
        "event_a_text_bundle": _event_text_bundle(event_a, entity_name_by_id=entity_name_by_id, chunk_lookup=chunk_lookup),
        "event_b_text_bundle": _event_text_bundle(event_b, entity_name_by_id=entity_name_by_id, chunk_lookup=chunk_lookup),
        "event_a_source_contexts": _event_source_contexts(event_a, chunk_lookup),
        "event_b_source_contexts": _event_source_contexts(event_b, chunk_lookup),
        "event_a_neighborhood": _event_neighborhood(
            events,
            event_a_index,
            entity_name_by_id=entity_name_by_id,
            chunk_lookup=chunk_lookup,
        ),
        "event_b_neighborhood": _event_neighborhood(
            events,
            event_b_index,
            entity_name_by_id=entity_name_by_id,
            chunk_lookup=chunk_lookup,
        ),
    }
    request: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": HYBRID_LINK_SYSTEM},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        "temperature": temperature,
        "max_completion_tokens": int(max_completion_tokens),
    }
    raw = ""
    try:
        request["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(**request)
        raw = _strip_code_fences((response.choices[0].message.content or "").strip())
    except Exception:
        request.pop("response_format", None)
        try:
            response = client.chat.completions.create(**request)
            raw = _strip_code_fences((response.choices[0].message.content or "").strip())
        except Exception:
            return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def _collapse_same_event_clusters(
    merged_graph: dict[str, Any],
    same_pairs: list[tuple[int, int]],
) -> tuple[dict[str, Any], dict[int, str], int]:
    events = _ensure_list(merged_graph.get("events"))
    uf = _UnionFind(len(events))
    for ia, ib in same_pairs:
        if not (0 <= ia < len(events) and 0 <= ib < len(events)):
            continue
        uf.union(ia, ib)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(events)):
        groups[uf.find(i)].append(i)
    if all(len(v) == 1 for v in groups.values()):
        return merged_graph, {i: _safe_text(events[i].get("event_id")) for i in range(len(events)) if isinstance(events[i], dict)}, 0

    sorted_groups = sorted(groups.values(), key=lambda g: min(g))
    idx_to_newid: dict[int, str] = {}
    new_events: list[dict[str, Any]] = []
    for gid, members in enumerate(sorted_groups, start=1):
        representative = deepcopy(events[min(members)])
        new_id = f"EV{gid}"
        representative["event_id"] = new_id
        merged_source_refs = []
        for m in members:
            ev = events[m]
            refs = _ensure_list(ev.get("source_refs"))
            for ref in refs:
                if isinstance(ref, dict):
                    merged_source_refs.append(ref)
            idx_to_newid[m] = new_id
        if merged_source_refs:
            representative["source_refs"] = merged_source_refs
        new_events.append(representative)

    old_id_to_new: dict[str, str] = {}
    for idx, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        old_id = _safe_text(event.get("event_id"))
        new_id = idx_to_newid.get(idx)
        if old_id and new_id and old_id not in old_id_to_new:
            old_id_to_new[old_id] = new_id

    def _remap_edge_list(edge_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        seen = set()
        for edge in edge_list:
            if not isinstance(edge, dict):
                continue
            src = old_id_to_new.get(_safe_text(edge.get("from_event")), _safe_text(edge.get("from_event")))
            dst = old_id_to_new.get(_safe_text(edge.get("to_event")), _safe_text(edge.get("to_event")))
            if not src or not dst or src == dst:
                continue
            relation = _safe_text(edge.get("relation"))
            key = (src, dst, relation)
            if key in seen:
                continue
            seen.add(key)
            remapped = dict(edge)
            remapped["from_event"] = src
            remapped["to_event"] = dst
            out.append(remapped)
        return out

    out_graph = {
        **merged_graph,
        "events": new_events,
        "temporal_edges": _remap_edge_list(_ensure_list(merged_graph.get("temporal_edges"))),
        "causal_edges": _remap_edge_list(_ensure_list(merged_graph.get("causal_edges"))),
    }
    collapsed = len(events) - len(new_events)
    return out_graph, idx_to_newid, max(0, collapsed)


def _add_llm_edges(
    merged_graph: dict[str, Any],
    decisions: list[dict[str, Any]],
) -> tuple[int, int]:
    temporal = _ensure_list(merged_graph.get("temporal_edges"))
    causal = _ensure_list(merged_graph.get("causal_edges"))
    t_seen = {(str(e.get("from_event")), str(e.get("to_event")), str(e.get("relation"))) for e in temporal if isinstance(e, dict)}
    c_seen = {(str(e.get("from_event")), str(e.get("to_event")), str(e.get("relation"))) for e in causal if isinstance(e, dict)}
    t_added = 0
    c_added = 0
    for d in decisions:
        a = _safe_text(d.get("collapsed_event_a") or d.get("event_a"))
        b = _safe_text(d.get("collapsed_event_b") or d.get("event_b"))
        if not a or not b or a == b:
            continue
        temporal_d = d.get("temporal") or {}
        if isinstance(temporal_d, dict) and bool(temporal_d.get("add")):
            direction = _safe_text(temporal_d.get("direction"))
            relation = _safe_text(temporal_d.get("relation")) or "RELATED_TO"
            src, dst = (a, b) if direction == "A_to_B" else ((b, a) if direction == "B_to_A" else ("", ""))
            if src and dst:
                key = (src, dst, relation)
                if key not in t_seen:
                    temporal.append(
                        {
                            "from_event": src,
                            "to_event": dst,
                            "relation": relation,
                            "evidence": {"sentence_ids": [], "snippets": []},
                            "confidence": float(d.get("confidence", 0.0) or 0.0),
                            "source": "hybrid_llm",
                        }
                    )
                    t_seen.add(key)
                    t_added += 1
        causal_d = d.get("causal") or {}
        if isinstance(causal_d, dict) and bool(causal_d.get("add")):
            direction = _safe_text(causal_d.get("direction"))
            relation = _safe_text(causal_d.get("relation")) or "CAUSES"
            src, dst = (a, b) if direction == "A_to_B" else ((b, a) if direction == "B_to_A" else ("", ""))
            if src and dst:
                key = (src, dst, relation)
                if key not in c_seen:
                    causal.append(
                        {
                            "from_event": src,
                            "to_event": dst,
                            "relation": relation,
                            "evidence": {"sentence_ids": [], "snippets": []},
                            "confidence": float(d.get("confidence", 0.0) or 0.0),
                            "source": "hybrid_llm",
                        }
                    )
                    c_seen.add(key)
                    c_added += 1
    merged_graph["temporal_edges"] = temporal
    merged_graph["causal_edges"] = causal
    return t_added, c_added


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid merge of chunk-level event mini-graphs.")
    parser.add_argument("--input", required=True, help="Input JSONL with doc.chunks[].graph or mini_graph.")
    parser.add_argument("--output", required=True, help="Output JSONL with merged/hybrid doc-level graph.")
    parser.add_argument("--model", default=None, help="Override LLM model for pair adjudication.")
    parser.add_argument("--embedding-model", default="text-embedding-3-small", help="Embedding model name.")
    parser.add_argument("--max-docs", type=int, default=None, help="Max documents to process.")
    parser.add_argument("--top-k", type=int, default=6, help="Top-k nearest neighbors per event.")
    parser.add_argument("--min-similarity", type=float, default=0.60, help="Primary retrieval similarity threshold.")
    parser.add_argument("--max-pairs", type=int, default=80, help="Max LLM pair adjudications per doc.")
    parser.add_argument("--max-relation-pairs", type=int, default=None, help="Cap relation-discovery adjudications per doc.")
    parser.add_argument("--max-duplicate-pairs", type=int, default=None, help="Cap duplicate adjudications per doc.")
    parser.add_argument("--semantic-rescue-model", default="text-embedding-3-small", help="Optional embedding model for rescue retrieval when primary retrieval is lexical.")
    parser.add_argument("--semantic-rescue-candidates", type=int, default=24, help="Max near-miss pairs to send through semantic rescue retrieval.")
    parser.add_argument("--max-completion-tokens", type=int, default=500, help="Token budget per pair adjudication.")
    parser.add_argument("--disable-same-event-collapse", action="store_true", help="Do not collapse events marked same_event=true.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    client, model, temperature = _make_client(args.model)
    merger = MiniGraphMergeAgent()
    generated_at = datetime.now(timezone.utc).isoformat()

    written = 0
    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            if args.max_docs is not None and written >= args.max_docs:
                break
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue

            merged = merger.merge_document(doc)
            merged_graph = merged.get("merged_graph") or {}
            events = _ensure_list(merged_graph.get("events"))
            entity_name_by_id = {
                _safe_text(ent.get("entity_id")): _safe_text(ent.get("name"))
                for ent in _ensure_list(merged_graph.get("entities"))
                if isinstance(ent, dict) and _safe_text(ent.get("entity_id"))
            }
            chunk_lookup = _build_chunk_lookup(doc)
            if len(events) < 2:
                merged["hybrid_merge"] = {
                    "enabled": True,
                    "llm_model": model,
                    "embedding_model": args.embedding_model,
                    "processed_at_utc": generated_at,
                    "num_candidates": 0,
                    "num_pairs_sent_to_llm": 0,
                    "num_same_event_links": 0,
                    "num_temporal_added": 0,
                    "num_causal_added": 0,
                    "num_events_collapsed": 0,
                    "notes": "Skipped hybrid linking: fewer than 2 merged events.",
                }
                fout.write(json.dumps(merged, default=str) + "\n")
                fout.flush()
                written += 1
                print(
                    f"[progress] wrote doc {written}: {merged.get('doc_id') or doc.get('doc_id') or 'unknown'} "
                    f"(events={len(_ensure_list(merged_graph.get('events')))}, "
                    f"temporal={len(_ensure_list(merged_graph.get('temporal_edges')))}, "
                    f"causal={len(_ensure_list(merged_graph.get('causal_edges')))} )",
                    flush=True,
                )
                continue

            if args.embedding_model == "local_lexical_cosine":
                candidate_rows = _pair_candidates_local_lexical(
                    events,
                    top_k=args.top_k,
                    min_similarity=args.min_similarity,
                    entity_name_by_id=entity_name_by_id,
                    chunk_lookup=chunk_lookup,
                )
                candidate_rows.extend(
                    _semantic_rescue_candidates(
                        client=client,
                        semantic_model=_safe_text(args.semantic_rescue_model),
                        events=events,
                        lexical_rows=candidate_rows,
                        min_similarity=float(args.min_similarity),
                        max_candidates=int(args.semantic_rescue_candidates),
                        entity_name_by_id=entity_name_by_id,
                        chunk_lookup=chunk_lookup,
                    )
                )
            else:
                texts = [_event_text_bundle(ev, entity_name_by_id=entity_name_by_id, chunk_lookup=chunk_lookup) for ev in events]
                embeddings = _embed_texts(client, args.embedding_model, texts)
                candidate_rows = [
                    {"i": i, "j": j, "score": sim, "priority": sim, "bucket": "mixed", "rescue": False, "features": {}}
                    for i, j, sim in _pair_candidates(events, embeddings, top_k=args.top_k, min_similarity=args.min_similarity)
                ]
            duplicate_candidates = [row for row in candidate_rows if str(row.get("bucket", "")).startswith("duplicate")]
            relation_candidates = [row for row in candidate_rows if not str(row.get("bucket", "")).startswith("duplicate")]
            relation_quota = args.max_relation_pairs
            duplicate_quota = args.max_duplicate_pairs
            if relation_quota is None or duplicate_quota is None:
                derived_relation_quota = max(1, math.ceil(args.max_pairs * 0.67))
                derived_duplicate_quota = max(1, args.max_pairs - derived_relation_quota)
                relation_quota = derived_relation_quota if relation_quota is None else relation_quota
                duplicate_quota = derived_duplicate_quota if duplicate_quota is None else duplicate_quota
            relation_quota = min(len(relation_candidates), max(0, int(relation_quota)))
            duplicate_quota = min(len(duplicate_candidates), max(0, int(duplicate_quota)))
            if relation_quota + duplicate_quota > args.max_pairs:
                duplicate_quota = max(0, args.max_pairs - relation_quota)
            candidates = sorted(
                relation_candidates,
                key=lambda x: (float(x.get("priority", 0.0)), float(x.get("score", 0.0))),
                reverse=True,
            )[:relation_quota]
            candidates.extend(
                sorted(
                    duplicate_candidates,
                    key=lambda x: (float(x.get("priority", 0.0)), float(x.get("score", 0.0))),
                    reverse=True,
                )[:duplicate_quota]
            )
            selected_keys = {(int(row["i"]), int(row["j"])) for row in candidates}
            if len(candidates) < args.max_pairs:
                leftovers = [row for row in relation_candidates if (int(row["i"]), int(row["j"])) not in selected_keys]
                need = args.max_pairs - len(candidates)
                candidates.extend(
                    sorted(
                        leftovers,
                        key=lambda x: (float(x.get("priority", 0.0)), float(x.get("score", 0.0))),
                        reverse=True,
                    )[:need]
                )
                selected_keys = {(int(row["i"]), int(row["j"])) for row in candidates}
            if len(candidates) < args.max_pairs:
                leftovers = [row for row in duplicate_candidates if (int(row["i"]), int(row["j"])) not in selected_keys]
                need = args.max_pairs - len(candidates)
                candidates.extend(
                    sorted(
                        leftovers,
                        key=lambda x: (float(x.get("priority", 0.0)), float(x.get("score", 0.0))),
                        reverse=True,
                    )[:need]
                )

            decisions: list[dict[str, Any]] = []
            same_pairs: list[tuple[int, int]] = []
            for cand in candidates:
                i = int(cand["i"])
                j = int(cand["j"])
                sim = float(cand.get("score", 0.0))
                ev_a = events[i]
                ev_b = events[j]
                decision = _call_link_llm(
                    client=client,
                    model=model,
                    temperature=temperature,
                    doc_id=_safe_text(merged.get("doc_id") or doc.get("doc_id")),
                    event_a=ev_a,
                    event_b=ev_b,
                    events=events,
                    event_a_index=i,
                    event_b_index=j,
                    entity_name_by_id=entity_name_by_id,
                    chunk_lookup=chunk_lookup,
                    candidate_bucket=_safe_text(cand.get("bucket")),
                    candidate_features=cand.get("features") if isinstance(cand.get("features"), dict) else {},
                    max_completion_tokens=args.max_completion_tokens,
                )
                if not decision:
                    continue
                row = {
                    "event_a": _safe_text(ev_a.get("event_id")),
                    "event_b": _safe_text(ev_b.get("event_id")),
                    "original_event_a": _safe_text(ev_a.get("event_id")),
                    "original_event_b": _safe_text(ev_b.get("event_id")),
                    "event_a_index": i,
                    "event_b_index": j,
                    "event_a_key": f"{_safe_text(ev_a.get('event_id'))}@@idx:{i}",
                    "event_b_key": f"{_safe_text(ev_b.get('event_id'))}@@idx:{j}",
                    "event_a_identity": _event_identity_snapshot(ev_a, index=i, entity_name_by_id=entity_name_by_id),
                    "event_b_identity": _event_identity_snapshot(ev_b, index=j, entity_name_by_id=entity_name_by_id),
                    "similarity": float(sim),
                    "lexical_score": float(cand.get("lexical_score", sim) or sim),
                    "candidate_bucket": _safe_text(cand.get("bucket")),
                    "candidate_priority": float(cand.get("priority", sim) or sim),
                    "candidate_features": cand.get("features") if isinstance(cand.get("features"), dict) else {},
                    "retrieval_rescue": bool(cand.get("rescue")),
                    **decision,
                }
                decisions.append(row)
                if bool(decision.get("same_event")):
                    same_pairs.append((i, j))

            events_collapsed = 0
            old_to_new: dict[int, str] = {}
            if same_pairs and not args.disable_same_event_collapse:
                merged_graph, old_to_new, events_collapsed = _collapse_same_event_clusters(merged_graph, same_pairs)
            if old_to_new:
                for d in decisions:
                    d["collapsed_event_a"] = old_to_new.get(int(d.get("event_a_index", -1)), _safe_text(d.get("original_event_a")))
                    d["collapsed_event_b"] = old_to_new.get(int(d.get("event_b_index", -1)), _safe_text(d.get("original_event_b")))

            t_added, c_added = _add_llm_edges(merged_graph, decisions)

            merged["merged_graph"] = merged_graph
            merged["merge_stats"] = {
                **(merged.get("merge_stats") or {}),
                "num_events": len(_ensure_list(merged_graph.get("events"))),
                "num_temporal_edges": len(_ensure_list(merged_graph.get("temporal_edges"))),
                "num_causal_edges": len(_ensure_list(merged_graph.get("causal_edges"))),
            }
            merged["hybrid_merge"] = {
                "enabled": True,
                "llm_model": model,
                "embedding_model": args.embedding_model,
                "processed_at_utc": generated_at,
                "num_candidates": len(candidates),
                "num_duplicate_candidates": len([row for row in candidates if str(row.get("bucket", "")).startswith("duplicate")]),
                "num_relation_candidates": len([row for row in candidates if not str(row.get("bucket", "")).startswith("duplicate")]),
                "num_duplicate_candidates_total": len(duplicate_candidates),
                "num_relation_candidates_total": len(relation_candidates),
                "num_rescue_candidates": len([row for row in candidates if bool(row.get("rescue"))]),
                "num_pairs_sent_to_llm": len(decisions),
                "num_same_event_links": len(same_pairs),
                "num_temporal_added": t_added,
                "num_causal_added": c_added,
                "num_events_collapsed": events_collapsed,
                "top_k": int(args.top_k),
                "min_similarity": float(args.min_similarity),
                "max_pairs": int(args.max_pairs),
                "max_relation_pairs": int(relation_quota),
                "max_duplicate_pairs": int(duplicate_quota),
                "semantic_rescue_model": _safe_text(args.semantic_rescue_model),
                "semantic_rescue_candidates": int(args.semantic_rescue_candidates),
                "adjudications": decisions,
            }

            fout.write(json.dumps(merged, default=str) + "\n")
            fout.flush()
            written += 1
            print(
                f"[progress] wrote doc {written}: {merged.get('doc_id') or doc.get('doc_id') or 'unknown'} "
                f"(events={len(_ensure_list(merged_graph.get('events')))}, "
                f"temporal={len(_ensure_list(merged_graph.get('temporal_edges')))}, "
                f"causal={len(_ensure_list(merged_graph.get('causal_edges')))} )",
                flush=True,
            )

    print(f"Done. Wrote {written} docs to {args.output}")


if __name__ == "__main__":
    main()
