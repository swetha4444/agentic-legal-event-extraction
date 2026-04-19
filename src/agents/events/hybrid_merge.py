"""
Hybrid doc-level merge refinement:
- start from deterministic merged graph
- propose cross-event candidates via embeddings
- ask LLM for identity + temporal/causal links
- apply validated decisions with deterministic fallbacks
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Any


def _ensure_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _norm_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _strip_code_fences(text: str) -> str:
    raw = (text or "").strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if match:
        return match.group(1).strip()
    return raw


def _parse_json(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    text = _strip_code_fences(raw)
    if not text:
        return None, "empty response"
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, str(exc)
    if not isinstance(obj, dict):
        return None, "json is not object"
    return obj, None


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (na * nb)


class _UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {x: x for x in items}

    def find(self, x: str) -> str:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != x:
            nxt = self.parent[x]
            self.parent[x] = root
            x = nxt
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


HYBRID_LINK_SYSTEM = """You are linking two legal events from different chunks.

Decide:
1) same_event: true only if both event objects describe the same real-world event.
2) temporal_relation: one of BEFORE, AFTER, OVERLAP, NONE for A -> B.
3) causal_relation: one of CAUSES, PRECONDITION, ENABLES, PREVENTS, NONE for A -> B.

Rules:
- Be conservative. Use NONE when uncertain.
- Do not infer from role names alone.
- Ground decisions in trigger, participants, time, and evidence snippets.
- Output JSON only with keys:
  same_event, temporal_relation, causal_relation, confidence, rationale
"""

HYBRID_LINK_USER = """DOC: {doc_id}

EVENT A:
{event_a}

EVENT B:
{event_b}
"""


def _event_view(event: dict[str, Any], entity_name_by_id: dict[str, str]) -> dict[str, Any]:
    participants = []
    for p in _ensure_list(event.get("participants")):
        if not isinstance(p, dict):
            continue
        eid = str(p.get("entity_id") or "")
        participants.append(
            {
                "entity_id": eid,
                "entity_name": entity_name_by_id.get(eid, ""),
                "role": str(p.get("role") or ""),
                "mention": (
                    (p.get("mention") or {}).get("span_text")
                    if isinstance(p.get("mention"), dict)
                    else str(p.get("mention") or "")
                ),
            }
        )
    return {
        "event_id": event.get("event_id"),
        "event_type": event.get("event_type"),
        "main_verb": event.get("main_verb"),
        "trigger": event.get("trigger"),
        "time": event.get("time"),
        "participants": participants,
        "evidence": event.get("evidence"),
        "source_refs": event.get("source_refs"),
    }


def _event_embed_text(event: dict[str, Any], entity_name_by_id: dict[str, str]) -> str:
    parts: list[str] = []
    parts.append(str(event.get("event_type") or ""))
    parts.append(str(event.get("main_verb") or (event.get("trigger") or {}).get("span_text") or ""))
    time_obj = event.get("time") or {}
    parts.append(str(time_obj.get("normalized") or time_obj.get("raw_span") or ""))
    for p in _ensure_list(event.get("participants")):
        if not isinstance(p, dict):
            continue
        eid = str(p.get("entity_id") or "")
        parts.append(f"{p.get('role')}: {entity_name_by_id.get(eid, eid)}")
    ev = event.get("evidence") or {}
    snippets = _ensure_list(ev.get("snippets"))
    if snippets:
        s0 = snippets[0]
        if isinstance(s0, dict):
            parts.append(str(s0.get("text") or ""))
    return " | ".join([_norm_text(x) for x in parts if str(x).strip()])


@dataclass
class HybridMergeRefiner:
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o"
    similarity_threshold: float = 0.78
    top_k: int = 4
    max_candidates: int = 120
    temperature: float = 0.0
    max_completion_tokens: int = 700

    def refine(
        self,
        *,
        doc_id: str,
        merged_graph: dict[str, Any],
        client: Any,
        apply_llm: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        entities = _ensure_list(merged_graph.get("entities"))
        events = _ensure_list(merged_graph.get("events"))
        event_ids = [str(ev.get("event_id") or "") for ev in events if isinstance(ev, dict)]
        event_by_id = {str(ev.get("event_id") or ""): ev for ev in events if isinstance(ev, dict)}
        entity_name_by_id = {str(e.get("entity_id") or ""): str(e.get("name") or "") for e in entities if isinstance(e, dict)}

        candidates = self._candidate_pairs(
            events=events,
            entity_name_by_id=entity_name_by_id,
            client=client,
            use_api_embeddings=apply_llm,
        )
        if self.max_candidates and len(candidates) > self.max_candidates:
            candidates = candidates[: self.max_candidates]

        llm_calls = 0
        llm_errors = 0
        llm_error_samples: list[str] = []
        identity_edges: list[tuple[str, str]] = []
        temporal_edges: list[dict[str, Any]] = []
        causal_edges: list[dict[str, Any]] = []

        for eid_a, eid_b, score in candidates:
            if not apply_llm:
                continue
            a = event_by_id.get(eid_a)
            b = event_by_id.get(eid_b)
            if not isinstance(a, dict) or not isinstance(b, dict):
                continue
            decision, err = self._link_decision(
                doc_id=doc_id,
                event_a=_event_view(a, entity_name_by_id),
                event_b=_event_view(b, entity_name_by_id),
                client=client,
            )
            llm_calls += 1
            if err:
                llm_errors += 1
                if len(llm_error_samples) < 5:
                    llm_error_samples.append(err)
                continue
            if bool(decision.get("same_event")):
                identity_edges.append((eid_a, eid_b))

            temp_rel = str(decision.get("temporal_relation") or "NONE").upper()
            if temp_rel != "NONE":
                temporal_edges.append(
                    {
                        "from_event": eid_a,
                        "to_event": eid_b,
                        "relation": temp_rel,
                        "confidence": float(decision.get("confidence") or 0.0),
                        "evidence": {
                            "sentence_ids": [],
                            "snippets": [{"text": str(decision.get("rationale") or f"Hybrid link score={score:.3f}")}],
                        },
                    }
                )
            caus_rel = str(decision.get("causal_relation") or "NONE").upper()
            if caus_rel != "NONE":
                causal_edges.append(
                    {
                        "from_event": eid_a,
                        "to_event": eid_b,
                        "relation": caus_rel,
                        "confidence": float(decision.get("confidence") or 0.0),
                        "evidence": {
                            "sentence_ids": [],
                            "snippets": [{"text": str(decision.get("rationale") or f"Hybrid link score={score:.3f}")}],
                        },
                    }
                )

        deduped_graph, dedupe_stats = self._apply_identity_unions(
            graph=merged_graph,
            event_ids=event_ids,
            identity_edges=identity_edges,
        )
        deduped_graph["temporal_edges"] = _ensure_list(deduped_graph.get("temporal_edges")) + temporal_edges
        deduped_graph["causal_edges"] = _ensure_list(deduped_graph.get("causal_edges")) + causal_edges
        deduped_graph["hybrid_provenance"] = {
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model,
            "similarity_threshold": self.similarity_threshold,
            "top_k": self.top_k,
            "candidates_considered": len(candidates),
            "identity_links_added": len(identity_edges),
            "temporal_links_added": len(temporal_edges),
            "causal_links_added": len(causal_edges),
            "llm_calls": llm_calls,
            "llm_errors": llm_errors,
            "llm_error_samples": llm_error_samples,
        }
        return deduped_graph, dedupe_stats

    def _candidate_pairs(
        self,
        *,
        events: list[dict[str, Any]],
        entity_name_by_id: dict[str, str],
        client: Any,
        use_api_embeddings: bool,
    ) -> list[tuple[str, str, float]]:
        vectors: dict[str, list[float]] = {}
        texts: dict[str, str] = {}
        for ev in events:
            if not isinstance(ev, dict):
                continue
            eid = str(ev.get("event_id") or "")
            if not eid:
                continue
            text = _event_embed_text(ev, entity_name_by_id)
            texts[eid] = text
            emb = self._embed_one(text, client=client, use_api_embeddings=use_api_embeddings)
            vectors[eid] = emb
        ids = list(vectors.keys())
        scored: list[tuple[str, str, float]] = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                score = _cosine(vectors[a], vectors[b])
                if score >= self.similarity_threshold:
                    scored.append((a, b, score))
        scored.sort(key=lambda x: x[2], reverse=True)

        if self.top_k <= 0:
            return scored
        per_node: dict[str, int] = {}
        kept: list[tuple[str, str, float]] = []
        for a, b, s in scored:
            if per_node.get(a, 0) >= self.top_k or per_node.get(b, 0) >= self.top_k:
                continue
            kept.append((a, b, s))
            per_node[a] = per_node.get(a, 0) + 1
            per_node[b] = per_node.get(b, 0) + 1
        return kept

    def _embed_one(self, text: str, *, client: Any, use_api_embeddings: bool) -> list[float]:
        if use_api_embeddings:
            try:
                r = client.embeddings.create(model=self.embedding_model, input=text)
                vec = (r.data[0].embedding or []) if getattr(r, "data", None) else []
                return [float(x) for x in vec]
            except Exception:
                pass
        # deterministic lexical fallback: cheap hashed vector
        dim = 256
        vec = [0.0] * dim
        for tok in re.findall(r"[a-z0-9_]+", _norm_text(text)):
            vec[hash(tok) % dim] += 1.0
        return vec

    def _link_decision(
        self,
        *,
        doc_id: str,
        event_a: dict[str, Any],
        event_b: dict[str, Any],
        client: Any,
    ) -> tuple[dict[str, Any], str | None]:
        user = HYBRID_LINK_USER.format(
            doc_id=doc_id,
            event_a=json.dumps(event_a, ensure_ascii=False, indent=2),
            event_b=json.dumps(event_b, ensure_ascii=False, indent=2),
        )
        request: dict[str, Any] = {
            "model": self.llm_model,
            "messages": [
                {"role": "system", "content": HYBRID_LINK_SYSTEM},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_completion_tokens": self.max_completion_tokens,
            "response_format": {"type": "json_object"},
        }
        raw = ""
        try:
            r = client.chat.completions.create(**request)
            raw = (r.choices[0].message.content or "").strip()
        except Exception as exc:
            return {}, str(exc)

        parsed, err = _parse_json(raw)
        if err or parsed is None:
            return {}, err or "parse error"
        if "same_event" not in parsed:
            return {}, "missing same_event"
        parsed["temporal_relation"] = str(parsed.get("temporal_relation") or "NONE").upper()
        parsed["causal_relation"] = str(parsed.get("causal_relation") or "NONE").upper()
        if parsed["temporal_relation"] not in {"BEFORE", "AFTER", "OVERLAP", "NONE"}:
            parsed["temporal_relation"] = "NONE"
        if parsed["causal_relation"] not in {"CAUSES", "PRECONDITION", "ENABLES", "PREVENTS", "NONE"}:
            parsed["causal_relation"] = "NONE"
        return parsed, None

    def _apply_identity_unions(
        self,
        *,
        graph: dict[str, Any],
        event_ids: list[str],
        identity_edges: list[tuple[str, str]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        uf = _UnionFind(event_ids)
        for a, b in identity_edges:
            if a in uf.parent and b in uf.parent:
                uf.union(a, b)
        rep_to_ids: dict[str, list[str]] = {}
        for eid in event_ids:
            rep = uf.find(eid)
            rep_to_ids.setdefault(rep, []).append(eid)
        if all(len(v) == 1 for v in rep_to_ids.values()):
            return graph, {"identity_merges": 0}

        event_by_id = {str(e.get("event_id") or ""): e for e in _ensure_list(graph.get("events")) if isinstance(e, dict)}
        representative_map: dict[str, str] = {}
        merged_events: list[dict[str, Any]] = []
        for rep, ids in rep_to_ids.items():
            keep = sorted(ids)[0]
            representative_map.update({x: keep for x in ids})
            base = dict(event_by_id[keep])
            src_refs = []
            for eid in ids:
                ev = event_by_id.get(eid) or {}
                src_refs.extend(_ensure_list(ev.get("source_refs")))
            base["source_refs"] = src_refs
            base["event_id"] = keep
            merged_events.append(base)

        def remap_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
            out: list[dict[str, Any]] = []
            seen: set[tuple[str, str, str]] = set()
            for e in edges:
                if not isinstance(e, dict):
                    continue
                a = representative_map.get(str(e.get("from_event") or ""), str(e.get("from_event") or ""))
                b = representative_map.get(str(e.get("to_event") or ""), str(e.get("to_event") or ""))
                rel = str(e.get("relation") or "")
                if not a or not b:
                    continue
                key = (a, b, rel)
                if key in seen:
                    continue
                seen.add(key)
                out.append({**e, "from_event": a, "to_event": b})
            return out

        out_graph = dict(graph)
        out_graph["events"] = merged_events
        out_graph["temporal_edges"] = remap_edges(_ensure_list(graph.get("temporal_edges")))
        out_graph["causal_edges"] = remap_edges(_ensure_list(graph.get("causal_edges")))
        merges = sum(max(0, len(v) - 1) for v in rep_to_ids.values())
        return out_graph, {"identity_merges": merges}
