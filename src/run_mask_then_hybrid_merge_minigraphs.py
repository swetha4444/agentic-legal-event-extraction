#!/usr/bin/env python3
"""
Combined pipeline (local-Qwen only, no API calls):
1) Mask/normalize each chunk mini-graph with local Qwen (role-style labels)
2) Deterministic document merge of chunk mini-graphs
3) Retrieval + local Qwen adjudication for same_event / temporal / causal links
4) Optional event collapse for same_event clusters
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __name__ == "__main__":
    _src = Path(__file__).resolve().parent
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from agents.events import MiniGraphMergeAgent
from agents.events.minigraph_mask import (
    MINIGRAPH_MASK_SYSTEM,
    MINIGRAPH_MASK_USER,
    masked_graph_is_compatible,
)


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
- Keep relations concise (e.g., BEFORE, AFTER, OVERLAP, CAUSES, ENABLES, PREVENTS, RELATED_TO).
"""


def _ensure_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _pick_graph(chunk: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    if chunk.get("graph"):
        return "graph", chunk["graph"]
    if chunk.get("mini_graph"):
        return "mini_graph", chunk["mini_graph"]
    return None, {}


def _event_text_bundle(event: dict[str, Any]) -> str:
    participants = []
    for p in _ensure_list(event.get("participants")):
        if not isinstance(p, dict):
            continue
        role = _safe_text(p.get("role")).upper() or "CONTEXT"
        ent = _safe_text(p.get("entity_id"))
        mention = p.get("mention") or {}
        mention_text = _safe_text(mention.get("span_text") or mention.get("text")) if isinstance(mention, dict) else ""
        participants.append(f"{role}:{ent}:{mention_text}")
    evidence = event.get("evidence") or {}
    snippets = []
    if isinstance(evidence, dict):
        for s in _ensure_list(evidence.get("snippets"))[:3]:
            snippets.append(_safe_text(s.get("text")) if isinstance(s, dict) else _safe_text(s))
    src_chunks = []
    for ref in _ensure_list(event.get("source_refs")):
        if isinstance(ref, dict):
            cid = _safe_text(ref.get("chunk_id"))
            if cid:
                src_chunks.append(cid)
    return " | ".join(
        [
            f"event_id={_safe_text(event.get('event_id'))}",
            f"type={_safe_text(event.get('event_type')).upper()}",
            f"verb={_safe_text(event.get('main_verb'))}",
            f"time={json.dumps(event.get('time') or {}, ensure_ascii=False)}",
            f"participants={';'.join(participants)}",
            f"snippets={';'.join(snippets)}",
            f"source_chunks={','.join(sorted(set(src_chunks)))}",
        ]
    )


def _event_chunk_set(event: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for ref in _ensure_list(event.get("source_refs")):
        if isinstance(ref, dict):
            cid = _safe_text(ref.get("chunk_id"))
            if cid:
                out.add(cid)
    return out


def _tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"[^A-Za-z0-9_]+", (text or "").lower()) if t]


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


def _pair_candidates(events: list[dict[str, Any]], top_k: int, min_similarity: float) -> list[tuple[int, int, float]]:
    if not events:
        return []
    bundles = [_event_text_bundle(ev) for ev in events]
    bows = [_bow(b) for b in bundles]

    per_i: list[list[tuple[float, int]]] = [[] for _ in events]
    for i in range(len(events)):
        ci = _event_chunk_set(events[i])
        for j in range(i + 1, len(events)):
            cj = _event_chunk_set(events[j])
            if ci and cj and not ci.isdisjoint(cj):
                continue
            sim = _cosine_counter(bows[i], bows[j])
            if sim < min_similarity:
                continue
            per_i[i].append((sim, j))
            per_i[j].append((sim, i))

    pairs: set[tuple[int, int]] = set()
    for i, nbrs in enumerate(per_i):
        for _, j in sorted(nbrs, key=lambda x: x[0], reverse=True)[: max(1, top_k)]:
            a, b = sorted((i, j))
            pairs.add((a, b))

    out: list[tuple[int, int, float]] = []
    for i, j in sorted(pairs):
        out.append((i, j, _cosine_counter(bows[i], bows[j])))
    return out


def _strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()
    return text


def _extract_json_object(raw: str) -> dict[str, Any] | None:
    text = _strip_code_fences(raw)
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


class _LocalQwenChat:
    def __init__(self, model_path: str, *, device: str = "auto", max_new_tokens: int = 1024) -> None:
        try:
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Local Qwen mode requires 'torch' and 'transformers'. "
                "Install them in your runtime environment."
            ) from exc

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        model_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = "auto"
        elif device == "cpu":
            model_kwargs["torch_dtype"] = torch.float32
            model_kwargs["device_map"] = "cpu"
        else:
            if torch.cuda.is_available():
                model_kwargs["torch_dtype"] = torch.float16
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["torch_dtype"] = torch.float32
                model_kwargs["device_map"] = "cpu"

        self._model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        self._max_new_tokens = int(max_new_tokens)

    def complete_json(self, *, system: str, user: str, temperature: float, max_new_tokens: int | None = None) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer([prompt], return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens or self._max_new_tokens),
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if float(temperature) > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = float(temperature)
            gen_kwargs["top_p"] = 0.9
        else:
            gen_kwargs["do_sample"] = False

        with self._torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        gen_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def _mask_minigraph_local(
    *,
    graph: dict[str, Any],
    chunk_text: str,
    chunk_id: str,
    case_name: str,
    qwen: _LocalQwenChat,
    temperature: float,
    max_new_tokens: int,
) -> tuple[dict[str, Any], str | None]:
    user = MINIGRAPH_MASK_USER.format(
        chunk_id=chunk_id or "chunk",
        case_name=case_name or "",
        text=chunk_text or "(no chunk text provided)",
        graph_json=json.dumps(graph, ensure_ascii=False, indent=2),
    )
    try:
        raw = qwen.complete_json(
            system=MINIGRAPH_MASK_SYSTEM,
            user=user,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
    except Exception as exc:
        return graph, str(exc)

    parsed = _extract_json_object(raw)
    if not isinstance(parsed, dict):
        return graph, "parse failed"

    if not masked_graph_is_compatible(graph, parsed):
        return graph, "masked graph failed structural validation (IDs or edges)"

    return parsed, None


def _call_link_llm_local(
    *,
    qwen: _LocalQwenChat,
    event_a: dict[str, Any],
    event_b: dict[str, Any],
    temperature: float,
    max_new_tokens: int,
) -> tuple[dict[str, Any] | None, str | None]:
    user = {
        "event_a": event_a,
        "event_b": event_b,
        "event_a_text_bundle": _event_text_bundle(event_a),
        "event_b_text_bundle": _event_text_bundle(event_b),
    }
    try:
        raw = qwen.complete_json(
            system=HYBRID_LINK_SYSTEM,
            user=json.dumps(user, ensure_ascii=False),
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
    except Exception as exc:
        return None, f"exception: {exc}"
    parsed = _extract_json_object(raw)
    if not isinstance(parsed, dict):
        raw_short = raw[:1200] + ("..." if len(raw) > 1200 else "")
        return None, f"parse_failed: {raw_short}"
    return parsed, None


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


def _collapse_same_event_clusters(merged_graph: dict[str, Any], same_pairs: list[tuple[str, str]]) -> tuple[dict[str, Any], dict[str, str], int]:
    events = _ensure_list(merged_graph.get("events"))
    id_to_idx = {str(ev.get("event_id")): i for i, ev in enumerate(events) if isinstance(ev, dict) and ev.get("event_id")}
    uf = _UnionFind(len(events))
    for a, b in same_pairs:
        ia, ib = id_to_idx.get(a), id_to_idx.get(b)
        if ia is None or ib is None:
            continue
        uf.union(ia, ib)
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(events)):
        groups[uf.find(i)].append(i)
    if all(len(v) == 1 for v in groups.values()):
        return merged_graph, {eid: eid for eid in id_to_idx}, 0

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
            for ref in _ensure_list(ev.get("source_refs")):
                if isinstance(ref, dict):
                    merged_source_refs.append(ref)
            idx_to_newid[m] = new_id
        if merged_source_refs:
            representative["source_refs"] = merged_source_refs
        new_events.append(representative)

    old_to_new = {old_id: idx_to_newid.get(idx, old_id) for old_id, idx in id_to_idx.items()}

    def _remap_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        seen = set()
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            src_old = _safe_text(edge.get("from_event"))
            dst_old = _safe_text(edge.get("to_event"))
            src = old_to_new.get(src_old, src_old)
            dst = old_to_new.get(dst_old, dst_old)
            relation = _safe_text(edge.get("relation"))
            if not src or not dst or src == dst:
                continue
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
        "temporal_edges": _remap_edges(_ensure_list(merged_graph.get("temporal_edges"))),
        "causal_edges": _remap_edges(_ensure_list(merged_graph.get("causal_edges"))),
    }
    return out_graph, old_to_new, max(0, len(events) - len(new_events))


def _add_llm_edges(merged_graph: dict[str, Any], decisions: list[dict[str, Any]]) -> tuple[int, int]:
    temporal = _ensure_list(merged_graph.get("temporal_edges"))
    causal = _ensure_list(merged_graph.get("causal_edges"))
    t_seen = {(str(e.get("from_event")), str(e.get("to_event")), str(e.get("relation"))) for e in temporal if isinstance(e, dict)}
    c_seen = {(str(e.get("from_event")), str(e.get("to_event")), str(e.get("relation"))) for e in causal if isinstance(e, dict)}
    t_added = 0
    c_added = 0
    for d in decisions:
        a = _safe_text(d.get("event_a"))
        b = _safe_text(d.get("event_b"))
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
                            "source": "hybrid_llm_local_qwen",
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
                            "source": "hybrid_llm_local_qwen",
                        }
                    )
                    c_seen.add(key)
                    c_added += 1
    merged_graph["temporal_edges"] = temporal
    merged_graph["causal_edges"] = causal
    return t_added, c_added


def main() -> None:
    parser = argparse.ArgumentParser(description="Mask mini-graphs and run hybrid doc-level merge using local Qwen only.")
    parser.add_argument("--input", required=True, help="Input JSONL with doc.chunks[].graph or mini_graph.")
    parser.add_argument("--output", required=True, help="Output JSONL with merged/hybrid doc-level graph.")
    parser.add_argument("--qwen-model-path", default="/datasets/ai/qwen3/", help="Local Qwen model path.")
    parser.add_argument("--qwen-device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--qwen-max-new-tokens", type=int, default=1024)
    parser.add_argument("--mask-temperature", type=float, default=0.0)
    parser.add_argument("--link-temperature", type=float, default=0.0)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--mask-max-completion-tokens", type=int, default=8192)
    parser.add_argument("--dry-run-mask", action="store_true", help="Skip LLM masking and keep graphs as-is.")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--min-similarity", type=float, default=0.20, help="Lexical cosine threshold (0-1).")
    parser.add_argument("--max-pairs", type=int, default=80)
    parser.add_argument("--link-max-completion-tokens", type=int, default=500)
    parser.add_argument("--disable-same-event-collapse", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    qwen = _LocalQwenChat(args.qwen_model_path, device=args.qwen_device, max_new_tokens=args.qwen_max_new_tokens)
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

            masked_chunks = []
            for chunk in doc.get("chunks") or []:
                chunk = dict(chunk)
                field, graph = _pick_graph(chunk)
                if not graph or not isinstance(graph, dict):
                    masked_chunks.append(chunk)
                    continue
                if args.dry_run_mask:
                    masked = deepcopy(graph)
                    err = None
                else:
                    masked, err = _mask_minigraph_local(
                        graph=graph,
                        chunk_text=str(chunk.get("text") or ""),
                        chunk_id=str(chunk.get("chunk_id") or ""),
                        case_name=str(doc.get("case_name") or doc.get("title") or doc.get("doc_id") or ""),
                        qwen=qwen,
                        temperature=float(args.mask_temperature),
                        max_new_tokens=int(args.mask_max_completion_tokens),
                    )
                chunk["masked_graph"] = masked
                if field:
                    chunk[field] = masked
                if err:
                    chunk["mask_error"] = err
                masked_chunks.append(chunk)

            masked_doc = {
                **{k: v for k, v in doc.items() if k != "chunks"},
                "chunks": masked_chunks,
                "mask_model": f"local_qwen:{args.qwen_model_path}",
                "masked_at_utc": generated_at,
            }

            merged = merger.merge_document(masked_doc)
            merged_graph = merged.get("merged_graph") or {}
            events = _ensure_list(merged_graph.get("events"))

            if len(events) >= 2:
                candidates = _pair_candidates(events, top_k=args.top_k, min_similarity=args.min_similarity)
                candidates = sorted(candidates, key=lambda x: x[2], reverse=True)[: max(0, args.max_pairs)]

                decisions: list[dict[str, Any]] = []
                same_pairs: list[tuple[str, str]] = []
                link_failures = 0
                link_failure_samples: list[dict[str, Any]] = []
                for i, j, sim in candidates:
                    ev_a = events[i]
                    ev_b = events[j]
                    decision, link_error = _call_link_llm_local(
                        qwen=qwen,
                        event_a=ev_a,
                        event_b=ev_b,
                        temperature=float(args.link_temperature),
                        max_new_tokens=int(args.link_max_completion_tokens),
                    )
                    if not decision:
                        link_failures += 1
                        if len(link_failure_samples) < 3:
                            link_failure_samples.append(
                                {
                                    "event_a": _safe_text(ev_a.get("event_id")),
                                    "event_b": _safe_text(ev_b.get("event_id")),
                                    "similarity": float(sim),
                                    "error": link_error or "unknown",
                                }
                            )
                        continue
                    row = {
                        "event_a": _safe_text(ev_a.get("event_id")),
                        "event_b": _safe_text(ev_b.get("event_id")),
                        "similarity": float(sim),
                        **decision,
                    }
                    decisions.append(row)
                    if bool(decision.get("same_event")):
                        same_pairs.append((row["event_a"], row["event_b"]))

                events_collapsed = 0
                old_to_new = {}
                if same_pairs and not args.disable_same_event_collapse:
                    merged_graph, old_to_new, events_collapsed = _collapse_same_event_clusters(merged_graph, same_pairs)
                if old_to_new:
                    for d in decisions:
                        d["event_a"] = old_to_new.get(_safe_text(d.get("event_a")), _safe_text(d.get("event_a")))
                        d["event_b"] = old_to_new.get(_safe_text(d.get("event_b")), _safe_text(d.get("event_b")))

                t_added, c_added = _add_llm_edges(merged_graph, decisions)
                merged["hybrid_merge"] = {
                    "enabled": True,
                    "llm_model": f"local_qwen:{args.qwen_model_path}",
                    "embedding_model": "local_lexical_cosine",
                    "processed_at_utc": generated_at,
                    "num_candidates": len(candidates),
                    "num_pairs_sent_to_llm": len(decisions),
                    "num_same_event_links": len(same_pairs),
                    "num_temporal_added": t_added,
                    "num_causal_added": c_added,
                    "num_events_collapsed": events_collapsed,
                    "num_link_failures": link_failures,
                    "link_failure_samples": link_failure_samples,
                    "top_k": int(args.top_k),
                    "min_similarity": float(args.min_similarity),
                    "max_pairs": int(args.max_pairs),
                    "adjudications": decisions,
                }
            else:
                merged["hybrid_merge"] = {
                    "enabled": True,
                    "llm_model": f"local_qwen:{args.qwen_model_path}",
                    "embedding_model": "local_lexical_cosine",
                    "processed_at_utc": generated_at,
                    "num_candidates": 0,
                    "num_pairs_sent_to_llm": 0,
                    "num_same_event_links": 0,
                    "num_temporal_added": 0,
                    "num_causal_added": 0,
                    "num_events_collapsed": 0,
                    "notes": "Skipped hybrid linking: fewer than 2 merged events.",
                }

            merged["merged_graph"] = merged_graph
            merged["merge_stats"] = {
                **(merged.get("merge_stats") or {}),
                "num_events": len(_ensure_list(merged_graph.get("events"))),
                "num_temporal_edges": len(_ensure_list(merged_graph.get("temporal_edges"))),
                "num_causal_edges": len(_ensure_list(merged_graph.get("causal_edges"))),
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
