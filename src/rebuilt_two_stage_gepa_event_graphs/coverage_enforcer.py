#!/usr/bin/env python3
"""Deterministic post-processing to enforce verb-sentence event coverage."""
import copy
import re
from typing import Any

_SENTENCE_PATTERN = re.compile(r"\[S(\d+)\]\s*([\s\S]*?)(?=\n\[S\d+\]|$)")
_WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z'\-]*")

_LIGHT_VERBS = {
    "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "do", "does", "did",
}


def parse_sentence_map(text: str) -> dict[int, str]:
    out: dict[int, str] = {}
    for match in _SENTENCE_PATTERN.finditer(text or ""):
        sid = int(match.group(1))
        sentence = match.group(2).strip().replace("\n", " ")
        out[sid] = sentence
    return out


def _looks_like_verb(word: str) -> bool:
    w = (word or "").lower()
    if len(w) < 3:
        return False
    if w in _LIGHT_VERBS:
        return False
    if w.endswith("ed") or w.endswith("ing"):
        return True
    if w in {
        "hire", "hired", "report", "reported", "assign", "assigned",
        "plan", "planned", "cause", "caused", "damage", "damaged",
        "cancel", "cancelled", "secure", "secured", "disclose", "disclosed",
        "terminate", "terminated", "ask", "asked", "respond", "responded",
        "state", "stated", "proceed", "proceeded", "impress", "impressed",
        "congratulate", "congratulated", "shock", "shocked", "change", "changed",
    }:
        return True
    return False


def contains_action_verb(sentence: str) -> bool:
    words = [m.group(0) for m in _WORD_PATTERN.finditer(sentence or "")]
    if not words:
        return False
    for word in words:
        if _looks_like_verb(word):
            return True
    return False


def _next_event_id(events: list[dict[str, Any]]) -> str:
    max_id = 0
    for event in events:
        eid = str(event.get("event_id") or "")
        match = re.match(r"EV(\d+)$", eid)
        if match:
            max_id = max(max_id, int(match.group(1)))
    return f"EV{max_id + 1}"


def _event_evidence_sentence_ids(graph: dict[str, Any]) -> set[int]:
    used: set[int] = set()
    for event in graph.get("events") or []:
        for sid in (event.get("evidence") or {}).get("sentence_ids") or []:
            if str(sid).isdigit():
                used.add(int(sid))
    return used


def enforce_verb_sentence_coverage(
    graph: dict[str, Any],
    text: str,
    *,
    fallback_event_type: str = "SUPPORTING_FACT",
    fallback_confidence: float = 0.35,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Add fallback events for uncovered verb-bearing sentences."""
    out = copy.deepcopy(graph or {"entities": [], "events": [], "temporal_edges": [], "causal_edges": []})
    out.setdefault("entities", [])
    out.setdefault("events", [])
    out.setdefault("temporal_edges", [])
    out.setdefault("causal_edges", [])

    sentence_map = parse_sentence_map(text)
    verb_sentence_ids = sorted(sid for sid, sentence in sentence_map.items() if contains_action_verb(sentence))
    covered_ids = _event_evidence_sentence_ids(out)
    missing_ids = [sid for sid in verb_sentence_ids if sid not in covered_ids]

    added_events = []
    for sid in missing_ids:
        sentence = sentence_map.get(sid, "")
        words = [m.group(0) for m in _WORD_PATTERN.finditer(sentence)]
        main_verb = ""
        for word in words:
            if _looks_like_verb(word):
                main_verb = word.lower()
                break
        if not main_verb and words:
            main_verb = words[0].lower()

        event_id = _next_event_id(out["events"])
        event = {
            "event_id": event_id,
            "event_type": fallback_event_type,
            "main_verb": main_verb,
            "trigger": {
                "sentence_id": sid,
                "span_text": main_verb,
            },
            "participants": [],
            "time": {
                "kind": "UNKNOWN",
                "raw_span": "",
                "normalized": None,
                "confidence": 0.0,
            },
            "evidence": {
                "sentence_ids": [sid],
                "snippets": [{"sentence_id": sid, "text": sentence}],
            },
            "confidence": fallback_confidence,
        }
        out["events"].append(event)
        added_events.append(event_id)

    metrics = {
        "num_sentences": len(sentence_map),
        "num_verb_sentences": len(verb_sentence_ids),
        "num_covered_verb_sentences_before": len([sid for sid in verb_sentence_ids if sid in covered_ids]),
        "num_added_fallback_events": len(added_events),
        "uncovered_verb_sentence_ids_before": missing_ids,
        "added_event_ids": added_events,
    }
    return out, metrics
