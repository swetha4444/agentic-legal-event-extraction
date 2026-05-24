"""Serialize merged EKG rows into compact text for transformer or ablation baselines."""

from __future__ import annotations

from typing import Any, Dict, List


def _trigger_text(event: Dict[str, Any]) -> str:
    t = event.get("trigger") or {}
    if isinstance(t, dict):
        return str(t.get("span_text") or "").strip()
    return ""


def linearize_merged_graph(
    merged_graph: Dict[str, Any],
    max_events: int = 120,
    max_entities: int = 80,
) -> str:
    events = list((merged_graph or {}).get("events") or [])[:max_events]
    entities = list((merged_graph or {}).get("entities") or [])[:max_entities]

    ent_lines: List[str] = []
    for ent in entities:
        nm = str(ent.get("name") or "").strip()
        if not nm:
            continue
        role = str(ent.get("canonical_role") or "UNKNOWN").strip()
        ent_lines.append(f"- {nm} (role={role})")

    ev_lines: List[str] = []
    for ev in events:
        et = str(ev.get("event_type") or "UNKNOWN").strip()
        trig = _trigger_text(ev)
        sids = ev.get("sentence_ids") or []
        sid_txt = ""
        if isinstance(sids, list) and sids:
            sid_txt = ",".join(str(x) for x in sids[:6])
        if trig:
            ev_lines.append(f"- [{et}] {trig} (sent_ids={sid_txt})")
        else:
            ev_lines.append(f"- [{et}] (sent_ids={sid_txt})")

    return (
        "ENTITIES:\n"
        + ("\n".join(ent_lines) if ent_lines else "- (none)")
        + "\n\nEVENTS:\n"
        + ("\n".join(ev_lines) if ev_lines else "- (none)")
    )


def linearize_ekg_row(row: Dict[str, Any]) -> str:
    mg = row.get("merged_graph") or {}
    header = f"DOC_ID: {row.get('doc_id','')}\nNUM_CHUNKS: {row.get('num_chunks','')}\n\n"
    return header + linearize_merged_graph(mg)
