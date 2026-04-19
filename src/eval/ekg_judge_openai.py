#!/usr/bin/env python3
"""
Compare baseline vs hybrid event knowledge graphs (EKG) with LLM-as-judge via OpenAI-compatible chat completions.

- Inter: same temperature, multiple models (different "raters").
- Intra: one or more models, each at multiple temperatures (stability / test–retest).

Uses AGENT_API_KEY and llm.api_base from config (see agents.config_loader).

Example:
  .venv/bin/python src/eval/ekg_judge_openai.py \\
    --baseline-json data/outputs/baseline/graph/llm_5docs.json \\
    --hybrid-jsonl data/outputs/hybrid_merge_api/hybrid_api_1doc_20260408_201517.jsonl \\
    --facts-jsonl "/work/pi_dagarwal_umass_edu/project_1/sriram/outputs/fact_extraction/US Dataset/holdout40_eval_train60_legalbert/us_holdout40_hybrid_legalbert_train60_predictions.jsonl \\
    --inter-models gpt5,  meta.llama3-3-70b, mistral-large, deepseek-r1  \\
    --inter-temperature 0.0 \\
    --intra-models gpt5 mistral-large \\
    --intra-temperatures 0.0 0.5 1.0 \\
    --output data/outputs/eval/ekg_judge_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Repo root on path for `agents`
_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agents.config_loader import get_llm_config  # noqa: E402


def _load_dotenv_project() -> None:
    try:
        from dotenv import load_dotenv

        env_file = _ROOT / ".env"
        if env_file.is_file():
            load_dotenv(env_file, override=True)
    except ImportError:
        pass


def _api_key() -> str:
    _load_dotenv_project()
    key = (os.environ.get("AGENT_API_KEY") or "").strip()
    if not key:
        raise ValueError(
            "Set AGENT_API_KEY in project .env or environment. "
            "See agents.config_loader."
        )
    return key


def _make_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install openai: pip install openai") from exc
    cfg = get_llm_config()
    return OpenAI(api_key=_api_key(), base_url=cfg.get("api_base"), timeout=180.0)


def _strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        return m.group(1).strip()
    return text


def _parse_verdict_json(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    text = _strip_code_fences(raw)
    if not text:
        return None, "empty"
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        return None, str(e)
    if not isinstance(obj, dict):
        return None, "not an object"
    return obj, None


def load_baseline_by_doc(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for row in data.get("docs") or []:
        did = str(row.get("doc_id") or "")
        if did:
            out[did] = row.get("graph") or {}
    return out


def load_hybrid_by_doc(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            did = str(row.get("doc_id") or "")
            if did:
                out[did] = row.get("merged_graph") or row.get("graph") or {}
    return out


def _record_doc_id(rec: dict[str, Any]) -> str:
    """Same key order as baseline/graph-creation/llm-model.py."""
    return str(rec.get("doc_id") or rec.get("case_id") or rec.get("title") or "")


def load_facts_by_doc(path: Path, max_chars_per_doc: int) -> dict[str, str]:
    """Build labeled fact text per doc (pred_label == 1)."""
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            did = _record_doc_id(rec)
            if not did:
                continue
            sents = rec.get("sentences") or []
            parts: list[str] = []
            for s in sents:
                if not isinstance(s, dict):
                    continue
                pl = s.get("pred_label")
                if pl != 1 and pl != "1":
                    continue
                sid = s.get("id")
                text = (s.get("text") or "").strip()
                if text:
                    parts.append(f"[S{sid}] {text}")
            blob = "\n".join(parts)
            if max_chars_per_doc > 0 and len(blob) > max_chars_per_doc:
                blob = blob[: max_chars_per_doc] + "\n[... truncated ...]"
            out[did] = blob
    return out


def _count_grounded_events(events: list[Any]) -> int:
    n = 0
    for e in events:
        if not isinstance(e, dict):
            continue
        ev = e.get("evidence") or {}
        ids = ev.get("sentence_ids") or []
        trig = (e.get("trigger") or {}).get("sentence_id")
        if ids or trig is not None:
            n += 1
    return n


def graph_digest(name: str, g: dict[str, Any]) -> str:
    ents = g.get("entities") or []
    ev = g.get("events") or []
    te = g.get("temporal_edges") or []
    ce = g.get("causal_edges") or []
    grounded = _count_grounded_events(ev)
    lines = [
        f"=== {name} ===",
        f"entities={len(ents)} events={len(ev)} temporal_edges={len(te)} causal_edges={len(ce)}",
        f"events_with_evidence_or_trigger_sentence_id={grounded}/{len(ev)}",
    ]
    types = [str(e.get("event_type") or "?") for e in ev if isinstance(e, dict)]
    if types:
        shown = types[:45]
        tail = ", ..." if len(types) > 45 else ""
        lines.append("event_types: " + ", ".join(shown) + tail)
    lines.append("sample temporal_edges:")
    for e in te[:12]:
        if isinstance(e, dict):
            lines.append(f"  {e.get('from_event')} --{e.get('relation')}--> {e.get('to_event')}")
    lines.append("sample causal_edges:")
    for e in ce[:12]:
        if isinstance(e, dict):
            lines.append(f"  {e.get('from_event')} --{e.get('relation')}--> {e.get('to_event')}")
    return "\n".join(lines)


JUDGE_SYSTEM = """You are an expert evaluator of legal event knowledge graphs (EKG).
Strict structure of the graph is: Events are graph nodes; temporal and causal relations are edges between events.
The source material is a set of fact sentences from a legal complaint (labeled [S#]).

You compare two candidate graphs for the SAME document:
- BASELINE: one-shot LLM extraction from the fact sentences.
- HYBRID: chunk-level extraction merged with hybrid refinement (deterministic + semantic/LLM linking), that understand the context of the document chunkwise and merges them proeprly.

Rate each dimension as exactly one of: BASELINE_BETTER, HYBRID_BETTER, TIE
Use TIE only when neither is clearly better on that dimension.

Dimensions:
1. EVENT_COMPLETENESS — captures the important events implied by the facts
2. EVIDENCE_GROUNDING — stronger ties to source sentences / evidence fields
3. EDGE_QUALITY — temporal and causal edges are plausible and not random cross-links
4. OVERALL_WINNER — which EKG better represents the document overall

Return STRICT JSON ONLY (no markdown), one object with exactly these keys:
{"EVENT_COMPLETENESS":"","EVIDENCE_GROUNDING":"","EDGE_QUALITY":"","OVERALL_WINNER":""}
Values must be BASELINE_BETTER or HYBRID_BETTER."""


def build_user_message(doc_id: str, facts_text: str, baseline_d: str, hybrid_d: str) -> str:
    facts_block = facts_text.strip() or "(no fact sentences provided for this doc_id)"
    return f"""document_id: {doc_id}

FACT SENTENCES (pred_label=1; ground truth text for judging support):
{facts_block}

{baseline_d}

{hybrid_d}

Return the JSON verdict object only."""


def chat_complete(
    client: Any,
    *,
    model: str,
    temperature: float,
    messages: list[dict[str, str]],
    max_completion_tokens: int,
) -> str:
    req: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_completion_tokens": max_completion_tokens,
    }
    try:
        resp = client.chat.completions.create(**req)
        return (resp.choices[0].message.content or "").strip()
    except TypeError:
        req.pop("max_completion_tokens", None)
        req["max_tokens"] = max_completion_tokens
        resp = client.chat.completions.create(**req)
        return (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        err = str(exc).lower()
        if "max_completion_tokens" in err or "unknown parameter" in err:
            req.pop("max_completion_tokens", None)
            req["max_tokens"] = max_completion_tokens
            resp = client.chat.completions.create(**req)
            return (resp.choices[0].message.content or "").strip()
        if "only temperature=1 is supported" in err:
            req["temperature"] = 1.0
            resp = client.chat.completions.create(**req)
            return (resp.choices[0].message.content or "").strip()
        raise


def run_one_judge(
    client: Any,
    *,
    model: str,
    temperature: float,
    doc_id: str,
    facts_text: str,
    baseline_graph: dict[str, Any],
    hybrid_graph: dict[str, Any],
    max_completion_tokens: int,
) -> dict[str, Any]:
    b_d = graph_digest("BASELINE_EKG", baseline_graph)
    h_d = graph_digest("HYBRID_EKG", hybrid_graph)
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": build_user_message(doc_id, facts_text, b_d, h_d)},
    ]
    t0 = time.perf_counter()
    raw = chat_complete(
        client,
        model=model,
        temperature=temperature,
        messages=messages,
        max_completion_tokens=max_completion_tokens,
    )
    elapsed = time.perf_counter() - t0
    verdict, err = _parse_verdict_json(raw)
    return {
        "doc_id": doc_id,
        "model": model,
        "temperature": temperature,
        "elapsed_sec": round(elapsed, 3),
        "verdict": verdict,
        "parse_error": err,
        "raw_response_preview": raw[:2000] + ("..." if len(raw) > 2000 else ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-as-judge: baseline vs hybrid EKG (OpenAI-compatible API).")
    parser.add_argument("--baseline-json", type=Path, required=True)
    parser.add_argument("--hybrid-jsonl", type=Path, required=True)
    parser.add_argument("--facts-jsonl", type=Path, required=True, help="Per-doc fact sentences (e.g. LegalBERT predictions JSONL).")
    parser.add_argument("--inter-models", nargs="*", default=[], help="Models for inter-rater mode (each at --inter-temperature).")
    parser.add_argument("--inter-temperature", type=float, default=0.0)
    parser.add_argument("--intra-model", type=str, default="", help="Single model for intra (use --intra-models for several).")
    parser.add_argument(
        "--intra-models",
        nargs="*",
        default=[],
        help="Models for intra-rater mode (each run at every --intra-temperature).",
    )
    parser.add_argument("--intra-temperatures", nargs="*", type=float, default=[], help="Temperatures for intra-rater mode.")
    parser.add_argument("--max-facts-chars", type=int, default=20000, help="Truncate fact text per doc; 0 = no limit.")
    parser.add_argument("--max-completion-tokens", type=int, default=2048)
    parser.add_argument("--doc-ids", nargs="*", default=[], help="Optional subset of doc_id strings.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    intra_models_list: list[str] = list(args.intra_models) if args.intra_models else []
    if not intra_models_list and args.intra_model.strip():
        intra_models_list = [args.intra_model.strip()]
    has_intra = bool(args.intra_temperatures and intra_models_list)

    if not args.inter_models and not has_intra:
        parser.error(
            "Provide --inter-models and/or --intra-temperatures with --intra-model or --intra-models.",
        )

    if args.intra_temperatures and not intra_models_list:
        parser.error("--intra-temperatures requires --intra-model and/or --intra-models.")

    baseline = load_baseline_by_doc(args.baseline_json)
    hybrid = load_hybrid_by_doc(args.hybrid_jsonl)
    common = sorted(set(baseline) & set(hybrid))
    if args.doc_ids:
        want = set(args.doc_ids)
        common = [d for d in common if d in want]
    if not common:
        raise SystemExit("No overlapping doc_id between baseline and hybrid (after optional filter).")

    max_fc = args.max_facts_chars
    facts_map = load_facts_by_doc(args.facts_jsonl, max_fc if max_fc > 0 else 10**9)

    client = _make_client()
    generated_at = datetime.now(timezone.utc).isoformat()

    inter_rows: list[dict[str, Any]] = []
    intra_rows: list[dict[str, Any]] = []

    for model in args.inter_models:
        for doc_id in common:
            facts_text = facts_map.get(doc_id, "")
            inter_rows.append(
                run_one_judge(
                    client,
                    model=model,
                    temperature=float(args.inter_temperature),
                    doc_id=doc_id,
                    facts_text=facts_text,
                    baseline_graph=baseline[doc_id],
                    hybrid_graph=hybrid[doc_id],
                    max_completion_tokens=args.max_completion_tokens,
                )
            )

    if intra_models_list and args.intra_temperatures:
        for model in intra_models_list:
            for temp in args.intra_temperatures:
                for doc_id in common:
                    facts_text = facts_map.get(doc_id, "")
                    intra_rows.append(
                        run_one_judge(
                            client,
                            model=model,
                            temperature=float(temp),
                            doc_id=doc_id,
                            facts_text=facts_text,
                            baseline_graph=baseline[doc_id],
                            hybrid_graph=hybrid[doc_id],
                            max_completion_tokens=args.max_completion_tokens,
                        )
                    )

    payload = {
        "generated_at_utc": generated_at,
        "baseline_json": str(args.baseline_json.resolve()),
        "hybrid_jsonl": str(args.hybrid_jsonl.resolve()),
        "facts_jsonl": str(args.facts_jsonl.resolve()),
        "doc_ids": common,
        "inter": {
            "models": list(args.inter_models),
            "temperature": args.inter_temperature,
            "results": inter_rows,
        },
        "intra": {
            "models": list(intra_models_list),
            "model": (intra_models_list[0] if len(intra_models_list) == 1 else None),
            "temperatures": list(args.intra_temperatures),
            "results": intra_rows,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Docs evaluated: {len(common)}")
    print(f"Inter calls: {len(inter_rows)}  Intra calls: {len(intra_rows)}")


if __name__ == "__main__":
    main()
