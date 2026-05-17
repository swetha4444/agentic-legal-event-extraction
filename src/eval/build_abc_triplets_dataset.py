#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path


TOPIC_RULES: dict[str, list[str]] = {
    "employment_discrimination": [
        r"\btitle vii\b",
        r"\bhostile work environment\b",
        r"\bretaliation\b",
        r"\bwrongful termination\b",
        r"\bhr\b",
        r"\beeoc\b",
        r"\bdiscrimination\b",
    ],
    "civil_rights_police": [
        r"\b42 u\.?s\.?c\.?\s*1983\b",
        r"\bexcessive force\b",
        r"\bunlawful arrest\b",
        r"\bfalse arrest\b",
        r"\bfourth amendment\b",
        r"\bsearch and seizure\b",
    ],
    "contract_commercial": [
        r"\bbreach of contract\b",
        r"\bagreement\b",
        r"\bpurchase order\b",
        r"\binvoice\b",
        r"\bcommercial\b",
        r"\bindemnif\w+\b",
    ],
    "personal_injury": [
        r"\bnegligence\b",
        r"\bslip and fall\b",
        r"\bpremises liability\b",
        r"\bduty of care\b",
        r"\bpersonal injury\b",
        r"\bdamages\b",
    ],
    "disability_accommodation": [
        r"\bada\b",
        r"\breasonable accommodation\b",
        r"\bdisabilit\w+\b",
        r"\binteractive process\b",
    ],
    "wage_hour": [
        r"\bfair labor standards act\b",
        r"\bflsa\b",
        r"\bovertime\b",
        r"\bunpaid wages\b",
        r"\bminimum wage\b",
    ],
    "housing_property": [
        r"\beviction\b",
        r"\blandlord\b",
        r"\btenant\b",
        r"\blease\b",
        r"\bproperty\b",
        r"\bforeclosure\b",
    ],
    "consumer_finance_debt": [
        r"\bfdcpa\b",
        r"\bdebt collection\b",
        r"\bcredit report\b",
        r"\bconsumer\b",
        r"\bloan\b",
        r"\bmortgage\b",
    ],
    "education_university": [
        r"\btitle ix\b",
        r"\buniversity\b",
        r"\bschool district\b",
        r"\bstudent\b",
        r"\bprofessor\b",
    ],
    "medical_healthcare": [
        r"\bmedical\b",
        r"\bphysician\b",
        r"\bhospital\b",
        r"\bmalpractice\b",
        r"\btreatment\b",
    ],
}

EVENT_RULES: dict[str, list[str]] = {
    "hiring": [r"\bhired\b", r"\bemployed\b", r"\bposition\b"],
    "termination": [r"\bterminated\b", r"\bdischarged\b", r"\bfired\b", r"\bconstructive discharge\b"],
    "complaint_filed": [r"\bcomplaint\b", r"\blawsuit\b", r"\bcivil action\b", r"\bcauses of action\b"],
    "internal_complaint": [r"\breported\b", r"\bcomplained\b", r"\bhr\b", r"\bsupervisor\b"],
    "retaliation": [r"\bretaliation\b", r"\bretaliat\w+\b"],
    "harassment": [r"\bharass\w+\b", r"\bhostile work environment\b"],
    "injury_or_harm": [r"\binjury\b", r"\bhurt\b", r"\bassault\b", r"\btrauma\b"],
    "arrest_search": [r"\barrest\b", r"\bdetained\b", r"\bsearch\b", r"\bseizure\b"],
    "contract_breach": [r"\bbreach\b", r"\bfailed to perform\b", r"\bnonpayment\b", r"\bdefault\b"],
    "accommodation_request": [r"\baccommodation\b", r"\brequest(ed)? accommodation\b", r"\binteractive process\b"],
}

STOPWORDS = {
    "the", "and", "for", "that", "with", "this", "from", "are", "was", "were", "has", "have", "had",
    "into", "their", "there", "here", "such", "shall", "would", "could", "should", "under", "over",
    "plaintiff", "defendant", "court", "complaint", "action", "case", "count", "against", "alleges",
    "alleged", "federal", "state", "district", "civil", "rights", "damages", "relief", "jurisdiction",
}


@dataclass
class DocFeatures:
    path: str
    court: str
    topics: set[str]
    events: set[str]
    tokens: set[str]


def _text_from_doc(path: Path) -> str:
    raw = json.loads(path.read_text(encoding="utf-8"))
    parts: list[str] = []
    for sec in raw.get("sections") or []:
        name = sec.get("name")
        if isinstance(name, str):
            parts.append(name)
        for sent in sec.get("sentences") or []:
            if isinstance(sent, str):
                parts.append(sent)
    return "\n".join(parts)


def _extract_topics(text: str) -> set[str]:
    out: set[str] = set()
    for topic, pats in TOPIC_RULES.items():
        if any(re.search(p, text) for p in pats):
            out.add(topic)
    return out or {"other_general_civil"}


def _extract_events(text: str) -> set[str]:
    out: set[str] = set()
    for event, pats in EVENT_RULES.items():
        if any(re.search(p, text) for p in pats):
            out.add(event)
    return out


def _extract_tokens(text: str) -> set[str]:
    toks = re.findall(r"[a-z][a-z0-9_]{2,}", text)
    return {t for t in toks if t not in STOPWORDS}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _event_overlap(a: DocFeatures, b: DocFeatures) -> int:
    return len(a.events & b.events)


def _topic_overlap(a: DocFeatures, b: DocFeatures) -> int:
    return len(a.topics & b.topics)


def build_doc_features(processed_root: Path) -> list[DocFeatures]:
    docs: list[DocFeatures] = []
    for court_dir in sorted(processed_root.glob("courtlistener_processed_*")):
        if not court_dir.is_dir():
            continue
        court = court_dir.name.replace("courtlistener_processed_", "")
        for p in sorted(court_dir.glob("*.json")):
            text = _text_from_doc(p).lower()
            docs.append(
                DocFeatures(
                    path=str(p.resolve()),
                    court=court,
                    topics=_extract_topics(text),
                    events=_extract_events(text),
                    tokens=_extract_tokens(text),
                )
            )
    return docs


def choose_triplets(docs: list[DocFeatures], target: int, seed: int) -> list[dict[str, str]]:
    rnd = random.Random(seed)
    by_court: dict[str, list[DocFeatures]] = {}
    for d in docs:
        by_court.setdefault(d.court, []).append(d)
    for arr in by_court.values():
        rnd.shuffle(arr)

    # Round-robin anchors by court for diversity across all datasets.
    anchors: list[DocFeatures] = []
    while True:
        progressed = False
        for court in sorted(by_court):
            arr = by_court[court]
            if arr:
                anchors.append(arr.pop())
                progressed = True
        if not progressed:
            break

    out: list[dict[str, str]] = []
    used_signatures: set[tuple[str, str, str]] = set()

    for a in anchors:
        positives = [
            d
            for d in docs
            if d.path != a.path
            and _topic_overlap(a, d) >= 1
            and (_event_overlap(a, d) >= 1 or (_topic_overlap(a, d) >= 2))
        ]
        if not positives:
            positives = [
                d
                for d in docs
                if d.path != a.path
                and (_topic_overlap(a, d) >= 1 or _event_overlap(a, d) >= 1)
            ]
        if not positives:
            continue
        positives.sort(
            key=lambda d: (
                _event_overlap(a, d),
                _topic_overlap(a, d),
                _jaccard(a.tokens, d.tokens),
                1 if d.court != a.court else 0,
            ),
            reverse=True,
        )
        pos = positives[0]

        negatives = [
            d
            for d in docs
            if d.path not in {a.path, pos.path}
            and _topic_overlap(a, d) == 0
            and _event_overlap(a, d) == 0
            and _jaccard(a.tokens, d.tokens) > 0.03
        ]
        if not negatives:
            negatives = [
                d
                for d in docs
                if d.path not in {a.path, pos.path}
                and _event_overlap(a, d) == 0
                and _topic_overlap(a, d) <= 1
                and _jaccard(a.tokens, d.tokens) > 0.03
            ]
        if not negatives:
            continue
        negatives.sort(
            key=lambda d: (
                _jaccard(a.tokens, d.tokens),
                1 if d.court != a.court else 0,
            ),
            reverse=True,
        )
        neg = negatives[0]

        b_is_similar = rnd.random() < 0.5
        b, c = (pos, neg) if b_is_similar else (neg, pos)
        similar = "b" if b_is_similar else "c"
        shared_topics = sorted(a.topics & pos.topics)
        shared_events = sorted(a.events & pos.events)
        negative_lex = _jaccard(a.tokens, neg.tokens)
        desc = (
            f"Anchor and {similar} share contextual signals: topics={shared_topics[:3]} "
            f"events={shared_events[:3] if shared_events else []}. "
            f"The other document is a lexical hard negative (token_jaccard={negative_lex:.3f}) "
            f"but lacks anchor's topic/event structure."
        )
        sig = (a.path, b.path, c.path)
        if sig in used_signatures:
            continue
        used_signatures.add(sig)
        out.append(
            {
                "anchor_doc_path": a.path,
                "b_doc_path": b.path,
                "c_doc_path": c.path,
                "which_is_similar_to_anchor": similar,
                "description": desc,
            }
        )
        if len(out) >= target:
            break

    return out


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "anchor_doc_path",
                "b_doc_path",
                "c_doc_path",
                "which_is_similar_to_anchor",
                "description",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build A/B/C legal similarity triplet dataset from processed CourtListener docs.")
    ap.add_argument(
        "--processed-root",
        type=Path,
        default=Path("courtlistener-pipeline-generic"),
        help="Folder containing courtlistener_processed_* directories.",
    )
    ap.add_argument("--target-size", type=int, default=110, help="Number of triplets to generate (recommend 100-120).")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-csv", type=Path, default=Path("data/eval/abc_triplets_110.csv"))
    ap.add_argument("--out-jsonl", type=Path, default=Path("data/eval/abc_triplets_110.jsonl"))
    args = ap.parse_args()

    docs = build_doc_features(args.processed_root)
    if len(docs) < args.target_size:
        raise SystemExit(f"Not enough docs: found {len(docs)} docs for target {args.target_size}.")

    rows = choose_triplets(docs, args.target_size, args.seed)
    if len(rows) < 100:
        raise SystemExit(
            f"Only generated {len(rows)} triplets. Try smaller target-size or relax negative constraints."
        )

    write_csv(rows, args.out_csv)
    write_jsonl(rows, args.out_jsonl)
    print(f"docs={len(docs)} triplets={len(rows)}")
    print(f"csv={args.out_csv}")
    print(f"jsonl={args.out_jsonl}")


if __name__ == "__main__":
    main()
