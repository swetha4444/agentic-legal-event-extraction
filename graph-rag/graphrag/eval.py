from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .retrieval import RetrievalConfig, retrieve
from .store import GraphStore


@dataclass
class EvalQuestion:
    qid: str
    question: str
    expected_event_ids: list[str]


def load_questions(path: str | Path) -> list[EvalQuestion]:
    questions: list[EvalQuestion] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            questions.append(
                EvalQuestion(
                    qid=str(row["id"]),
                    question=str(row["question"]),
                    expected_event_ids=[str(x) for x in row.get("expected_event_ids") or []],
                )
            )
    return questions


def _score_question(found_event_ids: list[str], expected_event_ids: list[str]) -> dict[str, Any]:
    expected = list(dict.fromkeys(expected_event_ids))
    found = list(dict.fromkeys(found_event_ids))
    found_set = set(found)
    expected_set = set(expected)
    hits = [eid for eid in found if eid in expected_set]
    missed = [eid for eid in expected if eid not in found_set]
    precision = len(hits) / len(found) if found else 0.0
    recall = len(hits) / len(expected) if expected else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "hits": hits,
        "missed": missed,
        "precision_at_k": precision,
        "recall_at_k": recall,
        "f1_at_k": f1,
    }


def run_retrieval_eval(
    *,
    db_path: str | Path,
    question_path: str | Path,
    top_k: int = 8,
    seed_limit: int = 12,
    graph_hops: int = 2,
) -> dict[str, Any]:
    store = GraphStore(db_path)
    questions = load_questions(question_path)
    rows: list[dict[str, Any]] = []

    totals = {
        "count": 0,
        "precision_at_k": 0.0,
        "recall_at_k": 0.0,
        "f1_at_k": 0.0,
        "full_match_count": 0,
    }

    for item in questions:
        bundle = retrieve(
            store,
            item.question,
            config=RetrievalConfig(
                seed_limit=seed_limit,
                top_k=top_k,
                graph_hops=graph_hops,
            ),
        )
        found_ids = [hit.event_id for hit in bundle.events]
        scores = _score_question(found_ids, item.expected_event_ids)
        row = {
            "id": item.qid,
            "question": item.question,
            "expected_event_ids": item.expected_event_ids,
            "retrieved_event_ids": found_ids,
            **scores,
            "retrieval_summary": bundle.summary,
        }
        rows.append(row)
        totals["count"] += 1
        totals["precision_at_k"] += scores["precision_at_k"]
        totals["recall_at_k"] += scores["recall_at_k"]
        totals["f1_at_k"] += scores["f1_at_k"]
        if not scores["missed"]:
            totals["full_match_count"] += 1

    count = max(totals["count"], 1)
    aggregate = {
        "question_count": totals["count"],
        "avg_precision_at_k": totals["precision_at_k"] / count,
        "avg_recall_at_k": totals["recall_at_k"] / count,
        "avg_f1_at_k": totals["f1_at_k"] / count,
        "full_match_rate": totals["full_match_count"] / count,
        "top_k": top_k,
        "seed_limit": seed_limit,
        "graph_hops": graph_hops,
    }
    return {
        "aggregate": aggregate,
        "questions": rows,
    }
