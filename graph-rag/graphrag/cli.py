from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import get_llm_config
from .eval import run_retrieval_eval
from .llm import generate_answer
from .prompting import build_prompt
from .retrieval import RetrievalConfig, retrieve
from .store import GraphStore, build_database


def _default_db() -> str:
    root = Path(__file__).resolve().parent.parent
    return str(root / ".artifacts" / "graphrag.sqlite")


def _cmd_index(args: argparse.Namespace) -> int:
    stats = build_database(
        db_path=args.db,
        merged_jsonl=args.merged_jsonl,
        chunk_jsonl=args.chunk_jsonl,
        reset=not args.no_reset,
    )
    print(json.dumps({"db": args.db, "stats": stats}, indent=2))
    return 0


def _bundle_to_json(bundle) -> dict:
    return {
        "summary": bundle.summary,
        "events": [
            {
                "doc_id": hit.doc_id,
                "event_id": hit.event_id,
                "event_type": hit.event_type,
                "score": hit.score,
                "lexical_score": hit.lexical_score,
                "graph_score": hit.graph_score,
                "entity_score": hit.entity_score,
                "temporal_score": hit.temporal_score,
                "evidence_score": hit.evidence_score,
                "is_supporting_fact": hit.is_supporting_fact,
                "neighbors": hit.neighbors,
                "chunk_ids": hit.chunk_ids,
                "text": hit.text,
            }
            for hit in bundle.events
        ],
        "entities": bundle.entities,
        "edges": bundle.edges,
        "chunks": bundle.chunks,
    }


def _cmd_query(args: argparse.Namespace) -> int:
    store = GraphStore(args.db)
    bundle = retrieve(
        store,
        args.question,
        config=RetrievalConfig(
            seed_limit=args.seed_limit,
            top_k=args.top_k,
            graph_hops=args.graph_hops,
        ),
    )
    payload = _bundle_to_json(bundle)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _cmd_prompt(args: argparse.Namespace) -> int:
    store = GraphStore(args.db)
    bundle = retrieve(
        store,
        args.question,
        config=RetrievalConfig(
            seed_limit=args.seed_limit,
            top_k=args.top_k,
            graph_hops=args.graph_hops,
        ),
    )
    messages = build_prompt(bundle)
    print(json.dumps(messages, indent=2, ensure_ascii=False))
    return 0


def _cmd_answer(args: argparse.Namespace) -> int:
    store = GraphStore(args.db)
    bundle = retrieve(
        store,
        args.question,
        config=RetrievalConfig(
            seed_limit=args.seed_limit,
            top_k=args.top_k,
            graph_hops=args.graph_hops,
        ),
    )
    messages = build_prompt(bundle)
    llm_cfg = get_llm_config()
    answer = generate_answer(
        messages,
        model=args.model,
        api_key=llm_cfg["api_key"],
        base_url=args.base_url or llm_cfg["api_base"],
        temperature=args.temperature,
    )
    print(answer)
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    report = run_retrieval_eval(
        db_path=args.db,
        question_path=args.questions,
        top_k=args.top_k,
        seed_limit=args.seed_limit,
        graph_hops=args.graph_hops,
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    llm_cfg = get_llm_config()
    parser = argparse.ArgumentParser(description="Graph-native RAG prototype over legal event graphs.")
    sub = parser.add_subparsers(dest="command", required=True)

    index = sub.add_parser("index", help="Index merged graph JSONL into a local SQLite DB.")
    index.add_argument("--merged-jsonl", required=True, help="Merged graph JSONL to read.")
    index.add_argument("--chunk-jsonl", default=None, help="Optional chunk graph JSONL for provenance text.")
    index.add_argument("--db", default=_default_db(), help="Local SQLite database path inside this folder.")
    index.add_argument("--no-reset", action="store_true", help="Append instead of resetting the DB.")
    index.set_defaults(func=_cmd_index)

    for name, func in (("query", _cmd_query), ("prompt", _cmd_prompt), ("answer", _cmd_answer)):
        cmd = sub.add_parser(name, help=f"{name.title()} over the indexed graph DB.")
        cmd.add_argument("--db", default=_default_db(), help="Local SQLite database path inside this folder.")
        cmd.add_argument("--question", required=True, help="User question.")
        cmd.add_argument("--top-k", type=int, default=8, help="How many events/facts to return.")
        cmd.add_argument("--seed-limit", type=int, default=12, help="How many lexical seeds to expand from.")
        cmd.add_argument("--graph-hops", type=int, default=2, help="How many graph expansion hops to use.")
        if name == "answer":
            cmd.add_argument(
                "--model",
                default=llm_cfg["model"],
                help="OpenAI-compatible chat model name.",
            )
            cmd.add_argument("--base-url", default=None, help="Optional OpenAI-compatible base URL.")
            cmd.add_argument("--temperature", type=float, default=0.0, help="Generation temperature.")
        cmd.set_defaults(func=func)

    eval_cmd = sub.add_parser("eval", help="Run a retrieval evaluation question set.")
    eval_cmd.add_argument("--db", default=_default_db(), help="Local SQLite database path inside this folder.")
    eval_cmd.add_argument(
        "--questions",
        default=str(Path(__file__).resolve().parent.parent / "eval" / "questions_smoke.jsonl"),
        help="Question set JSONL path.",
    )
    eval_cmd.add_argument("--top-k", type=int, default=8, help="How many events/facts to return.")
    eval_cmd.add_argument("--seed-limit", type=int, default=12, help="How many lexical seeds to expand from.")
    eval_cmd.add_argument("--graph-hops", type=int, default=2, help="How many graph expansion hops to use.")
    eval_cmd.add_argument("--output", default=None, help="Optional JSON report path inside this folder.")
    eval_cmd.set_defaults(func=_cmd_eval)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
