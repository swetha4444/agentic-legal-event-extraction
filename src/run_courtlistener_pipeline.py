"""
Data pipeline from CourtListener: cluster-first (only cases with court opinion).
We fetch from the clusters API (case law) so every record has a court opinion; then for each
cluster we fetch docket metadata and parties by docket_id (parties from docket/Parties API, or caption).

Usage (from project root):
  export COURTLISTENER_API_TOKEN=your_token
  python -m src.run_courtlistener_pipeline [--config FILE] [--max-cases N] [--out FILE]
"""
import argparse
import json
from pathlib import Path
from typing import Optional

import yaml

from src.data.courtlistener_client import fetch_clusters


def _load_config_file(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _pick_document_text(rec: dict, prefer_complaint: bool = True) -> str:
    """Return best available document text: complaint snippet first, else first doc snippet."""
    docs = rec.get("documents") or []
    if not docs:
        return ""
    if prefer_complaint:
        for d in docs:
            desc = (d.get("description") or "").lower()
            if "complaint" in desc:
                return (d.get("snippet") or "").strip()
    if docs:
        return (docs[0].get("snippet") or "").strip()
    return ""


def run(
    max_cases: Optional[int] = None,
    nature_of_suit: Optional[str] = None,
    out_path: Optional[str] = None,
    config_path: Optional[Path] = None,
):
    """Run cluster-first pipeline: only cases with court opinion; enrich with docket metadata and parties by docket_id."""
    default_config = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    path_to_load = config_path or default_config
    cfg = _load_config_file(path_to_load)
    cl_cfg = cfg.get("courtlistener", {})
    paths_cfg = cfg.get("paths", {})

    nature_of_suit = nature_of_suit or cl_cfg.get("nature_of_suit") or "442"
    out_path = out_path or paths_cfg.get("courtlistener_output_file") or "data/processed/courtlistener_recap.jsonl"

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = fetch_clusters(
        max_results=max_cases,
        nature_of_suit_filter=nature_of_suit,
        include_opinion_text=True,
        include_docket_parties=True,
    )

    count = 0
    with open(out_path, "w") as f:
        for rec in rows:
            doc_text = (rec.get("opinion_text") or "").strip()
            out = {
                "case_id": rec.get("cluster_id") or rec.get("docket_id") or str(count),
                "cluster_id": rec.get("cluster_id"),
                "docket_id": rec.get("docket_id"),
                "docket_number": rec.get("docket_number") or "",
                "court_id": rec.get("court_id"),
                "court": rec.get("court"),
                "court_citation_string": rec.get("court_citation_string"),
                "case_name": rec.get("case_name"),
                "case_name_full": rec.get("case_name_full"),
                "date_filed": rec.get("date_filed"),
                "date_terminated": rec.get("date_terminated"),
                "date_argued": rec.get("date_argued"),
                "nature_of_suit": rec.get("nature_of_suit"),
                "cause": rec.get("cause"),
                "document_text": doc_text,
                "absolute_url": rec.get("absolute_url"),
                "parties": rec.get("parties", []),
                "party_id": rec.get("party_id", []),
                "attorneys": rec.get("attorneys", []),
                "attorney_id": rec.get("attorney_id", []),
                "firms": rec.get("firms", []),
                "firm_id": rec.get("firm_id", []),
                "assigned_to": rec.get("assigned_to"),
                "assigned_to_id": rec.get("assigned_to_id"),
                "referred_to": rec.get("referred_to"),
                "referred_to_id": rec.get("referred_to_id"),
                "jurisdiction_type": rec.get("jurisdiction_type"),
                "jury_demand": rec.get("jury_demand"),
                "pacer_case_id": rec.get("pacer_case_id"),
                "documents": rec.get("documents", []),
            }
            if rec.get("opinion_text"):
                out["opinion_text"] = rec.get("opinion_text")
            f.write(json.dumps(out, default=str) + "\n")
            count += 1

    print(f"Wrote {count} cases to {out_path}")
    return count


def main():
    p = argparse.ArgumentParser(
        description="CourtListener pipeline: cluster-first (only cases with court opinion). Enrich with docket metadata and parties by docket_id."
    )
    p.add_argument("--config", type=Path, default=None, help="Config YAML path")
    p.add_argument("--max-cases", type=int, default=None, help="Max records to write")
    p.add_argument("--nature-of-suit", default=None, help="Nature of suit filter (e.g. 442 for employment)")
    p.add_argument("--out", default=None, help="Output JSONL path")
    args = p.parse_args()
    run(
        max_cases=args.max_cases,
        nature_of_suit=args.nature_of_suit,
        out_path=args.out,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
