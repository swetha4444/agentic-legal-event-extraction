"""
Data pipeline from CourtListener. Uses clusters API by default (case law + opinion text; same auth as
curl .../api/rest/v4/clusters/ --header "Authorization: Token <token>"). Optional: RECAP search.

Usage (from project root):
  export COURTLISTENER_API_TOKEN=your_token
  python -m src.run_courtlistener_pipeline [--config FILE] [--max-cases N] [--out FILE]
  python -m src.run_courtlistener_pipeline --source recap ...   # RECAP search (may require subscription)
"""
import argparse
import json
from pathlib import Path
from typing import Optional

import yaml

from src.data.courtlistener_client import fetch_clusters, search_recap


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
    q: Optional[str] = None,
    filed_before: Optional[str] = None,
    filed_after: Optional[str] = None,
    district_only: bool = True,
    description_contains: Optional[str] = None,
    out_path: Optional[str] = None,
    config_path: Optional[Path] = None,
    source: str = "clusters",
    include_opinion_text: Optional[bool] = None,
):
    default_config = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    path_to_load = config_path or default_config
    cfg = _load_config_file(path_to_load)
    cl_cfg = cfg.get("courtlistener", {})
    paths_cfg = cfg.get("paths", {})

    source = (source or cl_cfg.get("source") or "clusters").lower()
    nature_of_suit = nature_of_suit or cl_cfg.get("nature_of_suit") or "442"
    q = q or cl_cfg.get("query") or "employment"
    filed_before = filed_before or cl_cfg.get("filed_before")
    filed_after = filed_after or cl_cfg.get("filed_after")
    district_only = district_only if district_only is not None else cl_cfg.get("district_only", True)
    description_contains = description_contains or cl_cfg.get("description_contains")
    include_opinion = include_opinion_text if include_opinion_text is not None else cl_cfg.get("include_opinion_text", True)
    out_path = out_path or paths_cfg.get("courtlistener_output_file") or "data/processed/courtlistener_recap.jsonl"

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if source == "recap":
        rows = search_recap(
            q=q,
            nature_of_suit=nature_of_suit,
            filed_before=filed_before,
            filed_after=filed_after,
            district_only=district_only,
            max_results=max_cases,
            description_contains=description_contains,
        )
    else:
        # clusters API (default): same auth as curl .../api/rest/v4/clusters/; optionally fetch opinion text per cluster
        rows = fetch_clusters(
            max_results=max_cases,
            nature_of_suit_filter=nature_of_suit if source == "clusters" else None,
            include_opinion_text=include_opinion,
        )

    count = 0
    with open(out_path, "w") as f:
        for rec in rows:
            if source == "recap":
                doc_text = _pick_document_text(rec, prefer_complaint=True)
            else:
                doc_text = rec.get("opinion_text") or ""  # clusters: we fetched it
            out = {
                "case_id": rec.get("cluster_id") or rec.get("docket_id") or str(count),
                "cluster_id": rec.get("cluster_id"),
                "docket_id": rec.get("docket_id"),
                "docket_number": rec.get("docket_number") or "",
                "court_id": rec.get("court_id"),
                "case_name": rec.get("case_name"),
                "case_name_full": rec.get("case_name_full"),
                "date_filed": rec.get("date_filed"),
                "date_terminated": rec.get("date_terminated"),
                "nature_of_suit": rec.get("nature_of_suit"),
                "cause": rec.get("cause"),
                "document_text": doc_text,
                "absolute_url": rec.get("absolute_url"),
            }
            f.write(json.dumps(out, default=str) + "\n")
            count += 1

    print(f"Wrote {count} cases to {out_path}")
    return count


def main():
    p = argparse.ArgumentParser(
        description="CourtListener pipeline: clusters API (default) or RECAP search; write to JSONL."
    )
    p.add_argument("--config", type=Path, default=None, help="Config YAML path")
    p.add_argument("--source", default="clusters", choices=("clusters", "recap"),
                   help="clusters = case law API (default; same auth as curl .../clusters/). recap = RECAP search (may need subscription)")
    p.add_argument("--max-cases", type=int, default=None, help="Max records to write")
    p.add_argument("--nature-of-suit", default=None, help="Nature of suit filter (e.g. 442 for employment)")
    p.add_argument("--query", "-q", default=None, help="Search query for RECAP (e.g. employment)")
    p.add_argument("--filed-before", default=None, help="Filed before date (YYYY-MM-DD); RECAP only")
    p.add_argument("--filed-after", default=None, help="Filed after date (YYYY-MM-DD); RECAP only")
    p.add_argument("--no-district-only", action="store_true", help="Include appellate/bankruptcy (RECAP only)")
    p.add_argument("--description-contains", default=None, help="Document description filter (RECAP only)")
    p.add_argument("--no-opinion-text", action="store_true", help="Clusters only: skip fetching opinion text (faster)")
    p.add_argument("--out", default=None, help="Output JSONL path")
    args = p.parse_args()
    district_only = False if args.no_district_only else None
    run(
        max_cases=args.max_cases,
        nature_of_suit=args.nature_of_suit,
        q=args.query,
        filed_before=args.filed_before,
        filed_after=args.filed_after,
        district_only=district_only,
        description_contains=args.description_contains,
        out_path=args.out,
        config_path=args.config,
        source=args.source,
        include_opinion_text=False if args.no_opinion_text else None,
    )


if __name__ == "__main__":
    main()
