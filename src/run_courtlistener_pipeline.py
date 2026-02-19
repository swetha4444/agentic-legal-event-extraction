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
import logging
from pathlib import Path
from typing import Optional

import yaml

from src.data.courtlistener_client import (
    fetch_clusters,
    fetch_complaints_for_docket,
    fetch_docket,
    fetch_opinion_for_docket,
    fetch_parties_for_docket,
    search_recap_complaint_docket_ids,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


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


def _recap_first_record_to_output(
    docket: dict,
    opinion_text: str,
    cluster_id: Optional[int],
    complaint_text: str,
    complaints: list,
    parties: list,
    count: int,
) -> dict:
    """Build output dict from RECAP-first path (docket + opinion + complaints). Same shape as cluster output."""
    case_name = docket.get("case_name") or docket.get("case_name_short") or ""
    return {
        "case_id": cluster_id or docket.get("id") or count,
        "cluster_id": cluster_id,
        "docket_id": docket.get("id"),
        "docket_number": docket.get("docket_number") or "",
        "court_id": docket.get("court_id"),
        "court": docket.get("court"),
        "court_citation_string": None,
        "case_name": case_name,
        "case_name_full": docket.get("case_name_full") or "",
        "date_filed": docket.get("date_filed"),
        "date_terminated": docket.get("date_terminated"),
        "date_argued": docket.get("date_argued"),
        "nature_of_suit": docket.get("nature_of_suit") or "",
        "cause": docket.get("cause") or "",
        "document_text": opinion_text,
        "opinion_text": opinion_text,
        "complaint_text": complaint_text,
        "complaints": complaints,
        "absolute_url": docket.get("absolute_url"),
        "parties": parties,
        "party_id": [],
        "attorneys": [],
        "attorney_id": [],
        "firms": [],
        "firm_id": [],
        "assigned_to": docket.get("assigned_to_str"),
        "assigned_to_id": None,
        "referred_to": docket.get("referred_to_str"),
        "referred_to_id": None,
        "jurisdiction_type": docket.get("jurisdiction_type"),
        "jury_demand": docket.get("jury_demand"),
        "pacer_case_id": docket.get("pacer_case_id"),
        "documents": [],
    }


def _cluster_record_to_output(rec: dict, doc_text: str, complaint_text: str, complaints: list, count: int) -> dict:
    """Build output dict from a cluster API record (same shape as run() writes)."""
    return {
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
        "opinion_text": doc_text,
        "complaint_text": complaint_text,
        "complaints": complaints,
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

    logger.info("Starting pipeline: nature_of_suit=%s, out=%s", nature_of_suit, out_path)
    rows = fetch_clusters(
        max_results=max_cases,
        nature_of_suit_filter=nature_of_suit,
        include_opinion_text=True,
        include_docket_parties=True,
    )

    count = 0
    skipped = 0
    with open(out_path, "w") as f:
        for rec in rows:
            doc_text = (rec.get("opinion_text") or "").strip()
            complaint_text = (rec.get("complaint_text") or "").strip()
            complaints = rec.get("complaints") or []
            has_opinion = bool(doc_text)
            has_complaint = bool(complaint_text) or len(complaints) > 0
            if not (has_opinion and has_complaint):
                skipped += 1
                logger.info("Skip (no opinion or no complaint): %s (docket %s)", rec.get("case_name") or rec.get("cluster_id"), rec.get("docket_id"))
                continue
            logger.info("Case %s: %s (docket %s)", count + 1, rec.get("case_name") or rec.get("cluster_id"), rec.get("docket_id"))
            out = _cluster_record_to_output(rec, doc_text, complaint_text, complaints, count)
            f.write(json.dumps(out, default=str) + "\n")
            count += 1

    logger.info("Wrote %s cases to %s (skipped %s without both opinion and complaint)", count, out_path, skipped)
    return count


def run_recap_first(
    max_cases: Optional[int] = None,
    out_path: Optional[str] = None,
    config_path: Optional[Path] = None,
):
    """
    RECAP-first pipeline: search for dockets that have complaint documents, then fetch opinion per docket.
    Higher hit rate than cluster-first when many clusters lack RECAP complaints.
    """
    default_config = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    path_to_load = config_path or default_config
    cfg = _load_config_file(path_to_load)
    paths_cfg = cfg.get("paths", {})
    out_path = out_path or paths_cfg.get("courtlistener_output_file") or "data/processed/courtlistener_recap.jsonl"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    max_docket_ids = (max_cases or 400) * 4
    logger.info("RECAP-first: searching for docket_ids with complaints (max %s), then fetching opinion per docket; out=%s", max_docket_ids, out_path)
    count = 0
    skipped = 0
    written_docket_ids = set()
    with open(out_path, "w") as f:
        for docket_id in search_recap_complaint_docket_ids(max_docket_ids=max_docket_ids):
            if max_cases is not None and count >= max_cases:
                break
            if docket_id in written_docket_ids:
                continue
            docket = fetch_docket(docket_id)
            if not docket:
                skipped += 1
                continue
            opinion_text, cluster_id = fetch_opinion_for_docket(docket_id)
            opinion_text = (opinion_text or "").strip()
            complaints_list, complaint_text = fetch_complaints_for_docket(docket_id)
            complaint_text = (complaint_text or "").strip()
            has_opinion = bool(opinion_text)
            has_complaint = bool(complaint_text) or len(complaints_list) > 0
            if not (has_opinion and has_complaint):
                logger.info("Skip (no opinion or no complaint): %s (docket %s)", docket.get("case_name") or docket_id, docket_id)
                skipped += 1
                continue
            written_docket_ids.add(docket_id)
            parties = fetch_parties_for_docket(docket_id)
            out = _recap_first_record_to_output(
                docket, opinion_text, cluster_id, complaint_text, complaints_list, parties, count,
            )
            f.write(json.dumps(out, default=str) + "\n")
            count += 1
            logger.info("Case %s: %s (docket %s) -> %s complaint(s)", count, docket.get("case_name") or docket_id, docket_id, len(complaints_list))

    logger.info("Wrote %s cases to %s (skipped %s)", count, out_path, skipped)
    return count


def run_from_file(
    from_path: Path,
    out_path: Optional[str] = None,
    config_path: Optional[Path] = None,
    only_both: bool = True,
    max_records: Optional[int] = None,
    max_cases: Optional[int] = None,
):
    """
    Read existing JSONL (e.g. courtlistener_recap.jsonl), take docket_id from each record,
    search CourtListener for complaints by docket_id, enrich and write.
    If only_both=True, write only records that have both opinion and complaint.
    If max_cases=N, stop after writing N such records (keeps searching input until N non-empty).
    """
    default_config = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    path_to_load = config_path or default_config
    cfg = _load_config_file(path_to_load)
    paths_cfg = cfg.get("paths", {})
    out_path = out_path or paths_cfg.get("courtlistener_output_file") or "data/processed/courtlistener_recap.jsonl"
    out_path = Path(out_path)
    cl_cfg = cfg.get("courtlistener", {})
    from_path = Path(from_path)
    if not from_path.exists():
        raise FileNotFoundError(f"Input file not found: {from_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Enriching from file: %s -> %s (only_both=%s, max_cases=%s)", from_path, out_path, only_both, max_cases)
    count = 0
    skipped = 0
    seen = 0
    written_docket_ids = set()
    with open(from_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            if max_cases is not None and count >= max_cases:
                break
            seen += 1
            if max_records is not None and seen > max_records:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("Skip invalid JSON at line %s: %s", seen, e)
                skipped += 1
                continue
            docket_id = rec.get("docket_id")
            doc_text = (rec.get("document_text") or rec.get("opinion_text") or "").strip()
            if docket_id is None:
                logger.info("Skip (no docket_id): %s", rec.get("case_name") or rec.get("case_id"))
                skipped += 1
                continue
            if docket_id in written_docket_ids:
                logger.info("Skip (duplicate docket): %s (docket %s)", rec.get("case_name") or rec.get("case_id"), docket_id)
                skipped += 1
                continue
            complaints_list, complaint_text = fetch_complaints_for_docket(docket_id)
            rec["complaint_text"] = complaint_text
            rec["complaints"] = complaints_list
            if not rec.get("opinion_text") and doc_text:
                rec["opinion_text"] = doc_text
            has_opinion = bool(doc_text)
            has_complaint = bool(complaint_text) or len(complaints_list) > 0
            if only_both and not (has_opinion and has_complaint):
                logger.info("Skip (no opinion or no complaint): %s (docket %s)", rec.get("case_name") or rec.get("case_id"), docket_id)
                skipped += 1
                continue
            written_docket_ids.add(docket_id)
            logger.info("Case %s: %s (docket %s) -> %s complaint(s)", count + 1, rec.get("case_name") or rec.get("case_id"), docket_id, len(complaints_list))
            fout.write(json.dumps(rec, default=str) + "\n")
            count += 1
            if max_cases is not None and count >= max_cases:
                break

        # If we need more and have a limit, fetch from cluster API until we reach max_cases
        if max_cases is not None and count < max_cases:
            remaining = max_cases - count
            count_before_cluster = count
            nature_of_suit = cl_cfg.get("nature_of_suit") or "442"
            logger.info("Input exhausted with %s cases; fetching up to %s more from cluster API (nature_of_suit=%s)", count, remaining, nature_of_suit)
            buffer = min(500, remaining * 2)
            rows = fetch_clusters(
                max_results=remaining + buffer,
                nature_of_suit_filter=nature_of_suit,
                include_opinion_text=True,
                include_docket_parties=True,
            )
            for rec in rows:
                if count >= max_cases:
                    break
                docket_id = rec.get("docket_id")
                if docket_id is not None and docket_id in written_docket_ids:
                    continue
                doc_text = (rec.get("opinion_text") or "").strip()
                complaint_text = (rec.get("complaint_text") or "").strip()
                complaints = rec.get("complaints") or []
                if not doc_text or (not complaint_text and not complaints):
                    continue
                written_docket_ids.add(docket_id)
                out = _cluster_record_to_output(rec, doc_text, complaint_text, complaints, count)
                fout.write(json.dumps(out, default=str) + "\n")
                count += 1
                logger.info("Case %s (cluster): %s (docket %s)", count, rec.get("case_name") or rec.get("cluster_id"), docket_id)
            logger.info("Added %s cases from cluster API (%s total)", count - count_before_cluster, count)
    logger.info("Wrote %s cases to %s (skipped %s)", count, out_path, skipped)
    return count


def main():
    p = argparse.ArgumentParser(
        description="CourtListener pipeline: cluster-first (only cases with court opinion). Enrich with docket metadata and parties by docket_id."
    )
    p.add_argument("--config", type=Path, default=None, help="Config YAML path")
    p.add_argument("--max-cases", type=int, default=None, help="Max records to write (cluster mode). With --from-file: stop after this many records with both opinion and complaint")
    p.add_argument("--nature-of-suit", default=None, help="Nature of suit filter (e.g. 442 for employment)")
    p.add_argument("--out", default=None, help="Output JSONL path")
    p.add_argument("--from-file", type=Path, default=None, help="Read existing JSONL and enrich with complaints by docket_id (e.g. data/processed/courtlistener_recap.jsonl)")
    p.add_argument("--no-filter", action="store_true", help="With --from-file: write all enriched records; default is only records with both opinion and complaint")
    p.add_argument("--max-records", type=int, default=None, help="With --from-file: max input records to process")
    args = p.parse_args()
    if args.from_file is not None:
        run_from_file(
            from_path=args.from_file,
            out_path=args.out,
            config_path=args.config,
            only_both=not args.no_filter,
            max_records=args.max_records,
            max_cases=args.max_cases,
        )
    elif not args.recap_first:
        run(
            max_cases=args.max_cases,
            nature_of_suit=args.nature_of_suit,
            out_path=args.out,
            config_path=args.config,
        )


if __name__ == "__main__":
    main()
