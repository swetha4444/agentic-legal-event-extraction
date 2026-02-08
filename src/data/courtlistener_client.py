"""
CourtListener API client: clusters (case law) and RECAP search.
Clusters API: GET /api/rest/v4/clusters/ — works with standard token (Authorization: Token <token>).
RECAP Search: GET /api/rest/v4/search/?type=r — may require subscription.
"""
import os
import re
from pathlib import Path
from typing import Any, Iterator, Optional

import requests
import yaml

CLUSTERS_URL = "https://www.courtlistener.com/api/rest/v4/clusters/"
SEARCH_URL = "https://www.courtlistener.com/api/rest/v4/search/"

# Court IDs for U.S. Circuit Courts (appellate) — exclude when district_only=True
APPELLATE_COURT_PREFIXES = ("ca", "uscfc", "usca", "arb", "neb", "nysb")


def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def _is_district_court(court_id: Optional[str]) -> bool:
    """Return True if court_id looks like a district (not appellate/bankruptcy) court."""
    if not court_id or not isinstance(court_id, str):
        return False
    c = court_id.lower().strip()
    if not c:
        return False
    return not any(c.startswith(p) for p in APPELLATE_COURT_PREFIXES)


def _get_token(api_token: Optional[str] = None) -> str:
    """Resolve CourtListener API token from arg, env, or config (including token_file)."""
    config = _load_config()
    cl_cfg = config.get("courtlistener", {})
    token = (
        api_token
        or os.environ.get("COURTLISTENER_API_TOKEN")
        or cl_cfg.get("api_token")
    )
    if not token and cl_cfg.get("token_file"):
        token_path = Path(cl_cfg["token_file"])
        if not token_path.is_absolute():
            token_path = Path(__file__).resolve().parents[2] / token_path
        if token_path.exists():
            token = token_path.read_text().strip()
    if not token:
        raise ValueError(
            "CourtListener API token required. Either:\n"
            "  1. export COURTLISTENER_API_TOKEN=your_token\n"
            "  2. Set courtlistener.api_token or courtlistener.token_file in config/config.yaml\n"
            "Get a token: https://www.courtlistener.com/help/api/ (sign in, REST API access)."
        )
    return token.strip()


def _fetch_opinion_text(opinion_url: str, headers: dict) -> str:
    """GET one opinion by URL; return plain text (plain_text, or strip html_with_citations)."""
    if not opinion_url or not opinion_url.strip():
        return ""
    url = opinion_url.strip()
    if url.startswith("/"):
        url = "https://www.courtlistener.com" + url
    # Request only text fields to reduce payload (CourtListener recommends field selection)
    if "?" in url:
        url = f"{url}&fields=plain_text,html_with_citations"
    else:
        url = f"{url}?fields=plain_text,html_with_citations"
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        text = (data.get("plain_text") or "").strip()
        if text:
            return text
        html = (data.get("html_with_citations") or data.get("html") or "").strip()
        if not html:
            return ""
        # Strip tags for plain text
        return re.sub(r"<[^>]+>", " ", html).replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"').strip()
    except Exception:
        return ""


def fetch_clusters(
    api_token: Optional[str] = None,
    base_url: str = CLUSTERS_URL,
    max_results: Optional[int] = None,
    nature_of_suit_filter: Optional[str] = None,
    include_opinion_text: bool = False,
) -> Iterator[dict]:
    """
    Fetch case law (opinion clusters) from CourtListener clusters API.
    Uses same auth as: curl "https://www.courtlistener.com/api/rest/v4/clusters/" --header "Authorization: Token <token>"
    Yields one dict per cluster with: cluster_id, docket_id, case_name, case_name_full, date_filed, nature_of_suit, absolute_url, etc.
    If include_opinion_text=True, fetches the first sub_opinion per cluster and adds opinion_text (plain text of the decision).
    """
    token = _get_token(api_token)
    headers = {"Authorization": f"Token {token}"}
    params: dict[str, Any] = {}
    count = 0
    url: Optional[str] = base_url

    while url:
        resp = requests.get(url, params=params if url == base_url else None, headers=headers, timeout=60)
        if resp.status_code == 401:
            msg = (
                "CourtListener API returned 401 Unauthorized. "
                "Set the token in this shell: export COURTLISTENER_API_TOKEN=your_token "
                "Or in config: courtlistener.token_file (e.g. .courtlistener_token) or courtlistener.api_token."
            )
            try:
                body = resp.json()
                if body:
                    msg += f" Response: {body}"
            except Exception:
                pass
            raise requests.HTTPError(msg, response=resp)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results") or []
        for item in results:
            rec = _normalize_cluster_result(item)
            if nature_of_suit_filter and rec.get("nature_of_suit"):
                nos = (rec.get("nature_of_suit") or "").lower()
                if nature_of_suit_filter.lower() not in nos:
                    continue
            if include_opinion_text:
                sub = item.get("sub_opinions") or []
                opinion_url = sub[0] if sub else None
                rec["opinion_text"] = _fetch_opinion_text(opinion_url, headers) if opinion_url else ""
            yield rec
            count += 1
            if max_results is not None and count >= max_results:
                return
        url = data.get("next")
        if url:
            params = None
    return


def _normalize_cluster_result(item: dict) -> dict:
    """Map clusters API result to a flat record for JSONL (same shape as RECAP where possible)."""
    return {
        "cluster_id": item.get("id"),
        "docket_id": item.get("docket_id"),
        "case_name": item.get("case_name") or item.get("case_name_short") or "",
        "case_name_full": item.get("case_name_full") or "",
        "date_filed": item.get("date_filed"),
        "nature_of_suit": item.get("nature_of_suit") or "",
        "absolute_url": (item.get("absolute_url") or "").strip() or None,
        "slug": item.get("slug") or "",
        "docket_number": "",  # clusters don't include it; would need docket fetch
        "court_id": None,
        "date_terminated": None,
        "cause": "",
        "documents": [],
    }


def search_recap(
    api_token: Optional[str] = None,
    base_url: str = SEARCH_URL,
    q: Optional[str] = None,
    nature_of_suit: Optional[str] = None,
    filed_before: Optional[str] = None,
    filed_after: Optional[str] = None,
    court_id: Optional[str] = None,
    district_only: bool = True,
    max_results: Optional[int] = None,
    description_contains: Optional[str] = None,
) -> Iterator[dict]:
    """
    Search RECAP (PACER) dockets via CourtListener Search API (type=r).
    Yields one dict per docket with: docket_id, case_name, court_id, date_filed,
    nature_of_suit, docket_number, documents (list of doc info), etc.

    - api_token: CourtListener API token (or set COURTLISTENER_API_TOKEN).
    - q: Free-text query (e.g. "employment").
    - nature_of_suit: Filter by NOS code or phrase (e.g. "442" or "Civil Rights: Jobs").
    - filed_before / filed_after: ISO date string (YYYY-MM-DD).
    - court_id: Restrict to one court (e.g. "dcd").
    - district_only: If True, skip results whose court_id is appellate/bankruptcy.
    - max_results: Stop after yielding this many dockets.
    - description_contains: Add to q to filter by document description (e.g. "complaint").
    """
    token = _get_token(api_token)
    params: dict[str, Any] = {"type": "r"}
    if q:
        params["q"] = q
    if nature_of_suit is not None:
        params["nature_of_suit"] = nature_of_suit
    if filed_before is not None:
        params["filed_before"] = filed_before
    if filed_after is not None:
        params["filed_after"] = filed_after
    if court_id is not None:
        params["court"] = court_id
    if description_contains:
        # Document description filter (RECAP field "description")
        desc_q = f'description:("{description_contains}")'
        params["q"] = (params.get("q") or "") + (" " + desc_q if params.get("q") else desc_q)

    headers = {"Authorization": f"Token {token}"}
    count = 0
    url: Optional[str] = base_url

    while url:
        resp = requests.get(url, params=params if url == base_url else None, headers=headers, timeout=60)
        if resp.status_code == 401:
            try:
                body = resp.json()
            except Exception:
                body = resp.text or ""
            msg = (
                "CourtListener API returned 401 Unauthorized. Check that your token is correct and not expired. "
                "RECAP Search API access may require a Free Law Project membership or API subscription."
            )
            if body:
                msg += f" Response: {body}"
            raise requests.HTTPError(msg, response=resp)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results") or []
        for item in results:
            cid = item.get("court_id") or item.get("court")
            if district_only and not _is_district_court(cid):
                continue
            yield _normalize_recap_result(item)
            count += 1
            if max_results is not None and count >= max_results:
                return
        url = data.get("next")
        if url:
            params = None  # next is full URL with cursor; pass no params on next request

    return


def _normalize_recap_result(item: dict) -> dict:
    """Map Search API RECAP result to a flat-ish record for JSONL."""
    docs = item.get("recap_documents") or item.get("recapDocuments") or []
    doc_list = []
    for d in docs if isinstance(docs, list) else []:
        if isinstance(d, dict):
            doc_list.append({
                "description": d.get("description") or d.get("short_description") or "",
                "snippet": (d.get("snippet") or "")[:2000] if d.get("snippet") else "",
                "document_number": d.get("document_number"),
                "entry_number": d.get("entry_number"),
                "id": d.get("id"),
            })
        else:
            doc_list.append({"description": str(d), "snippet": ""})

    return {
        "docket_id": item.get("docket_id"),
        "case_name": item.get("caseName") or item.get("case_name") or "",
        "case_name_full": item.get("caseNameFull") or item.get("case_name_full") or "",
        "court_id": item.get("court_id") or item.get("court"),
        "court": item.get("court"),  # sometimes full name
        "docket_number": item.get("docketNumber") or item.get("docket_number") or "",
        "date_filed": item.get("dateFiled") or item.get("date_filed"),
        "date_terminated": item.get("dateTerminated") or item.get("date_terminated"),
        "nature_of_suit": item.get("suitNature") or item.get("nature_of_suit") or "",
        "cause": item.get("cause") or "",
        "documents": doc_list,
        "absolute_url": item.get("absolute_url") or "",
    }
