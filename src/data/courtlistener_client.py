"""
CourtListener API client: clusters (case law) and RECAP search.
Clusters API: GET /api/rest/v4/clusters/ — works with standard token (Authorization: Token <token>).
RECAP Search: GET /api/rest/v4/search/?type=r — may require subscription.
"""
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Iterator, Optional

import requests
import yaml

logger = logging.getLogger(__name__)

# Timeouts (seconds). Cluster/list can be slow; use longer timeout + retries.
REQUEST_TIMEOUT_CLUSTERS = 120
REQUEST_TIMEOUT_DEFAULT = 45
REQUEST_RETRIES = 3
REQUEST_RETRY_BACKOFF = 10

CLUSTERS_URL = "https://www.courtlistener.com/api/rest/v4/clusters/"
DOCKETS_URL = "https://www.courtlistener.com/api/rest/v4/dockets/"
DOCKET_ENTRIES_URL = "https://www.courtlistener.com/api/rest/v4/docket-entries/"
PARTIES_URL = "https://www.courtlistener.com/api/rest/v4/parties/"
SEARCH_URL = "https://www.courtlistener.com/api/rest/v4/search/"
RECAP_DOCUMENTS_URL = "https://www.courtlistener.com/api/rest/v4/recap-documents/"

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


def _get_with_retries(
    url: str,
    headers: dict,
    timeout: int = REQUEST_TIMEOUT_DEFAULT,
    params: Optional[dict] = None,
    retries: int = REQUEST_RETRIES,
    backoff: int = REQUEST_RETRY_BACKOFF,
) -> requests.Response:
    """GET with retries on ReadTimeout and ConnectionError."""
    last_exc = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout, params=params)
            return resp
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError) as e:
            last_exc = e
            if attempt < retries - 1:
                wait = backoff * (attempt + 1)
                logger.warning("Request timeout/error (attempt %s/%s), retrying in %ss: %s", attempt + 1, retries, wait, e)
                time.sleep(wait)
    raise last_exc


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


def fetch_opinion_for_docket(
    docket_id: Any,
    api_token: Optional[str] = None,
) -> tuple[str, Optional[int]]:
    """
    Get court opinion text for a docket by finding its cluster and fetching the first opinion.

    CourtListener uses one docket table: RECAP search returns docket_id from that table, and
    clusters are joined to the same docket. So filtering clusters by docket=<docket_id> returns
    the opinion cluster(s) for the same case. We take the first cluster and its first sub_opinion
    (if multiple opinions exist, e.g. majority + dissent, we do not currently prefer lead opinion).

    Returns (opinion_text, cluster_id). Use for hybrid: parties from RECAP, opinion from cluster.
    """
    if docket_id is None:
        return "", None
    token = _get_token(api_token)
    headers = {"Authorization": f"Token {token}"}
    try:
        resp = requests.get(
            CLUSTERS_URL,
            params={"docket": docket_id},
            headers=headers,
            timeout=30,
        )
        if resp.status_code != 200:
            return "", None
        data = resp.json()
        results = data.get("results") or []
        if not results:
            return "", None
        item = results[0]
        # Defensive: ensure returned cluster is for this docket (same case)
        cluster_docket_id = item.get("docket_id")
        if cluster_docket_id is None and item.get("docket"):
            # API may return docket as URL, e.g. .../dockets/5354075/
            docket_url = str(item.get("docket", ""))
            if "/dockets/" in docket_url:
                try:
                    cluster_docket_id = int(docket_url.strip("/").rsplit("/", 1)[-1])
                except (ValueError, IndexError):
                    pass
        if cluster_docket_id is not None and int(cluster_docket_id) != int(docket_id):
            return "", None
        cluster_id = item.get("id")
        sub = item.get("sub_opinions") or []
        opinion_url = sub[0] if sub else None
        text = _fetch_opinion_text(opinion_url, headers) if opinion_url else ""
        return text, cluster_id
    except Exception:
        return "", None


def _fetch_docket(docket_id: Any, headers: dict) -> Optional[dict]:
    """GET one docket by ID; return parsed JSON or None. Docket does not include parties list."""
    if docket_id is None:
        return None
    try:
        url = f"{DOCKETS_URL.rstrip('/')}/{int(docket_id)}/"
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:
        return None


def _fetch_parties_for_docket(docket_id: Any, headers: dict) -> list:
    """GET parties for a docket; return list of party names (and optionally roles). Empty if 403/no access."""
    if docket_id is None:
        return []
    try:
        resp = requests.get(
            PARTIES_URL,
            params={"docket": docket_id, "filter_nested_results": "true"},
            headers=headers,
            timeout=15,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        results = data.get("results") or []
        names = []
        for p in results:
            name = (p.get("name") or "").strip()
            if name:
                names.append(name)
        return names
    except Exception:
        return []


def _fetch_recap_document_plain_text(doc_id: Any, headers: dict) -> str:
    """GET full plain_text for a RECAP document by ID. Returns '' if 403/no access."""
    if doc_id is None:
        return ""
    try:
        url = f"{RECAP_DOCUMENTS_URL.rstrip('/')}/{int(doc_id)}/"
        resp = requests.get(url, params={"fields": "plain_text"}, headers=headers, timeout=20)
        if resp.status_code != 200:
            return ""
        data = resp.json()
        return (data.get("plain_text") or "").strip()
    except Exception:
        return ""


def _fetch_complaint_text_via_docket_entries(docket_id: Any, headers: dict, max_docs: int = 10) -> tuple[list, str]:
    """
    Fetch complaint text via Docket Entries API: GET docket-entries/?docket=X.
    Each entry can include nested recap_documents with full plain_text (per CourtListener PACER docs).
    Returns (list of complaint doc dicts, concatenated plain_text). Returns ([], "") if API unavailable or no complaints.
    """
    if docket_id is None:
        return [], ""
    out: list = []
    parts: list = []
    url: Optional[str] = f"{DOCKET_ENTRIES_URL}?docket={docket_id}"
    try:
        while url and len(out) < max_docs:
            try:
                resp = _get_with_retries(url, headers, timeout=REQUEST_TIMEOUT_DEFAULT, params=None)
            except Exception:
                break
            if resp.status_code != 200:
                break
            data = resp.json()
            for entry in data.get("results", []):
                desc = (entry.get("description") or "").lower()
                short = (entry.get("short_description") or "").lower()
                if "complaint" not in desc and "complaint" not in short:
                    continue
                docs = entry.get("recap_documents") or entry.get("recapDocuments") or []
                if not docs and entry.get("recap_document"):
                    docs = [entry["recap_document"]]
                for doc in docs:
                    if len(out) >= max_docs:
                        break
                    if not isinstance(doc, dict):
                        continue
                    plain = (doc.get("plain_text") or "").strip()
                    if plain:
                        parts.append(plain)
                    out.append({
                        "id": doc.get("id"),
                        "description": entry.get("description") or doc.get("description") or "",
                        "short_description": entry.get("short_description") or doc.get("short_description") or "",
                        "snippet": (plain or doc.get("snippet") or "")[:10000],
                        "entry_number": entry.get("entry_number") or doc.get("entry_number"),
                        "entry_date_filed": entry.get("entry_date_filed") or doc.get("entry_date_filed"),
                        "document_number": doc.get("document_number") or entry.get("document_number"),
                        "absolute_url": doc.get("absolute_url") or "",
                        "is_available": doc.get("is_available"),
                    })
            url = data.get("next")
    except Exception:
        pass
    complaint_text = "\n\n---\n\n".join(parts) if parts else ""
    return out, complaint_text.strip()


def _fetch_complaints_for_docket(docket_id: Any, headers: dict, max_docs: int = 10) -> tuple[list, str]:
    """
    Fetch complaint-like RECAP documents for a docket.
    Tries Docket Entries API first (nested recap_documents with plain_text); falls back to Search API (type=rd).
    Returns (list of complaint doc dicts, best available full complaint text).
    """
    if docket_id is None:
        return [], ""
    # Prefer Docket Entries API when available (includes full plain_text in nested recap_documents)
    out, complaint_text = _fetch_complaint_text_via_docket_entries(docket_id, headers, max_docs=max_docs)
    if complaint_text and out:
        return out, complaint_text
    out = []
    try:
        q = (
            f'docket_id:{docket_id} AND document_type:"PACER Document" '
            f'AND (description:complaint OR short_description:complaint)'
        )
        params: dict[str, Any] = {"type": "rd", "q": q}
        # Request snippet (and plain_text if returned) so we have fallback when recap-documents 403
        params["fields"] = "snippet,plain_text,id,description,short_description,entry_number,entry_date_filed,document_number,absolute_url,is_available"
        url = SEARCH_URL
        while url and len(out) < max_docs:
            try:
                resp = _get_with_retries(url, headers, timeout=REQUEST_TIMEOUT_DEFAULT, params=params if url == SEARCH_URL else None)
            except Exception:
                break
            if resp.status_code != 200:
                break
            data = resp.json()
            for d in data.get("results", []):
                # Search may return snippet or plain_text; keep both for building complaint_text
                raw_snippet = d.get("snippet") or ""
                raw_plain = (d.get("plain_text") or "")[:500000]
                snippet = (raw_snippet or raw_plain or "")[:10000]
                out.append({
                    "id": d.get("id"),
                    "description": d.get("description") or "",
                    "short_description": d.get("short_description") or "",
                    "snippet": snippet,
                    "plain_text_from_search": raw_plain[:10000] if raw_plain else "",
                    "entry_number": d.get("entry_number"),
                    "entry_date_filed": d.get("entry_date_filed"),
                    "document_number": d.get("document_number"),
                    "absolute_url": d.get("absolute_url") or "",
                    "is_available": d.get("is_available"),
                })
                if len(out) >= max_docs:
                    break
            url = data.get("next")
            params = {}
        out.sort(key=lambda x: (0 if x.get("entry_number") == 1 else 1, x.get("entry_number") or 999, x.get("entry_date_filed") or "9999"))
    except Exception:
        pass

    complaint_text = ""
    if out:
        # Try full plain_text from recap-documents API (up to 5 docs to limit requests); else use search snippet
        max_plain_fetches = 5
        parts = []
        for i, doc in enumerate(out):
            doc_id = doc.get("id")
            full = ""
            if doc_id and i < max_plain_fetches:
                full = _fetch_recap_document_plain_text(doc_id, headers)
            if full:
                parts.append(full)
            else:
                snippet = (doc.get("plain_text_from_search") or doc.get("snippet") or "").strip()
                if snippet:
                    parts.append(snippet)
        if parts:
            complaint_text = "\n\n---\n\n".join(parts)
        if not complaint_text and out:
            logger.debug("Complaints found for docket %s but no text (recap-documents may require subscription)", docket_id)
    # Return out without internal key for API consumers
    for doc in out:
        doc.pop("plain_text_from_search", None)
    return out, complaint_text.strip()


def fetch_complaints_for_docket(
    docket_id: Any,
    api_token: Optional[str] = None,
    max_docs: int = 10,
) -> tuple[list, str]:
    """
    Fetch complaint RECAP documents for a docket by ID (Search API type=rd).
    Returns (list of complaint doc dicts, full complaint text).
    Use this to enrich existing records that have docket_id (e.g. from courtlistener_recap.jsonl).
    """
    token = _get_token(api_token)
    headers = {"Authorization": f"Token {token}"}
    return _fetch_complaints_for_docket(docket_id, headers, max_docs=max_docs)


def _docket_id_from_cluster(item: dict) -> Optional[int]:
    """Get numeric docket_id from cluster item (may be in docket_id or in docket URL)."""
    did = item.get("docket_id")
    if did is not None:
        try:
            return int(did)
        except (TypeError, ValueError):
            pass
    url = item.get("docket")
    if url and "/dockets/" in str(url):
        try:
            return int(str(url).strip("/").rsplit("/", 1)[-1])
        except (ValueError, IndexError):
            pass
    return None


def fetch_clusters(
    api_token: Optional[str] = None,
    base_url: str = CLUSTERS_URL,
    max_results: Optional[int] = None,
    nature_of_suit_filter: Optional[str] = None,
    include_opinion_text: bool = False,
    include_docket_parties: bool = False,
) -> Iterator[dict]:
    """
    Fetch case law (opinion clusters) from CourtListener clusters API.
    Yields one dict per cluster with: cluster_id, docket_id, case_name, date_filed, nature_of_suit, etc.
    If include_opinion_text=True, fetches the first sub_opinion and adds opinion_text.
    If include_docket_parties=True, for each cluster fetches the docket (metadata) and parties by docket_id and merges into the record (parties from API override caption-derived parties).
    """
    token = _get_token(api_token)
    headers = {"Authorization": f"Token {token}"}
    params: dict[str, Any] = {}
    count = 0
    url: Optional[str] = base_url

    while url:
        resp = _get_with_retries(
            url,
            headers,
            timeout=REQUEST_TIMEOUT_CLUSTERS,
            params=params if url == base_url else None,
        )
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
        if results and not params:
            logger.info("Fetching cluster page (batch of %s)", len(results))
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
            if include_docket_parties:
                did = _docket_id_from_cluster(item) or rec.get("docket_id")
                if did is not None:
                    docket_meta = _fetch_docket(did, headers)
                    if docket_meta:
                        rec["docket_number"] = docket_meta.get("docket_number") or rec.get("docket_number") or ""
                        rec["court_id"] = docket_meta.get("court_id") or rec.get("court_id")
                        rec["date_terminated"] = docket_meta.get("date_terminated") or rec.get("date_terminated")
                        rec["date_filed"] = docket_meta.get("date_filed") or rec.get("date_filed")
                        rec["cause"] = docket_meta.get("cause") or rec.get("cause") or ""
                        rec["assigned_to"] = docket_meta.get("assigned_to_str") or rec.get("assigned_to") or ""
                        rec["referred_to"] = docket_meta.get("referred_to_str") or rec.get("referred_to") or ""
                    api_parties = _fetch_parties_for_docket(did, headers)
                    if api_parties:
                        rec["parties"] = api_parties
                    complaints_list, complaint_text = _fetch_complaints_for_docket(did, headers)
                    rec["complaints"] = complaints_list
                    rec["complaint_text"] = complaint_text
                    if complaints_list:
                        logger.info("  docket %s: %s complaint(s), opinion %s chars", did, len(complaints_list), len(rec.get("opinion_text") or ""))
                if "assigned_to" not in rec:
                    rec["assigned_to"] = ""
                if "referred_to" not in rec:
                    rec["referred_to"] = ""
                if "complaints" not in rec:
                    rec["complaints"] = []
                if "complaint_text" not in rec:
                    rec["complaint_text"] = ""
            yield rec
            count += 1
            if max_results is not None and count >= max_results:
                return
        url = data.get("next")
        if url:
            params = None
    return


def _parties_from_caption(case_name: str, case_name_full: str) -> list:
    """Extract party-like names from case caption (e.g. 'Plaintiff v. Defendant')."""
    text = (case_name_full or case_name or "").strip()
    if not text:
        return []
    for sep in (" v. ", " v ", " V. ", " V "):
        if sep in text:
            parts = [p.strip() for p in text.split(sep, 1)]
            if len(parts) == 2 and parts[0] and parts[1]:
                return parts
    return [text] if text else []


def _normalize_cluster_result(item: dict) -> dict:
    """Map clusters API result to a flat record for JSONL (same shape as RECAP where possible)."""
    case_name = item.get("case_name") or item.get("case_name_short") or ""
    case_name_full = item.get("case_name_full") or ""
    parties = _parties_from_caption(case_name, case_name_full)
    return {
        "cluster_id": item.get("id"),
        "docket_id": item.get("docket_id"),
        "case_name": case_name,
        "case_name_full": case_name_full,
        "date_filed": item.get("date_filed"),
        "nature_of_suit": item.get("nature_of_suit") or "",
        "absolute_url": (item.get("absolute_url") or "").strip() or None,
        "slug": item.get("slug") or "",
        "docket_number": "",
        "court_id": None,
        "date_terminated": None,
        "cause": "",
        "documents": [],
        "parties": parties,
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
        resp = _get_with_retries(
            url,
            headers,
            timeout=REQUEST_TIMEOUT_CLUSTERS,
            params=params if url == base_url else None,
        )
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
    """Map Search API RECAP result to a flat record for JSONL; save all useful RECAP fields."""
    docs = item.get("recap_documents") or item.get("recapDocuments") or []
    doc_list = []
    for d in docs if isinstance(docs, list) else []:
        if isinstance(d, dict):
            doc_list.append({
                "description": d.get("description") or d.get("short_description") or "",
                "short_description": d.get("short_description") or "",
                "snippet": (d.get("snippet") or "")[:2000] if d.get("snippet") else "",
                "document_number": d.get("document_number"),
                "entry_number": d.get("entry_number"),
                "entry_date_filed": d.get("entry_date_filed"),
                "id": d.get("id"),
                "absolute_url": d.get("absolute_url") or "",
                "document_type": d.get("document_type") or "",
            })
        else:
            doc_list.append({"description": str(d), "snippet": ""})

    party_raw = item.get("party") or []
    parties = party_raw if isinstance(party_raw, list) else [party_raw] if party_raw else []
    attorney_raw = item.get("attorney") or []
    attorneys = attorney_raw if isinstance(attorney_raw, list) else [attorney_raw] if attorney_raw else []
    firm_raw = item.get("firm") or []
    firms = firm_raw if isinstance(firm_raw, list) else [firm_raw] if firm_raw else []

    return {
        "docket_id": item.get("docket_id"),
        "case_name": item.get("caseName") or item.get("case_name") or "",
        "case_name_full": item.get("caseNameFull") or item.get("case_name_full") or "",
        "court_id": item.get("court_id") or item.get("court"),
        "court": item.get("court") or "",
        "court_citation_string": item.get("court_citation_string") or "",
        "docket_number": item.get("docketNumber") or item.get("docket_number") or "",
        "date_filed": item.get("dateFiled") or item.get("date_filed"),
        "date_terminated": item.get("dateTerminated") or item.get("date_terminated"),
        "date_argued": item.get("dateArgued") or item.get("date_argued"),
        "nature_of_suit": item.get("suitNature") or item.get("nature_of_suit") or "",
        "cause": item.get("cause") or "",
        "documents": doc_list,
        "absolute_url": item.get("docket_absolute_url") or item.get("absolute_url") or "",
        "parties": parties,
        "party_id": item.get("party_id") or [],
        "attorneys": attorneys,
        "attorney_id": item.get("attorney_id") or [],
        "firms": firms,
        "firm_id": item.get("firm_id") or [],
        "assigned_to": item.get("assignedTo") or item.get("assigned_to") or "",
        "assigned_to_id": item.get("assigned_to_id"),
        "referred_to": item.get("referredTo") or item.get("referred_to") or "",
        "referred_to_id": item.get("referred_to_id"),
        "jurisdiction_type": item.get("jurisdictionType") or item.get("jurisdiction_type") or "",
        "jury_demand": item.get("juryDemand") or item.get("jury_demand") or "",
        "pacer_case_id": item.get("pacer_case_id"),
    }
