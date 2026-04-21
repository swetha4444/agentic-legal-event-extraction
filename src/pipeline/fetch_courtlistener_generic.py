#!/usr/bin/env python3
"""
Generic CourtListener fetch: search URLs (any court / filters) and/or docket landing pages.

Output matches the E.D. Pa. pipeline: each .txt file is

  Source URL: <document URL>
  Title: <page title>

  <full text from document <pre>>

Examples:

  # New York Southern (nature of suit 442, employment)
  python fetch_courtlistener_generic.py \\
    --search-url "https://www.courtlistener.com/?q=&type=r&order_by=score+desc&available_only=on&nature_of_suit=442&court=nysd" \\
    --output-dir ./courtlistener_cases_nysd --max-cases 50

  # Middle District of Pennsylvania
  python fetch_courtlistener_generic.py \\
    --search-url "https://www.courtlistener.com/?q=&type=r&order_by=score+desc&available_only=on&nature_of_suit=442&court=pawd" \\
    --output-dir ./courtlistener_cases_pawd

  # Single case (docket page — script resolves to document #1)
  python fetch_courtlistener_generic.py \\
    --docket-url "https://www.courtlistener.com/docket/71540902/green-v-university-of-mississippi-medical-center/" \\
    --output-dir ./courtlistener_cases_oneoff

  # JSON config (see pipeline.example.json)
  python fetch_courtlistener_generic.py --config pipeline.example.json --output-dir ./out

Optional: set variables in the repo `.env` (loaded automatically if python-dotenv is installed).
CLI flags override env. See `.env.example` (COURTLISTENER_* keys).

Requires: Chrome + chromedriver (or Firefox + geckodriver with --firefox).
On cluster/login nodes (PEP 668): run `bash setup_venv.sh` in this folder, then `source .venv/bin/activate`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

# URL path pattern for "document #1" (initial doc): /docket/{id}/1/...
INITIAL_DOC_HREF_RE = re.compile(r"/docket/\d+/1(?:/|$)")


def _load_project_dotenv() -> None:
    """Load repo `.env` when running from any cwd (optional python-dotenv)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    here = Path(__file__).resolve()
    for d in (here.parent, *here.parents):
        envf = d / ".env"
        if envf.is_file():
            load_dotenv(envf, override=False)
            return


def _env_int(key: str, default: int) -> int:
    v = (os.environ.get(key) or "").strip()
    if not v:
        return default
    return int(v)


def _env_float(key: str, default: float) -> float:
    v = (os.environ.get(key) or "").strip()
    if not v:
        return default
    return float(v)


def _env_str(key: str, default: str | None = None) -> str | None:
    v = (os.environ.get(key) or "").strip()
    return v if v else default


def _env_bool(key: str) -> bool:
    return (os.environ.get(key) or "").strip().lower() in ("1", "true", "yes", "on")


def _has_display() -> bool:
    return bool((os.environ.get("DISPLAY") or "").strip() or (os.environ.get("WAYLAND_DISPLAY") or "").strip())


def _env_url_list(key: str) -> list[str]:
    raw = (os.environ.get(key) or "").strip()
    if not raw:
        return []
    return [normalize_search_url(u) for u in raw.splitlines() if u.strip()]


def normalize_search_url(url: str) -> str:
    """Strip fragment (#...) so the server receives the same query as the browser."""
    return url.strip().split("#", 1)[0].strip()


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "document"


def _get_driver(*, use_firefox: bool, use_headless: bool):
    # HPC/login nodes often do not have a GUI display.
    if not use_headless and not _has_display():
        print("No DISPLAY detected; forcing headless browser mode.", flush=True)
        use_headless = True

    if use_firefox:
        from selenium import webdriver
        from selenium.webdriver.firefox.options import Options

        opts = Options()
        if use_headless:
            opts.add_argument("-headless")
        return webdriver.Firefox(options=opts)
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    opts = Options()
    if use_headless:
        opts.add_argument("--headless=new")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--window-size=1920,1080")
        opts.add_argument("--remote-debugging-port=9222")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    return webdriver.Chrome(options=opts)


def get_soup(url: str, use_browser: bool = False) -> BeautifulSoup:
    if use_browser:
        driver = _get_driver(use_firefox=False, use_headless=False)
        try:
            driver.get(url)
            time.sleep(2.0)
            return BeautifulSoup(driver.page_source, "html.parser")
        finally:
            driver.quit()
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def get_search_page_url(base_search_url: str, page: int) -> str:
    base_search_url = normalize_search_url(base_search_url)
    if page <= 1:
        return base_search_url
    parsed = urlparse(base_search_url)
    query = parse_qs(parsed.query)
    query["page"] = [str(page)]
    new_query = urlencode(query, doseq=True)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, ""))


def find_initial_document_links_on_page(soup: BeautifulSoup, base_url: str) -> list[str]:
    seen: set[str] = set()
    by_text: list[str] = []
    by_href: list[str] = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not INITIAL_DOC_HREF_RE.search(href):
            continue
        full = urljoin(base_url, href)
        if full in seen:
            continue
        seen.add(full)
        if a.get_text(strip=True).lower() == "initial document":
            by_text.append(full)
        else:
            by_href.append(full)

    return by_text + by_href


def extract_docket_numeric_id(url: str) -> str | None:
    m = re.search(r"/docket/(\d+)", url)
    return m.group(1) if m else None


def is_document_page_url(url: str) -> bool:
    """True if URL points to a specific attachment (/docket/{id}/{docnum}/...)."""
    parts = [p for p in urlparse(url).path.split("/") if p]
    if len(parts) < 4:
        return False
    if parts[0] != "docket":
        return False
    return parts[2].isdigit()


def resolve_docket_to_initial_document_url(docket_or_doc_url: str, driver) -> str:
    """
    If URL is already a document page, return it (normalized).
    If URL is a docket landing page, load it and find a link to document #1 for that docket.
    """
    u = normalize_search_url(docket_or_doc_url)
    if is_document_page_url(u):
        return u
    docket_id = extract_docket_numeric_id(u)
    if not docket_id:
        raise ValueError(f"Could not parse docket id from URL: {u!r}")

    driver.get(u)
    time.sleep(2.5)
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(0.5)
    soup = BeautifulSoup(driver.page_source, "html.parser")

    pat = re.compile(rf"/docket/{re.escape(docket_id)}/1(?:/|$)")
    by_text: list[str] = []
    by_href: list[str] = []
    seen: set[str] = set()

    for a in soup.find_all("a", href=True):
        href = a.get("href") or ""
        if not pat.search(href):
            continue
        full = urljoin(u, href)
        if full in seen:
            continue
        seen.add(full)
        if a.get_text(strip=True).lower() == "initial document":
            by_text.append(full)
        else:
            by_href.append(full)

    ordered = by_text + by_href
    if ordered:
        return ordered[0]

    raise RuntimeError(
        f"No link to document #1 found on docket page. Open in browser and check: {u}"
    )


def extract_document_text(doc_url: str, driver=None) -> tuple[str, str]:
    if driver is not None:
        driver.get(doc_url)
        time.sleep(2.0)
        soup = BeautifulSoup(driver.page_source, "html.parser")
    else:
        try:
            soup = get_soup(doc_url)
        except Exception:
            soup = get_soup(doc_url, use_browser=True)
    pre = soup.find("pre", id="document-text") or soup.find("pre")
    if pre is None:
        raise RuntimeError("No <pre> document text found on page.")
    text = pre.get_text("\n")
    if not text.strip():
        print("    WARNING: extracted text is empty (may be scanned-only).")
    title_el = soup.find("h1") or soup.find("title")
    title = title_el.get_text(strip=True) if title_el else "document"
    return title, text


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run(
    *,
    search_specs: list[tuple[str, int]],
    docket_urls: list[str],
    output_dir: str,
    seen_urls_file: str,
    max_cases: int,
    request_delay: float,
    max_pages: int,
    debug_save_html: bool,
    debug_html_path: str,
    use_firefox: bool,
    use_headless: bool,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    seen_urls: set[str] = set()
    if os.path.exists(seen_urls_file):
        with open(seen_urls_file, encoding="utf-8") as f:
            seen_urls = {line.strip() for line in f if line.strip()}
        print(f"Resuming: {len(seen_urls)} URLs already in {seen_urls_file}")

    # Same as original paed fetcher: counter follows resume list length
    saved_count = len(seen_urls)

    driver = _get_driver(use_firefox=use_firefox, use_headless=use_headless)
    try:
        # 1) Explicit docket / one-off document URLs
        for raw in docket_urls:
            if saved_count >= max_cases:
                break
            u = normalize_search_url(raw)
            try:
                doc_url = resolve_docket_to_initial_document_url(u, driver)
            except Exception as e:
                print(f"  [docket] SKIP {u}: {e}")
                continue
            if doc_url in seen_urls:
                print(f"  [docket] already seen: {doc_url}")
                continue
            try:
                time.sleep(request_delay)
                title, text = extract_document_text(doc_url, driver=driver)
                saved_count += 1
                seen_urls.add(doc_url)
                slug = slugify(title)
                filename = f"{saved_count:04d}_{slug}.txt"
                out_path = os.path.join(output_dir, filename)
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(f"Source URL: {doc_url}\n")
                    f.write(f"Title: {title}\n\n")
                    f.write(text)
                print(f"  Saved #{saved_count}: {filename} (docket)")
                with open(seen_urls_file, "a", encoding="utf-8") as f:
                    f.write(doc_url + "\n")
            except Exception as e:
                print(f"  ERROR {doc_url}: {e}")

        # 2) Paginated search URLs (in order; each URL may have its own start page)
        for base, sp0 in search_specs:
            if saved_count >= max_cases:
                break
            base = normalize_search_url(base)
            page = max(1, int(sp0))
            while saved_count < max_cases and page <= max_pages:
                search_url = get_search_page_url(base, page)
                print(f"Search page {page}: {search_url}")

                driver.get(search_url)
                time.sleep(3.0)
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1.0)
                soup = BeautifulSoup(driver.page_source, "html.parser")
                time.sleep(request_delay)

                doc_urls = find_initial_document_links_on_page(soup, search_url)
                if not doc_urls:
                    print("  No initial document links on this page.")
                    if page == max(1, int(sp0)) and debug_save_html:
                        with open(debug_html_path, "w", encoding="utf-8") as f:
                            f.write(soup.prettify())
                        print(f"  Saved page HTML to {debug_html_path} for debugging.")
                    break

                for url in doc_urls:
                    if saved_count >= max_cases:
                        break
                    if url in seen_urls:
                        continue
                    try:
                        time.sleep(request_delay)
                        title, text = extract_document_text(url, driver=driver)
                        saved_count += 1
                        seen_urls.add(url)
                        slug = slugify(title)
                        filename = f"{saved_count:04d}_{slug}.txt"
                        out_path = os.path.join(output_dir, filename)
                        with open(out_path, "w", encoding="utf-8") as f:
                            f.write(f"Source URL: {url}\n")
                            f.write(f"Title: {title}\n\n")
                            f.write(text)
                        print(f"  Saved #{saved_count}: {filename}")
                        with open(seen_urls_file, "a", encoding="utf-8") as f:
                            f.write(url + "\n")
                    except Exception as e:
                        print(f"  ERROR {url}: {e}")

                page += 1

    finally:
        driver.quit()

    print("Done.")


def main() -> None:
    _load_project_dotenv()

    p = argparse.ArgumentParser(
        description="Fetch CourtListener initial documents into .txt (same format as E.D. Pa. pipeline)."
    )
    p.add_argument(
        "--search-url",
        action="append",
        default=[],
        help="CourtListener search results URL (repeat for multiple searches). Fragment (#...) is ignored.",
    )
    p.add_argument(
        "--docket-url",
        action="append",
        default=[],
        help="Docket landing page or document URL; non-document URLs are resolved to attachment #1.",
    )
    p.add_argument(
        "--config",
        default=None,
        help="JSON file with keys: searches (list of {url, start_page?}), docket_urls (list of strings). "
        "Default: COURTLISTENER_CONFIG if set in .env.",
    )
    p.add_argument(
        "--output-dir",
        default=_env_str("COURTLISTENER_OUTPUT_DIR") or "courtlistener_cases",
        help="Directory for .txt output (env: COURTLISTENER_OUTPUT_DIR).",
    )
    p.add_argument(
        "--seen-urls-file",
        default=_env_str("COURTLISTENER_SEEN_URLS_FILE") or "seen_urls.txt",
        help="Resume dedupe file (env: COURTLISTENER_SEEN_URLS_FILE).",
    )
    p.add_argument(
        "--max-cases",
        type=int,
        default=_env_int("COURTLISTENER_MAX_CASES", 1500),
        help="Stop when running document count reaches this (env: COURTLISTENER_MAX_CASES).",
    )
    p.add_argument(
        "--start-page",
        type=int,
        default=_env_int("COURTLISTENER_START_PAGE", 1),
        help="First search results page (env: COURTLISTENER_START_PAGE).",
    )
    p.add_argument(
        "--request-delay",
        type=float,
        default=_env_float("COURTLISTENER_REQUEST_DELAY", 1.0),
        help="Seconds between requests (env: COURTLISTENER_REQUEST_DELAY).",
    )
    p.add_argument(
        "--max-pages",
        type=int,
        default=_env_int("COURTLISTENER_MAX_PAGES", 500),
        help="Safety cap per search URL (env: COURTLISTENER_MAX_PAGES).",
    )
    p.add_argument(
        "--debug-save-html",
        action="store_true",
        help="Save HTML when first search page has no links (env: COURTLISTENER_DEBUG_SAVE_HTML=1).",
    )
    p.add_argument(
        "--debug-html-path",
        default=_env_str("COURTLISTENER_DEBUG_HTML_PATH") or "debug_search_page.html",
        help="Debug HTML path (env: COURTLISTENER_DEBUG_HTML_PATH).",
    )
    p.add_argument("--firefox", action="store_true", help="Firefox (env: COURTLISTENER_USE_FIREFOX=1).")
    p.add_argument("--headless", action="store_true", help="Headless browser (env: COURTLISTENER_HEADLESS=1).")
    args = p.parse_args()

    search_urls_cli = list(args.search_url)
    docket_urls_cli = list(args.docket_url)
    search_urls_env = _env_url_list("COURTLISTENER_SEARCH_URLS")
    if not search_urls_env:
        su1 = _env_str("COURTLISTENER_SEARCH_URL")
        if su1:
            search_urls_env = [normalize_search_url(su1)]
    docket_urls_env = _env_url_list("COURTLISTENER_DOCKET_URLS")
    if not docket_urls_env:
        du1 = _env_str("COURTLISTENER_DOCKET_URL")
        if du1:
            docket_urls_env = [normalize_search_url(du1)]
    search_urls = search_urls_cli if search_urls_cli else search_urls_env
    docket_urls = docket_urls_cli if docket_urls_cli else docket_urls_env

    search_specs: list[tuple[str, int]] = []
    for u in search_urls:
        search_specs.append((normalize_search_url(u), int(args.start_page)))

    config_path = args.config or _env_str("COURTLISTENER_CONFIG")
    if config_path:
        cfg = load_config(config_path)
        for item in cfg.get("searches") or []:
            if isinstance(item, str):
                search_specs.append((normalize_search_url(item), int(args.start_page)))
            elif isinstance(item, dict) and item.get("url"):
                sp = int(item.get("start_page", args.start_page))
                search_specs.append((normalize_search_url(item["url"]), sp))
        docket_urls.extend(cfg.get("docket_urls") or [])

    if not search_specs and not docket_urls:
        p.error(
            "Provide --search-url / --docket-url, COURTLISTENER_SEARCH_URL(S) / COURTLISTENER_DOCKET_URL(S) in .env, "
            "or --config / COURTLISTENER_CONFIG."
        )

    seen_path = args.seen_urls_file
    if not os.path.isabs(seen_path):
        seen_path = os.path.join(args.output_dir, seen_path)

    use_firefox = bool(args.firefox or _env_bool("COURTLISTENER_USE_FIREFOX"))
    use_headless = bool(args.headless or _env_bool("COURTLISTENER_HEADLESS"))
    debug_save_html = bool(args.debug_save_html or _env_bool("COURTLISTENER_DEBUG_SAVE_HTML"))

    run(
        search_specs=search_specs,
        docket_urls=docket_urls,
        output_dir=args.output_dir,
        seen_urls_file=seen_path,
        max_cases=args.max_cases,
        request_delay=args.request_delay,
        max_pages=args.max_pages,
        debug_save_html=debug_save_html,
        debug_html_path=args.debug_html_path,
        use_firefox=use_firefox,
        use_headless=use_headless,
    )


if __name__ == "__main__":
    main()
