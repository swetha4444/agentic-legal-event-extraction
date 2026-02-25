#!/usr/bin/env python3
"""
Fetch initial document (complaint) text from CourtListener search results.
Automatically follows pagination to the next page when the current page is done.

Uses Selenium with a single browser session for search and document pages
(CourtListener blocks plain requests and loads document text via JavaScript).

Usage:
    pip install requests beautifulsoup4 selenium
    python fetch_courtlistener_cases.py

You need Chrome and chromedriver (or set USE_FIREFOX=True and geckodriver).
On Mac with Homebrew: brew install chromedriver

Configure MAX_CASES and OUTPUT_DIR below as needed.
"""

import os
import re
import time
from urllib.parse import urljoin, urlparse, parse_qs, urlencode, urlunparse

import requests
from bs4 import BeautifulSoup

# Use Firefox instead of Chrome if you prefer (requires geckodriver)
USE_FIREFOX = False

# CloudFront blocks headless browsers. Set to False to open a visible Chrome window.
USE_HEADLESS = False

# Base search URL (no page param for first page)
BASE_SEARCH_URL = (
    "https://www.courtlistener.com/"
    "?type=r&q=&type=r&order_by=score+desc&nature_of_suit=442"
    "&available_only=on&court=paed"
)

OUTPUT_DIR = "/Users/vishnuvardhan/Desktop/698/untitled folder/courtlistener_cases"
MAX_CASES = 1500
REQUEST_DELAY = 1.0

# Start fetching search results from this page (1-based).
# Example: set START_PAGE = 120 to start at page 120.
# If you want to start *after* page 120, set START_PAGE = 121.
START_PAGE = 95

# Optional: save first search page HTML for debugging (e.g. if no links found)
DEBUG_SAVE_HTML = True
DEBUG_HTML_PATH = "debug_search_page.html"

# Resume: file tracking already-fetched URLs (one URL per line)
SEEN_URLS_FILE = "courtlistener_seen_urls.txt"


def slugify(text: str) -> str:
    """Turn a case title into a filesystem-safe slug."""
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "document"


def _get_driver():
    """Create browser (Chrome or Firefox). CloudFront blocks headless; use visible browser if needed."""
    if USE_FIREFOX:
        from selenium import webdriver
        from selenium.webdriver.firefox.options import Options
        opts = Options()
        if USE_HEADLESS:
            opts.add_argument("--headless")
        return webdriver.Firefox(options=opts)
    else:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        opts = Options()
        if USE_HEADLESS:
            opts.add_argument("--headless")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--disable-blink-features=AutomationControlled")
        opts.add_experimental_option("excludeSwitches", ["enable-automation"])
        return webdriver.Chrome(options=opts)


def get_page_html_with_browser(url: str, wait_seconds: float = 2.0) -> str:
    """Load URL in a real browser and return page HTML (works when site blocks requests or uses JS)."""
    driver = _get_driver()
    try:
        driver.get(url)
        time.sleep(wait_seconds)  # allow content to render (including JS)
        return driver.page_source
    finally:
        driver.quit()


def get_soup(url: str, use_browser: bool = False) -> BeautifulSoup:
    """Fetch URL and return BeautifulSoup. use_browser=True for pages that need JS or block requests."""
    if use_browser:
        html = get_page_html_with_browser(url)
        return BeautifulSoup(html, "html.parser")
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


def get_search_page_url(page: int) -> str:
    """Return the full URL for a given search result page (1-based)."""
    if page <= 1:
        return BASE_SEARCH_URL
    parsed = urlparse(BASE_SEARCH_URL)
    query = parse_qs(parsed.query)
    query["page"] = [str(page)]
    new_query = urlencode(query, doseq=True)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, ""))


# URL path pattern for "document #1" (initial doc) on CourtListener: /docket/{id}/1/...
INITIAL_DOC_HREF_RE = re.compile(r"/docket/\d+/1(?:/|$)")


def find_initial_document_links_on_page(soup: BeautifulSoup, base_url: str) -> list[str]:
    """
    Find all initial document links on a search results page.
    Uses href pattern /docket/{id}/1/... (document #1); prefers links with text "Initial Document".
    """
    seen = set()
    by_text = []
    by_href = []

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


def has_next_page(soup: BeautifulSoup, current_page: int) -> bool:
    """Check if there is a 'Next' pagination link or a link to the next page number."""
    # Look for "Next" link
    for a in soup.find_all("a", href=True):
        text = a.get_text(strip=True)
        href = a.get("href", "")
        if re.match(r"^Next\s*$", text, re.I):
            return True
        # Look for link to next page (e.g. page=2 when we're on page 1)
        if f"page={current_page + 1}" in href.replace("&amp;", "&"):
            return True
    return False


def extract_document_text(doc_url: str, driver=None) -> tuple[str, str]:
    """
    Fetch document page and extract full text from <pre id="document-text"> or first <pre>.
    Uses shared driver if provided (faster); otherwise opens a new browser per page.
    Returns (title, full_text).
    """
    if driver is not None:
        driver.get(doc_url)
        time.sleep(2)  # allow JS to render
        soup = BeautifulSoup(driver.page_source, "html.parser")
    else:
        # Try requests first (faster when it works)
        try:
            soup = get_soup(doc_url)
        except Exception:
            soup = get_soup(doc_url, use_browser=True)
    # Prefer id="document-text" for main document content; fallback to first <pre>
    pre = soup.find("pre", id="document-text") or soup.find("pre")
    if pre is None:
        raise RuntimeError("No <pre> document text found on page.")
    text = pre.get_text("\n")
    if not text.strip():
        print("    WARNING: extracted text is empty (may be scanned-only).")
    title_el = soup.find("h1") or soup.find("title")
    title = title_el.get_text(strip=True) if title_el else "document"
    return title, text


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Resume: skip URLs we've already fetched
    seen_urls = set()
    seen_path = os.path.join(os.path.dirname(OUTPUT_DIR), SEEN_URLS_FILE)
    if os.path.exists(seen_path):
        with open(seen_path, encoding="utf-8") as f:
            seen_urls = set(line.strip() for line in f if line.strip())
        print(f"Resuming: {len(seen_urls)} URLs already fetched.")
    saved_count = len(seen_urls)
    page = max(1, int(START_PAGE))

    # Use single browser session for all pages (search + documents)
    driver = _get_driver()
    try:
        while saved_count < MAX_CASES:
            search_url = get_search_page_url(page)
            print(f"Page {page}: {search_url}")

            driver.get(search_url)
            time.sleep(3)  # allow full page load
            # Scroll to bottom to trigger any lazy-loaded content
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)
            soup = BeautifulSoup(driver.page_source, "html.parser")
            time.sleep(REQUEST_DELAY)

            doc_urls = find_initial_document_links_on_page(soup, search_url)
            if not doc_urls:
                print("  No initial document links on this page.")
                if page == max(1, int(START_PAGE)):
                    if DEBUG_SAVE_HTML:
                        with open(DEBUG_HTML_PATH, "w", encoding="utf-8") as f:
                            f.write(soup.prettify())
                        print(f"  Saved page HTML to {DEBUG_HTML_PATH} for debugging.")
                    break  # Stop only if first page has no links
                # Otherwise try next page (might be loading or different layout)
                page += 1
                continue

            for url in doc_urls:
                if saved_count >= MAX_CASES:
                    break
                if url in seen_urls:
                    continue
                try:
                    time.sleep(REQUEST_DELAY)
                    title, text = extract_document_text(url, driver=driver)
                    saved_count += 1
                    seen_urls.add(url)
                    slug = slugify(title)
                    filename = f"{saved_count:04d}_{slug}.txt"
                    out_path = os.path.join(OUTPUT_DIR, filename)
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(f"Source URL: {url}\n")
                        f.write(f"Title: {title}\n\n")
                        f.write(text)
                    print(f"  Saved #{saved_count}: {filename}")
                    # Persist seen URLs for resume
                    with open(seen_path, "a", encoding="utf-8") as f:
                        f.write(url + "\n")
                except Exception as e:
                    print(f"  ERROR {url}: {e}")

            if saved_count >= MAX_CASES:
                break
            # Always try next page (more reliable than parsing pagination links)
            page += 1
            if page > 250:  # safety: search has ~210 pages
                print("  Reached page limit. Done.")
                break
    finally:
        driver.quit()

    print("Done.")


if __name__ == "__main__":
    main()
