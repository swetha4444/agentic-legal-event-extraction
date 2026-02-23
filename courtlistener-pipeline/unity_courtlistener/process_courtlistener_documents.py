#!/usr/bin/env python3
"""
Process CourtListener case documents: split into sections and extract clean sentences.
- Sections: INTRODUCTION, JURISDICTION AND VENUE, PARTIES, FACTUAL ALLEGATIONS, etc.
- Sentences: no leading numbers (1., 2.), no junk, clean text ending with period.
- Optional: use LLM (OpenAI) for section extraction via --use-llm, --llm-fallback, or --use-both.

How to run (from the folder containing courtlistener_cases/ and this script):

  # Rule-based only (no API key needed)
  python3 process_courtlistener_documents.py
  python3 process_courtlistener_documents.py --limit 5

  # LLM only (needs OPENAI_API_KEY)
  export OPENAI_API_KEY="your-key"
  python3 process_courtlistener_documents.py --use-llm

  # Rule-based first; use LLM only if result looks incomplete
  python3 process_courtlistener_documents.py --llm-fallback

  # Run both and merge results (recommended for best coverage)
  python3 process_courtlistener_documents.py --use-both

  # Use Groq (free tier, OpenAI-compatible)
  export OPENAI_API_KEY="your-groq-api-key"
  export OPENAI_BASE_URL="https://api.groq.com/openai/v1"
  python3 process_courtlistener_documents.py --use-llm --llm-model llama-3.1-8b-instant

Output goes to courtlistener_processed/ (or --output-dir). The script prints the
mode at start (e.g. "Mode: rule-based only" or "Mode: LLM only") and per-file
"[LLM]", "[merged rule+LLM]", or "[LLM fallback used]" when the LLM is used.
"""

import re
import os
import json
from pathlib import Path

# Max characters per LLM request (stay within model context; ~1 char ≈ 0.25 token)
# Single-request limit; when doc is longer we chunk (see CHUNK_OVERLAP).
LLM_MAX_CHARS = 200_000
# When chunking long docs: chunk size and overlap (chars) so we don't cut mid-section
LLM_CHUNK_SIZE = 150_000
LLM_CHUNK_OVERLAP = 8_000


# Common section headers (ALL CAPS). We detect by pattern, not fixed list.
SECTION_HEADER_RE = re.compile(
    r'^\s*([A-Z][A-Z0-9\s\-&/,\.\'\(\)]+)\s*$',  # line that is mostly/all caps
    re.MULTILINE
)

# Known section heading names (normalized: lowercase, single space) so we don't miss
# Title Case, mixed case, or typos. Add variants and common misspellings.
KNOWN_SECTION_NAMES = frozenset([
    'introduction', 'intro',
    'jurisdiction', 'venue', 'jurisdiction and venue', 'jurisdiction and venue',
    'parties', 'the parties',
    'facts', 'factual background', 'factual allegations', 'factual history',
    'background', 'factual background',
    'cause of action', 'first cause of action', 'second cause of action',
    'third cause of action', 'fourth cause of action', 'fifth cause of action',
    'sixth cause of action', 'seventh cause of action', 'eighth cause of action',
    'v cause of action', 'vi cause of action', 've cause of action',
    'count i', 'count ii', 'count iii', 'count iv', 'count v', 'count vi',
    'count 1', 'count 2', 'count 3', 'count 4',
    'prayer for relief', 'prayer for relief',
    'relief requested', 'request for relief',
    'jury trial demanded', 'demand for jury trial', 'jury trial demand',
    'exhaustion of administrative remedies', 'administrative exhaustion',
    'basis for instant action', 'basis for action',
    'procedural prerequisites', 'procedural prerequisite',
    'nature of the action', 'nature of action',
    'civil complaint', 'complaint',
    'designation form', 'arbitration certification',
    'disability discrimination and retaliation under the ada',
    'retaliation under the fmla', 'violation of title vii', 'violation of phra',
    'discrimination under the americans with disabilities act',
    'failure to accommodate under the americans with disabilities act',
    'retaliation under the americans with disabilities act',
    'interference and retaliation under the family medical leave act',
    'discrimination under pennsylvania human relations act',
    'retaliation under the pennsylvania human relations act',
    'aiding and abetting under the pennsylvania human relations act',
    'discrimination under the philadelphia fair practices ordinance',
    'verification', 'demand for jury trial',
    'violation of title vii', 'violation of title vii--sexual harassment and retaliation',
])

# Typos / OCR-style fixes when normalizing a line for section matching
SECTION_TYPO_FIXES = [
    (r'counti\b', 'count i'),   # COUNTI -> count i
    (r'count\s*it\b', 'count ii'),  # Count IT -> count ii
    (r'\bve\s+cause', 'v cause'),   # VE CAUSE -> v cause
    (r'\bvi\s+cause', 'vi cause'),
    (r'jurisidction', 'jurisdiction'),
    (r'diserimination', 'discrimination'),
    (r'title\s*vu\b', 'title vii'),
    (r'title\s*vii\b', 'title vii'),
]

# Lines that look like section headers but are form/caption junk (don't treat as section)
NOT_SECTION_PATTERNS = [
    re.compile(r'^defendants?\s*$', re.I),
    re.compile(r'^plaintiffs?\s*$', re.I),
    re.compile(r'^in the united states district court\s*$', re.I),
    re.compile(r'^for the eastern district of pennsylvania\s*$', re.I),
    re.compile(r'^civil cover sheet\s*$', re.I),
    re.compile(r'^civil action\s*$', re.I),
    re.compile(r'^civil action no\.?\s*$', re.I),
    re.compile(r'^attorneys?\s*\(', re.I),
    re.compile(r'^county of residence', re.I),
    re.compile(r'^nature of suit', re.I),
    re.compile(r'^basis of jurisdiction', re.I),
    re.compile(r'^citizenship of principal parties', re.I),
    re.compile(r'^the tract of land', re.I),
    re.compile(r'^related case', re.I),
    re.compile(r'^date signature of attorney', re.I),
    re.compile(r'^amount applying ifp', re.I),
    re.compile(r'^mag\.?\s*judge', re.I),
    re.compile(r'^except in u\.s\. plaintiff', re.I),
    re.compile(r'^the vanguard group', re.I),
    re.compile(r'^deborah blair.*civil action', re.I),
    re.compile(r'^designation form\s*$', re.I),
    re.compile(r'^arbitration certification\s*$', re.I),
    re.compile(r'^case management track designation', re.I),
    re.compile(r'^contract torts forfeit', re.I),
    re.compile(r'^receipt\s*#', re.I),
    re.compile(r'^l\s*\(a\)\s*plaintiffs defendants', re.I),
    re.compile(r'^complaint:.*rule 23', re.I),
    re.compile(r'^\d+ u\.s\.c\.', re.I),  # "42 U.S.C. 2000e" form line
    re.compile(r'^exhibit\s+[a-z]\s*$', re.I),
    re.compile(r'^dismissal and notice of rights\s*$', re.I),
    re.compile(r'^confidential\s*\(', re.I),
    re.compile(r'^notice of suit rights\s*$', re.I),
    re.compile(r'^\(?except in u\.s\. plaintiff', re.I),
    re.compile(r'^note: inland', re.I),
    re.compile(r'^for office use only', re.I),
    re.compile(r'^united states district court\s*$', re.I),
    re.compile(r'^v\.\s*;\s*\(.*jurors demanded', re.I),
    re.compile(r'^eeoc form\s+\d+', re.I),
    re.compile(r'^-\s*notice of suit rights\s*-', re.I),
    re.compile(r'^l\s*_contract', re.I),
    re.compile(r'^\[\s*contract torts', re.I),
    re.compile(r'^\(?in u\.s\. plaintiff cases only', re.I),
    re.compile(r'^[ivx]+\.\s*related case\s*\(s\)\s*$', re.I),
    re.compile(r'^:\s*jury trial by twelve', re.I),
]

# Junk patterns to drop from text
JUNK_PATTERNS = [
    re.compile(r'^Source URL:.*', re.IGNORECASE),
    re.compile(r'^Title:.*', re.IGNORECASE),
    re.compile(r'^Case\s+\d+[\:\-]\d+[\w\-]*\s+Document\s+\d+.*', re.IGNORECASE),
    re.compile(r'^\s*Page\s+\d+\s+of\s+\d+\s*$', re.IGNORECASE),
    re.compile(r'^Filed\s+\d{2}/\d{2}/\d{2}.*', re.IGNORECASE),
    re.compile(r'^IN THE UNITED STATES DISTRICT COURT\s*$', re.IGNORECASE),
    re.compile(r'^FOR THE EASTERN DISTRICT OF PENNSYLVANIA\s*$', re.IGNORECASE),
    re.compile(r'^\s*\d+\s*$'),  # standalone page number
    re.compile(r'^JS 44.*', re.IGNORECASE),
    re.compile(r'^CIVIL COVER SHEET\s*$', re.IGNORECASE),
]

# Paragraph number at start of line: "1.", "2.", "  17.  ", "  68.     "
PARA_NUM_RE = re.compile(r'^\s*\d+\.\s*')

# Letter list at start: "A.", "B."
LETTER_NUM_RE = re.compile(r'^\s*[A-Z]\.\s*')


def is_junk_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    for pat in JUNK_PATTERNS:
        if pat.match(s):
            return True
    # Lines that are only digits and spaces
    if re.match(r'^[\d\s]+$', s):
        return True
    # Very short lines that look like document art (e.g. ":", "v.")
    if len(s) <= 2 and s in (':', 'v.', 'v'):
        return True
    # Caption-style: "AKZO NOBEL COATINGS INC., :" or "Defendant. : JURY TRIAL DEMANDED"
    if re.search(r'\s+:\s*$', s) or re.match(r'^[A-Z\s\-]+,?\s*:\s*$', s):
        return True
    if re.match(r'^(Defendant|Plaintiff)\.?\s*:\s*', s, re.I):
        return True
    if s.endswith('JURY TRIAL DEMANDED') and len(s) < 50:
        return True
    if re.match(r'^[vV]\.\s*:\s*CIVIL ACTION', s):
        return True
    if re.match(r'^CIVIL ACTION NO\.?\s*$', s, re.I):
        return True
    return False


def _normalize_for_section_match(line: str) -> str:
    """Normalize line for matching against known section names (lowercase, fix typos, collapse space)."""
    s = ' '.join(line.strip().split()).lower()
    for pat, repl in SECTION_TYPO_FIXES:
        s = re.sub(pat, repl, s, flags=re.I)
    return s


def is_section_header(line: str) -> bool:
    """True if line looks like a section header: ALL CAPS or matches known section names (including Title Case / typos)."""
    s = line.strip()
    # Too short to be a meaningful section (allow "Venue", "Facts", "Parties" = 5-7 chars)
    if len(s) < 4:
        return False
    # Caption-style lines (court header): contain " : " or end with " :" or "."
    if ' : ' in s or s.endswith(' :') or s.rstrip().endswith(':') or re.match(r'^[A-Z\s]+\s*:\s*$', s):
        return False
    if re.search(r'\b(Defendant|Plaintiff)\.?\s*$', s, re.I) and len(s) < 50:
        return False

    # 1) Check known section names (Title Case, mixed case, typos)
    normalized = _normalize_for_section_match(s)
    if normalized in KNOWN_SECTION_NAMES:
        for pat in NOT_SECTION_PATTERNS:
            if pat.search(normalized):
                return False
        return True

    # 2) Check if line is a "Count I/II/III" or "COUNT I" / "COUNTI" style
    if re.match(r'^count\s+[i]+\s*$', normalized) or re.match(r'^count\s+\d+\s*$', normalized):
        return True
    if re.match(r'^count\s+[i]+\s+', normalized):  # "Count I Age Discrimination..."
        return True

    # 3) ALL CAPS section header (at least 6 chars, predominantly uppercase)
    if len(s) < 6:
        return False
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return False
    caps = sum(1 for c in letters if c.isupper())
    if caps < 0.9 * len(letters):
        return False
    # Exclude form/caption junk that is all caps
    for pat in NOT_SECTION_PATTERNS:
        if pat.search(normalized):
            return False
    return True


def normalize_section_name(line: str) -> str:
    """Return a clean section title (strip, collapse space)."""
    return ' '.join(line.strip().split())


def strip_paragraph_number(line: str) -> str:
    """Remove leading '1.' or '  17.   ' from line."""
    return PARA_NUM_RE.sub('', line).strip()


def strip_letter_number(line: str) -> str:
    """Remove leading 'A.' or 'B.' from line."""
    return LETTER_NUM_RE.sub('', line).strip()


def clean_line(line: str) -> str:
    """Remove paragraph/letter numbering and extra spaces."""
    s = strip_paragraph_number(line)
    s = strip_letter_number(s)
    return ' '.join(s.split())


# Abbreviations that should not trigger sentence split (period is part of abbreviation)
ABBREVS = frozenset(
    'Dr. Mr. Mrs. Ms. Sr. Jr. Ph.D. U.S. U.S.C. et. seq. etc. No. Inc. Ltd. Corp. Co. v. cf. id. al. P.S.'.split()
)


def _ends_with_abbrev(text: str) -> bool:
    """True if text ends with a known abbreviation (so we should not split here)."""
    t = text.strip().replace('\u2024', '.').replace('\u00b7', '.')
    if not t:
        return False
    words = re.split(r'\s+', t)
    last_word = words[-1] if words else ''
    # Normalize: remove periods for comparison (so U.S. -> us, Dr. -> dr)
    key = last_word.replace('.', '').lower()
    if key in ('dr', 'mr', 'mrs', 'ms', 'sr', 'jr', 'us', 'usc', 'et', 'seq', 'etc', 'no', 'inc', 'ltd', 'corp', 'co', 'v', 'cf', 'id', 'al', 'ps', 'ph'):
        return True
    if last_word in ABBREVS:
        return True
    if len(last_word) <= 4 and last_word.endswith('.'):
        return True
    return False


def split_into_sentences(paragraph: str) -> list[str]:
    """Split paragraph into sentences. Avoid splitting after common abbreviations."""
    if not paragraph or not paragraph.strip():
        return []
    text = ' '.join(paragraph.split())
    text = text.replace('\u2024', '.').replace('\u00b7', '.')
    sentences = []
    start = 0  # start of current sentence
    pos = 0
    n = len(text)
    while pos < n:
        match = re.search(r'[\.\?\!]\s+(?=[A-Z\"\']|\d|\$)|[\.\?\!]\s*$', text[pos:])
        if not match:
            fragment = text[start:].strip()
            if fragment:
                if not fragment.endswith(('.', '!', '?')):
                    fragment = fragment + '.'
                sentences.append(fragment)
            break
        candidate = text[start : pos + match.start() + 1].strip()
        if _ends_with_abbrev(candidate):
            # Period is abbreviation; extend current sentence (don't split)
            pos = pos + match.end()
            continue
        if candidate:
            sentences.append(candidate)
        start = pos + match.end()
        pos = start
    return sentences


def extract_sections_and_sentences(filepath: str) -> dict:
    """
    Read a CourtListener txt file and return a dict:
    { "sections": [ { "name": "INTRODUCTION", "sentences": ["...", "..."] }, ... ] }
    """
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    # Drop junk lines and normalize
    cleaned_lines = []
    for line in lines:
        line = line.replace('\f', '\n').rstrip()
        if is_junk_line(line):
            continue
        cleaned_lines.append(line)

    # Build list of (line_index, section_name) for section headers
    section_breaks = []  # (index, section_name)
    current_section = "INTRODUCTION"  # default for content before first header

    for i, line in enumerate(cleaned_lines):
        if is_section_header(line):
            name = normalize_section_name(line)
            # Skip if it's clearly a caption (e.g. "CIVIL ACTION NO.")
            if re.match(r'^(CIVIL ACTION|JURY TRIAL|COMPLAINT)\s*$', name, re.I):
                continue
            section_breaks.append((i, name))
    section_breaks.sort(key=lambda x: x[0])
    # Ensure content before first header belongs to first section (e.g. INTRODUCTION)
    if section_breaks and section_breaks[0][0] > 0:
        first_name = section_breaks[0][1]
        section_breaks.insert(0, (0, first_name))

    if not section_breaks:
        section_breaks = [(0, "FULL DOCUMENT")]

    # Assign each line to a section and collect paragraph text
    def section_at_line(idx):
        name = "INTRODUCTION"
        for (j, n) in section_breaks:
            if idx >= j:
                name = n
        return name

    # Group consecutive lines by section, then merge into paragraphs and split sentences
    sections = {}  # section_name -> list of lines (raw)
    for i, line in enumerate(cleaned_lines):
        if is_section_header(line):
            continue  # don't add header as content
        sec = section_at_line(i)
        if sec not in sections:
            sections[sec] = []
        sections[sec].append(line)

    # Section names that are signature/firm blocks - drop from output
    skip_section_names = {'UEBLER LAW LLC', 'LAW LLC'}
    seen_sections = set()
    result_sections = []
    for sec_name in [s[1] for s in section_breaks]:
        if sec_name in seen_sections or sec_name not in sections:
            continue
        seen_sections.add(sec_name)
        if any(skip in sec_name for skip in ('LAW LLC', 'Attorney for', 'Signature', 'VERIFICATION')):
            continue
        if sec_name in skip_section_names:
            continue
        raw_lines = sections[sec_name]

        # Clean each line (strip paragraph numbers) and join into one block
        cleaned = [clean_line(line) for line in raw_lines if clean_line(line)]
        block = ' '.join(cleaned)

        # Split block into sentences
        all_sentences = []
        for sent in split_into_sentences(block):
            sent = ' '.join(sent.split()).strip()
            if not sent:
                continue
            if not sent.endswith(('.', '!', '?')):
                sent = sent + '.'
            if len(sent) < 15:
                continue
            if re.match(r'^[\d\.\s\-]+$', sent):
                continue
            all_sentences.append(sent)

        result_sections.append({"name": sec_name, "sentences": all_sentences})

    return {"file": os.path.basename(filepath), "sections": result_sections}


# --- LLM-based extraction (OpenAI) ---
# Requires: pip install openai  and  OPENAI_API_KEY in environment (or pass api_key=).

LLM_SYSTEM_PROMPT = """You are a legal document parser. Your task is to split a court complaint or similar legal document into sections and extract clean sentences for each section.

Rules:
1. Identify section headers (e.g. INTRODUCTION, JURISDICTION AND VENUE, PARTIES, FACTUAL ALLEGATIONS, FACTUAL BACKGROUND, FACTS, COUNT I, COUNT II, PRAYER FOR RELIEF, etc.). Headings may be in ALL CAPS, Title Case, or have typos.
2. For each section, output a list of clean sentences. Each sentence must:
   - End with a period (or ? or !).
   - Have no leading paragraph numbers (e.g. remove "1." or "17.").
   - Have no leading letter list markers (e.g. "A." or "B.").
   - Be a single, readable sentence (no line-break junk or extra spaces).
3. Ignore form pages (cover sheets, JS 44, designation forms, signature blocks, EEOC notices). Only include substantive complaint sections.
4. Output in PLAIN TEXT only (no JSON, no markdown, no code fences). Use exactly this format:

SECTION: SECTION_NAME_AS_IN_DOC
- First sentence of this section.
- Second sentence of this section.

SECTION: NEXT_SECTION_NAME
- First sentence of next section.
- Second sentence of next section.

Do not include any explanation before or after the sections. Only output the sections in this format."""


def _prepare_text_for_llm(filepath: str, max_chars: int | None = None) -> str:
    """Read file and return cleaned text for LLM (strip junk; no truncation)."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    text = text.replace("\f", "\n")
    lines = text.split("\n")
    kept = []
    for line in lines:
        s = line.strip()
        if not s:
            kept.append("")
            continue
        if re.match(r"^Source URL:\s*https?://", s, re.I):
            continue
        if re.match(r"^Title:\s*", s, re.I):
            continue
        if re.match(r"^Case\s+\d+[\:\-]\d+.*Document\s+\d+.*Filed.*Page\s+\d+\s+of", s, re.I):
            continue
        if re.match(r"^\s*\d+\s*$", s) and len(kept) > 0 and kept[-1].strip() == "":
            continue
        kept.append(line)
    return "\n".join(kept)


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks. No truncation: last chunk can be shorter."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def _merge_llm_section_lists(lists: list[list[dict]]) -> list[dict]:
    """Merge multiple section lists (from chunked LLM calls). Same section name -> combine sentences, dedupe."""
    by_name: dict[str, list[str]] = {}
    name_order: list[str] = []  # preserve first-seen order of section names

    def norm_name(n: str) -> str:
        return " ".join(n.strip().lower().split())

    def norm_sent(s: str) -> str:
        return " ".join(s.strip().split())

    for sec_list in lists:
        for sec in sec_list:
            name = (sec.get("name") or sec.get("section") or "UNNAMED").strip()
            sents = sec.get("sentences", sec.get("sentence", []))
            if not isinstance(sents, list):
                sents = [sents] if sents else []
            key = norm_name(name)
            if key not in by_name:
                name_order.append(key)
                by_name[key] = []
            seen = {norm_sent(x) for x in by_name[key]}
            for s in sents:
                s = " ".join(str(s).split()).strip()
                if len(s) < 10:
                    continue
                if norm_sent(s) in seen:
                    continue
                seen.add(norm_sent(s))
                by_name[key].append(s)

    # Use first-seen display name for each key (from first list)
    display_names: dict[str, str] = {}
    for sec_list in lists:
        for sec in sec_list:
            name = (sec.get("name") or sec.get("section") or "UNNAMED").strip()
            key = norm_name(name)
            if key not in display_names:
                display_names[key] = name

    result = []
    for key in name_order:
        result.append({"name": display_names.get(key, key), "sentences": by_name[key]})
    return result


def _parse_llm_sections_response(raw: str) -> list[dict]:
    """Parse LLM response (JSON or plain text sections) into list of {name, sentences}."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw)

    # If the model still returns JSON, support that path first.
    stripped = raw.lstrip()
    if stripped.startswith("{"):
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            # Try to recover by extracting the largest JSON object substring.
            start = stripped.find("{")
            end = stripped.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise
            data = json.loads(stripped[start : end + 1])
        sections = data.get("sections", [])
        if not isinstance(sections, list):
            raise ValueError("LLM response missing or invalid 'sections' list")
        result = []
        for sec in sections:
            name = sec.get("name") or sec.get("section") or "UNNAMED"
            if isinstance(name, list):
                name = name[0] if name else "UNNAMED"
            sents = sec.get("sentences", sec.get("sentence", []))
            if not isinstance(sents, list):
                sents = [sents] if sents else []
            cleaned = []
            for s in sents:
                s = " ".join(str(s).split()).strip()
                if len(s) < 10:
                    continue
                if not s.endswith((".", "!", "?")):
                    s = s + "."
                cleaned.append(s)
            result.append({"name": str(name).strip(), "sentences": cleaned})
        return result

    # Otherwise, parse the plain-text SECTION / bullet format.
    sections: list[dict] = []
    current_name: str | None = None
    current_sents: list[str] = []

    def flush_current() -> None:
        nonlocal current_name, current_sents
        if current_name and current_sents:
            sections.append({"name": current_name.strip(), "sentences": current_sents})
        current_name = None
        current_sents = []

    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        # New section header: "SECTION: Name"
        if re.match(r"^(SECTION|Section)\s*:", s):
            flush_current()
            name = s.split(":", 1)[1].strip() or "UNNAMED"
            current_name = name
            current_sents = []
            continue

        # Bullet sentence line: "- ..." or similar
        # Strip bullet markers and leading numbering.
        s = re.sub(r"^[-•*]\s*", "", s)
        s = re.sub(r"^\d+[\)\.\]]\s*", "", s)
        s = " ".join(s.split()).strip()
        if not s:
            continue
        if len(s) < 10:
            continue
        if not s.endswith((".", "!", "?")):
            s = s + "."
        if current_name is None:
            current_name = "UNNAMED"
            current_sents = []
        current_sents.append(s)

    flush_current()
    return sections


def extract_sections_and_sentences_llm(
    filepath: str,
    api_key: str | None = None,
    model: str = "gpt-4o-mini",
    max_chars: int = LLM_MAX_CHARS,
    chunk_size: int = LLM_CHUNK_SIZE,
    chunk_overlap: int = LLM_CHUNK_OVERLAP,
) -> dict:
    """
    Use an LLM to extract sections and clean sentences. For docs longer than max_chars,
    sends overlapping chunks and merges section lists (no truncation). Returns same
    structure as extract_sections_and_sentences.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("LLM extraction requires: pip install openai")

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY not set and no api_key provided")

    text = _prepare_text_for_llm(filepath)
    if not text.strip():
        return {"file": os.path.basename(filepath), "sections": []}

    # Support Groq / Ollama / any OpenAI-compatible API via OPENAI_BASE_URL.
    # Disable automatic retries so 429s etc. surface quickly and we can fall back.
    base_url = os.environ.get("OPENAI_BASE_URL")
    client_kwargs = {"api_key": key, "max_retries": 0}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    # Groq free/on-demand tiers can reject large prompts with a 413 (token limit / TPM enforcement).
    # Use smaller max_chars + chunk sizes when pointed at Groq, with env overrides.
    if base_url and "groq.com" in base_url.lower():
        groq_max_chars = int(os.environ.get("COURTLISTENER_GROQ_MAX_CHARS", "16000"))
        groq_chunk_size = int(os.environ.get("COURTLISTENER_GROQ_CHUNK_SIZE", "14000"))
        groq_chunk_overlap = int(os.environ.get("COURTLISTENER_GROQ_CHUNK_OVERLAP", "800"))
        max_chars = min(max_chars, groq_max_chars)
        chunk_size = min(chunk_size, groq_chunk_size)
        chunk_overlap = min(chunk_overlap, groq_chunk_overlap)

    if len(text) <= max_chars:
        # Single request
        if os.environ.get("COURTLISTENER_LLM_QUIET") != "1":
            print("    [LLM] 1 request", flush=True)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Extract sections and clean sentences from this court document:\n\n{text}"},
                ],
                temperature=0.1,
            )
            raw = response.choices[0].message.content.strip()
            result_sections = _parse_llm_sections_response(raw)
        except Exception as e:
            # If provider rejects prompt as too large, retry via chunking.
            msg = str(e)
            if "Request too large" in msg or "Error code: 413" in msg or "rate_limit_exceeded" in msg:
                chunks = _chunk_text(text, chunk_size, chunk_overlap)
                if os.environ.get("COURTLISTENER_LLM_QUIET") != "1":
                    print(f"    [LLM] retry as {len(chunks)} chunks (provider size limit)", flush=True)
                section_lists = []
                for i, chunk in enumerate(chunks):
                    part_label = f" (Part {i + 1} of {len(chunks)})" if len(chunks) > 1 else ""
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": LLM_SYSTEM_PROMPT},
                            {"role": "user", "content": f"Extract sections and clean sentences from this court document{part_label}:\n\n{chunk}"},
                        ],
                        temperature=0.1,
                    )
                    raw = response.choices[0].message.content.strip()
                    section_lists.append(_parse_llm_sections_response(raw))
                result_sections = _merge_llm_section_lists(section_lists)
            else:
                raise
    else:
        # Chunked: send overlapping chunks, merge section lists
        chunks = _chunk_text(text, chunk_size, chunk_overlap)
        if os.environ.get("COURTLISTENER_LLM_QUIET") != "1":
            print(f"    [LLM] {len(chunks)} chunks (long doc)", flush=True)
        section_lists = []
        for i, chunk in enumerate(chunks):
            part_label = f" (Part {i + 1} of {len(chunks)})" if len(chunks) > 1 else ""
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Extract sections and clean sentences from this court document{part_label}:\n\n{chunk}"},
                ],
                temperature=0.1,
            )
            raw = response.choices[0].message.content.strip()
            section_lists.append(_parse_llm_sections_response(raw))
        result_sections = _merge_llm_section_lists(section_lists)

    return {"file": os.path.basename(filepath), "sections": result_sections}


def _merge_rule_and_llm(rule_doc: dict, llm_doc: dict) -> dict:
    """Merge rule-based and LLM extraction: same section names get combined sentences (deduped)."""
    by_key: dict[str, tuple[str, list[str]]] = {}  # key -> (display_name, sentences)

    def norm_name(n: str) -> str:
        return " ".join((n or "").strip().lower().split())

    def norm_sent(s: str) -> str:
        return " ".join((s or "").strip().split())

    def add_sections(sections: list[dict]) -> None:
        for sec in sections:
            name = (sec.get("name") or "UNNAMED").strip()
            sents = sec.get("sentences", [])
            key = norm_name(name)
            if key not in by_key:
                by_key[key] = (name, [])
            seen = {norm_sent(x) for x in by_key[key][1]}
            for s in sents:
                if not s or len(s.strip()) < 10:
                    continue
                ns = norm_sent(s)
                if ns in seen:
                    continue
                seen.add(ns)
                by_key[key][1].append(s.strip())

    add_sections(rule_doc.get("sections", []))
    add_sections(llm_doc.get("sections", []))

    # Order: rule-based order first (no dupes), then LLM-only sections
    order = []
    seen = set()
    for s in rule_doc.get("sections", []):
        k = norm_name(s.get("name", ""))
        if k in by_key and k not in seen:
            order.append(k)
            seen.add(k)
    for k in by_key:
        if k not in seen:
            order.append(k)

    sections = [{"name": by_key[k][0], "sentences": by_key[k][1]} for k in order if by_key[k][1]]
    return {"file": rule_doc.get("file", ""), "sections": sections}


def _should_use_llm_fallback(rule_based_doc: dict) -> bool:
    """Return True if rule-based result looks incomplete (few sections or missing key sections)."""
    sections = rule_based_doc.get("sections", [])
    if len(sections) < 2:
        return True
    names_lower = {s.get("name", "").lower() for s in sections}
    # If we have at least one of these, rule-based is probably ok
    key = {"introduction", "parties", "jurisdiction", "venue", "factual", "facts", "background", "prayer", "count"}
    has_key = any(any(k in n for k in key) for n in names_lower)
    if not has_key:
        return True
    total_sentences = sum(len(s.get("sentences", [])) for s in sections)
    if total_sentences < 5:
        return True
    return False


def format_output(doc: dict, format_type: str = "text") -> str:
    """Format extracted doc as text or markdown."""
    lines = []
    lines.append(f"# {doc['file']}\n")
    for sec in doc["sections"]:
        if not sec["sentences"]:
            continue
        lines.append(f"\n## {sec['name']}\n")
        for s in sec["sentences"]:
            lines.append(s)
            if not s.endswith('\n'):
                lines.append("\n")
    return "".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract sections and clean sentences from CourtListener case docs")
    parser.add_argument("--input-dir", default="courtlistener_cases", help="Input directory with .txt files")
    parser.add_argument("--output-dir", default="courtlistener_processed", help="Output directory for processed files")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N files (0 = all)")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    parser.add_argument("--use-llm", action="store_true", help="Use OpenAI LLM for section extraction (needs OPENAI_API_KEY)")
    parser.add_argument("--llm-fallback", action="store_true", help="Use rule-based first; if result looks incomplete, retry with LLM")
    parser.add_argument("--use-both", action="store_true", help="Run both rule-based and LLM; merge sections and sentences (deduplicated)")
    parser.add_argument("--llm-model", default="gpt-4o-mini", help="OpenAI model for LLM extraction (default: gpt-4o-mini)")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    input_dir = base / args.input_dir
    output_dir = base / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(input_dir.glob("*.txt"))
    if args.limit:
        txt_files = txt_files[: args.limit]

    use_llm_direct = args.use_llm and not args.llm_fallback

    # Print mode so user can see if LLM is used
    if args.use_both:
        print("Mode: rule-based + LLM (merge)", flush=True)
    elif use_llm_direct:
        print("Mode: LLM only", flush=True)
    elif args.llm_fallback:
        print("Mode: rule-based with LLM fallback if incomplete", flush=True)
    else:
        print("Mode: rule-based only (no LLM)", flush=True)
    print(f"Processing {len(txt_files)} file(s)...\n", flush=True)

    for path in txt_files:
        try:
            if args.use_both:
                rule_doc = extract_sections_and_sentences(str(path))
                try:
                    llm_doc = extract_sections_and_sentences_llm(str(path), model=args.llm_model)
                    doc = _merge_rule_and_llm(rule_doc, llm_doc)
                    print(f"  [merged rule+LLM]", flush=True)
                except Exception as llm_err:
                    print(f"  (LLM failed: {llm_err}; using rule-based only)", flush=True)
                    doc = rule_doc
            elif use_llm_direct:
                doc = extract_sections_and_sentences_llm(str(path), model=args.llm_model)
            else:
                doc = extract_sections_and_sentences(str(path))
                if args.llm_fallback and _should_use_llm_fallback(doc):
                    try:
                        doc = extract_sections_and_sentences_llm(str(path), model=args.llm_model)
                        print(f"  [LLM fallback used]", flush=True)
                    except Exception as llm_err:
                        print(f"  (LLM fallback failed: {llm_err}; keeping rule-based result)", flush=True)
            out_name = path.stem + "_processed.txt"
            out_path = output_dir / out_name
            if args.format == "json":
                with open(out_path.with_suffix(".json"), "w", encoding="utf-8") as f:
                    json.dump(doc, f, indent=2)
            else:
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(format_output(doc))
            print(f"Processed: {path.name} -> {out_name}")
        except Exception as e:
            print(f"Error processing {path.name}: {e}")

    print(f"Done. Output in {output_dir}")


if __name__ == "__main__":
    main()
