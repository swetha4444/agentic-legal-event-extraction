"""
Dataset Preprocessing Module
Cleans raw legal complaint text from CourtListener JSONL data.
Designed to produce plain, human-readable text for fact annotation.
"""

import re
from typing import List, Dict, Tuple, Optional


class TextCleaner:
    """
    Cleans raw court complaint text by removing noise and normalizing formatting.
    Goal: produce plain readable text suitable for human annotation (fact / not-fact).
    Each method is a discrete cleaning step that can be toggled on/off.
    """

    # --- Section header patterns (used for detection & structuring) ---
    SECTION_HEADERS = [
        r"(?:I+\.?\s+)?BACKGROUND",
        r"(?:I+\.?\s+)?FACTS",
        r"(?:I+\.?\s+)?FACTUAL\s+ALLEGATIONS",
        r"(?:I+\.?\s+)?RELEVANT\s+FACTS",
        r"(?:I+\.?\s+)?PROCEDURAL\s+HISTORY",
        r"(?:I+\.?\s+)?JURISDICTION",
        r"(?:I+\.?\s+)?VENUE",
        r"(?:I+\.?\s+)?PARTIES",
        r"(?:I+\.?\s+)?COUNT\s+[IVX0-9]+",
        r"(?:I+\.?\s+)?DISCUSSION",
        r"(?:I+\.?\s+)?ANALYSIS",
        r"(?:I+\.?\s+)?CONCLUSION",
        r"(?:I+\.?\s+)?LEGAL\s+STANDARD",
        r"(?:I+\.?\s+)?PRAYER\s+FOR\s+RELIEF",
        r"(?:I+\.?\s+)?REQUEST\s+FOR\s+RELIEF",
        r"MEMORANDUM",
        r"OPINION",
        r"DECISION",
        r"COMPLAINT",
        r"(?:I+\.?\s+)?PRELIMINARY\s+STATEMENT",
        r"(?:I+\.?\s+)?INTRODUCTION",
        r"(?:I+\.?\s+)?CAUSES?\s+OF\s+ACTION",
        r"(?:I+\.?\s+)?JURY\s+DEMAND",
        r"(?:I+\.?\s+)?JURISDICTION\s+AND\s+VENUE",
    ]

    def clean(self, text: str) -> str:
        """
        Apply all cleaning steps in order.
        Returns fully cleaned, human-readable text.
        """
        # Phase 1: Decode and remove structural junk
        text = self.decode_unicode_escapes(text)
        text = self.remove_civil_cover_sheet(text)
        text = self.remove_court_filing_headers(text)
        text = self.remove_spaced_ocr_text(text)

        # Phase 2: Remove non-narrative content
        text = self.remove_footnote_blocks(text)
        text = self.remove_separator_lines(text)
        text = self.remove_page_numbers(text)
        text = self.clean_footnote_markers(text)
        text = self.remove_url_lines(text)
        text = self.remove_signature_blocks(text)
        text = self.remove_duplicate_captions(text)

        # Phase 3: Normalize to plain readable text
        text = self.strip_legal_symbols(text)
        # Note: We NO LONGER normalize whitespace here.
        # We need the newlines preserved so that strip_caption can reliably find 
        # the start of the first numbered paragraph ("\n1. "). 
        # Whitespace normalization (flattenting) now happens at the very end in preprocess_case.
        return text.strip()

    # =========================================================================
    # NEW cleaning steps
    # =========================================================================

    @staticmethod
    def decode_unicode_escapes(text: str) -> str:
        """
        Convert Unicode escape sequences to their actual characters,
        then replace smart quotes/dashes with plain ASCII equivalents.
        """
        # Handle raw \\uXXXX sequences that weren't decoded by JSON parser
        def _replace_unicode(match):
            try:
                return chr(int(match.group(1), 16))
            except (ValueError, OverflowError):
                return match.group(0)

        text = re.sub(r'\\u([0-9a-fA-F]{4})', _replace_unicode, text)

        # Smart quotes -> straight quotes
        text = text.replace('\u201c', '"').replace('\u201d', '"')  # " "
        text = text.replace('\u2018', "'").replace('\u2019', "'")  # ' '
        # Em/en dashes -> plain dash
        text = text.replace('\u2014', '-').replace('\u2013', '-')
        # Bullet / misc
        text = text.replace('\u2022', '-')  # bullet
        text = text.replace('\u2716', '')   # heavy X mark (cover sheet checkboxes)
        text = text.replace('\u2717', '')   # ballot X
        text = text.replace('\u2713', '')   # check mark
        text = text.replace('\u25cf', '')   # black circle
        text = text.replace('\u25cb', '')   # white circle
        text = text.replace('\u00b7', '-')  # middle dot

        return text

    @staticmethod
    def remove_court_filing_headers(text: str) -> str:
        """
        Remove court filing header lines in all known formats.
        """
        # Format: Case 2:26-cv-01059 Document 1 Filed 02/19/26 Page 6 of 37
        text = re.sub(
            r'(?m)^.*Case\s+\d+:\d+-\w+-\d+.*(?:Page|Pag\.?)\s+\d+\s+of\s+\d+.*$',
            '', text
        )
        # Format: 2:26-cv-00666-DCN   Date Filed 02/17/26   Entry Number 1   Page 1 of 5
        text = re.sub(
            r'(?m)^.*\d+:\d+-\w+-\d+.*Date\s+Filed.*Page\s+\d+\s+of\s+\d+.*$',
            '', text
        )
        # Format: Entered on FLSD Docket 02/19/2026 Page 1 of 25
        text = re.sub(
            r'(?m)^.*Entered\s+on\s+\w+\s+Docket.*Page\s+\d+\s+of\s+\d+.*$',
            '', text
        )
        # Standalone docket number lines: 2:26-cv-00666-DCN
        text = re.sub(
            r'(?m)^\s*\d+:\d+-\w+-\d+(?:-[A-Z]+)?\s*$',
            '', text
        )
        # "Document X Filed MM/DD/YY" standalone
        text = re.sub(
            r'(?m)^.*Document\s+\d+\s+Filed\s+\d{2}/\d{2}/\d{2,4}.*$',
            '', text
        )
        # Format: Doc. #: 1 Filed: 02/17/26 Page: 1 of 13 PageID #: 1
        # Also: Doc # 1 Filed: 02/16/26 Page 1 of 5 - Page ID # 1
        text = re.sub(
            r'(?m)^.*Doc\.?\s*#:?\s*\d+\s+Filed:?\s*\d{2}/\d{2}/\d{2,4}.*$',
            '', text
        )
        # Master catch-all: any line with "Filed" + date + "Page" + "of" (covers all header variants)
        text = re.sub(
            r'(?m)^.*Filed:?\s*\d{2}/\d{2}/\d{2,4}.*Page:?\s+\d+\s+of\s+\d+.*$',
            '', text
        )
        # Any remaining "Page X of Y" standalone lines
        text = re.sub(
            r'(?m)^\s*Page:?\s+\d+\s+of\s+\d+.*$',
            '', text
        )
        # Catch-all: any line containing "PageID #" or "Page ID #"
        text = re.sub(
            r'(?m)^.*Page\s*ID\s*#.*$',
            '', text
        )
        return text

    @staticmethod
    def remove_civil_cover_sheet(text: str) -> str:
        """
        Remove the JS 44 civil cover sheet that precedes actual complaint text.
        Detects the form by its distinctive spaced-out characters or header.
        """
        # Pattern 1: Detect JS 44 header (spaced or normal)
        js44_pattern = re.compile(
            r'J\s*S\s*4\s*4.*?(?=(?:UNITED\s+STATES\s+DISTRICT\s+COURT|'
            r'IN\s+THE\s+UNITED\s+STATES|COMPLAINT|PLAINTIFF))',
            re.DOTALL | re.IGNORECASE
        )
        text = js44_pattern.sub('', text)

        # Pattern 2: Cover sheet form sections (all caps spaced out)
        # e.g., "C O N T R A C T   T O R T S   F O R F E I T U R E"
        text = re.sub(
            r'(?m)^[A-Z\s/&()]{30,}$',
            '',
            text
        )

        return text

    @staticmethod
    def remove_spaced_ocr_text(text: str) -> str:
        """
        Remove lines where characters are separated by spaces (bad OCR / form data).
        E.g., 'T h e J S 4 4 ci vil c o v er s h e et'
        Heuristic: if >40% of non-space chars are single chars separated by spaces.
        """
        cleaned_lines = []
        for line in text.split('\n'):
            stripped = line.strip()
            if not stripped:
                cleaned_lines.append(line)
                continue

            # Count single-char tokens
            tokens = stripped.split()
            if len(tokens) < 5:
                cleaned_lines.append(line)
                continue

            single_char_count = sum(1 for t in tokens if len(t) == 1)
            ratio = single_char_count / len(tokens)

            # If more than 50% of tokens are single characters, it's spaced OCR
            if ratio > 0.50:
                continue  # drop the line
            else:
                cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    @staticmethod
    def remove_footnote_blocks(text: str) -> str:
        """
        Remove multi-line footnote blocks that contain citations, URLs, and 'Id.' references.
        These appear as numbered footnotes and inline citation noise.
        """
        # 1. Identify and strip blocks starting with horizontal dividers (e.g., __________)
        # These are strong signals of the transition from body text to footer.
        lines = text.split('\n')
        cleaned_lines = []
        in_footer_zone = False
        
        for line in lines:
            stripped = line.strip()
            
            # Detect divider: 3 or more underscores, dashes, or asterisks
            if re.match(r'^[ \t]*[_\-\*]{3,}[ \t]*$', line):
                in_footer_zone = True
                continue # Drop the divider line
                
            if in_footer_zone:
                # If we are in the footer zone, we drop lines that look like footnotes
                # until we hit a line that looks like a new paragraph (e.g. starts with "1- " or a heading)
                # But here, we just drop until the end of the text/section or until we see 
                # something that DEFINITELY isn't a footnote.
                if re.match(r'^\s*(\d{1,2}|Id\.|See|Cf\.|Office|THE|RAND|http|www\.)', stripped):
                    continue # Drop footnote content
                # If it's a very short line or empty, keep dropping
                if not stripped:
                    continue
                # If we see a line starting with a capital letter and it's long, 
                # maybe we left the footer? (Unlikely in most complaints before a page break)
                # For now, let's just drop lines that look like citations.
            
            cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)

        # 2. Pattern-based removal for cases where dividers are missing
        # Remove lines that are just "Id." or "Id. at X." or "Id. at X-Y."
        text = re.sub(r'(?m)^\s*(?:\d{1,2}\s+)?Id\.(?:\s+at\s+[\d\-]+\.?)?\s*$', '', text)

        # Remove numbered footnote blocks (e.g., "3 THE DAILY BEAST, VisionQuest...")
        text = re.sub(
            r'(?m)^\s*\d{1,2}\s+(?:THE\s|See|Cf\.|In\s+the\s+Matter|Office\s+of|'
            r'Letter\s+From|NEW\s|SAN\s|WNEP|VENANGO|PENNSYLVANIA\\s+RECORD|'
            r'RAND\s|U\\.S\\.\\s+Department|http|www\.).*$',
            '', text
        )

        # Remove lines that are primarily URLs
        text = re.sub(r'(?m)^.*https?://\S{40,}.*$', '', text)

        return text

    @staticmethod
    def remove_url_lines(text: str) -> str:
        """Remove lines that contain URLs (leftover citation/footnote noise)."""
        text = re.sub(r'(?m)^.*https?://\S+.*$', '', text)
        # Also remove lines that are just www. links
        text = re.sub(r'(?m)^.*www\.\S+.*$', '', text)
        return text

    @staticmethod
    def remove_signature_blocks(text: str) -> str:
        """
        Remove signature blocks at the end of the document:
        - 'Respectfully submitted' and everything after
        - '/s/ Name' signatures
        - Attorney info blocks (firm, address, phone)
        - 'Dated:' lines
        """
        # Cut everything from "Respectfully submitted" to end
        match = re.search(r'(?i)\n\s*Respectfully\s+submitted', text)
        if match:
            text = text[:match.start()]

        # Also try "Attorneys for Plaintiff" as a cutoff if near end
        lines = text.split('\n')
        total = len(lines)
        for i in range(total - 1, max(total - 30, 0), -1):
            if re.search(r'(?i)Attorneys?\s+for\s+(?:the\s+)?Plaintif', lines[i]):
                text = '\n'.join(lines[:i])
                break

        return text

    @staticmethod
    def remove_duplicate_captions(text: str) -> str:
        """
        Remove repeated caption blocks that appear between pages:
        - Repeated firm names and addresses (LEVY KONIGSBERG, LLP...)
        - Repeated party vs. party blocks
        - Repeated court name blocks
        """
        # Remove repeated firm headers (appear on multiple pages)
        text = re.sub(
            r'(?m)(?:^.*(?:LLP|LLC|P\.?C\.?|PLLC|ESQ\.?).*\n'
            r'(?:^.*(?:Bar\s+No\.|Esq\.).*\n)*'
            r'(?:^.*(?:Penn\s+Center|Boulevard|Blvd|Suite\s+\d|Street|Ave).*\n)*'
            r'(?:^.*(?:\d{3}[-)]\d{3,4}|\d{5}).*\n)*'
            r'(?:^.*(?:Pro\s+Hac|Motion\s+Forthcoming).*\n)*)',
            '', text,
            flags=re.IGNORECASE
        )

        # Remove repeated "UNITED STATES DISTRICT COURT" blocks
        # Keep only the first occurrence
        parts = re.split(r'(UNITED\s+STATES\s+DISTRICT\s+COURT)', text)
        if len(parts) > 3:
            # Keep first occurrence, remove the rest
            result = parts[0] + parts[1] + parts[2]
            for i in range(3, len(parts), 2):
                if i + 1 < len(parts):
                    # Skip the header, keep the text after removing
                    # a few lines of repeated court/party info
                    after = parts[i + 1] if i + 1 < len(parts) else ''
                    # Remove up to the next section/paragraph number
                    cleaned_after = re.sub(
                        r'^.*?(?=\d+\.\s+|\n[A-Z]{3,})',
                        '', after, count=1, flags=re.DOTALL
                    )
                    result += cleaned_after
                else:
                    result += parts[i]
            text = result

        # Remove "Plaintiff v. Defendant" blocks that repeat
        text = re.sub(
            r"(?m)^\s*[A-Z][A-Z\s,.]+,?\s*\n\s*Plaintiff\s*\n\s*v\.\s*\n"
            r"\s*[A-Z][A-Z\s,.]+,?\s*\n\s*Defendant\s*$",
            '', text
        )

        return text

    @staticmethod
    def strip_legal_symbols(text: str) -> str:
        """
        Remove or replace special legal symbols to produce plain readable text.
        The goal is text that a human annotator can read without confusion.
        """
        # Section symbol
        text = text.replace('\u00a7', 'Section')
        text = re.sub(r'Section\s+Section', 'Section', text)  # avoid double

        # Paragraph symbol
        text = text.replace('\u00b6', '')

        # Dagger / double dagger
        text = text.replace('\u2020', '').replace('\u2021', '')

        # Copyright, trademark, registered
        text = text.replace('\u00a9', '').replace('\u2122', '').replace('\u00ae', '')

        # Ellipsis
        text = text.replace('\u2026', '...')

        # Non-breaking space
        text = text.replace('\u00a0', ' ')

        # Any remaining non-ASCII that isn't basic punctuation, letters, or digits
        # Keep common accented chars but remove unusual symbols
        text = re.sub(r'[^\x00-\x7F\u00c0-\u00ff]', '', text)

        return text

    # =========================================================================
    # Original cleaning steps (preserved)
    # =========================================================================

    @staticmethod
    def remove_separator_lines(text: str) -> str:
        """Remove visual separator lines: _____, -----, =====, ....."""
        text = re.sub(r"(?m)^\s*[_]{5,}\s*$", "", text)
        text = re.sub(r"(?m)^\s*[-]{5,}\s*$", "", text)
        text = re.sub(r"(?m)^\s*[=]{5,}\s*$", "", text)
        text = re.sub(r"(?m)^\s*[.]{5,}\s*$", "", text)
        # Also remove dot leaders within lines (e.g., "Table of Contents ........... v")
        text = re.sub(r'[.]{4,}', ' ', text)
        # Remove repeated colons from caption column formatting (: : : : :)
        text = re.sub(r'(?m)^\s*(?::\s*){2,}$', '', text)
        # Also remove isolated * or ) that come from caption formatting
        text = re.sub(r'(?m)^\s*[*)\]]+\s*$', '', text)
        return text

    @staticmethod
    def remove_page_numbers(text: str) -> str:
        """Remove standalone page numbers (e.g., '- 3 -' or just '12' on a line)."""
        text = re.sub(r"(?m)^\s*-?\s*\d{1,3}\s*-?\s*$", "", text)
        return text

    @staticmethod
    def clean_footnote_markers(text: str) -> str:
        """
        Remove inline footnote numbers that appear mid-text.
        E.g., 'proceeding.3 Although' -> 'proceeding. Although'
        Or 'kids15' -> 'kids' (if it looks like a footnote link)
        """
        # Remove markers after punctuation followed by space and Capital
        text = re.sub(r"(?<=[.!?])\s*\d{1,2}(?=\s+[A-Z])", "", text)
        
        # Remove markers attached to words (like "kids15")
        # We look for a lowercase letter followed by 1-2 digits, 
        # but NOT if it's followed by more digits (like part of a year or zip code)
        text = re.sub(r"([a-z])\d{1,2}(?!\d)", r"\1", text)
        
        return text

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """
        Flatten text into a single continuous string:
        - Replace all newlines with spaces
        - Collapse multiple spaces into one
        - The dashboard will handle line breaks for display.
        """
        # Replace all newlines with spaces
        text = text.replace('\n', ' ')
        # Collapse multiple spaces into one
        text = re.sub(r'\s{2,}', ' ', text)
        return text

    @staticmethod
    def normalize_citations(text: str) -> str:
        """Normalize spacing in common legal citations."""
        text = re.sub(r"Pa\.\s*C\.\s*S\.", "Pa.C.S.", text)
        text = re.sub(r"Pa\.\s*Super\.", "Pa.Super.", text)
        return text

    @staticmethod
    def reformat_paragraph_numbers(text: str) -> str:
        """
        Change paragraph numbers from "1. " to "1- " at the start of lines.
        This prevents the dashboard from splitting the number and the sentence.
        """
        return re.sub(r'(?m)^([ \t]*\d+)\.', r'\1-', text)

    @staticmethod
    def remove_mid_document_subtitles(text: str) -> str:
        """
        Remove standalone all-caps lines that serve as section headers 
        (e.g., "FACTS", "PARTIES", "JURISDICTION AND VENUE").
        These usually appear between numbered paragraphs.
        """
        # Match lines that are all caps (allowing for spaces and ampersands) 
        # and are not immediately preceded by a number in the same line.
        # We also check that it's a standalone line.
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            # If line is all caps, contains only letters/spaces/&/and, and is reasonably short
            if stripped and stripped.isupper() and re.match(r'^[A-Z\s&]+$', stripped) and len(stripped) < 100:
                # If it doesn't look like part of a numbered list (e.g. "1. FACTS")
                if not re.match(r'^\d+[-.]', stripped):
                    continue # Skip this subtitle line
            cleaned_lines.append(line)
        return '\n'.join(cleaned_lines)

    # =========================================================================
    # Section detection (structuring)
    # =========================================================================

    def detect_sections(self, text: str) -> List[Dict[str, str]]:
        """
        Detect major section headers and split text into labeled sections.
        Returns list of {"header": "...", "text": "..."} dicts.
        """
        header_positions: List[Tuple[int, int, str]] = []

        for pattern in self.SECTION_HEADERS:
            for match in re.finditer(rf"^\s*{pattern}\s*$", text, re.MULTILINE | re.IGNORECASE):
                label = match.group().strip().upper()
                header_positions.append((match.start(), match.end(), label))

        if not header_positions:
            return [{"header": "BODY", "text": text.strip()}]

        header_positions.sort(key=lambda x: x[0])
        sections = []

        preamble = text[: header_positions[0][0]].strip()
        if preamble:
            sections.append({"header": "PREAMBLE", "text": preamble})

        for i, (start, end, label) in enumerate(header_positions):
            if i + 1 < len(header_positions):
                section_text = text[end : header_positions[i + 1][0]].strip()
            else:
                section_text = text[end:].strip()
            sections.append({"header": label, "text": section_text})

        return sections

    def strip_caption(self, text: str) -> str:
        """
        Remove the case caption (party names, court header) that appears
        before the first meaningful section header or the first numbered paragraph.
        """
        # First, try to find the first numbered paragraph (specifically starting with number 1, optionally indented)
        # We look for a newline, optional indentation, the number 1, a period, spaces, and an uppercase letter.
        match = re.search(r'(?m)^[ \t]*1\.\s+[A-Z]', text)
        if match:
             return text[match.start():].strip()

        # Fallback to the old method if no numbered paragraphs are found
        for pattern in [
            r"(?i)\n\s*memorandum\s*\n",
            r"(?i)\n\s*opinion\s*\n",
            r"(?i)\n\s*decision\s*\n",
            r"(?i)\n\s*complaint\s*\n",
            r"(?i)\n\s*civil\s+action\s+complaint\s*\n",
            r"(?i)\n\s*plaintiff's\s+complaint\s*\n",
            r"(?i)\n\s*preliminary\s+statement\s*\n",
            r"(?i)\n\s*introduction\s*\n",
        ]:
            match = re.search(pattern, text)
            if match:
                return text[match.start():].strip()
        return text


def preprocess_case(case_data: dict) -> dict:
    """
    Preprocess a single case record from the JSONL dataset.
    Works on any dataset size - just pass one case at a time.

    Args:
        case_data: dict with at least 'document_text' field

    Returns:
        dict with cleaned text, sections, and metadata
    """
    cleaner = TextCleaner()
    raw_text = case_data.get("document_text", "")

    if not raw_text:
        return {
            "case_id": case_data.get("case_id"),
            "case_name": case_data.get("case_name"),
            "clean_text": "",
            "sections": [],
            "metadata": {"original_length": 0, "cleaned_length": 0, "reduction_pct": "0%", "sections_found": []},
        }

    # 1. Clean the text
    clean_text = cleaner.clean(raw_text)

    # 2. Strip caption using the regex for numbered paragraphs
    body_text = cleaner.strip_caption(clean_text)

    # 3. Apply final refinements: remove mid-doc subtitles and reformat numbers
    body_text = cleaner.remove_mid_document_subtitles(body_text)
    body_text = cleaner.reformat_paragraph_numbers(body_text)

    # 4. Detect sections on the text BEFORE we flatten the newlines
    sections = cleaner.detect_sections(body_text)

    # 5. NOW normalize whitespace (flatten) for the main body and all sections
    body_text = cleaner.normalize_whitespace(body_text)
    for section in sections:
         section["text"] = cleaner.normalize_whitespace(section["text"])

    # 6. Build metadata
    orig_len = len(raw_text)
    clean_len = len(body_text)
    reduction = round((1 - clean_len / orig_len) * 100) if orig_len else 0

    return {
        "case_id": case_data.get("case_id"),
        "case_name": case_data.get("case_name"),
        "date_filed": case_data.get("date_filed"),
        "clean_text": body_text,
        "sections": sections,
        "metadata": {
            "original_length": orig_len,
            "cleaned_length": clean_len,
            "reduction_pct": f"{reduction}%",
            "sections_found": [s["header"] for s in sections],
        },
    }
