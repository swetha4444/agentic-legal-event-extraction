"""
Dataset Preprocessing Module
Cleans raw legal opinion text from CourtListener JSONL data.
Designed to scale from 10-case sample to full dataset.
"""

import re
from typing import List, Dict, Tuple, Optional


class TextCleaner:
    """
    Cleans raw court opinion text by removing noise and normalizing formatting.
    Each method is a discrete cleaning step that can be toggled on/off.
    """

    # --- Section header patterns (used for detection & structuring) ---
    SECTION_HEADERS = [
        r"(?i)(?:I+\.?\s+)?BACKGROUND",
        r"(?i)(?:I+\.?\s+)?FACTS",
        r"(?i)(?:I+\.?\s+)?FACTUAL\s+ALLEGATIONS", # Common in Complaints
        r"(?i)(?:I+\.?\s+)?RELEVANT\s+FACTS",
        r"(?i)(?:I+\.?\s+)?PROCEDURAL\s+HISTORY",
        r"(?i)(?:I+\.?\s+)?JURISDICTION", # Common in Complaints
        r"(?i)(?:I+\.?\s+)?VENUE", # Common in Complaints
        r"(?i)(?:I+\.?\s+)?PARTIES", # Common in Complaints
        r"(?i)(?:I+\.?\s+)?COUNT\s+[IVX0-9]+", # Common in Complaints (Count I, Count 1, etc)
        r"(?i)(?:I+\.?\s+)?DISCUSSION",
        r"(?i)(?:I+\.?\s+)?ANALYSIS",
        r"(?i)(?:I+\.?\s+)?CONCLUSION",
        r"(?i)(?:I+\.?\s+)?LEGAL\s+STANDARD",
        r"(?i)(?:I+\.?\s+)?PRAYER\s+FOR\s+RELIEF", # Common in Complaints
        r"(?i)(?:I+\.?\s+)?REQUEST\s+FOR\s+RELIEF", # Common in Complaints
        r"(?i)MEMORANDUM",
        r"(?i)OPINION",
        r"(?i)DECISION",
        r"(?i)COMPLAINT", # Sometimes a header itself
    ]

    def clean(self, text: str) -> str:
        """
        Apply all cleaning steps in order.
        Returns the fully cleaned text.
        """
        text = self.remove_separator_lines(text)
        text = self.remove_page_numbers(text)
        text = self.clean_footnote_markers(text)
        text = self.normalize_whitespace(text)
        text = self.normalize_citations(text)
        return text.strip()

    # -------------------------------------------------------------------------
    # Individual cleaning steps
    # -------------------------------------------------------------------------

    @staticmethod
    def remove_separator_lines(text: str) -> str:
        """Remove visual separator lines: _____, -----, ====="""
        # Remove lines that are mostly underscores, dashes, or equals (5+ chars)
        text = re.sub(r"(?m)^\s*[_]{5,}\s*$", "", text)
        text = re.sub(r"(?m)^\s*[-]{5,}\s*$", "", text)
        text = re.sub(r"(?m)^\s*[=]{5,}\s*$", "", text)
        return text

    @staticmethod
    def remove_page_numbers(text: str) -> str:
        """Remove standalone page numbers (e.g., '- 3 -' or just '12' on a line)."""
        # Pattern: optional dash, whitespace, 1-3 digits, whitespace, optional dash
        text = re.sub(r"(?m)^\s*-?\s*\d{1,3}\s*-?\s*$", "", text)
        return text

    @staticmethod
    def clean_footnote_markers(text: str) -> str:
        """
        Remove inline footnote numbers that appear mid-text.
        E.g., 'proceeding.3 Although' -> 'proceeding. Although'
        Careful not to remove numbers that are part of citations like '§ 5328'.
        """
        # Match a digit after sentence-ending punctuation followed by a space and uppercase
        text = re.sub(r"(?<=[.!?])\s*\d{1,2}(?=\s+[A-Z])", "", text)
        return text

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """
        Collapse excessive whitespace:
        - 3+ consecutive newlines -> 2 newlines
        - 2+ consecutive spaces -> 1 space
        - Leading/trailing whitespace per line
        """
        # Collapse multiple blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Collapse multiple spaces (but not newlines)
        text = re.sub(r"[^\S\n]{2,}", " ", text)
        # Strip each line
        lines = [line.strip() for line in text.split("\n")]
        return "\n".join(lines)

    @staticmethod
    def normalize_citations(text: str) -> str:
        """Normalize spacing in common legal citations."""
        # Normalize "Pa. C.S." variations -> "Pa.C.S."
        text = re.sub(r"Pa\.\s*C\.\s*S\.", "Pa.C.S.", text)
        # Normalize "Pa. Super." -> "Pa.Super."
        text = re.sub(r"Pa\.\s*Super\.", "Pa.Super.", text)
        # Ensure space after § if missing
        text = re.sub(r"§(\d)", r"§ \1", text)
        return text

    # -------------------------------------------------------------------------
    # Section detection (structuring)
    # -------------------------------------------------------------------------

    def detect_sections(self, text: str) -> List[Dict[str, str]]:
        """
        Detect major section headers and split text into labeled sections.
        Returns list of {"header": "...", "text": "..."} dicts.
        """
        # Find all section header positions
        header_positions: List[Tuple[int, int, str]] = []  # (start, end, label)

        for pattern in self.SECTION_HEADERS:
            for match in re.finditer(rf"(?m)^\s*{pattern}\s*$", text):
                label = match.group().strip().upper()
                header_positions.append((match.start(), match.end(), label))

        if not header_positions:
            # No sections found — return whole text as single block
            return [{"header": "BODY", "text": text.strip()}]

        # Sort by position in document
        header_positions.sort(key=lambda x: x[0])

        sections = []

        # Text before first header
        preamble = text[: header_positions[0][0]].strip()
        if preamble:
            sections.append({"header": "PREAMBLE", "text": preamble})

        # Each section runs from its header end to the next header start
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
        before the first meaningful section header.
        """
        for pattern in [
            r"(?i)\n\s*memorandum\s*\n",
            r"(?i)\n\s*opinion\s*\n",
            r"(?i)\n\s*decision\s*\n",
            r"(?i)\n\s*complaint\s*\n",
            r"(?i)\n\s*civil\s+action\s+complaint\s*\n",
        ]:
            match = re.search(pattern, text)
            if match:
                return text[match.start():].strip()
        return text


def preprocess_case(case_data: dict) -> dict:
    """
    Preprocess a single case record from the JSONL dataset.
    Works on any dataset size — just pass one case at a time.
    
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
            "metadata": {"original_length": 0, "cleaned_length": 0, "reduction_pct": "0%"},
        }

    # 1. Clean the text
    clean_text = cleaner.clean(raw_text)

    # 2. Strip caption
    body_text = cleaner.strip_caption(clean_text)

    # 3. Detect sections
    sections = cleaner.detect_sections(body_text)

    # 4. Build metadata
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
