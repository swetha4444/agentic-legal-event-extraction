"""
Chunking for event extraction pipeline.
First step: split document text into chunks (by chars, paragraphs, or sentences)
so downstream steps (facts, events, EKG) can run on bounded segments.
"""
from dataclasses import dataclass, field
from typing import List, Optional
import re


@dataclass
class Chunk:
    """A single chunk of document text with metadata for downstream steps."""

    chunk_id: str
    text: str
    start_char: int
    end_char: int
    start_sentence_id: Optional[int] = None  # 0-based index of first sentence in doc, if known
    end_sentence_id: Optional[int] = None
    section_label: Optional[str] = None  # e.g. "FACTS", "BACKGROUND"
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.text)


def chunk_by_characters(
    text: str,
    chunk_size: int = 12_000,
    overlap: int = 200,
    chunk_id_prefix: str = "chunk",
) -> List[Chunk]:
    """
    Split text into overlapping character-based chunks.
    No truncation: last chunk can be shorter.
    """
    text = text or ""
    if not text.strip():
        return []
    if len(text) <= chunk_size:
        return [
            Chunk(
                chunk_id=f"{chunk_id_prefix}_0",
                text=text,
                start_char=0,
                end_char=len(text),
            )
        ]
    chunks: List[Chunk] = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        segment = text[start:end]
        if segment.strip():
            chunks.append(
                Chunk(
                    chunk_id=f"{chunk_id_prefix}_{idx}",
                    text=segment,
                    start_char=start,
                    end_char=end,
                    metadata={"chunk_size": chunk_size, "overlap": overlap},
                )
            )
            idx += 1
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def chunk_by_paragraphs(
    text: str,
    max_chars_per_chunk: Optional[int] = None,
    chunk_id_prefix: str = "chunk",
) -> List[Chunk]:
    """
    Split text by double newlines (paragraphs). Optionally merge small
    paragraphs so no chunk exceeds max_chars_per_chunk.
    """
    text = text or ""
    if not text.strip():
        return []
    raw_paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not raw_paras:
        return [
            Chunk(
                chunk_id=f"{chunk_id_prefix}_0",
                text=text.strip(),
                start_char=0,
                end_char=len(text),
            )
        ]

    if max_chars_per_chunk is None:
        return [
            Chunk(
                chunk_id=f"{chunk_id_prefix}_{i}",
                text=p,
                start_char=text.find(p),
                end_char=text.find(p) + len(p),
            )
            for i, p in enumerate(raw_paras)
        ]

    chunks: List[Chunk] = []
    current: List[str] = []
    current_len = 0
    idx = 0
    pos = 0
    for p in raw_paras:
        if current_len + len(p) + 2 > max_chars_per_chunk and current:
            combined = "\n\n".join(current)
            start = text.find(combined, pos)
            chunks.append(
                Chunk(
                    chunk_id=f"{chunk_id_prefix}_{idx}",
                    text=combined,
                    start_char=start,
                    end_char=start + len(combined),
                )
            )
            idx += 1
            pos = start + len(combined)
            current = []
            current_len = 0
        current.append(p)
        current_len += len(p) + 2
    if current:
        combined = "\n\n".join(current)
        start = text.find(combined, pos)
        chunks.append(
            Chunk(
                chunk_id=f"{chunk_id_prefix}_{idx}",
                text=combined,
                start_char=start,
                end_char=start + len(combined),
            )
        )
    return chunks


def chunk_by_sentences(
    sentences: List[str],
    max_sentences_per_chunk: int = 50,
    overlap_sentences: int = 5,
    chunk_id_prefix: str = "chunk",
) -> List[Chunk]:
    """
    Split a list of sentences into chunks with optional overlap.
    Each chunk gets text = " ".join(sentences) and start_sentence_id / end_sentence_id set.
    """
    if not sentences:
        return []
    if len(sentences) <= max_sentences_per_chunk:
        text = " ".join(s.strip() for s in sentences if s.strip())
        return [
            Chunk(
                chunk_id=f"{chunk_id_prefix}_0",
                text=text,
                start_char=0,
                end_char=len(text),
                start_sentence_id=0,
                end_sentence_id=len(sentences) - 1,
            )
        ]
    step = max(1, max_sentences_per_chunk - overlap_sentences)
    chunks: List[Chunk] = []
    idx = 0
    start = 0
    while start < len(sentences):
        end = min(start + max_sentences_per_chunk, len(sentences))
        seg = sentences[start:end]
        text = " ".join(s.strip() for s in seg if s.strip())
        if text:
            chunks.append(
                Chunk(
                    chunk_id=f"{chunk_id_prefix}_{idx}",
                    text=text,
                    start_char=0,
                    end_char=len(text),
                    start_sentence_id=start,
                    end_sentence_id=end - 1,
                )
            )
            idx += 1
        start += step
        if end >= len(sentences):
            break
    return chunks


# Common legal section headers for optional section-aware chunking
LEGAL_SECTION_PATTERN = re.compile(
    r"^\s*(?:"
    r"FACTS?|BACKGROUND|PROCEDURAL\s+HISTORY|SUMMARY|"
    r"ANALYSIS|DISCUSSION|CONCLUSION|HOLDING|"
    r"JURISDICTION|PARTIES|CLAIMS?|RELIEF\s+REQUESTED"
    r")\s*[:\s]",
    re.IGNORECASE,
)


def chunk_by_sections(
    text: str,
    section_pattern: Optional[re.Pattern] = None,
    max_chars_per_chunk: Optional[int] = None,
    chunk_id_prefix: str = "chunk",
) -> List[Chunk]:
    """
    Split text by detected section headers (e.g. FACTS, BACKGROUND).
    Lines matching section_pattern start a new section; lines between go into that section.
    If section_pattern is None, uses LEGAL_SECTION_PATTERN.
    """
    text = text or ""
    if not text.strip():
        return []
    pattern = section_pattern or LEGAL_SECTION_PATTERN
    lines = text.split("\n")
    sections: List[tuple[str, List[str]]] = []
    current_label = "PREAMBLE"
    current_lines: List[str] = []

    for line in lines:
        m = pattern.match(line)
        if m:
            if current_lines:
                sections.append((current_label, current_lines))
                current_lines = []
            current_label = line.strip()[:50]
        current_lines.append(line)

    if current_lines:
        sections.append((current_label, current_lines))

    chunks: List[Chunk] = []
    pos = 0
    for i, (label, sec_lines) in enumerate(sections):
        block = "\n".join(sec_lines).strip()
        if not block:
            continue
        start = text.find(block, pos)
        if start < 0:
            start = pos
        end = start + len(block)
        pos = end

        if max_chars_per_chunk and len(block) > max_chars_per_chunk:
            sub = chunk_by_characters(
                block,
                chunk_size=max_chars_per_chunk,
                overlap=200,
                chunk_id_prefix=f"{chunk_id_prefix}_{i}",
            )
            for c in sub:
                c.section_label = label
                c.start_char = start + c.start_char
                c.end_char = start + c.end_char
                chunks.append(c)
        else:
            chunks.append(
                Chunk(
                    chunk_id=f"{chunk_id_prefix}_{i}",
                    text=block,
                    start_char=start,
                    end_char=end,
                    section_label=label,
                )
            )
    return chunks


def chunk_document(
    text: str,
    strategy: str = "char",
    chunk_size: int = 12_000,
    overlap: int = 200,
    max_sentences_per_chunk: int = 50,
    overlap_sentences: int = 5,
    sentences: Optional[List[str]] = None,
    chunk_id_prefix: str = "chunk",
) -> List[Chunk]:
    """
    One-entry chunking: choose strategy and return list of Chunk.

    - strategy "char": chunk by characters (chunk_size, overlap).
    - strategy "paragraph": chunk by double-newline; optional chunk_size = max chars per chunk.
    - strategy "sentence": requires sentences list; chunks by max_sentences_per_chunk and overlap_sentences.
    - strategy "section": chunk by legal section headers; optional chunk_size = max chars per section chunk.
    """
    text = text or ""
    if strategy == "char":
        return chunk_by_characters(text, chunk_size=chunk_size, overlap=overlap, chunk_id_prefix=chunk_id_prefix)
    if strategy == "paragraph":
        return chunk_by_paragraphs(
            text,
            max_chars_per_chunk=chunk_size if chunk_size else None,
            chunk_id_prefix=chunk_id_prefix,
        )
    if strategy == "sentence":
        if sentences is None:
            raise ValueError("chunk_document(..., strategy='sentence') requires sentences=...")
        return chunk_by_sentences(
            sentences,
            max_sentences_per_chunk=max_sentences_per_chunk,
            overlap_sentences=overlap_sentences,
            chunk_id_prefix=chunk_id_prefix,
        )
    if strategy == "section":
        return chunk_by_sections(
            text,
            max_chars_per_chunk=chunk_size if chunk_size else None,
            chunk_id_prefix=chunk_id_prefix,
        )
    raise ValueError(f"Unknown chunk strategy: {strategy}. Use 'char', 'paragraph', 'sentence', or 'section'.")
