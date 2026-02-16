"""
LegalBERT-based facts extraction agent.
Same interface as LLMFactsExtractor: extract(text, case_name="", docket_number="") -> str.
Uses a fine-tuned LegalBERT sentence classifier (see scripts/train_legal_bert_facts.py).
"""
from pathlib import Path
from typing import Optional, Union

from processing.legal_bert_facts import LegalBERTFactExtractor


class LegalBERTFactsExtractor:
    """Extract court-established facts using a LegalBERT sentence classifier (fact vs non-fact)."""

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: Optional[str] = None,
        first_contiguous_only: bool = False,
    ):
        """
        Args:
            checkpoint_path: Path to trained model dir (from train_legal_bert_facts.py).
            device: 'cuda', 'cpu', or None (auto).
            first_contiguous_only: If True, keep only the first contiguous block of fact sentences.
        """
        self._extractor = LegalBERTFactExtractor(
            checkpoint_path=checkpoint_path,
            device=device,
            first_contiguous_only=first_contiguous_only,
        )

    def extract(self, text: str, case_name: str = "", docket_number: str = "") -> str:
        """Extract court-established facts. case_name and docket_number are unused (kept for API parity with LLM agent)."""
        return self._extractor.extract(text)
