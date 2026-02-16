"""
LegalBERT-based facts extraction: sentence-level binary classifier (fact vs non-fact).
Train on rhetorical-role or custom sentence labels; run inference on opinion text.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Union

try:
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        RobertaTokenizer,
    )
except ImportError as e:
    raise ImportError(
        "LegalBERT extractor requires torch and transformers. "
        "Install with: pip install torch transformers"
    ) from e

# Lazy spacy load to avoid import-time download
_nlp = None
_nlp_available: Optional[bool] = None


def _get_nlp():
    global _nlp, _nlp_available
    if _nlp_available is False:
        return None
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
            _nlp_available = True
        except OSError:
            try:
                import spacy
                from spacy.cli import download
                download("en_core_web_sm")
                _nlp = spacy.load("en_core_web_sm")
                _nlp_available = True
            except Exception:
                _nlp_available = False
        except ImportError:
            _nlp_available = False
    return _nlp


def _split_sentences_fallback(text: str, min_len: int) -> List[str]:
    """Fallback when spacy not available: split on newlines, then on . ! ?"""
    lines = [s.strip() for s in text.splitlines() if s.strip()]
    out = []
    for line in lines:
        if len(line) < min_len:
            continue
        # If line looks like a single sentence, keep as-is; else split on . ! ?
        parts = re.split(r"(?<=[.!?])\s+", line)
        for p in parts:
            p = p.strip()
            if len(p) >= min_len:
                out.append(p)
    return out if out else [s for s in lines if len(s) >= min_len]


def split_sentences(text: str, min_len: int = 10) -> List[str]:
    """Split text into sentences using spaCy if available; else fallback (newline + clause split). Drops very short fragments."""
    nlp = _get_nlp()
    if nlp is not None:
        doc = nlp(text)
        return [s.text.strip() for s in doc.sents if len(s.text.strip()) >= min_len]
    return _split_sentences_fallback(text, min_len)


def _resolve_checkpoint_path(path: Union[str, Path]) -> str:
    """If path is a directory without model weights, use the latest checkpoint-* subdir (Trainer saves per-epoch)."""
    p = Path(path).resolve()
    if not p.is_dir():
        return str(p)
    for name in ("model.safetensors", "pytorch_model.bin"):
        if (p / name).exists():
            return str(p)
    # No weights in root; look for checkpoint-<step> subdirs
    checkpoints = []
    try:
        entries = list(p.iterdir())
    except OSError:
        entries = []
    for d in entries:
        if d.is_dir() and d.name.startswith("checkpoint-"):
            try:
                step = int(d.name.split("-")[1])
                if (d / "model.safetensors").exists() or (d / "pytorch_model.bin").exists():
                    checkpoints.append((step, d))
            except (IndexError, ValueError):
                continue
    if checkpoints:
        checkpoints.sort(key=lambda x: x[0], reverse=True)
        return str(checkpoints[0][1])
    # Nothing found: give a clear error
    subdirs = [e.name for e in entries if e.is_dir()]
    raise OSError(
        f"No model weights (model.safetensors or pytorch_model.bin) in {p}. "
        f"Subdirs found: {subdirs or '(none)'}. "
        "Point --checkpoint to a specific checkpoint, e.g. data/models/lexlm_facts_marro/checkpoint-<step>, "
        "or re-run training so the final model is saved to the output dir."
    )


class LegalBERTFactExtractor:
    """
    Extract court-established facts by classifying each sentence as fact (1) or non-fact (0),
    then concatenating fact sentences. Requires a fine-tuned checkpoint (train with
    scripts/train_legal_bert_facts.py on rhetorical-role or custom labeled data).
    """

    MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
    MAX_LENGTH = 256
    BATCH_SIZE = 32
    LABEL_FACT = 1
    LABEL_NON_FACT = 0

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        max_length: int = MAX_LENGTH,
        batch_size: int = BATCH_SIZE,
        first_contiguous_only: bool = False,
    ):
        """
        Args:
            checkpoint_path: Path to fine-tuned model dir (config.json + pytorch_model.bin).
                            If None, uses MODEL_NAME with randomly initialized head (not useful for inference).
            model_name: Base model to load; default nlpaueb/legal-bert-base-uncased.
            device: 'cuda', 'cpu', or None (auto).
            max_length: Max token length per sentence.
            batch_size: Batch size for inference.
            first_contiguous_only: If True, stop at first run of fact sentences (avoid later "fact" blocks).
        """
        self.model_name = model_name or self.MODEL_NAME
        self.max_length = max_length
        self.batch_size = batch_size
        self.first_contiguous_only = first_contiguous_only

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        load_path = _resolve_checkpoint_path(checkpoint_path) if checkpoint_path else self.model_name
        # Load model first when using a checkpoint: its config tells us the correct tokenizer class.
        # AutoTokenizer can otherwise pick the wrong class (e.g. XLM) from tokenizer_config.
        self.model = AutoModelForSequenceClassification.from_pretrained(
            load_path,
            num_labels=2,
        )
        if checkpoint_path:
            model_type = getattr(self.model.config, "model_type", None)
            if model_type == "roberta":
                self.tokenizer = RobertaTokenizer.from_pretrained(load_path)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        self.model.to(self.device)
        self.model.eval()

    def _predict_batch(self, sentences: List[str]) -> List[int]:
        """Return list of predicted labels (0 or 1) for each sentence."""
        out = []
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i : i + self.batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self.model(**enc).logits
            preds = logits.argmax(dim=-1).cpu().tolist()
            out.extend(preds)
        return out

    def predict_labels(self, sentences: List[str]) -> List[int]:
        """Predict fact (1) vs non-fact (0) for each sentence. For evaluation."""
        if not sentences:
            return []
        return self._predict_batch(sentences)

    def extract(self, text: str) -> str:
        """
        Extract facts from opinion text: split into sentences, classify each,
        concatenate sentences predicted as fact. Returns plain text (paragraphs).
        """
        if not (text or "").strip():
            return ""

        sentences = split_sentences(text)
        if not sentences:
            return ""

        preds = self._predict_batch(sentences)
        fact_sentences = [s for s, p in zip(sentences, preds) if p == self.LABEL_FACT]

        if self.first_contiguous_only and fact_sentences:
            # Keep only the first contiguous block of fact predictions
            chosen = []
            for s, p in zip(sentences, preds):
                if p == self.LABEL_FACT:
                    chosen.append(s)
                elif chosen:
                    break
            fact_sentences = chosen

        return "\n\n".join(_paragraphize(fact_sentences))


def _paragraphize(sentences: List[str], max_sents_per_para: int = 5) -> List[str]:
    """Group sentences into paragraphs (e.g. up to 5 sentences per paragraph)."""
    if not sentences:
        return []
    paras = []
    for i in range(0, len(sentences), max_sents_per_para):
        block = sentences[i : i + max_sents_per_para]
        paras.append(" ".join(block))
    return paras
