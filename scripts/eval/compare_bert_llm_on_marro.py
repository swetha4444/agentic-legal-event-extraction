#!/usr/bin/env python3
"""
Run LegalBERT and LLM on the same MARRO doc(s); report comparable metrics (Acc, P, R, F1).
"""
import argparse
import re
import sys
from pathlib import Path

# Project root (script lives in scripts/eval/)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def load_marro_doc(path: Path):
    """Return (sentences, gold_labels). Gold: 1 = FAC, 0 = rest."""
    sentences, gold = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            sent, label = line.split("\t", 1)
            sent = sent.strip()
            if not sent:
                continue
            sentences.append(sent)
            gold.append(1 if label.strip().upper() == "FAC" else 0)
    return sentences, gold


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def metrics(gold: list, pred: list):
    if len(pred) != len(gold):
        return None
    correct = sum(1 for p, g in zip(pred, gold) if p == g)
    acc = correct / len(gold)
    tp = sum(1 for p, g in zip(pred, gold) if p == 1 and g == 1)
    fp = sum(1 for p, g in zip(pred, gold) if p == 1 and g == 0)
    fn = sum(1 for p, g in zip(pred, gold) if p == 0 and g == 1)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"acc": acc, "p": prec, "r": rec, "f1": f1, "n": len(gold), "n_fact_gold": sum(gold), "n_fact_pred": sum(pred)}


def main():
    p = argparse.ArgumentParser(description="Compare LegalBERT vs LLM on MARRO doc(s). Same test set, same metrics.")
    p.add_argument("--doc", type=Path, nargs="*", default=None, help="One or more MARRO .txt files")
    p.add_argument("--docs", type=Path, default=None, help="Directory of MARRO .txt files")
    p.add_argument("--bert", type=Path, required=True, help="LegalBERT/LexLM checkpoint dir")
    p.add_argument("--llm", type=str, default=None, help="LLM model name (e.g. gpt4o). If omitted, only BERT.")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    if args.docs is not None:
        args.docs = args.docs.resolve()
        doc_paths = sorted(args.docs.glob("*.txt"))
    elif args.doc:
        doc_paths = [Path(d).resolve() for d in args.doc]
    else:
        print("Provide --doc <file(s)> or --docs <dir>")
        sys.exit(1)
    if not doc_paths:
        print("No documents to evaluate.")
        sys.exit(1)
    if not args.bert.resolve().is_dir():
        print(f"Checkpoint not found: {args.bert}")
        sys.exit(1)

    from processing.legal_bert_facts import LegalBERTFactExtractor
    bert_extractor = LegalBERTFactExtractor(checkpoint_path=str(args.bert), device=args.device)

    llm_extractor = None
    if args.llm:
        try:
            from agents.facts import LLMFactsExtractor
            llm_extractor = LLMFactsExtractor(model_name=args.llm)
        except Exception as e:
            print(f"LLM not available: {e}. Run without --llm for BERT only.")
            args.llm = None

    all_gold, all_bert_pred, all_llm_pred = [], [], []

    for doc_path in doc_paths:
        if not doc_path.is_file():
            continue
        sentences, gold = load_marro_doc(doc_path)
        if not sentences:
            continue
        pred_bert = bert_extractor.predict_labels(sentences)
        all_gold.extend(gold)
        all_bert_pred.extend(pred_bert)
        if llm_extractor is not None:
            doc_text = "\n\n".join(sentences)
            try:
                out = llm_extractor.extract(text=doc_text, case_name="", docket_number="")
            except Exception as e:
                print(f"LLM failed on {doc_path.name}: {e}")
                out = ""
            out_norm = _norm(out)
            all_llm_pred.extend([1 if _norm(s) in out_norm else 0 for s in sentences])

    m_bert = metrics(all_gold, all_bert_pred)
    if m_bert is None:
        sys.exit(1)
    print("=" * 60)
    print(f"Test docs: {len(doc_paths)}  |  Sentences: {m_bert['n']}  |  Gold facts: {m_bert['n_fact_gold']}")
    print("=" * 60)
    print("LegalBERT/LexLM")
    print(f"  Accuracy: {m_bert['acc']:.4f}  |  Fact P: {m_bert['p']:.4f}  R: {m_bert['r']:.4f}  F1: {m_bert['f1']:.4f}  (pred fact: {m_bert['n_fact_pred']})")
    if llm_extractor is not None and all_llm_pred:
        m_llm = metrics(all_gold, all_llm_pred)
        if m_llm:
            print(f"LLM ({args.llm})")
            print(f"  Accuracy: {m_llm['acc']:.4f}  |  Fact P: {m_llm['p']:.4f}  R: {m_llm['r']:.4f}  F1: {m_llm['f1']:.4f}  (pred fact: {m_llm['n_fact_pred']})")
    print("=" * 60)


if __name__ == "__main__":
    main()
