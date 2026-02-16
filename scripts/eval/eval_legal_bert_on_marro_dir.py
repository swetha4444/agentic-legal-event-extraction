#!/usr/bin/env python3
"""
Evaluate LegalBERT/LexLM on MARRO .txt file(s) or directory.
Single file: --docs path/to/doc.txt. Directory: --docs path/to/dir (all .txt).
Reports per-doc metrics and aggregate (micro F1, macro F1, mean accuracy).
"""
import argparse
import sys
from pathlib import Path

# Project root (script lives in scripts/eval/)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from processing.legal_bert_facts import LegalBERTFactExtractor


def load_marro_doc_with_labels(path: Path):
    """Return (sentences, gold_labels). 1 = FAC, 0 = rest."""
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


def metrics(gold, pred):
    """Return (acc, prec, rec, f1, tp, fp, fn) for fact class."""
    correct = sum(1 for p, g in zip(pred, gold) if p == g)
    acc = correct / len(gold) if gold else 0.0
    tp = sum(1 for p, g in zip(pred, gold) if p == 1 and g == 1)
    fp = sum(1 for p, g in zip(pred, gold) if p == 1 and g == 0)
    fn = sum(1 for p, g in zip(pred, gold) if p == 0 and g == 1)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return acc, prec, rec, f1, tp, fp, fn


def main():
    p = argparse.ArgumentParser(description="Eval BERT/LexLM on MARRO doc(s); report aggregate")
    p.add_argument("--checkpoint", type=Path, required=True, help="Path to trained model dir")
    p.add_argument("--docs", type=Path, required=True, help="MARRO .txt file or dir of .txt files")
    p.add_argument("--device", default=None)
    p.add_argument("--quiet", action="store_true", help="Only print aggregate (no per-doc table)")
    args = p.parse_args()

    args.checkpoint = args.checkpoint.resolve()
    args.docs = args.docs.resolve()
    if not args.checkpoint.is_dir():
        print(f"Error: checkpoint not found {args.checkpoint}")
        sys.exit(1)
    if args.docs.is_file():
        doc_paths = [args.docs]
    elif args.docs.is_dir():
        doc_paths = sorted(args.docs.glob("*.txt"))
    else:
        print(f"Error: --docs must be a file or dir: {args.docs}")
        sys.exit(1)
    if not doc_paths:
        print(f"No .txt files in {args.docs}")
        sys.exit(1)

    extractor = LegalBERTFactExtractor(checkpoint_path=str(args.checkpoint), device=args.device)

    all_tp, all_fp, all_fn = 0, 0, 0
    doc_f1s = []
    doc_accs = []
    rows = []

    for path in doc_paths:
        sentences, gold = load_marro_doc_with_labels(path)
        if not sentences:
            continue
        pred = extractor.predict_labels(sentences)
        if len(pred) != len(gold):
            continue
        acc, prec, rec, f1, tp, fp, fn = metrics(gold, pred)
        all_tp += tp
        all_fp += fp
        all_fn += fn
        doc_f1s.append(f1)
        doc_accs.append(acc)
        rows.append((path.name, len(gold), sum(gold), sum(pred), acc, prec, rec, f1))

    if not rows:
        print("No docs with valid sentences.")
        sys.exit(1)

    if not args.quiet:
        print("Checkpoint:", args.checkpoint)
        if "holdout" not in str(args.checkpoint).lower():
            print("(in-domain: docs were in training unless you used run_*_train_holdout.sh)")
        print()
        print(f"{'Doc':<28} {'N':>5} {'GoldF':>6} {'PredF':>6} {'Acc':>7} {'P':>6} {'R':>6} {'F1':>6}")
        print("-" * 76)
        for name, n, gf, pf, acc, p, r, f1 in rows:
            print(f"{name:<28} {n:>5} {gf:>6} {pf:>6} {acc:>7.4f} {p:>6.4f} {r:>6.4f} {f1:>6.4f}")
        print("-" * 76)

    n_docs = len(rows)
    micro_prec = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    micro_rec = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0
    macro_f1 = sum(doc_f1s) / n_docs
    mean_acc = sum(doc_accs) / n_docs

    print(f"Docs: {n_docs}")
    print(f"Micro (pooled) — P: {micro_prec:.4f}  R: {micro_rec:.4f}  F1: {micro_f1:.4f}")
    print(f"Macro (avg per-doc) — F1: {macro_f1:.4f}  Acc: {mean_acc:.4f}")


if __name__ == "__main__":
    main()
