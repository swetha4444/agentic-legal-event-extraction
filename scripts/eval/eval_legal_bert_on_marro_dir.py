#!/usr/bin/env python3
"""
Evaluate LegalBERT/LexLM on MARRO .txt file(s) or directory.
Single file: --docs path/to/doc.txt. Directory: --docs path/to/dir (all .txt).
Reports per-doc metrics and aggregate (micro F1, macro F1, mean accuracy).
"""
import argparse
import random
import sys
from datetime import datetime
from pathlib import Path

# Project root (script lives in scripts/eval/)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from processing.legal_bert_facts import LegalBERTFactExtractor


COUNTRY_TO_DIR = {
    "uk": ROOT / "data" / "MARRO" / "UK-dataset",
    "india": ROOT / "data" / "MARRO" / "IN-dataset",
    "in": ROOT / "data" / "MARRO" / "IN-dataset",
}


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


def resolve_doc_paths(docs_path: Path):
    docs_path = docs_path.resolve()
    if docs_path.is_file():
        return [docs_path]
    if docs_path.is_dir():
        return sorted(docs_path.glob("*.txt"))
    raise FileNotFoundError(f"--docs must be a file or dir: {docs_path}")


def prompt_country():
    while True:
        raw = input("Choose country [uk/india]: ").strip().lower()
        if raw in COUNTRY_TO_DIR:
            return raw
        print("Invalid choice. Enter 'uk' or 'india'.")


def prompt_limit():
    while True:
        raw = input("Enter sentence limit (e.g. 100, or blank for all): ").strip()
        if raw == "":
            return None
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
        print("Invalid limit. Enter a positive integer or blank.")


def balanced_sample_counts(lengths, limit, rng):
    """Allocate up to `limit` examples across docs with near-equal per-doc counts."""
    total_available = sum(lengths)
    if limit is None or limit >= total_available:
        return list(lengths)
    if limit <= 0:
        return [0] * len(lengths)

    non_empty = [idx for idx, n in enumerate(lengths) if n > 0]
    if not non_empty:
        return [0] * len(lengths)

    counts = [0] * len(lengths)
    base = limit // len(non_empty)
    for idx in non_empty:
        counts[idx] = min(base, lengths[idx])

    remaining = limit - sum(counts)
    while remaining > 0:
        candidates = [idx for idx in non_empty if counts[idx] < lengths[idx]]
        if not candidates:
            break
        rng.shuffle(candidates)
        for idx in candidates:
            if remaining == 0:
                break
            if counts[idx] < lengths[idx]:
                counts[idx] += 1
                remaining -= 1

    return counts


def save_run_output(lines, prefix: str):
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{prefix}_{stamp}.txt"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def main():
    p = argparse.ArgumentParser(description="Eval BERT/LexLM on MARRO doc(s); report aggregate")
    p.add_argument("--checkpoint", type=Path, required=True, help="Path to trained model dir")
    p.add_argument("--docs", type=Path, default=None, help="MARRO .txt file or dir of .txt files")
    p.add_argument(
        "--country",
        type=str,
        choices=["uk", "india", "in"],
        default=None,
        help="Shortcut for MARRO country directory (data/MARRO/UK-dataset or IN-dataset)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Total sentence limit sampled across selected .txt files (balanced per file).",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed for balanced sampling.")
    p.add_argument("--device", default=None)
    p.add_argument("--quiet", action="store_true", help="Only print aggregate (no per-doc table)")
    p.add_argument("--no-save-output", action="store_true", help="Do not save run output to scripts/eval/outputs.")
    args = p.parse_args()
    run_lines = []

    def emit(line=""):
        print(line)
        run_lines.append(str(line))

    args.checkpoint = args.checkpoint.resolve()

    if not args.checkpoint.is_dir():
        print(f"Error: checkpoint not found {args.checkpoint}")
        sys.exit(1)

    if args.docs is None:
        if args.country is None:
            if sys.stdin.isatty():
                args.country = prompt_country()
            else:
                emit("Error: provide --country {uk|india} when running non-interactively.")
                sys.exit(1)
        if args.limit is None:
            if sys.stdin.isatty():
                args.limit = prompt_limit()
            else:
                emit("Error: provide --limit when running non-interactively with --country.")
                sys.exit(1)

    if args.docs is not None:
        try:
            doc_paths = resolve_doc_paths(args.docs)
        except FileNotFoundError as exc:
            emit(f"Error: {exc}")
            sys.exit(1)
    else:
        country_key = args.country or "uk"
        country_dir = COUNTRY_TO_DIR[country_key]
        doc_paths = sorted(country_dir.glob("*.txt"))

    if not doc_paths:
        target_path = args.docs if args.docs is not None else COUNTRY_TO_DIR[args.country or "uk"]
        emit(f"No .txt files in {target_path}")
        sys.exit(1)
    if args.limit is not None and args.limit <= 0:
        emit("Error: --limit must be a positive integer.")
        sys.exit(1)

    extractor = LegalBERTFactExtractor(checkpoint_path=str(args.checkpoint), device=args.device)
    rng = random.Random(args.seed)

    loaded_docs = []
    for path in doc_paths:
        sentences, gold = load_marro_doc_with_labels(path)
        if sentences:
            loaded_docs.append((path, sentences, gold))
    if not loaded_docs:
        emit("No docs with valid sentences.")
        sys.exit(1)

    per_doc_lengths = [len(sentences) for _, sentences, _ in loaded_docs]
    sample_counts = balanced_sample_counts(per_doc_lengths, args.limit, rng)

    all_tp, all_fp, all_fn = 0, 0, 0
    doc_f1s = []
    doc_accs = []
    rows = []
    sampled_total = 0
    full_total = sum(per_doc_lengths)

    for (path, sentences, gold), sample_n in zip(loaded_docs, sample_counts):
        if sample_n <= 0:
            continue
        if sample_n < len(sentences):
            picked = sorted(rng.sample(range(len(sentences)), sample_n))
            sampled_sentences = [sentences[i] for i in picked]
            sampled_gold = [gold[i] for i in picked]
        else:
            sampled_sentences = sentences
            sampled_gold = gold

        sampled_total += len(sampled_sentences)
        pred = extractor.predict_labels(sampled_sentences)
        if len(pred) != len(sampled_gold):
            continue
        acc, prec, rec, f1, tp, fp, fn = metrics(sampled_gold, pred)
        all_tp += tp
        all_fp += fp
        all_fn += fn
        doc_f1s.append(f1)
        doc_accs.append(acc)
        rows.append((path.name, len(sampled_gold), sum(sampled_gold), sum(pred), acc, prec, rec, f1))

    if not rows:
        emit("No docs with valid sentences.")
        sys.exit(1)

    if not args.quiet:
        emit(f"Checkpoint: {args.checkpoint}")
        if "holdout" not in str(args.checkpoint).lower():
            emit("(in-domain: docs were in training unless you used run_*_train_holdout.sh)")
        emit()
        emit(f"{'Doc':<28} {'N':>5} {'GoldF':>6} {'PredF':>6} {'Acc':>7} {'P':>6} {'R':>6} {'F1':>6}")
        emit("-" * 76)
        for name, n, gf, pf, acc, p, r, f1 in rows:
            emit(f"{name:<28} {n:>5} {gf:>6} {pf:>6} {acc:>7.4f} {p:>6.4f} {r:>6.4f} {f1:>6.4f}")
        emit("-" * 76)
        sampled_msg = f"sampled {sampled_total}/{full_total} sentence(s)"
        if args.limit is None:
            sampled_msg = f"using all {sampled_total} sentence(s)"
        emit(f"Selection: {sampled_msg} (seed={args.seed})")

    n_docs = len(rows)
    micro_prec = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    micro_rec = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0
    macro_f1 = sum(doc_f1s) / n_docs
    mean_acc = sum(doc_accs) / n_docs

    emit(f"Docs: {n_docs}")
    emit(f"Micro (pooled) — P: {micro_prec:.4f}  R: {micro_rec:.4f}  F1: {micro_f1:.4f}")
    emit(f"Macro (avg per-doc) — F1: {macro_f1:.4f}  Acc: {mean_acc:.4f}")

    if not args.no_save_output:
        out_path = save_run_output(run_lines, prefix="eval_out")
        emit(f"Saved output: {out_path}")


if __name__ == "__main__":
    main()
