#!/usr/bin/env python3
"""
eval_irr.py — Inter-Rater Reliability evaluation for minigraph annotations.

Computes Fleiss' kappa, Krippendorff's alpha (nominal + ordinal),
pairwise Cohen's kappa, and percent agreement on the `alignment_label`
field from human annotations stored in Supabase.

Usage:
    python scripts/eval_irr.py                # both GEPA & FrameNet
    python scripts/eval_irr.py --mode gepa    # GEPA only
    python scripts/eval_irr.py --mode framenet
    python scripts/eval_irr.py --csv          # also export CSV

Requires: supabase, krippendorff, statsmodels, scikit-learn
"""

import argparse
import csv
import itertools
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# ── Add annotation_app to path so we can reuse the Supabase client ──────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "annotation_app"))

from supabase_client import get_supabase_client  # noqa: E402


def _fallback_supabase_client():
    """
    Fallback: parse config.js directly (handles multi-line key format)
    and create a Supabase client.
    """
    import re
    config_js = PROJECT_ROOT / "annotation_dashboard" / "config.js"
    if not config_js.exists():
        return None
    text = config_js.read_text()

    url_match = re.search(r'SUPABASE_URL:\s*"([^"]+)"', text)
    key_match = re.search(r'SUPABASE_ANON_KEY:\s*\n?\s*"([^"]+)"', text)
    if not url_match or not key_match:
        return None

    try:
        from supabase import create_client
        return create_client(url_match.group(1), key_match.group(1))
    except Exception as e:
        print(f"   Fallback connection error: {e}")
        return None


# ── Constants ────────────────────────────────────────────────────────────────
LABEL_ORDER = ["aligned", "partially_aligned", "misaligned"]
LABEL_TO_INT = {label: i for i, label in enumerate(LABEL_ORDER)}

# Legacy UUID → readable name (same map as app.py)
LEGACY_UUID_MAP = {
    "218b4166-e5d5-4f64-86b6-2adf3c14ee70": "heo",
}

TABLE_CONFIGS = {
    "gepa": {
        "ann_table": "event_annotations",
        "label": "GEPA Minigraph",
    },
    "framenet": {
        "ann_table": "framenet_annotations",
        "label": "FrameNet Minigraph",
    },
}


def _resolve_user_id(uid: str) -> str:
    return LEGACY_UUID_MAP.get(uid, uid)


# ── Data loading ─────────────────────────────────────────────────────────────

def fetch_annotations(sb, ann_table: str) -> list[dict]:
    """
    Fetch all completed annotations from a Supabase annotation table.
    Returns a list of dicts with keys: chunk_key, user_id, alignment_label.
    """
    rows = []
    page_size = 1000
    offset = 0

    while True:
        resp = (
            sb.table(ann_table)
            .select("chunk_key, user_id, alignment_label")
            .eq("annotation_completed", True)
            .range(offset, offset + page_size - 1)
            .execute()
        )
        batch = resp.data
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size

    # Resolve legacy UUIDs
    for r in rows:
        r["user_id"] = _resolve_user_id(r["user_id"])

    return rows


def build_rating_matrix(annotations: list[dict]) -> tuple[
    list[str], list[str], np.ndarray
]:
    """
    Build a rater × item matrix from annotation rows.

    Returns:
        chunk_keys: list of chunk_key strings  (items, row axis)
        raters:     list of user_id strings    (raters, column axis)
        matrix:     np.ndarray of shape (n_items, n_raters)
                    values are label indices (0/1/2) or np.nan for missing
    """
    # Group by chunk_key
    by_chunk: dict[str, dict[str, str]] = defaultdict(dict)
    rater_set: set[str] = set()

    for ann in annotations:
        ck = ann["chunk_key"]
        uid = ann["user_id"]
        label = ann.get("alignment_label")
        if label not in LABEL_TO_INT:
            continue
        by_chunk[ck][uid] = label
        rater_set.add(uid)

    chunk_keys = sorted(by_chunk.keys())
    raters = sorted(rater_set)
    rater_idx = {r: i for i, r in enumerate(raters)}

    matrix = np.full((len(chunk_keys), len(raters)), np.nan)
    for row, ck in enumerate(chunk_keys):
        for uid, label in by_chunk[ck].items():
            col = rater_idx[uid]
            matrix[row, col] = LABEL_TO_INT[label]

    return chunk_keys, raters, matrix


# ── Metric computation ───────────────────────────────────────────────────────

def compute_percent_agreement(matrix: np.ndarray) -> float:
    """
    For each item rated by 2+ raters, compute the fraction of rater-pairs
    that agree, then average across items.
    """
    agreements = []
    for row in matrix:
        valid = row[~np.isnan(row)]
        if len(valid) < 2:
            continue
        n_pairs = 0
        n_agree = 0
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                n_pairs += 1
                if valid[i] == valid[j]:
                    n_agree += 1
        agreements.append(n_agree / n_pairs)
    return float(np.mean(agreements)) if agreements else 0.0


def compute_fleiss_kappa(matrix: np.ndarray, n_categories: int = 3) -> float:
    """
    Compute Fleiss' kappa for items rated by 2+ raters.
    Direct implementation that handles variable number of raters per item.

    Uses the generalized formula:
        κ = (P̄ - P̄_e) / (1 - P̄_e)
    where P̄ is the mean observed agreement and P̄_e is chance agreement.
    """
    # Build category count table: each row is [count_cat0, count_cat1, count_cat2]
    # Only include items with 2+ ratings
    table_rows = []
    n_raters_per_item = []
    for row in matrix:
        valid = row[~np.isnan(row)].astype(int)
        if len(valid) < 2:
            continue
        counts = [0] * n_categories
        for val in valid:
            counts[val] += 1
        table_rows.append(counts)
        n_raters_per_item.append(len(valid))

    if not table_rows:
        return float("nan")

    table = np.array(table_rows, dtype=float)
    n_items = len(table_rows)
    n_raters_arr = np.array(n_raters_per_item, dtype=float)

    # P_i = (1 / (n_i * (n_i - 1))) * (sum_j(n_ij^2) - n_i) for each item i
    P_i = np.zeros(n_items)
    for i in range(n_items):
        ni = n_raters_arr[i]
        P_i[i] = (np.sum(table[i] ** 2) - ni) / (ni * (ni - 1))

    P_bar = np.mean(P_i)

    # p_j = proportion of all ratings that fall in category j
    total_ratings = np.sum(table)
    p_j = np.sum(table, axis=0) / total_ratings

    P_e = np.sum(p_j ** 2)

    if P_e >= 1.0:
        return float("nan")

    kappa = (P_bar - P_e) / (1.0 - P_e)
    return float(kappa)


def compute_krippendorff_alpha(
    matrix: np.ndarray, level: str = "nominal"
) -> float:
    """
    Compute Krippendorff's alpha. Handles missing data (np.nan).

    level: 'nominal' or 'ordinal'
    """
    import krippendorff

    # krippendorff expects reliability_data shaped (n_raters, n_items)
    # with np.nan for missing values
    reliability_data = matrix.T  # transpose to (raters, items)

    return float(
        krippendorff.alpha(
            reliability_data=reliability_data,
            level_of_measurement=level,
            value_domain=list(range(len(LABEL_ORDER))),
        )
    )


def compute_pairwise_cohens_kappa(
    matrix: np.ndarray, raters: list[str]
) -> list[tuple[str, str, float, int]]:
    """
    Compute Cohen's kappa for every pair of raters on items they both rated.
    Returns list of (rater_a, rater_b, kappa, n_shared_items).
    """
    from sklearn.metrics import cohen_kappa_score

    results = []
    for i, j in itertools.combinations(range(len(raters)), 2):
        # Find items where both raters have a rating
        mask = ~np.isnan(matrix[:, i]) & ~np.isnan(matrix[:, j])
        shared = mask.sum()
        if shared < 2:
            results.append((raters[i], raters[j], float("nan"), int(shared)))
            continue

        ratings_i = matrix[mask, i].astype(int)
        ratings_j = matrix[mask, j].astype(int)

        try:
            kappa = cohen_kappa_score(ratings_i, ratings_j)
        except Exception:
            kappa = float("nan")

        results.append((raters[i], raters[j], float(kappa), int(shared)))

    return results


def label_distribution(matrix: np.ndarray) -> dict[str, float]:
    """Return the percentage of each label across all valid ratings."""
    valid = matrix[~np.isnan(matrix)].astype(int)
    total = len(valid)
    if total == 0:
        return {label: 0.0 for label in LABEL_ORDER}
    dist = {}
    for idx, label in enumerate(LABEL_ORDER):
        count = int((valid == idx).sum())
        dist[label] = round(100 * count / total, 1)
    return dist


def interpret_kappa(value: float) -> str:
    """Landis-Koch interpretation for kappa/alpha values."""
    if np.isnan(value):
        return "N/A"
    if value < 0:
        return "less than chance"
    if value <= 0.20:
        return "slight"
    if value <= 0.40:
        return "fair"
    if value <= 0.60:
        return "moderate"
    if value <= 0.80:
        return "substantial"
    return "almost perfect"


# ── Reporting ────────────────────────────────────────────────────────────────

def print_report(
    mode_label: str,
    chunk_keys: list[str],
    raters: list[str],
    matrix: np.ndarray,
) -> dict:
    """Print a formatted report and return the results dict."""
    n_items = len(chunk_keys)
    n_multi = sum(1 for row in matrix if np.sum(~np.isnan(row)) >= 2)

    # Compute metrics
    pct_agree = compute_percent_agreement(matrix)
    fleiss_k = compute_fleiss_kappa(matrix)
    kripp_nom = compute_krippendorff_alpha(matrix, level="nominal")
    kripp_ord = compute_krippendorff_alpha(matrix, level="ordinal")
    pairwise = compute_pairwise_cohens_kappa(matrix, raters)
    dist = label_distribution(matrix)

    # Per-rater counts
    rater_counts = {}
    for i, r in enumerate(raters):
        rater_counts[r] = int(np.sum(~np.isnan(matrix[:, i])))

    separator = "═" * 60
    print()
    print(separator)
    print(f"  {mode_label} — Inter-Rater Reliability")
    print(separator)
    print(f"  Raters:                 {len(raters)} ({', '.join(raters)})")
    for r in raters:
        print(f"    {r:>20s}:  {rater_counts[r]} annotations")
    print()
    print(f"  Total chunks rated:     {n_items}")
    print(f"  Chunks with 2+ ratings: {n_multi}")
    print()
    print(f"  Fleiss' kappa:          {fleiss_k:+.4f}  ({interpret_kappa(fleiss_k)})")
    print(f"  Krippendorff α nominal: {kripp_nom:+.4f}  ({interpret_kappa(kripp_nom)})")
    print(f"  Krippendorff α ordinal: {kripp_ord:+.4f}  ({interpret_kappa(kripp_ord)})")
    print(f"  Percent agreement:      {pct_agree * 100:.1f}%")
    print()
    print("  Pairwise Cohen's kappa:")
    for ra, rb, kappa, n_shared in pairwise:
        k_str = f"{kappa:+.4f}" if not np.isnan(kappa) else "  N/A "
        print(f"    {ra} ↔ {rb}:  {k_str}  ({n_shared} shared items, {interpret_kappa(kappa)})")
    print()
    print("  Label distribution (all ratings):")
    for label, pct in dist.items():
        bar = "█" * int(pct / 2)
        print(f"    {label:>20s}:  {pct:5.1f}%  {bar}")
    print(separator)
    print()

    return {
        "mode": mode_label,
        "n_raters": len(raters),
        "raters": raters,
        "n_items": n_items,
        "n_multi_rated": n_multi,
        "fleiss_kappa": fleiss_k,
        "krippendorff_alpha_nominal": kripp_nom,
        "krippendorff_alpha_ordinal": kripp_ord,
        "percent_agreement": pct_agree,
        "pairwise_kappa": pairwise,
        "label_distribution": dist,
        "rater_counts": rater_counts,
    }


def export_csv(results: list[dict], output_path: Path):
    """Export IRR results to a CSV file."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "mode", "n_raters", "n_items", "n_multi_rated",
            "fleiss_kappa", "kripp_alpha_nominal", "kripp_alpha_ordinal",
            "percent_agreement",
            "aligned_pct", "partially_aligned_pct", "misaligned_pct",
        ])
        for r in results:
            writer.writerow([
                r["mode"],
                r["n_raters"],
                r["n_items"],
                r["n_multi_rated"],
                f"{r['fleiss_kappa']:.4f}",
                f"{r['krippendorff_alpha_nominal']:.4f}",
                f"{r['krippendorff_alpha_ordinal']:.4f}",
                f"{r['percent_agreement']:.4f}",
                r["label_distribution"].get("aligned", 0),
                r["label_distribution"].get("partially_aligned", 0),
                r["label_distribution"].get("misaligned", 0),
            ])

        # Pairwise section
        writer.writerow([])
        writer.writerow(["mode", "rater_a", "rater_b", "cohens_kappa", "n_shared_items"])
        for r in results:
            for ra, rb, kappa, n_shared in r["pairwise_kappa"]:
                writer.writerow([
                    r["mode"], ra, rb,
                    f"{kappa:.4f}" if not np.isnan(kappa) else "N/A",
                    n_shared,
                ])

    print(f"📄 CSV exported to {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute inter-rater reliability for minigraph annotations."
    )
    parser.add_argument(
        "--mode",
        choices=["gepa", "framenet", "both"],
        default="both",
        help="Which graph type to evaluate (default: both).",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Export results to CSV.",
    )
    parser.add_argument(
        "--min-raters",
        type=int,
        default=None,
        help="Minimum number of raters required per chunk. Default: all found raters.",
    )
    args = parser.parse_args()

    # Connect to Supabase
    sb = get_supabase_client()
    if not sb:
        print("   Primary client failed, trying fallback config parser...")
        sb = _fallback_supabase_client()
    if not sb:
        print("❌ Could not connect to Supabase. Check your env vars or config.js.")
        sys.exit(1)
    print("✅ Connected to Supabase")

    modes = (
        list(TABLE_CONFIGS.keys())
        if args.mode == "both"
        else [args.mode]
    )

    all_results = []

    for mode in modes:
        cfg = TABLE_CONFIGS[mode]
        print(f"\n⏳ Fetching {cfg['label']} annotations from `{cfg['ann_table']}`...")
        annotations = fetch_annotations(sb, cfg["ann_table"])
        print(f"   → {len(annotations)} completed annotations fetched")

        if not annotations:
            print(f"   ⚠️  No annotations found in `{cfg['ann_table']}`. Skipping.")
            continue

        chunk_keys, raters, matrix = build_rating_matrix(annotations)

        if len(raters) < 2:
            print(f"   ⚠️  Only {len(raters)} rater(s) found. Need 2+ for IRR. Skipping.")
            continue

        min_raters = args.min_raters if args.min_raters is not None else len(raters)
        valid_rows = [i for i, row in enumerate(matrix) if np.sum(~np.isnan(row)) >= min_raters]
        
        if len(valid_rows) < len(matrix):
            print(f"   ℹ️  Filtered from {len(matrix)} to {len(valid_rows)} chunks having {min_raters}+ ratings.")
            matrix = matrix[valid_rows]
            chunk_keys = [chunk_keys[i] for i in valid_rows]
            
        if not chunk_keys:
            print(f"   ⚠️  No chunks found with {min_raters}+ ratings. Skipping.")
            continue

        result = print_report(cfg["label"], chunk_keys, raters, matrix)
        all_results.append(result)

    # Export CSV
    if args.csv and all_results:
        csv_path = PROJECT_ROOT / "data" / "outputs" / "irr_results.csv"
        export_csv(all_results, csv_path)

    if not all_results:
        print("\n⚠️  No results to report. Make sure annotations exist in Supabase.")
        sys.exit(1)


if __name__ == "__main__":
    main()
