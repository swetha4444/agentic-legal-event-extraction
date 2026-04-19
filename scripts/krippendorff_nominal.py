"""Krippendorff's alpha (nominal metric), no dependencies beyond numpy.

Rows = raters, columns = units. Use np.nan for missing.
Coincidence matrix follows Krippendorff (e.g. Wikipedia / official formulation).
"""

from __future__ import annotations

import numpy as np


def krippendorff_alpha_nominal(data: np.ndarray) -> float:
    """
    Args:
        data: (n_raters, n_units), integer category codes; np.nan = missing.

    Returns:
        alpha in (-inf, 1]; nan if undefined.
    """
    d = np.asarray(data, dtype=float)
    if d.ndim != 2:
        raise ValueError("data must be 2D (n_raters, n_units)")

    valid = ~np.isnan(d)
    if valid.sum() < 2:
        return float("nan")

    cats = np.unique(d[valid].astype(int))
    cat_to_i = {int(c): i for i, c in enumerate(cats)}
    K = len(cats)
    n_units = d.shape[1]

    O = np.zeros((K, K), dtype=float)
    for j in range(n_units):
        col = d[:, j]
        m = np.sum(~np.isnan(col))
        if m < 2:
            continue
        counts = np.zeros(K, dtype=float)
        for r in range(d.shape[0]):
            if not np.isnan(col[r]):
                counts[cat_to_i[int(col[r])]] += 1
        denom = m - 1.0
        for c in range(K):
            for k in range(K):
                delta = 1.0 if c == k else 0.0
                O[c, k] += (counts[c] * counts[k] - delta * counts[c]) / denom

    n_e = float(np.sum(O))
    if n_e <= 0:
        return float("nan")

    # Nominal: disagreement 0 same category, 1 different
    D_o = 0.0
    for c in range(K):
        for k in range(K):
            if c != k:
                D_o += O[c, k]
    D_o /= n_e

    p = np.zeros(K)
    for c in range(K):
        p[c] = np.sum(O[c, :]) / n_e

    D_e = 0.0
    for c in range(K):
        for k in range(K):
            if c != k:
                D_e += p[c] * p[k]

    if D_e <= 1e-15:
        return 1.0 if D_o <= 1e-15 else float("nan")

    return float(1.0 - D_o / D_e)


def alpha_per_column(data: np.ndarray) -> list[float]:
    """One alpha per unit (column): shape (n_raters, 1) each."""
    out = []
    for j in range(data.shape[1]):
        col = data[:, j : j + 1]
        out.append(krippendorff_alpha_nominal(col))
    return out


if __name__ == "__main__":
    B, T, M = 0, 1, 2
    inter = np.array(
        [
            [M, M, M, T, M],
            [M, M, M, B, M],
            [M, M, M, T, M],
            [M, M, M, M, M],
            [M, M, M, T, M],
        ],
        dtype=float,
    )
    intra = np.array(
        [
            [M, B, M, T, M],
            [M, M, M, T, M],
            [M, M, M, B, M],
            [T, M, M, T, M],
            [M, B, M, T, M],
        ],
        dtype=float,
    )
    labels = ["Event Completeness", "Hallucination", "Evidence", "Edge Quality", "Overall"]
    print("Krippendorff's alpha (nominal), 5 raters x 5 dimensions")
    print(f"  Inter-rater: {krippendorff_alpha_nominal(inter):.4f}")
    print(f"  Intra-rater: {krippendorff_alpha_nominal(intra):.4f}")
    print("Per-dimension (inter):")
    for lab, a in zip(labels, alpha_per_column(inter)):
        print(f"  {lab:22s}  {a:.4f}")
    print("Per-dimension (intra):")
    for lab, a in zip(labels, alpha_per_column(intra)):
        print(f"  {lab:22s}  {a:.4f}")
