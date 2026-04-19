#!/usr/bin/env python3
"""Compute kappa for hybrid vs baseline across 5 docs."""
import numpy as np


def enc(v):
    if "HYBRID" in v:
        return 2
    if "TIE" in v:
        return 1
    return 0


DIMS = ["EVENT_COMPLETENESS", "EVIDENCE_GROUNDING", "EDGE_QUALITY", "OVERALL_WINNER"]

JUDGES = [
    {"EVENT_COMPLETENESS": "HYBRID_BETTER", "EVIDENCE_GROUNDING": "HYBRID_BETTER", "EDGE_QUALITY": "HYBRID_BETTER", "OVERALL_WINNER": "HYBRID_BETTER"},
    {"EVENT_COMPLETENESS": "HYBRID_BETTER", "EVIDENCE_GROUNDING": "HYBRID_BETTER", "EDGE_QUALITY": "HYBRID_BETTER", "OVERALL_WINNER": "HYBRID_BETTER"},
    {"EVENT_COMPLETENESS": "HYBRID_BETTER", "EVIDENCE_GROUNDING": "HYBRID_BETTER", "EDGE_QUALITY": "HYBRID_BETTER", "OVERALL_WINNER": "HYBRID_BETTER"},
    {"EVENT_COMPLETENESS": "HYBRID_BETTER", "EVIDENCE_GROUNDING": "HYBRID_BETTER", "EDGE_QUALITY": "HYBRID_BETTER", "OVERALL_WINNER": "HYBRID_BETTER"},
    {"EVENT_COMPLETENESS": "BASELINE_BETTER", "EVIDENCE_GROUNDING": "HYBRID_BETTER", "EDGE_QUALITY": "HYBRID_BETTER", "OVERALL_WINNER": "HYBRID_BETTER"},
]

DOC_NAMES = [
    "Miczulski v. Alix",
    "Poole v. Ampler Pizza",
    "Sermarini v. Step Up",
    "Bleiler v. Chester",
    "Bacon v. M.A.G.",
]


def fleiss_kappa(matrix):
    N, K = matrix.shape
    n = int(matrix.sum(axis=1)[0])
    if n <= 1:
        return float("nan")
    P_i = (np.sum(matrix ** 2, axis=1) - n) / (n * (n - 1))
    P_bar = float(np.mean(P_i))
    p_j = np.sum(matrix, axis=0) / (N * n)
    P_e = float(np.sum(p_j ** 2))
    if P_e == 1.0:
        return 1.0
    return (P_bar - P_e) / (1.0 - P_e)


def interpret(k):
    if k > 0.8:
        return "Almost Perfect"
    if k > 0.6:
        return "Substantial"
    if k > 0.4:
        return "Moderate"
    if k > 0.2:
        return "Fair"
    if k > 0.0:
        return "Slight"
    return "Poor"


def main():
    matrix = np.zeros((len(DIMS), 3))
    for i, dim in enumerate(DIMS):
        for j in JUDGES:
            matrix[i, enc(j[dim])] += 1

    kappa = fleiss_kappa(matrix)

    print("=" * 70)
    print("HYBRID vs BASELINE - LLM-as-Judge (5 docs, Cursor fast model)")
    print("Model: Claude Haiku (Cursor built-in fast tier)")
    print("=" * 70)
    print()

    print("Per-doc verdicts:")
    header = f"  {'Doc':<25s}  {'COMPL':>5s}  {'GRND':>5s}  {'EDGE':>5s}  {'OVRL':>5s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for j, name in zip(JUDGES, DOC_NAMES):
        vals = [j[d][0] for d in DIMS]
        print(f"  {name:<25s}  {'  '.join(f'{v:>5s}' for v in vals)}")

    print()
    print("Dimension agreement across 5 docs:")
    for i, d in enumerate(DIMS):
        b, t, h = int(matrix[i, 0]), int(matrix[i, 1]), int(matrix[i, 2])
        print(f"  {d:30s}  {b}B  {t}T  {h}H")

    print()
    print(f"Fleiss' kappa = {kappa:.4f}  ({interpret(kappa)})")
    print()

    total = len(JUDGES) * len(DIMS)
    h_count = sum(1 for j in JUDGES for d in DIMS if "HYBRID" in j[d])
    b_count = sum(1 for j in JUDGES for d in DIMS if "BASELINE" in j[d])
    t_count = total - h_count - b_count
    print(f"Total ratings: {total}")
    print(f"  HYBRID_BETTER:   {h_count}/{total} ({100 * h_count / total:.0f}%)")
    print(f"  BASELINE_BETTER: {b_count}/{total} ({100 * b_count / total:.0f}%)")
    print(f"  TIE:             {t_count}/{total} ({100 * t_count / total:.0f}%)")
    print()
    print("OVERALL_WINNER: HYBRID_BETTER 5/5 docs (100%)")


if __name__ == "__main__":
    main()
