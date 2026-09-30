#!/usr/bin/env python3
"""Compute Fleiss' kappa for inter- and intra-rater LLM judge agreement."""

import sys
import numpy as np


def enc(v):
    if "MERGED" in v:
        return 2
    if "TIE" in v:
        return 1
    return 0


DIMS = [
    "EVENT_COMPLETENESS",
    "EVIDENCE_GROUNDING",
    "EDGE_QUALITY",
    "OVERALL_WINNER",
]

INTER = [
    {"EVENT_COMPLETENESS": "MERGED_BETTER", "EVIDENCE_GROUNDING": "MERGED_BETTER", "EDGE_QUALITY": "TIE", "OVERALL_WINNER": "MERGED_BETTER"},
    {"EVENT_COMPLETENESS": "MERGED_BETTER", "EVIDENCE_GROUNDING": "MERGED_BETTER", "EDGE_QUALITY": "BASELINE_BETTER", "OVERALL_WINNER": "MERGED_BETTER"},
    {"EVENT_COMPLETENESS": "MERGED_BETTER", "EVIDENCE_GROUNDING": "MERGED_BETTER", "EDGE_QUALITY": "TIE", "OVERALL_WINNER": "MERGED_BETTER"},
    {"EVENT_COMPLETENESS": "MERGED_BETTER", "EVIDENCE_GROUNDING": "MERGED_BETTER", "EDGE_QUALITY": "MERGED_BETTER", "OVERALL_WINNER": "MERGED_BETTER"},
    {"EVENT_COMPLETENESS": "MERGED_BETTER", "EVIDENCE_GROUNDING": "MERGED_BETTER", "EDGE_QUALITY": "TIE", "OVERALL_WINNER": "MERGED_BETTER"},
]

INTRA = [
    {"EVENT_COMPLETENESS": "MERGED_BETTER", "EVIDENCE_GROUNDING": "MERGED_BETTER", "EDGE_QUALITY": "TIE", "OVERALL_WINNER": "MERGED_BETTER"},
    {"EVENT_COMPLETENESS": "MERGED_BETTER", "EVIDENCE_GROUNDING": "MERGED_BETTER", "EDGE_QUALITY": "TIE", "OVERALL_WINNER": "MERGED_BETTER"},
    {"EVENT_COMPLETENESS": "MERGED_BETTER", "EVIDENCE_GROUNDING": "MERGED_BETTER", "EDGE_QUALITY": "BASELINE_BETTER", "OVERALL_WINNER": "MERGED_BETTER"},
    {"EVENT_COMPLETENESS": "TIE", "EVIDENCE_GROUNDING": "MERGED_BETTER", "EDGE_QUALITY": "TIE", "OVERALL_WINNER": "MERGED_BETTER"},
    {"EVENT_COMPLETENESS": "MERGED_BETTER", "EVIDENCE_GROUNDING": "MERGED_BETTER", "EDGE_QUALITY": "TIE", "OVERALL_WINNER": "MERGED_BETTER"},
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


def build_matrix(judges, dims):
    matrix = np.zeros((len(dims), 3))
    for i, dim in enumerate(dims):
        for j in judges:
            matrix[i, enc(j[dim])] += 1
    return matrix


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
    inter_m = build_matrix(INTER, DIMS)
    intra_m = build_matrix(INTRA, DIMS)
    ik = fleiss_kappa(inter_m)
    ak = fleiss_kappa(intra_m)

    print("=" * 65)
    print("INTER-RATER AGREEMENT (5 different judge personas)")
    print("=" * 65)
    for i, d in enumerate(DIMS):
        votes = [j[d][0] for j in INTER]
        print(f"  {d:30s} {' '.join(votes)}  [{int(inter_m[i,0])}B {int(inter_m[i,1])}T {int(inter_m[i,2])}M]")
    print(f"\n  Fleiss' kappa = {ik:.4f}  ({interpret(ik)})")

    print()
    print("=" * 65)
    print("INTRA-RATER AGREEMENT (same prompt, 5 runs ~ temp variation)")
    print("=" * 65)
    for i, d in enumerate(DIMS):
        votes = [j[d][0] for j in INTRA]
        print(f"  {d:30s} {' '.join(votes)}  [{int(intra_m[i,0])}B {int(intra_m[i,1])}T {int(intra_m[i,2])}M]")
    print(f"\n  Fleiss' kappa = {ak:.4f}  ({interpret(ak)})")

    print()
    print("=" * 65)
    print("OVERALL WINNER CONSENSUS (all 10 judges)")
    print("=" * 65)
    all_ov = [j["OVERALL_WINNER"] for j in INTER + INTRA]
    print(f"  MERGED_BETTER: {all_ov.count('MERGED_BETTER')}/10")
    print(f"  BASELINE_BETTER: {all_ov.count('BASELINE_BETTER')}/10")
    print(f"  TIE: {all_ov.count('TIE')}/10")


if __name__ == "__main__":
    main()
