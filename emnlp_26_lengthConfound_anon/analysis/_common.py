"""Tiny shared primitives used across every analysis script.

Currently a single export: `auroc`, computed via Mann-Whitney U with
average-rank tie handling.  No NumPy dependency, no SciPy — pure Python.
"""
from __future__ import annotations


def auroc(scores_pos: list[float], scores_neg: list[float]) -> float:
    """AUROC via Mann-Whitney U / (n_pos * n_neg).

    `scores_pos` are the scores for the class we treat as positive
    throughout the paper:
      - for `lift_em = +1`: positive class is "retrieval would have helped"
      - for `closed_em = 0`: positive class is "closed-book incorrect"
    """
    if not scores_pos or not scores_neg:
        return float("nan")
    combined = [(s, 1) for s in scores_pos] + [(s, 0) for s in scores_neg]
    combined.sort(key=lambda x: x[0])
    n = len(combined)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    R_pos = sum(r for r, (_, lbl) in zip(ranks, combined) if lbl == 1)
    n_p = len(scores_pos)
    n_n = len(scores_neg)
    U = R_pos - n_p * (n_p + 1) / 2
    return U / (n_p * n_n)
