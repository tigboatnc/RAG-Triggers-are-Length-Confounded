"""TRY 4 — Pre-discretization signal analysis.

Does the per-step CRV-derived scalar carry signal BEFORE we threshold it?
If trace-level AUROC for predicting label_correct from max(score_trace)
is ~0.5, the signal isn't there and no threshold tweak will save it.
If AUROC > 0.55, the signal exists and the work is in extracting it.

Computed:
  - Per-step AUROC of each scalar (using label_correct of the parent trace)
  - Trace-level AUROC of max(score)
  - Trace-level AUROC of mean(score)
  - Mann-Whitney U two-sided p for trace-level max(score) ~ correctness
"""
from __future__ import annotations
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load_traces, label, sd

ARCHS = ["phi", "gemma", "qwen", "phi_hotpot_v2", "phi_2wiki_v2"]
KEYS  = ["drift", "variance", "etc", "hvr", "dragin"]


def auroc(scores_pos: list[float], scores_neg: list[float]) -> float:
    """AUROC via Mann-Whitney U / (n_p * n_n). scores_pos = scores in the 'positive' class."""
    if not scores_pos or not scores_neg:
        return float("nan")
    # rank all
    combined = [(s, 1) for s in scores_pos] + [(s, 0) for s in scores_neg]
    combined.sort(key=lambda x: x[0])
    n = len(combined)
    # tied-rank handling
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-indexed mean rank
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    R_pos = sum(r for r, (_, lbl) in zip(ranks, combined) if lbl == 1)
    n_p = len(scores_pos); n_n = len(scores_neg)
    U = R_pos - n_p * (n_p + 1) / 2
    return U / (n_p * n_n)


def trace_features(t: dict, key: str) -> dict:
    sc = sd._trace_scores(t)
    vs = [float(v) for v in sc[key] if v is not None]
    if not vs: return {"max": float("nan"), "mean": float("nan"), "sum": float("nan")}
    return {"max": max(vs), "mean": sum(vs) / len(vs), "sum": sum(vs)}


print(f"PRE-DISCRETIZATION AUROC  (label_INCORRECT vs label_CORRECT; >0.5 ⇒ higher score on incorrect)\n")
print(f"{'arch':16} | " + " | ".join(f"{k:^18}" for k in KEYS))
print(f"{'':16} | " + " | ".join(f"{'step':>5} {'trace':>5} {'mean':>5}" for _ in KEYS))
print("-" * (16 + 3 + 21 * len(KEYS)))

for arch in ARCHS:
    try:
        traces = load_traces(arch)
    except (KeyError, FileNotFoundError):
        continue
    if not traces:
        continue
    row = f"{arch:16} | "
    for key in KEYS:
        # Per-step: collect (score, parent_label) for every step
        pos_step, neg_step = [], []
        # Per-trace max + mean
        pos_max, neg_max = [], []
        pos_mean, neg_mean = [], []
        for t in traces:
            sc = sd._trace_scores(t)
            lbl = label(t)
            for v in sc[key]:
                if v is not None:
                    (neg_step if lbl else pos_step).append(float(v))
            feats = trace_features(t, key)
            if feats["max"] == feats["max"]:
                (neg_max if lbl else pos_max).append(feats["max"])
                (neg_mean if lbl else pos_mean).append(feats["mean"])
        au_step = auroc(pos_step, neg_step)
        au_max  = auroc(pos_max,  neg_max)
        au_mean = auroc(pos_mean, neg_mean)
        row += f"{au_step:>5.3f} {au_max:>5.3f} {au_mean:>5.3f} | "
    print(row)

print()
print("Reading: AUROC > 0.55 ⇒ signal exists pre-discretization; AUROC ≈ 0.50 ⇒ no signal.")
print("'step' = per-step AUROC (large n_steps, noisy individual signal)")
print("'trace' = per-trace max(score) AUROC (what gates effectively use)")
print("'mean' = per-trace mean(score) AUROC (alternative summary)")
