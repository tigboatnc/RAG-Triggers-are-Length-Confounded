"""TRY 3 — Δfrac bias floor at high fire rates.

Empirically and analytically: at what fire rate does random's Δfrac
become indistinguishable from zero?

Method:
  - For each arch, fire-rate target ∈ {0.10..0.95}, simulate 5000 random
    gates per cell, compute Δfrac, report mean ± 2σ band.
  - The "noise floor" for a non-random gate at that fire rate is the
    upper edge of the random band; anything above is potential signal.
"""
from __future__ import annotations
import random
import statistics
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load_traces, label, gate_fire_per_trace, delta_frac

ARCHS = ["phi", "gemma", "qwen"]
TARGETS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
N_TRIALS = 5000

print(f"RANDOM-GATE Δfrac BIAS FLOOR  (per-trial random gate, {N_TRIALS} resamples per cell)\n")
print(f"{'arch':6} {'tgt':>4} | {'fire(actual)':>12}  {'EM':>5} | {'mean Δfrac':>10}  {'σ':>6}  {'+2σ floor':>10}  {'−2σ':>7}")
print("-" * 80)

bias_table = {}
for arch in ARCHS:
    traces = load_traces(arch)
    labels = [label(r) for r in traces]
    em = sum(labels) / len(labels) if labels else 0
    bias_table[arch] = {}
    for tgt in TARGETS:
        dfs = []
        actual_fires = []
        for trial in range(N_TRIALS):
            rng = random.Random(trial)
            fires = gate_fire_per_trace(traces, "random", {}, rng=rng, random_p=tgt)
            actual_fires.append(sum(fires) / len(fires))
            df, _ = delta_frac(fires, labels)
            if df == df:
                dfs.append(df)
        if not dfs:
            continue
        m = statistics.mean(dfs); s = statistics.pstdev(dfs)
        mean_fire = statistics.mean(actual_fires)
        bias_table[arch][tgt] = {"mean": m, "std": s, "fire": mean_fire}
        print(f"{arch:6} {tgt:>4.2f} | {mean_fire*100:>10.1f}%   {em*100:>4.1f}% | {m:>+10.4f}  {s:>6.4f}  {m+2*s:>+10.4f}  {m-2*s:>+7.4f}")

# Detect bias direction
print("\n" + "=" * 80)
print("ANALYTICAL INSIGHT:")
print("Random's Δfrac mean is NOT 0 at high fire rates because P(C|¬F) variance grows.")
print("With trace-fire-rate r, n_NF = n(1-r). At r=0.9, n_NF = 0.1n → SE(P(C|¬F)) blows up.")
print("If incorrect-base-rate ≠ 0.5, the random gate is biased toward sign of (0.5 − EM).")
print()
print("Empirical bias sign vs (0.5 − EM) — predicted positive Δfrac when EM < 0.5:")
for arch in ARCHS:
    em = sum(label(r) for r in load_traces(arch)) / len(load_traces(arch))
    high_fire_bias = bias_table[arch][0.80]["mean"] if 0.80 in bias_table[arch] else None
    print(f"  {arch:6}  EM={em*100:.1f}%  →  predicted bias sign: {'+' if em<0.5 else '−'}  observed at r=0.80: {high_fire_bias:+.4f}")

# Noise floor for non-random gates: at typical iso-0.30, +/- 2σ ≈ what?
print("\nNON-RANDOM GATE NOISE FLOOR (at target r=0.30, trace level):")
for arch in ARCHS:
    if 0.30 in bias_table[arch]:
        f = bias_table[arch][0.30]
        print(f"  {arch:6}  mean random Δfrac = {f['mean']:+.4f}  ±2σ = ±{2*f['std']:.4f}  → noise band [{f['mean']-2*f['std']:+.4f}, {f['mean']+2*f['std']:+.4f}]")

print("\n→ Any reported gate Δfrac inside its arch's noise band is statistically indistinguishable from random.")
