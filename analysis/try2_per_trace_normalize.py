"""TRY 2 — Per-trace normalization.

Replace absolute threshold with within-trace z-score threshold.
Hypothesis: scale-invariant per-trace normalization makes gate rankings
more stable across architectures and datasets.

For each gate ∈ {drift, variance, etc, dragin}, compute:
  - absolute-τ Δfrac (current method)
  - z-score-τ Δfrac (normalized: (x - μ_trace) / σ_trace > z*)
  - max-norm Δfrac (x / max(x_trace) > q*)
Calibrate τ / z* / q* on calibration split, evaluate on val+test.
"""
from __future__ import annotations
import math
import statistics
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (load_traces, label, fit_threshold, delta_frac, split_traces,
                     V1_TRACES, sd)


def trace_zscore_scores(t, key):
    sc = sd._trace_scores(t)
    vs = [float(v) for v in sc[key] if v is not None]
    if len(vs) < 2: return [0.0] * len(vs)
    m = statistics.mean(vs); s = statistics.pstdev(vs)
    if s == 0: return [0.0] * len(vs)
    return [(v - m) / s for v in vs]


def trace_maxnorm_scores(t, key):
    """Use median+1MAD normalization instead of max-norm (max-norm is degenerate
    because every trace has at least one step = 1.0)."""
    sc = sd._trace_scores(t)
    vs = [float(v) for v in sc[key] if v is not None]
    if len(vs) < 2: return [0.0] * len(vs)
    s = sorted(vs); med = s[len(s)//2]
    mad = sorted([abs(v - med) for v in vs])[len(vs)//2]
    if mad == 0: mad = 1e-6
    return [(v - med) / mad for v in vs]


def fires_with_normalized(traces, key, threshold, normalizer):
    """normalizer: callable trace -> normalized per-step score list"""
    out = []
    for t in traces:
        norm = normalizer(t, key)
        out.append(any(v > threshold for v in norm))
    return out


def fit_normalized_threshold(traces, key, target, normalizer, candidates):
    per_trace = [normalizer(t, key) for t in traces]
    all_scores = sorted(v for ts in per_trace for v in ts)
    if not all_scores: return 0.0
    cand_vals = sorted({all_scores[int(q * (len(all_scores) - 1))] for q in candidates})
    best, bd = cand_vals[-1], float("inf")
    for thr in cand_vals:
        fired = sum(1 for ts in per_trace if any(v > thr for v in ts))
        rate = fired / len(traces)
        d = abs(rate - target)
        if d < bd:
            bd, best = d, thr
    return float(best)


TARGET = 0.30
ARCHS = ["phi", "gemma", "qwen"]
GATES = ["drift", "variance", "etc", "dragin"]
CANDS = (.50,.60,.70,.75,.80,.85,.88,.90,.93,.95,.97,.99)

print(f"PER-TRACE NORMALIZATION SWEEP  (target trace-fire-rate = {TARGET})\n")
print(f"{'arch':6} {'gate':10} | {'abs τ':>8} {'Δfrac':>8} | {'zscore τ':>10} {'Δfrac':>8} | {'maxnorm τ':>11} {'Δfrac':>8}")
print("-" * 95)

results = {}
for arch in ARCHS:
    traces = load_traces(arch)
    splits = split_traces(traces)
    cal = splits["calibration"]
    eval_ = splits["validation"] + splits["test"]
    if not cal or not eval_:
        continue
    eval_labels = [label(r) for r in eval_]
    results[arch] = {}
    for g in GATES:
        # Absolute
        tau_abs = fit_threshold(cal, g, TARGET, "trace")
        thr_abs = {"drift": tau_abs if g=="drift" else 0, "variance": tau_abs if g=="variance" else 0,
                   "etc": tau_abs if g=="etc" else 0, "dragin": tau_abs if g=="dragin" else 0, "hvr":0}
        from _common import gate_fire_per_trace
        fires = gate_fire_per_trace(eval_, g, thr_abs, random_p=TARGET)
        df_abs, _ = delta_frac(fires, eval_labels)

        # Z-score
        zstar = fit_normalized_threshold(cal, g, TARGET, trace_zscore_scores, CANDS)
        fires_z = fires_with_normalized(eval_, g, zstar, trace_zscore_scores)
        df_z, _ = delta_frac(fires_z, eval_labels)

        # Max-norm
        qstar = fit_normalized_threshold(cal, g, TARGET, trace_maxnorm_scores, CANDS)
        fires_m = fires_with_normalized(eval_, g, qstar, trace_maxnorm_scores)
        df_m, _ = delta_frac(fires_m, eval_labels)

        results[arch][g] = {"abs": df_abs, "z": df_z, "max": df_m,
                            "tau_abs": tau_abs, "zstar": zstar, "qstar": qstar}
        print(f"{arch:6} {g:10} | {tau_abs:>8.3f} {df_abs:>+8.3f} | {zstar:>10.3f} {df_z:>+8.3f} | {qstar:>11.3f} {df_m:>+8.3f}")

# Cross-arch stability: which method has lower variance of Δfrac across archs?
print("\n" + "=" * 75)
print("CROSS-ARCH STABILITY  (lower std = method ranks gates more consistently across archs):")
print(f"{'gate':10} | {'absolute (μ, σ)':>22} | {'z-score (μ, σ)':>22} | {'max-norm (μ, σ)':>22}")
for g in GATES:
    def _filt(method):
        return [results[a][g][method] for a in ARCHS
                if g in results[a] and results[a][g][method] == results[a][g][method]]
    abs_vals = _filt("abs"); z_vals = _filt("z"); m_vals = _filt("max")
    if not m_vals: m_vals = [float("nan")]
    if not abs_vals: abs_vals = [float("nan")]
    if not z_vals: z_vals = [float("nan")]
    def _ms(v):
        clean = [x for x in v if x == x]
        if not clean: return float("nan"), float("nan")
        return statistics.mean(clean), (statistics.pstdev(clean) if len(clean)>1 else 0.0)
    am, asd = _ms(abs_vals); zm, zsd = _ms(z_vals); mm, msd = _ms(m_vals)
    print(f"{g:10} | ({am:+.3f}, {asd:.3f})         ({zm:+.3f}, {zsd:.3f})           ({mm:+.3f}, {msd:.3f})")

# Headline: best method per gate
print("\nBEST NORMALIZATION (highest mean Δfrac across archs):")
for g in GATES:
    def _avg(method):
        v = [results[a][g][method] for a in ARCHS
             if g in results.get(a, {}) and results[a][g][method] == results[a][g][method]]
        return statistics.mean(v) if v else float("nan")
    means = {"abs": _avg("abs"), "z": _avg("z"), "max": _avg("max")}
    valid = {k: v for k, v in means.items() if v == v}
    best = max(valid, key=valid.get) if valid else "n/a"
    print(f"  {g:10}: {best:8} (mean Δfrac = {valid.get(best, float('nan')):+.3f})  "
          f"[abs={means['abs']:+.3f}, z={means['z']:+.3f}, max={means['max']:+.3f}]")
