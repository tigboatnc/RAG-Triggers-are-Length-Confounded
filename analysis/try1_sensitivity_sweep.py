"""TRY 1 — Calibration sensitivity sweep.

For each (arch × granularity × target × gate), calibrate on the 'calibration'
split, evaluate Δfrac on validation+test (zero-leakage, mirrors v1 #20).
Report who wins where.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (load_traces, label, fit_threshold, gate_fire_per_trace,
                     delta_frac, split_traces, V1_TRACES)

ARCHS = ["phi", "gemma", "qwen"]
GRANULARITIES = ["step", "trace"]
TARGETS = [0.10, 0.20, 0.30, 0.40]
GATES = ["drift", "variance", "or", "etc", "hvr", "dragin"]

results = []
print(f"{'arch':6} {'gran':5} {'tgt':5} | " + "  ".join(f"{g:>8}" for g in GATES) + "  ||  winner  (Δ)")
print("-" * 110)
for arch in ARCHS:
    traces = load_traces(arch)
    splits = split_traces(traces)
    cal = splits["calibration"]
    eval_ = splits["validation"] + splits["test"]
    if not cal or not eval_:
        print(f"{arch}: split empty"); continue
    eval_labels = [label(r) for r in eval_]
    for gran in GRANULARITIES:
        for tgt in TARGETS:
            thresh = {
                "drift":    fit_threshold(cal, "drift",    tgt, gran),
                "variance": fit_threshold(cal, "variance", tgt, gran),
                "etc":      fit_threshold(cal, "etc",      tgt, gran),
                "dragin":   fit_threshold(cal, "dragin",   tgt, gran),
                "hvr":      0.0,
            }
            dfs = {}
            for g in GATES:
                fires = gate_fire_per_trace(eval_, g, thresh, random_p=tgt)
                df, _ = delta_frac(fires, eval_labels)
                dfs[g] = df
            results.append({"arch": arch, "gran": gran, "tgt": tgt, "dfs": dfs})
            # winner (excluding hvr if only 1 fire)
            valid = {g: d for g, d in dfs.items() if not (d != d)}  # not NaN
            winner = max(valid, key=valid.get) if valid else "—"
            row = f"{arch:6} {gran:5} {tgt:5.2f} | "
            for g in GATES:
                v = dfs[g]
                row += f"{v:>+8.3f}  " if v == v else f"{'  n/a':>8}  "
            row += f"||  {winner:8} ({dfs.get(winner, 0):+.3f})"
            print(row)

# Aggregated views
print("\n" + "=" * 80)
print("WINNER FREQUENCY (across 24 cells = 3 archs × 2 gran × 4 targets):")
from collections import Counter
winners = []
for r in results:
    valid = {g: d for g, d in r["dfs"].items() if d == d}
    if valid:
        winners.append(max(valid, key=valid.get))
c = Counter(winners)
for g in GATES:
    print(f"  {g:10}  wins {c.get(g, 0):>2} / {len(results)} cells")

print("\nPER-ARCH WINNERS:")
for arch in ARCHS:
    sub = [r for r in results if r["arch"] == arch]
    cw = Counter()
    for r in sub:
        valid = {g: d for g, d in r["dfs"].items() if d == d}
        if valid:
            cw[max(valid, key=valid.get)] += 1
    print(f"  {arch:6}: " + ", ".join(f"{g}={cw[g]}" for g in GATES if cw[g] > 0))

print("\nPER-GATE Δfrac DISTRIBUTION (across all 24 cells):")
import statistics
for g in GATES:
    vals = [r["dfs"][g] for r in results if r["dfs"][g] == r["dfs"][g]]
    if vals:
        print(f"  {g:10}  mean={statistics.mean(vals):+.3f}  median={statistics.median(vals):+.3f}  "
              f"min={min(vals):+.3f}  max={max(vals):+.3f}  std={statistics.pstdev(vals):.3f}  "
              f"n_cells={len(vals)}")
