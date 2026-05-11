"""TRY 5 — Best-gate-per-trace oracle + ensemble upper bound.

Two related questions:
  (a) Recall ceiling: of all incorrect traces, what fraction has at-least-one
      gate that fires? (Union recall.) If this is much higher than any single
      gate's recall, an ensemble could outperform any individual gate.
  (b) Per-pair disjoint coverage: do drift and variance fire on disjoint pools
      of incorrect traces (v1 #22's claim) or overlapping ones?
  (c) Best-per-trace oracle Δfrac: if we could pick the best gate per trace,
      what Δfrac would we get? Gives the absolute ceiling.

Calibrate all gates at trace-iso-0.30 on calibration split; evaluate on val+test.
"""
from __future__ import annotations
import sys
import itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (load_traces, label, fit_threshold, gate_fire_per_trace,
                     delta_frac, split_traces)

ARCHS = ["phi", "gemma", "qwen"]
GATES = ["drift", "variance", "etc", "hvr", "dragin"]  # exclude OR (a derivative), random
TARGET = 0.30

for arch in ARCHS:
    traces = load_traces(arch)
    splits = split_traces(traces)
    cal = splits["calibration"]
    eval_ = splits["validation"] + splits["test"]
    if not cal or not eval_:
        continue
    eval_labels = [label(r) for r in eval_]
    n = len(eval_)
    n_incorrect = sum(1 for l in eval_labels if not l)

    thresh = {
        "drift":    fit_threshold(cal, "drift",    TARGET, "trace"),
        "variance": fit_threshold(cal, "variance", TARGET, "trace"),
        "etc":      fit_threshold(cal, "etc",      TARGET, "trace"),
        "dragin":   fit_threshold(cal, "dragin",   TARGET, "trace"),
        "hvr":      0.0,
    }

    # Per-trace gate-fire matrix
    fires = {g: gate_fire_per_trace(eval_, g, thresh, random_p=TARGET) for g in GATES}

    print(f"\n=== {arch}  (n_eval={n}, n_incorrect={n_incorrect}, EM={(n-n_incorrect)/n*100:.1f}%) ===")

    print(f"\nGate-level recall on INCORRECT traces (P(gate fires | incorrect)):")
    incorrect_recalls = {}
    for g in GATES:
        fired_and_incorrect = sum(1 for i in range(n) if fires[g][i] and not eval_labels[i])
        recall = fired_and_incorrect / n_incorrect if n_incorrect else 0
        incorrect_recalls[g] = recall
        df, info = delta_frac(fires[g], eval_labels)
        print(f"  {g:10}  recall={recall:.3f}  fire-rate={info['fire_rate']:.3f}  Δfrac={df:+.3f}")

    # Union recall (any gate fires)
    union_recall = sum(1 for i in range(n) if any(fires[g][i] for g in GATES) and not eval_labels[i]) / n_incorrect
    print(f"\n  UNION (any gate fires) recall on incorrect = {union_recall:.3f}")
    print(f"  Best single-gate recall = {max(incorrect_recalls.values()):.3f}")
    print(f"  Δrecall (union − best single) = {union_recall - max(incorrect_recalls.values()):+.3f}")

    # Per-pair Jaccard (disjoint coverage) on the incorrect-trace set
    print(f"\nPer-pair OVERLAP on incorrect traces (Jaccard of fire-on-incorrect sets, 1.0 = identical, 0.0 = disjoint):")
    sets = {g: {i for i in range(n) if fires[g][i] and not eval_labels[i]} for g in GATES}
    print(f"  {'':10}  " + "  ".join(f"{g:>10}" for g in GATES))
    for g1 in GATES:
        row = f"  {g1:10}  "
        for g2 in GATES:
            if g1 == g2:
                row += f"{'—':>10}  "
            else:
                u = sets[g1] | sets[g2]; i = sets[g1] & sets[g2]
                j = len(i) / len(u) if u else 0
                row += f"{j:>10.2f}  "
        print(row)

    # Best-per-trace ORACLE (cheating ceiling)
    # For each trace, pick the gate that fires iff trace is incorrect.
    oracle_correct = sum(1 for i in range(n)
                         if (not eval_labels[i] and any(fires[g][i] for g in GATES))
                         or (eval_labels[i] and not any(fires[g][i] for g in GATES)))
    print(f"\n  ORACLE-best-gate-per-trace correct decisions: {oracle_correct}/{n} = {oracle_correct/n*100:.1f}%")
    # Compare to majority-class
    base = max(n_incorrect, n - n_incorrect) / n
    print(f"  vs majority-class baseline: {base*100:.1f}%")

    # Ensemble Δfrac: OR of all gates
    ens_fires = [any(fires[g][i] for g in GATES) for i in range(n)]
    df_ens, info_ens = delta_frac(ens_fires, eval_labels)
    print(f"  Ensemble (any gate fires) Δfrac = {df_ens:+.3f}  (fire-rate {info_ens['fire_rate']:.3f})")

    # AND ensemble (all fire)
    and_fires = [all(fires[g][i] for g in GATES) for i in range(n)]
    df_and, info_and = delta_frac(and_fires, eval_labels)
    print(f"  Ensemble (ALL gates fire) Δfrac = {df_and:+.3f}  (fire-rate {info_and['fire_rate']:.3f})")

print("\n" + "=" * 80)
print("READING:")
print("  - If union_recall >> best_single_recall: ensemble has headroom.")
print("  - If Jaccards are low: gates cover disjoint incorrect-trace pools (v1 #22).")
print("  - If Jaccards are high: gates are redundant; ensembling won't help much.")
