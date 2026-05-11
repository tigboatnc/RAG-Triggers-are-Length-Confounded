"""TRY 7 — Full-feature AUROC bake-off.

For every per-step scalar feature on disk, compute trace-level AUROC.
v1 only tested {drift_cosine, variance_l2_window, mean_token_entropy→etc,
hvr, dragin_rind_max}. The data has ~15 more features that were never gated on.

Per-step scalars on disk (from inspection of a phi v1 trace step record):
  boundary_token_entropy, dragin_n_content_tokens, dragin_rind_max,
  drift_cosine, hedge_count, hidden_norm, hvr, mean_token_entropy,
  n_tokens_in_step, u_cons, u_ent, u_verb, variance_l2_window, verify_count
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load_traces, label, auroc

ARCHS = ["phi", "gemma", "qwen", "phi_hotpot_v2", "phi_2wiki_v2"]

# Step-level features to test (skip non-numeric: step_text, boundary_token_id, etc)
STEP_FEATURES = [
    "drift_cosine", "variance_l2_window", "hidden_norm",
    "mean_token_entropy", "boundary_token_entropy",
    "hvr", "dragin_rind_max", "dragin_n_content_tokens",
    "hedge_count", "verify_count", "n_tokens_in_step",
    "u_cons", "u_ent", "u_verb",
]
TRACE_FEATURES = ["n_steps", "tokens_generated", "wall_clock_s"]

# Aggregations to try
AGGS = ["max", "mean", "sum", "last", "first"]


def safe_float(v):
    try: return float(v)
    except: return None


def aggregate(steps: list, key: str, agg: str) -> float | None:
    vals = [safe_float(s.get(key)) for s in steps]
    vals = [v for v in vals if v is not None]
    if not vals: return None
    if agg == "max":  return max(vals)
    if agg == "min":  return min(vals)
    if agg == "mean": return sum(vals) / len(vals)
    if agg == "sum":  return sum(vals)
    if agg == "last": return vals[-1]
    if agg == "first": return vals[0]
    return None


print("FULL-FEATURE AUROC BAKE-OFF (trace-level, prediction target = INCORRECT)\n")
print("Per-feature per-aggregation AUROC, across architectures. AUROC > 0.55 in bold-equivalent.")

for arch in ARCHS:
    try:
        rows = load_traces(arch, min_steps=1)
    except (KeyError, FileNotFoundError):
        continue
    if not rows: continue
    labels = [label(r) for r in rows]
    print(f"\n=== {arch}  n={len(rows)}  EM={(1-sum(not l for l in labels)/len(labels))*100:.1f}% (P(incorrect)={sum(not l for l in labels)/len(labels):.2f}) ===")
    print(f"{'feature':28}  " + "  ".join(f"{a:>6}" for a in AGGS))
    print("-" * 70)
    for key in STEP_FEATURES:
        row = f"{key:28}  "
        for agg in AGGS:
            pos = []; neg = []
            for r, l in zip(rows, labels):
                steps = r.get("steps") or []
                v = aggregate(steps, key, agg)
                if v is None: continue
                (pos if not l else neg).append(v)
            au = auroc(pos, neg)
            row += f"{au:>6.3f}  " if au == au else f"{'  n/a':>6}  "
        print(row)

    # Trace-level features
    print(f"\n  trace-level features:")
    for tkey in TRACE_FEATURES:
        pos = [float(r.get(tkey, 0)) for r, l in zip(rows, labels) if not l]
        neg = [float(r.get(tkey, 0)) for r, l in zip(rows, labels) if l]
        au = auroc(pos, neg)
        print(f"  {tkey:28}  {au:>6.3f}")

    # Top-3 features for this arch across all (feature, agg) pairs
    candidates = []
    for key in STEP_FEATURES:
        for agg in AGGS:
            pos = []; neg = []
            for r, l in zip(rows, labels):
                v = aggregate(r.get("steps") or [], key, agg)
                if v is None: continue
                (pos if not l else neg).append(v)
            au = auroc(pos, neg)
            if au == au:
                candidates.append((f"{key}/{agg}", au))
    for tkey in TRACE_FEATURES:
        pos = [float(r.get(tkey, 0)) for r, l in zip(rows, labels) if not l]
        neg = [float(r.get(tkey, 0)) for r, l in zip(rows, labels) if l]
        au = auroc(pos, neg)
        if au == au:
            candidates.append((f"{tkey}/trace", au))
    candidates.sort(key=lambda x: -abs(x[1] - 0.5))
    print(f"\n  TOP 5 features for {arch} by |AUROC − 0.5|:")
    for name, au in candidates[:5]:
        sign = "↑" if au > 0.5 else "↓"
        print(f"    {name:36}  AUROC = {au:.3f}  {sign} on incorrect")
