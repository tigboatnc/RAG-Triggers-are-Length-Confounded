"""ITERATION 2 — Feature-selection stability under split-half.

For each (arch, dataset), test:
  - Across 50 random calibrate/eval splits, what fraction agree on the
    top-1 feature?
  - What's the eval-half AUROC of the calibrate-half-selected top feature,
    averaged over the splits?
  - This is the precondition for any "pick a feature on cal, freeze it for
    test" claim at n=500.

Pool of features = best-of-aggregation per per-step scalar, plus a few
trajectory shapes (the strongest ones from try9).
"""
from __future__ import annotations
import random
import sys
import statistics
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load_traces, label, auroc

ARCHS = ["phi", "gemma", "qwen", "phi_hotpot_v2", "phi_2wiki_v2"]


def f(v):
    try: return float(v) if v is not None else None
    except: return None


def step_vals(r, key):
    return [v for v in (f(s.get(key)) for s in r.get("steps") or []) if v is not None]


# Candidate features (name, extractor)
def mk_pointwise(key, agg):
    def fn(r):
        vs = step_vals(r, key)
        if not vs: return None
        if agg == "max": return max(vs)
        if agg == "mean": return sum(vs)/len(vs)
        if agg == "sum": return sum(vs)
        if agg == "last": return vs[-1]
        if agg == "first": return vs[0]
    return fn


def feat_monotonicity(key):
    def fn(r):
        vs = step_vals(r, key)
        if len(vs) < 2: return None
        return sum(1 for i in range(len(vs)-1) if vs[i+1]>vs[i])/(len(vs)-1)
    return fn


def feat_range(key):
    def fn(r):
        vs = step_vals(r, key)
        return max(vs)-min(vs) if vs else None
    return fn


def feat_slope(key):
    def fn(r):
        vs = step_vals(r, key)
        if len(vs) < 2: return None
        n=len(vs); sx=sum(range(n)); sy=sum(vs); sxx=sum(i*i for i in range(n)); sxy=sum(i*v for i,v in enumerate(vs))
        d=n*sxx-sx*sx
        return (n*sxy-sx*sy)/d if d else None
    return fn


def feat_trace(key):
    def fn(r): return float(r.get(key, 0))
    return fn


CANDS = []
for key in ["drift_cosine","variance_l2_window","hidden_norm","mean_token_entropy",
            "boundary_token_entropy","dragin_rind_max","u_cons","u_verb"]:
    for agg in ("max","mean","sum","last","first"):
        CANDS.append((f"{key}/{agg}", mk_pointwise(key, agg)))
    CANDS.append((f"{key}/monoton",  feat_monotonicity(key)))
    CANDS.append((f"{key}/range",    feat_range(key)))
    CANDS.append((f"{key}/slope",    feat_slope(key)))

for tkey in ("n_steps","tokens_generated","wall_clock_s"):
    CANDS.append((f"{tkey}/trace", feat_trace(tkey)))


def featurize(rows):
    out = []
    for r in rows:
        feat = {}
        for name, fn in CANDS:
            try: feat[name] = fn(r)
            except: feat[name] = None
        out.append((feat, label(r)))
    return out


def feat_auroc(records, name):
    pos = [rec[0][name] for rec in records if not rec[1] and rec[0].get(name) is not None]
    neg = [rec[0][name] for rec in records if rec[1] and rec[0].get(name) is not None]
    return auroc(pos, neg)


N_SPLITS = 50
print(f"ITER 2 — Feature-selection stability ({N_SPLITS} random calibrate/eval split-halves)\n")

for arch in ARCHS:
    try: rows = load_traces(arch, min_steps=3)
    except: continue
    if not rows: continue
    recs = featurize(rows)
    n = len(recs)

    top_picks = []
    eval_aurocs = []
    full_aurocs = {}
    for sd_ in range(N_SPLITS):
        rng = random.Random(sd_)
        idx = list(range(n)); rng.shuffle(idx)
        cal = [recs[i] for i in idx[:n//2]]; ev = [recs[i] for i in idx[n//2:]]
        # On calibration: pick feature with max |AUROC − 0.5|
        cal_aurocs = {}
        for name, _ in CANDS:
            au = feat_auroc(cal, name)
            if au == au:
                cal_aurocs[name] = au
        if not cal_aurocs: continue
        best = max(cal_aurocs, key=lambda n: abs(cal_aurocs[n] - 0.5))
        top_picks.append(best)
        # Evaluate on held-out: AUROC of that feature
        ev_au = feat_auroc(ev, best)
        # Use cal-determined sign
        if cal_aurocs[best] < 0.5:
            ev_au = 1 - ev_au if ev_au == ev_au else ev_au
        if ev_au == ev_au:
            eval_aurocs.append(ev_au)

    # Also compute full-data AUROC for reporting
    for name, _ in CANDS:
        au = feat_auroc(recs, name)
        if au == au:
            full_aurocs[name] = au
    if not full_aurocs: continue
    full_best = max(full_aurocs, key=lambda n: abs(full_aurocs[n] - 0.5))

    pick_counts = Counter(top_picks)
    most_common, count = pick_counts.most_common(1)[0] if pick_counts else ("—", 0)
    mean_eval = statistics.mean(eval_aurocs) if eval_aurocs else float("nan")
    std_eval  = statistics.pstdev(eval_aurocs) if len(eval_aurocs) > 1 else 0
    agreement = count / N_SPLITS if N_SPLITS else 0

    print(f"=== {arch}  (n={n}) ===")
    print(f"  full-data best feature: {full_best:35} AUROC = {full_aurocs[full_best]:.3f}")
    print(f"  most-common cal pick:   {most_common:35} chosen in {count}/{N_SPLITS} splits ({agreement*100:.0f}%)")
    print(f"  eval-half AUROC of cal-picked feature: mean={mean_eval:.3f}  std={std_eval:.3f}  "
          f"(95% interval ≈ [{mean_eval-2*std_eval:.3f}, {mean_eval+2*std_eval:.3f}])")
    # Top 5 most-picked features
    print(f"  top 5 most-picked features across splits:")
    for name, c in pick_counts.most_common(5):
        ad = full_aurocs.get(name)
        print(f"    {name:35} picked {c}/{N_SPLITS}  full-data AUROC={ad:.3f}")
    print()
