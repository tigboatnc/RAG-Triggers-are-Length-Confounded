"""ITERATION 4 — u_cons paired with each other feature: does anything push to 0.70+?

For Phi v2 (the actual run target), pair u_cons/max with every other feature
and z-norm combine. Identify the single best pair per dataset.

Also: test a learned-weight LR combiner via Welch's t-statistic weighting on the
calibration half, evaluate on held-out half (cheap, training-free in spirit —
just label-conditional means).
"""
from __future__ import annotations
import random
import sys
import statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load_traces, label, auroc

DATASETS = ["phi_hotpot_v2", "phi_2wiki_v2"]


def f(v):
    try: return float(v) if v is not None else None
    except: return None


def step_vals(r, key):
    return [v for v in (f(s.get(key)) for s in r.get("steps") or []) if v is not None]


def mk_features(r):
    """Return all candidate trace-level features."""
    out = {}
    for key in ["drift_cosine","variance_l2_window","hidden_norm","mean_token_entropy",
                "boundary_token_entropy","dragin_rind_max","u_cons","u_verb"]:
        vs = step_vals(r, key)
        if not vs:
            for agg in ("max","mean","sum","first","last","monoton","range","slope"):
                out[f"{key}/{agg}"] = None
            continue
        out[f"{key}/max"] = max(vs)
        out[f"{key}/mean"] = sum(vs)/len(vs)
        out[f"{key}/sum"] = sum(vs)
        out[f"{key}/first"] = vs[0]
        out[f"{key}/last"] = vs[-1]
        if len(vs) >= 2:
            out[f"{key}/monoton"] = sum(1 for i in range(len(vs)-1) if vs[i+1]>vs[i])/(len(vs)-1)
            out[f"{key}/range"] = max(vs)-min(vs)
            n=len(vs); sx=sum(range(n)); sy=sum(vs); sxx=sum(i*i for i in range(n)); sxy=sum(i*v for i,v in enumerate(vs))
            d=n*sxx-sx*sx
            out[f"{key}/slope"] = (n*sxy-sx*sy)/d if d else None
        else:
            out[f"{key}/monoton"] = None; out[f"{key}/range"] = None; out[f"{key}/slope"] = None
    out["n_steps"] = float(r.get("n_steps",0))
    out["tokens_generated"] = float(r.get("tokens_generated",0))
    out["wall_clock_s"] = float(r.get("wall_clock_s",0))
    return out


def featurize(rows):
    return [(mk_features(r), label(r)) for r in rows]


def feat_auroc(records, name):
    pos = [rec[0].get(name) for rec in records if not rec[1] and rec[0].get(name) is not None]
    neg = [rec[0].get(name) for rec in records if rec[1] and rec[0].get(name) is not None]
    return auroc(pos, neg)


def znorm_cal(records, name):
    vs = [rec[0].get(name) for rec in records if rec[0].get(name) is not None]
    if len(vs) < 2: return None, None
    m = statistics.mean(vs); s = statistics.pstdev(vs)
    return m, (s if s>0 else 1.0)


def combiner_auroc(records_eval, features_signs_norms):
    """features_signs_norms: list of (name, sign, (mean, std))."""
    pos, neg = [], []
    for rec in records_eval:
        s = 0; n = 0
        for name, sign, (m, sd_) in features_signs_norms:
            v = rec[0].get(name)
            if v is None: continue
            s += sign * (v - m) / sd_; n += 1
        if n == 0: continue
        (neg if rec[1] else pos).append(s)
    return auroc(pos, neg)


for arch in DATASETS:
    try: rows = load_traces(arch, min_steps=2)
    except: continue
    if not rows: continue
    recs = featurize(rows)
    n = len(recs)
    print(f"\n=== {arch}  n={n}  EM={sum(l for _,l in recs)/n*100:.1f}% ===\n")

    # Single-feature baseline: u_cons/max
    au_solo = feat_auroc(recs, "u_cons/max")
    print(f"baseline u_cons/max AUROC = {au_solo:.3f}\n")

    # Pair u_cons/max with every other feature.
    print(f"PAIRWISE COMBINERS (u_cons/max + other), full-data AUROC. Sorted by AUROC:")
    pair_results = []
    cands = sorted(recs[0][0].keys())
    for other in cands:
        if other == "u_cons/max": continue
        au_other = feat_auroc(recs, other)
        if au_other != au_other: continue
        # Build combiner with both, sign by direction
        s1 = 1 if au_solo > 0.5 else -1
        s2 = 1 if au_other > 0.5 else -1
        # z-norm on full data (not split-half — just diagnostic)
        m1, sd1 = znorm_cal(recs, "u_cons/max")
        m2, sd2 = znorm_cal(recs, other)
        if m1 is None or m2 is None: continue
        au_pair = combiner_auroc(recs, [("u_cons/max", s1, (m1,sd1)), (other, s2, (m2,sd2))])
        pair_results.append((other, au_other, au_pair, au_pair - max(au_solo, au_other)))
    pair_results.sort(key=lambda x: -x[2])
    print(f"  {'partner':36}  {'solo':>6}  {'pair':>6}  {'Δ vs best-component':>20}")
    for other, au_other, au_pair, delta in pair_results[:10]:
        flag = " ★" if delta > 0.01 else "  "
        print(f"  {other:36}  {au_other:>6.3f}  {au_pair:>6.3f}  {delta:>+20.3f}{flag}")

    # Triple: u_cons + tokens + something
    print(f"\nTRIPLES (u_cons/max + tokens_generated + other), full-data:")
    au_tok = feat_auroc(recs, "tokens_generated")
    m_tok, sd_tok = znorm_cal(recs, "tokens_generated")
    triple_results = []
    for other in cands:
        if other in ("u_cons/max","tokens_generated"): continue
        au_other = feat_auroc(recs, other)
        if au_other != au_other: continue
        s1 = 1 if au_solo > 0.5 else -1
        s2 = 1 if au_tok > 0.5 else -1
        s3 = 1 if au_other > 0.5 else -1
        m1, sd1 = znorm_cal(recs, "u_cons/max")
        m3, sd3 = znorm_cal(recs, other)
        au_tri = combiner_auroc(recs, [("u_cons/max",s1,(m1,sd1)),
                                       ("tokens_generated",s2,(m_tok,sd_tok)),
                                       (other,s3,(m3,sd3))])
        triple_results.append((other, au_other, au_tri))
    triple_results.sort(key=lambda x: -x[2])
    print(f"  baseline pair u_cons + tokens AUROC = ", end="")
    s1 = 1 if au_solo > 0.5 else -1; s2 = 1 if au_tok > 0.5 else -1
    m1, sd1 = znorm_cal(recs, "u_cons/max")
    print(f"{combiner_auroc(recs, [('u_cons/max',s1,(m1,sd1)), ('tokens_generated',s2,(m_tok,sd_tok))]):.3f}")
    print(f"  {'3rd partner':36}  {'solo':>6}  {'triple':>6}")
    for other, au_other, au_tri in triple_results[:8]:
        print(f"  {other:36}  {au_other:>6.3f}  {au_tri:>6.3f}")

    # Cross-validated pair (split-half, 50 reps): u_cons + best-pair-component
    print(f"\nCROSS-VALIDATED u_cons/max + tokens_generated (50 random split-halves):")
    aurocs = []
    for sd_ in range(50):
        rng = random.Random(sd_)
        idx = list(range(n)); rng.shuffle(idx)
        cal = [recs[i] for i in idx[:n//2]]; ev = [recs[i] for i in idx[n//2:]]
        # Calibrate signs and z-norms on cal
        au1c = feat_auroc(cal, "u_cons/max"); au2c = feat_auroc(cal, "tokens_generated")
        m1c, sd1c = znorm_cal(cal, "u_cons/max"); m2c, sd2c = znorm_cal(cal, "tokens_generated")
        if m1c is None or m2c is None: continue
        s1 = 1 if au1c > 0.5 else -1; s2 = 1 if au2c > 0.5 else -1
        au = combiner_auroc(ev, [("u_cons/max",s1,(m1c,sd1c)), ("tokens_generated",s2,(m2c,sd2c))])
        if au == au:
            aurocs.append(au)
    if aurocs:
        m = statistics.mean(aurocs); sd_ = statistics.pstdev(aurocs)
        print(f"  eval AUROC mean = {m:.3f}  std = {sd_:.3f}  95% band ≈ [{m-2*sd_:.3f}, {m+2*sd_:.3f}]")
