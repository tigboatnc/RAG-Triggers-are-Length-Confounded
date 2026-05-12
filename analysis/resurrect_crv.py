"""TEN-TRY RESURRECTION OF CRV/LATENT — last serious attempt.

Each "try" tests a different reframing of hidden-state-derived signals. The
goal is to find any combination that lands AUROC ≥ 0.65 on lift_em=+1 (the
deployment metric) on Phi v2, beating or matching DRAGIN's 0.732 in-domain
and 0.594 OOD.

If nothing here lands, the latent/CRV framing is genuinely dead.
"""
from __future__ import annotations
import json
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _common import auroc, sd

RUN_DIR = Path(__file__).resolve().parents[1] / "runs/v2_exp1_phi-4-mini-instruct"


def f(v):
    try: return float(v) if v is not None else None
    except: return None


def sv(r, key):
    return [v for v in (f(s.get(key)) for s in r.get("steps") or []) if v is not None]


def load(label):
    closed = RUN_DIR / ({"hot": "D1_closed_hotpot.jsonl", "wiki": "D2_closed_wiki.jsonl"}[label])
    oracle = RUN_DIR / ({"hot": "E1_oracle_hotpot.jsonl", "wiki": "E2_oracle_wiki.jsonl"}[label])
    c = {r["id"]: r for r in (json.loads(l) for l in open(closed))}
    o = {r["id"]: r for r in (json.loads(l) for l in open(oracle))}
    rows = []
    for qid, cr in c.items():
        if qid not in o: continue
        cem = int(bool(cr.get("label_correct_em", cr.get("em", False))))
        oem = int(bool(o[qid].get("label_correct_em", o[qid].get("em", False))))
        lift = +1 if oem > cem else (-1 if oem < cem else 0)
        rows.append({**cr, "_lift_em": lift, "_closed_em": cem})
    return [r for r in rows if (r.get("steps") or [])]


def trace_auroc(rows, feature_fn, target_fn):
    pos = []; neg = []
    for r in rows:
        v = feature_fn(r)
        if v is None: continue
        if target_fn(r): pos.append(v)
        else: neg.append(v)
    return auroc(pos, neg), len(pos), len(neg)


def is_incorrect(r): return not r["_closed_em"]
def is_lift_pos(r): return r["_lift_em"] == +1


HOT = load("hot")
WIKI = load("wiki")
print(f"Loaded: HOT n={len(HOT)} (lift+1={sum(1 for r in HOT if is_lift_pos(r))}, base={sum(1 for r in HOT if is_lift_pos(r))/len(HOT):.3f})")
print(f"        WIKI n={len(WIKI)} (lift+1={sum(1 for r in WIKI if is_lift_pos(r))}, base={sum(1 for r in WIKI if is_lift_pos(r))/len(WIKI):.3f})")
print()


def report(label, fn, target_label, target_fn):
    print(f"  {label:48}", end="")
    for nm, ds in (("Hot", HOT), ("Wiki", WIKI)):
        au, np_, nn = trace_auroc(ds, fn, target_fn)
        print(f"  {nm} {au:.3f} (n+={np_}/n-={nn})  ", end="")
    print()


def section(n, title):
    print(f"\n{'='*80}\nTRY R{n}: {title}\n{'='*80}")


# ----------------------------------------------------------------------
# R1: hidden_norm trajectory monotonicity & shape (no drift_cosine math)
section(1, "hidden_norm trajectory geometry (Atkinson-aligned)")
def hidden_monoton(r):
    vs = sv(r, "hidden_norm")
    if len(vs) < 2: return None
    return sum(1 for i in range(len(vs)-1) if vs[i+1]>vs[i])/(len(vs)-1)
def hidden_slope(r):
    vs = sv(r, "hidden_norm")
    if len(vs) < 2: return None
    n=len(vs); sx=sum(range(n)); sy=sum(vs); sxx=sum(i*i for i in range(n)); sxy=sum(i*v for i,v in enumerate(vs))
    d=n*sxx-sx*sx
    return (n*sxy-sx*sy)/d if d else None
def hidden_range(r):
    vs = sv(r, "hidden_norm")
    return max(vs)-min(vs) if vs else None
def hidden_last(r):
    vs = sv(r, "hidden_norm")
    return vs[-1] if vs else None
def hidden_terminal_growth(r):
    vs = sv(r, "hidden_norm")
    if len(vs) < 4: return None
    return (sum(vs[-3:])/3) - (sum(vs[:3])/3)

print(f"\n  TARGET: closed-book INCORRECTNESS")
for nm, fn in [("hidden_norm/monoton", hidden_monoton),
               ("hidden_norm/slope",   hidden_slope),
               ("hidden_norm/range",   hidden_range),
               ("hidden_norm/last",    hidden_last),
               ("hidden_norm/terminal_growth", hidden_terminal_growth)]:
    report(nm, fn, "incorrect", is_incorrect)
print(f"\n  TARGET: LIFT_EM=+1 (deployment metric)")
for nm, fn in [("hidden_norm/monoton", hidden_monoton),
               ("hidden_norm/slope",   hidden_slope),
               ("hidden_norm/range",   hidden_range),
               ("hidden_norm/last",    hidden_last),
               ("hidden_norm/terminal_growth", hidden_terminal_growth)]:
    report(nm, fn, "lift+1", is_lift_pos)


# ----------------------------------------------------------------------
# R2: variance_l2_window monotonicity (the strongest from try9, now lift-conditional)
section(2, "variance_l2_window trajectory geometry — lift-conditional")
def var_monoton(r):
    vs = sv(r, "variance_l2_window")
    if len(vs) < 2: return None
    return sum(1 for i in range(len(vs)-1) if vs[i+1]>vs[i])/(len(vs)-1)
def var_slope(r):
    vs = sv(r, "variance_l2_window")
    if len(vs) < 2: return None
    n=len(vs); sx=sum(range(n)); sy=sum(vs); sxx=sum(i*i for i in range(n)); sxy=sum(i*v for i,v in enumerate(vs))
    d=n*sxx-sx*sx
    return (n*sxy-sx*sy)/d if d else None
def var_range(r):
    vs = sv(r, "variance_l2_window")
    return max(vs)-min(vs) if vs else None
def var_terminal_growth(r):
    vs = sv(r, "variance_l2_window")
    if len(vs) < 4: return None
    return (sum(vs[-3:])/3) - (sum(vs[:3])/3)

print(f"\n  TARGET: closed-book INCORRECTNESS")
for nm, fn in [("variance_l2_window/monoton", var_monoton),
               ("variance_l2_window/slope", var_slope),
               ("variance_l2_window/range", var_range),
               ("variance_l2_window/terminal_growth", var_terminal_growth)]:
    report(nm, fn, "incorrect", is_incorrect)
print(f"\n  TARGET: LIFT_EM=+1 (deployment metric)")
for nm, fn in [("variance_l2_window/monoton", var_monoton),
               ("variance_l2_window/slope", var_slope),
               ("variance_l2_window/range", var_range),
               ("variance_l2_window/terminal_growth", var_terminal_growth)]:
    report(nm, fn, "lift+1", is_lift_pos)


# ----------------------------------------------------------------------
# R3: drift_cosine LATE-TRACE (last 2 steps), not max-over-trace
section(3, "drift_cosine at the answer-commitment moment (last steps)")
def drift_last(r):
    vs = sv(r, "drift_cosine")
    return vs[-1] if vs else None
def drift_penult(r):
    vs = sv(r, "drift_cosine")
    return vs[-2] if len(vs) >= 2 else None
def drift_last2_mean(r):
    vs = sv(r, "drift_cosine")
    if len(vs) < 2: return None
    return (vs[-1] + vs[-2]) / 2
def drift_acceleration(r):
    vs = sv(r, "drift_cosine")
    if len(vs) < 3: return None
    return vs[-1] - vs[-2]  # how much drift grew on the final step
print(f"\n  TARGET: LIFT_EM=+1")
for nm, fn in [("drift_cosine/last",         drift_last),
               ("drift_cosine/penult",       drift_penult),
               ("drift_cosine/last2_mean",   drift_last2_mean),
               ("drift_cosine/acceleration", drift_acceleration)]:
    report(nm, fn, "lift+1", is_lift_pos)


# ----------------------------------------------------------------------
# R4: latent + DRAGIN interaction (z-norm linear combiner)
section(4, "latent × DRAGIN complementarity — Chen INSIDE-style fusion")
def zscore_combine(rows, fns_with_dirs):
    # First pass: compute mean/std per feature
    feats = [[fn(r) for fn in fns_with_dirs.keys()] for r in rows]
    stats = []
    for j in range(len(fns_with_dirs)):
        vals = [f[j] for f in feats if f[j] is not None]
        if not vals: stats.append((0, 1)); continue
        m = statistics.mean(vals)
        s = statistics.pstdev(vals) if len(vals)>1 else 1.0
        stats.append((m, s or 1.0))
    # Second pass: z-norm + signed sum
    out = []
    for f in feats:
        s = 0; ok = True
        for j, (fn, sign) in enumerate(fns_with_dirs.items()):
            if f[j] is None: ok = False; break
            m, sd_ = stats[j]
            s += sign * (f[j] - m) / sd_
        out.append(s if ok else None)
    return out

def latent_var_max(r):
    vs = sv(r, "variance_l2_window")
    return max(vs) if vs else None
def latent_var_monoton(r):
    vs = sv(r, "variance_l2_window")
    if len(vs)<2: return None
    return sum(1 for i in range(len(vs)-1) if vs[i+1]>vs[i])/(len(vs)-1)
def dragin_max_(r):
    vs = sv(r, "dragin_rind_max")
    return max(vs) if vs else None

for nm, ds in (("Hot", HOT), ("Wiki", WIKI)):
    print(f"\n  {nm}:")
    # baseline
    au_v, _, _ = trace_auroc(ds, latent_var_max, is_lift_pos)
    au_d, _, _ = trace_auroc(ds, dragin_max_, is_lift_pos)
    au_vm, _, _ = trace_auroc(ds, latent_var_monoton, is_lift_pos)
    print(f"    variance/max alone        AUROC(lift+1) = {au_v:.3f}")
    print(f"    dragin/max alone          AUROC(lift+1) = {au_d:.3f}")
    print(f"    variance/monoton alone    AUROC(lift+1) = {au_vm:.3f}")
    # pair: variance/max + dragin
    combo = zscore_combine(ds, {latent_var_max: 1 if au_v>0.5 else -1,
                                dragin_max_:     1 if au_d>0.5 else -1})
    pos = [v for v, r in zip(combo, ds) if v is not None and is_lift_pos(r)]
    neg = [v for v, r in zip(combo, ds) if v is not None and not is_lift_pos(r)]
    print(f"    variance/max + dragin     AUROC(lift+1) = {auroc(pos, neg):.3f}")
    # pair: variance/monoton + dragin
    combo = zscore_combine(ds, {latent_var_monoton: 1 if au_vm>0.5 else -1,
                                dragin_max_:        1 if au_d>0.5 else -1})
    pos = [v for v, r in zip(combo, ds) if v is not None and is_lift_pos(r)]
    neg = [v for v, r in zip(combo, ds) if v is not None and not is_lift_pos(r)]
    print(f"    variance/monoton + dragin AUROC(lift+1) = {auroc(pos, neg):.3f}")
    # triple
    combo = zscore_combine(ds, {latent_var_max: 1 if au_v>0.5 else -1,
                                latent_var_monoton: 1 if au_vm>0.5 else -1,
                                dragin_max_:     1 if au_d>0.5 else -1})
    pos = [v for v, r in zip(combo, ds) if v is not None and is_lift_pos(r)]
    neg = [v for v, r in zip(combo, ds) if v is not None and not is_lift_pos(r)]
    print(f"    var/max + var/monoton + dragin AUROC(lift+1) = {auroc(pos, neg):.3f}")


# ----------------------------------------------------------------------
# R5: hidden_norm × variance × drift trajectory composite
section(5, "multi-scalar CRV composite — Kossen-style internal probe")
for nm, ds in (("Hot", HOT), ("Wiki", WIKI)):
    print(f"\n  {nm}:")
    # individual AUROCs for direction calibration
    feats = {hidden_monoton: "hidden_monoton",
             var_monoton: "var_monoton",
             hidden_terminal_growth: "hidden_terminal_growth",
             var_terminal_growth: "var_terminal_growth",
             drift_last: "drift_last"}
    dirs = {}
    for fn in feats:
        au, _, _ = trace_auroc(ds, fn, is_lift_pos)
        dirs[fn] = 1 if au > 0.5 else -1
        print(f"    {feats[fn]:30}  AUROC = {au:.3f}")
    combo = zscore_combine(ds, dirs)
    pos = [v for v, r in zip(combo, ds) if v is not None and is_lift_pos(r)]
    neg = [v for v, r in zip(combo, ds) if v is not None and not is_lift_pos(r)]
    print(f"    COMPOSITE                       AUROC = {auroc(pos, neg):.3f}")


# ----------------------------------------------------------------------
# R6: pairwise step coherence (cross-step trajectory, not adjacent)
section(6, "cross-step coherence matrix — internal trajectory consistency")
def pairwise_drift_mean(r, key):
    vs = sv(r, key)
    if len(vs) < 3: return None
    # all pairs |v_i - v_j|, mean
    n = len(vs)
    diffs = [abs(vs[i] - vs[j]) for i in range(n) for j in range(i+1, n)]
    return sum(diffs) / len(diffs) if diffs else None

def pw_hidden(r): return pairwise_drift_mean(r, "hidden_norm")
def pw_var(r): return pairwise_drift_mean(r, "variance_l2_window")
def pw_drift(r): return pairwise_drift_mean(r, "drift_cosine")
print(f"\n  TARGET: LIFT_EM=+1")
for nm, fn in [("hidden_norm pairwise mean diff", pw_hidden),
               ("variance pairwise mean diff", pw_var),
               ("drift_cosine pairwise mean diff", pw_drift)]:
    report(nm, fn, "lift+1", is_lift_pos)


# ----------------------------------------------------------------------
# R7: lift-positive class SELECTIVITY (does CRV detect fixable wrongs?)
section(7, "Is CRV selective for FIXABLE (lift+1) within incorrect class?")
def is_lift_pos_within_incorrect(r):
    if r["_closed_em"]: return None  # exclude correct
    return r["_lift_em"] == +1

for nm, ds in (("Hot", HOT), ("Wiki", WIKI)):
    print(f"\n  {nm} (within incorrect class only):")
    incorrect = [r for r in ds if not r["_closed_em"]]
    for feat_nm, fn in [("hidden_norm/monoton", hidden_monoton),
                        ("variance/monoton", var_monoton),
                        ("drift_cosine/last", drift_last),
                        ("hidden_norm/terminal_growth", hidden_terminal_growth),
                        ("hidden_norm/last", hidden_last)]:
        pos = [fn(r) for r in incorrect if r["_lift_em"] == +1 and fn(r) is not None]
        neg = [fn(r) for r in incorrect if r["_lift_em"] != +1 and fn(r) is not None]
        au = auroc(pos, neg)
        print(f"    {feat_nm:30}  AUROC(lift+1 | incorrect) = {au:.3f}  (n+={len(pos)}, n-={len(neg)})")


# ----------------------------------------------------------------------
# R8: hidden_norm magnitude at the answer step (commitment moment)
section(8, "hidden_norm AT the answer step — geometric commitment")
def hidden_terminal_normalized(r):
    """hidden_norm at last step / mean(hidden_norm of earlier steps)"""
    vs = sv(r, "hidden_norm")
    if len(vs) < 3: return None
    early = sum(vs[:-1]) / (len(vs) - 1)
    if early == 0: return None
    return vs[-1] / early
print(f"\n  TARGET: LIFT_EM=+1")
for nm, fn in [("hidden_norm last / early-mean", hidden_terminal_normalized),
               ("hidden_norm last",              hidden_last),
               ("hidden_norm last − early-mean", lambda r: (sv(r,"hidden_norm")[-1] - sum(sv(r,"hidden_norm")[:-1])/(len(sv(r,"hidden_norm"))-1)) if len(sv(r,"hidden_norm"))>=2 else None)]:
    report(nm, fn, "lift+1", is_lift_pos)


# ----------------------------------------------------------------------
# R9: drift_cosine + variance OR-rule, but with shape conditioning
section(9, "shape-conditioned OR rule (v1 #22 revisited with shape features)")
print(f"\nv1 OR rule: fire if drift>τ_drift OR variance>τ_var (pointwise).")
print(f"Replace with: fire if (variance monoton > q) OR (hidden_norm monoton > q).")
def or_shape_score(r):
    v = var_monoton(r); h = hidden_monoton(r)
    if v is None or h is None: return None
    return max(v, h)  # OR-equivalent at score level
print(f"\n  TARGET: INCORRECTNESS")
report("OR(var_monoton, hidden_monoton) max", or_shape_score, "incorrect", is_incorrect)
print(f"\n  TARGET: LIFT_EM=+1")
report("OR(var_monoton, hidden_monoton) max", or_shape_score, "lift+1", is_lift_pos)


# ----------------------------------------------------------------------
# R10: pre-registered combined "CRV trajectory" deployment evaluation
section(10, "deployment evaluation at 5% / 10% / 20% fire-rate")
# Pick best feature from above: variance_l2_window/monoton on Hot
print(f"\nFor each candidate latent feature, deployment-edge precision:")
for nm, fn in [("variance_l2_window/monoton", var_monoton),
               ("hidden_norm/monoton",        hidden_monoton),
               ("hidden_norm/terminal_growth", hidden_terminal_growth),
               ("drift_cosine/last",          drift_last)]:
    print(f"\n  Feature: {nm}")
    for label, ds in (("Hot", HOT), ("Wiki", WIKI)):
        vals = [(fn(r), r) for r in ds if fn(r) is not None]
        if not vals: continue
        # direction (higher = more incorrect?)
        au, _, _ = trace_auroc(ds, fn, is_lift_pos)
        higher_is_lift = au > 0.5
        vals_sorted = sorted([v for v,_ in vals], reverse=higher_is_lift)
        n = len(vals)
        base_lift = sum(1 for _, r in vals if is_lift_pos(r)) / n
        for tgt in (0.05, 0.10, 0.20):
            k = int(tgt * n)
            if k == 0: continue
            tau = vals_sorted[k-1]
            fired = [(v, r) for v, r in vals if (v >= tau if higher_is_lift else v <= tau)]
            n_fire = len(fired)
            n_help = sum(1 for _, r in fired if is_lift_pos(r))
            n_harm = sum(1 for _, r in fired if r["_lift_em"] == -1)
            prec = n_help / n_fire if n_fire else 0
            print(f"    {label}  target {tgt*100:>4.0f}%  fire={n_fire/n*100:5.1f}%  P(help|fire)={prec:.3f} (base {base_lift:.3f}, lift {prec/base_lift if base_lift else 0:.2f}×)  harm={n_harm}/{n_fire}")
