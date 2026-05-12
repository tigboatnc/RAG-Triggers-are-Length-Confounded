"""ITERATION 7 — Direct head-to-head: variance/max vs u_cons.max vs DRAGIN
on the lift-conditional metric at deployment fire rates.

Earlier we showed u_cons.max gives 3.3× lift at 5% fire on Hot. What about
the v1 latent variance_l2_window/max? AUROC says 0.654, so it should produce
a similar deployment table.
"""
from __future__ import annotations
import json
import sys
import statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
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
        u = sv(cr, "u_cons"); u_max = max(u) if u else None
        v = sv(cr, "variance_l2_window"); v_max = max(v) if v else None
        d = sv(cr, "dragin_rind_max"); d_max = max(d) if d else None
        drift = sv(cr, "drift_cosine"); drift_last = drift[-1] if drift else None
        hn = sv(cr, "hidden_norm"); hn_range = (max(hn) - min(hn)) if hn else None
        rows.append({"id": qid, "u_cons_max": u_max, "var_max": v_max,
                     "dragin_max": d_max, "drift_last": drift_last,
                     "hn_range": hn_range, "lift_em": lift, "closed_em": cem})
    return [r for r in rows if r["u_cons_max"] is not None]


def is_help(r): return r["lift_em"] == +1
def is_incorrect(r): return not r["closed_em"]


def deploy_table(rows, key, target_fires=(0.05, 0.10, 0.15, 0.20, 0.30)):
    vals = sorted([(r[key], r) for r in rows if r[key] is not None], key=lambda x: -x[0])
    n = len(vals)
    base_help = sum(1 for _, r in vals if is_help(r)) / n
    base_inc = sum(1 for _, r in vals if is_incorrect(r)) / n
    print(f"    base helpful={base_help:.3f}, base incorrect={base_inc:.3f}")
    print(f"    {'target':>6} {'fire':>6} {'P(help|F)':>10} {'lift':>5} {'P(harm|F)':>10} {'P(inc|F)':>9} {'Δprec_inc':>10}")
    for tgt in target_fires:
        k = max(1, int(tgt * n))
        fired = vals[:k]
        n_fire = len(fired)
        n_help = sum(1 for _, r in fired if is_help(r))
        n_harm = sum(1 for _, r in fired if r["lift_em"] == -1)
        n_inc  = sum(1 for _, r in fired if is_incorrect(r))
        prec_help = n_help / n_fire
        prec_inc  = n_inc / n_fire
        print(f"    {tgt:>5.2f}  {n_fire/n*100:>5.1f}% {prec_help:>10.3f} {prec_help/base_help if base_help else 0:>5.2f}× {n_harm/n_fire:>10.3f} {prec_inc:>9.3f} {prec_inc-base_inc:>+10.3f}")


def paired_auroc_diff(rows, key_a, key_b, target_fn, B=1000):
    import random
    n = len(rows)
    diffs = []
    for sd_ in range(B):
        rng = random.Random(sd_)
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        pa = [r[key_a] for r in sample if target_fn(r) and r[key_a] is not None]
        na = [r[key_a] for r in sample if not target_fn(r) and r[key_a] is not None]
        pb = [r[key_b] for r in sample if target_fn(r) and r[key_b] is not None]
        nb = [r[key_b] for r in sample if not target_fn(r) and r[key_b] is not None]
        au_a = auroc(pa, na); au_b = auroc(pb, nb)
        if au_a == au_a and au_b == au_b:
            diffs.append(au_a - au_b)
    if not diffs: return None
    diffs.sort()
    m = statistics.mean(diffs)
    lo = diffs[int(0.025*(len(diffs)-1))]; hi = diffs[int(0.975*(len(diffs)-1))]
    return {"mean": m, "ci_lo": lo, "ci_hi": hi, "P_gt": sum(1 for d in diffs if d>0)/len(diffs)}


for label, ds in (("HotpotQA", load("hot")), ("2WikiMHQA", load("wiki"))):
    print(f"\n{'='*80}\n=== {label}  n={len(ds)} ===\n{'='*80}")
    # AUROC summary
    print(f"\n  AUROC for predicting lift_em=+1:")
    for k in ("u_cons_max", "var_max", "dragin_max", "drift_last", "hn_range"):
        pos = [r[k] for r in ds if is_help(r) and r[k] is not None]
        neg = [r[k] for r in ds if not is_help(r) and r[k] is not None]
        au = auroc(pos, neg)
        print(f"    {k:14}  AUROC = {au:.3f}  (n+={len(pos)}/n-={len(neg)})")

    print(f"\n  AUROC for predicting INCORRECTNESS:")
    for k in ("u_cons_max", "var_max", "dragin_max", "drift_last", "hn_range"):
        pos = [r[k] for r in ds if is_incorrect(r) and r[k] is not None]
        neg = [r[k] for r in ds if not is_incorrect(r) and r[k] is not None]
        au = auroc(pos, neg)
        print(f"    {k:14}  AUROC = {au:.3f}")

    # Paired diffs vs DRAGIN
    print(f"\n  Paired AUROC differences (vs DRAGIN, B=1000 bootstrap):")
    for k in ("u_cons_max", "var_max", "drift_last", "hn_range"):
        d = paired_auroc_diff(ds, k, "dragin_max", is_help, B=1000)
        if d:
            sig = " ***" if d["ci_lo"] > 0 else ("  -" if d["ci_hi"] < 0 else "")
            print(f"    {k:14}  Δ={d['mean']:+.3f}  95% CI [{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}]  P(>0)={d['P_gt']:.3f}{sig}")

    print(f"\n  DEPLOYMENT TABLE for v1 LATENT (variance_l2_window/max):")
    deploy_table(ds, "var_max")
    print(f"\n  DEPLOYMENT TABLE for u_cons.max:")
    deploy_table(ds, "u_cons_max")
    print(f"\n  DEPLOYMENT TABLE for DRAGIN.max:")
    deploy_table(ds, "dragin_max")
    print(f"\n  DEPLOYMENT TABLE for drift_cosine/last (CRV at commitment moment):")
    deploy_table(ds, "drift_last")
