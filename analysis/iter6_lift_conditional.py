"""FINAL — u_cons.max lift-conditional AUROC and precision @ n=500.

Now that E1/E2 are at full 500, compute the lift-conditional version of
the u_cons hypothesis. Target = lift_em ∈ {+1 (helpful), 0 (neutral), -1 (harmful)}.

Compare:
  - AUROC for predicting lift_em=+1 from u_cons.max
  - At q70-τ deployment threshold, P(lift_em=+1 | fired)  (the actual paper claim)
"""
from __future__ import annotations
import json
import random
import statistics
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import auroc, sd

RUN_DIR = Path(__file__).resolve().parents[1] / "runs/v2_exp1_phi-4-mini-instruct"


def f(v):
    try: return float(v) if v is not None else None
    except: return None


def step_vals(r, key):
    return [v for v in (f(s.get(key)) for s in r.get("steps") or []) if v is not None]


def umax(r):
    vs = step_vals(r, "u_cons")
    return max(vs) if vs else None


def dragin_max(r):
    vs = step_vals(r, "dragin_rind_max")
    return max(vs) if vs else None


def load_closed_oracle(closed_path, oracle_path):
    closed = {r["id"]: r for r in (json.loads(l) for l in open(closed_path))}
    oracle = {r["id"]: r for r in (json.loads(l) for l in open(oracle_path))}
    merged = []
    for qid, c in closed.items():
        if qid not in oracle: continue
        o = oracle[qid]
        c_em = int(bool(c.get("label_correct_em", c.get("em", False))))
        o_em = int(bool(o.get("label_correct_em", o.get("em", False))))
        # lift_em = oracle helped (+1), neutral (0), harmful (-1)
        if o_em > c_em: lift = +1
        elif o_em < c_em: lift = -1
        else: lift = 0
        u = umax(c); d = dragin_max(c)
        merged.append({"id": qid, "u_cons_max": u, "dragin_max": d,
                       "closed_em": c_em, "oracle_em": o_em, "lift_em": lift,
                       "n_steps": int(c.get("n_steps", 0))})
    return merged


for label, cpath, opath in [
    ("HotpotQA (in-domain)", RUN_DIR/"D1_closed_hotpot.jsonl", RUN_DIR/"E1_oracle_hotpot.jsonl"),
    ("2WikiMHQA (OOD)",      RUN_DIR/"D2_closed_wiki.jsonl",   RUN_DIR/"E2_oracle_wiki.jsonl"),
]:
    if not cpath.exists() or not opath.exists():
        continue
    rows = load_closed_oracle(cpath, opath)
    rows = [r for r in rows if r["u_cons_max"] is not None and r["n_steps"] >= 1]
    n = len(rows)
    helpful = [r for r in rows if r["lift_em"] == +1]
    neutral = [r for r in rows if r["lift_em"] == 0]
    harmful = [r for r in rows if r["lift_em"] == -1]
    print(f"\n=== {label}  n={n}  (helpful={len(helpful)} / neutral={len(neutral)} / harmful={len(harmful)}, base_helpful={len(helpful)/n:.3f}) ===")

    # AUROC: u_cons.max → lift_em=+1
    pos_u = [r["u_cons_max"] for r in rows if r["lift_em"] == +1]
    neg_u = [r["u_cons_max"] for r in rows if r["lift_em"] != +1]  # neutral + harmful
    au_u = auroc(pos_u, neg_u)
    pos_d = [r["dragin_max"] for r in rows if r["lift_em"] == +1 and r["dragin_max"] is not None]
    neg_d = [r["dragin_max"] for r in rows if r["lift_em"] != +1 and r["dragin_max"] is not None]
    au_d = auroc(pos_d, neg_d)
    print(f"  AUROC(u_cons.max → lift_em=+1)  = {au_u:.3f}")
    print(f"  AUROC(dragin.max → lift_em=+1)  = {au_d:.3f}")

    # Paired bootstrap, 1000 reps at observed n
    diffs = []
    for sd_ in range(1000):
        rng = random.Random(sd_)
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        pu = [r["u_cons_max"] for r in sample if r["lift_em"] == +1]
        nu = [r["u_cons_max"] for r in sample if r["lift_em"] != +1]
        pd = [r["dragin_max"] for r in sample if r["lift_em"] == +1 and r["dragin_max"] is not None]
        nd = [r["dragin_max"] for r in sample if r["lift_em"] != +1 and r["dragin_max"] is not None]
        a_u = auroc(pu, nu); a_d = auroc(pd, nd)
        if a_u == a_u and a_d == a_d:
            diffs.append(a_u - a_d)
    if diffs:
        diffs.sort()
        m = statistics.mean(diffs)
        lo = diffs[int(0.025*(len(diffs)-1))]
        hi = diffs[int(0.975*(len(diffs)-1))]
        ppos = sum(1 for d in diffs if d > 0) / len(diffs)
        print(f"  Paired u_cons − dragin AUROC: mean={m:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  P(u_cons > dragin) = {ppos:.3f}")

    # Deployment q70-τ: P(lift_em=+1 | fired)
    u_sorted = sorted(r["u_cons_max"] for r in rows)
    print(f"\n  u_cons.max gate at various fire-rate targets (deployment edge):")
    print(f"  {'target':>7}  {'τ':>8}  {'fire%':>6}  {'P(lift+1|fired)':>18}  {'base':>6}  {'Δprec':>7}  {'P(harm|fired)':>15}")
    for tgt in (0.05, 0.10, 0.20, 0.30, 0.50):
        tau = u_sorted[int((1-tgt)*(len(u_sorted)-1))]
        fired = [r for r in rows if r["u_cons_max"] > tau]
        n_fire = len(fired)
        if n_fire == 0:
            print(f"  {tgt:>7.2f}  {tau:>8.4f}  no fires"); continue
        n_help_fire = sum(1 for r in fired if r["lift_em"] == +1)
        n_harm_fire = sum(1 for r in fired if r["lift_em"] == -1)
        base = len(helpful) / n
        prec = n_help_fire / n_fire
        harm_rate = n_harm_fire / n_fire
        print(f"  {tgt:>7.2f}  {tau:>8.4f}  {n_fire/n*100:>5.1f}%  {prec:>18.3f}  {base:>6.3f}  {prec-base:>+7.3f}  {harm_rate:>15.3f}")
