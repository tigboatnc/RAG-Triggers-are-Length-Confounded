"""ITERATION 5 — Pre-registration stress test.

For the candidate hypothesis "u_cons/max AUROC ≥ 0.60 on Phi v2 (HotpotQA &
2Wiki) at n=500", check:

  (a) Bootstrap AUROC at projected n=500 (resample existing 200 with replacement
      to size 500, 1000 reps). What's the 5th-percentile AUROC? That's our
      worst plausible n=500 outcome under the current data distribution.

  (b) Does u_cons.max PAIRED-significantly beat dragin_rind_max.max (the only
      v1 gate that survived audit)? Bootstrap the AUROC difference (paired by
      trace via per-trace u_cons − per-trace dragin contributions).

  (c) Low-fire-rate precision at q90, q95, q97 — the deployment edge. For
      retrieval to be worth triggering, P(incorrect | gate fired) needs to
      clearly exceed base-rate P(incorrect).

  (d) Pre-registered THRESHOLD: pick τ as the q70 of cal-split u_cons.max
      (target trace-fire-rate = 30%). On held-out test, report fire-rate AND
      P(incorrect | fired) and ΔPrecision (vs base rate).
"""
from __future__ import annotations
import random
import statistics
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load_traces, label, auroc

DATASETS = ["phi_hotpot_v2", "phi_2wiki_v2"]


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


def tokens(r):
    return float(r.get("tokens_generated", 0))


for arch in DATASETS:
    try: rows = load_traces(arch, min_steps=1)
    except: continue
    if not rows: continue
    triples = []
    for r in rows:
        u = umax(r); d = dragin_max(r); t = tokens(r); l = label(r)
        if u is None or d is None: continue
        triples.append((u, d, t, l))
    if not triples: continue
    n = len(triples)
    n_incorrect = sum(1 for *_, l in triples if not l)
    base_p_inc = n_incorrect / n
    print(f"\n=== {arch}  n={n}  base P(incorrect) = {base_p_inc:.3f} ===\n")

    # (a) Bootstrap AUROC at n=500
    print(f"(a) Projected n=500 bootstrap (resample to 500 with replacement, 1000 reps):")
    for stat_name, idx in [("u_cons.max", 0), ("dragin_max", 1), ("tokens_gen", 2)]:
        aus = []
        for sd_ in range(1000):
            rng = random.Random(sd_)
            sample = [triples[rng.randrange(n)] for _ in range(500)]
            pos = [s[idx] for s in sample if not s[-1]]; neg = [s[idx] for s in sample if s[-1]]
            au = auroc(pos, neg)
            if au == au: aus.append(au)
        aus.sort()
        if not aus: continue
        m = statistics.mean(aus); s5 = aus[int(0.05*(len(aus)-1))]; s95 = aus[int(0.95*(len(aus)-1))]
        print(f"  {stat_name:12}  median={statistics.median(aus):.3f}  "
              f"90% interval [{s5:.3f}, {s95:.3f}]  P(AUROC>0.60)={sum(1 for a in aus if a>0.60)/len(aus):.2f}  "
              f"P(AUROC>0.65)={sum(1 for a in aus if a>0.65)/len(aus):.2f}")

    # (b) Paired test: u_cons.max AUROC vs dragin_max AUROC at n=500
    print(f"\n(b) Paired bootstrap (u_cons.max − dragin_max AUROC), 1000 reps at n=500:")
    diffs = []
    for sd_ in range(1000):
        rng = random.Random(sd_)
        sample = [triples[rng.randrange(n)] for _ in range(500)]
        pos_u = [s[0] for s in sample if not s[-1]]; neg_u = [s[0] for s in sample if s[-1]]
        pos_d = [s[1] for s in sample if not s[-1]]; neg_d = [s[1] for s in sample if s[-1]]
        au_u = auroc(pos_u, neg_u); au_d = auroc(pos_d, neg_d)
        if au_u == au_u and au_d == au_d:
            diffs.append(au_u - au_d)
    diffs.sort()
    if diffs:
        m_d = statistics.mean(diffs); lo = diffs[int(0.025*(len(diffs)-1))]; hi = diffs[int(0.975*(len(diffs)-1))]
        p_pos = sum(1 for d in diffs if d > 0) / len(diffs)
        print(f"  mean diff = {m_d:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  P(u_cons > dragin) = {p_pos:.3f}")

    # (c) Low-fire-rate precision
    print(f"\n(c) u_cons.max as gate at low fire rates (deployment-edge precision):")
    u_vals_sorted = sorted([t[0] for t in triples])
    print(f"  {'target fire':>11}  {'τ':>8}  {'actual fire':>11}  {'P(incorrect|fired)':>20}  {'Δ vs base':>9}  {'recall':>7}  {'lift':>5}")
    for tgt in (0.05, 0.10, 0.15, 0.20, 0.30, 0.50):
        tau = u_vals_sorted[int((1-tgt)*(len(u_vals_sorted)-1))]
        fired = [t[0] > tau for t in triples]
        tp = sum(1 for f_,t in zip(fired,triples) if f_ and not t[-1])
        fp = sum(1 for f_,t in zip(fired,triples) if f_ and t[-1])
        fn = sum(1 for f_,t in zip(fired,triples) if not f_ and not t[-1])
        prec = tp/(tp+fp) if tp+fp else float("nan")
        recall = tp/(tp+fn) if tp+fn else float("nan")
        delta = prec - base_p_inc
        lift = prec / base_p_inc if base_p_inc else float("nan")
        print(f"  {tgt:>11.2f}  {tau:>8.4f}  {sum(fired)/n:>11.3f}  {prec:>20.3f}  {delta:>+9.3f}  {recall:>7.3f}  {lift:>5.2f}")

    # (d) Pre-registered τ: q70 of u_cons.max on first half, eval on second half
    print(f"\n(d) Pre-registered τ = q70(u_cons.max) on calibrate-half, eval on held-out half (50 split-halves):")
    fire_rates, precisions, recalls = [], [], []
    for sd_ in range(50):
        rng = random.Random(sd_)
        idx = list(range(n)); rng.shuffle(idx)
        cal = [triples[i] for i in idx[:n//2]]; ev = [triples[i] for i in idx[n//2:]]
        cal_u = sorted([t[0] for t in cal])
        tau = cal_u[int(0.7*(len(cal_u)-1))]
        fired = [t[0] > tau for t in ev]
        tp = sum(1 for f_,t in zip(fired,ev) if f_ and not t[-1])
        fp = sum(1 for f_,t in zip(fired,ev) if f_ and t[-1])
        fn = sum(1 for f_,t in zip(fired,ev) if not f_ and not t[-1])
        if tp+fp:
            fire_rates.append(sum(fired)/len(ev))
            precisions.append(tp/(tp+fp))
            recalls.append(tp/(tp+fn) if tp+fn else 0)
    if precisions:
        print(f"  fire-rate    mean={statistics.mean(fire_rates):.3f}  std={statistics.pstdev(fire_rates):.3f}")
        print(f"  precision    mean={statistics.mean(precisions):.3f}  std={statistics.pstdev(precisions):.3f}  "
              f"vs base {base_p_inc:.3f}  ⇒ Δ_precision = {statistics.mean(precisions)-base_p_inc:+.3f}")
        print(f"  recall       mean={statistics.mean(recalls):.3f}  std={statistics.pstdev(recalls):.3f}")
