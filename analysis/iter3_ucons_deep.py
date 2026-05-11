"""ITERATION 3 — u_cons deep dive: what IS the consistency signal?

u_cons is the consistency sampler statistic: at each reasoning-step boundary,
K=3 stochastic completions are drawn and an agreement statistic is computed
(higher = more disagreement = less consistent). On Phi v2 traces it has the
strongest single-feature AUROC of any on-disk feature (0.65-0.70).

Questions:
  (a) What's the distribution of u_cons by trace correctness, across both
      datasets? Bimodal? Long-tailed?
  (b) Is u_cons concentrated at specific reasoning steps? First? Last?
      Middle? Compare AUROC of {first, last, max, mean, peak-position}.
  (c) Threshold-based gate: at trace-iso-0.30, what's u_cons trace-level
      precision (P(incorrect | gate fires))?
  (d) Does u_cons add signal ON TOP of trace length? Compute partial AUROC:
      bin traces by tokens_generated, compute u_cons AUROC within each bin.
  (e) AUROC stability across n: subsample n ∈ {30, 50, 80, 100, all}, 
      bootstrap 50 times per n. Confidence band.
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


for arch in DATASETS:
    try: rows = load_traces(arch, min_steps=2)
    except: continue
    if not rows: continue
    labels = [label(r) for r in rows]
    n = len(rows)
    n_correct = sum(labels); n_incorrect = n - n_correct
    print(f"\n=== {arch}  n={n} (correct={n_correct}, incorrect={n_incorrect}, EM={n_correct/n*100:.1f}%) ===\n")

    # (a) Distribution
    u_max_correct = [max(step_vals(r,"u_cons") or [0]) for r,l in zip(rows,labels) if l and step_vals(r,"u_cons")]
    u_max_incorrect = [max(step_vals(r,"u_cons") or [0]) for r,l in zip(rows,labels) if not l and step_vals(r,"u_cons")]
    def quants(xs, ps=(0.1,0.25,0.5,0.75,0.9)):
        s=sorted(xs)
        return tuple(s[int(p*(len(s)-1))] for p in ps) if s else (float("nan"),)*5
    qc = quants(u_max_correct); qi = quants(u_max_incorrect)
    print(f"(a) u_cons/max distribution by class:")
    print(f"    correct   q10={qc[0]:.3f} q25={qc[1]:.3f} q50={qc[2]:.3f} q75={qc[3]:.3f} q90={qc[4]:.3f}  (n={len(u_max_correct)})")
    print(f"    incorrect q10={qi[0]:.3f} q25={qi[1]:.3f} q50={qi[2]:.3f} q75={qi[3]:.3f} q90={qi[4]:.3f}  (n={len(u_max_incorrect)})")
    print(f"    median shift = {qi[2]-qc[2]:+.3f}")

    # (b) Per-step-position AUROC: first / mid / last / max
    print(f"\n(b) Per-position u_cons AUROC:")
    for agg in ("first","last","max","mean","sum"):
        pos=[]; neg=[]
        for r,l in zip(rows,labels):
            vs = step_vals(r,"u_cons")
            if not vs: continue
            v = {"first":vs[0],"last":vs[-1],"max":max(vs),"mean":sum(vs)/len(vs),"sum":sum(vs)}[agg]
            (neg if l else pos).append(v)
        au = auroc(pos, neg)
        print(f"    u_cons/{agg:6}  AUROC = {au:.3f}  ({len(pos)+len(neg)} usable traces)")

    # (b-cont) Position of MAX u_cons in trace, by class
    pos_correct = []; pos_incorrect = []
    for r,l in zip(rows,labels):
        vs = step_vals(r,"u_cons")
        if len(vs) < 2: continue
        peak_pos = vs.index(max(vs)) / (len(vs)-1)  # 0..1
        (pos_correct if l else pos_incorrect).append(peak_pos)
    if pos_correct and pos_incorrect:
        print(f"    Position of max u_cons (0=first, 1=last):")
        print(f"      correct mean position = {statistics.mean(pos_correct):.2f}")
        print(f"      incorrect mean position = {statistics.mean(pos_incorrect):.2f}")

    # (c) Threshold-based: trace-iso-0.30 precision
    print(f"\n(c) u_cons/max as gate at trace-iso-0.30:")
    u_max_all = sorted([max(step_vals(r,"u_cons") or [0]) for r in rows if step_vals(r,"u_cons")])
    if u_max_all:
        # Find threshold producing ~0.30 trace fire rate
        for target in (0.10,0.20,0.30,0.50):
            # threshold = (1-target)-quantile of trace-max
            tau = u_max_all[int((1-target)*(len(u_max_all)-1))]
            fired = [(max(step_vals(r,"u_cons") or [0]) > tau) for r in rows]
            tp = sum(1 for f_,l in zip(fired,labels) if f_ and not l)  # fired and incorrect
            fp = sum(1 for f_,l in zip(fired,labels) if f_ and l)
            fn = sum(1 for f_,l in zip(fired,labels) if not f_ and not l)
            tn = sum(1 for f_,l in zip(fired,labels) if not f_ and l)
            prec = tp/(tp+fp) if tp+fp else float("nan")
            recall = tp/(tp+fn) if tp+fn else float("nan")
            print(f"    target {target:.2f}  τ={tau:.4f}  fire-rate={sum(fired)/n:.2f}  "
                  f"P(incorrect | fired)={prec:.3f}  recall={recall:.3f}  "
                  f"base-rate-P(incorrect)={n_incorrect/n:.3f}")

    # (d) Partial AUROC conditioned on trace length tertile
    print(f"\n(d) u_cons/max AUROC within trace-length tertiles (control for length):")
    toks = [(r.get("tokens_generated", 0), max(step_vals(r,"u_cons") or [0]), l) for r,l in zip(rows,labels) if step_vals(r,"u_cons")]
    if toks:
        toks.sort()
        t1 = toks[len(toks)//3][0]; t2 = toks[2*len(toks)//3][0]
        for tag, sub in [(f"short ≤{t1}",  [(v,l_) for t,v,l_ in toks if t<=t1]),
                         (f"mid",            [(v,l_) for t,v,l_ in toks if t1<t<=t2]),
                         (f"long >{t2}",  [(v,l_) for t,v,l_ in toks if t>t2])]:
            pos=[v for v,l_ in sub if not l_]; neg=[v for v,l_ in sub if l_]
            au = auroc(pos, neg)
            print(f"    {tag:18}  n={len(sub):3}  AUROC = {au:.3f}  (EM={sum(1 for _,l_ in sub if l_)/max(1,len(sub))*100:.0f}%)")

    # (e) AUROC stability across subsample n
    print(f"\n(e) u_cons/max AUROC bootstrap stability across n:")
    u_vals = [(max(step_vals(r,"u_cons") or [0]), l) for r,l in zip(rows,labels) if step_vals(r,"u_cons")]
    for sub_n in (30, 50, 80, min(150, len(u_vals)), len(u_vals)):
        if sub_n > len(u_vals): continue
        aurocs = []
        for sd_ in range(100):
            rng = random.Random(sd_)
            sample = rng.sample(u_vals, sub_n)
            pos = [v for v,l_ in sample if not l_]; neg = [v for v,l_ in sample if l_]
            au = auroc(pos, neg)
            if au == au:
                aurocs.append(au)
        if not aurocs: continue
        m = statistics.mean(aurocs); s = statistics.pstdev(aurocs)
        print(f"    n={sub_n:3}  AUROC mean={m:.3f}  std={s:.3f}  95% band ≈ [{m-2*s:.3f}, {m+2*s:.3f}]")
