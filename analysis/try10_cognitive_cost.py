"""TRY 10 — Cognitive-cost reframe (length-as-signal, overthinking detection).

Per 2026 overthinking literature (Wu, Chen, Fu, Zhang_100Days):
the real bottleneck is tokens spent on doomed trajectories, not retrieval timing.
Reformulate the problem: predict "this trace is going to be wasted" (long AND
wrong) early — not "should we retrieve" but "should we abort and restart."

Key questions:
  (a) Is trace length alone a strong predictor of incorrectness?
  (b) Conditional on length, does any per-step feature add signal?
  (c) Is there a position in the trace where the length-vs-correctness
      decision becomes confident?
  (d) Predict-from-prefix: at step k (k=1,2,3...), can we predict whether
      this trace will be wrong? This is the "abort early" detector.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load_traces, label, auroc

ARCHS = ["phi", "gemma", "qwen", "phi_hotpot_v2", "phi_2wiki_v2"]


def f(v):
    try: return float(v) if v is not None else None
    except: return None


print("COGNITIVE-COST REFRAME: predict 'wasted trace' (long + wrong) from early signals\n")

for arch in ARCHS:
    try:
        rows = load_traces(arch, min_steps=1)
    except (KeyError, FileNotFoundError):
        continue
    if not rows: continue
    labels = [label(r) for r in rows]
    print(f"=== {arch}  (n={len(rows)}, EM={(sum(labels)/len(labels))*100:.1f}%) ===")

    # (a) Length alone
    for key in ("n_steps", "tokens_generated", "wall_clock_s"):
        pos = [float(r.get(key, 0)) for r, l in zip(rows, labels) if not l]
        neg = [float(r.get(key, 0)) for r, l in zip(rows, labels) if l]
        au = auroc(pos, neg)
        m_pos = sum(pos)/len(pos) if pos else 0
        m_neg = sum(neg)/len(neg) if neg else 0
        print(f"  {key:18}  AUROC={au:.3f}  mean(incorrect)={m_pos:.1f}  mean(correct)={m_neg:.1f}  ratio={m_pos/(m_neg+1e-9):.2f}×")

    # (b) Conditional: top features when conditioning on n_steps being above median.
    # Quick proxy: split traces by length tertile, AUROC of each feature within each tertile.
    n_steps_list = sorted([r.get("n_steps", 0) for r in rows])
    if len(n_steps_list) < 6: continue
    t1 = n_steps_list[len(n_steps_list)//3]
    t2 = n_steps_list[2 * len(n_steps_list)//3]
    short = [(r, l) for r, l in zip(rows, labels) if r.get("n_steps", 0) <= t1]
    mid   = [(r, l) for r, l in zip(rows, labels) if t1 < r.get("n_steps", 0) <= t2]
    long_ = [(r, l) for r, l in zip(rows, labels) if r.get("n_steps", 0) > t2]
    print(f"\n  Length tertiles: short ≤{t1} (n={len(short)}, EM={sum(l for _,l in short)/max(1,len(short))*100:.0f}%)  "
          f"mid (n={len(mid)}, EM={sum(l for _,l in mid)/max(1,len(mid))*100:.0f}%)  "
          f"long >{t2} (n={len(long_)}, EM={sum(l for _,l in long_)/max(1,len(long_))*100:.0f}%)")

    # (c) Predict-from-prefix: at step k, use only steps[0..k-1] features. AUROC at k=1,2,3.
    print(f"\n  PREDICT-FROM-PREFIX: AUROC for predicting incorrect from first-k steps only")
    print(f"  {'feature':28} {'k=1':>6} {'k=2':>6} {'k=3':>6} {'k=5':>6}")
    for key in ("mean_token_entropy", "drift_cosine", "hidden_norm",
                "hedge_count", "verify_count", "u_cons", "u_ent", "u_verb",
                "boundary_token_entropy", "dragin_rind_max", "n_tokens_in_step"):
        row = f"  {key:28} "
        for k in (1, 2, 3, 5):
            pos, neg = [], []
            for r, l in zip(rows, labels):
                steps = r.get("steps") or []
                vs = [f(s.get(key)) for s in steps[:k]]
                vs = [v for v in vs if v is not None]
                if not vs: continue
                # use mean of first k as the predictor
                m = sum(vs) / len(vs)
                (neg if l else pos).append(m)
            au = auroc(pos, neg)
            row += f"{au:>6.3f} " if au == au else f"{' n/a ':>6}"
        print(row)
    print()
