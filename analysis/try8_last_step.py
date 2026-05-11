"""TRY 8 — Last-step (commitment moment) features.

Hypothesis: the LAST reasoning step before answer commit carries more signal
than aggregating across the whole trace. The model commits at the answer step;
its hidden state, entropy, and uncertainty there should be most diagnostic.

Compares last-step AUROC vs max-over-trace AUROC vs mean-over-trace AUROC
for every step feature. If last-step > max-over-trace, the trigger-during-
reasoning literature has been smearing the signal by aggregating.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load_traces, label, auroc

ARCHS = ["phi", "gemma", "qwen", "phi_hotpot_v2", "phi_2wiki_v2"]
FEATURES = ["drift_cosine", "variance_l2_window", "hidden_norm",
            "mean_token_entropy", "boundary_token_entropy",
            "hvr", "dragin_rind_max", "hedge_count", "verify_count",
            "u_cons", "u_ent", "u_verb", "n_tokens_in_step"]


def f(v):
    try: return float(v) if v is not None else None
    except: return None


def step_vals(r, key):
    return [f(s.get(key)) for s in (r.get("steps") or [])]


print("LAST-STEP vs WHOLE-TRACE: where is the signal?")
print("(AUROC > 0.55 = real signal; >0.6 = strong)\n")

best_overall = []
for arch in ARCHS:
    try:
        rows = load_traces(arch, min_steps=3)
    except (KeyError, FileNotFoundError):
        continue
    if not rows: continue
    labels = [label(r) for r in rows]
    print(f"\n=== {arch}  (n={len(rows)}, n_incorrect={sum(1 for l in labels if not l)}) ===")
    print(f"{'feature':24}  {'max':>6}  {'mean':>6}  {'last':>6}  {'last−max':>9}")
    for key in FEATURES:
        au_max = auroc(
            [max([v for v in step_vals(r, key) if v is not None] or [0])
             for r, l in zip(rows, labels) if not l and any(v is not None for v in step_vals(r, key))],
            [max([v for v in step_vals(r, key) if v is not None] or [0])
             for r, l in zip(rows, labels) if l and any(v is not None for v in step_vals(r, key))],
        )
        au_mean = auroc(
            [sum(v for v in step_vals(r, key) if v is not None) / max(1, sum(1 for v in step_vals(r, key) if v is not None))
             for r, l in zip(rows, labels) if not l],
            [sum(v for v in step_vals(r, key) if v is not None) / max(1, sum(1 for v in step_vals(r, key) if v is not None))
             for r, l in zip(rows, labels) if l],
        )
        pos_last, neg_last = [], []
        for r, l in zip(rows, labels):
            vs = [v for v in step_vals(r, key) if v is not None]
            if not vs: continue
            (neg_last if l else pos_last).append(vs[-1])
        au_last = auroc(pos_last, neg_last)
        diff = (au_last - au_max) if (au_last == au_last and au_max == au_max) else float("nan")
        flag = ""
        if au_last == au_last and abs(au_last - 0.5) > abs(au_max - 0.5) + 0.02:
            flag = "  ★ last > max"
        elif au_max == au_max and abs(au_max - 0.5) > abs(au_last - 0.5) + 0.02:
            flag = "  ☆ max > last"
        print(f"{key:24}  {au_max:>6.3f}  {au_mean:>6.3f}  {au_last:>6.3f}  {diff:>+9.3f}{flag}")
        best_overall.append((arch, key, "last", au_last))
        best_overall.append((arch, key, "max",  au_max))
        best_overall.append((arch, key, "mean", au_mean))

print("\n" + "=" * 70)
print("TOP 10 (feature, agg, arch) pairs by |AUROC − 0.5|:")
best_overall = [b for b in best_overall if b[3] == b[3]]
best_overall.sort(key=lambda x: -abs(x[3] - 0.5))
for arch, key, agg, au in best_overall[:10]:
    print(f"  {arch:16}  {key:24}  ({agg})  AUROC = {au:.3f}")
