"""Follow-up: fuse the credible competitive baselines and test against
DRAGIN solo. Also stratify by multi-step subset (since single-step
traces dilute every internal signal).

Findings from extra_baselines.py to follow up on:

  Hot raw AUROC vs lift+1 (n=500):
    DRAGIN.max               0.670
    tokens (length)          0.700        ← beats DRAGIN (not sig)
    testoni: word count      0.692
    testoni: self_q count    0.654
    correa: lowconf_count    0.654
    halt: H_range            0.664
    SemEnt-light             0.668

  Hot residualized (vs log_tokens):
    DRAGIN_resid             0.508
    testoni: self_q count    0.597        ← strongest length-residualized
    halt: H_terminal_growth  0.584
    SemEnt-light             0.581
    halt: H_range            0.566

Hypothesis: the *next-gen RIND* should be a fusion of DRAGIN + a
length-residualized behavioral/entropy signal. Test:

  fused = z(DRAGIN) + z(self_q_count) + z(halt_H_range) + z(SemEnt-light)
        (signs all +1 — high value = more uncertain → predict lift+1)

  Also: restrict to multi-step traces only (n_steps >= 2) and re-run.
"""
from __future__ import annotations
import json
import math
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _common import auroc
import re

ROOT = HERE.parent
DATASETS = [
    ("HotpotQA  (in-domain)",
     ROOT / "runs/v2_exp1_phi-4-mini-instruct/D1_closed_hotpot.jsonl",
     ROOT / "runs/v2_exp1_phi-4-mini-instruct/E1_oracle_hotpot.jsonl"),
    ("2WikiMHQA (OOD)",
     ROOT / "runs/v2_exp1_phi-4-mini-instruct/D2_closed_wiki.jsonl",
     ROOT / "runs/v2_exp1_phi-4-mini-instruct/E2_oracle_wiki.jsonl"),
]
QMARK = re.compile(r"\?")


def f(v):
    try: return float(v) if v is not None else None
    except Exception: return None


def step_vals(r, key):
    return [f(s.get(key)) for s in r.get("steps") or []]


def trace_max(r, key):
    vs = [v for v in step_vals(r, key) if v is not None]
    return max(vs) if vs else 0.0


def trace_range(r, key):
    vs = [v for v in step_vals(r, key) if v is not None]
    if len(vs) < 2: return 0.0
    return max(vs) - min(vs)


def trace_std(r, key):
    vs = [v for v in step_vals(r, key) if v is not None]
    if len(vs) < 2: return 0.0
    return statistics.pstdev(vs)


def step_texts(r):
    return [str(s.get("step_text") or "") for s in r.get("steps") or []]


def load_paired(cp, op):
    closed = {r["id"]: r for r in (json.loads(l) for l in cp.open())}
    oracle = {r["id"]: r for r in (json.loads(l) for l in op.open())}
    rows = []
    for qid, c in closed.items():
        if qid not in oracle:
            continue
        o = oracle[qid]
        c_em = int(bool(c.get("label_correct_em", c.get("em", False))))
        o_em = int(bool(o.get("label_correct_em", o.get("em", False))))
        lift = +1 if o_em > c_em else (-1 if o_em < c_em else 0)
        full_text = " ".join(step_texts(c))
        rows.append({
            "id": qid,
            "closed_em": c_em, "lift_em": lift,
            "n_steps": int(c.get("n_steps", 0)),
            "tokens": int(c.get("tokens_generated", 0)),
            "log_tokens": math.log(max(1, int(c.get("tokens_generated", 0))) + 1),
            "dragin": trace_max(c, "dragin_rind_max"),
            "self_q": float(len(QMARK.findall(full_text))),
            "halt_H_range": trace_range(c, "mean_token_entropy"),
            "halt_H_max": trace_max(c, "mean_token_entropy"),
            "halt_H_terminal": (lambda H: (H[-1] - H[0]) if len(H) >= 2 else 0.0)([v for v in step_vals(c, "mean_token_entropy") if v is not None]),
            "scg_u_std": trace_std(c, "u_cons"),
            "scg_u_mean": (lambda U: (sum(U)/len(U)) if U else 0.0)([v for v in step_vals(c, "u_cons") if v is not None]),
        })
    return rows


def auroc_pos(rows, key, is_pos):
    p = [r[key] for r in rows if is_pos(r)]
    n = [r[key] for r in rows if not is_pos(r)]
    if not p or not n:
        return float("nan")
    return auroc(p, n)


def zscores(rows, key):
    vs = [r[key] for r in rows]
    mu = sum(vs) / len(vs); sd = statistics.pstdev(vs) or 1.0
    return [(v - mu) / sd for v in vs]


def fuse(rows, keys):
    """Equal-weight z-score sum."""
    acc = [0.0] * len(rows)
    for k in keys:
        zs = zscores(rows, k)
        for i in range(len(rows)):
            acc[i] += zs[i]
    return acc


def ols_residual(rows, key, x_key="log_tokens"):
    n = len(rows)
    xs = [r[x_key] for r in rows]; ys = [r[key] for r in rows]
    mx = sum(xs)/n; my = sum(ys)/n
    num = sum((xs[i]-mx)*(ys[i]-my) for i in range(n))
    den = sum((xs[i]-mx)**2 for i in range(n))
    b = num/den if den > 0 else 0.0; a = my - b*mx
    return [ys[i] - (a + b*xs[i]) for i in range(n)]


def boot_diff(rows, k_a, k_b, is_pos, B=2000, seed=11):
    rng = random.Random(seed); n = len(rows); diffs = []
    for _ in range(B):
        s = [rows[rng.randrange(n)] for _ in range(n)]
        a = auroc_pos(s, k_a, is_pos); b = auroc_pos(s, k_b, is_pos)
        if a == a and b == b: diffs.append(a - b)
    if not diffs: return float("nan"), float("nan"), float("nan"), float("nan")
    diffs.sort()
    return (statistics.mean(diffs),
            diffs[int(0.025*(len(diffs)-1))],
            diffs[int(0.975*(len(diffs)-1))],
            sum(1 for d in diffs if d > 0)/len(diffs))


COMBOS = [
    ("dragin",),
    ("dragin", "self_q"),
    ("dragin", "halt_H_range"),
    ("dragin", "scg_u_std"),
    ("dragin", "self_q", "halt_H_range"),
    ("dragin", "self_q", "halt_H_range", "scg_u_std"),
    ("dragin", "self_q", "halt_H_range", "scg_u_std", "halt_H_terminal"),
    # And length-residualized fusion
]


def main():
    for label, cp, op in DATASETS:
        rows_all = load_paired(cp, op)
        rows_multi = [r for r in rows_all if r["n_steps"] >= 2]
        for rows, name in ((rows_all, "ALL"), (rows_multi, "MULTI (n_steps≥2)")):
            n = len(rows); n_h = sum(1 for r in rows if r["lift_em"] == +1)
            if n_h < 5:
                continue
            print(f"\n{'='*92}\n=== {label}  [{name}]   n={n}  lift+1={n_h} ({n_h/n:.3f}) ===")

            is_help = lambda r: r["lift_em"] == +1
            # Add fused columns
            for combo in COMBOS:
                rows_copy = [dict(r) for r in rows]
                vals = fuse(rows_copy, combo)
                for r, v in zip(rows, vals):
                    r[f"fuse_{'_'.join(combo)}"] = v
            # Also length-residualized: residualize DRAGIN, self_q, halt_H_range, scg_u_std individually, then fuse residuals
            for k in ("dragin", "self_q", "halt_H_range", "scg_u_std", "halt_H_terminal", "halt_H_max", "scg_u_mean"):
                resids = ols_residual(rows, k, x_key="log_tokens")
                for r, v in zip(rows, resids):
                    r[f"{k}_resid"] = v
            # Fused residuals
            for combo in (("dragin_resid",),
                          ("dragin_resid", "self_q_resid"),
                          ("dragin_resid", "self_q_resid", "halt_H_range_resid"),
                          ("dragin_resid", "self_q_resid", "halt_H_range_resid", "scg_u_std_resid")):
                vals = fuse(rows, combo)
                for r, v in zip(rows, vals):
                    r[f"fuse_{'+'.join(combo)}"] = v

            # Headline table
            print(f"  {'predictor':<55} {'AUROC':>7} {'ΔvsDRAGIN':>10}   {'95% CI':>20}")
            print(f"  {'-'*55} {'-'*7} {'-'*10}   {'-'*20}")

            # Singles
            for k, lbl in (("dragin","dragin.max"), ("tokens","tokens"),
                           ("self_q","self_q count"), ("halt_H_range","halt:H_range"),
                           ("scg_u_std","SemEnt-light"), ("scg_u_mean","SelfCheck(u_mean)")):
                au = auroc_pos(rows, k, is_help)
                m, lo, hi, p = boot_diff(rows, k, "dragin", is_help, B=1500)
                print(f"  {lbl:<55} {au:>7.3f} {(au-auroc_pos(rows,'dragin',is_help)):>+10.3f}   [{lo:+.3f},{hi:+.3f}]")

            print()
            # Fused
            base_dragin = auroc_pos(rows, "dragin", is_help)
            for combo in COMBOS:
                k = f"fuse_{'_'.join(combo)}"
                au = auroc_pos(rows, k, is_help)
                m, lo, hi, p = boot_diff(rows, k, "dragin", is_help, B=1500)
                sig = "✓" if (lo > 0 or hi < 0) else " "
                print(f"  fuse z[{', '.join(combo)}]"[:55].ljust(55) + f" {au:>7.3f} {(au-base_dragin):>+10.3f}   [{lo:+.3f},{hi:+.3f}] {sig}")

            print()
            # Residualized
            for combo in (("dragin_resid",),
                          ("dragin_resid", "self_q_resid"),
                          ("dragin_resid", "self_q_resid", "halt_H_range_resid"),
                          ("dragin_resid", "self_q_resid", "halt_H_range_resid", "scg_u_std_resid")):
                k = f"fuse_{'+'.join(combo)}"
                au = auroc_pos(rows, k, is_help)
                m, lo, hi, p = boot_diff(rows, k, "dragin_resid", is_help, B=1500)
                sig = "✓" if (lo > 0 or hi < 0) else " "
                au_solo = auroc_pos(rows, "dragin_resid", is_help)
                print(f"  RESIDUAL fuse z[{'+'.join(c.replace('_resid','') for c in combo)}]"[:55].ljust(55) + f" {au:>7.3f} {(au-au_solo):>+10.3f}   [{lo:+.3f},{hi:+.3f}] {sig}")


if __name__ == "__main__":
    main()
