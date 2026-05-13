"""Length-confound audit v2 — addressing review critique of v2-10.

Improvements over `length_audit.py`:

  1. EXPANDED BASELINE POOL.
     Internal signals tested are no longer just {u_cons, dragin, ent}.
     Added: ETC (Δ²H = max abs second-difference of mean_token_entropy),
     SELFDOUBT Tier-1 HVR, drift_cosine.max, variance_l2_window.max,
     hidden_norm.range. These are the v1-vintage CRV / textual gates.

  2. PRECISION-AT-BUDGET, not AUROC, as headline.
     Dynamic-RAG triggers are deployed at "fire on top-k% most risky"
     budgets. We report P(lift_em=+1 | fired) at k ∈ {5%, 10%, 20%}
     for every signal, with AUROC as secondary.

  3. CONTINUOUS LENGTH-RESIDUALIZATION, not tertile binning.
     For every signal, fit residual = signal − OLS(signal ~ log_tokens).
     Then re-rank by residual and recompute precision-at-budget.
     This avoids the tertile sample-size problem and treats length as
     continuous (which is what it is).

  4. BOOTSTRAP CIs for the deltas.
     B=2000 paired bootstrap on (signal − tokens) AUROC and on
     (signal_residual − chance) AUROC. CIs that exclude zero are
     starred.

  5. CLARIFICATION OF QUESTION-TYPE FINDING.
     Compute u_cons.max AUROC AND length-residualized u_cons AUROC on
     each question-type subset separately. Tells us whether the
     "u_cons wins comparison" finding holds AFTER length is controlled.

Operating-point philosophy: the headline metric is **lift in
P(retrieval-helpful | fired) over base helpful rate**, at a 5% trace
fire-rate budget. This is what a deployment paper actually cares about.
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

ROOT = HERE.parent
DATASETS = [
    ("HotpotQA  (in-domain)",
     ROOT / "runs/v2_exp1_phi-4-mini-instruct/D1_closed_hotpot.jsonl",
     ROOT / "runs/v2_exp1_phi-4-mini-instruct/E1_oracle_hotpot.jsonl"),
    ("2WikiMHQA (OOD)",
     ROOT / "runs/v2_exp1_phi-4-mini-instruct/D2_closed_wiki.jsonl",
     ROOT / "runs/v2_exp1_phi-4-mini-instruct/E2_oracle_wiki.jsonl"),
]


def f(v):
    try: return float(v) if v is not None else None
    except Exception: return None


def step_vals(r, key):
    return [f(s.get(key)) for s in r.get("steps") or []]


def tmax(r, key):
    vs = [v for v in step_vals(r, key) if v is not None]
    return max(vs) if vs else None


def trange(r, key):
    vs = [v for v in step_vals(r, key) if v is not None]
    if len(vs) < 2: return 0.0
    return max(vs) - min(vs)


def etc_score(r):
    """ETC (Li 2025): max |Δ²H| of mean_token_entropy series."""
    H = [v for v in step_vals(r, "mean_token_entropy") if v is not None]
    if len(H) < 3:
        return 0.0
    d2 = [abs(H[i+1] - 2*H[i] + H[i-1]) for i in range(1, len(H)-1)]
    return max(d2) if d2 else 0.0


def hvr_max(r):
    """SELFDOUBT Tier-1: HVR (hedge-to-verify ratio) per step, max over trace."""
    vs = [v for v in step_vals(r, "hvr") if v is not None]
    return max(vs) if vs else 0.0


def load_paired(cp: Path, op: Path):
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
        u = tmax(c, "u_cons"); d = tmax(c, "dragin_rind_max"); e = tmax(c, "mean_token_entropy")
        if u is None or d is None or e is None:
            continue
        rows.append({
            "id": qid, "type": c.get("type"), "level": c.get("level"),
            "closed_em": c_em, "lift_em": lift,
            "n_steps": int(c.get("n_steps", 0)),
            "tokens": int(c.get("tokens_generated", 0)),
            "log_tokens": math.log(max(1, int(c.get("tokens_generated", 0))) + 1),
            "wall_s": float(c.get("wall_clock_s") or 0),
            "u_cons_max": u, "dragin_max": d, "ent_max": e,
            "etc": etc_score(c),
            "hvr_max": hvr_max(c),
            "drift_max": tmax(c, "drift_cosine") or 0.0,
            "var_max": tmax(c, "variance_l2_window") or 0.0,
            "hidden_range": trange(c, "hidden_norm"),
        })
    return rows


def auroc_pos(rows, key, is_pos):
    p = [r[key] for r in rows if is_pos(r)]
    n = [r[key] for r in rows if not is_pos(r)]
    if not p or not n:
        return float("nan")
    return auroc(p, n)


def precision_at_k(rows, key, k_frac, is_pos):
    """Top-k%-by-`key`. Returns (n_fire, P(is_pos|fire), base_rate)."""
    n = len(rows)
    if n == 0:
        return 0, float("nan"), float("nan")
    base = sum(1 for r in rows if is_pos(r)) / n
    pairs = sorted(rows, key=lambda r: r[key], reverse=True)
    n_fire = max(1, int(round(k_frac * n)))
    fired = pairs[:n_fire]
    n_pos = sum(1 for r in fired if is_pos(r))
    return n_fire, n_pos / n_fire, base


def ols_residual(rows, key, x_key="log_tokens"):
    """Fit y = a + b*x via OLS, return y_i - (a + b*x_i) as a list aligned with rows."""
    n = len(rows)
    xs = [r[x_key] for r in rows]
    ys = [r[key] for r in rows]
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    den = sum((xs[i] - mx) ** 2 for i in range(n))
    b = num / den if den > 0 else 0.0
    a = my - b * mx
    return [ys[i] - (a + b * xs[i]) for i in range(n)], (a, b)


def bootstrap_auroc_diff(rows, key_a, key_b, is_pos, B=2000, seed=11):
    rng = random.Random(seed)
    n = len(rows)
    diffs = []
    for _ in range(B):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        a = auroc_pos(sample, key_a, is_pos)
        b = auroc_pos(sample, key_b, is_pos)
        if a == a and b == b:
            diffs.append(a - b)
    if not diffs:
        return float("nan"), float("nan"), float("nan"), float("nan")
    diffs.sort()
    m = statistics.mean(diffs)
    lo = diffs[int(0.025 * (len(diffs) - 1))]
    hi = diffs[int(0.975 * (len(diffs) - 1))]
    ppos = sum(1 for d in diffs if d > 0) / len(diffs)
    return m, lo, hi, ppos


# Higher value = more uncertain (for AUROC where label=incorrect/helpful).
# For lift_em=+1 detection: high values of these signals SHOULD predict "retrieval will help",
# which empirically aligns for entropy/uncertainty signals. Sign is set so high→positive class.
SIGNAL_DEFS = (
    # (key, display, family, sign)
    ("tokens",       "tokens_generated",        "external", +1),
    ("wall_s",       "wall_clock_s",            "external", +1),
    ("n_steps",      "n_steps",                 "external", +1),
    ("u_cons_max",   "u_cons.max",              "internal-K3",       +1),
    ("dragin_max",   "dragin_rind_max.max",     "internal-attn",     +1),
    ("ent_max",      "ent.max",                 "internal-logit",    +1),
    ("etc",          "etc (Δ²H max)",           "internal-logit",    +1),
    ("hvr_max",      "selfdoubt_t1 (hvr.max)",  "external-text",     +1),
    ("drift_max",    "drift_cosine.max (CRV)",  "internal-hidden",   +1),
    ("var_max",      "variance_l2_window.max",  "internal-hidden",   +1),
    ("hidden_range", "hidden_norm.range",       "internal-hidden",   +1),
)


def banner(s):
    print(f"\n{'='*92}\n{s}\n{'='*92}")


def main():
    for label, cp, op in DATASETS:
        rows = load_paired(cp, op)
        n = len(rows)
        n_inc = sum(1 for r in rows if r["closed_em"] == 0)
        n_help = sum(1 for r in rows if r["lift_em"] == +1)
        banner(f"{label}   n={n}  incorrect={n_inc}  lift+1={n_help} (base={n_help/n:.3f})")
        is_help = lambda r: r["lift_em"] == +1
        is_inc = lambda r: r["closed_em"] == 0

        # ----- (1) Raw AUROC + precision-at-budget for all 11 signals -----
        print("\n--- (1) RAW headline table — TARGET=lift_em=+1 ---")
        print(f"  {'signal':<26} {'family':<18} {'AUROC':>7} {'P@5%':>7} {'P@10%':>7} {'P@20%':>7}   {'Δprec@5%':>9}")
        base_help = n_help / n
        for key, name, family, _ in SIGNAL_DEFS:
            au = auroc_pos(rows, key, is_help)
            _, p5, _ = precision_at_k(rows, key, 0.05, is_help)
            _, p10, _ = precision_at_k(rows, key, 0.10, is_help)
            _, p20, _ = precision_at_k(rows, key, 0.20, is_help)
            print(f"  {name:<26} {family:<18} {au:>7.3f} {p5:>7.3f} {p10:>7.3f} {p20:>7.3f}   {p5-base_help:>+9.3f}")
        print(f"  {'BASE RATE':<26} {'':<18} {'':<7} {base_help:>7.3f} {base_help:>7.3f} {base_help:>7.3f}")

        # ----- (2) Length-residualized AUROC + P@k -----
        print("\n--- (2) LENGTH-RESIDUALIZED (each signal regressed on log_tokens) — TARGET=lift+1 ---")
        # Add residual columns
        for key, _, _, _ in SIGNAL_DEFS:
            if key == "tokens":
                # residual of tokens against log_tokens — definitionally near-zero
                continue
            resids, (a, b) = ols_residual(rows, key, x_key="log_tokens")
            for r, x in zip(rows, resids):
                r[f"{key}_resid"] = x
        print(f"  {'signal':<26} {'raw AUROC':>10} {'resid AUROC':>12} {'ΔAUROC':>8}   {'raw P@5%':>9} {'resid P@5%':>11} {'Δ P@5%':>8}")
        for key, name, family, _ in SIGNAL_DEFS:
            if key == "tokens":
                # tokens vs itself: skip
                continue
            au_raw = auroc_pos(rows, key, is_help)
            au_res = auroc_pos(rows, f"{key}_resid", is_help)
            _, p5_raw, _ = precision_at_k(rows, key, 0.05, is_help)
            _, p5_res, _ = precision_at_k(rows, f"{key}_resid", 0.05, is_help)
            print(f"  {name:<26} {au_raw:>10.3f} {au_res:>12.3f} {au_res-au_raw:>+8.3f}   {p5_raw:>9.3f} {p5_res:>11.3f} {p5_res-p5_raw:>+8.3f}")

        # ----- (3) Paired bootstrap CI for (signal − tokens) AUROC, both raw and residual -----
        print("\n--- (3) PAIRED BOOTSTRAP (signal − tokens) AUROC →lift+1, B=2000 ---")
        print(f"  {'signal':<26} {'raw Δ vs tokens':>16} {'95% CI':>22} {'sig?':>6}")
        for key, name, family, _ in SIGNAL_DEFS:
            if key == "tokens":
                continue
            m, lo, hi, p = bootstrap_auroc_diff(rows, key, "tokens", is_help, B=2000)
            sig = "✓" if (lo > 0 or hi < 0) else "—"
            print(f"  {name:<26} {m:>+16.3f}  [{lo:>+6.3f},{hi:>+6.3f}]  {sig:>6}")

        # ----- (4) Per-question-type, BOTH raw and length-residualized -----
        print("\n--- (4) PER QUESTION TYPE — TARGET=lift+1 ---")
        print("       raw AUROC | length-residualized AUROC (signal − log_tokens regressed out)")
        types = sorted(set(r["type"] for r in rows if r["type"]))
        print(f"  types present: {types}")
        # Headers
        SHOW = ("u_cons_max", "dragin_max", "ent_max", "etc", "var_max", "tokens")
        SHOW_NAMES = {"u_cons_max": "u_cons", "dragin_max": "dragin", "ent_max": "ent",
                      "etc": "etc", "var_max": "var", "tokens": "tokens"}
        for t in types:
            sub = [r for r in rows if r["type"] == t]
            n_t = len(sub); n_h = sum(1 for r in sub if is_help(r))
            if n_h < 3 or n_t < 20:
                print(f"\n  type={t}  n={n_t}  lift+1={n_h}   [skipped — too few]"); continue
            # Refit residual within the subset (proper conditioning)
            for key, _, _, _ in SIGNAL_DEFS:
                if key == "tokens": continue
                resids, _ = ols_residual(sub, key, x_key="log_tokens")
                for r, x in zip(sub, resids):
                    r[f"{key}_resid_sub"] = x
            print(f"\n  type={t}  n={n_t}  lift+1={n_h} (base={n_h/n_t:.3f})")
            print(f"    {'signal':<10} {'raw AUROC':>10} {'resid AUROC':>13} {'Δ':>7}   {'raw P@10%':>10} {'resid P@10%':>13}")
            for key in SHOW:
                name = SHOW_NAMES[key]
                au_raw = auroc_pos(sub, key, is_help)
                if key == "tokens":
                    au_res = au_raw  # not residualized vs itself
                else:
                    au_res = auroc_pos(sub, f"{key}_resid_sub", is_help)
                _, p10_raw, _ = precision_at_k(sub, key, 0.10, is_help)
                if key == "tokens":
                    p10_res = p10_raw
                else:
                    _, p10_res, _ = precision_at_k(sub, f"{key}_resid_sub", 0.10, is_help)
                print(f"    {name:<10} {au_raw:>10.3f} {au_res:>13.3f} {au_res-au_raw:>+7.3f}   {p10_raw:>10.3f} {p10_res:>13.3f}")


if __name__ == "__main__":
    main()
