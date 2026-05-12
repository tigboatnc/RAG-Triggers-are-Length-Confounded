"""Idea 1 — Early vs Late uncertainty spike (Song detection-dilemma /
Wu inverted-U framing).

Hypothesis. Within incorrect traces:
  early spike (low normalised argmax_position) → factual gap → retrieval-fixable (lift_em=+1)
  late  spike (high normalised argmax_position) → logical collapse → unrecoverable (lift_em ≤ 0)

We use the existing v2 Phi closed-book + oracle traces. No new compute.

For each per-step scalar feature f ∈ {u_cons, mean_token_entropy,
drift_cosine, variance_l2_window, dragin_rind_max, hidden_norm,
boundary_token_entropy}, compute per trace:
  - max-value (already-studied)
  - argmax_step
  - normalised position = argmax_step / max(1, n_steps_with_feat - 1) ∈ [0, 1]
  - early-third indicator: pos < 1/3
  - late-third indicator:  pos > 2/3

Reported, per dataset (HotpotQA in-domain ; 2Wiki OOD):

  AUROC of `pos` for predicting:
    (a) closed-book INCORRECT  (all traces, label_correct_em == 0)
    (b) lift_em == +1   (all traces; competitor: u_cons.max)
    (c) lift_em == +1   restricted to incorrect-only subset ("fixable | incorrect")
    (d) lift_em == −1 vs +1  ("unrecoverable vs fixable" within incorrect)

  Deployment compound gate: fire iff (f.max > q70(f.max)) AND (pos < 1/3 for "fixable" /
  pos > 2/3 for "unrecoverable").  Report precision and base rate.

  Wu inverted-U: stratify by n_steps tertile and recompute (b)/(c).

Run with `gzn` conda env.
"""
from __future__ import annotations
import json
import statistics
import sys
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _common import auroc

RUN_DIR = HERE.parent / "runs/v2_exp1_phi-4-mini-instruct"
DATASETS = [
    ("HotpotQA  (in-domain)", RUN_DIR / "D1_closed_hotpot.jsonl", RUN_DIR / "E1_oracle_hotpot.jsonl"),
    ("2WikiMHQA (OOD)",       RUN_DIR / "D2_closed_wiki.jsonl",   RUN_DIR / "E2_oracle_wiki.jsonl"),
]
FEATURES = [
    "u_cons", "mean_token_entropy", "drift_cosine", "variance_l2_window",
    "dragin_rind_max", "hidden_norm", "boundary_token_entropy",
]


def f(v):
    try: return float(v) if v is not None else None
    except Exception: return None


def step_series(r, key):
    vals = []
    for s in r.get("steps") or []:
        v = f(s.get(key))
        vals.append(v)
    return vals


def feat_stats(r, key):
    vs = [(i, v) for i, v in enumerate(step_series(r, key)) if v is not None]
    if not vs:
        return None
    n = len(vs)
    if n == 1:
        i, v = vs[0]
        return {"max": v, "argmax": i, "pos": 0.5, "n_feat_steps": n}
    imax, vmax = max(vs, key=lambda x: x[1])
    pos = imax / (n - 1)
    return {"max": vmax, "argmax": imax, "pos": pos, "n_feat_steps": n}


def load_pair(closed_path: Path, oracle_path: Path):
    closed = {r["id"]: r for r in (json.loads(l) for l in closed_path.open())}
    oracle = {r["id"]: r for r in (json.loads(l) for l in oracle_path.open())}
    out = []
    for qid, c in closed.items():
        if qid not in oracle:
            continue
        o = oracle[qid]
        c_em = int(bool(c.get("label_correct_em", c.get("em", False))))
        o_em = int(bool(o.get("label_correct_em", o.get("em", False))))
        lift = +1 if o_em > c_em else (-1 if o_em < c_em else 0)
        row = {
            "id": qid,
            "closed_em": c_em, "oracle_em": o_em, "lift_em": lift,
            "n_steps": int(c.get("n_steps", 0)),
        }
        ok = True
        for key in FEATURES:
            s = feat_stats(c, key)
            if s is None:
                ok = False
                break
            row[f"{key}_max"] = s["max"]
            row[f"{key}_pos"] = s["pos"]
            row[f"{key}_n"] = s["n_feat_steps"]
        if ok and row["n_steps"] >= 2:
            out.append(row)
    return out


def auroc_pos_higher_is_class(rows, key, is_pos):
    """AUROC for `key` predicting is_pos(row)==True."""
    pos = [r[key] for r in rows if is_pos(r)]
    neg = [r[key] for r in rows if not is_pos(r)]
    if not pos or not neg:
        return float("nan"), len(pos), len(neg)
    return auroc(pos, neg), len(pos), len(neg)


def bootstrap_ci(rows, scoring_fn, B=1000, seed=0):
    import random
    rng = random.Random(seed)
    n = len(rows)
    vals = []
    for _ in range(B):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        try:
            v = scoring_fn(sample)
            if v == v:
                vals.append(v)
        except Exception:
            pass
    if not vals:
        return float("nan"), float("nan"), float("nan")
    vals.sort()
    lo = vals[int(0.025 * (len(vals) - 1))]
    hi = vals[int(0.975 * (len(vals) - 1))]
    return statistics.mean(vals), lo, hi


def precision_at_compound(rows, key_max, key_pos, tau_max, early=True, third=1/3):
    fired = [r for r in rows if r[key_max] > tau_max and
             ((r[key_pos] < third) if early else (r[key_pos] > 1 - third))]
    n_fire = len(fired)
    if n_fire == 0:
        return None
    n_help = sum(1 for r in fired if r["lift_em"] == +1)
    n_harm = sum(1 for r in fired if r["lift_em"] == -1)
    n_inc = sum(1 for r in fired if r["closed_em"] == 0)
    return {"n_fire": n_fire, "frac": n_fire / len(rows),
            "p_help": n_help / n_fire, "p_harm": n_harm / n_fire,
            "p_inc": n_inc / n_fire}


def main():
    for label, cp, op in DATASETS:
        if not cp.exists() or not op.exists():
            print(f"-- skip {label} (missing)"); continue
        rows = load_pair(cp, op)
        n = len(rows)
        n_inc = sum(1 for r in rows if r["closed_em"] == 0)
        n_help = sum(1 for r in rows if r["lift_em"] == +1)
        n_harm = sum(1 for r in rows if r["lift_em"] == -1)
        n_neut = sum(1 for r in rows if r["lift_em"] == 0)
        base_help = n_help / n
        print(f"\n{'='*80}\n=== {label}  n={n}  ===")
        print(f"  incorrect={n_inc} ({n_inc/n:.3f})  lift+1={n_help}  lift0={n_neut}  lift-1={n_harm}  base_help={base_help:.3f}")

        # ------ (1) AUROC of normalized spike POSITION for each feature ------
        print("\n  --- AUROC of normalised argmax-position (pos = argmax_step / (n-1)) ---")
        print(f"  {'feature':<24}  {'→INCORRECT':>11}  {'→lift+1':>9}  {'→lift+1|inc':>12}  {'→harm vs help':>14}")
        for key in FEATURES:
            pk = f"{key}_pos"
            au_inc, _, _ = auroc_pos_higher_is_class(rows, pk, lambda r: r["closed_em"] == 0)
            au_help, _, _ = auroc_pos_higher_is_class(rows, pk, lambda r: r["lift_em"] == +1)
            incorrect_only = [r for r in rows if r["closed_em"] == 0]
            au_fix_inc, _, _ = auroc_pos_higher_is_class(incorrect_only, pk, lambda r: r["lift_em"] == +1)
            within_lift_pm = [r for r in rows if r["lift_em"] in (+1, -1)]
            # higher pos → harm (we test pos predicting lift_em=-1 within {-1, +1})
            au_harm, _, _ = auroc_pos_higher_is_class(within_lift_pm, pk, lambda r: r["lift_em"] == -1)
            print(f"  {key:<24}  {au_inc:>11.3f}  {au_help:>9.3f}  {au_fix_inc:>12.3f}  {au_harm:>14.3f}")

        # ------ (2) Compound gate (f.max > q70) AND early/late-third ------
        print("\n  --- Compound deployment gate:  fire ⇔ (feat.max > q70) AND (pos in early/late third) ---")
        print(f"  base helpful = {base_help:.3f}   base incorrect = {n_inc/n:.3f}")
        print(f"  {'feature':<24} {'half':<6} {'n_fire':>6} {'frac':>6} {'P(help|F)':>10} {'P(harm|F)':>10} {'P(inc|F)':>9}")
        for key in FEATURES:
            mk = f"{key}_max"; pk = f"{key}_pos"
            maxes = sorted(r[mk] for r in rows)
            tau = maxes[int(0.70 * (len(maxes) - 1))]
            for half, early in (("early", True), ("late", False)):
                res = precision_at_compound(rows, mk, pk, tau, early=early, third=1/3)
                if res is None:
                    print(f"  {key:<24} {half:<6} (no fires)")
                else:
                    print(f"  {key:<24} {half:<6} {res['n_fire']:>6d} {res['frac']:>6.3f} "
                          f"{res['p_help']:>10.3f} {res['p_harm']:>10.3f} {res['p_inc']:>9.3f}")

        # ------ (3) Wu inverted-U: stratify by n_steps tertile ------
        ns = sorted(r["n_steps"] for r in rows)
        if len(ns) >= 3:
            q33 = ns[len(ns)//3]; q66 = ns[2*len(ns)//3]
        else:
            q33 = q66 = ns[-1]
        def tertile(r):
            if r["n_steps"] <= q33: return "short"
            if r["n_steps"] >= q66: return "long"
            return "mid"
        print(f"\n  --- Wu inverted-U: AUROC by n_steps tertile (split at <={q33} / >={q66}) ---")
        for ter in ("short", "mid", "long"):
            sub = [r for r in rows if tertile(r) == ter]
            n_s = len(sub); n_help_s = sum(1 for r in sub if r["lift_em"] == +1)
            n_inc_s = sum(1 for r in sub if r["closed_em"] == 0)
            print(f"  tertile={ter}  n={n_s}  helpful={n_help_s}  incorrect={n_inc_s}")
            for key in ("u_cons", "mean_token_entropy", "drift_cosine"):
                mk = f"{key}_max"; pk = f"{key}_pos"
                au_max_help, _, _ = auroc_pos_higher_is_class(sub, mk, lambda r: r["lift_em"] == +1)
                au_pos_help, _, _ = auroc_pos_higher_is_class(sub, pk, lambda r: r["lift_em"] == +1)
                inc_sub = [r for r in sub if r["closed_em"] == 0]
                au_pos_fix, _, _ = auroc_pos_higher_is_class(inc_sub, pk, lambda r: r["lift_em"] == +1)
                print(f"    {key:<22} max→help={au_max_help:.3f}  pos→help={au_pos_help:.3f}  pos→fix|inc={au_pos_fix:.3f}  (n_inc={len(inc_sub)})")

        # ------ (4) Paired bootstrap: u_cons.max  vs  (u_cons.max × early-pos indicator) ------
        # composite score = u_cons.max if pos<1/3 else -inf-ish (i.e. only fire on early spikes)
        # We rank by this composite vs u_cons.max alone for predicting lift_em=+1.
        # Bootstrap the AUROC delta.
        print(f"\n  --- Paired bootstrap: AUROC for lift_em=+1, key feature × early-spike (pos<1/3) ---")
        for key in ("u_cons", "mean_token_entropy", "drift_cosine"):
            mk = f"{key}_max"; pk = f"{key}_pos"
            def s_max(sample, mk=mk):
                pos = [r[mk] for r in sample if r["lift_em"] == +1]
                neg = [r[mk] for r in sample if r["lift_em"] != +1]
                return auroc(pos, neg)
            def s_composite(sample, mk=mk, pk=pk):
                ranked_pos = [r[mk] if r[pk] < 1/3 else -1e9 for r in sample if r["lift_em"] == +1]
                ranked_neg = [r[mk] if r[pk] < 1/3 else -1e9 for r in sample if r["lift_em"] != +1]
                return auroc(ranked_pos, ranked_neg)
            def s_diff(sample, mk=mk, pk=pk):
                return s_composite(sample, mk, pk) - s_max(sample, mk)
            m_max, _, _ = bootstrap_ci(rows, s_max, B=1000, seed=11)
            m_cmp, _, _ = bootstrap_ci(rows, s_composite, B=1000, seed=11)
            m_d, lo, hi = bootstrap_ci(rows, s_diff, B=1000, seed=11)
            print(f"    {key:<22} max alone={m_max:.3f}  early×max={m_cmp:.3f}  Δ={m_d:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
