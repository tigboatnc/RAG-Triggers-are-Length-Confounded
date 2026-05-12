"""Cross-architecture summary — Phi (v2 full n=500) + Gemma + Qwen (xarch n=200).

Produces the 3-arch × 2-dataset grid of:

  (1) AUROC(u_cons.max → closed-book INCORRECT)
  (2) AUROC(dragin_rind_max.max → closed-book INCORRECT)
  (3) Paired bootstrap (u_cons.max − dragin_rind_max.max), 95% CI
  (4) AUROC(u_cons.max → lift_em=+1)            (deployment metric)
  (5) AUROC(dragin_rind_max.max → lift_em=+1)
  (6) Idea 1 — AUROC(mean_token_entropy.argmax_pos → lift_em=+1 | INCORRECT)
  (7) q70 deployment table: P(help|fire), P(harm|fire), Δprec

All analyses are dataset-paired within an arch. The Bonferroni-corrected
significance bar (α = 0.05 / 4 = 0.0125) is applied to the u_cons − dragin
paired AUROC; the others are reported as descriptive numbers.

Run with `gzn` conda env. Reads:

  Phi (legacy v2 Phi run):
    runs/v2_exp1_phi-4-mini-instruct/D{1,2}.jsonl + E{1,2}.jsonl
  Gemma:
    runs/v2_xarch_gemma-4-26b-a4b-it/D{1,2}.jsonl + E{1,2}.jsonl
  Qwen:
    runs/v2_xarch_qwen3.6-35b-a3b-instruct/D{1,2}.jsonl + E{1,2}.jsonl

Missing files for a given arch are reported and that arch is skipped — no
hard failure (lets the local analyst run this incrementally as the compute
box commits per-arch outputs).
"""
from __future__ import annotations
import json
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _common import auroc

ROOT = HERE.parent

ARCH_DIRS = {
    "phi-4-mini-instruct":      ROOT / "runs/v2_exp1_phi-4-mini-instruct",
    "gemma-4-26b-a4b-it":       ROOT / "runs/v2_xarch_gemma-4-26b-a4b-it",
    "qwen3.6-35b-a3b-instruct": ROOT / "runs/v2_xarch_qwen3.6-35b-a3b-instruct",
}
DATASETS = [
    ("HotpotQA  (in-domain)", "D1_closed_hotpot.jsonl", "E1_oracle_hotpot.jsonl"),
    ("2WikiMHQA (OOD)",       "D2_closed_wiki.jsonl",   "E2_oracle_wiki.jsonl"),
]


def f(v):
    try: return float(v) if v is not None else None
    except Exception: return None


def step_series(r, key):
    return [f(s.get(key)) for s in r.get("steps") or []]


def feat_max_and_pos(r, key):
    """Returns (max_value, normalised_argmax_position) — both None if no valid steps."""
    vs = [(i, v) for i, v in enumerate(step_series(r, key)) if v is not None]
    if not vs:
        return None, None
    if len(vs) == 1:
        return vs[0][1], 0.5
    imax, vmax = max(vs, key=lambda x: x[1])
    return vmax, imax / (len(vs) - 1)


def load_arch_dataset(arch_dir: Path, closed_name: str, oracle_name: str) -> list[dict] | None:
    cp = arch_dir / closed_name
    op = arch_dir / oracle_name
    if not cp.exists() or not op.exists():
        return None
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
        u_max, u_pos = feat_max_and_pos(c, "u_cons")
        d_max, d_pos = feat_max_and_pos(c, "dragin_rind_max")
        e_max, e_pos = feat_max_and_pos(c, "mean_token_entropy")
        if u_max is None or d_max is None or e_max is None:
            continue
        rows.append({
            "id": qid,
            "closed_em": c_em, "oracle_em": o_em, "lift_em": lift,
            "n_steps": int(c.get("n_steps", 0)),
            "u_cons_max": u_max, "u_cons_pos": u_pos,
            "dragin_max": d_max, "dragin_pos": d_pos,
            "ent_max": e_max,   "ent_pos": e_pos,
        })
    return rows


def auroc_class(rows, score_key, is_pos):
    pos = [r[score_key] for r in rows if is_pos(r)]
    neg = [r[score_key] for r in rows if not is_pos(r)]
    if not pos or not neg:
        return float("nan")
    return auroc(pos, neg)


def paired_bootstrap_auroc_diff(rows, key_a, key_b, is_pos, B=2000, seed=11):
    rng = random.Random(seed)
    n = len(rows)
    diffs = []
    for _ in range(B):
        s = [rows[rng.randrange(n)] for _ in range(n)]
        a = auroc_class(s, key_a, is_pos)
        b = auroc_class(s, key_b, is_pos)
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


def q70_deployment(rows, score_key, target=0.30):
    vs = sorted(r[score_key] for r in rows)
    if not vs:
        return None
    tau = vs[int((1 - target) * (len(vs) - 1))]
    fired = [r for r in rows if r[score_key] > tau]
    if not fired:
        return None
    n = len(rows); nf = len(fired)
    n_help = sum(1 for r in fired if r["lift_em"] == +1)
    n_harm = sum(1 for r in fired if r["lift_em"] == -1)
    base_help = sum(1 for r in rows if r["lift_em"] == +1) / n
    return {
        "tau": tau, "n_fire": nf, "fire_rate": nf / n,
        "p_help": n_help / nf, "p_harm": n_harm / nf,
        "base_help": base_help, "delta_prec": n_help / nf - base_help,
    }


def analyse_one(label, rows):
    n = len(rows)
    n_inc = sum(1 for r in rows if r["closed_em"] == 0)
    n_help = sum(1 for r in rows if r["lift_em"] == +1)
    n_harm = sum(1 for r in rows if r["lift_em"] == -1)
    multi = [r for r in rows if r["n_steps"] >= 2]
    print(f"\n  --- {label}  n={n}  incorrect={n_inc}  lift+1={n_help}  lift-1={n_harm}  multi-step={len(multi)} ---")

    inc = lambda r: r["closed_em"] == 0
    hlp = lambda r: r["lift_em"] == +1

    au_u_inc = auroc_class(rows, "u_cons_max", inc)
    au_d_inc = auroc_class(rows, "dragin_max", inc)
    au_u_hlp = auroc_class(rows, "u_cons_max", hlp)
    au_d_hlp = auroc_class(rows, "dragin_max", hlp)
    print(f"    AUROC →INCORRECT:  u_cons.max={au_u_inc:.3f}   dragin.max={au_d_inc:.3f}")
    print(f"    AUROC →lift+1:     u_cons.max={au_u_hlp:.3f}   dragin.max={au_d_hlp:.3f}")

    m, lo, hi, pp = paired_bootstrap_auroc_diff(rows, "u_cons_max", "dragin_max", inc)
    bonf = "✓ Bonf" if (lo > 0 or hi < 0) else "—"
    print(f"    PAIRED u_cons − dragin (→INCORRECT): {m:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  P(>0)={pp:.3f}  {bonf}")
    m, lo, hi, pp = paired_bootstrap_auroc_diff(rows, "u_cons_max", "dragin_max", hlp)
    bonf = "✓ Bonf" if (lo > 0 or hi < 0) else "—"
    print(f"    PAIRED u_cons − dragin (→lift+1):    {m:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  P(>0)={pp:.3f}  {bonf}")

    # Idea-1 within incorrect: spike-pos of entropy → fixable
    if multi:
        inc_multi = [r for r in multi if r["closed_em"] == 0]
        au_pos_fix = auroc_class(inc_multi, "ent_pos", lambda r: r["lift_em"] == +1)
        au_dragin_pos_fix = auroc_class(inc_multi, "dragin_pos", lambda r: r["lift_em"] == +1)
        au_ucons_pos_fix = auroc_class(inc_multi, "u_cons_pos", lambda r: r["lift_em"] == +1)
        print(f"    IDEA-1 (multi-step n={len(multi)}, incorrect-only n={len(inc_multi)}):")
        print(f"      AUROC(ent_pos → fix|inc)     = {au_pos_fix:.3f}")
        print(f"      AUROC(dragin_pos → fix|inc)  = {au_dragin_pos_fix:.3f}")
        print(f"      AUROC(u_cons_pos → fix|inc)  = {au_ucons_pos_fix:.3f}")

    print(f"    DEPLOYMENT (u_cons.max gate, base_help={n_help/n:.3f}):")
    for tgt in (0.05, 0.10, 0.30):
        r = q70_deployment(rows, "u_cons_max", target=tgt)
        if r:
            print(f"      target={tgt:.2f}  fire={r['fire_rate']*100:5.1f}%  P(help|F)={r['p_help']:.3f}  "
                  f"P(harm|F)={r['p_harm']:.3f}  Δprec={r['delta_prec']:+.3f}")
    print(f"    DEPLOYMENT (dragin.max gate):")
    for tgt in (0.05, 0.10, 0.30):
        r = q70_deployment(rows, "dragin_max", target=tgt)
        if r:
            print(f"      target={tgt:.2f}  fire={r['fire_rate']*100:5.1f}%  P(help|F)={r['p_help']:.3f}  "
                  f"P(harm|F)={r['p_harm']:.3f}  Δprec={r['delta_prec']:+.3f}")


def main():
    for arch, arch_dir in ARCH_DIRS.items():
        print(f"\n{'='*88}\n=== {arch}  (dir: {arch_dir.relative_to(ROOT)}) ===")
        if not arch_dir.exists():
            print(f"  [SKIP] arch run dir missing")
            continue
        for label, closed, oracle in DATASETS:
            rows = load_arch_dataset(arch_dir, closed, oracle)
            if rows is None:
                print(f"  [SKIP] {label}: missing {closed} or {oracle}")
                continue
            if not rows:
                print(f"  [SKIP] {label}: 0 paired rows")
                continue
            analyse_one(label, rows)


if __name__ == "__main__":
    main()
