"""R1 Phi paired audit — full trigger-pool, both targets.

Consumes the just-landed R1 Phi cells:
    runs/v2_headline_phi-4-mini-instruct/{D1,D2,E1,E2}_*.jsonl
    (n=200 each; closed-book × paired oracle re-decode)

Emits the full headline-grade audit:
    - full trigger pool (16 signals + 4 AID variants), trace-max scalars
    - both targets: `lift_em = +1` (paper headline) and `closed_em = 0`
    - raw AUROC + length-residualized AUROC (signal ~ log(tokens+1))
    - paired bootstrap (B=10,000) of (signal − tokens) AUROC
    - per-question-type stratification (bridge / comparison / inference / etc.)
    - precision-at-budget at fire rates {5%, 10%, 20%}
    - V2 sanity-check comparison row (existing n=393/452 cohort)

Usage:
    python analysis/r1_phi_paired_audit.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _common import auroc as _auroc

# Reuse helpers from r1_phi_d1_closed_audit
sys.path.insert(0, str(HERE))
from r1_phi_d1_closed_audit import (
    ANSWER_RE, _f, step_vals, tmax, trange,
    etc_score, hvr_max, flare_max_per_step, seakr_k5_logdet,
    extract_answer, per_step_answers,
    flip_rate_majority, flip_rate_single, late_flips_count, min_majority_inv,
    auroc_pos, precision_at_k, ols_residual, bootstrap_auroc_diff,
    SIGNALS,
)

ROOT = HERE.parent


def load_paired(cp: Path, op: Path):
    if not cp.exists() or not op.exists():
        return []
    closed = {r["id"]: r for r in (json.loads(l) for l in cp.open() if l.strip())}
    oracle = {r["id"]: r for r in (json.loads(l) for l in op.open() if l.strip())}
    rows = []
    for qid, c in closed.items():
        if qid not in oracle:
            continue
        o = oracle[qid]
        c_em = int(bool(c.get("label_correct_em", c.get("em", False))))
        o_em = int(bool(o.get("label_correct_em", o.get("em", False))))
        lift = +1 if o_em > c_em else (-1 if o_em < c_em else 0)
        steps = c.get("steps") or []
        psa5 = per_step_answers(steps, k_top=5)
        rows.append({
            "id": qid,
            "type": c.get("type"),
            "level": c.get("level"),
            "closed_em": c_em,
            "oracle_em": o_em,
            "lift_em": lift,
            "n_steps": int(c.get("n_steps", 0)),
            "tokens": int(c.get("tokens_generated", 0)),
            "log_tokens": math.log(max(1, int(c.get("tokens_generated", 0))) + 1),
            "wall_s": float(c.get("wall_clock_s") or 0.0),
            "u_cons_max": tmax(c, "u_cons") or 0.0,
            "dragin_max": tmax(c, "dragin_rind_max") or 0.0,
            "ent_max": tmax(c, "mean_token_entropy") or 0.0,
            "etc": etc_score(c),
            "hvr_max": hvr_max(c),
            "drift_max": tmax(c, "drift_cosine") or 0.0,
            "var_max": tmax(c, "variance_l2_window") or 0.0,
            "hidden_range": trange(c, "hidden_norm"),
            "flare_max": flare_max_per_step(c),
            "seakr_k5_logdet": seakr_k5_logdet(c),
            "flip_k5": flip_rate_majority(psa5),
            "flip_k1": flip_rate_single(psa5),
            "late_flips": late_flips_count(psa5, len(steps)),
            "min_majority": min_majority_inv(psa5),
        })
    return rows


def banner(s: str):
    print(f"\n{'=' * 100}")
    print(s)
    print("=" * 100)


def report_target(rows, label: str, target_name: str, label_fn, B: int):
    n = len(rows)
    pop = sum(1 for r in rows if label_fn(r))
    base = pop / n if n else float("nan")
    banner(f"{label}    target={target_name}    n={n}    pop={pop}    base_rate={base:.3f}")

    print("\n--- (1) RAW AUROC + P@k ---")
    print(f"  {'signal':<32} {'pop':>5} {'AUROC':>7} {'P@5%':>7} {'P@10%':>7} {'P@20%':>7}   {'ΔP@5%':>7}")
    for key, name, family in SIGNALS:
        valid = [r for r in rows if r.get(key) is not None]
        npop = sum(1 for r in valid if label_fn(r))
        if npop < 2 or npop > len(valid) - 2:
            print(f"  {name:<32} {len(valid):>5}   (skipped — degenerate)")
            continue
        au = auroc_pos(rows, key, label_fn)
        _, p5, _ = precision_at_k(rows, key, 0.05, label_fn)
        _, p10, _ = precision_at_k(rows, key, 0.10, label_fn)
        _, p20, _ = precision_at_k(rows, key, 0.20, label_fn)
        d5 = p5 - base if p5 == p5 else float("nan")
        print(f"  {name:<32} {len(valid):>5} {au:>7.3f} {p5:>7.3f} {p10:>7.3f} {p20:>7.3f}   {d5:>+7.3f}")

    print("\n--- (2) LENGTH-RESIDUALIZED AUROC (signal ~ log_tokens) ---")
    print(f"  {'signal':<32} {'raw AUROC':>10} {'resid AUROC':>12} {'ΔAUROC':>8}   {'survives?':>10}")
    for key, name, family in SIGNALS:
        if key == "tokens":
            continue
        valid = [r for r in rows if r.get(key) is not None]
        if not valid:
            continue
        resid_map, _ = ols_residual(valid, key, x_key="log_tokens")
        for r in valid:
            r[f"{key}_resid"] = resid_map.get(id(r))
        au_raw = auroc_pos(rows, key, label_fn)
        au_res = auroc_pos(rows, f"{key}_resid", label_fn)
        if au_raw != au_raw or au_res != au_res:
            print(f"  {name:<32}   (degenerate)")
            continue
        delta = au_res - au_raw
        survives = "✓ >0.5" if au_res > 0.50 else "—"
        print(f"  {name:<32} {au_raw:>10.3f} {au_res:>12.3f} {delta:>+8.3f}   {survives:>10}")

    print(f"\n--- (3) PAIRED BOOTSTRAP (signal − tokens) AUROC, B={B} ---")
    print(f"  {'signal':<32} {'Δ vs tokens':>13} {'95% CI':>22} {'sig?':>6}")
    for key, name, family in SIGNALS:
        if key == "tokens":
            continue
        m, lo, hi, _ = bootstrap_auroc_diff(rows, key, "tokens", label_fn, B=B)
        sig = "✓" if (lo == lo and hi == hi and (lo > 0 or hi < 0)) else "—"
        # Direction matters: positive = signal beats tokens
        marker = ""
        if lo == lo and lo > 0:
            marker = "  ← BEATS tokens"
        elif hi == hi and hi < 0:
            marker = "  (worse than tokens)"
        print(f"  {name:<32} {m:>+13.3f}  [{lo:>+6.3f},{hi:>+6.3f}]  {sig:>6}{marker}")

    # Per-question-type
    print("\n--- (4) PER-QUESTION-TYPE AUROC (raw) ---")
    types = sorted({r["type"] for r in rows if r.get("type")})
    if types:
        print("  " + f"{'signal':<32}" + " ".join(f"{t[:14]:>14}" for t in types))
        for key, name, _ in SIGNALS:
            line = f"  {name:<32}"
            for t in types:
                sub = [r for r in rows if r.get("type") == t and r.get(key) is not None]
                p = [r[key] for r in sub if label_fn(r)]
                n_neg = [r[key] for r in sub if not label_fn(r)]
                if not p or not n_neg:
                    line += f" {'n/a':>14}"
                else:
                    line += f" {_auroc(p, n_neg):>14.3f}"
            print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="v2_headline")
    ap.add_argument("--arch", default="phi-4-mini-instruct")
    ap.add_argument("--bootstrap-b", type=int, default=10_000)
    ap.add_argument("--include-v2-comparison", action="store_true")
    args = ap.parse_args()

    arch_dir = ROOT / f"runs/{args.prefix}_{args.arch}"
    cells = [
        ("HotpotQA-distractor (in-domain)", "D1_closed_hotpot.jsonl", "E1_oracle_hotpot.jsonl"),
        ("2WikiMHQA-distractor (OOD)",       "D2_closed_wiki.jsonl",   "E2_oracle_wiki.jsonl"),
    ]

    all_rows = []
    for cell_label, df, of in cells:
        rows = load_paired(arch_dir / df, arch_dir / of)
        if not rows:
            print(f"[skip] {cell_label} — no paired traces under {arch_dir}")
            continue
        n = len(rows)
        n_inc = sum(1 for r in rows if r["closed_em"] == 0)
        n_help = sum(1 for r in rows if r["lift_em"] == +1)
        n_hurt = sum(1 for r in rows if r["lift_em"] == -1)
        print(f"\n{cell_label}: {n} paired traces  closed_inc={n_inc}  lift+1={n_help}  lift-1={n_hurt}  base_help={n_help/n:.3f}")

        report_target(rows, f"R1 {args.arch} × {cell_label}", "lift_em == +1 (RETRIEVAL HELPED)", lambda r: r["lift_em"] == +1, B=args.bootstrap_b)
        report_target(rows, f"R1 {args.arch} × {cell_label}", "closed_em == 0 (closed-book incorrect)", lambda r: r["closed_em"] == 0, B=args.bootstrap_b)

        for r in rows:
            r["_cell"] = cell_label
        all_rows.extend(rows)

    if len(all_rows) > 200 and len([r for r in all_rows if r["lift_em"] == +1]) >= 8:
        banner(f"R1 {args.arch} — POOLED across cells (n={len(all_rows)})")
        report_target(all_rows, f"R1 {args.arch} POOLED", "lift_em == +1", lambda r: r["lift_em"] == +1, B=args.bootstrap_b)

    if args.include_v2_comparison:
        print("\n" + "#" * 100)
        print("# V2 SANITY-CHECK comparison (existing n=393/452 paired Phi cohort)")
        print("#" * 100)
        v2_dir = ROOT / "runs/v2_exp1_phi-4-mini-instruct"
        for cell_label, df, of in cells:
            rows = load_paired(v2_dir / df, v2_dir / of)
            if not rows:
                continue
            n = len(rows)
            n_help = sum(1 for r in rows if r["lift_em"] == +1)
            print(f"\nV2 {cell_label}: {n} paired traces, lift+1={n_help}")
            report_target(rows, f"V2 {args.arch} × {cell_label}", "lift_em == +1 (V2 SANITY)", lambda r: r["lift_em"] == +1, B=args.bootstrap_b)


if __name__ == "__main__":
    main()
