"""A1 — K=1 (Datta-style) vs K=5 (ours) flip_rate ablation.

Goal (per `_rq1-plan.md` lit-precedent block, 2026-05-14):
    Substantiate the K=5 multi-sample contribution against the Datta et al. (2026,
    arxiv:2604.22266) forced-answer-completion precedent.

    Datta et al. compute AnswerSwitches = sum_i [A_i != A_{i+1}] from a single
    greedy intermediate prediction per reasoning step (forced answer completion).
    Our flip_rate uses K=5 boundary-completion answers per step and majority-votes.

    This script reads existing P0 pilot traces (rq1-pilot-v1 schema, K=5 samples
    per step), re-derives BOTH K=1 (samples[0] only) and K=5 (majority over
    samples[0..4]) flip_rate, and runs a paired bootstrap (B=10,000) of the
    AUROC delta against `lift_em = +1` and `closed_em = 0`.

Honest caveat:
    "K=1" here is the FIRST sample of an existing K=5 stochastic capture — NOT a
    perfectly faithful Datta-style greedy forced-completion. Datta uses
    deterministic greedy decoding after appending the answer prefix; our
    `samples[0]` is the first stochastic completion under the configured K=5
    sampler. Results should be read as a directional proxy. A perfectly faithful
    replication would re-decode P0 with K=1 greedy + answer-prefix forcing, which
    is ~equivalent to the existing N2 `prefix_answer_now` capture in `rq1-pilot-v2`
    (cancelled). This proxy is good enough for a pre-R1 directional read.

Outputs:
    - stdout: per-cell tables of K=1 vs K=5 raw and residualized AUROC + paired
      bootstrap CI of (K=5 - K=1) for both `lift_em=+1` and `closed_em=0` targets
    - JSON dump: analysis/a1_k1_vs_k5_p0.json (for paper Table 2)

Usage:
    python analysis/a1_k1_vs_k5_ablation.py
    # Or override defaults:
    python analysis/a1_k1_vs_k5_ablation.py --bootstrap-b 10000 --seed 20260514
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _common import auroc as _auroc

ROOT = HERE.parent

ANSWER_RE = re.compile(r"\bAnswer\s*:\s*(.+?)(?:[\n.]|$)", re.IGNORECASE)


def auroc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    return _auroc(pos, neg)


def normalize_answer(s: str) -> str:
    s = re.sub(r"[^a-z0-9 ]", "", s.strip().lower())[:30].strip()
    return s


def extract_answer(text: str) -> str | None:
    if not text:
        return None
    m = ANSWER_RE.search(text)
    if not m:
        return None
    a = normalize_answer(m.group(1))
    return a if a else None


def per_step_answers(steps: list[dict], k_top: int) -> list[list[str]]:
    """For each step, return the list of extracted answers from samples[0..k_top-1]."""
    out: list[list[str]] = []
    for s in steps:
        samples = s.get("samples", []) or []
        ans = []
        for samp in samples[:k_top]:
            a = extract_answer(samp.get("text", ""))
            if a:
                ans.append(a)
        out.append(ans)
    return out


def flip_rate_from_step_answers(per_step: list[list[str]], mode: str) -> float | None:
    """Compute flip_rate over answer-bearing steps.

    mode='majority': per step, take majority answer (Counter most_common). Compare
                     consecutive answer-bearing steps; flip if majority differs.
                     This is our K=5 definition.
    mode='single':   per step, take answers[0] (the first sample). Compare
                     consecutive; flip if answer differs. This is the K=1 /
                     Datta-style baseline (using samples[0] as the proxy for
                     greedy forced-answer-completion).
    """
    answer_steps: list[str] = []
    for ans in per_step:
        if not ans:
            continue
        if mode == "majority":
            counts = Counter(ans)
            answer_steps.append(counts.most_common(1)[0][0])
        elif mode == "single":
            answer_steps.append(ans[0])
        else:
            raise ValueError(mode)
    if len(answer_steps) < 2:
        return None
    flips = sum(1 for i in range(1, len(answer_steps)) if answer_steps[i] != answer_steps[i - 1])
    return flips / max(1, len(answer_steps) - 1)


def residualize_ols(scores: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Return residuals of `scores ~ x` (OLS, single regressor + intercept)."""
    valid = np.isfinite(scores) & np.isfinite(x)
    if valid.sum() < 3:
        return scores - np.nanmean(scores[valid]) if valid.any() else scores
    X = np.column_stack([np.ones(valid.sum()), x[valid]])
    beta, *_ = np.linalg.lstsq(X, scores[valid], rcond=None)
    pred = beta[0] + beta[1] * x
    return scores - pred


def paired_bootstrap_delta(
    s_a: np.ndarray,
    s_b: np.ndarray,
    labels: np.ndarray,
    B: int,
    seed: int,
) -> tuple[float, float, float]:
    """Paired bootstrap: resample indices with replacement, compute
    AUROC(s_b | labels) - AUROC(s_a | labels) on each resample. Returns
    (point_estimate, ci_lo_2.5%, ci_hi_97.5%)."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    pos_mask = labels == 1
    neg_mask = labels == 0
    point = auroc(list(s_b[pos_mask]), list(s_b[neg_mask])) - auroc(
        list(s_a[pos_mask]), list(s_a[neg_mask])
    )
    deltas = np.empty(B, dtype=float)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        lb = labels[idx]
        if lb.sum() == 0 or lb.sum() == n:
            deltas[b] = float("nan")
            continue
        a_b = auroc(list(s_a[idx][lb == 1]), list(s_a[idx][lb == 0]))
        b_b = auroc(list(s_b[idx][lb == 1]), list(s_b[idx][lb == 0]))
        deltas[b] = b_b - a_b
    finite = deltas[np.isfinite(deltas)]
    if len(finite) == 0:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(finite, [2.5, 97.5])
    return point, float(lo), float(hi)


def cell_label(arch: str) -> str:
    """Short cell label (phi/gemma/qwen)."""
    if "phi" in arch:
        return "phi"
    if "gemma" in arch:
        return "gemma"
    if "qwen" in arch:
        return "qwen"
    return arch


def load_paired_pilot_cells(
    archs: list[str],
    datasets: list[tuple[str, str, str]],
    pilot_root: str = "v2_pilot",
) -> list[dict]:
    """Load pilot paired traces. Returns flat list of trace dicts with keys:
    arch, ds, cell, c (closed trace), o (oracle trace), c_em, o_em, lift."""
    rows: list[dict] = []
    for arch in archs:
        arch_short = cell_label(arch)
        for ds, cf, of in datasets:
            cp = ROOT / f"runs/{pilot_root}_{arch}/{cf}"
            op = ROOT / f"runs/{pilot_root}_{arch}/{of}"
            if not cp.exists() or not op.exists():
                continue
            closed = {r["id"]: r for r in (json.loads(l) for l in cp.open() if l.strip())}
            oracle = {r["id"]: r for r in (json.loads(l) for l in op.open() if l.strip())}
            for qid, c in closed.items():
                if qid not in oracle:
                    continue
                o = oracle[qid]
                c_em = int(bool(c.get("label_correct_em", c.get("em", False))))
                o_em = int(bool(o.get("label_correct_em", o.get("em", False))))
                lift = +1 if o_em > c_em else (-1 if o_em < c_em else 0)
                rows.append(
                    {
                        "arch": arch_short,
                        "ds": ds,
                        "cell": f"{arch_short}_{ds}",
                        "c": c,
                        "o": o,
                        "c_em": c_em,
                        "o_em": o_em,
                        "lift": lift,
                    }
                )
    return rows


def derive_features(rows: list[dict]) -> None:
    """Mutates rows: adds flip_rate_k1, flip_rate_k5, tokens_log."""
    for r in rows:
        steps = r["c"].get("steps", []) or []
        psa = per_step_answers(steps, k_top=5)
        r["flip_rate_k1"] = flip_rate_from_step_answers(psa, mode="single")
        r["flip_rate_k5"] = flip_rate_from_step_answers(psa, mode="majority")
        r["tokens_log"] = math.log(int(r["c"].get("tokens_generated", 0)) + 1)


def report_target(
    rows: list[dict],
    target_name: str,
    label_fn,
    B: int,
    seed: int,
) -> dict:
    """Per-cell + overall: K=1 / K=5 raw + residualized AUROC + paired-bootstrap delta."""
    out = {"target": target_name, "cells": {}, "overall": {}}

    cells = defaultdict(list)
    for r in rows:
        cells[r["cell"]].append(r)

    print(f"\n{'=' * 90}")
    print(f"TARGET: {target_name}")
    print(f"{'=' * 90}")
    print(
        f"{'cell':<12} {'n':>4} {'pop':>4} {'K1_raw':>7} {'K5_raw':>7} {'Δraw':>8}  "
        f"{'K1_res':>7} {'K5_res':>7} {'Δres':>8}  {'Δraw 95%CI':>20} {'Δres 95%CI':>20}"
    )
    print("-" * 130)

    def cell_metrics(cell_rows: list[dict]) -> dict:
        valid = [r for r in cell_rows if r["flip_rate_k1"] is not None and r["flip_rate_k5"] is not None]
        if len(valid) < 4:
            return {"n": len(valid), "skipped": True}
        labels = np.array([1 if label_fn(r) else 0 for r in valid], dtype=int)
        pop = int(labels.sum())
        if pop < 2 or pop > len(valid) - 2:
            return {"n": len(valid), "pop": pop, "skipped": True}
        s_k1 = np.array([float(r["flip_rate_k1"]) for r in valid], dtype=float)
        s_k5 = np.array([float(r["flip_rate_k5"]) for r in valid], dtype=float)
        x_log = np.array([float(r["tokens_log"]) for r in valid], dtype=float)

        a_k1 = auroc(list(s_k1[labels == 1]), list(s_k1[labels == 0]))
        a_k5 = auroc(list(s_k5[labels == 1]), list(s_k5[labels == 0]))

        r_k1 = residualize_ols(s_k1, x_log)
        r_k5 = residualize_ols(s_k5, x_log)
        ar_k1 = auroc(list(r_k1[labels == 1]), list(r_k1[labels == 0]))
        ar_k5 = auroc(list(r_k5[labels == 1]), list(r_k5[labels == 0]))

        d_raw, lo_raw, hi_raw = paired_bootstrap_delta(s_k1, s_k5, labels, B=B, seed=seed)
        d_res, lo_res, hi_res = paired_bootstrap_delta(r_k1, r_k5, labels, B=B, seed=seed + 1)

        return {
            "n": len(valid),
            "pop": pop,
            "k1_raw": a_k1,
            "k5_raw": a_k5,
            "delta_raw_point": d_raw,
            "delta_raw_ci": (lo_raw, hi_raw),
            "k1_res": ar_k1,
            "k5_res": ar_k5,
            "delta_res_point": d_res,
            "delta_res_ci": (lo_res, hi_res),
        }

    for cell_name in sorted(cells.keys()):
        m = cell_metrics(cells[cell_name])
        out["cells"][cell_name] = m
        if m.get("skipped"):
            print(f"{cell_name:<12} {m.get('n', 0):>4} {m.get('pop', 0):>4} (skipped — too few samples or degenerate labels)")
            continue
        print(
            f"{cell_name:<12} {m['n']:>4} {m['pop']:>4} "
            f"{m['k1_raw']:>7.3f} {m['k5_raw']:>7.3f} {m['delta_raw_point']:>+8.3f}  "
            f"{m['k1_res']:>7.3f} {m['k5_res']:>7.3f} {m['delta_res_point']:>+8.3f}  "
            f"[{m['delta_raw_ci'][0]:>+6.3f},{m['delta_raw_ci'][1]:>+6.3f}]  "
            f"[{m['delta_res_ci'][0]:>+6.3f},{m['delta_res_ci'][1]:>+6.3f}]"
        )

    print("-" * 130)
    m_all = cell_metrics(rows)
    out["overall"] = m_all
    if not m_all.get("skipped"):
        print(
            f"{'OVERALL':<12} {m_all['n']:>4} {m_all['pop']:>4} "
            f"{m_all['k1_raw']:>7.3f} {m_all['k5_raw']:>7.3f} {m_all['delta_raw_point']:>+8.3f}  "
            f"{m_all['k1_res']:>7.3f} {m_all['k5_res']:>7.3f} {m_all['delta_res_point']:>+8.3f}  "
            f"[{m_all['delta_raw_ci'][0]:>+6.3f},{m_all['delta_raw_ci'][1]:>+6.3f}]  "
            f"[{m_all['delta_res_ci'][0]:>+6.3f},{m_all['delta_res_ci'][1]:>+6.3f}]"
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap-b", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=20260514)
    ap.add_argument(
        "--archs",
        nargs="+",
        default=["phi-4-mini-instruct", "gemma-4-26b-a4b-it", "qwen2.5-7b-instruct"],
    )
    ap.add_argument("--out", type=str, default="analysis/a1_k1_vs_k5_p0.json")
    ap.add_argument("--pilot-root", type=str, default="v2_pilot")
    args = ap.parse_args()

    datasets = [
        ("hot", "D1_closed_hotpot.jsonl", "E1_oracle_hotpot.jsonl"),
        ("wiki", "D2_closed_wiki.jsonl", "E2_oracle_wiki.jsonl"),
    ]
    rows = load_paired_pilot_cells(args.archs, datasets, pilot_root=args.pilot_root)
    print(f"\nLoaded {len(rows)} paired traces across {len({r['cell'] for r in rows})} cells.")
    print(
        "(per cell up to n=25 closed×oracle pairs from rq1-pilot-v1 schema; K=5 sampler captures.)"
    )

    derive_features(rows)

    n_with_k1 = sum(1 for r in rows if r["flip_rate_k1"] is not None)
    n_with_k5 = sum(1 for r in rows if r["flip_rate_k5"] is not None)
    print(
        f"flip_rate_k1 derivable: {n_with_k1}/{len(rows)}    "
        f"flip_rate_k5 derivable: {n_with_k5}/{len(rows)}"
    )

    print(
        "\nNote: K=1 here uses samples[0].text (first stochastic completion) as the\n"
        "Datta-style greedy proxy. Not a perfectly faithful replication of\n"
        "forced-answer-completion (which uses deterministic greedy + answer-prefix\n"
        "forcing). Read as directional. R1 will run the proper K=1 ablation if needed."
    )

    res_lift = report_target(
        rows, "lift_em == +1 (retrieval would have helped)",
        label_fn=lambda r: r["lift"] == +1, B=args.bootstrap_b, seed=args.seed,
    )
    res_inc = report_target(
        rows, "closed_em == 0 (closed-book incorrect)",
        label_fn=lambda r: r["c_em"] == 0, B=args.bootstrap_b, seed=args.seed + 100,
    )

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "a1-k1-vs-k5-p0-v1",
        "n_traces": len(rows),
        "n_cells": len({r["cell"] for r in rows}),
        "bootstrap_b": args.bootstrap_b,
        "seed": args.seed,
        "archs": args.archs,
        "pilot_root": args.pilot_root,
        "datasets": [d[0] for d in datasets],
        "lift_target": res_lift,
        "incorrect_target": res_inc,
        "caveat": (
            "K=1 uses samples[0].text (first stochastic completion) as Datta-style proxy. "
            "Not faithful greedy forced-completion. Read as directional pre-R1."
        ),
    }

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, float) and not math.isfinite(o):
            return None
        if isinstance(o, np.floating):
            v = float(o)
            return None if not math.isfinite(v) else v
        if isinstance(o, np.integer):
            return int(o)
        return o

    out_path.write_text(json.dumps(_clean(payload), indent=2))
    print(f"\nWrote {out_path.relative_to(ROOT)}")
    print("\nNext step (Day 2 of week-of-2026-05-14 plan): re-run this script with")
    print("--pilot-root v2_headline once R1 lands, to get the publishable K=1 vs K=5")
    print("ablation table at n=200/cell.")


if __name__ == "__main__":
    main()
