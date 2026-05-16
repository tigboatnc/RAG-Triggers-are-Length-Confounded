"""Follow-up: per-trace rank-fusion ensemble of AID min_majority and M9 CABV*LAAS.

Hypothesis from `moonshot_explore.py` per-cell breakdown on lift_em=+1:

    cell             min_majority    M9 CABV*LAAS
    Phi_hotpot       0.669 (win)     0.520
    Phi_wiki         0.409 (DIES)    0.734 (win)
    Qwen_hotpot      0.659           0.687 (marg win)
    Qwen_wiki        0.619           0.655 (marg win)

The two signals are COMPLEMENTARY across cells. min_majority captures the
within-step answer-distribution narrowness; M9 captures the joint pathology
of (answer-distribution drift over steps) AND (sample-length CV), which is
high precisely on the (Phi × 2Wiki) cell where min_majority collapses.

We test three fusion rules:

    F1 = max(rank(min_majority), rank(M9))         — trigger if EITHER is high
    F2 = mean(rank(min_majority), rank(M9))        — average
    F3 = sqrt(rank(min_majority) * rank(M9))       — geometric (both must agree)

Ranks are per-cell percentile to neutralise scale.

Reported: raw AUROC, residualised AUROC, paired-bootstrap vs tokens AND
vs the stronger of the two component signals on each cell.
"""
from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from r1_phi_d1_closed_audit import (
    auroc_pos,
    bootstrap_auroc_diff,
    ols_residual,
)
from moonshot_explore import (
    is_lift_pos,
    is_closed_neg,
    load_arch,
)

ROOT = HERE.parent


def _percentile_rank(vals: list[float | None]) -> list[float | None]:
    """Per-cell percentile rank in [0, 1]; None preserved as None."""
    idx_v = [(i, v) for i, v in enumerate(vals) if v is not None]
    if not idx_v:
        return [None] * len(vals)
    sorted_iv = sorted(idx_v, key=lambda iv: iv[1])
    n = len(sorted_iv)
    rank_of: dict[int, float] = {}
    for r, (i, _) in enumerate(sorted_iv):
        rank_of[i] = r / max(1, n - 1)
    out: list[float | None] = []
    for i, v in enumerate(vals):
        out.append(rank_of[i] if v is not None else None)
    return out


def add_per_cell_ranks(rows: list[dict], keys: list[str], cell_key: tuple[str, ...]) -> None:
    """Mutate rows: add r_<key> per-cell percentile rank for each key."""
    cells: dict[tuple, list[int]] = {}
    for i, r in enumerate(rows):
        c = tuple(r[k] for k in cell_key)
        cells.setdefault(c, []).append(i)
    for c, idxs in cells.items():
        for k in keys:
            vals = [rows[i].get(k) for i in idxs]
            ranks = _percentile_rank(vals)
            for i, rk in zip(idxs, ranks):
                rows[i][f"r_{k}"] = rk


def fuse(rows: list[dict], k1: str, k2: str, mode: str) -> None:
    out_key = f"F_{mode}"
    for r in rows:
        v1 = r.get(f"r_{k1}")
        v2 = r.get(f"r_{k2}")
        if v1 is None or v2 is None:
            r[out_key] = None
            continue
        if mode == "max":
            r[out_key] = max(v1, v2)
        elif mode == "mean":
            r[out_key] = 0.5 * (v1 + v2)
        elif mode == "geom":
            r[out_key] = math.sqrt(v1 * v2)


def banner(s):
    print("\n" + "=" * 100)
    print(s)
    print("=" * 100)


def report_one(rows, label, keys: list[tuple[str, str]], B: int):
    banner(label + "  (lift_em=+1)")
    pop = sum(1 for r in rows if is_lift_pos(r))
    n = len(rows)
    print(f"  n={n}  pop={pop}  base={pop/n:.3f}\n")

    # Pre-compute min_majority length-residualised once for this rowset, so
    # the Δ-vs-min_majority bootstrap can compare residualised signal vs
    # residualised reference survivor (matches the §3.7 paper formula).
    mm_valid = [r for r in rows if r.get("min_majority") is not None]
    if mm_valid:
        mm_res_map, _ = ols_residual(mm_valid, "min_majority", x_key="log_tokens")
        for r in mm_valid:
            r["__res_mm__"] = mm_res_map[id(r)]

    print(f"  {'signal':<48} {'valid':>6} {'rawAUC':>7} {'resAUC':>7}    {'Δ vs tokens':>22}    {'Δ vs min_majority':>24}")
    for key, name in keys:
        valid = [r for r in rows if r.get(key) is not None]
        if len(valid) < 6:
            print(f"  {name:<48} {len(valid):>6}  insufficient")
            continue
        au = auroc_pos(valid, key, is_lift_pos)
        # synthesise log_tokens key on each row already exists
        res_map, _ = ols_residual(valid, key, x_key="log_tokens")
        for r in valid:
            r["__res__"] = res_map[id(r)]
        res_au = auroc_pos(valid, "__res__", is_lift_pos)
        if key == "tokens":
            d_tok = "(baseline)            "
        else:
            # Δ = AUROC(s_resid) − AUROC(tokens_generated)  (matches paper §3.7)
            m, lo, hi, _ = bootstrap_auroc_diff(valid, "__res__", "tokens", is_lift_pos, B=B, seed=11)
            star = " *" if (lo > 0 or hi < 0) else "  "
            d_tok = f"{m:+.3f} [{lo:+.3f},{hi:+.3f}]{star}"
        if key == "min_majority":
            d_mm = "(reference)             "
        else:
            valid_both = [r for r in rows if r.get(key) is not None and r.get("min_majority") is not None]
            if len(valid_both) < 6:
                d_mm = "n/a"
            else:
                # Δ = AUROC(s_resid) − AUROC(min_majority_resid):
                # consistent residualised head-to-head against the reference survivor.
                m2, lo2, hi2, _ = bootstrap_auroc_diff(valid_both, "__res__", "__res_mm__", is_lift_pos, B=B, seed=13)
                star2 = " *" if (lo2 > 0 or hi2 < 0) else "  "
                d_mm = f"{m2:+.3f} [{lo2:+.3f},{hi2:+.3f}]{star2}"
        for r in valid:
            r.pop("__res__", None)
        print(f"  {name:<48} {len(valid):>6} {au:>7.3f} {res_au:>7.3f}    {d_tok}    {d_mm}")

    for r in mm_valid:
        r.pop("__res_mm__", None)


def per_cell_table(rows, keys: list[tuple[str, str]]):
    banner("Per-cell raw AUROC (lift_em=+1)")
    cells = sorted({(r["arch"], r["dataset"]) for r in rows})
    hdr = "  " + f"{'cell':<24} {'n':>4} {'pop':>4} | " + "  ".join(f"{name[:14]:>14}" for _, name in keys)
    print(hdr)
    for arch, ds in cells:
        sub = [r for r in rows if r["arch"] == arch and r["dataset"] == ds]
        pop = sum(1 for r in sub if is_lift_pos(r))
        line = f"  {arch+'_'+ds:<24} {len(sub):>4} {pop:>4} | "
        for key, _name in keys:
            v = [r for r in sub if r.get(key) is not None]
            au = auroc_pos(v, key, is_lift_pos) if len(v) >= 6 and pop >= 3 else float("nan")
            line += f"  {au:>14.3f}"
        print(line)


def main():
    B = 5000
    phi = load_arch("phi-4-mini-instruct", ROOT)
    qwen = load_arch("qwen2.5-7b-instruct", ROOT)
    gemma = load_arch("gemma-4-26b-a4b-it", ROOT)
    print(f"\nLoaded:  Phi paired = {len(phi)},  Qwen paired = {len(qwen)},  Gemma paired = {len(gemma)},  pooled = {len(phi)+len(qwen)+len(gemma)}")

    all_rows = phi + qwen + gemma
    rows_no_phi = qwen + gemma
    # cell-aware percentile-rank fusion (per arch × dataset to neutralise scale heterogeneity)
    add_per_cell_ranks(all_rows, ["min_majority", "M9_cabv_x_laas", "M1_cabv", "dragin_max"], ("arch", "dataset"))
    fuse(all_rows, "min_majority", "M9_cabv_x_laas", "max")
    fuse(all_rows, "min_majority", "M9_cabv_x_laas", "mean")
    fuse(all_rows, "min_majority", "M9_cabv_x_laas", "geom")
    fuse(all_rows, "min_majority", "M1_cabv", "max")  # extra sanity ensemble

    keys_to_test = [
        ("tokens",          "tokens_generated"),
        ("dragin_max",      "DRAGIN-RIND.max"),
        ("min_majority",    "AID min_majority"),
        ("M1_cabv",         "M1 CABV"),
        ("M9_cabv_x_laas",  "M9 CABV*LAAS"),
        ("F_max",           "F1 max(min_maj, M9)"),
        ("F_mean",          "F2 mean(min_maj, M9)"),
        ("F_geom",          "F3 geom(min_maj, M9)"),
    ]

    report_one(all_rows, "Phi + Qwen2.5 + Gemma pooled (R1, n=1200) — 3-arch", keys_to_test, B)
    report_one(rows_no_phi, "Qwen2.5 + Gemma pooled (R1, n=800) — Phi-DROPPED scenario", keys_to_test, B)
    report_one(phi, "Phi pooled (R1, n=400)", keys_to_test, B)
    report_one(qwen, "Qwen2.5 pooled (R1, n=400)", keys_to_test, B)
    report_one(gemma, "Gemma pooled (R1, n=400)", keys_to_test, B)

    per_cell_table(all_rows, keys_to_test)


if __name__ == "__main__":
    main()
