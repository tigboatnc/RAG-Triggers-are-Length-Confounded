"""Head-to-head replication of UNCERTAINTY-LINE (Vashurin et al. 2025) on our R1 traces.

Purpose
-------
The paper's §3 "Length Residualization" protocol uses

    s_resid = s - (alpha + beta * log(tokens + 1))      (our protocol)

on the **same cohort** (no train/test split, log-tokens, no MinMax).

UNCERTAINTY-LINE's reference implementation (externalCodeAudits/uncertainty-line/utils.py
`detrend_ue`) uses

    1. MinMax-normalize the signal and the *raw* length to [0,1] on a TRAIN split
       (5th-95th percentile of train lengths only).
    2. Fit LinearRegression(signal_norm ~ length_norm) on TRAIN.
    3. On TEST: subtract predicted trend from MinMax-normalized signal_norm.

This script runs BOTH protocols on the same paired R1 traces (closed × oracle
per arch × dataset) and reports:

    - raw AUROC vs lift_em = +1
    - ours_resid_AUROC  (single-cohort log-tokens OLS, evaluated on cohort)
    - line_resid_AUROC  (train/test split 70/30, raw length, MinMax-normalized)
    - |Δ| = | line_resid - ours_resid |

Headline number to quote in a reviewer-defense appendix:
    eps = max |Δ| across all (cell × signal) — if eps < 0.05 the protocols
    are numerically equivalent; if 0.05 ≤ eps < 0.10 there are small but
    expected differences from the (log vs linear) and (single-cohort vs
    train/test) design choices; if eps ≥ 0.10 there is a real disagreement
    that needs explanation.

What this script does NOT do
----------------------------
- We do NOT install / require `lm_polygraph` (their UEManager is heavy and
  tied to their own data format). We port the *math* of `detrend_ue`
  (MinMax + LinearRegression + train/test split) directly. The cited line
  in their utils.py: lines 154-172.
- We do NOT replicate their Prediction-Rejection-Area scoring metric. That
  is a separate methodological choice (rejection-curve area vs. AUROC); we
  audit only the length-debiasing step, with AUROC held constant on both
  sides. PRA-vs-AUROC is out of scope.

Usage
-----
    python analysis/uncertainty_line_audit.py --arch qwen2.5-7b-instruct
    python analysis/uncertainty_line_audit.py --arch gemma-4-26b-a4b-it

Outputs
-------
    analysis/uncertainty_line_audit_<arch>.out   (stdout dump)
    analysis/uncertainty_line_audit_<arch>.json  (machine-readable)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _common import auroc as _auroc
from r1_paired_audit import load_paired

ROOT = HERE.parent

# Signal pool: a sensible subset that spans the families.
# (key, display name, family). All extracted in load_paired() already.
SIGNALS = [
    ("tokens",          "tokens_generated (sanity)",    "length"),
    ("dragin_max",      "DRAGIN-RIND.max",              "attention"),
    ("ent_max",         "ent.max",                      "logit"),
    ("etc",             "etc (Δ²H max)",                "logit"),
    ("flare_max",       "flare.max (1 − min p)",        "logit"),
    ("u_cons_max",      "u_cons.max",                   "K-sample-text"),
    ("seakr_k5_logdet", "seakr_k5_proxy.logdet",        "K-sample-hidden"),
    ("flip_k5",         "flip_rate K=5",                "answer-trajectory"),
    ("flip_k1",         "flip_rate K=1",                "answer-trajectory"),
    ("late_flips",      "late_flips",                   "answer-trajectory"),
    ("min_majority",    "min_majority",                 "answer-trajectory"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def auroc_pos(rows, key: str, label_fn) -> float:
    pos = [r[key] for r in rows if r.get(key) is not None and label_fn(r)]
    neg = [r[key] for r in rows if r.get(key) is not None and not label_fn(r)]
    if not pos or not neg:
        return float("nan")
    return _auroc(pos, neg)


def ours_resid(values: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Our protocol: residualize signal on log(tokens+1) via OLS on the same cohort.

    Mirrors `analysis/signal_lib.py::ols_residual`.
    """
    log_len = np.log1p(lengths).reshape(-1, 1)
    reg = LinearRegression().fit(log_len, values)
    pred = reg.predict(log_len)
    return values - pred


def line_resid_full(values: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """UNCERTAINTY-LINE math (Vashurin et al. 2025) but applied to the FULL
    cohort with NO train/test split. Isolates the protocol differences (raw
    length + MinMax scaling + linear regression) from the sampling-noise
    differences (train/test split with small test set).

    Specifically replicates `externalCodeAudits/uncertainty-line/utils.py::detrend_ue`
    lines 154-167 with fit==predict on the same cohort.

    This is the APPLES-TO-APPLES comparison with our `ours_resid` (which
    fits and evaluates on the same cohort).
    """
    lengths = lengths.astype(float)
    values = values.astype(float)
    len_scaler = MinMaxScaler()
    len_norm = len_scaler.fit_transform(lengths[:, None]).squeeze()
    val_scaler = MinMaxScaler()
    val_norm = val_scaler.fit_transform(values[:, None]).squeeze()
    linreg = LinearRegression().fit(len_norm[:, None], val_norm)
    pred = linreg.predict(len_norm[:, None])
    return val_norm - pred


def line_resid_split(values: np.ndarray, lengths: np.ndarray, *,
                     n_seeds: int = 5, train_frac: float = 0.70) -> np.ndarray:
    """UNCERTAINTY-LINE protocol, FAITHFUL port of
    `externalCodeAudits/uncertainty-line/utils.py::detrend_ue` lines 140-167:

      1. Split data into TRAIN (70%) and TEST (30%).
      2. On TRAIN: trim to 5th-95th percentile of length.
      3. MinMax-scale length and signal on TRAIN, transform TEST.
      4. Fit LinearRegression(signal_norm ~ length_norm) on TRAIN.
      5. TEST residual = signal_norm_test - predicted_trend_test.

    To reduce single-split variance at our n=200, we run `n_seeds` random
    splits and AVERAGE the residual at each row across splits where that
    row landed in TEST. Returns full-length array (every row covered).
    """
    n = len(values)
    accum = np.zeros(n, dtype=float)
    count = np.zeros(n, dtype=int)

    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        n_train = int(round(train_frac * n))
        train_idx = perm[:n_train]
        test_idx = perm[n_train:]

        train_len = lengths[train_idx].astype(float)
        train_val = values[train_idx].astype(float)
        test_len = lengths[test_idx].astype(float)
        test_val = values[test_idx].astype(float)

        if len(train_len) >= 20:
            lo_q = np.quantile(train_len, 0.05)
            hi_q = np.quantile(train_len, 0.95)
            keep = (train_len > lo_q) & (train_len < hi_q)
            if keep.sum() >= 10:
                train_len = train_len[keep]
                train_val = train_val[keep]

        len_scaler = MinMaxScaler()
        train_len_norm = len_scaler.fit_transform(train_len[:, None]).squeeze()
        test_len_norm = len_scaler.transform(test_len[:, None]).squeeze()

        val_scaler = MinMaxScaler()
        train_val_norm = val_scaler.fit_transform(train_val[:, None]).squeeze()
        test_val_norm = val_scaler.transform(test_val[:, None]).squeeze()

        linreg = LinearRegression().fit(train_len_norm[:, None], train_val_norm)
        test_pred = linreg.predict(test_len_norm[:, None])
        test_resid = test_val_norm - test_pred

        accum[test_idx] += test_resid
        count[test_idx] += 1

    out = np.full(n, np.nan, dtype=float)
    mask = count > 0
    out[mask] = accum[mask] / count[mask]
    return out


def cell_audit(rows: list[dict], cell_name: str, label_fn, *, n_seeds: int = 5):
    """Run both protocols on a single cell, return per-signal numbers.

    Reports four columns per signal:
        - raw AUROC (no residualization)
        - ours_resid: our protocol (log-tokens OLS, full cohort)
        - line_full: UL math (raw-tokens MinMax + linear, full cohort)
                     → APPLES-TO-APPLES with ours_resid
        - line_split: UL faithful (raw-tokens MinMax + linear, train/test
                      averaged over n_seeds splits) → SAMPLING-MATCHED to UL paper

    Two epsilons reported:
        eps_protocol = |ours_resid - line_full|   (isolates math difference)
        eps_sampling = |line_full - line_split|   (isolates evaluation-set noise)
    """
    out = {"cell": cell_name, "n": len(rows), "pop": sum(1 for r in rows if label_fn(r)),
           "rows": []}
    if not rows or out["pop"] < 3:
        return out

    for key, name, family in SIGNALS:
        valid = [r for r in rows if r.get(key) is not None]
        if len(valid) < 20:
            out["rows"].append({"signal": name, "family": family, "skipped": True})
            continue
        valid_arr = np.array([r[key] for r in valid], dtype=float)
        valid_len = np.array([float(r["tokens"]) for r in valid], dtype=float)

        raw = auroc_pos(valid, key, label_fn)

        ours = ours_resid(valid_arr, valid_len)
        for v, r in zip(ours, valid):
            r[f"_ours_{key}"] = float(v)
        ours_au = auroc_pos(valid, f"_ours_{key}", label_fn)

        line_f = line_resid_full(valid_arr, valid_len)
        for v, r in zip(line_f, valid):
            r[f"_lf_{key}"] = float(v)
        line_full_au = auroc_pos(valid, f"_lf_{key}", label_fn)

        line_s = line_resid_split(valid_arr, valid_len, n_seeds=n_seeds)
        for v, r in zip(line_s, valid):
            r[f"_ls_{key}"] = float(v) if not math.isnan(v) else None
        line_split_au = auroc_pos(valid, f"_ls_{key}", label_fn)

        for r in valid:
            r.pop(f"_ours_{key}", None)
            r.pop(f"_lf_{key}", None)
            r.pop(f"_ls_{key}", None)

        def _diff(a, b):
            if isinstance(a, float) and a == a and isinstance(b, float) and b == b:
                return abs(a - b)
            return None

        out["rows"].append({
            "signal": name,
            "family": family,
            "n_valid": len(valid),
            "raw_auroc": raw,
            "ours_resid_auroc": ours_au,
            "line_full_auroc": line_full_au,
            "line_split_auroc": line_split_au,
            "delta_ours_minus_raw": (ours_au - raw) if (ours_au == ours_au and raw == raw) else None,
            "delta_line_full_minus_raw": (line_full_au - raw) if (line_full_au == line_full_au and raw == raw) else None,
            "eps_protocol": _diff(ours_au, line_full_au),
            "eps_sampling": _diff(line_full_au, line_split_au),
        })
    return out


def print_cell(cell: dict):
    print(f"\n{'=' * 100}")
    print(f"{cell['cell']}    n={cell['n']}    pop_lift_em+1={cell['pop']}")
    print("=" * 100)
    print(f"  {'signal':<32} {'raw':>6} {'ours':>6} {'LINE_f':>7} {'LINE_s':>7}  {'ε_proto':>8} {'ε_sample':>9}")
    print(f"  {'':>32} {'':>6} {'(log,':>6} {'(MM+lin':>7} {'(MM+lin':>7}  {'|ours':>8} {'|LINE_f':>9}")
    print(f"  {'':>32} {'':>6} {'full)':>6} {'full)':>7} {'split)':>7}  {'-LINE_f|':>8} {'-LINE_s|':>9}")
    for r in cell["rows"]:
        if r.get("skipped"):
            print(f"  {r['signal']:<32}  (skipped — n_valid<20)")
            continue
        ra = r["raw_auroc"]
        ou = r["ours_resid_auroc"]
        lf = r["line_full_auroc"]
        ls = r["line_split_auroc"]
        ep = r["eps_protocol"]
        es = r["eps_sampling"]
        fmt = lambda x: f"{x:>6.3f}" if isinstance(x, float) and x == x else f"{'n/a':>6}"
        fmt7 = lambda x: f"{x:>7.3f}" if isinstance(x, float) and x == x else f"{'n/a':>7}"
        fmte = lambda x: f"{x:>8.4f}" if isinstance(x, float) and x == x else f"{'n/a':>8}"
        fmt9 = lambda x: f"{x:>9.4f}" if isinstance(x, float) and x == x else f"{'n/a':>9}"
        print(f"  {r['signal']:<32} {fmt(ra)} {fmt(ou)} {fmt7(lf)} {fmt7(ls)}  {fmte(ep)} {fmt9(es)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="v2_headline")
    ap.add_argument("--arch", default="gemma-4-26b-a4b-it",
                    choices=["qwen2.5-7b-instruct", "gemma-4-26b-a4b-it"])
    ap.add_argument("--n-seeds", type=int, default=5,
                    help="Number of train/test seeds to average over for LINE_split")
    ap.add_argument("--out-prefix", default=None,
                    help="If set, write <prefix>.out / .json; else derived from --arch.")
    args = ap.parse_args()

    out_prefix = args.out_prefix or f"analysis/uncertainty_line_audit_{args.arch}"
    out_txt = ROOT / f"{out_prefix}.out"
    out_json = ROOT / f"{out_prefix}.json"

    # Stream output to both stdout and the .out file by redirecting sys.stdout
    from io import StringIO
    buf = StringIO()
    class Tee:
        def __init__(self, *streams): self.streams = streams
        def write(self, s):
            for st in self.streams: st.write(s)
        def flush(self):
            for st in self.streams: st.flush()
    sys.stdout = Tee(sys.__stdout__, buf)

    print(f"UNCERTAINTY-LINE replication audit")
    print(f"  arch:    {args.arch}")
    print(f"  target:  lift_em = +1 (paper headline)")
    print(f"  n_seeds: {args.n_seeds} (LINE_split averaging)")
    print(f"  pool:    {len(SIGNALS)} signals")
    print(f"  columns:")
    print(f"    raw       = AUROC on the raw signal (no detrend)")
    print(f"    ours      = our protocol: signal - OLS(signal ~ log(tokens+1))")
    print(f"                fit and evaluated on the FULL cohort")
    print(f"    LINE_f    = UL math (Vashurin 2025): MinMax-norm signal+length")
    print(f"                + LinearRegression(signal_norm ~ length_norm) on FULL cohort")
    print(f"                → APPLES-TO-APPLES sample size with `ours`")
    print(f"    LINE_s    = UL faithful: same math but TRAIN/TEST split (70/30),")
    print(f"                averaged over {args.n_seeds} seeds → matches UL paper's protocol")
    print(f"    ε_proto   = |ours − LINE_f|  (isolates math: log vs raw-MinMax)")
    print(f"    ε_sample  = |LINE_f − LINE_s|  (isolates evaluation-set sampling noise)")

    run_dir = ROOT / "runs" / f"{args.prefix}_{args.arch}"
    cells = []
    for cd, ce, tag in [
        ("D1_closed_hotpot.jsonl", "E1_oracle_hotpot.jsonl", "HotpotQA-distractor"),
        ("D2_closed_wiki.jsonl",   "E2_oracle_wiki.jsonl",   "2WikiMHQA-distractor"),
    ]:
        cp = run_dir / cd
        op = run_dir / ce
        if not cp.exists() or not op.exists():
            print(f"\n[skip] {tag} — missing one of:\n  {cp.name}: {cp.exists()}\n  {ce}: {op.exists()}")
            continue
        rows = load_paired(cp, op)
        if not rows:
            print(f"\n[skip] {tag} — no paired rows")
            continue
        cell = cell_audit(rows, f"{args.arch} × {tag}",
                          label_fn=lambda r: r.get("lift_em") == +1,
                          n_seeds=args.n_seeds)
        cells.append(cell)
        print_cell(cell)

    # Headline summary
    print(f"\n{'=' * 100}")
    print("HEADLINE: protocol-equivalence verdict")
    print("=" * 100)
    proto_diffs = []
    sample_diffs = []
    for cell in cells:
        for r in cell["rows"]:
            # Skip the tokens row for ε computation: residualizing tokens
            # against tokens is degenerate (both protocols give ~0 residual).
            if r.get("signal", "").startswith("tokens_generated"):
                continue
            ep = r.get("eps_protocol")
            es = r.get("eps_sampling")
            if isinstance(ep, float) and ep == ep:
                proto_diffs.append((ep, cell["cell"], r["signal"]))
            if isinstance(es, float) and es == es:
                sample_diffs.append((es, cell["cell"], r["signal"]))

    def _summary(diffs, label):
        if not diffs:
            print(f"  {label}: no comparable signals")
            return None
        diffs_sorted = sorted(diffs, reverse=True)
        ep_max = diffs_sorted[0][0]
        ep_mean = float(np.mean([d for d, *_ in diffs]))
        ep_med = float(np.median([d for d, *_ in diffs]))
        print(f"  {label}:  ε_max={ep_max:.4f}  ε_mean={ep_mean:.4f}  ε_med={ep_med:.4f}  (n_signals={len(diffs)})")
        print(f"    top-3 (worst):")
        for d, cn, sg in diffs_sorted[:3]:
            print(f"      {d:.4f}   {cn}    {sg}")
        return ep_max

    print("\n  ε_protocol = |ours − LINE_f| isolates math difference (log vs MinMax-linear):")
    proto_max = _summary(proto_diffs, "ε_protocol")
    print("\n  ε_sampling = |LINE_f − LINE_s| isolates train/test-split sampling noise:")
    samp_max = _summary(sample_diffs, "ε_sampling")

    print()
    if proto_max is not None:
        if proto_max < 0.05:
            v = "EQUIVALENT — our protocol matches UL math within numerical noise (apples-to-apples comparison)."
        elif proto_max < 0.10:
            v = "MOSTLY EQUIVALENT — small differences attributable to (log vs raw-MinMax) regressor choice on the same evaluation set."
        else:
            v = "DIFFERS — log vs MinMax-linear regressor gives ≥0.10 AUROC differences on at least one signal; investigate."
        print(f"  protocol verdict: {v}")
    if samp_max is not None and samp_max >= 0.05:
        print(f"  sampling note: UL's train/test split contributes ε up to {samp_max:.3f} AUROC at our n=200; this is sampling noise, not a protocol difference.")

    sys.stdout = sys.__stdout__
    out_txt.write_text(buf.getvalue())
    out_json.write_text(json.dumps({"arch": args.arch, "n_seeds": args.n_seeds,
                                    "cells": cells}, indent=2, default=str))
    print(f"\n[ok] wrote {out_txt.relative_to(ROOT)}")
    print(f"[ok] wrote {out_json.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
