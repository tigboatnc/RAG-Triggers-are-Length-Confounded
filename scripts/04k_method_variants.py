"""Phase 4 / #22 follow-up — robustness checks + alternative OR-gate variants
(no compute, offline on existing JSONLs).

Pre-registered (in this docstring + the _AGENT.md #22 §variants section
that will be appended after this script lands its output) BEFORE the
test split is opened on any new gate.

Five new gates evaluated alongside #20's `latent` (OR over drift+variance):

  Robustness checks on the Tier 1 fused-LR gate
  ---------------------------------------------
  R1  latent_fused_balanced
      Same 5-feature L2-regularized LR as #22 Tier 1, but with
      `class_weight="balanced"` to address the ~62% correct-rate
      class imbalance. Threshold still iso-fire-rate 0.30.

  R2  latent_fused_f1opt
      Same 5-feature LR (un-balanced), but the threshold is set to
      the value that MAXIMIZES step-level F1 on the calibration
      split, rather than the iso-fire-rate 0.30 quantile.

  Alternative two-feature OR variants
  -----------------------------------
  V1  latent_or_drift_entropy
      OR of (drift_cosine, mean_token_entropy). Both thresholds set
      at iso-fire-rate 0.30 quantile on calibration. Motivated by
      the Qwen LR weights (mean_token_entropy = +0.31 strongest
      positive feature; variance_l2 = -0.35 wrong-signed).

  V2  latent_or_drift_hiddennorm
      OR of (drift_cosine, hidden_norm). Hidden_norm = +0.27 on
      Qwen, +0.11 on Gemma, near-zero on Phi.

  V3  latent_entropy_only
      Single-threshold gate on mean_token_entropy. Sanity check —
      if V1 wins, is it the entropy that's doing the work?

Reporting:
  - All five gates evaluated alongside the existing #20 pool
    (latent, etc, selfdoubt_t1, dragin, random).
  - Bootstrap recipe identical to #20: 10000-resample paired
    percentile, n_steps >= 3 eligibility filter.
  - Validation + test reported separately.

Usage:
    /Users/sanegi/miniconda3/envs/gzn/bin/python scripts/04k_method_variants.py --include-test
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import warnings
from importlib import import_module
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.linear_model import LogisticRegressionCV

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

sd = import_module("04g_sota_showdown")  # type: ignore

DEFAULT_MODELS = [
    "phi-4-mini-instruct",
    "gemma-4-26b-a4b-it",
    "qwen3.6-35b-a3b-instruct",
]
DEFAULT_FIRE_RATE = 0.30
DEFAULT_BOOTSTRAP = 10000
DEFAULT_MIN_STEPS = 3
DEFAULT_SEED = 20260511

GEOM_FEATURES = [
    "drift_cosine",
    "variance_l2_window",
    "hidden_norm",
    "mean_token_entropy",
    "boundary_token_entropy",
]


# ----------------------------------------------------------------------------
# Feature extraction
# ----------------------------------------------------------------------------

def _extract_step_matrix(traces: list[dict], feature_names: list[str]):
    rows: list[list[float]] = []
    labels: list[int] = []
    keys: list[tuple[int, int]] = []
    for ti, t in enumerate(traces):
        label = 0 if bool(t.get("label_correct_em", t.get("em", False))) else 1
        for si, s in enumerate(t.get("steps") or []):
            vec: list[float] = []
            skip = False
            for fname in feature_names:
                v = s.get(fname)
                if v is None:
                    skip = True
                    break
                vec.append(float(v))
            if skip:
                continue
            rows.append(vec)
            labels.append(label)
            keys.append((ti, si))
    if not rows:
        return np.zeros((0, len(feature_names))), np.zeros(0, dtype=int), []
    return np.asarray(rows, dtype=float), np.asarray(labels, dtype=int), keys


def _per_step_scalar(traces: list[dict], feature_name: str) -> dict[tuple[int, int], float]:
    out: dict[tuple[int, int], float] = {}
    for ti, t in enumerate(traces):
        for si, s in enumerate(t.get("steps") or []):
            v = s.get(feature_name)
            if v is not None:
                out[(ti, si)] = float(v)
    return out


# ----------------------------------------------------------------------------
# Classifier helpers (mirror 04j)
# ----------------------------------------------------------------------------

def _fit_lr(X: np.ndarray, y: np.ndarray, class_weight=None):
    mu = X.mean(axis=0)
    sd_ = X.std(axis=0); sd_[sd_ == 0] = 1.0
    Xs = (X - mu) / sd_
    if len(set(y.tolist())) < 2:
        return None, {"mu": mu.tolist(), "sd": sd_.tolist(), "degenerate": True}
    clf = LogisticRegressionCV(
        penalty="l2", cv=5, scoring="neg_log_loss", max_iter=2000, Cs=10,
        class_weight=class_weight,
    )
    clf.fit(Xs, y)
    return clf, {
        "mu": mu.tolist(), "sd": sd_.tolist(),
        "C_chosen": float(clf.C_[0]),
        "weights": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
        "class_weight": class_weight,
        "degenerate": False,
    }


def _proba(clf, X: np.ndarray, meta: dict) -> np.ndarray:
    if clf is None or meta.get("degenerate"):
        return np.full(len(X), 0.5, dtype=float)
    mu = np.asarray(meta["mu"], dtype=float)
    sd_ = np.asarray(meta["sd"], dtype=float); sd_[sd_ == 0] = 1.0
    Xs = (X - mu) / sd_
    return clf.predict_proba(Xs)[:, 1]


def _iso_fire_threshold(scores: np.ndarray, fire_rate: float) -> float:
    if len(scores) == 0:
        return 0.5
    s = np.sort(scores)
    k = (len(s) - 1) * (1.0 - fire_rate)
    lo = int(np.floor(k))
    hi = min(lo + 1, len(s) - 1)
    return float(s[lo] + (s[hi] - s[lo]) * (k - lo))


def _f1_optimal_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Pick the threshold (over unique score values) that maximises step-level F1
    with `fired = score > thr` and `positive = label==1` (incorrect)."""
    if len(scores) == 0:
        return 0.5
    candidates = np.unique(scores)
    if len(candidates) > 200:
        # Subsample candidates to keep this O(n) on large pools.
        candidates = np.quantile(candidates, np.linspace(0, 1, 200))
    best_f1, best_thr = -1.0, float(candidates[0])
    P = float(labels.sum())
    for thr in candidates:
        pred = scores > thr
        tp = float(np.sum(pred & (labels == 1)))
        fp = float(np.sum(pred & (labels == 0)))
        fn = P - tp
        if tp + fp == 0:
            continue
        prec = tp / (tp + fp)
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if prec + rec == 0:
            continue
        f1 = 2 * prec * rec / (prec + rec)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr


# ----------------------------------------------------------------------------
# Per-trace first-fire helpers
# ----------------------------------------------------------------------------

def _first_fire_score(traces: list[dict], scores_by_row: dict[tuple[int, int], float],
                      threshold: float, gate_name: str) -> list[dict]:
    out: list[dict] = []
    for ti, t in enumerate(traces):
        n_steps = int(t.get("n_steps", len(t.get("steps") or [])))
        first_fire = None
        for si in range(len(t.get("steps") or [])):
            p = scores_by_row.get((ti, si))
            if p is None:
                continue
            if p > threshold:
                first_fire = si + 1
                break
        out.append({
            "id": t.get("id"),
            "n_steps": n_steps,
            "label_correct": bool(t.get("label_correct_em", t.get("em", False))),
            "first_fire": {gate_name: first_fire},
        })
    return out


def _first_fire_or_two(traces: list[dict],
                       a_by_row: dict[tuple[int, int], float], tau_a: float,
                       b_by_row: dict[tuple[int, int], float], tau_b: float,
                       gate_name: str) -> list[dict]:
    out: list[dict] = []
    for ti, t in enumerate(traces):
        n_steps = int(t.get("n_steps", len(t.get("steps") or [])))
        first_fire = None
        for si in range(len(t.get("steps") or [])):
            a = a_by_row.get((ti, si))
            b = b_by_row.get((ti, si))
            af = a is not None and a > tau_a
            bf = b is not None and b > tau_b
            if af or bf:
                first_fire = si + 1
                break
        out.append({
            "id": t.get("id"),
            "n_steps": n_steps,
            "label_correct": bool(t.get("label_correct_em", t.get("em", False))),
            "first_fire": {gate_name: first_fire},
        })
    return out


def _merge_first_fire(into: list[dict], src: list[dict], key: str) -> list[dict]:
    by_id = {r["id"]: r for r in src}
    for r in into:
        srec = by_id.get(r["id"])
        r["first_fire"][key] = srec["first_fire"].get(key) if srec else None
    return into


# ----------------------------------------------------------------------------
# Trace loading
# ----------------------------------------------------------------------------

def _load_split(jpath: Path, min_steps: int):
    rows = []
    with jpath.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    calib = [r for r in rows if r.get("split") == "calibration" and int(r.get("n_steps", 0)) >= min_steps]
    val   = [r for r in rows if r.get("split") == "validation"  and int(r.get("n_steps", 0)) >= min_steps]
    test  = [r for r in rows if r.get("split") == "test"        and int(r.get("n_steps", 0)) >= min_steps]
    return calib, val, test


# ----------------------------------------------------------------------------
# Per-arch pipeline
# ----------------------------------------------------------------------------

NEW_GATES = [
    "latent_fused_balanced",
    "latent_fused_f1opt",
    "latent_or_drift_entropy",
    "latent_or_drift_hiddennorm",
    "latent_entropy_only",
]


def run_one_arch(model: str, args, rng: random.Random) -> dict:
    runs_dir = Path("runs")
    jpath = runs_dir / f"phase4_qa_{model}_{args.layer_mode}{args.suffix}.jsonl"
    if not jpath.exists():
        return {"error": f"missing {jpath}"}

    calib, val, test = _load_split(jpath, args.min_steps)
    if not calib or not val:
        return {"error": f"insufficient calibration ({len(calib)}) or validation ({len(val)})"}

    # ----- LR fits (R1 + R2) ----- ------------------------------------------
    Xc, yc, _ = _extract_step_matrix(calib, GEOM_FEATURES)

    # R1: class-balanced
    clf_bal, meta_bal = _fit_lr(Xc, yc, class_weight="balanced")
    pc_bal = _proba(clf_bal, Xc, meta_bal)
    thr_bal_iso = _iso_fire_threshold(pc_bal, args.target_fire_rate)

    # R2: unbalanced (same as #22), but F1-optimal threshold
    clf_unb, meta_unb = _fit_lr(Xc, yc, class_weight=None)
    pc_unb = _proba(clf_unb, Xc, meta_unb)
    thr_unb_f1 = _f1_optimal_threshold(pc_unb, yc)

    def _lr_firsts(clf, meta, threshold: float, traces: list[dict], gate_name: str) -> list[dict]:
        X_, _, k_ = _extract_step_matrix(traces, GEOM_FEATURES)
        if len(X_) == 0:
            return [{
                "id": t.get("id"),
                "n_steps": int(t.get("n_steps", 0)),
                "label_correct": bool(t.get("label_correct_em", t.get("em", False))),
                "first_fire": {gate_name: None},
            } for t in traces]
        probas = _proba(clf, X_, meta)
        by_row = {key: float(p) for key, p in zip(k_, probas)}
        return _first_fire_score(traces, by_row, threshold, gate_name)

    # ----- V1 / V2 / V3 (alternative OR variants) --------------------------
    drift_c = _per_step_scalar(calib, "drift_cosine")
    ent_c   = _per_step_scalar(calib, "mean_token_entropy")
    hnorm_c = _per_step_scalar(calib, "hidden_norm")

    drift_vals = np.asarray(list(drift_c.values()))
    ent_vals   = np.asarray(list(ent_c.values()))
    hnorm_vals = np.asarray(list(hnorm_c.values()))
    tau_drift  = _iso_fire_threshold(drift_vals, args.target_fire_rate)
    tau_ent    = _iso_fire_threshold(ent_vals,   args.target_fire_rate)
    tau_hnorm  = _iso_fire_threshold(hnorm_vals, args.target_fire_rate)

    # Frozen-from-#20 latent gate config + #20 random_p (for paired-margin parity)
    bootstrap_json = Path(args.bootstrap_json)
    with bootstrap_json.open() as f:
        boot20 = json.load(f)
    arch_rec20 = boot20["per_arch"][model]
    thr20 = arch_rec20["thresholds"]
    mode20 = arch_rec20["latent_mode_chosen"]
    rp = float(arch_rec20.get("random_p", args.target_fire_rate))

    # Build a per-split combined-records list, starting from the #20 pool.
    def _build_split_record(traces: list[dict]) -> list[dict]:
        pt = sd.firsts_per_trace(traces, thr20, mode20, rp, rng)

        # R1 / R2 (LR variants)
        pt = _merge_first_fire(pt, _lr_firsts(clf_bal, meta_bal, thr_bal_iso, traces, "latent_fused_balanced"),
                               "latent_fused_balanced")
        pt = _merge_first_fire(pt, _lr_firsts(clf_unb, meta_unb, thr_unb_f1, traces, "latent_fused_f1opt"),
                               "latent_fused_f1opt")

        # V1 — drift OR mean_token_entropy
        drift_t = _per_step_scalar(traces, "drift_cosine")
        ent_t   = _per_step_scalar(traces, "mean_token_entropy")
        hnorm_t = _per_step_scalar(traces, "hidden_norm")
        pt = _merge_first_fire(pt,
            _first_fire_or_two(traces, drift_t, tau_drift, ent_t, tau_ent, "latent_or_drift_entropy"),
            "latent_or_drift_entropy")
        # V2 — drift OR hidden_norm
        pt = _merge_first_fire(pt,
            _first_fire_or_two(traces, drift_t, tau_drift, hnorm_t, tau_hnorm, "latent_or_drift_hiddennorm"),
            "latent_or_drift_hiddennorm")
        # V3 — entropy alone
        pt = _merge_first_fire(pt,
            _first_fire_score(traces, ent_t, tau_ent, "latent_entropy_only"),
            "latent_entropy_only")
        return pt

    val_pt = _build_split_record(val)
    test_pt = _build_split_record(test) if args.include_test else None

    arch_out: dict = {
        "model": model,
        "n_calib": len(calib),
        "n_val": len(val),
        "n_test": len(test),
        "calibration_thresholds": {
            "tau_drift_iso30": tau_drift,
            "tau_mean_entropy_iso30": tau_ent,
            "tau_hidden_norm_iso30": tau_hnorm,
            "thr_fused_balanced_iso30": thr_bal_iso,
            "thr_fused_unb_f1opt": thr_unb_f1,
        },
        "fitted_lr": {
            "balanced": {**meta_bal, "thr_iso30": thr_bal_iso},
            "unbalanced_f1opt": {**meta_unb, "thr_f1opt": thr_unb_f1},
        },
        "frozen_from_20": {"mode": mode20, "thresholds": thr20, "random_p": rp},
        "validation": _eval_pool(val_pt, args.bootstrap, rng),
    }
    if test_pt is not None:
        arch_out["test"] = _eval_pool(test_pt, args.bootstrap, rng)
    return arch_out


EXTENDED_POOL = sd.POOL + NEW_GATES


def _eval_pool(pt: list[dict], B: int, rng: random.Random) -> dict:
    per_gate: dict[str, dict] = {}
    for g in EXTENDED_POOL:
        fired = sum(1 for t in pt if t["first_fire"].get(g) is not None)
        fire_rate = fired / len(pt) if pt else 0.0
        ci = sd.boot_ci(pt, g, B, rng)
        ci["trace_fire_rate"] = fire_rate
        per_gate[g] = ci
    margins: dict[str, dict] = {}
    for ng in NEW_GATES:
        for opp in ("latent", "dragin", "etc", "random"):
            margins[f"{ng}_minus_{opp}"] = sd.boot_margin_ci(pt, ng, opp, B, rng)
    return {"n": len(pt), "per_gate": per_gate, "margins": margins}


# ----------------------------------------------------------------------------
# Pretty-print
# ----------------------------------------------------------------------------

def _print_arch(model: str, rec: dict) -> None:
    if "error" in rec:
        print(f"[arch] {model}: ERROR {rec['error']}")
        return
    print(f"[arch] {model}")
    print(f"  n_calib={rec['n_calib']}  n_val={rec['n_val']}  n_test={rec['n_test']}")
    ct = rec["calibration_thresholds"]
    print(f"  tau_drift_iso30          = {ct['tau_drift_iso30']:.4f}")
    print(f"  tau_mean_entropy_iso30   = {ct['tau_mean_entropy_iso30']:.4f}")
    print(f"  tau_hidden_norm_iso30    = {ct['tau_hidden_norm_iso30']:.4f}")
    print(f"  thr_fused_balanced_iso30 = {ct['thr_fused_balanced_iso30']:.4f}")
    print(f"  thr_fused_unb_f1opt      = {ct['thr_fused_unb_f1opt']:.4f}")
    bm = rec["fitted_lr"]["balanced"]
    if not bm.get("degenerate"):
        print(f"  balanced LR weights (after standardisation):")
        for fname, w in zip(GEOM_FEATURES, bm["weights"]):
            print(f"    {fname:24} = {w:+.4f}")
        print(f"  balanced C_chosen = {bm['C_chosen']:.4g}")
    print()
    for label in ("validation", "test"):
        if label not in rec:
            continue
        block = rec[label]
        print(f"  --- {label.upper()} (n={block['n']}) ---")
        print(f"    {'gate':32}  fire%   point      95% CI                 sig")
        for g in EXTENDED_POOL:
            ci = block["per_gate"].get(g, {})
            point = ci.get("point"); lo = ci.get("ci_lo"); hi = ci.get("ci_hi")
            fr = ci.get("trace_fire_rate", 0.0)
            print(f"    {g:32}  {fr*100:5.1f}  {sd._fmt(point)}   [{sd._fmt(lo)}, {sd._fmt(hi)}]   {sd._sig(lo, hi)}")
        print(f"    --- key paired margins (new − baseline) ---")
        for ng in NEW_GATES:
            for opp in ("latent", "dragin", "etc", "random"):
                mci = block["margins"].get(f"{ng}_minus_{opp}", {})
                pgt = mci.get("p_gt_zero")
                p_str = "n/a" if pgt is None else f"P(>0)={pgt:.3f}"
                print(f"    {ng}-{opp:<13}  {sd._fmt(mci.get('point'))}   "
                      f"[{sd._fmt(mci.get('ci_lo'))}, {sd._fmt(mci.get('ci_hi'))}]   "
                      f"{p_str}   {sd._sig(mci.get('ci_lo'), mci.get('ci_hi'))}")
        print()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--layer-mode", default="mid_stack")
    p.add_argument("--suffix", default="_dragin")
    p.add_argument("--target-fire-rate", type=float, default=DEFAULT_FIRE_RATE)
    p.add_argument("--bootstrap", "-B", type=int, default=DEFAULT_BOOTSTRAP)
    p.add_argument("--min-steps", type=int, default=DEFAULT_MIN_STEPS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--include-test", action="store_true")
    p.add_argument("--bootstrap-json", default="runs/phase4_bootstrap_dragin.json")
    p.add_argument("--out", default="runs/phase4_method_variants.json")
    args = p.parse_args()

    rng = random.Random(args.seed)
    result: dict = {"args": vars(args), "per_arch": {}}

    print(f"[..] #22 follow-up: robustness checks + alternative OR variants")
    print(f"     new gates: {NEW_GATES}")
    print(f"     target_fire_rate={args.target_fire_rate}  B={args.bootstrap}  "
          f"min_steps={args.min_steps}  seed={args.seed}  include_test={args.include_test}")
    print()

    for model in args.models:
        rec = run_one_arch(model, args, rng)
        result["per_arch"][model] = rec
        _print_arch(model, rec)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[ok] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
