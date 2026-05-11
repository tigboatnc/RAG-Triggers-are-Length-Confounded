"""Phase 4 / #22 — multi-feature fusion gate (Tier 1, offline, no compute).

Re-uses the per-step scalars already in `runs/phase4_qa_*_mid_stack_dragin.jsonl`
and fits a per-architecture L2-regularised logistic regression on the
calibration split, using `label_correct_em == False` as the target. The
trained classifier emits a per-step P(incorrect); we threshold this at the
iso-fire-rate 0.30 quantile of the calibration distribution to produce a
new gate `latent_fused` (and a `latent_fused_plus` variant that also uses
HVR as a 6th feature).

Pre-registered in `_AGENT.md` #22 before the test split is opened. The
new gates enter the comparison pool alongside the existing
`{latent, etc, selfdoubt_t1, dragin, random}` — *not* as replacements.
The current OR gate (`latent` per #20) remains the canonical baseline
headline; this script tests whether richer fusion does materially
better on each architecture.

Discipline (L5-clean):
  - Fitting + cross-validation + quantile-threshold all happen on
    calibration rows only.
  - 2-of-3 consecutive-step variant is computed from the same calibrated
    probability series; no additional fitting.
  - Z-normalisation variant uses per-trace mean/std computed on the
    calibration split's feature distribution alone for fitting; per-
    trace mean/std on val/test rows is computed *within each trace*
    (no leakage across rows).
  - Validation + test reported separately; bootstrap recipe identical
    to #20.

Usage:
    /Users/sanegi/miniconda3/envs/gzn/bin/python scripts/04j_fused_gate.py
    /Users/sanegi/miniconda3/envs/gzn/bin/python scripts/04j_fused_gate.py --include-test --bootstrap 10000
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegressionCV

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Re-use bootstrap primitives from #20 so the recipe is identical.
sd = import_module("04g_sota_showdown")  # type: ignore

DEFAULT_MODELS = [
    "phi-4-mini-instruct",
    "gemma-4-26b-a4b-it",
    "qwen3.6-35b-a3b-instruct",
]

# Variant 1 (geometry-only): 5 features from CRV + forward-pass logits.
GEOM_FEATURES = [
    "drift_cosine",
    "variance_l2_window",
    "hidden_norm",
    "mean_token_entropy",
    "boundary_token_entropy",
]
# Variant 2 (geometry + text): adds HVR as a 6th feature.
GEOM_PLUS_FEATURES = GEOM_FEATURES + ["hvr"]

DEFAULT_FIRE_RATE = 0.30
DEFAULT_BOOTSTRAP = 10000
DEFAULT_MIN_STEPS = 3
DEFAULT_SEED = 20260511
DEFAULT_TWO_OF_THREE = True


# ----------------------------------------------------------------------------
# Feature extraction (offline, from existing JSONL)
# ----------------------------------------------------------------------------

def _extract_step_matrix(traces: list[dict], feature_names: list[str]) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Returns X (N_steps, n_features), y (N_steps,), and a row->(trace_idx, step_idx) map.
    Skips steps with any None in the requested feature set."""
    rows: list[list[float]] = []
    labels: list[int] = []
    keys: list[tuple[int, int]] = []
    for ti, t in enumerate(traces):
        # Trace-level label: 1 if INCORRECT (the "fire-earlier-on-incorrect" target).
        label = 0 if bool(t.get("label_correct_em", t.get("em", False))) else 1
        steps = t.get("steps") or []
        for si, s in enumerate(steps):
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


def _fit_classifier(X: np.ndarray, y: np.ndarray) -> tuple[LogisticRegressionCV, dict]:
    """Fit L2-regularised logistic with 5-fold CV on the regularization C.
    Standardise features internally (zero mean / unit variance) to keep
    the L2 penalty comparable across heterogeneous scales (drift in [0,2]
    vs hidden_norm in [10, 50] vs entropy in [0, 5])."""
    mu = X.mean(axis=0)
    sd_ = X.std(axis=0)
    sd_[sd_ == 0] = 1.0
    Xs = (X - mu) / sd_
    # If only one class present (all correct OR all incorrect), CV will fail;
    # bail out with a degenerate classifier (uniform predictions).
    if len(set(y.tolist())) < 2:
        return None, {"mu": mu.tolist(), "sd": sd_.tolist(), "degenerate": True}
    clf = LogisticRegressionCV(
        penalty="l2",
        cv=5,
        scoring="neg_log_loss",
        max_iter=2000,
        Cs=10,  # geometric grid from 1e-4 to 1e4
    )
    clf.fit(Xs, y)
    meta = {
        "mu": mu.tolist(),
        "sd": sd_.tolist(),
        "C_chosen": float(clf.C_[0]),
        "weights_after_standardisation": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
        "calib_pos_rate": float(y.mean()),
        "n_calib_steps": int(len(y)),
        "degenerate": False,
    }
    return clf, meta


def _predict_proba(clf, X: np.ndarray, meta: dict) -> np.ndarray:
    if clf is None or meta.get("degenerate"):
        return np.full(len(X), 0.5, dtype=float)
    mu = np.asarray(meta["mu"], dtype=float)
    sd_ = np.asarray(meta["sd"], dtype=float)
    sd_[sd_ == 0] = 1.0
    Xs = (X - mu) / sd_
    return clf.predict_proba(Xs)[:, 1]


def _first_fire_fused(traces: list[dict], probas_by_row: dict[tuple[int, int], float],
                      threshold: float, two_of_three: bool = False) -> list[dict]:
    """For each trace, find the first step where the fused probability
    exceeds `threshold`. If two_of_three, fire only at the first step where
    >=2 of the last 3 step scores exceed threshold."""
    out: list[dict] = []
    for ti, t in enumerate(traces):
        n_steps = int(t.get("n_steps", len(t.get("steps") or [])))
        # Collect per-step probas in order, filling missing steps with NaN.
        steps = t.get("steps") or []
        scores: list[float | None] = []
        for si in range(len(steps)):
            scores.append(probas_by_row.get((ti, si)))
        first_fire: int | None = None
        if two_of_three:
            recent: list[bool] = []
            for i, p in enumerate(scores):
                if p is None:
                    recent.append(False)
                else:
                    recent.append(p > threshold)
                if len(recent) > 3:
                    recent.pop(0)
                if len(recent) == 3 and sum(recent) >= 2:
                    first_fire = i + 1
                    break
        else:
            for i, p in enumerate(scores):
                if p is None:
                    continue
                if p > threshold:
                    first_fire = i + 1
                    break
        out.append({
            "id": t.get("id"),
            "n_steps": n_steps,
            "label_correct": bool(t.get("label_correct_em", t.get("em", False))),
            "first_fire": {"latent_fused": first_fire},
        })
    return out


def _quantile_threshold_from_probas(probas: np.ndarray, target_fire_rate: float) -> float:
    """Step-level threshold so P(p > thr | calibration step) ≈ target."""
    if len(probas) == 0:
        return 0.5
    s = np.sort(probas)
    k = (len(s) - 1) * (1.0 - target_fire_rate)
    lo = int(np.floor(k))
    hi = min(lo + 1, len(s) - 1)
    return float(s[lo] + (s[hi] - s[lo]) * (k - lo))


# ----------------------------------------------------------------------------
# Bootstrap reusing #20 primitives
# ----------------------------------------------------------------------------

def boot_ci_simple(per_trace: list[dict], gate: str, B: int, rng: random.Random) -> dict:
    return sd.boot_ci(per_trace, gate, B, rng)


def boot_margin_simple(per_trace: list[dict], a: str, b: str, B: int, rng: random.Random) -> dict:
    return sd.boot_margin_ci(per_trace, a, b, B, rng)


# ----------------------------------------------------------------------------
# Merge in the existing pool first-fires (so margins to ETC / DRAGIN / etc.
# can be computed without rerunning #20). We do this by piggy-backing on
# #20's `firsts_per_trace` exactly.
# ----------------------------------------------------------------------------

def _eligible_traces(jpath: Path, min_steps: int) -> tuple[list[dict], list[dict], list[dict]]:
    """Returns (calib, val, test) lists, each filtered to n_steps >= min_steps."""
    rows: list[dict] = []
    with jpath.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    calib = [r for r in rows if r.get("split") == "calibration" and int(r.get("n_steps", 0)) >= min_steps]
    val   = [r for r in rows if r.get("split") == "validation"  and int(r.get("n_steps", 0)) >= min_steps]
    test  = [r for r in rows if r.get("split") == "test"        and int(r.get("n_steps", 0)) >= min_steps]
    return calib, val, test


def _eligible_traces_unfiltered(jpath: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """Same but no min_steps filter, for the existing-pool firsts_per_trace."""
    rows: list[dict] = []
    with jpath.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    calib = [r for r in rows if r.get("split") == "calibration"]
    val   = [r for r in rows if r.get("split") == "validation"]
    test  = [r for r in rows if r.get("split") == "test"]
    return calib, val, test


# ----------------------------------------------------------------------------
# Combined per-trace record (one row per trace, with first_fire for every gate)
# ----------------------------------------------------------------------------

def _combine_first_fires(
    fused_pt: list[dict],
    existing_pt: list[dict],
) -> list[dict]:
    """Merge the fused first_fire dicts into the #20-pool first_fire dicts
    by trace id, so paired margins can be computed across the union pool.
    Both lists are expected to be in the same order; we still match by id
    defensively in case eligibility filtering diverged.
    """
    by_id = {r["id"]: r for r in fused_pt}
    out: list[dict] = []
    for r in existing_pt:
        f = by_id.get(r["id"])
        if f is None:
            # Step features missing -> fused gate didn't fire
            r["first_fire"]["latent_fused"] = None
            r["first_fire"]["latent_fused_2of3"] = None
        else:
            r["first_fire"]["latent_fused"] = f["first_fire"].get("latent_fused")
            r["first_fire"]["latent_fused_2of3"] = f["first_fire"].get("latent_fused_2of3")
        out.append(r)
    return out


# ----------------------------------------------------------------------------
# Main per-arch pipeline
# ----------------------------------------------------------------------------

def run_one_arch(model: str, args: argparse.Namespace, rng: random.Random) -> dict:
    runs_dir = Path("runs")
    jpath = runs_dir / f"phase4_qa_{model}_{args.layer_mode}{args.suffix}.jsonl"
    if not jpath.exists():
        return {"error": f"missing {jpath}"}

    calib, val, test = _eligible_traces(jpath, args.min_steps)
    if not calib or not val:
        return {"error": f"insufficient calibration ({len(calib)}) or validation ({len(val)}) traces"}

    # Variant 1: geometry-only.
    Xc, yc, kc = _extract_step_matrix(calib, GEOM_FEATURES)
    clf, meta = _fit_classifier(Xc, yc)

    # Predict on calibration to set the iso-fire-rate threshold.
    pc = _predict_proba(clf, Xc, meta) if len(Xc) else np.zeros(0)
    threshold = _quantile_threshold_from_probas(pc, args.target_fire_rate)

    def _per_arch_fused_firsts(traces: list[dict], two_of_three: bool) -> list[dict]:
        X_, y_, k_ = _extract_step_matrix(traces, GEOM_FEATURES)
        if len(X_) == 0:
            return [{"id": t.get("id"), "n_steps": int(t.get("n_steps", 0)),
                     "label_correct": bool(t.get("label_correct_em", t.get("em", False))),
                     "first_fire": {"latent_fused": None}}
                    for t in traces]
        probas = _predict_proba(clf, X_, meta)
        probas_by_row: dict[tuple[int, int], float] = {key: float(p) for key, p in zip(k_, probas)}
        return _first_fire_fused(traces, probas_by_row, threshold, two_of_three=two_of_three)

    fused_val_pt = _per_arch_fused_firsts(val, two_of_three=False)
    fused_test_pt = _per_arch_fused_firsts(test, two_of_three=False) if args.include_test else None
    fused_val_pt_2 = _per_arch_fused_firsts(val, two_of_three=True)
    fused_test_pt_2 = _per_arch_fused_firsts(test, two_of_three=True) if args.include_test else None

    # Also fit a geometry-plus-text variant for comparison.
    Xc_p, yc_p, kc_p = _extract_step_matrix(calib, GEOM_PLUS_FEATURES)
    clf_p, meta_p = _fit_classifier(Xc_p, yc_p)
    pc_p = _predict_proba(clf_p, Xc_p, meta_p) if len(Xc_p) else np.zeros(0)
    threshold_p = _quantile_threshold_from_probas(pc_p, args.target_fire_rate)

    def _per_arch_fused_plus_firsts(traces: list[dict]) -> list[dict]:
        X_, y_, k_ = _extract_step_matrix(traces, GEOM_PLUS_FEATURES)
        if len(X_) == 0:
            return [{"id": t.get("id"), "n_steps": int(t.get("n_steps", 0)),
                     "label_correct": bool(t.get("label_correct_em", t.get("em", False))),
                     "first_fire": {"latent_fused_plus": None}}
                    for t in traces]
        probas = _predict_proba(clf_p, X_, meta_p)
        probas_by_row: dict[tuple[int, int], float] = {key: float(p) for key, p in zip(k_, probas)}
        out: list[dict] = []
        for ti, t in enumerate(traces):
            n_steps = int(t.get("n_steps", len(t.get("steps") or [])))
            steps = t.get("steps") or []
            first_fire = None
            for si in range(len(steps)):
                p = probas_by_row.get((ti, si))
                if p is None:
                    continue
                if p > threshold_p:
                    first_fire = si + 1
                    break
            out.append({
                "id": t.get("id"),
                "n_steps": n_steps,
                "label_correct": bool(t.get("label_correct_em", t.get("em", False))),
                "first_fire": {"latent_fused_plus": first_fire},
            })
        return out

    fused_plus_val_pt = _per_arch_fused_plus_firsts(val)
    fused_plus_test_pt = _per_arch_fused_plus_firsts(test) if args.include_test else None

    # Recompute the existing pool's per-trace firsts using #20's recipe so
    # the margins are computed against the SAME random-gate noise pattern.
    # We need #20's thresholds + chosen mode for this arch.
    bootstrap_json = Path(args.bootstrap_json)
    with bootstrap_json.open() as f:
        boot20 = json.load(f)
    arch_rec = boot20["per_arch"][model]
    thr20 = arch_rec["thresholds"]
    mode20 = arch_rec["latent_mode_chosen"]
    rp = float(arch_rec.get("random_p", args.target_fire_rate))

    existing_val_pt = sd.firsts_per_trace(val, thr20, mode20, rp, rng)
    existing_test_pt = sd.firsts_per_trace(test, thr20, mode20, rp, rng) if args.include_test else None

    # Merge fused gates into the per-trace records (same id-keyed).
    def _merge_gate(into: list[dict], src: list[dict], key: str) -> list[dict]:
        by_id = {r["id"]: r for r in src}
        for r in into:
            srec = by_id.get(r["id"])
            r["first_fire"][key] = srec["first_fire"].get(key) if srec else None
        return into

    val_pt = _merge_gate(existing_val_pt, fused_val_pt, "latent_fused")
    val_pt = _merge_gate(val_pt, fused_val_pt_2, "latent_fused_2of3")
    val_pt = _merge_gate(val_pt, fused_plus_val_pt, "latent_fused_plus")
    test_pt = existing_test_pt
    if test_pt is not None:
        test_pt = _merge_gate(test_pt, fused_test_pt, "latent_fused")
        test_pt = _merge_gate(test_pt, fused_test_pt_2, "latent_fused_2of3")
        test_pt = _merge_gate(test_pt, fused_plus_test_pt, "latent_fused_plus")

    # Bootstrap.
    arch_out: dict = {
        "model": model,
        "n_calib_eligible": len(calib),
        "n_val_eligible": len(val),
        "n_test_eligible": len(test) if args.include_test else None,
        "fitted": {"geometry_only": meta, "geometry_plus_text": meta_p},
        "threshold_iso_fire": {"geometry_only": threshold, "geometry_plus_text": threshold_p},
        "frozen_from_20": {"mode": mode20, "thresholds": thr20, "random_p": rp},
        "validation": _eval_pool(val_pt, args.bootstrap, rng),
    }
    if test_pt is not None:
        arch_out["test"] = _eval_pool(test_pt, args.bootstrap, rng)
    return arch_out


# Reused gate pool: original five plus the three new fused variants.
EXTENDED_POOL = sd.POOL + ["latent_fused", "latent_fused_2of3", "latent_fused_plus"]


def _eval_pool(pt: list[dict], B: int, rng: random.Random) -> dict:
    per_gate: dict[str, dict] = {}
    for g in EXTENDED_POOL:
        fired = sum(1 for t in pt if t["first_fire"].get(g) is not None)
        fire_rate = fired / len(pt) if pt else 0.0
        ci = sd.boot_ci(pt, g, B, rng)
        ci["trace_fire_rate"] = fire_rate
        per_gate[g] = ci
    # Margins: compare each fused variant against the existing #20 pool members.
    margins: dict[str, dict] = {}
    for fused in ("latent_fused", "latent_fused_2of3", "latent_fused_plus"):
        for opp in ("latent", "etc", "selfdoubt_t1", "dragin", "random"):
            mci = sd.boot_margin_ci(pt, fused, opp, B, rng)
            margins[f"{fused}_minus_{opp}"] = mci
    return {"n": len(pt), "per_gate": per_gate, "margins": margins}


# ----------------------------------------------------------------------------
# Pretty printer
# ----------------------------------------------------------------------------

def _print_arch(model: str, rec: dict) -> None:
    if "error" in rec:
        print(f"[arch] {model}: ERROR {rec['error']}")
        return
    print(f"[arch] {model}")
    print(f"  n_calib={rec['n_calib_eligible']}  n_val={rec['n_val_eligible']}  "
          f"n_test={rec['n_test_eligible']}")
    print(f"  threshold_iso_fire (geom): {rec['threshold_iso_fire']['geometry_only']:.4f}")
    if not rec["fitted"]["geometry_only"].get("degenerate"):
        ws = rec["fitted"]["geometry_only"]["weights_after_standardisation"]
        print(f"  geom weights (after standardisation; sign: +ve => fires on INCORRECT):")
        for fname, w in zip(GEOM_FEATURES, ws):
            print(f"    {fname:24} = {w:+.4f}")
        print(f"  geom C_chosen = {rec['fitted']['geometry_only']['C_chosen']:.4g}")
    if not rec["fitted"]["geometry_plus_text"].get("degenerate"):
        ws_p = rec["fitted"]["geometry_plus_text"]["weights_after_standardisation"]
        print(f"  geom+text weights:")
        for fname, w in zip(GEOM_PLUS_FEATURES, ws_p):
            print(f"    {fname:24} = {w:+.4f}")
        print(f"  geom+text C_chosen = {rec['fitted']['geometry_plus_text']['C_chosen']:.4g}")
    print()
    for label in ("validation", "test"):
        if label not in rec:
            continue
        block = rec[label]
        print(f"  --- {label.upper()} (n={block['n']}) ---")
        print(f"    {'gate':22}  fire%   point      95% CI                 sig")
        for g in EXTENDED_POOL:
            ci = block["per_gate"].get(g, {})
            point = ci.get("point")
            lo = ci.get("ci_lo")
            hi = ci.get("ci_hi")
            fr = ci.get("trace_fire_rate", 0.0)
            sig = sd._sig(lo, hi)
            print(f"    {g:22}  {fr*100:5.1f}  {sd._fmt(point)}   [{sd._fmt(lo)}, {sd._fmt(hi)}]   {sig}")
        print(f"    --- key paired margins (fused − baseline) ---")
        for fused in ("latent_fused", "latent_fused_plus"):
            for opp in ("latent", "dragin", "etc", "random"):
                mci = block["margins"].get(f"{fused}_minus_{opp}", {})
                pgt = mci.get("p_gt_zero")
                p_str = "n/a" if pgt is None else f"P(>0)={pgt:.3f}"
                print(f"    {fused}-{opp:<13}  {sd._fmt(mci.get('point'))}   "
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
    p.add_argument("--include-test", action="store_true",
                   help="Also evaluate on the test split (L5-OK; single-shot after the val pick is logged).")
    p.add_argument("--bootstrap-json", default="runs/phase4_bootstrap_dragin.json",
                   help="Source of #20-frozen thresholds + chosen mode + random_p per arch.")
    p.add_argument("--out", default="runs/phase4_fused_gate.json")
    args = p.parse_args()

    rng = random.Random(args.seed)
    result: dict = {"args": vars(args), "per_arch": {}}

    print(f"[..] Tier 1: multi-feature fusion gate (entry #22)")
    print(f"     features (geom-only)  = {GEOM_FEATURES}")
    print(f"     features (geom+text)  = {GEOM_PLUS_FEATURES}")
    print(f"     target_fire_rate={args.target_fire_rate}  B={args.bootstrap}  "
          f"min_steps={args.min_steps}  seed={args.seed}")
    print(f"     include_test={args.include_test}")
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
