"""Phase 4 / #20 — re-bootstrap on the new {latent, ETC, SELFDOUBT-T1, DRAGIN,
random} pool over the #19 DRAGIN-only JSONLs.

Strictly L5-compliant:

  1. Read each per-question JSONL (`runs/phase4_qa_<model>_mid_stack_dragin.jsonl`).
  2. Split rows by `split in {calibration, validation, test}`.
  3. On the CALIBRATION split only, fit per-arch per-gate thresholds so that
     each gate's per-step fire rate ≈ target (default 0.30, the iso-fire-rate
     target that every other gate in the comparison pool uses).
  4. Also on the CALIBRATION split, pick the latent aggregation mode
     (drift / variance / or) by calibration Δfrac.
  5. Freeze (mode, thresholds). Re-evaluate every gate on VALIDATION rows.
  6. 10000-resample paired percentile bootstrap of Δfrac per gate + margin
     CIs `latent − {etc, selfdoubt_t1, dragin, random}`.
  7. Write `runs/phase4_bootstrap_dragin.json` + pretty-print a single
     headline table to stdout.

The JSONLs already carry per-step scalars (`drift_cosine`, `variance_l2_window`,
`mean_token_entropy`, `hvr`, `dragin_rind_max`) so this analyzer needs no
GPU and no gate-class instantiation; it scores each step from the saved
features under the calibrated thresholds.

Sign convention for Δfrac (matches the existing showdown):
    Δfrac = mean_first_fire_frac(correct) - mean_first_fire_frac(incorrect)
    POSITIVE => gate fires *earlier* on incorrect traces (the desired
    behaviour for a retrieval-need detector).

Usage:
    conda activate gzn
    python scripts/04g_sota_showdown.py
    python scripts/04g_sota_showdown.py --target-fire-rate 0.30 --bootstrap 10000
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DEFAULT_MODELS = [
    "phi-4-mini-instruct",
    "gemma-4-26b-a4b-it",
    "qwen3.6-35b-a3b-instruct",
]

POOL = ["latent", "etc", "selfdoubt_t1", "dragin", "random"]
LATENT_MODES = ["drift", "variance", "or"]
DEFAULT_FIRE_RATE = 0.30


# ----------------------------------------------------------------------------
# Per-step score extraction (no learning, just lookups + Δ²H on the fly)
# ----------------------------------------------------------------------------

def _trace_scores(trace: dict) -> dict[str, list[float | None]]:
    """For one trace, return per-step scalars for every gate.

    `latent` is returned as TWO parallel arrays (`drift`, `variance`) so the
    caller can pick the aggregation mode at threshold-fit time.
    `etc` is the absolute second-difference of the mean-token-entropy series,
    computed online (entries 0 and 1 are None because Δ²H is undefined).
    """
    steps = trace.get("steps") or []
    drift = [float(s.get("drift_cosine", 0.0)) for s in steps]
    var = [float(s.get("variance_l2_window", 0.0)) for s in steps]
    hvr = [float(s.get("hvr", 0.0)) for s in steps]
    rind = [float(s.get("dragin_rind_max", 0.0)) for s in steps]
    H = [float(s.get("mean_token_entropy", 0.0)) for s in steps]
    etc: list[float | None] = []
    for t in range(len(steps)):
        if t < 2:
            etc.append(None)
        else:
            etc.append(abs(H[t] - 2.0 * H[t - 1] + H[t - 2]))
    return {
        "drift": drift,
        "variance": var,
        "hvr": hvr,
        "dragin": rind,
        "etc": etc,
    }


# ----------------------------------------------------------------------------
# Quantile-based threshold calibration (iso-fire-rate)
# ----------------------------------------------------------------------------

def _quantile_threshold(scores: list[float], target_fire_rate: float) -> float:
    """Threshold s.t. P(score > thr) ≈ target_fire_rate.
    Uses the empirical (1 - target)-th quantile of the non-None score pool.
    """
    if not scores:
        return float("inf")
    s = sorted(scores)
    k = (len(s) - 1) * (1.0 - target_fire_rate)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def calibrate_per_gate(traces: list[dict], target_fire_rate: float) -> dict[str, float]:
    """Calibrate scalar thresholds for latent (drift, variance), etc, dragin.

    selfdoubt_t1 is NOT iso-fire-rate calibrated — its native semantics
    from Pandey et al. are "fire iff HVR > 0" (the hedge-to-verify ratio is
    a quasi-binary signal whose iso-fire quantile collapses to 0). We freeze
    its threshold at exactly 0 so a strict `>` recovers the Tier-1 rule.
    """
    pool: dict[str, list[float]] = {
        "drift": [], "variance": [], "etc": [], "dragin": [],
    }
    for t in traces:
        sc = _trace_scores(t)
        pool["drift"].extend(sc["drift"])
        pool["variance"].extend(sc["variance"])
        for x in sc["etc"]:
            if x is not None:
                pool["etc"].append(float(x))
        pool["dragin"].extend(sc["dragin"])
    thr = {k: _quantile_threshold(v, target_fire_rate) for k, v in pool.items()}
    thr["selfdoubt_t1"] = 0.0  # native Tier-1 gate
    return thr


# ----------------------------------------------------------------------------
# First-fire per trace under frozen thresholds
# ----------------------------------------------------------------------------

def _first_fire_index(scores: list, threshold: float, mask_warmup: int = 0) -> int | None:
    for i, x in enumerate(scores):
        if i < mask_warmup or x is None:
            continue
        if float(x) > threshold:
            return i + 1
    return None


def _latent_first_fire(sc: dict, mode: str, tau_drift: float, tau_var: float) -> int | None:
    n = len(sc["drift"])
    for i in range(n):
        d = sc["drift"][i] > tau_drift
        v = sc["variance"][i] > tau_var
        if mode == "drift":
            fire = d
        elif mode == "variance":
            fire = v
        elif mode == "or":
            fire = d or v
        else:
            raise ValueError(mode)
        if fire:
            return i + 1
    return None


def firsts_per_trace(
    traces: list[dict],
    thresholds: dict[str, float],
    latent_mode: str,
    random_p: float,
    rng: random.Random,
) -> list[dict]:
    out: list[dict] = []
    for t in traces:
        sc = _trace_scores(t)
        n_steps = int(t.get("n_steps", len(sc["drift"])))
        ff: dict[str, int | None] = {}
        ff["latent"] = _latent_first_fire(sc, latent_mode, thresholds["drift"], thresholds["variance"])
        ff["etc"] = _first_fire_index(sc["etc"], thresholds["etc"])
        ff["selfdoubt_t1"] = _first_fire_index(sc["hvr"], thresholds["selfdoubt_t1"])
        ff["dragin"] = _first_fire_index(sc["dragin"], thresholds["dragin"])
        # Random gate: per-step Bernoulli at p; first index where it fires.
        rfire = None
        for i in range(n_steps):
            if rng.random() < random_p:
                rfire = i + 1
                break
        ff["random"] = rfire
        out.append({
            "id": t.get("id"),
            "n_steps": n_steps,
            "label_correct": bool(t.get("label_correct_em", t.get("em", False))),
            "first_fire": ff,
        })
    return out


# ----------------------------------------------------------------------------
# Δfrac + bootstrap (paired)
# ----------------------------------------------------------------------------

def delta_frac(per_trace: list[dict], gate: str) -> float | None:
    corr, inco = [], []
    for t in per_trace:
        ff = t["first_fire"].get(gate)
        if ff is None:
            continue
        frac = ff / max(1, t["n_steps"])
        (corr if t["label_correct"] else inco).append(frac)
    if not corr or not inco:
        return None
    return (sum(corr) / len(corr)) - (sum(inco) / len(inco))


def _pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    if not s:
        return float("nan")
    k = (len(s) - 1) * q
    lo = int(k); hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def boot_ci(per_trace: list[dict], gate: str, B: int, rng: random.Random) -> dict:
    n = len(per_trace)
    xs: list[float] = []
    for _ in range(B):
        boot = [per_trace[rng.randrange(n)] for _ in range(n)]
        d = delta_frac(boot, gate)
        if d is not None:
            xs.append(d)
    if not xs:
        return {"point": delta_frac(per_trace, gate), "ci_lo": None, "ci_hi": None, "n_valid": 0}
    return {
        "point": delta_frac(per_trace, gate),
        "ci_lo": _pct(xs, 0.025),
        "ci_hi": _pct(xs, 0.975),
        "n_valid": len(xs),
    }


def boot_margin_ci(per_trace: list[dict], a: str, b: str, B: int, rng: random.Random) -> dict:
    n = len(per_trace)
    xs: list[float] = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        boot = [per_trace[i] for i in idx]
        da = delta_frac(boot, a)
        db = delta_frac(boot, b)
        if da is not None and db is not None:
            xs.append(da - db)
    a0 = delta_frac(per_trace, a)
    b0 = delta_frac(per_trace, b)
    point = (a0 - b0) if (a0 is not None and b0 is not None) else None
    if not xs:
        return {"point": point, "ci_lo": None, "ci_hi": None, "n_valid": 0, "p_gt_zero": None}
    return {
        "point": point,
        "ci_lo": _pct(xs, 0.025),
        "ci_hi": _pct(xs, 0.975),
        "n_valid": len(xs),
        "p_gt_zero": sum(1 for x in xs if x > 0) / len(xs),
    }


# ----------------------------------------------------------------------------
# Latent-mode pick from calibration
# ----------------------------------------------------------------------------

def pick_latent_mode(
    calib_traces: list[dict],
    thresholds: dict[str, float],
    random_p: float,
    rng: random.Random,
) -> tuple[str, dict[str, float]]:
    scores: dict[str, float | None] = {}
    for mode in LATENT_MODES:
        # Use a fresh RNG seed per mode so random-gate noise doesn't leak across modes.
        local_rng = random.Random(20260511)
        pt = firsts_per_trace(calib_traces, thresholds, latent_mode=mode,
                              random_p=random_p, rng=local_rng)
        scores[mode] = delta_frac(pt, "latent")
    # Pick the mode with the highest non-None calibration Δfrac (ties broken by mode order).
    best = None
    for mode in LATENT_MODES:
        if scores[mode] is None:
            continue
        if best is None or scores[mode] > scores[best]:
            best = mode
    return best or "drift", scores


# ----------------------------------------------------------------------------
# Pretty formatting
# ----------------------------------------------------------------------------

def _fmt(x: float | None, w: int = 8) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return f"{'n/a':>{w}}"
    return f"{x:+.4f}".rjust(w)


def _sig(ci_lo, ci_hi) -> str:
    if ci_lo is None or ci_hi is None:
        return "n/a"
    if ci_lo > 0:
        return "*** > 0"
    if ci_hi < 0:
        return "*** < 0"
    return "n.s."


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--layer-mode", default="mid_stack")
    p.add_argument("--suffix", default="_dragin")
    p.add_argument("--target-fire-rate", type=float, default=DEFAULT_FIRE_RATE)
    p.add_argument("--bootstrap", "-B", type=int, default=10000)
    p.add_argument("--min-steps", type=int, default=3)
    p.add_argument("--seed", type=int, default=20260511)
    p.add_argument("--include-test", action="store_true",
                   help="Also report on the test split (single-shot replication, L5 OK).")
    p.add_argument("--out", default="runs/phase4_bootstrap_dragin.json")
    args = p.parse_args()

    rng = random.Random(args.seed)
    runs_dir = Path("runs")
    out_path = Path(args.out)
    result = {"args": vars(args), "per_arch": {}}

    print(f"[..] re-bootstrap on the {POOL} pool")
    print(f"     target_fire_rate={args.target_fire_rate}  B={args.bootstrap}  "
          f"min_steps={args.min_steps}  seed={args.seed}")
    print()

    for m in args.models:
        jpath = runs_dir / f"phase4_qa_{m}_{args.layer_mode}{args.suffix}.jsonl"
        if not jpath.exists():
            print(f"[!!] missing {jpath}; skipping {m}")
            continue
        rows: list[dict] = []
        with jpath.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        calib = [r for r in rows if r.get("split") == "calibration"]
        valid = [r for r in rows if r.get("split") == "validation"]
        test = [r for r in rows if r.get("split") == "test"]

        # Eligibility filter (min_steps), applied identically to all gates.
        calib_e = [r for r in calib if int(r.get("n_steps", 0)) >= args.min_steps]
        valid_e = [r for r in valid if int(r.get("n_steps", 0)) >= args.min_steps]
        test_e = [r for r in test if int(r.get("n_steps", 0)) >= args.min_steps]

        # 1. Threshold calibration on calibration split.
        thresholds = calibrate_per_gate(calib_e, args.target_fire_rate)
        random_p = float(args.target_fire_rate)
        # 2. Latent mode selection on calibration split.
        chosen_mode, mode_scores = pick_latent_mode(calib_e, thresholds, random_p, rng)

        # 3. Per-trace first-fire on validation.
        pt_valid = firsts_per_trace(valid_e, thresholds, chosen_mode, random_p, rng)
        # 4. Per-trace first-fire on test (if requested).
        pt_test = firsts_per_trace(test_e, thresholds, chosen_mode, random_p, rng) if args.include_test else None

        # Headline log line.
        print(f"[arch] {m}")
        print(f"  rows: calibration={len(calib)} (elig {len(calib_e)})  "
              f"validation={len(valid)} (elig {len(valid_e)})  "
              f"test={len(test)} (elig {len(test_e)})")
        print(f"  thresholds (iso-fire {args.target_fire_rate}):")
        for k in ("drift", "variance", "etc", "selfdoubt_t1", "dragin"):
            print(f"    {k:13} > {thresholds[k]:.4f}")
        print(f"  latent_mode_scores (calibration Δfrac per mode):")
        for k in LATENT_MODES:
            print(f"    {k:9} -> {_fmt(mode_scores[k])}")
        print(f"  latent_mode_chosen = {chosen_mode}")
        print()

        def _emit(label: str, pt: list[dict]) -> dict:
            print(f"  --- {label} (n={len(pt)}) ---")
            per_gate: dict[str, dict] = {}
            print(f"    {'gate':14}  fire%   point      95% CI                 sig")
            for g in POOL:
                fired = sum(1 for t in pt if t["first_fire"].get(g) is not None)
                fire_rate_trace = fired / len(pt) if pt else 0.0
                ci = boot_ci(pt, g, args.bootstrap, rng)
                ci["trace_fire_rate"] = fire_rate_trace
                per_gate[g] = ci
                print(f"    {g:14}  {fire_rate_trace*100:5.1f}  {_fmt(ci['point'])}   "
                      f"[{_fmt(ci['ci_lo'])}, {_fmt(ci['ci_hi'])}]   "
                      f"{_sig(ci['ci_lo'], ci['ci_hi'])}")
            print(f"    --- paired margins (latent − baseline) ---")
            margins: dict[str, dict] = {}
            for opp in ("etc", "selfdoubt_t1", "dragin", "random"):
                mci = boot_margin_ci(pt, "latent", opp, args.bootstrap, rng)
                margins[f"latent_minus_{opp}"] = mci
                p_str = "n/a" if mci.get("p_gt_zero") is None else f"P(>0)={mci['p_gt_zero']:.3f}"
                print(f"    latent-{opp:<13}  {_fmt(mci['point'])}   "
                      f"[{_fmt(mci['ci_lo'])}, {_fmt(mci['ci_hi'])}]   "
                      f"{p_str}   {_sig(mci.get('ci_lo'), mci.get('ci_hi'))}")
            print()
            return {"per_gate": per_gate, "margins": margins, "n": len(pt)}

        result["per_arch"][m] = {
            "n_calibration": len(calib),
            "n_calibration_eligible": len(calib_e),
            "n_validation": len(valid),
            "n_validation_eligible": len(valid_e),
            "n_test": len(test),
            "n_test_eligible": len(test_e),
            "thresholds": thresholds,
            "random_p": random_p,
            "latent_mode_chosen": chosen_mode,
            "latent_mode_calib_scores": mode_scores,
            "validation": _emit("VALIDATION", pt_valid),
        }
        if pt_test is not None:
            result["per_arch"][m]["test"] = _emit("TEST", pt_test)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[ok] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
