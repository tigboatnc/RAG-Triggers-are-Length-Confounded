"""Phase 4.7 — post-hoc add-on gates (SELFDOUBT-Tier1 HVR + ETC).

Reads the existing 4.1b per-question JSONLs and re-evaluates each step
record under two new gates that supersede the deprecated simulators:

  - ``SelfDoubtTier1Gate`` — true HVR per Pandey et al. (`arXiv:2604.06389v1`).
  - ``EtcGate``           — true second-derivative-of-entropy per Li et al.

Both gates are CPU-only post-hoc evaluators over the saved ``step_text``
and ``mean_token_entropy`` fields; no GPU needed (see `_AGENT.md` 2026-05-11 #16).

For each arch:
  1. Score every step under both new gates (continuous score, no firing yet).
  2. Calibrate the per-gate threshold on the ``calibration`` split (120 q)
     to hit a target per-step fire rate (default 0.30) — same L5-clean recipe
     as ``04_showdown.py``.
  3. With those thresholds frozen, compute first-fire-per-trace on the
     ``validation`` split (40 q) and the earliness Δfrac:

         Δfrac = mean(first_fire_frac | correct) - mean(first_fire_frac | incorrect)

     so that POSITIVE = gate fires earlier on hallucinated answers, matching
     the showdown's sign convention.
  4. 10000-resample percentile bootstrap of Δfrac and of the latent-minus-gate
     and gate-minus-random paired margins.

Outputs:
  - ``runs/phase4_bootstrap_v2_addons.json``   (per-arch CIs + margins)
  - pretty-printed summary table to stdout

Usage:
  python scripts/04e_addon_gates.py
  python scripts/04e_addon_gates.py --target-fire-rate 0.30 --bootstrap 10000
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from crv import SelfDoubtTier1Gate, EtcGate, hedge_verify_counts  # type: ignore[import-not-found]

# Reuse showdown loaders + bootstrap helpers.
from importlib import import_module
sd = import_module("04_showdown")  # type: ignore
bs = import_module("04c_bootstrap")  # type: ignore

DEFAULT_MODELS = sd.DEFAULT_MODELS
NEW_GATES = ["selfdoubt_t1", "etc"]


# -----------------------------------------------------------------------------
# Per-step score computation (no firing yet)
# -----------------------------------------------------------------------------

def _score_trace(trace: dict) -> dict:
    """Return per-step scores for selfdoubt_t1 (HVR) and etc (|D2H|).

    Pure functions of the saved step records; idempotent.
    """
    steps = trace.get("steps") or []
    hvr_scores: list[float] = []
    etc_scores: list[float | None] = []
    history: list[float] = []
    for s in steps:
        # HVR
        h, v = hedge_verify_counts(str(s.get("step_text", "")))
        hvr_scores.append(h / (v + 1))
        # ETC: |D2H_t| on step-mean entropy, needs 3 consecutive points.
        history.append(float(s.get("mean_token_entropy", 0.0)))
        if len(history) < 3:
            etc_scores.append(None)
        else:
            d2h = history[-1] - 2.0 * history[-2] + history[-3]
            etc_scores.append(abs(d2h))
    return {"selfdoubt_t1": hvr_scores, "etc": etc_scores}


# -----------------------------------------------------------------------------
# Threshold calibration on calibration split
# -----------------------------------------------------------------------------

def _calibrate_threshold(
    traces: list[dict],
    gate_name: str,
    target_fire_rate: float,
) -> float:
    """Pick threshold so that per-step fire rate ≈ target on these traces.

    For ``selfdoubt_t1`` (HVR) the natural threshold space is [0, ∞). For
    ``etc`` (|D2H|) it's also [0, ∞). We use the empirical
    ``(1 - target_fire_rate)``-quantile of the non-null score distribution.
    """
    all_scores: list[float] = []
    for t in traces:
        sc = _score_trace(t)
        for x in sc[gate_name]:
            if x is None:
                continue
            all_scores.append(float(x))
    if not all_scores:
        return float("inf")
    s = sorted(all_scores)
    # We want roughly target_fire_rate of scores to be > threshold.
    # Use the (1 - target_fire_rate)-th quantile.
    k = (len(s) - 1) * (1.0 - target_fire_rate)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


# -----------------------------------------------------------------------------
# First-fire-per-trace under fixed thresholds
# -----------------------------------------------------------------------------

def _first_fire(scores: list, threshold: float) -> int | None:
    for i, x in enumerate(scores):
        if x is None:
            continue
        if float(x) > threshold:
            return i + 1  # 1-indexed to match the showdown convention
    return None


def _build_per_trace(
    valid_traces: list[dict],
    thresholds: dict[str, float],
) -> list[dict]:
    out = []
    for t in valid_traces:
        sc = _score_trace(t)
        ff = {
            g: _first_fire(sc[g], thresholds[g])
            for g in NEW_GATES
        }
        out.append({
            "id": t.get("id"),
            "label_correct": bool(t.get("label_correct_em", False)),
            "n_steps": int(t.get("n_steps", 0)),
            "first_fire": ff,
        })
    return out


# -----------------------------------------------------------------------------
# Headline gate from existing summary (latent + random)
# -----------------------------------------------------------------------------

def _existing_per_trace_for_latent_random(
    model: str,
    suffix: str,
    layer_mode: str,
    summary: dict,
) -> list[dict]:
    """Re-derive first-fire for {latent, random} using the same machinery
    the bootstrap script uses, so we can put new gates side-by-side."""
    rows = sd._load_traces(model, layer_mode, suffix)
    valid = [r for r in rows if r.get("split") == "validation"]
    thresholds = summary[model]["thresholds"]
    chosen_mode = summary[model].get("latent_mode_chosen", "drift")
    return sd._firsts_per_trace(valid, thresholds, latent_mode=chosen_mode)


# -----------------------------------------------------------------------------
# Bootstrap on the combined per-trace records (latent + random + new gates)
# -----------------------------------------------------------------------------

def _merge_per_trace(existing: list[dict], addons: list[dict]) -> list[dict]:
    """Attach addon first_fire entries onto existing per-trace dicts.

    existing comes from sd._firsts_per_trace and carries latent/realm_sim/
    grip_sim/random first_fire; addons carries selfdoubt_t1/etc.
    Aligned by trace id (or by order if ids absent).
    """
    by_id = {a["id"]: a for a in addons if a.get("id") is not None}
    merged: list[dict] = []
    for e in existing:
        # The showdown stores trace id under different keys depending on version;
        # try a couple of fallbacks before degrading to positional.
        eid = e.get("id") or e.get("trace_id") or e.get("qid")
        addon = by_id.get(eid)
        if addon is None:
            # Position fallback only if lengths match.
            if len(existing) == len(addons):
                addon = addons[existing.index(e)]
        if addon is not None:
            e2 = dict(e)
            ff = dict(e.get("first_fire", {}))
            ff.update(addon["first_fire"])
            e2["first_fire"] = ff
            merged.append(e2)
        else:
            merged.append(e)
    return merged


def _fmt(x: float | None, w: int = 8) -> str:
    if x is None or (isinstance(x, float) and x != x):  # nan
        return f"{'n/a':>{w}}"
    return f"{x:+.{w-4}f}"


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--layer-mode", default="mid_stack")
    p.add_argument("--suffix", default="_v2")
    p.add_argument("--target-fire-rate", type=float, default=0.30)
    p.add_argument("--bootstrap", "-B", type=int, default=10000)
    p.add_argument("--min-steps", type=int, default=3)
    p.add_argument("--seed", type=int, default=20260511)
    p.add_argument("--summary-path", default=None,
                   help="defaults to runs/phase4_showdown<suffix>_summary.json")
    p.add_argument("--out", default=None,
                   help="defaults to runs/phase4_bootstrap<suffix>_addons.json")
    args = p.parse_args()

    summary_path = Path(args.summary_path or f"runs/phase4_showdown{args.suffix}_summary.json")
    summary = json.loads(summary_path.read_text())["per_arch"]
    out_path = Path(args.out or f"runs/phase4_bootstrap{args.suffix}_addons.json")
    rng = random.Random(args.seed)

    print(f"[..] Phase 4.7 add-on gates: SELFDOUBT-Tier1 (HVR) + ETC (|D2H|)")
    print(f"     target_fire_rate={args.target_fire_rate}, bootstrap B={args.bootstrap}, seed={args.seed}")
    print(f"     loading existing summary: {summary_path}")
    print()

    result = {"args": vars(args), "per_arch": {}}

    for m in args.models:
        rows = sd._load_traces(m, args.layer_mode, args.suffix)
        calib = [r for r in rows if r.get("split") == "calibration"]
        valid = [r for r in rows if r.get("split") == "validation"]

        # 1. Calibrate per-gate threshold on calibration split.
        thresholds = {
            g: _calibrate_threshold(calib, g, args.target_fire_rate)
            for g in NEW_GATES
        }

        # 2. Build addon per-trace for validation.
        addon_per_trace = _build_per_trace(valid, thresholds)
        # 3. Existing latent + random per-trace (from auto-mode summary).
        existing = _existing_per_trace_for_latent_random(m, args.suffix, args.layer_mode, summary)
        # Merge id-wise so bootstrap helpers can see both.
        merged = _merge_per_trace(existing, addon_per_trace)
        elig = [t for t in merged if t["n_steps"] >= args.min_steps]

        # Compute observed fire rates for sanity check.
        def _fr(elig, g):
            fired = sum(1 for t in elig if t["first_fire"].get(g) is not None)
            return fired / max(1, len(elig))
        obs_fr = {g: _fr(elig, g) for g in NEW_GATES + ["latent", "random"]}

        chosen_mode = summary[m].get("latent_mode_chosen", "drift")
        print(f"[arch] {m}  latent_mode={chosen_mode}  n_eligible={len(elig)}")
        print(f"  calibrated thresholds:  selfdoubt_t1.HVR > {thresholds['selfdoubt_t1']:.3f}   "
              f"etc.|D2H| > {thresholds['etc']:.3f}")
        print(f"  observed valid fire rates: "
              + "  ".join(f"{g}={obs_fr[g]:.2f}" for g in ["latent", "selfdoubt_t1", "etc", "random"]))

        # 4. Per-gate Δfrac CIs (only on new gates; existing pool was bootstrapped in #15).
        print(f"  {'gate':14}  point     95% CI            sig")
        per_gate: dict[str, dict] = {}
        for g in NEW_GATES:
            ci = bs.bootstrap_ci(elig, g, args.bootstrap, rng)
            per_gate[g] = ci
            if ci["ci_lo"] is None:
                sig = "n/a (no fires on one side)"
            elif ci["ci_lo"] > 0:
                sig = "*** > 0 (95%)"
            elif ci["ci_hi"] < 0:
                sig = "*** < 0 (95%)"
            else:
                sig = "n.s."
            print(f"  {g:14}  {_fmt(ci['point'])}  [{_fmt(ci['ci_lo'])}, {_fmt(ci['ci_hi'])}]   {sig}")

        # 5. Margin CIs: latent − {selfdoubt_t1, etc} and {selfdoubt_t1, etc} − random.
        print(f"  --- margins (paired bootstrap of Δfrac difference) ---")
        margins: dict[str, dict] = {}
        def _fmt_margin(mci: dict, lhs: str, rhs: str) -> str:
            p = mci.get("p_gt_zero")
            pstr = f"P(>0)={p:.3f}" if p is not None else "P(>0)=n/a"
            if mci["ci_lo"] is None:
                sig = "n/a (insufficient fires)"
            elif mci["ci_lo"] > 0:
                sig = "*** > 0"
            elif mci["ci_hi"] < 0:
                sig = "*** < 0"
            else:
                sig = "n.s."
            label = f"{lhs}-{rhs}"
            return (f"  {label:<24}  {_fmt(mci['point'])}  "
                    f"[{_fmt(mci['ci_lo'])}, {_fmt(mci['ci_hi'])}]   {pstr}   {sig}")

        for opp in NEW_GATES:
            mci = bs.bootstrap_margin_ci(elig, "latent", opp, args.bootstrap, rng)
            margins[f"latent_minus_{opp}"] = mci
            print(_fmt_margin(mci, "latent", opp))
        for g in NEW_GATES:
            mci = bs.bootstrap_margin_ci(elig, g, "random", args.bootstrap, rng)
            margins[f"{g}_minus_random"] = mci
            print(_fmt_margin(mci, g, "random"))

        result["per_arch"][m] = {
            "latent_mode_chosen": chosen_mode,
            "n_eligible": len(elig),
            "thresholds": thresholds,
            "observed_fire_rate": obs_fr,
            "per_gate_ci": per_gate,
            "margins": margins,
        }
        print()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[ok] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
