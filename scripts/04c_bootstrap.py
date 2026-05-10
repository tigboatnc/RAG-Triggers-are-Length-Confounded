"""Bootstrap 95 % CIs on per-arch earliness Δfrac.

Resamples validation traces with replacement (B=10000 by default), recomputes
each gate's earliness_delta_frac per resample, reports percentile CIs. Also
reports the load-bearing "latent - random" margin CI per arch.

Reads the canonical 4.1b JSONLs + the auto-mode summary so the latent gate
uses the calibration-chosen mode per architecture (no leakage).

Usage:
    python scripts/04c_bootstrap.py --suffix _v2 --bootstrap 10000

Output:
    runs/phase4_bootstrap_v2.json  -- per-arch CI tables + margin CIs
    + pretty-printed table to stdout
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Reuse the showdown's data loaders + gate machinery.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module
sd = import_module("04_showdown")  # type: ignore

GATE_ORDER = sd.GATE_ORDER
DEFAULT_MODELS = sd.DEFAULT_MODELS


def _delta_frac_for_gate(per_trace_subset: list[dict], gate: str) -> float | None:
    """Mean first-fire-frac on incorrect traces minus on correct traces.
    Returns None if either side has no fires.
    """
    corr_fracs, inco_fracs = [], []
    for t in per_trace_subset:
        ff = t["first_fire"].get(gate)
        if ff is None:
            continue
        n = max(1, t["n_steps"])
        frac = ff / n
        if t["label_correct"]:
            corr_fracs.append(frac)
        else:
            inco_fracs.append(frac)
    if not corr_fracs or not inco_fracs:
        return None
    # Sign convention matches scripts/04_showdown.py earliness_table:
    #   delta_frac = mean_frac_correct - mean_frac_incorrect
    # so that POSITIVE delta = gate fires *earlier* on incorrect traces
    # (small frac on incorrect, larger frac on correct).
    return (sum(corr_fracs) / len(corr_fracs)) - (sum(inco_fracs) / len(inco_fracs))


def _percentile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def bootstrap_ci(per_trace: list[dict], gate: str, B: int, rng: random.Random) -> dict:
    """B-resample bootstrap of Δfrac for a single gate. Returns mean + 95 % percentile CI."""
    n = len(per_trace)
    samples: list[float] = []
    for _ in range(B):
        boot = [per_trace[rng.randrange(n)] for _ in range(n)]
        d = _delta_frac_for_gate(boot, gate)
        if d is not None:
            samples.append(d)
    if not samples:
        return {"point": None, "ci_lo": None, "ci_hi": None, "n_valid": 0}
    point = _delta_frac_for_gate(per_trace, gate)
    return {
        "point": point,
        "ci_lo": _percentile(samples, 0.025),
        "ci_hi": _percentile(samples, 0.975),
        "n_valid": len(samples),
    }


def bootstrap_margin_ci(per_trace: list[dict], gate_a: str, gate_b: str, B: int, rng: random.Random) -> dict:
    """Bootstrap CI on (gate_a Δfrac − gate_b Δfrac) on the same resampled traces.
    This is the right test for "does latent beat random" — it accounts for the
    correlation across gates within a trace (they share the same trace lengths,
    EM labels, etc.).
    """
    n = len(per_trace)
    samples: list[float] = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        boot = [per_trace[i] for i in idx]
        a = _delta_frac_for_gate(boot, gate_a)
        b = _delta_frac_for_gate(boot, gate_b)
        if a is not None and b is not None:
            samples.append(a - b)
    if not samples:
        return {"point": None, "ci_lo": None, "ci_hi": None, "n_valid": 0, "p_gt_zero": None}
    a0 = _delta_frac_for_gate(per_trace, gate_a) or 0.0
    b0 = _delta_frac_for_gate(per_trace, gate_b) or 0.0
    return {
        "point": a0 - b0,
        "ci_lo": _percentile(samples, 0.025),
        "ci_hi": _percentile(samples, 0.975),
        "n_valid": len(samples),
        "p_gt_zero": sum(1 for x in samples if x > 0) / len(samples),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--layer-mode", default="mid_stack")
    p.add_argument("--suffix", default="_v2")
    p.add_argument("--summary-path", default=None,
                   help="defaults to runs/phase4_showdown<suffix>_summary.json")
    p.add_argument("--bootstrap", "-B", type=int, default=10000)
    p.add_argument("--min-steps", type=int, default=3)
    p.add_argument("--seed", type=int, default=20260510)
    p.add_argument("--out", default=None,
                   help="defaults to runs/phase4_bootstrap<suffix>.json")
    args = p.parse_args()

    summary_path = Path(args.summary_path or f"runs/phase4_showdown{args.suffix}_summary.json")
    summary = json.loads(summary_path.read_text())["per_arch"]
    out_path = Path(args.out or f"runs/phase4_bootstrap{args.suffix}.json")

    rng = random.Random(args.seed)
    result = {"args": vars(args), "per_arch": {}}

    print(f"[..] bootstrap B={args.bootstrap}, summary={summary_path}, seed={args.seed}")
    for m in args.models:
        if m not in summary:
            print(f"[!!] {m} not in summary; skipping.")
            continue
        rows = sd._load_traces(m, args.layer_mode, args.suffix)  # type: ignore
        valid = [r for r in rows if r.get("split") == "validation"]
        thresholds = summary[m]["thresholds"]
        chosen_mode = summary[m].get("latent_mode_chosen", "drift")

        per_trace = sd._firsts_per_trace(valid, thresholds, latent_mode=chosen_mode)
        elig = [t for t in per_trace if t["n_steps"] >= args.min_steps]

        nice = m.split("-")[0]
        print(f"\n[arch] {m}  mode={chosen_mode}  n_eligible={len(elig)}")
        print(f"  {'gate':10}  point     95% CI            sig (CI excludes 0)")
        per_gate = {}
        for g in GATE_ORDER:
            ci = bootstrap_ci(elig, g, args.bootstrap, rng)
            per_gate[g] = ci
            sig = ""
            if ci["ci_lo"] is not None and ci["ci_hi"] is not None:
                if ci["ci_lo"] > 0:
                    sig = "*** > 0"
                elif ci["ci_hi"] < 0:
                    sig = "*** < 0"
                else:
                    sig = "n.s."
            print(f"  {g:10}  {ci['point']:+.4f}   "
                  f"[{ci['ci_lo']:+.4f}, {ci['ci_hi']:+.4f}]   {sig}")

        # Margin CIs: latent vs each baseline (paired bootstrap).
        margins = {}
        print(f"  --- margins (paired bootstrap of Δfrac difference) ---")
        for opp in ("realm_sim", "grip_sim", "random"):
            mci = bootstrap_margin_ci(elig, "latent", opp, args.bootstrap, rng)
            margins[f"latent_minus_{opp}"] = mci
            sig = "*** > 0" if mci["ci_lo"] is not None and mci["ci_lo"] > 0 else (
                "*** < 0" if mci["ci_hi"] is not None and mci["ci_hi"] < 0 else "n.s."
            )
            print(f"  latent-{opp:<10}  {mci['point']:+.4f}   "
                  f"[{mci['ci_lo']:+.4f}, {mci['ci_hi']:+.4f}]   "
                  f"P(>0)={mci['p_gt_zero']:.3f}   {sig}")

        result["per_arch"][m] = {
            "latent_mode_chosen": chosen_mode,
            "n_eligible": len(elig),
            "per_gate_ci": per_gate,
            "margins": margins,
        }

    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n[ok] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
