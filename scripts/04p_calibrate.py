"""v2 calibration: k-fold CV on an existing closed-book trace file to pick
the latent gate's aggregation mode AND thresholds for ALL gates, with the
calibration cohort defined by --splits.

Outputs a thresholds JSON in the same shape downstream tools expect:

    {
      "thresholds": {"drift": ..., "variance": ..., "etc": ..., "selfdoubt_t1": ..., "dragin": ...},
      "latent_mode_chosen": "drift" | "variance" | "or",
      "random_p": 0.30,
      "cv_details": {...}
    }

The target fire rate is trace-level (NOT step-level), to match the v1 #25
lesson — step-level iso-0.30 produces trace-level 80-90% on Qwen and
collapses gate selectivity. Trace-level iso-0.30 means "in calibration,
~30% of traces should have at least one step where the gate fires."

Usage:

    python scripts/04p_calibrate.py \
        --closed-book runs/phase4_qa_phi-4-mini-instruct_mid_stack_dragin.jsonl \
        --splits calibration validation test  # use ALL v1 200-q rows as calibration cohort
        --target-trace-fire-rate 0.30 \
        --kfold 5 \
        --out runs/v2_phi_calibration.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from importlib import import_module
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sd = import_module("04g_sota_showdown")  # type: ignore

LATENT_MODES = ["drift", "variance", "or"]


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _all_step_scores(traces: list[dict], key: str) -> list[float]:
    """Pool per-step scores across traces via sd._trace_scores, so canonical
    field-name translation is applied (drift→drift_cosine, variance→
    variance_l2_window, dragin→dragin_rind_max, etc→Δ²H of mean_token_entropy,
    hvr→hvr). Reading raw step records with s.get(key) silently returns []
    for the un-translated keys and collapses every threshold to 0.0.
    """
    out: list[float] = []
    for t in traces:
        sc = sd._trace_scores(t)
        for v in sc.get(key, []):
            if v is not None:
                out.append(float(v))
    return out


def _trace_fire_rate(traces: list[dict], gate: str, thresholds: dict,
                     latent_mode: str, random_p: float, rng: random.Random) -> float:
    fired = 0
    for t in traces:
        sc = sd._trace_scores(t)
        if gate == "latent":
            idx = sd._latent_first_fire(sc, latent_mode, thresholds["drift"], thresholds["variance"])
        elif gate == "random":
            n_steps = int(t.get("n_steps", len(sc["drift"])))
            idx = None
            for i in range(n_steps):
                if rng.random() < random_p:
                    idx = i + 1
                    break
        else:
            score_key = {"etc": "etc", "selfdoubt_t1": "hvr", "dragin": "dragin"}[gate]
            idx = sd._first_fire_index(sc[score_key], thresholds[gate])
        if idx is not None:
            fired += 1
    return fired / len(traces) if traces else 0.0


def _trace_quantile_threshold(traces: list[dict], score_key: str,
                              target_trace_rate: float) -> float:
    """Pick a per-step threshold τ such that approximately `target_trace_rate`
    of traces have at least one step with score > τ.

    Strategy: sweep candidate τ over per-step score quantiles, find the
    smallest τ that brings trace-level fire rate <= target.
    """
    all_scores = sorted(_all_step_scores(traces, score_key))
    if not all_scores:
        return 0.0
    candidates = [all_scores[int(q * (len(all_scores) - 1))]
                  for q in (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99)]
    candidates = sorted(set(candidates))
    best_thr, best_diff = candidates[-1], float("inf")
    per_trace = [[float(v) for v in sd._trace_scores(t).get(score_key, []) if v is not None]
                 for t in traces]
    for thr in candidates:
        fired = sum(1 for vs in per_trace if any(v > thr for v in vs))
        rate = fired / len(traces)
        diff = abs(rate - target_trace_rate)
        if diff < best_diff:
            best_diff, best_thr = diff, thr
    return float(best_thr)


def _select_latent_mode(traces: list[dict], thresholds: dict, kfold: int,
                        seed: int, target_trace_rate: float) -> tuple[str, dict]:
    """k-fold CV: for each latent mode, compute mean Δfrac across the k folds.
    Pick the mode with the largest mean Δfrac (tie-break: largest minimum).
    """
    rng = random.Random(seed)
    shuffled = list(range(len(traces)))
    rng.shuffle(shuffled)
    folds = [shuffled[i::kfold] for i in range(kfold)]

    per_mode: dict[str, list[float]] = {m: [] for m in LATENT_MODES}
    for mode in LATENT_MODES:
        for fold_idx in range(kfold):
            test_idx = set(folds[fold_idx])
            fold_test = [traces[i] for i in test_idx]
            # Δfrac on this fold given the (whole-cohort) thresholds.
            random_p = target_trace_rate
            pt = sd.firsts_per_trace(fold_test, thresholds, mode, random_p,
                                     random.Random(seed + fold_idx))
            df = sd.delta_frac(pt, "latent")
            if df is not None:
                per_mode[mode].append(df)

    means = {m: (sum(v) / len(v) if v else None) for m, v in per_mode.items()}
    mins  = {m: (min(v)        if v else None) for m, v in per_mode.items()}
    # pick max mean; if tied, max min
    best = max(LATENT_MODES, key=lambda m: (means[m] if means[m] is not None else -1e9,
                                            mins[m]  if mins[m]  is not None else -1e9))
    return best, {"per_fold_deltafrac": per_mode, "mean_deltafrac": means,
                  "min_deltafrac": mins, "kfold": kfold}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--closed-book", required=True)
    p.add_argument("--splits", nargs="+", default=None,
                   help="Which split tags to include in the calibration cohort. "
                        "Default: all rows.")
    p.add_argument("--target-trace-fire-rate", type=float, default=0.30)
    p.add_argument("--kfold", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260511)
    p.add_argument("--min-steps", type=int, default=3,
                   help="Drop traces below this n_steps from calibration (gates can't fire).")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rows = _load_jsonl(Path(args.closed_book))
    if args.splits:
        rows = [r for r in rows if r.get("split") in set(args.splits)]
    rows = [r for r in rows if int(r.get("n_steps", 0)) >= args.min_steps]
    if not rows:
        print(f"[err] no traces match the filter"); return 1

    print(f"[..] calibration cohort: n={len(rows)}  target_trace_fire_rate={args.target_trace_fire_rate}")

    # Per-gate thresholds at trace-level iso-fire-rate.
    thresholds = {
        "drift":       _trace_quantile_threshold(rows, "drift",       args.target_trace_fire_rate),
        "variance":    _trace_quantile_threshold(rows, "variance",    args.target_trace_fire_rate),
        "etc":         _trace_quantile_threshold(rows, "etc",         args.target_trace_fire_rate),
        "selfdoubt_t1": 0.0,  # Pandey native rule: HVR > 0 (no calibration knob)
        "dragin":      _trace_quantile_threshold(rows, "dragin",      args.target_trace_fire_rate),
    }
    print(f"[..] thresholds (trace-level iso-{args.target_trace_fire_rate:.2f}):")
    for k, v in thresholds.items():
        print(f"      {k:14}  τ = {v:.4f}")

    # Cross-validate latent mode
    mode, cv = _select_latent_mode(rows, thresholds, args.kfold, args.seed,
                                   args.target_trace_fire_rate)
    print(f"[..] latent_mode chosen by {args.kfold}-fold CV: {mode!r}")
    for m in LATENT_MODES:
        mean = cv["mean_deltafrac"].get(m)
        mn   = cv["min_deltafrac"].get(m)
        if mean is None:
            print(f"      {m:9}  mean Δfrac = n/a")
        else:
            print(f"      {m:9}  mean Δfrac = {mean:+.4f}   min Δfrac = {mn:+.4f}")

    # Observed trace-level fire rates under final config
    rng = random.Random(args.seed)
    fire_rates = {
        g: _trace_fire_rate(rows, g, thresholds, mode, args.target_trace_fire_rate, rng)
        for g in ("latent", "etc", "selfdoubt_t1", "dragin", "random")
    }
    print(f"[..] observed trace-level fire rates on calibration cohort:")
    for g, r in fire_rates.items():
        print(f"      {g:14}  {r*100:5.1f}%")

    result = {
        "thresholds": thresholds,
        "latent_mode_chosen": mode,
        "random_p": args.target_trace_fire_rate,
        "calibration_cohort_n": len(rows),
        "splits_used": args.splits,
        "target_trace_fire_rate": args.target_trace_fire_rate,
        "observed_trace_fire_rates": fire_rates,
        "cv_details": cv,
        "args": vars(args),
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"[ok] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
