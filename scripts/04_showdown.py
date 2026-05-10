"""Phase 4.2 - 4.5 showdown analysis on Phase 4.1b JSONLs.

Reads per-question traces produced by ``04_run_qa.py`` (one file per
architecture under ``runs/phase4_qa_<model>_<layer_mode>{_v2}.jsonl``)
and computes everything Phase 4.2 / 4.3 / 4.4(a) / 4.5 needs, **offline,
no GPU required**:

  1. Per-architecture gate-threshold calibration on the ``calibration``
     split (120 q). Iso-fire-rate target (default 30 %) so all four gates
     are compared under the same firing budget.

  2. Earliness metric (4.2) on the ``validation`` split (40 q). For each
     (gate x arch), report mean first-fire step on label-correct vs
     label-incorrect traces, and the EARLINESS DELTA
     (mean_correct - mean_incorrect). Positive delta means the gate
     fires earlier on traces that end up wrong - the headline claim.

  3. Trajectory-drift table (4.3). Per (gate x arch), bin traces by
     first-fire decile and report mean EM/F1 per bin. Monotonic decay
     (early-fire -> bad outcome) is the signal.

  4. Ablation (4.4 a): re-derive latent decisions under
     ``mode in {drift, variance, or}`` at the same calibration recipe.
     Compare earliness deltas.

  5. Cost (4.5): per-(gate x arch) total wall-clock + tokens, plus
     latent's per-question forward-pass overhead breakout.

L5 leakage contract: thresholds are fit on ``split == "calibration"``
ONLY; metrics are reported on ``split == "validation"``;
``split == "test"`` is opened only at paper write-up. The script refuses
to touch test rows unless ``--include-test`` is passed.

Outputs:
  - ``runs/phase4_showdown_v2.csv``         (long format, all cells)
  - ``runs/phase4_showdown_v2_summary.json`` (machine-readable headline)
  - ``figs/phase4_earliness.png``            (4.2 headline figure)
  - ``figs/phase4_drift.png``                (4.3 trajectory drift)
  - ``figs/phase4_pareto.png``               (4.5 cost vs EM)

Usage:
  python scripts/04_showdown.py
  python scripts/04_showdown.py --target-fire-rate 0.20 --suffix _v2
  python scripts/04_showdown.py --layer-mode mid_stack --suffix _v2 --min-steps 3
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from crv import GripSimGate, RealmRetrieveSimGate  # type: ignore[import-not-found]


# Architectures we expect in the bracket. Order is the canonical paper order:
# Dense -> Sliding-window MoE -> Hybrid-Linear MoE.
DEFAULT_MODELS = [
    "phi-4-mini-instruct",
    "gemma-4-26b-a4b-it",
    "qwen3.6-35b-a3b-instruct",
]

GATE_ORDER = ["latent", "realm_sim", "grip_sim", "random"]


# -----------------------------------------------------------------------------
# I/O
# -----------------------------------------------------------------------------

def _load_traces(model: str, layer_mode: str, suffix: str) -> list[dict]:
    path = Path("runs") / f"phase4_qa_{model}_{layer_mode}{suffix}.jsonl"
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "error" not in r:
                rows.append(r)
    return rows


# -----------------------------------------------------------------------------
# Threshold calibration (iso fire rate)
# -----------------------------------------------------------------------------

def _quantile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def calibrate_thresholds(
    calib_traces: list[dict],
    target_fire_rate: float,
) -> dict[str, dict[str, float]]:
    """Pick per-gate thresholds to fire on roughly ``target_fire_rate`` of steps.

    Single-knob proxies (one threshold per gate) so the comparison stays
    interpretable. The 4.4(a) ablation re-runs latent in three different
    OR/AND modes under the SAME threshold; the 4.4(b)+(c) ablations land
    once last_layer + think_block sweeps exist.

    - latent : threshold on ``drift_cosine``  (mode='drift' for headline)
    - realm  : threshold on ``mean_token_entropy``  (RSU stand-in)
    - grip   : threshold on rolling-z(boundary_entropy) computed in-place
    - random : returns the rate; gate is Bernoulli at that rate
    """
    drifts: list[float] = []
    rsus: list[float] = []
    z_seq_per_trace: list[list[float]] = []  # for grip
    grip_window = 20

    for r in calib_traces:
        # GRIP needs a rolling window per trace; replicate at calibration time
        # by walking each trace's boundary_token_entropy through a fresh
        # _RollingStats. Reuse GripSimGate's internals via a fresh instance.
        gate = GripSimGate(z_threshold=0.0, strong_z_threshold=1e9, window_size=grip_window)
        per_trace_z: list[float] = []
        for s in r.get("steps", []):
            drifts.append(float(s.get("drift_cosine", 0.0)))
            rsus.append(float(s.get("mean_token_entropy", 0.0)))
            # We piggy-back on grip's evaluate to advance the rolling window
            # and get the same z-score the live trace saw.
            decision = gate.evaluate(s)
            # decision["score"] is z-score (per the grip impl)
            per_trace_z.append(float(decision["score"]))
        z_seq_per_trace.append(per_trace_z)

    # Iso-fire-rate quantile: pick top target_fire_rate of values.
    q = 1.0 - target_fire_rate
    tau_drift = _quantile(drifts, q) if drifts else 0.0
    tau_rsu = _quantile(rsus, q) if rsus else 0.0
    # GRIP warmup: drop the first ``grip_window`` z-scores per trace where the
    # rolling stats aren't full yet. Without this, the calibration distribution
    # is dominated by zeros (z=0 returned when std==0 in early steps), the
    # 70th-percentile of a mostly-zero distribution is 0, and GRIP fires on
    # every step where z>0 (≈half the trace). The discarded steps are still
    # scored at evaluation time -- this filter only affects threshold fitting.
    flat_z = [
        z for trz in z_seq_per_trace
        for i, z in enumerate(trz)
        if i >= grip_window  # warmup-discarded
    ]
    tau_strong_z = _quantile(flat_z, q) if flat_z else 0.0

    # Latent variance threshold for the 4.4(a) ablation cell mode='variance'.
    variances: list[float] = []
    for r in calib_traces:
        for s in r.get("steps", []):
            variances.append(float(s.get("variance_l2_window", 0.0)))
    tau_var = _quantile(variances, q) if variances else 0.0

    return {
        "latent": {
            "tau_drift": tau_drift,
            "tau_var": tau_var,
            "_n_steps_calib": len(drifts),
            "_quantile_q": q,
        },
        "realm_sim": {
            "tau_rsu_nats": tau_rsu,
            "_n_steps_calib": len(rsus),
        },
        "grip_sim": {
            # Single strong-z knob (matches our top-quantile target). We let
            # the weak-AND-marker path piggy-back on the same threshold so
            # the gate has a chance to fire on context, not just spikes.
            "tau_z": tau_strong_z * 0.6,
            "tau_strong_z": tau_strong_z,
            "window_size": 20,
            "_n_steps_calib": len(flat_z),
        },
        "random": {
            "p": target_fire_rate,
            "seed": 20260509,
        },
    }


# -----------------------------------------------------------------------------
# Re-evaluate gates under calibrated thresholds (offline)
# -----------------------------------------------------------------------------

def _latent_fired(step: dict, mode: str, thr: dict[str, float]) -> bool:
    drift = float(step.get("drift_cosine", 0.0))
    var = float(step.get("variance_l2_window", 0.0))
    if mode == "drift":
        return drift > thr["tau_drift"]
    if mode == "variance":
        return var > thr["tau_var"]
    if mode == "or":
        return (drift > thr["tau_drift"]) or (var > thr["tau_var"])
    raise ValueError(f"unknown latent mode: {mode}")


def _firsts_per_trace(
    traces: list[dict],
    thresholds: dict[str, dict[str, float]],
    latent_mode: str = "drift",
    rng_seed: int = 20260509,
) -> list[dict]:
    """For each trace, find the first step (1-indexed) at which each gate
    fires. Returns one dict per trace with keys: id, n_steps, em, f1,
    label_correct, first_fire[gate_name] (or None if never).
    """
    import random
    rng = random.Random(rng_seed)
    out = []
    for r in traces:
        steps = r.get("steps", [])
        # Fresh stateful gates per trace (rolling window resets).
        realm = RealmRetrieveSimGate(rsu_threshold_nats=thresholds["realm_sim"]["tau_rsu_nats"])
        grip = GripSimGate(
            z_threshold=thresholds["grip_sim"]["tau_z"],
            strong_z_threshold=thresholds["grip_sim"]["tau_strong_z"],
            window_size=int(thresholds["grip_sim"]["window_size"]),
        )
        ff = {g: None for g in GATE_ORDER}
        n_fires = {g: 0 for g in GATE_ORDER}
        p_random = float(thresholds["random"]["p"])
        for s in steps:
            si = int(s["step_idx"])
            if _latent_fired(s, latent_mode, thresholds["latent"]):
                n_fires["latent"] += 1
                if ff["latent"] is None:
                    ff["latent"] = si
            d = realm.evaluate(s)
            if d["fired"]:
                n_fires["realm_sim"] += 1
                if ff["realm_sim"] is None:
                    ff["realm_sim"] = si
            d = grip.evaluate(s)
            if d["fired"]:
                n_fires["grip_sim"] += 1
                if ff["grip_sim"] is None:
                    ff["grip_sim"] = si
            if rng.random() < p_random:
                n_fires["random"] += 1
                if ff["random"] is None:
                    ff["random"] = si
        out.append({
            "id": r["id"],
            "split": r.get("split", "unknown"),
            "n_steps": int(r.get("n_steps", 0)),
            "em": int(r.get("em", 0)),
            "f1": float(r.get("f1", 0.0)),
            "label_correct": bool(r.get("label_correct_em", False)),
            "tokens_generated": int(r.get("tokens_generated", 0)),
            "wall_clock_s": float(r.get("wall_clock_s", 0.0)),
            "first_fire": ff,
            "n_fires": n_fires,
        })
    return out


# -----------------------------------------------------------------------------
# Earliness + drift metrics
# -----------------------------------------------------------------------------

def _mean_or_none(xs: list[float]) -> float | None:
    return (sum(xs) / len(xs)) if xs else None


def earliness_table(
    per_trace: list[dict],
    min_steps: int = 3,
) -> list[dict]:
    """One row per gate with mean first-fire on correct vs incorrect traces.

    Filters out traces with ``n_steps < min_steps`` (no gating opportunity).
    Reports first-fire as a *fraction of trace length* in addition to absolute
    step, so we can compare across architectures with very different mean
    n_steps (Phi: ~3, Gemma: ~12, Qwen: 5-15 expected).
    """
    elig = [t for t in per_trace if t["n_steps"] >= min_steps]
    n_total = len(elig)
    n_correct = sum(1 for t in elig if t["label_correct"])
    n_incorrect = n_total - n_correct
    rows = []
    for g in GATE_ORDER:
        ffs_correct: list[float] = []
        ffs_incorrect: list[float] = []
        ffrac_correct: list[float] = []
        ffrac_incorrect: list[float] = []
        n_fired_correct = 0
        n_fired_incorrect = 0
        for t in elig:
            ff = t["first_fire"][g]
            if ff is None:
                continue
            f = ff / max(1, t["n_steps"])
            if t["label_correct"]:
                ffs_correct.append(ff)
                ffrac_correct.append(f)
                n_fired_correct += 1
            else:
                ffs_incorrect.append(ff)
                ffrac_incorrect.append(f)
                n_fired_incorrect += 1
        mean_ff_corr = _mean_or_none(ffs_correct)
        mean_ff_inco = _mean_or_none(ffs_incorrect)
        mean_frac_corr = _mean_or_none(ffrac_correct)
        mean_frac_inco = _mean_or_none(ffrac_incorrect)
        delta = (
            (mean_ff_corr - mean_ff_inco)
            if (mean_ff_corr is not None and mean_ff_inco is not None)
            else None
        )
        delta_frac = (
            (mean_frac_corr - mean_frac_inco)
            if (mean_frac_corr is not None and mean_frac_inco is not None)
            else None
        )
        rows.append({
            "gate": g,
            "n_eligible_traces": n_total,
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "fire_rate_overall": (n_fired_correct + n_fired_incorrect) / max(1, n_total),
            "mean_first_fire_correct": mean_ff_corr,
            "mean_first_fire_incorrect": mean_ff_inco,
            "earliness_delta_steps": delta,
            "mean_first_fire_frac_correct": mean_frac_corr,
            "mean_first_fire_frac_incorrect": mean_frac_inco,
            "earliness_delta_frac": delta_frac,
            "n_fired_correct": n_fired_correct,
            "n_fired_incorrect": n_fired_incorrect,
        })
    return rows


def trajectory_drift_table(
    per_trace: list[dict],
    gate: str,
    n_bins: int = 5,
    min_steps: int = 3,
) -> list[dict]:
    """Bin traces by first-fire fraction (where in the trace the gate first
    fired); report mean EM/F1 per bin. Monotonic EM decrease across bins
    (firing earlier in a trace -> trace ends worse) is the trajectory-drift
    signal.

    Traces where the gate never fired are reported in the "never_fired" bin.
    """
    elig = [t for t in per_trace if t["n_steps"] >= min_steps]
    fired = [t for t in elig if t["first_fire"][gate] is not None]
    never = [t for t in elig if t["first_fire"][gate] is None]

    fired_with_frac = [(t, t["first_fire"][gate] / max(1, t["n_steps"])) for t in fired]
    fired_with_frac.sort(key=lambda tf: tf[1])

    bins: list[list[dict]] = [[] for _ in range(n_bins)]
    for i, (t, _f) in enumerate(fired_with_frac):
        b = min(i * n_bins // max(1, len(fired_with_frac)), n_bins - 1)
        bins[b].append(t)

    out = []
    for bi, items in enumerate(bins):
        if not items:
            out.append({
                "bin": f"q{bi+1}",
                "n": 0,
                "mean_first_fire_frac": None,
                "mean_em": None,
                "mean_f1": None,
            })
            continue
        out.append({
            "bin": f"q{bi+1}",
            "n": len(items),
            "mean_first_fire_frac": sum(t["first_fire"][gate] / max(1, t["n_steps"]) for t in items) / len(items),
            "mean_em": sum(t["em"] for t in items) / len(items),
            "mean_f1": sum(t["f1"] for t in items) / len(items),
        })
    out.append({
        "bin": "never_fired",
        "n": len(never),
        "mean_first_fire_frac": None,
        "mean_em": (sum(t["em"] for t in never) / len(never)) if never else None,
        "mean_f1": (sum(t["f1"] for t in never) / len(never)) if never else None,
    })
    return out


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def _plot_earliness(per_arch_tables: dict[str, list[dict]], out_path: Path) -> None:
    import matplotlib.pyplot as plt
    archs = list(per_arch_tables.keys())
    fig, axes = plt.subplots(1, len(archs), figsize=(4 * len(archs), 4), sharey=True)
    if len(archs) == 1:
        axes = [axes]
    for ax, arch in zip(axes, archs):
        rows = per_arch_tables[arch]
        gates = [r["gate"] for r in rows]
        deltas = [r["earliness_delta_frac"] if r["earliness_delta_frac"] is not None else 0.0 for r in rows]
        colors = ["tab:blue" if g == "latent" else "tab:gray" for g in gates]
        ax.bar(gates, deltas, color=colors)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(arch, fontsize=10)
        ax.set_xlabel("gate")
        ax.tick_params(axis="x", rotation=20)
    axes[0].set_ylabel("earliness delta (frac)\n=  mean_first_fire_frac(correct) - mean_first_fire_frac(incorrect)\nhigher = fires earlier on bad traces")
    fig.suptitle("Phase 4.2 — gate earliness across architectures (validation split)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_drift(per_arch_drift: dict[str, list[dict]], out_path: Path) -> None:
    import matplotlib.pyplot as plt
    archs = list(per_arch_drift.keys())
    fig, axes = plt.subplots(1, len(archs), figsize=(4 * len(archs), 4), sharey=True)
    if len(archs) == 1:
        axes = [axes]
    for ax, arch in zip(axes, archs):
        rows = per_arch_drift[arch]
        binned = [r for r in rows if r["bin"] != "never_fired" and r["mean_em"] is not None]
        ax.plot(
            [r["mean_first_fire_frac"] for r in binned],
            [r["mean_em"] for r in binned],
            marker="o",
            label="binned by first_fire_frac",
        )
        never = [r for r in rows if r["bin"] == "never_fired"][0]
        if never["mean_em"] is not None:
            ax.axhline(never["mean_em"], color="gray", linestyle="--", label=f"never-fired baseline (n={never['n']})")
        ax.set_title(arch, fontsize=10)
        ax.set_xlabel("first-fire fraction of trace")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("mean EM in bin")
    fig.suptitle("Phase 4.3 — trajectory drift: latent gate first-fire-frac vs final EM", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_pareto(cost_rows: list[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt
    archs = sorted({r["arch"] for r in cost_rows})
    fig, ax = plt.subplots(figsize=(6, 4))
    markers = {"phi-4-mini-instruct": "o", "gemma-4-26b-a4b-it": "s", "qwen3.6-35b-a3b-instruct": "^"}
    for arch in archs:
        sub = [r for r in cost_rows if r["arch"] == arch]
        xs = [r["mean_wall_per_q_s"] for r in sub]
        ys = [r["mean_em"] for r in sub]
        labels = [r["gate"] for r in sub]
        ax.scatter(xs, ys, marker=markers.get(arch, "o"), s=60, label=arch)
        for x, y, lab in zip(xs, ys, labels):
            ax.annotate(lab, (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("mean wall-clock per question (s)  — lower is better")
    ax.set_ylabel("mean EM on validation  — higher is better")
    ax.set_title("Phase 4.5 — cost vs EM (note: gates do not change generation cost in 4.1b;\nthis is a per-arch view, gate annotation is informational)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def _fmt(x: float | None, fmt: str = "{:6.3f}") -> str:
    return "  n/a " if x is None else fmt.format(x)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--layer-mode", default="mid_stack")
    parser.add_argument("--suffix", default="_v2", help="JSONL suffix (default '_v2' for the 4.1b sweep).")
    parser.add_argument("--target-fire-rate", type=float, default=0.30,
                        help="Per-step iso-fire-rate target for threshold calibration.")
    parser.add_argument("--min-steps", type=int, default=3,
                        help="Earliness/drift filter: skip traces with fewer steps.")
    parser.add_argument("--include-test", action="store_true",
                        help="Open the test split. L5 leakage contract: only at write-up.")
    parser.add_argument("--include-ablation", action="store_true", default=True,
                        help="Run 4.4(a) latent drift/variance/or ablation.")
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--fig-dir", default="figs")
    args = parser.parse_args()

    # Discover available files
    available: dict[str, list[dict]] = {}
    for m in args.models:
        rows = _load_traces(m, args.layer_mode, args.suffix)
        if rows:
            available[m] = rows
    if not available:
        print(f"[!!] no JSONLs found at runs/phase4_qa_<model>_{args.layer_mode}{args.suffix}.jsonl")
        return 2

    print(f"[..] loaded traces:")
    for m, rs in available.items():
        from collections import Counter
        c = Counter(r.get("split", "unknown") for r in rs)
        print(f"      {m:32}  total={len(rs)}  splits={dict(c)}")

    # Per-arch pipeline.
    csv_rows: list[dict] = []
    per_arch_earliness: dict[str, list[dict]] = {}
    per_arch_drift: dict[str, list[dict]] = {}
    per_arch_thresholds: dict[str, dict] = {}
    per_arch_ablation: dict[str, dict[str, list[dict]]] = {}
    cost_rows: list[dict] = []
    headline_summary: dict[str, Any] = {
        "args": vars(args),
        "models": list(available.keys()),
        "per_arch": {},
    }

    for m, traces in available.items():
        print(f"\n{'='*72}\n[arch] {m}\n{'='*72}")
        calib = [t for t in traces if t.get("split") == "calibration"]
        valid = [t for t in traces if t.get("split") == "validation"]
        if not calib:
            print(f"  [!!] no calibration rows; skipping.")
            continue
        if not valid:
            print(f"  [!!] no validation rows; skipping.")
            continue

        # 1. Calibrate thresholds on calibration split.
        thresholds = calibrate_thresholds(calib, args.target_fire_rate)
        per_arch_thresholds[m] = thresholds
        print(f"  [calib]  fit on n={len(calib)} traces  target_fire_rate={args.target_fire_rate}")
        print(f"           latent.tau_drift = {thresholds['latent']['tau_drift']:.4f}")
        print(f"           latent.tau_var   = {thresholds['latent']['tau_var']:.4f}")
        print(f"           realm.tau_rsu    = {thresholds['realm_sim']['tau_rsu_nats']:.4f} nats")
        print(f"           grip.tau_z       = {thresholds['grip_sim']['tau_z']:.4f}")
        print(f"           grip.tau_strong  = {thresholds['grip_sim']['tau_strong_z']:.4f}")
        print(f"           random.p         = {thresholds['random']['p']:.4f}")

        # 2. Earliness on validation split.
        per_trace = _firsts_per_trace(valid, thresholds, latent_mode="drift")
        e_rows = earliness_table(per_trace, min_steps=args.min_steps)
        per_arch_earliness[m] = e_rows

        # Pretty-print earliness.
        elig = [t for t in per_trace if t["n_steps"] >= args.min_steps]
        n_corr = sum(1 for t in elig if t["label_correct"])
        n_inco = len(elig) - n_corr
        print(f"\n  [4.2]  earliness on validation (n_eligible={len(elig)}, correct={n_corr}, incorrect={n_inco}, min_steps={args.min_steps})")
        print(f"         { 'gate':10}  fire%   ff_corr  ff_inco  delta_steps  ff_corr_frac ff_inco_frac delta_frac")
        for r in e_rows:
            print(f"         {r['gate']:10}"
                  f"  {r['fire_rate_overall']*100:5.1f}%"
                  f"  {_fmt(r['mean_first_fire_correct'])}"
                  f"  {_fmt(r['mean_first_fire_incorrect'])}"
                  f"  {_fmt(r['earliness_delta_steps'])}"
                  f"     {_fmt(r['mean_first_fire_frac_correct'])}"
                  f"  {_fmt(r['mean_first_fire_frac_incorrect'])}"
                  f"  {_fmt(r['earliness_delta_frac'])}")

        # 3. Trajectory drift on validation (latent gate).
        d_rows = trajectory_drift_table(per_trace, "latent", n_bins=5, min_steps=args.min_steps)
        per_arch_drift[m] = d_rows
        print(f"\n  [4.3]  trajectory drift (gate=latent, 5 bins by first-fire-frac)")
        print(f"         { 'bin':12}  n   ff_frac  mean_EM  mean_F1")
        for r in d_rows:
            print(f"         {r['bin']:12}  {r['n']:>3}  {_fmt(r['mean_first_fire_frac'])}  {_fmt(r['mean_em'])}  {_fmt(r['mean_f1'])}")

        # 4.4(a) ablation: drift / variance / or for latent.
        if args.include_ablation:
            abl = {}
            for mode in ["drift", "variance", "or"]:
                pt = _firsts_per_trace(valid, thresholds, latent_mode=mode)
                abl[mode] = earliness_table(pt, min_steps=args.min_steps)
            per_arch_ablation[m] = abl
            print(f"\n  [4.4a] latent ablation: drift vs variance vs or  (only the 'latent' gate row)")
            print(f"         {'mode':10}  fire%   ff_corr  ff_inco  delta_steps  delta_frac")
            for mode, rows in abl.items():
                lat = next(r for r in rows if r["gate"] == "latent")
                print(f"         {mode:10}"
                      f"  {lat['fire_rate_overall']*100:5.1f}%"
                      f"  {_fmt(lat['mean_first_fire_correct'])}"
                      f"  {_fmt(lat['mean_first_fire_incorrect'])}"
                      f"  {_fmt(lat['earliness_delta_steps'])}"
                      f"  {_fmt(lat['earliness_delta_frac'])}")

        # 5. Cost — per-arch (gate is informational; gates don't change gen cost in 4.1b).
        mean_em = sum(t["em"] for t in elig) / max(1, len(elig))
        mean_f1 = sum(t["f1"] for t in elig) / max(1, len(elig))
        mean_wall = sum(t["wall_clock_s"] for t in elig) / max(1, len(elig))
        mean_tok = sum(t["tokens_generated"] for t in elig) / max(1, len(elig))
        for g in GATE_ORDER:
            cost_rows.append({
                "arch": m,
                "gate": g,
                "n_eligible": len(elig),
                "mean_em": mean_em,
                "mean_f1": mean_f1,
                "mean_wall_per_q_s": mean_wall,
                "mean_tokens_per_q": mean_tok,
            })
        print(f"\n  [4.5]  cost (validation, n={len(elig)}): EM={mean_em:.3f}  F1={mean_f1:.3f}  "
              f"mean_wall_per_q={mean_wall:.2f}s  mean_tokens={mean_tok:.1f}")

        # CSV rows for the long file.
        for r in e_rows:
            csv_rows.append({"section": "4.2 earliness", "arch": m, **r})
        for r in d_rows:
            csv_rows.append({"section": "4.3 drift_latent", "arch": m, **r})
        if args.include_ablation:
            for mode, rows in per_arch_ablation[m].items():
                for r in rows:
                    csv_rows.append({"section": f"4.4a ablation_{mode}", "arch": m, **r})

        headline_summary["per_arch"][m] = {
            "thresholds": thresholds,
            "earliness": e_rows,
            "drift_latent": d_rows,
            "ablation_4_4a": per_arch_ablation.get(m, {}),
            "cost": {
                "n_eligible": len(elig),
                "mean_em": mean_em,
                "mean_f1": mean_f1,
                "mean_wall_per_q_s": mean_wall,
                "mean_tokens_per_q": mean_tok,
            },
        }

    # Write outputs.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"phase4_showdown{args.suffix}.csv"
    if csv_rows:
        all_keys: list[str] = []
        seen = set()
        for r in csv_rows:
            for k in r.keys():
                if k not in seen:
                    all_keys.append(k); seen.add(k)
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_keys)
            w.writeheader()
            for r in csv_rows:
                w.writerow(r)
        print(f"\n[ok] wrote {csv_path} ({len(csv_rows)} rows)")

    summary_path = out_dir / f"phase4_showdown{args.suffix}_summary.json"
    with summary_path.open("w") as f:
        json.dump(headline_summary, f, indent=2, default=str)
    print(f"[ok] wrote {summary_path}")

    if per_arch_earliness:
        p = fig_dir / "phase4_earliness.png"
        _plot_earliness(per_arch_earliness, p); print(f"[ok] wrote {p}")
    if per_arch_drift:
        p = fig_dir / "phase4_drift.png"
        _plot_drift(per_arch_drift, p); print(f"[ok] wrote {p}")
    if cost_rows:
        p = fig_dir / "phase4_pareto.png"
        _plot_pareto(cost_rows, p); print(f"[ok] wrote {p}")

    # Headline reduction: latent-vs-baseline earliness delta per arch.
    print(f"\n{'='*72}\n[HEADLINE] latent-vs-baseline earliness delta (frac, validation)\n{'='*72}")
    print(f"  {'arch':32}  latent  realm   grip    random  | latent_wins?")
    for m, rows in per_arch_earliness.items():
        d = {r["gate"]: r["earliness_delta_frac"] for r in rows}
        latent = d.get("latent")
        baselines = [d.get("realm_sim"), d.get("grip_sim")]
        latent_wins = (
            latent is not None
            and all((b is not None and latent > b) for b in baselines)
        )
        cells = [d.get(g) for g in GATE_ORDER]
        cell_strs = "  ".join(_fmt(x, "{:+6.3f}") for x in cells)
        print(f"  {m:32}  {cell_strs}  | {'YES' if latent_wins else 'no '}")
    print(
        "\n  (positive delta_frac = gate fires earlier in the trace on incorrect"
        "\n   answers than on correct ones. The latent gate winning on every arch"
        "\n   is the architecture-agnostic earliness claim.)"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
