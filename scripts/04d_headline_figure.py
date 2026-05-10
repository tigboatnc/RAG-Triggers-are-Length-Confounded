"""Geometry-aware headline figure: arch x gate, Δfrac with 95 % bootstrap CIs.

Single panel. Each architecture is a group of 6 bars: drift, variance, OR
(the three latent modes from the 4.4(a) ablation), then ReaLM, GRIP, random.
The calibration-chosen latent mode per arch is highlighted (hatched + bold
edge). 95 % bootstrap CIs are drawn as error bars.

Reads:
    runs/phase4_showdown<suffix>_summary.json   (chosen mode per arch + thresholds + ablation)
    runs/phase4_bootstrap<suffix>.json          (95 % CIs for the latent + 3 baselines)

Writes:
    figs/phase4_headline_geometry_aware.png
    figs/phase4_headline_geometry_aware.pdf  (vector for the camera-ready)

We compute on-the-fly bootstrap CIs for the *non-chosen* latent modes
(drift/variance/or) so the figure shows uncertainty for every bar, not just
the chosen-mode latent + 3 baselines.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module
sd = import_module("04_showdown")
boot = import_module("04c_bootstrap")


NICE = {
    "phi-4-mini-instruct": "Phi-4-mini\n(dense GQA)",
    "gemma-4-26b-a4b-it": "Gemma-4-26B-A4B\n(sliding-window MoE)",
    "qwen3.6-35b-a3b-instruct": "Qwen3.6-35B-A3B\n(hybrid Gated DeltaNet MoE)",
}

GATE_LABELS = ["latent\ndrift", "latent\nvariance", "latent\nOR", "ReaLM-sim", "GRIP-sim", "random"]
GATE_COLORS = ["#4c72b0", "#4c72b0", "#4c72b0", "#dd8452", "#55a467", "#8172b3"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--suffix", default="_v2")
    p.add_argument("--bootstrap", "-B", type=int, default=10000)
    p.add_argument("--min-steps", type=int, default=3)
    p.add_argument("--seed", type=int, default=20260510)
    p.add_argument("--out-stem", default="figs/phase4_headline_geometry_aware")
    args = p.parse_args()

    summary = json.loads(Path(f"runs/phase4_showdown{args.suffix}_summary.json").read_text())["per_arch"]
    archs = list(summary.keys())

    rng = random.Random(args.seed)

    # Per-(arch, gate-bar) compute point + CI.
    cells: dict[str, dict[str, dict[str, float]]] = {}
    for m in archs:
        rows = sd._load_traces(m, "mid_stack", args.suffix)
        valid = [r for r in rows if r.get("split") == "validation"]
        thresholds = summary[m]["thresholds"]

        cells[m] = {}
        # Three latent modes.
        for mode in ("drift", "variance", "or"):
            pt = sd._firsts_per_trace(valid, thresholds, latent_mode=mode)
            elig = [t for t in pt if t["n_steps"] >= args.min_steps]
            ci = boot.bootstrap_ci(elig, "latent", args.bootstrap, rng)
            cells[m][f"latent_{mode}"] = ci

        # Baselines: use the OR-mode trace (gate decisions are independent of latent mode).
        pt = sd._firsts_per_trace(valid, thresholds, latent_mode="or")
        elig = [t for t in pt if t["n_steps"] >= args.min_steps]
        for g in ("realm_sim", "grip_sim", "random"):
            cells[m][g] = boot.bootstrap_ci(elig, g, args.bootstrap, rng)

    # Plot.
    fig, ax = plt.subplots(figsize=(13, 6))
    n_arch = len(archs)
    n_bars = 6
    bar_w = 0.13
    x_arch = np.arange(n_arch)

    bar_keys = ["latent_drift", "latent_variance", "latent_or", "realm_sim", "grip_sim", "random"]

    for bi, (key, label, color) in enumerate(zip(bar_keys, GATE_LABELS, GATE_COLORS)):
        xs = x_arch + (bi - (n_bars - 1) / 2) * bar_w
        ys = []
        err_lo = []
        err_hi = []
        edgecolors = []
        hatches = []
        for m in archs:
            ci = cells[m][key]
            ys.append(ci["point"] if ci["point"] is not None else 0.0)
            err_lo.append((ci["point"] - ci["ci_lo"]) if ci["ci_lo"] is not None else 0.0)
            err_hi.append((ci["ci_hi"] - ci["point"]) if ci["ci_hi"] is not None else 0.0)
            chosen = summary[m].get("latent_mode_chosen")
            is_chosen = (key == f"latent_{chosen}")
            edgecolors.append("black" if is_chosen else "none")
            hatches.append("///" if is_chosen else "")

        bars = ax.bar(
            xs, ys, bar_w,
            color=color, label=label,
            yerr=[err_lo, err_hi], capsize=3, ecolor="#333",
            edgecolor=edgecolors, linewidth=[2.0 if e == "black" else 0.0 for e in edgecolors],
        )
        for b, h in zip(bars, hatches):
            if h:
                b.set_hatch(h)

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x_arch)
    ax.set_xticklabels([NICE[m] for m in archs], fontsize=10)
    ax.set_ylabel(r"$\Delta_{\mathrm{frac}}$  (mean first-fire fraction: correct $-$ incorrect; positive = fires earlier on incorrect)", fontsize=10)
    ax.set_title(
        "Geometry-aware latent gating on HotpotQA-distractor (validation, n=40 unless noted) — 95% bootstrap CIs\n"
        "Hatched bar = calibration-chosen latent mode per architecture (no validation leakage).",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=9, ncol=3, framealpha=0.95)

    # Annotate sample sizes.
    for i, m in enumerate(archs):
        n = summary[m]["cost"]["n_eligible"]
        ax.text(i, ax.get_ylim()[1] * 0.92, f"n={n}", ha="center", fontsize=9, color="#444")

    plt.tight_layout()
    out_png = Path(f"{args.out_stem}.png")
    out_pdf = Path(f"{args.out_stem}.pdf")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_pdf)
    print(f"[ok] wrote {out_png}")
    print(f"[ok] wrote {out_pdf}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
