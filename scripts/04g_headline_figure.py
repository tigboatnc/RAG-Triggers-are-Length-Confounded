"""Headline figure for the new {latent, ETC, SELFDOUBT-T1, DRAGIN, random} pool.

Reads `runs/phase4_bootstrap_dragin.json` (produced by `04g_sota_showdown.py`)
and renders a two-panel figure:

    Panel A — Δfrac per gate, per arch (validation), with 95% CIs.
              The calibration-chosen latent mode is hatched + black-edged.
    Panel B — Paired margins `latent − baseline` per arch (validation),
              with 95% CIs. This is the punchline panel.

A single-panel test-split companion is also written to a separate filename.

The old `04d_headline_figure.py` (simulator baselines) is left in place
for the audit trail.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


NICE = {
    "phi-4-mini-instruct": "Phi-4-mini\n(dense GQA)",
    "gemma-4-26b-a4b-it": "Gemma-4-26B-A4B\n(sliding-window MoE)",
    "qwen3.6-35b-a3b-instruct": "Qwen3.6-35B-A3B\n(hybrid Gated DeltaNet MoE)",
}

POOL = ["latent", "etc", "selfdoubt_t1", "dragin", "random"]
POOL_LABELS = {
    "latent": "latent\n(calib-chosen)",
    "etc": "ETC",
    "selfdoubt_t1": "SELFDOUBT-T1",
    "dragin": "DRAGIN",
    "random": "random\n(rate-matched)",
}
POOL_COLORS = {
    "latent": "#4c72b0",
    "etc": "#dd8452",
    "selfdoubt_t1": "#55a467",
    "dragin": "#c44e52",
    "random": "#8172b3",
}
MARGIN_KEYS = ["etc", "selfdoubt_t1", "dragin", "random"]


def _err(ci: dict) -> tuple[float, float, float]:
    """Return (point, err_lo, err_hi) suitable for matplotlib `yerr`.

    Bars with missing CIs (n/a Δfrac, e.g. SELFDOUBT-T1) collapse to zero so
    they still show up as a flat bar with no error bar — fire% annotation
    above carries the substantive result.
    """
    p = ci.get("point")
    lo = ci.get("ci_lo")
    hi = ci.get("ci_hi")
    if p is None:
        return 0.0, 0.0, 0.0
    el = (p - lo) if lo is not None else 0.0
    eh = (hi - p) if hi is not None else 0.0
    return p, max(el, 0.0), max(eh, 0.0)


def _panel_per_gate(ax, archs, per_arch, split_key="validation"):
    n_arch = len(archs)
    n_bars = len(POOL)
    bar_w = 0.15
    x = np.arange(n_arch)
    for bi, gate in enumerate(POOL):
        xs = x + (bi - (n_bars - 1) / 2) * bar_w
        ys, el, eh, edgecolor, hatches, fire_pcts, point_isna = [], [], [], [], [], [], []
        for m in archs:
            cell = per_arch[m][split_key]
            ci = cell["per_gate"][gate]
            p, lo, hi = _err(ci)
            ys.append(p); el.append(lo); eh.append(hi)
            chosen = per_arch[m].get("latent_mode_chosen")
            is_chosen_latent = (gate == "latent")
            edgecolor.append("black" if is_chosen_latent else "none")
            hatches.append("///" if is_chosen_latent else "")
            fire_pcts.append(ci.get("trace_fire_rate", 0.0) * 100)
            point_isna.append(ci.get("point") is None)
        bars = ax.bar(
            xs, ys, bar_w,
            color=POOL_COLORS[gate], label=POOL_LABELS[gate],
            yerr=[el, eh], capsize=3, ecolor="#333",
            edgecolor=edgecolor, linewidth=[2.0 if e == "black" else 0.0 for e in edgecolor],
        )
        for b, h in zip(bars, hatches):
            if h:
                b.set_hatch(h)
        # Fire-% annotation above each bar (especially informative for SELFDOUBT-T1 @ ~0%).
        for xi, y, fr, isna in zip(xs, ys, fire_pcts, point_isna):
            tag = f"{fr:.0f}%"
            if isna:
                tag += "*"
            # Place above the bar top (or above 0-line for n/a bars at 0).
            y_anchor = max(y, 0.0)
            ax.text(xi, y_anchor + 0.012, tag, ha="center", va="bottom",
                    fontsize=6.5, color="#333")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([NICE[m] for m in archs], fontsize=9)
    ax.set_ylabel(
        r"$\Delta_{\mathrm{frac}}$"
        "\n(positive = fires earlier on incorrect)",
        fontsize=9,
    )
    ax.legend(loc="lower left", fontsize=7.5, ncol=5, framealpha=0.95,
              bbox_to_anchor=(0.0, 1.02))


def _panel_margins(ax, archs, per_arch, split_key="validation"):
    n_arch = len(archs)
    n_bars = len(MARGIN_KEYS)
    bar_w = 0.18
    x = np.arange(n_arch)
    for bi, opp in enumerate(MARGIN_KEYS):
        xs = x + (bi - (n_bars - 1) / 2) * bar_w
        ys, el, eh = [], [], []
        for m in archs:
            mci = per_arch[m][split_key]["margins"][f"latent_minus_{opp}"]
            p, lo, hi = _err(mci)
            ys.append(p); el.append(lo); eh.append(hi)
        ax.bar(
            xs, ys, bar_w,
            color=POOL_COLORS[opp], label=f"latent − {POOL_LABELS[opp].splitlines()[0]}",
            yerr=[el, eh], capsize=3, ecolor="#333",
            edgecolor="#222", linewidth=0.4,
        )
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([NICE[m] for m in archs], fontsize=9)
    ax.set_ylabel(
        r"paired margin"
        "\n(latent $-$ baseline $\\Delta_{\\mathrm{frac}}$)",
        fontsize=9,
    )
    ax.legend(loc="lower left", fontsize=7.5, ncol=4, framealpha=0.95,
              bbox_to_anchor=(0.0, 1.02))


def render_figure(per_arch, archs, split_key, title, out_stem):
    fig, (axA, axB) = plt.subplots(2, 1, figsize=(11.5, 9.0), sharex=True,
                                    gridspec_kw={"hspace": 0.35})
    _panel_per_gate(axA, archs, per_arch, split_key=split_key)
    _panel_margins(axB, archs, per_arch, split_key=split_key)

    # Chosen-mode annotations on Panel A: inside the panel near the bottom of
    # the y-range so they don't collide with the (now bbox-anchored) legend.
    y_lo = axA.get_ylim()[0]
    for i, m in enumerate(archs):
        mode = per_arch[m].get("latent_mode_chosen", "?")
        n_val = per_arch[m].get("n_validation_eligible", "?")
        axA.text(
            i, y_lo + 0.02,
            f"calib-chosen latent mode: {mode}   |   n_val_elig={n_val}",
            ha="center", fontsize=7, color="#444", style="italic",
        )

    fig.suptitle(title, fontsize=11, y=0.995)
    fig.text(
        0.01, 0.005,
        "Hatched bar = calibration-chosen latent aggregation (drift/variance/OR); "
        "% above each Panel-A bar is per-trace fire rate ('*' marks $\\Delta_{\\mathrm{frac}}$=n/a, "
        "no fires in one bucket). All thresholds frozen on the calibration split "
        "(no validation/test leakage).",
        fontsize=7, color="#555",
    )
    plt.tight_layout(rect=(0, 0.02, 1, 0.97))

    out_png = Path(f"{out_stem}.png")
    out_pdf = Path(f"{out_stem}.pdf")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"[ok] wrote {out_png}")
    print(f"[ok] wrote {out_pdf}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bootstrap-json", default="runs/phase4_bootstrap_dragin.json")
    p.add_argument("--out-stem", default="figs/phase4_headline_dragin")
    p.add_argument("--out-stem-test", default="figs/phase4_headline_dragin_test")
    args = p.parse_args()

    payload = json.loads(Path(args.bootstrap_json).read_text())
    per_arch = payload["per_arch"]
    archs = [m for m in [
        "phi-4-mini-instruct", "gemma-4-26b-a4b-it", "qwen3.6-35b-a3b-instruct",
    ] if m in per_arch]

    render_figure(
        per_arch, archs, split_key="validation",
        title=(
            "Geometry-aware latent gating vs training-free SOTA — HotpotQA-distractor, "
            "validation split (95% bootstrap CIs, B=10k)"
        ),
        out_stem=args.out_stem,
    )

    if all("test" in per_arch[m] for m in archs):
        render_figure(
            per_arch, archs, split_key="test",
            title=(
                "Single-shot test-split replication (no re-tuning; same frozen "
                "thresholds + chosen mode as validation)"
            ),
            out_stem=args.out_stem_test,
        )

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
