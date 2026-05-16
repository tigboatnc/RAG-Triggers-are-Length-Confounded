"""Survivor forest plot — the single selling figure.

Parses `analysis/e7_bonferroni.out` and renders a horizontal forest plot:
    y-axis = trigger signals, ranked by Δ point estimate (descending)
    x-axis = Δ = AUROC(s_resid) − AUROC(tokens), range ≈ [−0.15, +0.30]
    per row: filled point at Δ; thick bar at 95% CI; thin extension at Bonferroni 99.67% CI
    vertical dashed line at Δ=0 (length baseline)
    background banding: top 4 = Bonferroni-robust (white); middle 5 = nominal-95% only (light grey);
        bottom 8 = not significant (darker grey)
    color: step-level answer-trajectory family = blue; attention (DRAGIN) = red;
        logit + K=5-text published-pool = orange; everything else (hidden-state + text) = grey

Output: figs/survivor_forest.pdf (vector) and figs/survivor_forest.png (raster).

Usage:
    python analysis/plot_survivor_forest.py
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT_TXT = HERE / "e7_bonferroni.out"
FIG_DIR = ROOT / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Signal family map (drives color)
# --------------------------------------------------------------------------- #
ANSWER_TRAJECTORY = {
    "AID flip_rate K=5", "AID flip_rate K=1", "AID late_flips", "AID min_majority",
    "M1 CABV", "M8 ECT", "M9 CABV×LAAS",
}
ATTENTION = {"DRAGIN-RIND.max"}
LOGIT_PUBLISHED = {"ent.max", "etc (Δ²H max)", "flare.max (1-min p)", "u_cons.max"}
# everything else = hidden-state / text (grey)

# Display-name swap so figure matches the cleaned-up .tex (no "AID", no "M8")
DISPLAY = {
    "AID flip_rate K=5": "flip_rate K=5",
    "AID flip_rate K=1": "flip_rate K=1",
    "AID late_flips":    "late_flips",
    "AID min_majority":  "min_majority",
    "M8 ECT":            "ECT",
    "M9 CABV×LAAS":      "CABV × LAAS",
    "M1 CABV":           "CABV",
    "etc (Δ²H max)":     "ent.curv (Δ²H)",
    "flare.max (1-min p)": "flare.max",
    "hvr.max (SELFDOUBT-T1)": "hvr.max",
    "seakr_k5_proxy.logdet":  "seakr-K=5.logdet",
    "variance_l2_window.max": "var_l2_window.max",
}


def color_for(sig: str) -> str:
    if sig in ANSWER_TRAJECTORY:
        return "#1f77b4"  # blue — answer-trajectory class (the highlight)
    if sig in ATTENTION:
        return "#d62728"  # red — attention (DRAGIN)
    if sig in LOGIT_PUBLISHED:
        return "#ff7f0e"  # orange — logit / K=5-text published pool
    return "#7f7f7f"      # grey — hidden-state / text


# --------------------------------------------------------------------------- #
# Parse e7_bonferroni.out
# --------------------------------------------------------------------------- #
ROW_RE = re.compile(
    r"^\s+(\S.*?)\s{2,}"                              # signal
    r"([\d\.]+)\s+"                                    # resAUC
    r"([+\-][\d\.]+)\s+"                               # Δ
    r"\[([+\-][\d\.]+),([+\-][\d\.]+)\]\s+"            # 95% CI lo, hi
    r"\[([+\-][\d\.]+),([+\-][\d\.]+)\]\s+"            # 99.67% CI lo, hi
    r"(.+?)\s*$"                                       # tail (sig flags)
)


def parse() -> list[dict]:
    rows = []
    for ln in OUT_TXT.read_text().splitlines():
        m = ROW_RE.match(ln)
        if not m:
            continue
        signal = m.group(1).strip()
        delta = float(m.group(3))
        lo95, hi95 = float(m.group(4)), float(m.group(5))
        loB, hiB = float(m.group(6)), float(m.group(7))
        tail = m.group(8)
        sig95 = "✓" in tail
        sigB = "✓✓" in tail
        rows.append({
            "signal": signal,
            "delta": delta,
            "lo95": lo95, "hi95": hi95,
            "loB": loB, "hiB": hiB,
            "sig95": sig95, "sigB": sigB,
        })
    return rows


# --------------------------------------------------------------------------- #
# Build figure
# --------------------------------------------------------------------------- #
def main():
    rows = parse()
    # Sort by Δ descending so highest survivor sits at the top.
    rows = sorted(rows, key=lambda r: r["delta"], reverse=True)
    n = len(rows)

    fig_w, fig_h = 6.6, 5.6  # single-column EMNLP-short friendly (slightly taller)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)

    y_positions = np.arange(n)[::-1]  # row 0 at top
    y_labels = [DISPLAY.get(r["signal"], r["signal"]) for r in rows]

    # Background banding by significance tier ------------------------------- #
    bonf_idxs = [i for i, r in enumerate(rows) if r["sigB"]]
    nom_idxs  = [i for i, r in enumerate(rows) if r["sig95"] and not r["sigB"]]
    ns_idxs   = [i for i, r in enumerate(rows) if not r["sig95"]]

    def band(idxs, color, alpha):
        if not idxs:
            return
        # y_positions are reversed; idxs are forward
        ymax = max(y_positions[i] for i in idxs) + 0.5
        ymin = min(y_positions[i] for i in idxs) - 0.5
        ax.axhspan(ymin, ymax, color=color, alpha=alpha, zorder=0)

    band(bonf_idxs, "#ffffff", 0.0)            # white — no shading needed
    band(nom_idxs,  "#cccccc", 0.18)
    band(ns_idxs,   "#aaaaaa", 0.28)

    # Bonferroni CI (thin, extends beyond 95%) ------------------------------ #
    for i, r in enumerate(rows):
        y = y_positions[i]
        c = color_for(r["signal"])
        ax.plot([r["loB"], r["hiB"]], [y, y],
                color=c, alpha=0.40, linewidth=1.0, zorder=2)

    # 95% CI bar (thick) ---------------------------------------------------- #
    for i, r in enumerate(rows):
        y = y_positions[i]
        c = color_for(r["signal"])
        ax.plot([r["lo95"], r["hi95"]], [y, y],
                color=c, alpha=0.95, linewidth=3.0, zorder=3)

    # Point estimate -------------------------------------------------------- #
    for i, r in enumerate(rows):
        y = y_positions[i]
        c = color_for(r["signal"])
        # Different marker for Bonferroni-robust = filled square; others = circle
        marker = "s" if r["sigB"] else "o"
        ms = 7 if r["sigB"] else 6
        ax.scatter([r["delta"]], [y], color=c, marker=marker, s=ms*ms,
                   zorder=5, edgecolor="white", linewidth=0.6)

    # Reference line at Δ=0 (length baseline) ------------------------------- #
    ax.axvline(0, color="black", linestyle="--", linewidth=0.9,
               alpha=0.65, zorder=1)
    # Annotation BELOW the bars, just above the x-axis ticks
    ax.text(0.005, -0.55, r"length baseline",
            ha="left", va="center", fontsize=7.5, color="black",
            alpha=0.75, style="italic")

    # Bonferroni-robust separator (horizontal line under top 4) ------------- #
    n_bonf = len([r for r in rows if r["sigB"]])
    n_nom = len([r for r in rows if r["sig95"] and not r["sigB"]])
    if n_bonf > 0 and n_bonf < n:
        ax.axhline(y_positions[n_bonf - 1] - 0.5,
                   color="#444444", linestyle="-", linewidth=0.6, alpha=0.7)
    if n_nom > 0 and (n_bonf + n_nom) < n:
        ax.axhline(y_positions[n_bonf + n_nom - 1] - 0.5,
                   color="#444444", linestyle=":", linewidth=0.6, alpha=0.5)

    # Tier annotations (right margin, midpoint of tier) --------------------- #
    if n_bonf > 0:
        mid_bonf = (y_positions[0] + y_positions[n_bonf - 1]) / 2.0
        ax.text(0.325, mid_bonf, "Bonferroni-\nrobust",
                fontsize=7.0, ha="left", va="center",
                style="italic", color="#222222", fontweight="bold")
    if n_nom > 0:
        mid_nom = (y_positions[n_bonf] + y_positions[n_bonf + n_nom - 1]) / 2.0
        ax.text(0.325, mid_nom, "Nominal-\n95% only",
                fontsize=7.0, ha="left", va="center",
                style="italic", color="#444444")
    if n - n_bonf - n_nom > 0:
        mid_ns = (y_positions[n_bonf + n_nom] + y_positions[-1]) / 2.0
        ax.text(0.325, mid_ns, "Does not\nclear",
                fontsize=7.0, ha="left", va="center",
                style="italic", color="#666666")

    # Axes ------------------------------------------------------------------ #
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlim(-0.18, 0.32)
    ax.set_xlabel(r"$\Delta = $ AUROC$(s_{\mathrm{resid}}) - $ AUROC(tokens)",
                  fontsize=10)
    ax.set_xticks(np.arange(-0.15, 0.31, 0.05))
    ax.tick_params(axis="x", labelsize=8)
    ax.set_ylim(-0.7, n - 0.3)
    ax.grid(axis="x", linestyle=":", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    # Legend (family color key) --------------------------------------------- #
    legend_handles = [
        mpatches.Patch(color="#1f77b4", label="answer-trajectory (Datta-derived)"),
        mpatches.Patch(color="#d62728", label="attention (DRAGIN)"),
        mpatches.Patch(color="#ff7f0e", label="logit / $K{=}5$-text"),
        mpatches.Patch(color="#7f7f7f", label="hidden-state / text"),
    ]
    # Place legend ABOVE the plot to avoid overlapping bars
    ax.legend(handles=legend_handles,
              loc="lower center",
              bbox_to_anchor=(0.5, 1.02),
              ncol=2, fontsize=7.5, framealpha=0.9, frameon=True,
              borderpad=0.5, handlelength=1.2,
              columnspacing=1.2)

    # Title (figure title above the legend) --------------------------------- #
    fig.suptitle(
        r"Length-orthogonal trigger signals "
        r"(Qwen+Gemma, $n{=}800$, $B{=}10{,}000$, target $\mathrm{lift\_em}{=}{+}1$)",
        fontsize=9.5, y=0.995, style="italic")

    plt.tight_layout()
    out_pdf = FIG_DIR / "survivor_forest.pdf"
    out_png = FIG_DIR / "survivor_forest.png"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, bbox_inches="tight", dpi=200)
    print(f"[ok] wrote {out_pdf.relative_to(ROOT)}")
    print(f"[ok] wrote {out_png.relative_to(ROOT)}")
    print(f"     n_signals={n}  Bonferroni-robust={n_bonf}  nominal-only={n_nom}")


if __name__ == "__main__":
    main()
