"""Phase 2 — CRV signal visualisation.

Reads per-step JSONLs produced by `02_crv_probe.py` and renders three plots:

  figs/phase2_drift_per_arch.png
      Cosine drift (vs previous step's CRV) over the trace, faceted by
      architecture (3 rows) and prompt class (3 cols overlaid). One panel
      per architectural class — the comparison the EMNLP framing wants.

  figs/phase2_variance_per_arch.png
      Same layout for rolling L2-norm variance.

  figs/phase2_signal_summary.png
      Single panel: per-architecture mean drift + mean variance across
      the trace, with prompt-class spread as error bars. The "headline
      number" version of Phase 2 — does the latent-state signal exist
      uniformly across architectures?

Usage:
    python scripts/plot_phase2.py
    python scripts/plot_phase2.py --boundary sentence --layer-mode last_layer
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


# (label, color, model_dir_name) tuples — same order as Phase 1.4 plot.
SOURCES: list[tuple[str, str, str]] = [
    ("Phi-4-mini-instruct (Dense, full GQA)", "#1f77b4", "phi-4-mini-instruct"),
    ("Gemma 4 26B A4B (MoE, sliding+global)", "#ff7f0e", "gemma-4-26b-a4b-it"),
    ("Qwen3.6-35B-A3B (MoE, 3:1 Gated DeltaNet)", "#2ca02c", "qwen3.6-35b-a3b-instruct"),
]
PROMPT_CLASSES = ("factual", "multi-hop", "ambiguous")
PROMPT_LINESTYLES = {"factual": "-", "multi-hop": "--", "ambiguous": ":"}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def jsonl_path(model: str, prompt: str, boundary: str, layer_mode: str) -> Path:
    return Path("runs") / f"phase2_crv_{model}_{prompt}_{boundary}_{layer_mode}.jsonl"


def plot_drift(out_path: Path, boundary: str, layer_mode: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=False)
    skipped = []
    for ax, (label, color, model) in zip(axes, SOURCES):
        for prompt in PROMPT_CLASSES:
            rows = load_jsonl(jsonl_path(model, prompt, boundary, layer_mode))
            if not rows:
                skipped.append((label, prompt))
                continue
            xs = [r["token_idx_end"] for r in rows]
            ys = [r["drift_cosine"] for r in rows]
            ax.plot(
                xs, ys,
                color=color, linestyle=PROMPT_LINESTYLES[prompt],
                marker=".", markersize=4, linewidth=1.6,
                label=prompt,
                alpha=0.85,
            )
        ax.set_title(label, fontsize=11, loc="left")
        ax.set_ylabel("drift_cosine\n(1 − cos sim vs prev CRV)", fontsize=9)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
        ax.set_ylim(bottom=0)
    axes[-1].set_xlabel("token index (end of logical step)")
    fig.suptitle(
        f"Phase 2 — CRV cosine drift across the 5 000-token trace\n"
        f"boundary={boundary}, layer_mode={layer_mode}, last-layer mean-pooled CRV",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[ok] {out_path}")
    if skipped:
        print(f"  [missing data] {skipped}")


def plot_variance(out_path: Path, boundary: str, layer_mode: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=False)
    for ax, (label, color, model) in zip(axes, SOURCES):
        for prompt in PROMPT_CLASSES:
            rows = load_jsonl(jsonl_path(model, prompt, boundary, layer_mode))
            if not rows:
                continue
            xs = [r["token_idx_end"] for r in rows]
            ys = [r["variance_l2_window"] for r in rows]
            ax.plot(
                xs, ys,
                color=color, linestyle=PROMPT_LINESTYLES[prompt],
                marker=".", markersize=4, linewidth=1.6,
                label=prompt,
                alpha=0.85,
            )
        ax.set_title(label, fontsize=11, loc="left")
        ax.set_ylabel("rolling variance of\nCRV L2 norm (window=10)", fontsize=9)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
        ax.set_ylim(bottom=0)
    axes[-1].set_xlabel("token index (end of logical step)")
    fig.suptitle(
        f"Phase 2 — CRV L2-norm variance across the 5 000-token trace\n"
        f"boundary={boundary}, layer_mode={layer_mode}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[ok] {out_path}")


def plot_signal_summary(out_path: Path, boundary: str, layer_mode: str) -> None:
    """Per-architecture mean drift + mean variance over the trace, error
    bars = ±1 σ across prompt classes. Single-panel headline figure."""
    fig, (ax_d, ax_v) = plt.subplots(1, 2, figsize=(11, 4.5))
    arch_labels = []
    drift_means = []
    drift_stds = []
    var_means = []
    var_stds = []
    colors = []
    for label, color, model in SOURCES:
        per_prompt_drift = []
        per_prompt_var = []
        for prompt in PROMPT_CLASSES:
            rows = load_jsonl(jsonl_path(model, prompt, boundary, layer_mode))
            if not rows:
                continue
            # Skip step 1 — drift is 0 there by definition.
            ds = [r["drift_cosine"] for r in rows[1:]]
            vs = [r["variance_l2_window"] for r in rows[1:]]
            if ds:
                per_prompt_drift.append(float(np.mean(ds)))
            if vs:
                per_prompt_var.append(float(np.mean(vs)))
        if not per_prompt_drift:
            continue
        arch_labels.append(label.split(" (")[0])
        colors.append(color)
        drift_means.append(np.mean(per_prompt_drift))
        drift_stds.append(np.std(per_prompt_drift))
        var_means.append(np.mean(per_prompt_var))
        var_stds.append(np.std(per_prompt_var))

    xs = np.arange(len(arch_labels))
    ax_d.bar(xs, drift_means, yerr=drift_stds, color=colors, alpha=0.82, capsize=6)
    ax_d.set_xticks(xs)
    ax_d.set_xticklabels(arch_labels, rotation=14, ha="right")
    ax_d.set_ylabel("mean drift_cosine over trace")
    ax_d.set_title("Drift signal magnitude (Phase 2.5 prerequisite)")
    ax_d.grid(alpha=0.25, axis="y")

    ax_v.bar(xs, var_means, yerr=var_stds, color=colors, alpha=0.82, capsize=6)
    ax_v.set_xticks(xs)
    ax_v.set_xticklabels(arch_labels, rotation=14, ha="right")
    ax_v.set_ylabel("mean rolling L2-variance over trace")
    ax_v.set_title("Variance signal magnitude")
    ax_v.grid(alpha=0.25, axis="y")

    fig.suptitle(
        f"Phase 2 — does a CRV signal *exist* on every architectural class?\n"
        f"(error bars = ±1 σ across {len(PROMPT_CLASSES)} prompt classes; bars are mean post step 1)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[ok] {out_path}")


def print_headline(boundary: str, layer_mode: str) -> None:
    print()
    print("=" * 92)
    print(
        f"Phase 2 — per-architecture CRV signal summary  (boundary={boundary}, layer_mode={layer_mode})"
    )
    print("=" * 92)
    print(
        f"  {'model':<46}  {'mean drift':>11}  {'mean variance':>13}  {'n_steps':>8}"
    )
    print("-" * 92)
    for label, _color, model in SOURCES:
        all_drifts = []
        all_vars = []
        n_steps = 0
        for prompt in PROMPT_CLASSES:
            rows = load_jsonl(jsonl_path(model, prompt, boundary, layer_mode))
            if not rows:
                continue
            ds = [r["drift_cosine"] for r in rows[1:]]
            vs = [r["variance_l2_window"] for r in rows[1:]]
            all_drifts.extend(ds)
            all_vars.extend(vs)
            n_steps += len(rows)
        if not all_drifts:
            print(f"  {label:<46}  (no data)")
            continue
        print(
            f"  {label:<46}  {np.mean(all_drifts):>11.4f}  "
            f"{np.mean(all_vars):>13.4f}  {n_steps:>8}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boundary", default="sentence", choices=["sentence", "think_block", "fixed"])
    parser.add_argument("--layer-mode", default="last_layer", choices=["last_layer", "mid_stack", "last_and_mid"])
    parser.add_argument("--out-drift", default="figs/phase2_drift_per_arch.png")
    parser.add_argument("--out-variance", default="figs/phase2_variance_per_arch.png")
    parser.add_argument("--out-summary", default="figs/phase2_signal_summary.png")
    args = parser.parse_args()

    plot_drift(Path(args.out_drift), args.boundary, args.layer_mode)
    plot_variance(Path(args.out_variance), args.boundary, args.layer_mode)
    plot_signal_summary(Path(args.out_summary), args.boundary, args.layer_mode)
    print_headline(args.boundary, args.layer_mode)


if __name__ == "__main__":
    main()
