"""Phase 1.4 — three-way architectural-class comparison plot.

Reads checkpoint JSONLs from three Phase 1 runs (one per architectural class):
  - Phi-4-mini-instruct      (Dense SLM, full GQA attention)              — runs/phase1_4_phi.jsonl
  - Gemma 4 26B A4B          (Standard MoE, sliding-window + sparse global) — runs/phase1_4_gemma4_26b.jsonl
  - Qwen3.6-35B-A3B          (Hybrid Linear MoE, 3:1 Gated DeltaNet)        — runs/phase1_kv_pressure.jsonl  (Phase 1.3 data)

For each model, averages across the three prompt classes (factual / multi-hop /
ambiguous) at each 500-token checkpoint and plots one line per model with a
shaded ±1 σ band. Two stacked panels: VRAM-vs-tokens (top), rolling tok/s
(bottom). Saved to figs/phase1_4_arch_comparison.png.

This is the EMNLP-2026 bracketing plot: dense full attention (steep linear
O(N) reference) vs sliding-window+sparse-global (knee at sliding_window cap)
vs hybrid linear attention (essentially flat). Three-class spectrum.

Usage:
    python scripts/plot_phase1_4.py
    python scripts/plot_phase1_4.py --include-prompt-class-detail   # also save
                                                                     # per-prompt-class
                                                                     # detail plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


SOURCES: list[tuple[str, str, str, str]] = [
    # (label, color, jsonl_path, architectural-class subtitle)
    (
        "Phi-4-mini-instruct (Dense, full GQA)",
        "#1f77b4",
        "runs/phase1_4_phi.jsonl",
        "Dense SLM",
    ),
    (
        "Gemma 4 26B A4B (MoE, sliding+global)",
        "#ff7f0e",
        "runs/phase1_4_gemma4_26b.jsonl",
        "Standard MoE",
    ),
    (
        "Qwen3.6-35B-A3B (MoE, 3:1 Gated DeltaNet)",
        "#2ca02c",
        "runs/phase1_kv_pressure.jsonl",
        "Hybrid Linear MoE",
    ),
]


def load_rows(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def aggregate_by_step(rows: list[dict], field: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per checkpoint step, return (steps, mean, std) across prompt classes."""
    by_step: dict[int, list[float]] = {}
    for r in rows:
        step = int(r["step_idx"])
        by_step.setdefault(step, []).append(float(r[field]))
    steps = sorted(by_step.keys())
    means = np.array([np.mean(by_step[s]) for s in steps])
    stds = np.array([np.std(by_step[s]) for s in steps])
    return np.array(steps), means, stds


def plot_three_way(out_path: Path, sources: Iterable[tuple[str, str, str, str]]) -> None:
    fig, (ax_v, ax_t) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    skipped = []
    for label, color, jsonl_path, _arch in sources:
        rows = load_rows(jsonl_path)
        if not rows:
            skipped.append((label, jsonl_path))
            continue
        # VRAM (GiB)
        steps, vram_mean, vram_std = aggregate_by_step(rows, "vram_allocated_mb")
        vram_mean_gib = vram_mean / 1024.0
        vram_std_gib = vram_std / 1024.0
        ax_v.plot(steps, vram_mean_gib, marker="o", color=color, label=label, linewidth=2)
        ax_v.fill_between(
            steps,
            vram_mean_gib - vram_std_gib,
            vram_mean_gib + vram_std_gib,
            color=color, alpha=0.18, linewidth=0,
        )
        # Rolling tok/s
        steps_t, tps_mean, tps_std = aggregate_by_step(rows, "rolling_tok_per_s")
        ax_t.plot(steps_t, tps_mean, marker="o", color=color, label=label, linewidth=2)
        ax_t.fill_between(
            steps_t,
            tps_mean - tps_std,
            tps_mean + tps_std,
            color=color, alpha=0.18, linewidth=0,
        )

    ax_v.set_ylabel("VRAM allocated (GiB)")
    ax_v.set_title(
        "Phase 1.4 — KV-pressure across the 2026 attention-design spectrum\n"
        "5 000-token uninterrupted greedy generation, mean ±1σ across 3 prompt classes, A800 80 GB bf16"
    )
    ax_v.grid(alpha=0.3)
    ax_v.legend(loc="upper left", framealpha=0.92)

    ax_t.set_ylabel("Rolling tok/s (over 500-token window)")
    ax_t.set_xlabel("Tokens generated")
    ax_t.grid(alpha=0.3)
    ax_t.legend(loc="upper right", framealpha=0.92)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    print(f"[ok] plot written to {out_path}")
    if skipped:
        print(f"[!!] skipped (jsonl missing): {skipped}")


def plot_kv_growth_normalized(out_path: Path, sources: Iterable[tuple[str, str, str, str]]) -> None:
    """KV growth above each model's at-step-500 baseline. Same shape regardless
    of weight footprint, so the curves are directly comparable as memory-pressure
    signatures across architectures."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for label, color, jsonl_path, _arch in sources:
        rows = load_rows(jsonl_path)
        if not rows:
            continue
        steps, vram_mean, vram_std = aggregate_by_step(rows, "vram_allocated_mb")
        # Normalize: subtract step 500 baseline → "growth since first checkpoint"
        baseline = vram_mean[0]
        growth = vram_mean - baseline
        growth_std = vram_std
        ax.plot(steps, growth, marker="o", color=color, label=label, linewidth=2)
        ax.fill_between(
            steps,
            growth - growth_std,
            growth + growth_std,
            color=color, alpha=0.18, linewidth=0,
        )
    ax.set_xlabel("Tokens generated")
    ax.set_ylabel("VRAM growth since step 500 (MB)")
    ax.set_title(
        "Phase 1.4 — KV-pressure growth signature\n"
        "Normalised at step 500, mean ±1σ across 3 prompt classes"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.92)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    print(f"[ok] plot written to {out_path}")


def print_headline_summary(sources: Iterable[tuple[str, str, str, str]]) -> None:
    print()
    print("=" * 86)
    print("Phase 1.4 — headline numbers (mean across 3 prompt classes)")
    print("=" * 86)
    print(
        f"  {'model':<46}  {'tok/s @ 5000':>12}  {'KV growth (MB)':>14}  {'peak VRAM':>12}"
    )
    print("-" * 86)
    for label, _color, jsonl_path, _arch in sources:
        rows = load_rows(jsonl_path)
        if not rows:
            continue
        last_steps = [r for r in rows if r["step_idx"] == 5000]
        if not last_steps:
            print(f"  {label:<46}  (no step=5000 row)")
            continue
        tps = np.mean([r["rolling_tok_per_s"] for r in last_steps])
        kvg = np.mean([r["kv_growth_mb_since_step0"] for r in last_steps])
        pkv = np.mean([r["vram_peak_mb"] for r in last_steps])
        print(
            f"  {label:<46}  {tps:>12.2f}  {kvg:>14.1f}  {pkv:>10.0f} MB"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-main", default="figs/phase1_4_arch_comparison.png")
    parser.add_argument("--out-growth", default="figs/phase1_4_kv_growth.png")
    args = parser.parse_args()

    plot_three_way(Path(args.out_main), SOURCES)
    plot_kv_growth_normalized(Path(args.out_growth), SOURCES)
    print_headline_summary(SOURCES)


if __name__ == "__main__":
    main()
