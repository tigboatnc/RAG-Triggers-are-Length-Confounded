"""Appendix tables: Bonferroni-corrected survivor matrix + Spearman ρ per cell.

Two paper-appendix outputs in a single pass over the Qwen + Gemma headline pool:

  (A) Bonferroni-corrected paired-bootstrap CIs at k=15 (α=0.0033, ~99.67% CI)
      on the headline pool (n=800).  Side-by-side with the nominal-95% CIs.
      → Appendix A (sec:appendix-bonferroni in the .tex).

  (B) Length-vs-signal Spearman ρ table per cell.  Mandatory per Santilli 2025
      §3 / Table 2.  Rows = (arch × dataset) cells.  Cols = signal pool.
      Values = ρ(tokens_generated, signal) on CLOSED-BOOK traces.
      → Appendix E (sec:appendix-spearman in the .tex).

Bootstrap formula matches paper §3.2:
    Δ = AUROC(s_resid) − AUROC(tokens_generated)
with s_resid = OLS residual of signal ~ log(tokens+1) on the cohort.

Usage:
    python analysis/e7_bonferroni_and_spearman.py
    python analysis/e7_bonferroni_and_spearman.py --bootstrap-b 10000

Outputs:
    analysis/e7_bonferroni.out
    analysis/e7_spearman.out
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from signal_lib import (
    auroc_pos, bootstrap_auroc_diff, ols_residual,
)
from r1_paired_audit import load_paired as load_paired_base
from moonshot_explore import (
    cabv_score, fct_score, laas_score, normalize_final,
)

ROOT = HERE.parent

# Pool we want both Bonferroni and Spearman on.  Includes the answer-trajectory
# / answer-velocity signals (M1 CABV, M8 FCT, M9 CABV×LAAS) that are the
# headline-strongest survivors in §4.1.
SIGNALS = [
    ("tokens",           "tokens_generated"),
    ("dragin_max",       "DRAGIN-RIND.max"),
    ("ent_max",          "ent.max"),
    ("etc",              "etc (Δ²H max)"),
    ("flare_max",        "flare.max (1-min p)"),
    ("u_cons_max",       "u_cons.max"),
    ("seakr_k5_logdet",  "seakr_k5_proxy.logdet"),
    ("drift_max",        "drift_cosine.max"),
    ("var_max",          "var_l2_window.max"),
    ("hidden_range",     "hidden_norm.range"),
    ("hvr_max",          "hvr.max"),
    ("flip_k5",          "flip_rate K=5"),
    ("flip_k1",          "flip_rate K=1"),
    ("late_flips",       "late_flips"),
    ("min_majority",     "min_majority"),
    ("M1_cabv",          "CABV"),
    ("M8_fct",           "FCT"),
    ("M9_cabv_x_laas",   "CABV×LAAS"),
]
# k for Bonferroni correction (α = 0.05 / k), per paper §4.1 / Appendix A.
N_SIGNALS_FOR_BONFERRONI = 15


def load_paired(cp, op):
    """Wrapper: extends r1_paired_audit.load_paired with the answer-trajectory
    signals (M1 CABV, M8 FCT, M9 CABV×LAAS) computed from the per-step trace
    data."""
    base_rows = load_paired_base(cp, op)
    if not base_rows:
        return []
    closed = {r["id"]: r for r in (json.loads(l) for l in cp.open() if l.strip())}
    for row in base_rows:
        c = closed.get(row["id"])
        if not c:
            continue
        steps = c.get("steps") or []
        final_ans = normalize_final(c.get("model_answer_extracted") or "")
        cabv = cabv_score(steps)
        laas = laas_score(steps)
        fct = fct_score(steps, final_ans)
        row["M1_cabv"] = cabv
        row["M8_fct"] = fct
        row["M9_cabv_x_laas"] = (cabv * laas) if (cabv is not None and laas is not None) else None
    return base_rows


def spearman_rho(xs, ys) -> float:
    """Spearman ρ via the standard rank-correlation formula (with tie handling
    via the average-rank method).  Pure-numpy, no scipy dependency."""
    n = len(xs)
    if n < 4:
        return float("nan")
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    def _ranks(a):
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty(n, dtype=float)
        i = 0
        while i < n:
            j = i
            while j + 1 < n and a[order[j + 1]] == a[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0  # 1-indexed average rank
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx, ry = _ranks(xs), _ranks(ys)
    # Pearson on ranks == Spearman
    mx, my = rx.mean(), ry.mean()
    num = ((rx - mx) * (ry - my)).sum()
    den = math.sqrt(((rx - mx) ** 2).sum() * ((ry - my) ** 2).sum())
    return float(num / den) if den > 0 else float("nan")


def bootstrap_auroc_diff_ci(rows, key_a, key_b, is_pos, B, seed, alphas):
    """Paired bootstrap; returns dict {alpha: (m, lo, hi)} for each requested α.

    alphas should be the two-sided α level: 0.05 for 95% CI, 0.0033 for ~99.67% CI.
    """
    valid = [r for r in rows if r.get(key_a) is not None and r.get(key_b) is not None]
    n = len(valid)
    if n < 4:
        return {a: (float("nan"), float("nan"), float("nan")) for a in alphas}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        sample = [valid[i] for i in idx]
        a = auroc_pos(sample, key_a, is_pos)
        b = auroc_pos(sample, key_b, is_pos)
        if a == a and b == b:
            diffs.append(a - b)
    diffs.sort()
    out = {}
    for alpha in alphas:
        lo = diffs[int((alpha / 2.0) * (len(diffs) - 1))]
        hi = diffs[int((1.0 - alpha / 2.0) * (len(diffs) - 1))]
        m = float(np.mean(diffs))
        out[alpha] = (m, lo, hi)
    return out


def load_arch_cells(arch: str, root: Path):
    """Return list of (cell_label, rows) for one arch (per-dataset)."""
    base = root / "runs" / f"v2_headline_{arch}"
    cells = []
    for ds_label, df, of in [
        ("HotpotQA", "D1_closed_hotpot.jsonl", "E1_oracle_hotpot.jsonl"),
        ("2WikiMHQA", "D2_closed_wiki.jsonl",   "E2_oracle_wiki.jsonl"),
    ]:
        rows = load_paired(base / df, base / of)
        if not rows:
            continue
        cells.append((f"{arch} × {ds_label}", rows))
    return cells


# ============================================================================ #
# (A) Bonferroni-corrected paired bootstrap on the Phi-dropped headline pool
# ============================================================================ #

def run_bonferroni(qwen_rows, gemma_rows, B: int, out_path: Path):
    """Pool Qwen + Gemma at n=800; run paired bootstrap of
    Δ = AUROC(s_resid) − AUROC(tokens) at both 95% and 99.67% CIs."""
    pool = qwen_rows + gemma_rows
    n = len(pool)
    pop = sum(1 for r in pool if r["lift_em"] == +1)
    is_pos = lambda r: r["lift_em"] == +1

    lines = []
    lines.append(f"Bonferroni-corrected paired bootstrap, headline pool (Qwen + Gemma)")
    lines.append(f"  n={n}  pop={pop}  base={pop/n:.3f}  target=lift_em=+1  B={B}")
    lines.append(f"  Bonferroni at k={N_SIGNALS_FOR_BONFERRONI}: α={0.05/N_SIGNALS_FOR_BONFERRONI:.4f}, ~99.67% CI")
    lines.append("")
    lines.append("  Bootstrap formula: Δ = AUROC(s_resid) − AUROC(tokens_generated)")
    lines.append("  s_resid = OLS residual of signal on log(tokens+1)")
    lines.append("")
    header = (f"  {'signal':<30}  {'resAUC':>7}  "
              f"{'Δ':>7}  {'95% CI':>22}   {'99.67% CI':>22}   {'95%?':>4}  {'Bonf?':>5}")
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    # Pre-compute residuals once on the pool for each signal.
    for key, name in SIGNALS:
        valid = [r for r in pool if r.get(key) is not None]
        if len(valid) < 20:
            lines.append(f"  {name:<30}  (skipped — n_valid<20)")
            continue
        resid_map, _ = ols_residual(valid, key, x_key="log_tokens")
        for r in valid:
            r[f"__res_{key}__"] = resid_map[id(r)]

    # Tokens row first (baseline)
    valid_tok = [r for r in pool if r.get("tokens") is not None]
    tok_au = auroc_pos(valid_tok, "tokens", is_pos)
    res_tok_map, _ = ols_residual(valid_tok, "tokens", x_key="log_tokens")
    for r in valid_tok:
        r["__res_tokens__"] = res_tok_map[id(r)]
    res_tok_au = auroc_pos(valid_tok, "__res_tokens__", is_pos)
    lines.append(f"  {'tokens_generated':<30}  {res_tok_au:>7.3f}  "
                 f"{'—':>7}  {'(baseline, raw '+format(tok_au,'.3f')+')':>22}   {'—':>22}     —      —")

    for key, name in SIGNALS:
        if key == "tokens":
            continue
        valid = [r for r in pool if r.get(key) is not None and r.get(f"__res_{key}__") is not None]
        if len(valid) < 20:
            continue
        res_au = auroc_pos(valid, f"__res_{key}__", is_pos)
        out = bootstrap_auroc_diff_ci(
            valid, f"__res_{key}__", "tokens", is_pos,
            B=B, seed=11, alphas=(0.05, 0.05 / N_SIGNALS_FOR_BONFERRONI)
        )
        m95, lo95, hi95 = out[0.05]
        mB, loB, hiB = out[0.05 / N_SIGNALS_FOR_BONFERRONI]
        sig95 = "✓" if (lo95 > 0 or hi95 < 0) else "—"
        sigB = "✓✓" if (loB > 0 or hiB < 0) else "—"
        lines.append(
            f"  {name:<30}  {res_au:>7.3f}  "
            f"{m95:>+7.3f}  [{lo95:>+6.3f},{hi95:>+6.3f}]   "
            f"[{loB:>+6.3f},{hiB:>+6.3f}]   {sig95:>4}    {sigB:>5}"
        )

    lines.append("")
    lines.append("  Reading guide:")
    lines.append("    ✓     paired-significant at α=0.05 (nominal)")
    lines.append("    ✓✓    survives Bonferroni at k=15 (α=0.0033, ~99.67% CI)")
    lines.append("    —     CI straddles 0 at that α level")
    text = "\n".join(lines) + "\n"
    out_path.write_text(text)
    print(text)
    return text


# ============================================================================ #
# (B) Length-vs-signal Spearman ρ per cell
# ============================================================================ #

def run_spearman(qwen_cells, gemma_cells, out_path: Path):
    """For each (arch × dataset) cell, compute Spearman ρ between
    tokens_generated and each signal in the pool (on the CLOSED-BOOK traces,
    consistent with Santilli 2025 Table 2 convention)."""
    all_cells = qwen_cells + gemma_cells

    lines = []
    lines.append("Length-vs-signal Spearman ρ per cell")
    lines.append("  ρ = Spearman(tokens_generated, signal) on closed-book traces")
    lines.append("  Mandatory per Santilli et al. 2025 Table 2 convention.")
    lines.append("  Per Santilli: |ρ| ≥ 0.5 indicates a strong length confound on that signal.")
    lines.append("")
    cell_names = [c[0] for c in all_cells]
    header = f"  {'signal':<28}" + "".join(f"{c.split(' × ')[0][:8]}_{c.split(' × ')[1][:6]:>8}".rjust(16) for c in cell_names)
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for key, name in SIGNALS:
        if key == "tokens":
            continue
        line = f"  {name:<28}"
        for cell_label, rows in all_cells:
            xs = []
            ys = []
            for r in rows:
                t = r.get("tokens")
                v = r.get(key)
                if t is None or v is None:
                    continue
                xs.append(float(t))
                ys.append(float(v))
            if len(xs) < 10:
                line += f"{'n/a':>16}"
                continue
            rho = spearman_rho(xs, ys)
            star = "*" if (rho == rho and abs(rho) >= 0.5) else " "
            line += f"{rho:>+15.3f}{star}"
        lines.append(line)

    # Tokens × itself row for sanity (always ρ=1.0)
    lines.append("")
    lines.append("  Sanity: ρ(tokens, tokens) per cell:")
    for cell_label, rows in all_cells:
        xs = [float(r["tokens"]) for r in rows if r.get("tokens") is not None]
        lines.append(f"    {cell_label:<40}  ρ={spearman_rho(xs, xs):+.3f}  n={len(xs)}")

    lines.append("")
    lines.append("  Interpretation: Signals with |ρ| ≥ 0.5 (starred) are the ones for")
    lines.append("  which length-residualization should most change the AUROC.  Compare")
    lines.append("  against raw vs resid AUROC in Table 1 / Appendix A.")
    text = "\n".join(lines) + "\n"
    out_path.write_text(text)
    print(text)
    return text


# ============================================================================ #
# Main
# ============================================================================ #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap-b", type=int, default=10_000)
    args = ap.parse_args()

    print(f"Loading Qwen + Gemma headline traces...")
    qwen_cells = load_arch_cells("qwen2.5-7b-instruct", ROOT)
    gemma_cells = load_arch_cells("gemma-4-26b-a4b-it", ROOT)
    qwen_rows = [r for _, rs in qwen_cells for r in rs]
    gemma_rows = [r for _, rs in gemma_cells for r in rs]
    print(f"  Qwen:  {len(qwen_rows)} paired rows ({len(qwen_cells)} cells)")
    print(f"  Gemma: {len(gemma_rows)} paired rows ({len(gemma_cells)} cells)")

    print(f"\n=== (A) Bonferroni-corrected paired bootstrap (B={args.bootstrap_b}) ===\n")
    run_bonferroni(qwen_rows, gemma_rows, args.bootstrap_b,
                   ROOT / "analysis/e7_bonferroni.out")

    print(f"\n=== (B) Length-vs-signal Spearman ρ ===\n")
    run_spearman(qwen_cells, gemma_cells,
                 ROOT / "analysis/e7_spearman.out")


if __name__ == "__main__":
    main()
