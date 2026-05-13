"""Radical #1 + #2 fused — external-only dynamic-RAG trigger.

Hypothesis: the model's *external behaviour* (decode time, generation
length, surface-form hedging, multi-signal disagreement) is a sufficient
statistic for its internal uncertainty — zero hidden-state, zero
attention, zero stochastic-resampling access required.

Two feature families, both observable from the OUTPUT of any model
(including black-box HTTP API):

  EXTERNAL FAMILY 1 — Decode latency / length:
    - wall_clock_s          (total wall-clock of the generation)
    - tokens_generated      (total tokens emitted)
    - n_steps               (segmenter-detected reasoning step count)
    - tokens_per_step       (tokens_generated / max(1, n_steps))
    - step_len_std          (std of per-step token counts within a trace)

  EXTERNAL FAMILY 2 — Surface-form deliberation:
    - hedge_total           (sum hedge phrases across steps)
    - verify_total          (sum verify phrases across steps)
    - hvr_max               (Pandey HVR, max across steps)
    - hedge_per_step        (hedge_total / max(1, n_steps))

  SIGNAL-DISAGREEMENT COEFFICIENT — built ON TOP of family 1+2:
    - normalise each external feature to z-score (using all rows in the
      dataset as the calibration cohort), discretise at the 70th
      percentile to get a binary "alarm" per signal, count the alarms
      per trace, then compute `disagreement = min(n_alarms, K -
      n_alarms)` where K is the number of signals. High disagreement =
      half the signals scream uncertain, half say calm — the
      *confident-hallucination* fingerprint.

Comparison baselines (INTERNAL — require hidden-state / attention / K=3 sampling):
    - u_cons.max            (K=3 stochastic consistency sampler)
    - dragin_rind_max.max   (attention-based RIND)
    - mean_token_entropy.max (in-stream entropy, single-pass)

Outputs per dataset:
  1. Single-feature AUROC (each external + each internal) on two targets:
        (a) closed-book INCORRECT
        (b) lift_em == +1
  2. Signal-disagreement coefficient AUROC, with special attention to:
        (c) lift_em == -1 within {-1, +1}  ("harm vs help" — the
            confident-hallucination axis)
  3. Fused external-only score: equal-weight z-sum of all external
     features. AUROC vs all internal singles.
  4. Deployment table for the fused external score at q70 / q90 / q95
     fire rates.
  5. Best-pair / best-triple search across external features (greedy).

Run with `gzn` (or system python — no GPU, no transformers).
"""
from __future__ import annotations
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _common import auroc

ROOT = HERE.parent

DATASETS = [
    ("HotpotQA  (in-domain)",
     ROOT / "runs/v2_exp1_phi-4-mini-instruct/D1_closed_hotpot.jsonl",
     ROOT / "runs/v2_exp1_phi-4-mini-instruct/E1_oracle_hotpot.jsonl"),
    ("2WikiMHQA (OOD)",
     ROOT / "runs/v2_exp1_phi-4-mini-instruct/D2_closed_wiki.jsonl",
     ROOT / "runs/v2_exp1_phi-4-mini-instruct/E2_oracle_wiki.jsonl"),
]


def f(v):
    try: return float(v) if v is not None else None
    except Exception: return None


def step_vals(r, key):
    return [f(s.get(key)) for s in r.get("steps") or []]


def trace_max(r, key):
    vs = [v for v in step_vals(r, key) if v is not None]
    return max(vs) if vs else None


def trace_sum(r, key):
    vs = [v for v in step_vals(r, key) if v is not None]
    return sum(vs) if vs else 0.0


def featurise(r: dict) -> dict | None:
    n_steps = int(r.get("n_steps", 0))
    if n_steps < 1:
        return None
    step_token_counts = [int(s.get("n_tokens_in_step") or 0) for s in r.get("steps") or []]
    feats = {
        # External family 1: latency / length
        "wall_clock_s":    float(r.get("wall_clock_s") or 0),
        "tokens_generated": float(r.get("tokens_generated") or 0),
        "n_steps":         float(n_steps),
        "tokens_per_step": float(r.get("tokens_generated") or 0) / max(1, n_steps),
        "step_len_std":    float(statistics.pstdev(step_token_counts)) if len(step_token_counts) >= 2 else 0.0,
        # External family 2: surface-form deliberation
        "hedge_total":     float(trace_sum(r, "hedge_count")),
        "verify_total":    float(trace_sum(r, "verify_count")),
        "hvr_max":         float(trace_max(r, "hvr") or 0),
        "hedge_per_step":  float(trace_sum(r, "hedge_count")) / max(1, n_steps),
        # Internal baselines (these REQUIRE hidden-state / attention / K=3 sampling)
        "u_cons_max":      trace_max(r, "u_cons"),
        "dragin_max":      trace_max(r, "dragin_rind_max"),
        "ent_max":         trace_max(r, "mean_token_entropy"),
    }
    return feats


EXTERNAL_FEATURES = (
    "wall_clock_s", "tokens_generated", "n_steps", "tokens_per_step", "step_len_std",
    "hedge_total", "verify_total", "hvr_max", "hedge_per_step",
)
INTERNAL_FEATURES = ("u_cons_max", "dragin_max", "ent_max")


def load_paired(cp: Path, op: Path):
    closed = {r["id"]: r for r in (json.loads(l) for l in cp.open())}
    oracle = {r["id"]: r for r in (json.loads(l) for l in op.open())}
    rows = []
    for qid, c in closed.items():
        if qid not in oracle:
            continue
        o = oracle[qid]
        c_em = int(bool(c.get("label_correct_em", c.get("em", False))))
        o_em = int(bool(o.get("label_correct_em", o.get("em", False))))
        lift = +1 if o_em > c_em else (-1 if o_em < c_em else 0)
        feats = featurise(c)
        if feats is None:
            continue
        # Skip rows missing any internal baseline — keeps the comparison apples-to-apples.
        if any(feats[k] is None for k in INTERNAL_FEATURES):
            continue
        feats["lift_em"] = lift
        feats["closed_em"] = c_em
        feats["id"] = qid
        rows.append(feats)
    return rows


def zscores(rows: list[dict], key: str) -> list[float]:
    vals = [r[key] for r in rows]
    mu = statistics.mean(vals); sd = statistics.pstdev(vals) or 1.0
    return [(v - mu) / sd for v in vals]


def auroc_of(rows, key, is_pos):
    pos = [r[key] for r in rows if is_pos(r)]
    neg = [r[key] for r in rows if not is_pos(r)]
    if not pos or not neg:
        return float("nan")
    return auroc(pos, neg)


def fused_zscore(rows: list[dict], keys: tuple, signs: dict[str, int] = None) -> list[float]:
    """Equal-weight signed z-score sum of the given features."""
    signs = signs or {}
    n = len(rows)
    accum = [0.0] * n
    for k in keys:
        zs = zscores(rows, k)
        s = signs.get(k, +1)
        for i in range(n):
            accum[i] += s * zs[i]
    return accum


def signal_disagreement(rows: list[dict], keys: tuple, q: float = 0.70):
    """Per row, count signals above their q-quantile alarm threshold;
    return `min(n_alarms, K - n_alarms)` — peak at K/2 (max disagreement).
    """
    K = len(keys)
    # Compute q-quantile threshold for each signal
    thresholds = {}
    for k in keys:
        vs = sorted(r[k] for r in rows)
        thresholds[k] = vs[int(q * (len(vs) - 1))]
    out = []
    for r in rows:
        n_alarms = sum(1 for k in keys if r[k] > thresholds[k])
        out.append(min(n_alarms, K - n_alarms))
    return out


def deployment_table(rows, scores, label="external_fused"):
    pairs = sorted(zip(scores, rows), key=lambda x: x[0])
    n = len(pairs)
    n_help = sum(1 for r in rows if r["lift_em"] == +1)
    base_help = n_help / n
    n_inc = sum(1 for r in rows if r["closed_em"] == 0)
    print(f"      base helpful = {base_help:.3f}   base incorrect = {n_inc/n:.3f}")
    print(f"      {'target':>7}  {'fire':>5}  {'P(help|F)':>10}  {'P(harm|F)':>10}  {'P(inc|F)':>9}  {'Δprec':>7}")
    for tgt in (0.05, 0.10, 0.20, 0.30):
        k = int((1 - tgt) * (n - 1))
        tau = pairs[k][0]
        fired = [r for s, r in pairs if s > tau]
        if not fired:
            continue
        nf = len(fired)
        ph = sum(1 for r in fired if r["lift_em"] == +1) / nf
        pharm = sum(1 for r in fired if r["lift_em"] == -1) / nf
        pinc = sum(1 for r in fired if r["closed_em"] == 0) / nf
        print(f"      {tgt:>7.2f}  {nf/n*100:>4.1f}%  {ph:>10.3f}  {pharm:>10.3f}  {pinc:>9.3f}  {ph-base_help:>+7.3f}")


def main():
    for label, cp, op in DATASETS:
        rows = load_paired(cp, op)
        n = len(rows)
        n_inc = sum(1 for r in rows if r["closed_em"] == 0)
        n_help = sum(1 for r in rows if r["lift_em"] == +1)
        n_harm = sum(1 for r in rows if r["lift_em"] == -1)
        print(f"\n{'='*92}\n=== {label}   n={n}  incorrect={n_inc}  lift+1={n_help}  lift-1={n_harm} ===\n")

        # --- (1) Single-feature AUROC ---
        print("  --- SINGLE-FEATURE AUROC ---")
        print(f"  {'feature':<22}  {'→INCORRECT':>11}  {'→lift+1':>9}  {'→lift-1|±1':>12}")
        is_inc = lambda r: r["closed_em"] == 0
        is_help = lambda r: r["lift_em"] == +1
        # Among rows with lift_em ∈ {-1, +1}, is lift_em == -1?
        signed = lambda rows_: [r for r in rows_ if r["lift_em"] in (-1, +1)]
        for k in EXTERNAL_FEATURES + INTERNAL_FEATURES:
            au_inc = auroc_of(rows, k, is_inc)
            au_help = auroc_of(rows, k, is_help)
            au_harm = auroc_of(signed(rows), k, lambda r: r["lift_em"] == -1)
            mark = " (INTERNAL)" if k in INTERNAL_FEATURES else ""
            print(f"  {k:<22}  {au_inc:>11.3f}  {au_help:>9.3f}  {au_harm:>12.3f}{mark}")

        # --- (2) Signal-disagreement coefficient ---
        print("\n  --- SIGNAL-DISAGREEMENT COEFFICIENT (external only) ---")
        for keys, name in (
            (EXTERNAL_FEATURES, "all-9-external"),
            (("wall_clock_s", "hedge_total", "tokens_generated", "n_steps", "hvr_max"), "5-external-core"),
            (("wall_clock_s", "hedge_total", "verify_total", "hvr_max"), "4-behavioral"),
        ):
            disag = signal_disagreement(rows, keys, q=0.70)
            for r, d in zip(rows, disag):
                r["_disag"] = d
            au_inc = auroc_of(rows, "_disag", is_inc)
            au_help = auroc_of(rows, "_disag", is_help)
            au_harm = auroc_of(signed(rows), "_disag", lambda r: r["lift_em"] == -1)
            print(f"  disagreement({name}, q=0.70):  →inc={au_inc:.3f}  →lift+1={au_help:.3f}  →lift-1|±1={au_harm:.3f}")

        # --- (3) Equal-weight fused external ---
        print("\n  --- FUSED EXTERNAL (equal-weight z-sum) ---")
        # All external features point UP for uncertainty except `verify_total`
        # (more verification phrases can be a hedging sign too, leave +1).
        signs = {k: +1 for k in EXTERNAL_FEATURES}
        for combo in (
            ("wall_clock_s",),
            ("tokens_generated",),
            ("wall_clock_s", "tokens_generated"),
            ("wall_clock_s", "hedge_total"),
            ("wall_clock_s", "hvr_max"),
            ("wall_clock_s", "tokens_generated", "hvr_max"),
            EXTERNAL_FEATURES,
        ):
            zsum = fused_zscore(rows, combo, signs)
            for r, z in zip(rows, zsum):
                r["_fused"] = z
            au_inc = auroc_of(rows, "_fused", is_inc)
            au_help = auroc_of(rows, "_fused", is_help)
            au_harm = auroc_of(signed(rows), "_fused", lambda r: r["lift_em"] == -1)
            print(f"  fused{list(combo)}".ljust(70) +
                  f"  →inc={au_inc:.3f}  →lift+1={au_help:.3f}  →lift-1|±1={au_harm:.3f}")

        # --- (4) Best-pair / best-triple greedy search ---
        print("\n  --- GREEDY SEARCH: best external feature combinations (→ lift+1) ---")
        # Single best
        best = max(EXTERNAL_FEATURES, key=lambda k: auroc_of(rows, k, is_help))
        print(f"  best single:  {best:<22}  AUROC={auroc_of(rows, best, is_help):.3f}")
        # Best pair
        best_pair = None; best_pair_au = -1
        for a in EXTERNAL_FEATURES:
            for b in EXTERNAL_FEATURES:
                if a >= b: continue
                zs = fused_zscore(rows, (a, b), signs)
                for r, z in zip(rows, zs): r["_p"] = z
                au = auroc_of(rows, "_p", is_help)
                if au > best_pair_au:
                    best_pair_au, best_pair = au, (a, b)
        print(f"  best pair:    {best_pair!s:<35}  AUROC={best_pair_au:.3f}")
        # Best triple (greedy: add best 3rd to best_pair)
        rest = [k for k in EXTERNAL_FEATURES if k not in best_pair]
        best_third = None; best_triple_au = best_pair_au
        for c in rest:
            zs = fused_zscore(rows, (*best_pair, c), signs)
            for r, z in zip(rows, zs): r["_t"] = z
            au = auroc_of(rows, "_t", is_help)
            if au > best_triple_au:
                best_triple_au, best_third = au, c
        if best_third:
            print(f"  best triple:  {(*best_pair, best_third)!s:<35}  AUROC={best_triple_au:.3f}")
        else:
            print(f"  best triple:  no improvement over pair")

        # --- (5) Deployment table — fused EXTERNAL vs best INTERNAL ---
        print("\n  --- DEPLOYMENT (fused external — all 9) ---")
        zs = fused_zscore(rows, EXTERNAL_FEATURES, signs)
        deployment_table(rows, zs, label="external_all")
        print("\n  --- DEPLOYMENT (best external pair) ---")
        zs = fused_zscore(rows, best_pair, signs)
        deployment_table(rows, zs, label="external_pair")
        print("\n  --- DEPLOYMENT (u_cons.max — INTERNAL baseline) ---")
        deployment_table(rows, [r["u_cons_max"] for r in rows], label="u_cons")
        print("\n  --- DEPLOYMENT (dragin.max — INTERNAL baseline) ---")
        deployment_table(rows, [r["dragin_max"] for r in rows], label="dragin")


if __name__ == "__main__":
    main()
