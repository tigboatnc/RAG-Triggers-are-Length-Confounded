"""Moonshot exploratory triggers on existing R1 Phi + Qwen traces (n=200/cell).

We pivot from AID to ask: given the SAME on-disk data (no re-decode), can a
fundamentally different signal mechanism beat AID `min_majority` (current
headline survivor: resid AUROC 0.648 on Qwen pooled, paired-bootstrap-sig
beats `tokens_generated` with CI [+0.020, +0.224]) on the `lift_em = +1`
paired-oracle target?

Six new signals tested:

  M1 CABV  Continuous Answer-Belief Velocity
           Per step → K=5 answer empirical distribution p_t (Laplace smoothed
           over the union of seen answers across the trace) → mean
           JSD(p_t, p_{t+1}) over consecutive steps with both populated.
           Generalises AID flip_rate (binary edge count) and min_majority
           (within-step entropy) to a continuous between-step measure.

  M2 LAAS  Length-Asymmetry Across Samples
           Per step → coefficient of variation
           std(len(samples[k].text)) / mean(len(...)). Max over steps.
           Scale-invariant => length-orthogonal by construction.

  M3 CRA   Confidence-Recovery Asymmetry
           On the greedy per_token_max_logprob trajectory across all steps,
           identify "drops" (consecutive runs of prob < 0.5) and count
           recoveries (any subsequent prob >= 0.7). Score = 1 - recovered/total.
           High = unrecovered drops = destructive instability (Chen et al. 2026).

  M5 HSE   Hidden-State effective rank
           Per step → K=5 sample EOS hidden vectors → covariance C → effective
           rank exp(-Σ λ_i log λ_i) on normalised eigenvalues of C.
           Max over steps. Captures direction-of-uncertainty (anisotropy)
           that SeaKR-K5 logdet (volume) does not.

  M7 SCC   Sample-Centroid step Consistency
           Per step, centroid c_t of K=5 sample EOS hidden vectors.
           Compute mean cosine(c_t, c_{t-1}) across consecutive steps.
           Score = 1 - mean(cos). Geometric analogue of AID flip_rate.

  M8 ECT   Early-Commitment Time
           t_commit = first step at which the K=5 majority answer equals the
           final extracted answer AND stays equal for every remaining step.
           Score = t_commit / max(1, n_answer_bearing_steps - 1).
           Late commit = high score = harder question = retrieval-helpful.

  M9 CABV*LAAS  Product trigger (both must fire).

Targets:
  - lift_em = +1 (PAPER HEADLINE — paired closed × oracle)
  - closed_em = 0 (secondary / sanity)

Per signal we report:
  - raw AUROC, length-residualised AUROC
  - paired bootstrap (B=5000) of (signal − tokens) and (signal − min_majority)
    AUROC differences with 95% CIs.

This script does NOT modify any existing analyser; it is purely additive and
reads the same on-disk traces.

Usage:
    python analysis/moonshot_explore.py
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from r1_phi_d1_closed_audit import (
    auroc_pos,
    bootstrap_auroc_diff,
    extract_answer,
    flare_max_per_step,
    flip_rate_majority,
    flip_rate_single,
    late_flips_count,
    min_majority_inv,
    ols_residual,
    per_step_answers,
    seakr_k5_logdet,
    tmax,
    trange,
)

ROOT = HERE.parent


# --------------------------------------------------------------------------- #
# Moonshot signal extractors
# --------------------------------------------------------------------------- #
def cabv_score(steps, k_top: int = 5, eps: float = 1e-3) -> float | None:
    """M1 CABV — continuous answer-belief velocity.

    Per step compute the empirical answer distribution from K=5 samples
    (Laplace-smoothed over the union of all answers ever seen in this trace),
    then mean Jensen-Shannon distance between consecutive populated steps.
    """
    per_step = per_step_answers(steps, k_top=k_top)
    populated = [(i, ans) for i, ans in enumerate(per_step) if ans]
    if len(populated) < 2:
        return None
    universe = sorted({a for _, ans in populated for a in ans})
    if not universe:
        return None
    idx = {a: j for j, a in enumerate(universe)}
    K_uni = len(universe)

    dists = []
    for _, ans in populated:
        v = np.full(K_uni, eps, dtype=float)
        for a in ans:
            v[idx[a]] += 1.0
        v = v / v.sum()
        dists.append(v)

    def jsd(p, q):
        m = 0.5 * (p + q)
        # add tiny epsilon to avoid log(0) — distributions already smoothed
        kl_pm = float(np.sum(p * np.log((p + 1e-12) / (m + 1e-12))))
        kl_qm = float(np.sum(q * np.log((q + 1e-12) / (m + 1e-12))))
        return 0.5 * (kl_pm + kl_qm)

    deltas = [jsd(dists[i], dists[i + 1]) for i in range(len(dists) - 1)]
    if not deltas:
        return None
    return float(np.mean(deltas))


def laas_score(steps) -> float | None:
    """M2 LAAS — coefficient of variation of K=5 sample lengths per step,
    max over steps.
    """
    out = []
    for s in steps:
        samples = s.get("samples", []) or []
        lens = []
        for samp in samples:
            n = samp.get("n_tokens")
            if n is None:
                t = samp.get("text") or ""
                n = len(t.split())
            try:
                lens.append(float(n))
            except Exception:
                continue
        if len(lens) < 2:
            continue
        m = float(np.mean(lens))
        if m <= 0:
            continue
        sd = float(np.std(lens, ddof=0))
        out.append(sd / m)
    if not out:
        return None
    return float(np.max(out))


def cra_score(steps, drop_thr: float = 0.5, rec_thr: float = 0.7) -> float | None:
    """M3 CRA — confidence-recovery asymmetry.

    Concatenate per_token_max_logprob across all steps in order (these are the
    *greedy* token probabilities, already in [0,1]). Identify maximal
    consecutive runs of probs < drop_thr ("drops"). For each drop ending at
    token i, mark RECOVERED if any prob in the remaining trace exceeds
    rec_thr. Return 1 - recovered/total. High = many unrecovered drops
    (destructive in the Chen et al. sense).
    """
    seq: list[float] = []
    for s in steps:
        probs = s.get("per_token_max_logprob") or []
        for p in probs:
            try:
                if p is not None:
                    seq.append(float(p))
            except Exception:
                continue
    if len(seq) < 4:
        return None

    # find drop runs (consecutive probs < drop_thr)
    drops = []  # list of (start, end) inclusive
    i, n = 0, len(seq)
    while i < n:
        if seq[i] < drop_thr:
            j = i
            while j < n and seq[j] < drop_thr:
                j += 1
            drops.append((i, j - 1))
            i = j
        else:
            i += 1
    if not drops:
        return 0.0  # no drops at all = stable trace
    rec = 0
    for (_a, b) in drops:
        # any token after b with prob >= rec_thr counts as recovery
        tail = seq[b + 1 :]
        if tail and max(tail) >= rec_thr:
            rec += 1
    return 1.0 - rec / len(drops)


def hse_score(steps) -> float | None:
    """M5 HSE — effective rank of the K=5 sample EOS hidden cloud, max over
    steps. Effective rank = exp(H(λ_norm)) where λ_norm are eigenvalues of the
    sample covariance, normalised to sum to 1. Bounded in [1, K].
    """
    out = []
    for s in steps:
        samples = s.get("samples", []) or []
        H = []
        for samp in samples:
            h = samp.get("eos_hidden")
            if h is None:
                continue
            try:
                H.append(np.asarray(h, dtype=float))
            except Exception:
                continue
        if len(H) < 3:
            continue
        try:
            M = np.stack(H, axis=0)  # K x d
            M = M - M.mean(axis=0, keepdims=True)
            # K x K Gram is rank-min(K-1, d) and cheaper than d x d covariance
            G = (M @ M.T) / max(1, M.shape[1])
            evals = np.linalg.eigvalsh(G)
            evals = np.clip(evals, 1e-12, None)
            evals = evals / evals.sum()
            H_ent = -float(np.sum(evals * np.log(evals)))
            out.append(math.exp(H_ent))
        except Exception:
            continue
    if not out:
        return None
    return float(max(out))


def scc_score(steps) -> float | None:
    """M7 SCC — sample-centroid step (in)consistency.
    1 - mean(cos(c_t, c_{t-1})) over consecutive populated steps.
    """
    cents = []
    for s in steps:
        samples = s.get("samples", []) or []
        H = []
        for samp in samples:
            h = samp.get("eos_hidden")
            if h is None:
                continue
            try:
                H.append(np.asarray(h, dtype=float))
            except Exception:
                continue
        if not H:
            continue
        cents.append(np.mean(np.stack(H, axis=0), axis=0))
    if len(cents) < 2:
        return None
    cos_sims = []
    for i in range(1, len(cents)):
        a, b = cents[i - 1], cents[i]
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na <= 0 or nb <= 0:
            continue
        cos_sims.append(float(np.dot(a, b) / (na * nb)))
    if not cos_sims:
        return None
    return float(1.0 - np.mean(cos_sims))


def ect_score(steps, final_ans: str | None, k_top: int = 5) -> float | None:
    """M8 ECT — early commitment time / n_answer_bearing_steps.

    t_commit = smallest i such that for every j >= i with answers populated,
    the K=5 majority answer at step j equals `final_ans`. Returns the index
    normalised by the number of populated answer steps - 1 (so ECT in [0, 1]).
    Late commit = score near 1 = harder.
    """
    if not final_ans:
        return None
    final_ans_n = final_ans
    per_step = per_step_answers(steps, k_top=k_top)
    populated = [(i, ans) for i, ans in enumerate(per_step) if ans]
    if len(populated) < 2:
        return None
    majorities = []
    for i, ans in populated:
        m = Counter(ans).most_common(1)[0][0]
        majorities.append((i, m))
    # find smallest k such that majorities[k:] all equal final_ans_n
    n = len(majorities)
    t_commit = n - 1  # default to "committed only at last step"
    for k in range(n):
        tail = majorities[k:]
        if all(m == final_ans_n for _, m in tail):
            t_commit = k
            break
    if n - 1 == 0:
        return 0.0
    return t_commit / (n - 1)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def normalize_final(s: str) -> str:
    import re as _re
    s = _re.sub(r"[^a-z0-9 ]", "", (s or "").strip().lower())[:30].strip()
    return s


def load_paired(cp: Path, op: Path, arch: str, dataset: str) -> list[dict]:
    if not cp.exists() or not op.exists():
        return []
    closed = {r["id"]: r for r in (json.loads(l) for l in cp.open() if l.strip())}
    oracle = {r["id"]: r for r in (json.loads(l) for l in op.open() if l.strip())}
    rows = []
    for qid, c in closed.items():
        if qid not in oracle:
            continue
        o = oracle[qid]
        c_em = int(bool(c.get("label_correct_em", c.get("em", False))))
        o_em = int(bool(o.get("label_correct_em", o.get("em", False))))
        lift = +1 if o_em > c_em else (-1 if o_em < c_em else 0)
        steps = c.get("steps") or []
        psa5 = per_step_answers(steps, k_top=5)
        final_ans = normalize_final(c.get("model_answer_extracted") or "")
        toks = int(c.get("tokens_generated") or 0)

        cabv = cabv_score(steps)
        laas = laas_score(steps)
        cra = cra_score(steps)
        hse = hse_score(steps)
        scc = scc_score(steps)
        ect = ect_score(steps, final_ans)

        cabv_x_laas = (cabv * laas) if (cabv is not None and laas is not None) else None

        rows.append({
            "id": qid,
            "arch": arch,
            "dataset": dataset,
            "type": c.get("type"),
            "closed_em": c_em,
            "oracle_em": o_em,
            "lift_em": lift,
            "n_steps": int(c.get("n_steps", 0)),
            "tokens": toks,
            "log_tokens": math.log(max(1, toks) + 1),

            # baselines we want to head-to-head against
            "dragin_max": tmax(c, "dragin_rind_max") or 0.0,
            "ent_max": tmax(c, "mean_token_entropy") or 0.0,
            "flare_max": flare_max_per_step(c),
            "seakr_k5_logdet": seakr_k5_logdet(c),
            "flip_k5": flip_rate_majority(psa5),
            "flip_k1": flip_rate_single(psa5),
            "late_flips": late_flips_count(psa5, len(steps)),
            "min_majority": min_majority_inv(psa5),

            # MOONSHOTS
            "M1_cabv": cabv,
            "M2_laas": laas,
            "M3_cra": cra,
            "M5_hse": hse,
            "M7_scc": scc,
            "M8_ect": ect,
            "M9_cabv_x_laas": cabv_x_laas,
        })
    return rows


def load_arch(arch: str, root: Path) -> list[dict]:
    base = root / "runs" / f"v2_headline_{arch}"
    rows = []
    rows += load_paired(base / "D1_closed_hotpot.jsonl", base / "E1_oracle_hotpot.jsonl", arch, "hotpot")
    rows += load_paired(base / "D2_closed_wiki.jsonl", base / "E2_oracle_wiki.jsonl", arch, "wiki")
    return rows


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
SIGNALS_HEAD2HEAD = [
    # baselines & current survivors
    ("tokens",         "tokens_generated"),
    ("dragin_max",     "DRAGIN-RIND.max"),
    ("seakr_k5_logdet","SeaKR-K5.logdet"),
    ("ent_max",        "ent.max"),
    ("flare_max",      "flare.max"),
    ("flip_k5",        "AID flip_rate K=5"),
    ("flip_k1",        "AID flip_rate K=1"),
    ("late_flips",     "AID late_flips"),
    ("min_majority",   "AID min_majority    [survivor]"),
    # moonshots
    ("M1_cabv",        "** M1 CABV (continuous belief velocity)"),
    ("M2_laas",        "** M2 LAAS (sample-len CV)"),
    ("M3_cra",         "** M3 CRA (confidence-recovery asym)"),
    ("M5_hse",         "** M5 HSE (effective-rank cloud)"),
    ("M7_scc",         "** M7 SCC (centroid drift)"),
    ("M8_ect",         "** M8 ECT (early-commitment time)"),
    ("M9_cabv_x_laas", "** M9 CABV*LAAS"),
]


def banner(s):
    print("\n" + "=" * 100)
    print(s)
    print("=" * 100)


def is_lift_pos(r):
    return r["lift_em"] == +1


def is_closed_neg(r):
    return r["closed_em"] == 0


def report(rows, label: str, B: int):
    rows_with_tok = [r for r in rows if r.get("tokens")]
    # synthesise a `tokens` key for uniform handling
    for r in rows_with_tok:
        r["tokens"] = float(r["tokens"])

    for target_name, label_fn in [("lift_em=+1", is_lift_pos), ("closed_em=0", is_closed_neg)]:
        n = len(rows_with_tok)
        pop = sum(1 for r in rows_with_tok if label_fn(r))
        if pop < 4 or pop > n - 4:
            print(f"\n[{label}] target={target_name} skipped (pop={pop}/{n})")
            continue
        banner(f"[{label}] target={target_name}    n={n}    pop={pop}    base={pop/n:.3f}")

        # paired bootstrap baselines: reference signals are tokens AND min_majority
        print(f"\n  {'signal':<48} {'valid':>6} {'rawAUC':>7} {'resAUC':>7}   "
              f"{'Δ vs tokens':>14}        {'Δ vs min_maj':>14}")
        # precompute residuals once per signal
        for key, name in SIGNALS_HEAD2HEAD:
            valid = [r for r in rows_with_tok if r.get(key) is not None]
            if len(valid) < 6:
                print(f"  {name:<48} {len(valid):>6}  insufficient")
                continue
            au = auroc_pos(valid, key, label_fn)
            res_map, _ = ols_residual(valid, key, x_key="log_tokens")
            for r in valid:
                r["__res__"] = res_map[id(r)]
            res_au = auroc_pos(valid, "__res__", label_fn)

            if key == "tokens":
                d_tok_str = "(baseline)            "
            else:
                m, lo, hi, _ = bootstrap_auroc_diff(valid, key, "tokens", label_fn, B=B, seed=11)
                star = " *" if (lo > 0 or hi < 0) else "  "
                d_tok_str = f"{m:+.3f} [{lo:+.3f},{hi:+.3f}]{star}"

            if key == "min_majority":
                d_mm_str = "(reference survivor)"
            else:
                # only run vs min_majority if both are available on the same valid set
                valid_both = [r for r in rows_with_tok if r.get(key) is not None and r.get("min_majority") is not None]
                if len(valid_both) < 6:
                    d_mm_str = "n/a"
                else:
                    m2, lo2, hi2, _ = bootstrap_auroc_diff(valid_both, key, "min_majority", label_fn, B=B, seed=13)
                    star2 = " *" if (lo2 > 0 or hi2 < 0) else "  "
                    d_mm_str = f"{m2:+.3f} [{lo2:+.3f},{hi2:+.3f}]{star2}"

            print(f"  {name:<48} {len(valid):>6} {au:>7.3f} {res_au:>7.3f}   "
                  f"{d_tok_str:<22}  {d_mm_str}")
            for r in valid:
                r.pop("__res__", None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap-b", type=int, default=5000)
    args = ap.parse_args()

    phi = load_arch("phi-4-mini-instruct", ROOT)
    qwen = load_arch("qwen2.5-7b-instruct", ROOT)

    print(f"\nLoaded:  Phi paired = {len(phi)},  Qwen paired = {len(qwen)},  pooled = {len(phi)+len(qwen)}")

    if phi:
        report(phi, "Phi pooled (R1, n=400 paired across HotpotQA + 2Wiki)", args.bootstrap_b)
    if qwen:
        report(qwen, "Qwen2.5 pooled (R1, n=400 paired across HotpotQA + 2Wiki)", args.bootstrap_b)
    if phi and qwen:
        report(phi + qwen, "Phi + Qwen2.5 pooled (R1, n=800 paired)", args.bootstrap_b)

    # per-cell breakdown for the moonshots that survived overall
    banner("PER-CELL breakdown — lift_em=+1 raw AUROC for moonshots")
    print(f"  {'cell':<28} {'n':>4} {'pop':>4} | "
          f"{'min_maj':>8} {'M1':>8} {'M2':>8} {'M3':>8} {'M5':>8} {'M7':>8} {'M8':>8} {'M9':>8}")
    for arch_label, rows_arch in [("Phi", phi), ("Qwen2.5", qwen)]:
        for ds in ("hotpot", "wiki"):
            cell = [r for r in rows_arch if r["dataset"] == ds]
            if len(cell) < 8:
                continue
            pop = sum(1 for r in cell if is_lift_pos(r))
            if pop < 3:
                print(f"  {arch_label}_{ds:<22} {len(cell):>4} {pop:>4} | (pop too small)")
                continue
            def au(k):
                v = [r for r in cell if r.get(k) is not None]
                if len(v) < 6:
                    return float("nan")
                return auroc_pos(v, k, is_lift_pos)
            print(f"  {arch_label+'_'+ds:<28} {len(cell):>4} {pop:>4} | "
                  f"{au('min_majority'):>8.3f} {au('M1_cabv'):>8.3f} {au('M2_laas'):>8.3f} "
                  f"{au('M3_cra'):>8.3f} {au('M5_hse'):>8.3f} {au('M7_scc'):>8.3f} "
                  f"{au('M8_ect'):>8.3f} {au('M9_cabv_x_laas'):>8.3f}")


if __name__ == "__main__":
    main()
