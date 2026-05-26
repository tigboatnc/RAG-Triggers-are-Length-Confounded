"""Headline survivor matrix + answer-trajectory signal exploration.

Reads the `v2_headline_<arch>` paired traces (Qwen + Gemma) and computes
length-residualised AUROC of every signal in the audit pool against
`lift_em = +1` and `closed_em = 0` targets, with paired-bootstrap CIs of
Δ = AUROC(s_resid) − AUROC(tokens_generated).

Three signal families are introduced or formalised in this script (the
existing published-pool signals — DRAGIN, FLARE, SeaKR-proxy, ETC,
SelfCheckGPT-style u_cons — live in `signal_lib.py`):

  M1 CABV — Continuous Answer-Belief Velocity
    Per step → K=5 answer empirical distribution p_t (Laplace-smoothed
    over the union of answers seen in this trace) → mean JSD(p_t, p_{t+1})
    over consecutive populated steps.  Generalises the binary flip_rate
    edge-count to a continuous between-step measure.

  M2 LAAS — Length-Asymmetry Across Samples
    Per step → coefficient of variation std(len)/mean(len) of K=5 sample
    lengths; max over steps.  Scale-invariant by construction.

  M8 FCT — First-Commitment Time (paper headline; was "ECT" in early
    drafts, renamed for ETC-naming-collision with Li et al. 2025)
    t_commit = first step at which the K=5 majority answer equals the
    final extracted answer AND stays equal for every remaining step.
    Score = t_commit / max(1, n_answer_bearing_steps − 1).
    Late commit = high score = harder = retrieval-helpful.

  M9 CABV × LAAS  — product trigger (both must fire).

The other M-series (M3 CRA, M5 HSE, M7 SCC) were exploratory and never
beat the published pool; they are kept here as null-result reference.

Usage:
    python analysis/moonshot_explore.py --bootstrap-b 10000
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from signal_lib import (
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

# Architectures shipped with the paper's headline pool.
HEADLINE_ARCHS = ("qwen2.5-7b-instruct", "gemma-4-26b-a4b-it")


# --------------------------------------------------------------------------- #
# Answer-trajectory / multi-sample signal extractors
# --------------------------------------------------------------------------- #
def cabv_score(steps, k_top: int = 5, eps: float = 1e-3) -> float | None:
    """M1 CABV — continuous answer-belief velocity.

    Per step compute the empirical K=5 answer distribution (Laplace-smoothed
    over the union of all answers ever seen in this trace), then mean
    Jensen-Shannon distance between consecutive populated steps.
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


def fct_score(steps, final_ans: str | None, k_top: int = 5) -> float | None:
    """M8 FCT — first-commitment time (was ECT pre-renaming).

    t_commit = smallest i such that for every j >= i with answers populated,
    the K=5 majority answer at step j equals `final_ans`.  Returned value
    is t_commit / max(1, n_populated_steps - 1) so FCT ∈ [0, 1].  Late
    commit ⇒ score near 1 ⇒ harder ⇒ retrieval-helpful.
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
        fct = fct_score(steps, final_ans)
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

            # Published-pool baselines + answer-trajectory aggregations
            "dragin_max":      tmax(c, "dragin_rind_max") or 0.0,
            "ent_max":         tmax(c, "mean_token_entropy") or 0.0,
            "flare_max":       flare_max_per_step(c),
            "seakr_k5_logdet": seakr_k5_logdet(c),
            "flip_k5":         flip_rate_majority(psa5),
            "flip_k1":         flip_rate_single(psa5),
            "late_flips":      late_flips_count(psa5, len(steps)),
            "min_majority":    min_majority_inv(psa5),

            # Paper signals introduced or formalised here
            "M1_cabv":         cabv,
            "M2_laas":         laas,
            "M8_fct":          fct,
            "M9_cabv_x_laas":  cabv_x_laas,
        })
    return rows


def load_arch(arch: str, root: Path) -> list[dict]:
    base = root / "runs" / f"v2_headline_{arch}"
    rows = []
    rows += load_paired(base / "D1_closed_hotpot.jsonl", base / "E1_oracle_hotpot.jsonl", arch, "hotpot")
    rows += load_paired(base / "D2_closed_wiki.jsonl",   base / "E2_oracle_wiki.jsonl",   arch, "wiki")
    return rows


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
SIGNALS_HEAD2HEAD = [
    # baselines & published-pool survivors
    ("tokens",          "tokens_generated"),
    ("dragin_max",      "DRAGIN-RIND.max"),
    ("seakr_k5_logdet", "seakr_k5_proxy.logdet"),
    ("ent_max",         "ent.max"),
    ("flare_max",       "flare.max"),
    ("flip_k5",         "flip_rate K=5"),
    ("flip_k1",         "flip_rate K=1"),
    ("late_flips",      "late_flips"),
    ("min_majority",    "min_majority [reference]"),
    # answer-velocity + answer-trajectory novelties
    ("M1_cabv",         "M1 CABV (belief velocity)"),
    ("M2_laas",         "M2 LAAS (sample-len CV)"),
    ("M8_fct",          "FCT (first-commitment time)"),
    ("M9_cabv_x_laas",  "CABV × LAAS"),
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
    for r in rows_with_tok:
        r["tokens"] = float(r["tokens"])

    for target_name, label_fn in [("lift_em=+1", is_lift_pos), ("closed_em=0", is_closed_neg)]:
        n = len(rows_with_tok)
        pop = sum(1 for r in rows_with_tok if label_fn(r))
        if pop < 4 or pop > n - 4:
            print(f"\n[{label}] target={target_name} skipped (pop={pop}/{n})")
            continue
        banner(f"[{label}] target={target_name}    n={n}    pop={pop}    base={pop/n:.3f}")

        # Pre-compute min_majority length-residualised once for the target so
        # the Δ-vs-min_majority bootstrap can compare residualised signal vs
        # residualised reference (matches paper §3.2 formula).
        mm_valid = [r for r in rows_with_tok if r.get("min_majority") is not None]
        if mm_valid:
            mm_res_map, _ = ols_residual(mm_valid, "min_majority", x_key="log_tokens")
            for r in mm_valid:
                r["__res_mm__"] = mm_res_map[id(r)]

        print(f"\n  {'signal':<48} {'valid':>6} {'rawAUC':>7} {'resAUC':>7}   "
              f"{'Δ vs tokens':>14}        {'Δ vs min_maj':>14}")
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
                # Δ = AUROC(s_resid) − AUROC(tokens_generated)  — paper §3.2
                m, lo, hi, _ = bootstrap_auroc_diff(valid, "__res__", "tokens", label_fn, B=B, seed=11)
                star = " *" if (lo > 0 or hi < 0) else "  "
                d_tok_str = f"{m:+.3f} [{lo:+.3f},{hi:+.3f}]{star}"

            if key == "min_majority":
                d_mm_str = "(reference)"
            else:
                valid_both = [r for r in rows_with_tok if r.get(key) is not None and r.get("min_majority") is not None]
                if len(valid_both) < 6:
                    d_mm_str = "n/a"
                else:
                    m2, lo2, hi2, _ = bootstrap_auroc_diff(valid_both, "__res__", "__res_mm__", label_fn, B=B, seed=13)
                    star2 = " *" if (lo2 > 0 or hi2 < 0) else "  "
                    d_mm_str = f"{m2:+.3f} [{lo2:+.3f},{hi2:+.3f}]{star2}"

            print(f"  {name:<48} {len(valid):>6} {au:>7.3f} {res_au:>7.3f}   "
                  f"{d_tok_str:<22}  {d_mm_str}")
            for r in valid:
                r.pop("__res__", None)

        for r in mm_valid:
            r.pop("__res_mm__", None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap-b", type=int, default=10_000)
    args = ap.parse_args()

    arch_rows = {arch: load_arch(arch, ROOT) for arch in HEADLINE_ARCHS}
    qwen = arch_rows["qwen2.5-7b-instruct"]
    gemma = arch_rows["gemma-4-26b-a4b-it"]

    print(f"\nLoaded:  Qwen paired = {len(qwen)},  Gemma paired = {len(gemma)},  "
          f"pooled = {len(qwen) + len(gemma)}")

    if qwen:
        report(qwen, "Qwen2.5-7B-Instruct  (n=400 paired, HotpotQA + 2Wiki)", args.bootstrap_b)
    if gemma:
        report(gemma, "Gemma-4-26B-A4B-it  (n=400 paired, HotpotQA + 2Wiki)", args.bootstrap_b)
    if qwen and gemma:
        report(qwen + gemma, "Qwen + Gemma pooled — HEADLINE POOL (n=800)", args.bootstrap_b)

    # Per-cell breakdown — the per-cell raw AUROC table cited in §4 of the paper.
    banner("PER-CELL breakdown — lift_em=+1 raw AUROC")
    print(f"  {'cell':<28} {'n':>4} {'pop':>4} | "
          f"{'min_maj':>8} {'CABV':>8} {'LAAS':>8} {'FCT':>8} {'CABV×LAAS':>10}")
    for arch_label, rows_arch in [("Qwen2.5", qwen), ("Gemma", gemma)]:
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

            print(f"  {arch_label + '_' + ds:<28} {len(cell):>4} {pop:>4} | "
                  f"{au('min_majority'):>8.3f} {au('M1_cabv'):>8.3f} "
                  f"{au('M2_laas'):>8.3f} {au('M8_fct'):>8.3f} "
                  f"{au('M9_cabv_x_laas'):>10.3f}")


if __name__ == "__main__":
    main()
