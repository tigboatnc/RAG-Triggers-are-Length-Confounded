"""R1 closed-book audit on partial-landing R1 traces (Phi D1 only at first).

This script runs the **closed-book-only** half of the headline audit on whatever
R1 cells have closed-book traces written so far (D1, D2). It cannot compute
`lift_em = +1` analyses because those require paired oracle traces (E1, E2)
which haven't landed yet at script time.

Audited signal pool (12 trigger-pool signals + 4 AID variants):
    - Length / external:   tokens_generated, wall_clock_s, n_steps
    - Logit family:        ent.max (mean_token_entropy.max),
                           etc (Δ²H max of mean_token_entropy),
                           flare.max (1 - min(per_token_max_logprob_alt) per step → max)
    - Attention family:    dragin_rind_max.max
    - Hidden-state CRV:    drift_cosine.max, variance_l2_window.max,
                           hidden_norm.range
    - Text family:         hvr.max (SELFDOUBT-T1)
    - Multi-sample K=5:    seakr_k5_proxy (Gram log-det of K=5 sample EOS hiddens),
                           u_cons.max (exploratory — kept for completeness)
    - AID family:          flip_rate_k5 (majority over K=5 samples, our contribution),
                           flip_rate_k1 (Datta-style proxy, samples[0] only),
                           late_flips (V3 from _explore_ans_flip_variants),
                           min_majority (V5 from same: 1 - min step majority frac)

Targets:
    closed_em == 0  (closed-book incorrect; the only target available without oracle)

For each signal, we report:
    - Raw AUROC against closed_em=0
    - Length-residualized AUROC (signal ~ log(tokens+1), evaluate on residual)
    - Paired-bootstrap 95% CI on (signal AUROC - tokens AUROC), B=10,000
    - Precision-at-budget at fire rates {5%, 10%, 20%}, raw AND residualized

Sanity-comparison block at the end: replays length_audit_v2 on the existing
Phi v2 traces (n=393 paired HotpotQA) and prints the same closed_em=0 row for
each signal, so we can see whether R1 D1 (n=200) reproduces the v2 numbers.

Usage:
    python analysis/r1_phi_d1_closed_audit.py
    # Override:
    python analysis/r1_phi_d1_closed_audit.py --bootstrap-b 10000 \
        --d1 runs/v2_headline_phi-4-mini-instruct/D1_closed_hotpot.jsonl \
        --d2 runs/v2_headline_phi-4-mini-instruct/D2_closed_wiki.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _common import auroc as _auroc

ROOT = HERE.parent

ANSWER_RE = re.compile(r"\bAnswer\s*:\s*(.+?)(?:[\n.]|$)", re.IGNORECASE)


def _f(v):
    try:
        return float(v) if v is not None else None
    except Exception:
        return None


def step_vals(r, key):
    return [_f(s.get(key)) for s in r.get("steps") or []]


def tmax(r, key):
    vs = [v for v in step_vals(r, key) if v is not None]
    return max(vs) if vs else None


def trange(r, key):
    vs = [v for v in step_vals(r, key) if v is not None]
    if len(vs) < 2:
        return 0.0
    return max(vs) - min(vs)


def etc_score(r):
    H = [v for v in step_vals(r, "mean_token_entropy") if v is not None]
    if len(H) < 3:
        return 0.0
    d2 = [abs(H[i + 1] - 2 * H[i] + H[i - 1]) for i in range(1, len(H) - 1)]
    return max(d2) if d2 else 0.0


def hvr_max(r):
    vs = [v for v in step_vals(r, "hvr") if v is not None]
    return max(vs) if vs else 0.0


def flare_max_per_step(r):
    """FLARE: per step, 1 - min(per_token_max_logprob_alt) (probability not logprob).
    Then max over steps. Uses the pre-rep-penalty distribution (`_alt`) per the
    `pilot_audit.py` convention. Fall back to `per_token_max_logprob` if alt missing.
    """
    out = []
    for s in r.get("steps") or []:
        probs = s.get("per_token_max_logprob_alt") or s.get("per_token_max_logprob") or []
        probs = [p for p in probs if p is not None]
        if not probs:
            continue
        try:
            out.append(1.0 - float(min(probs)))
        except Exception:
            continue
    return max(out) if out else None


def seakr_k5_logdet(r, alpha=1e-3):
    """Gram log-det of K=5 sample EOS hidden states per step → max over steps.
    A K=5 proxy for SeaKR's K=20 eigen_score. Returns None if any step lacks
    enough sample hidden states.
    """
    out = []
    for s in r.get("steps") or []:
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
        if len(H) < 2:
            continue
        try:
            M = np.stack(H, axis=0)  # K x d
            G = M @ M.T  # K x K
            G = G + alpha * np.eye(G.shape[0])
            sign, logdet = np.linalg.slogdet(G)
            if sign > 0 and np.isfinite(logdet):
                out.append(float(logdet))
        except Exception:
            continue
    return max(out) if out else None


def normalize_answer(s: str) -> str:
    s = re.sub(r"[^a-z0-9 ]", "", s.strip().lower())[:30].strip()
    return s


def extract_answer(text: str):
    if not text:
        return None
    m = ANSWER_RE.search(text)
    if not m:
        return None
    a = normalize_answer(m.group(1))
    return a if a else None


def per_step_answers(steps, k_top: int):
    out = []
    for s in steps:
        samples = s.get("samples", []) or []
        ans = []
        for samp in samples[:k_top]:
            a = extract_answer(samp.get("text", ""))
            if a:
                ans.append(a)
        out.append(ans)
    return out


def flip_rate_majority(per_step):
    seq = []
    for ans in per_step:
        if not ans:
            continue
        seq.append(Counter(ans).most_common(1)[0][0])
    if len(seq) < 2:
        return None
    flips = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])
    return flips / max(1, len(seq) - 1)


def flip_rate_single(per_step):
    seq = []
    for ans in per_step:
        if not ans:
            continue
        seq.append(ans[0])
    if len(seq) < 2:
        return None
    flips = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])
    return flips / max(1, len(seq) - 1)


def late_flips_count(per_step, n_steps_total):
    """V3 from _explore_ans_flip_variants.py: flips in the last 50% of steps."""
    answer_steps = []
    for i, ans in enumerate(per_step):
        if not ans:
            continue
        answer_steps.append((i, set(ans)))
    if len(answer_steps) < 2:
        return None
    mid = n_steps_total // 2
    late = [(i, s) for i, s in answer_steps if i >= mid]
    if len(late) < 2:
        return 0.0
    return float(sum(1 for j in range(1, len(late)) if late[j][1] != late[j - 1][1]))


def min_majority_inv(per_step):
    """V5: 1 - min(majority fraction across steps). High = more uncertain."""
    fracs = []
    for ans in per_step:
        if not ans:
            continue
        c = Counter(ans).most_common(1)[0][1]
        fracs.append(c / len(ans))
    if not fracs:
        return None
    return 1.0 - min(fracs)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_closed(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.open():
        if not line.strip():
            continue
        c = json.loads(line)
        c_em = int(bool(c.get("label_correct_em", c.get("em", False))))
        steps = c.get("steps") or []

        # Multi-sample features
        psa5 = per_step_answers(steps, k_top=5)
        psa1 = [[a[0]] if a else [] for a in psa5]  # K=1 = first sample

        flip_k5 = flip_rate_majority(psa5)
        flip_k1 = flip_rate_single(psa5)
        late_f = late_flips_count(psa5, len(steps))
        min_maj = min_majority_inv(psa5)

        u = tmax(c, "u_cons")
        d = tmax(c, "dragin_rind_max")
        e = tmax(c, "mean_token_entropy")

        rows.append({
            "id": c.get("id"),
            "type": c.get("type"),
            "level": c.get("level"),
            "closed_em": c_em,
            "n_steps": int(c.get("n_steps", 0)),
            "tokens": int(c.get("tokens_generated", 0)),
            "log_tokens": math.log(max(1, int(c.get("tokens_generated", 0))) + 1),
            "wall_s": float(c.get("wall_clock_s") or 0.0),

            # Trigger pool from v2 audit
            "u_cons_max": u if u is not None else 0.0,
            "dragin_max": d if d is not None else 0.0,
            "ent_max": e if e is not None else 0.0,
            "etc": etc_score(c),
            "hvr_max": hvr_max(c),
            "drift_max": tmax(c, "drift_cosine") or 0.0,
            "var_max": tmax(c, "variance_l2_window") or 0.0,
            "hidden_range": trange(c, "hidden_norm"),

            # New for R1 (rq1-pilot-v1 schema)
            "flare_max": flare_max_per_step(c),
            "seakr_k5_logdet": seakr_k5_logdet(c),

            # AID family
            "flip_k5": flip_k5,
            "flip_k1": flip_k1,
            "late_flips": late_f,
            "min_majority": min_maj,
        })
    return rows


# --------------------------------------------------------------------------- #
# Stats helpers
# --------------------------------------------------------------------------- #
def auroc_pos(rows, key, is_pos):
    p = [r[key] for r in rows if is_pos(r) and r.get(key) is not None]
    n = [r[key] for r in rows if not is_pos(r) and r.get(key) is not None]
    if not p or not n:
        return float("nan")
    return _auroc(p, n)


def precision_at_k(rows, key, k_frac, is_pos):
    valid = [r for r in rows if r.get(key) is not None]
    n = len(valid)
    if n == 0:
        return 0, float("nan"), float("nan")
    base = sum(1 for r in valid if is_pos(r)) / n
    pairs = sorted(valid, key=lambda r: r[key], reverse=True)
    n_fire = max(1, int(round(k_frac * n)))
    fired = pairs[:n_fire]
    n_pos = sum(1 for r in fired if is_pos(r))
    return n_fire, n_pos / n_fire, base


def ols_residual(rows, key, x_key="log_tokens"):
    valid = [r for r in rows if r.get(key) is not None]
    n = len(valid)
    if n < 3:
        return {id(r): 0.0 for r in rows}, (0.0, 0.0)
    xs = np.array([r[x_key] for r in valid], dtype=float)
    ys = np.array([r[key] for r in valid], dtype=float)
    mx, my = xs.mean(), ys.mean()
    num = ((xs - mx) * (ys - my)).sum()
    den = ((xs - mx) ** 2).sum()
    b = num / den if den > 0 else 0.0
    a = my - b * mx
    out = {}
    for r in rows:
        if r.get(key) is None:
            out[id(r)] = None
        else:
            out[id(r)] = float(r[key] - (a + b * r[x_key]))
    return out, (a, b)


def bootstrap_auroc_diff(rows, key_a, key_b, is_pos, B=10_000, seed=11):
    valid = [r for r in rows if r.get(key_a) is not None and r.get(key_b) is not None]
    n = len(valid)
    if n < 4:
        return float("nan"), float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        sample = [valid[i] for i in idx]
        a = auroc_pos(sample, key_a, is_pos)
        b = auroc_pos(sample, key_b, is_pos)
        if a == a and b == b:
            diffs.append(a - b)
    if not diffs:
        return float("nan"), float("nan"), float("nan"), float("nan")
    diffs.sort()
    m = statistics.mean(diffs)
    lo = diffs[int(0.025 * (len(diffs) - 1))]
    hi = diffs[int(0.975 * (len(diffs) - 1))]
    ppos = sum(1 for d in diffs if d > 0) / len(diffs)
    return m, lo, hi, ppos


# --------------------------------------------------------------------------- #
# Signal definitions
# --------------------------------------------------------------------------- #
SIGNALS = (
    # (key, display, family)
    ("tokens",         "tokens_generated",          "external/length"),
    ("wall_s",         "wall_clock_s",              "external/length"),
    ("n_steps",        "n_steps",                   "external/length"),
    ("ent_max",        "ent.max",                   "logit"),
    ("etc",            "etc (Δ²H max)",             "logit"),
    ("flare_max",      "flare.max (1-min p)",       "logit"),
    ("dragin_max",     "dragin_rind_max.max",       "attention"),
    ("drift_max",      "drift_cosine.max",          "hidden CRV"),
    ("var_max",        "variance_l2_window.max",    "hidden CRV"),
    ("hidden_range",   "hidden_norm.range",         "hidden CRV"),
    ("hvr_max",        "selfdoubt_t1 (hvr.max)",    "text"),
    ("u_cons_max",     "u_cons.max (exploratory)",  "K-sample-text"),
    ("seakr_k5_logdet","seakr_k5_proxy.logdet",     "K-sample-hidden"),
    ("flip_k5",        "AID flip_rate (K=5 ours)",  "AID/K-sample-answer"),
    ("flip_k1",        "AID flip_rate (K=1 Datta)", "AID/K-sample-answer"),
    ("late_flips",     "AID late_flips",            "AID/K-sample-answer"),
    ("min_majority",   "AID min_majority (1-min)",  "AID/K-sample-answer"),
)


def banner(s: str):
    print(f"\n{'=' * 100}")
    print(s)
    print("=" * 100)


def report(rows: list[dict], label: str, B: int):
    n = len(rows)
    n_inc = sum(1 for r in rows if r["closed_em"] == 0)
    base_inc = n_inc / n if n else float("nan")
    banner(f"{label}    n={n}    incorrect={n_inc}    base_rate(incorrect)={base_inc:.3f}")
    is_inc = lambda r: r["closed_em"] == 0

    # Raw AUROC + P@k
    print("\n--- (1) RAW AUROC + P@k against closed_em == 0 ---")
    print(f"  {'signal':<32} {'family':<22} {'pop':>5} {'AUROC':>7} {'P@5%':>7} {'P@10%':>7} {'P@20%':>7}   {'ΔP@5%':>7}")
    for key, name, family in SIGNALS:
        pop = sum(1 for r in rows if r.get(key) is not None)
        au = auroc_pos(rows, key, is_inc)
        _, p5, _ = precision_at_k(rows, key, 0.05, is_inc)
        _, p10, _ = precision_at_k(rows, key, 0.10, is_inc)
        _, p20, _ = precision_at_k(rows, key, 0.20, is_inc)
        d5 = p5 - base_inc if p5 == p5 else float("nan")
        print(f"  {name:<32} {family:<22} {pop:>5} {au:>7.3f} {p5:>7.3f} {p10:>7.3f} {p20:>7.3f}   {d5:>+7.3f}")

    # Length-residualized AUROC
    print("\n--- (2) LENGTH-RESIDUALIZED AUROC (signal ~ log_tokens, evaluate residual) ---")
    print(f"  {'signal':<32} {'raw AUROC':>10} {'resid AUROC':>12} {'ΔAUROC':>8}   {'raw P@5%':>9} {'resid P@5%':>11}")
    for key, name, family in SIGNALS:
        if key == "tokens":
            continue
        valid = [r for r in rows if r.get(key) is not None]
        if not valid:
            continue
        resid_map, (a_, b_) = ols_residual(valid, key, x_key="log_tokens")
        for r in valid:
            r[f"{key}_resid"] = resid_map.get(id(r))
        au_raw = auroc_pos(rows, key, is_inc)
        au_res = auroc_pos(rows, f"{key}_resid", is_inc)
        _, p5_raw, _ = precision_at_k(rows, key, 0.05, is_inc)
        _, p5_res, _ = precision_at_k(rows, f"{key}_resid", 0.05, is_inc)
        delta = au_res - au_raw if (au_res == au_res and au_raw == au_raw) else float("nan")
        print(f"  {name:<32} {au_raw:>10.3f} {au_res:>12.3f} {delta:>+8.3f}   {p5_raw:>9.3f} {p5_res:>11.3f}")

    # Paired bootstrap vs tokens (RAW)
    print(f"\n--- (3) PAIRED BOOTSTRAP (signal − tokens) AUROC vs closed_em=0, B={B} ---")
    print(f"  {'signal':<32} {'Δraw vs tokens':>16} {'95% CI':>22} {'sig?':>6}")
    for key, name, family in SIGNALS:
        if key == "tokens":
            continue
        m, lo, hi, _ = bootstrap_auroc_diff(rows, key, "tokens", is_inc, B=B)
        sig = "✓" if (lo == lo and hi == hi and (lo > 0 or hi < 0)) else "—"
        print(f"  {name:<32} {m:>+16.3f}  [{lo:>+6.3f},{hi:>+6.3f}]  {sig:>6}")

    # Per-question-type
    print("\n--- (4) PER-QUESTION-TYPE AUROC (raw, vs closed_em=0) ---")
    types = sorted({r["type"] for r in rows if r.get("type")})
    if types:
        head = "  " + f"{'signal':<32}" + " ".join(f"{t[:14]:>14}" for t in types) + "   pop"
        print(head)
        for key, name, _ in SIGNALS:
            line = f"  {name:<32}"
            for t in types:
                sub = [r for r in rows if r.get("type") == t and r.get(key) is not None]
                p = [r[key] for r in sub if r["closed_em"] == 0]
                n = [r[key] for r in sub if r["closed_em"] == 1]
                if not p or not n:
                    line += f" {'n/a':>14}"
                else:
                    line += f" {_auroc(p, n):>14.3f}"
            line += f"   ({len(rows)})"
            print(line)


def report_a1_ablation(rows: list[dict], label: str, B: int):
    """A1 K=1 vs K=5 ablation (closed_em=0 target only — lift_em blocked here)."""
    banner(f"A1 K=1 vs K=5 — {label} (closed_em=0 target only; lift_em blocked until oracle lands)")
    is_inc = lambda r: r["closed_em"] == 0
    valid = [r for r in rows if r.get("flip_k1") is not None and r.get("flip_k5") is not None]
    n = len(valid)
    n_inc = sum(1 for r in valid if is_inc(r))
    print(f"  n_valid={n}    incorrect={n_inc}")

    a_k1 = auroc_pos(valid, "flip_k1", is_inc)
    a_k5 = auroc_pos(valid, "flip_k5", is_inc)

    # Residualized
    rmap1, _ = ols_residual(valid, "flip_k1")
    rmap5, _ = ols_residual(valid, "flip_k5")
    for r in valid:
        r["flip_k1_resid"] = rmap1.get(id(r))
        r["flip_k5_resid"] = rmap5.get(id(r))
    ar_k1 = auroc_pos(valid, "flip_k1_resid", is_inc)
    ar_k5 = auroc_pos(valid, "flip_k5_resid", is_inc)

    m_raw, lo_raw, hi_raw, _ = bootstrap_auroc_diff(valid, "flip_k1", "flip_k5", is_inc, B=B, seed=20260514)
    m_res, lo_res, hi_res, _ = bootstrap_auroc_diff(valid, "flip_k1_resid", "flip_k5_resid", is_inc, B=B, seed=20260515)

    # bootstrap_auroc_diff returns (a - b), so K=5 - K=1 = -m
    m_raw_kdelta, lo_raw_kdelta, hi_raw_kdelta = -m_raw, -hi_raw, -lo_raw
    m_res_kdelta, lo_res_kdelta, hi_res_kdelta = -m_res, -hi_res, -lo_res

    print(f"\n  metric                         K=1     K=5     Δ(K5-K1)    95% CI")
    print(f"  ---------------------------- -------  ------ ----------- -------------------")
    print(f"  raw AUROC                    {a_k1:>7.3f} {a_k5:>7.3f}  {m_raw_kdelta:>+11.3f}  [{lo_raw_kdelta:>+6.3f},{hi_raw_kdelta:>+6.3f}]")
    print(f"  length-residualized AUROC    {ar_k1:>7.3f} {ar_k5:>7.3f}  {m_res_kdelta:>+11.3f}  [{lo_res_kdelta:>+6.3f},{hi_res_kdelta:>+6.3f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d1", type=str, default="runs/v2_headline_phi-4-mini-instruct/D1_closed_hotpot.jsonl")
    ap.add_argument("--d2", type=str, default="runs/v2_headline_phi-4-mini-instruct/D2_closed_wiki.jsonl")
    ap.add_argument("--bootstrap-b", type=int, default=10_000)
    ap.add_argument("--out", type=str, default="analysis/r1_phi_d1_closed_audit.json")
    ap.add_argument("--seed", type=int, default=20260514)
    ap.add_argument("--include-v2-comparison", action="store_true",
                    help="Also load runs/v2_exp1_phi-4-mini-instruct/D1_closed_hotpot.jsonl as a sanity-check comparison cohort.")
    args = ap.parse_args()

    d1_path = ROOT / args.d1
    d2_path = ROOT / args.d2

    print(f"Loading R1 D1: {d1_path}")
    d1_rows = load_closed(d1_path)
    print(f"  {len(d1_rows)} traces  (incorrect={sum(1 for r in d1_rows if r['closed_em']==0)})")

    print(f"\nLoading R1 D2 (likely partial): {d2_path}")
    d2_rows = load_closed(d2_path)
    print(f"  {len(d2_rows)} traces  (incorrect={sum(1 for r in d2_rows if r['closed_em']==0)})")

    if d1_rows:
        report(d1_rows, "R1 — Phi-4-mini-instruct × HotpotQA-distractor (D1, closed-book)", B=args.bootstrap_b)
        report_a1_ablation(d1_rows, "R1 D1", B=args.bootstrap_b)

    if len(d2_rows) >= 30:
        report(d2_rows, "R1 — Phi × 2WikiMHQA-distractor (D2, closed-book, partial)", B=args.bootstrap_b)
        report_a1_ablation(d2_rows, "R1 D2 (partial)", B=args.bootstrap_b)
    elif d2_rows:
        print(f"\n[skip D2 audit — only {len(d2_rows)} traces, too small for stable AUROC + bootstrap]")

    if args.include_v2_comparison:
        print("\n" + "#" * 100)
        print("# SANITY-CHECK comparison: existing v2 Phi cohort (n=393 paired HotpotQA).")
        print("# These numbers are from `runs/v2_exp1_phi-4-mini-instruct/D1_closed_hotpot.jsonl`.")
        print("# Same closed-book cohort; we run the same closed_em=0 audit. R1 D1 (n=200)")
        print("# should reproduce direction-of-effect on every signal.")
        print("#" * 100)
        v2_path = ROOT / "runs/v2_exp1_phi-4-mini-instruct/D1_closed_hotpot.jsonl"
        v2_rows = load_closed(v2_path)
        if v2_rows:
            report(v2_rows, "V2 SANITY — Phi × HotpotQA (n=393, prior-cohort comparison)",
                   B=min(args.bootstrap_b, 10_000))

    out_path = ROOT / args.out
    payload = {
        "schema_version": "r1-d1-closed-audit-v1",
        "n_d1": len(d1_rows),
        "n_d2": len(d2_rows),
        "bootstrap_b": args.bootstrap_b,
        "seed": args.seed,
        "d1_path": str(d1_path.relative_to(ROOT)) if d1_path.exists() else None,
        "d2_path": str(d2_path.relative_to(ROOT)) if d2_path.exists() else None,
        "summary": "stdout dump in r1_phi_d1_closed_audit.out; rerun on full R1 once oracle lands.",
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out_path.relative_to(ROOT)}")
    print("\nNote: lift_em=+1 analyses BLOCKED until E1_oracle_hotpot.jsonl lands.")
    print("Re-run with `pilot_audit.py --prefix v2_headline --bootstrap-b 10000` once R1 oracle lands.")


if __name__ == "__main__":
    main()
