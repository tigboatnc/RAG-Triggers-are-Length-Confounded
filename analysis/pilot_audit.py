"""RQ1 Experiment P0 — pilot audit on the rq1-pilot-v1 schema.

Consumes:
  runs/v2_pilot_<arch>/{D1_closed_hotpot, D2_closed_wiki,
                        E1_oracle_hotpot, E2_oracle_wiki}.jsonl
for each arch in {phi-4-mini-instruct, gemma-4-26b-a4b-it,
qwen3.6-35b-a3b-instruct}, and emits a per-cell trigger-pool table.

For each (arch × dataset) cell, computes raw + length-residualized AUROC
+ paired bootstrap (B=2000) vs `tokens_generated` for:

  Existing v2-pool signals (length_audit_v2.py compatible):
    tokens_generated, wall_clock_s, n_steps,
    dragin_rind_max.max, mean_token_entropy.max, etc (Δ²H max),
    selfdoubt_t1 (hvr.max), drift_cosine.max, variance_l2_window.max,
    hidden_norm.range, u_cons.max (exploratory).

  NEW pilot-only signals (rq1-pilot-v1 schema):
    flare.min_logprob_post_rep      → 1 - min(per_token_max_logprob) per step → max
    flare.min_logprob_pre_rep       → same on pre-penalty distribution
    seakr_K5_proxy.gram_logdet      → log det(G + α I) over K=5 sample EOS hidden
                                      states; max over steps.
    selfcheck_ngram.unigram_neg_logprob → SelfCheckGPT-Ngram-1: surprisal of greedy
                                      step text under unigram model built from K
                                      sample texts; max over steps.
    capture_overhead_s              → diagnostic, not a trigger.

SelfCheckGPT-NLI is GPU-side (DeBERTa-v3-large-MNLI); enabled with
``--with-nli``. Otherwise the offline run is CPU-only.

Pilot is direction-of-effect only — n=25 per cell is too small for
paired-bootstrap headline significance. Reports CIs anyway; flags are
indicative, not load-bearing.
"""
from __future__ import annotations
import argparse
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _common import auroc

ROOT = HERE.parent
ARCHS = ["phi-4-mini-instruct", "gemma-4-26b-a4b-it", "qwen3.6-35b-a3b-instruct"]
DATASETS = [
    ("HotpotQA  (in-domain)", "D1_closed_hotpot.jsonl", "E1_oracle_hotpot.jsonl"),
    ("2WikiMHQA (OOD)",       "D2_closed_wiki.jsonl",   "E2_oracle_wiki.jsonl"),
]


# --------------------------------------------------------------------- #
# Per-trace feature extractors
# --------------------------------------------------------------------- #
def f(v):
    try: return float(v) if v is not None else None
    except Exception: return None


def step_vals(r, key):
    return [f(s.get(key)) for s in r.get("steps") or []]


def tmax(r, key):
    vs = [v for v in step_vals(r, key) if v is not None]
    return max(vs) if vs else None


def trange(r, key):
    vs = [v for v in step_vals(r, key) if v is not None]
    if len(vs) < 2: return 0.0
    return max(vs) - min(vs)


def etc_score(r):
    """ETC (Li 2025): max |Δ²H| of mean_token_entropy series."""
    H = [v for v in step_vals(r, "mean_token_entropy") if v is not None]
    if len(H) < 3: return 0.0
    d2 = [abs(H[i+1] - 2*H[i] + H[i-1]) for i in range(1, len(H)-1)]
    return max(d2) if d2 else 0.0


def hvr_max(r):
    vs = [v for v in step_vals(r, "hvr") if v is not None]
    return max(vs) if vs else 0.0


def flare_min_logprob_step(step, key="per_token_max_logprob"):
    """Per-step FLARE signal: 1 - min(per-token-max-prob).

    Returns None if the field is missing (i.e. v2 trace, not rq1-pilot-v1).
    """
    arr = step.get(key)
    if not arr: return None
    try:
        vals = [float(p) for p in arr if p is not None]
    except Exception:
        return None
    if not vals: return None
    return float(1.0 - min(vals))


def flare_max_over_trace(r, key="per_token_max_logprob"):
    vals = []
    for step in (r.get("steps") or []):
        v = flare_min_logprob_step(step, key=key)
        if v is not None: vals.append(v)
    return max(vals) if vals else None


def seakr_k5_logdet_step(step, alpha: float = 1e-3):
    """SeaKR-K5 proxy per step: log det(G + α I).

    G is the K×K Gram matrix of the per-sample EOS hidden vectors.
    Returns None if hidden states are missing or K<2.
    """
    samples = step.get("samples")
    if not samples: return None
    H = []
    for s in samples:
        h = s.get("eos_hidden")
        if h is None: continue
        H.append(h)
    if len(H) < 2: return None
    try:
        import numpy as np  # local import; analysis box has it
    except Exception:
        return None
    arr = np.asarray(H, dtype=np.float64)  # [K, D]
    # Normalize each row (matches SeaKR's L2-normed hidden treatment).
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    arr = arr / norms
    G = arr @ arr.T  # [K, K]
    K = G.shape[0]
    G = G + alpha * np.eye(K)
    try:
        sign, logabsdet = np.linalg.slogdet(G)
        if sign <= 0: return None
        return float(logabsdet)
    except Exception:
        return None


def seakr_k5_max_over_trace(r):
    vals = []
    for step in (r.get("steps") or []):
        v = seakr_k5_logdet_step(step)
        if v is not None: vals.append(v)
    return max(vals) if vals else None


def selfcheck_ngram_step(step, n: int = 1):
    """SelfCheckGPT-Ngram-n: max neg-log-prob of the greedy step text under
    a unigram model built from the K sample texts.

    Returns None if samples are missing.
    """
    samples = step.get("samples")
    if not samples: return None
    sample_texts = [s.get("text", "") for s in samples if s.get("text")]
    if not sample_texts: return None
    step_text = step.get("step_text", "") or ""
    if not step_text: return None
    # Tokenize cheaply (whitespace + lowercase).
    def toks(t): return [w for w in t.lower().split() if w]
    bag = []
    for s in sample_texts:
        bag.extend(toks(s))
    if not bag: return None
    counts = defaultdict(int)
    for w in bag:
        counts[w] += 1
    V = len(counts)
    N = sum(counts.values())
    # Add-one smoothing.
    def neg_log_prob(w):
        return -math.log((counts.get(w, 0) + 1) / (N + V + 1))
    surprisals = [neg_log_prob(w) for w in toks(step_text)]
    return max(surprisals) if surprisals else None


def selfcheck_ngram_max_over_trace(r):
    vals = []
    for step in (r.get("steps") or []):
        v = selfcheck_ngram_step(step)
        if v is not None: vals.append(v)
    return max(vals) if vals else None


# --------------------------------------------------------------------- #
# Pair load + feature dump
# --------------------------------------------------------------------- #
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
        # Required (legacy) fields for cohort filter
        u = tmax(c, "u_cons"); d = tmax(c, "dragin_rind_max"); e = tmax(c, "mean_token_entropy")
        if d is None or e is None:
            continue
        row = {
            "id": qid, "type": c.get("type"), "level": c.get("level"),
            "closed_em": c_em, "lift_em": lift,
            "n_steps": int(c.get("n_steps", 0)),
            "tokens": int(c.get("tokens_generated", 0)),
            "log_tokens": math.log(max(1, int(c.get("tokens_generated", 0))) + 1),
            "wall_s": float(c.get("wall_clock_s") or 0),
            # Legacy v2 pool
            "u_cons_max": u if u is not None else 0.0,
            "dragin_max": d, "ent_max": e,
            "etc": etc_score(c),
            "hvr_max": hvr_max(c),
            "drift_max": tmax(c, "drift_cosine") or 0.0,
            "var_max": tmax(c, "variance_l2_window") or 0.0,
            "hidden_range": trange(c, "hidden_norm"),
            # NEW pilot-only signals (None on v2 traces)
            "flare_post": flare_max_over_trace(c, "per_token_max_logprob"),
            "flare_pre":  flare_max_over_trace(c, "per_token_max_logprob_alt"),
            "seakr_k5":   seakr_k5_max_over_trace(c),
            "sc_ngram":   selfcheck_ngram_max_over_trace(c),
            # Bookkeeping
            "schema":     c.get("capture_schema_version", "v2"),
            "K":          int(c.get("sampler_K") or 0),
            "cap_overhead_s": float(c.get("per_step_capture_overhead_s") or 0.0),
        }
        rows.append(row)
    return rows


# --------------------------------------------------------------------- #
# Audit machinery (mirrors length_audit_v2.py)
# --------------------------------------------------------------------- #
def auroc_pos(rows, key, is_pos):
    p = [r[key] for r in rows if is_pos(r) and r.get(key) is not None]
    n = [r[key] for r in rows if not is_pos(r) and r.get(key) is not None]
    if not p or not n: return float("nan")
    return auroc(p, n)


def precision_at_k(rows, key, k_frac, is_pos):
    rows_ = [r for r in rows if r.get(key) is not None]
    n = len(rows_)
    if n == 0: return 0, float("nan"), float("nan")
    base = sum(1 for r in rows if is_pos(r)) / max(1, len(rows))
    pairs = sorted(rows_, key=lambda r: r[key], reverse=True)
    n_fire = max(1, int(round(k_frac * n)))
    fired = pairs[:n_fire]
    n_pos = sum(1 for r in fired if is_pos(r))
    return n_fire, n_pos / n_fire, base


def ols_residual(rows, key, x_key="log_tokens"):
    pts = [(r[x_key], r[key]) for r in rows if r.get(key) is not None]
    if len(pts) < 3: return [None] * len(rows), (0.0, 0.0)
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    n = len(pts); mx = sum(xs)/n; my = sum(ys)/n
    num = sum((xs[i]-mx)*(ys[i]-my) for i in range(n))
    den = sum((xs[i]-mx)**2 for i in range(n))
    b = num/den if den > 0 else 0.0; a = my - b*mx
    out = []
    for r in rows:
        v = r.get(key)
        if v is None: out.append(None)
        else: out.append(float(v - (a + b * r[x_key])))
    return out, (a, b)


def bootstrap_auroc_diff(rows, key_a, key_b, is_pos, B=2000, seed=11):
    rng = random.Random(seed)
    n = len(rows)
    diffs = []
    for _ in range(B):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        a = auroc_pos(sample, key_a, is_pos)
        b = auroc_pos(sample, key_b, is_pos)
        if a == a and b == b:
            diffs.append(a - b)
    if not diffs: return float("nan"), float("nan"), float("nan"), float("nan")
    diffs.sort()
    m = statistics.mean(diffs)
    lo = diffs[int(0.025 * (len(diffs) - 1))]
    hi = diffs[int(0.975 * (len(diffs) - 1))]
    ppos = sum(1 for d in diffs if d > 0) / len(diffs)
    return m, lo, hi, ppos


# (key, display, family, exploratory?)
SIGNAL_DEFS = (
    ("tokens",        "tokens_generated",          "external",        False),
    ("wall_s",        "wall_clock_s",              "external",        False),
    ("n_steps",       "n_steps",                   "external",        False),
    ("dragin_max",    "dragin_rind_max.max",       "internal-attn",   False),
    ("ent_max",       "ent.max",                   "internal-logit",  False),
    ("etc",           "etc (Δ²H max)",             "internal-logit",  False),
    ("hvr_max",       "selfdoubt_t1 (hvr.max)",    "external-text",   False),
    ("drift_max",     "drift_cosine.max",          "internal-hidden", True),
    ("var_max",       "variance_l2_window.max",    "internal-hidden", True),
    ("hidden_range",  "hidden_norm.range",         "internal-hidden", True),
    ("u_cons_max",    "u_cons.max",                "internal-K3",     True),
    # NEW pilot signals
    ("flare_post",    "FLARE.min_p (post-rep)",    "internal-logit",  False),
    ("flare_pre",     "FLARE.min_p (pre-rep)",     "internal-logit",  False),
    ("seakr_k5",      "SeaKR.K5_logdet",           "internal-hidden-K5", False),
    ("sc_ngram",      "SelfCheckGPT.ngram",        "internal-K5-text", False),
)


def banner(s):
    print(f"\n{'='*92}\n{s}\n{'='*92}")


def report_cell(label, rows):
    n = len(rows)
    n_inc = sum(1 for r in rows if r["closed_em"] == 0)
    n_help = sum(1 for r in rows if r["lift_em"] == +1)
    base_help = (n_help / n) if n else 0.0
    banner(f"{label}   n={n}  incorrect={n_inc}  lift+1={n_help} (base={base_help:.3f})")
    if n < 5 or n_help < 2:
        print(f"[skip] cohort too small for AUROC ({n} traces, {n_help} helpful)")
        return

    is_help = lambda r: r["lift_em"] == +1

    # (1) Raw AUROC + precision-at-budget
    print("\n--- (1) RAW headline table — TARGET=lift_em=+1 ---")
    print(f"  {'signal':<28} {'family':<22} {'AUROC':>7} {'P@10%':>7} {'P@20%':>7}   {'Δprec@10%':>10}  {'pop':>4}")
    for key, name, family, _ in SIGNAL_DEFS:
        au = auroc_pos(rows, key, is_help)
        pop = sum(1 for r in rows if r.get(key) is not None)
        _, p10, _ = precision_at_k(rows, key, 0.10, is_help)
        _, p20, _ = precision_at_k(rows, key, 0.20, is_help)
        au_s = f"{au:>7.3f}" if au == au else f"{'n/a':>7}"
        p10_s = f"{p10:>7.3f}" if p10 == p10 else f"{'n/a':>7}"
        p20_s = f"{p20:>7.3f}" if p20 == p20 else f"{'n/a':>7}"
        d_s = f"{(p10 - base_help):>+10.3f}" if p10 == p10 else f"{'n/a':>10}"
        print(f"  {name:<28} {family:<22} {au_s} {p10_s} {p20_s}   {d_s}  {pop:>4}/{n}")
    print(f"  {'BASE RATE':<28} {'':<22} {'':<7} {base_help:>7.3f} {base_help:>7.3f}")

    # (2) Length-residualized AUROC
    print("\n--- (2) LENGTH-RESIDUALIZED (regress on log_tokens) — TARGET=lift+1 ---")
    for key, _, _, _ in SIGNAL_DEFS:
        if key == "tokens": continue
        resids, _ = ols_residual(rows, key, x_key="log_tokens")
        for r, x in zip(rows, resids):
            r[f"{key}_resid"] = x
    print(f"  {'signal':<28} {'raw AUROC':>10} {'resid AUROC':>12} {'ΔAUROC':>8}")
    for key, name, _, _ in SIGNAL_DEFS:
        if key == "tokens": continue
        au_raw = auroc_pos(rows, key, is_help)
        au_res = auroc_pos(rows, f"{key}_resid", is_help)
        if au_raw != au_raw or au_res != au_res:
            print(f"  {name:<28} {'n/a':>10} {'n/a':>12} {'n/a':>8}")
            continue
        print(f"  {name:<28} {au_raw:>10.3f} {au_res:>12.3f} {au_res-au_raw:>+8.3f}")

    # (3) Paired bootstrap CI vs tokens (B=2000; pilot-scale, indicative only)
    print("\n--- (3) PAIRED BOOTSTRAP (signal − tokens) AUROC →lift+1, B=2000 ---")
    print("  (CIs at n=25 per cell are wide; indicative direction-of-effect only.)")
    for key, name, _, _ in SIGNAL_DEFS:
        if key == "tokens": continue
        rows_ok = [r for r in rows if r.get(key) is not None and r.get("tokens") is not None]
        if len(rows_ok) < 5:
            print(f"  {name:<28} {'n/a (insufficient rows)':>40}")
            continue
        m, lo, hi, p = bootstrap_auroc_diff(rows_ok, key, "tokens", is_help, B=2000)
        sig = "✓" if (lo > 0 or hi < 0) else "—"
        print(f"  {name:<28} Δ={m:>+7.3f}  [{lo:>+6.3f}, {hi:>+6.3f}]  P>0={p:>4.2f}  {sig}")

    # (4) Capture-overhead diagnostic (G2 cost-vs-signal accounting)
    over = [r["cap_overhead_s"] for r in rows if r.get("cap_overhead_s")]
    if over:
        print(f"\n--- (4) Capture-overhead diagnostic ---")
        print(f"  per-trace K=5 sampler+hidden capture: mean={sum(over)/len(over):.2f}s  "
              f"min={min(over):.2f}s  max={max(over):.2f}s  n_with_overhead={len(over)}/{n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "runs"),
                    help="Pilot run-dir root (default: <repo>/runs)")
    ap.add_argument("--with-nli", action="store_true",
                    help="Also compute SelfCheckGPT-NLI (DeBERTa-v3-large-MNLI; "
                         "needs ~1.5 GB GPU). Off by default.")
    args = ap.parse_args()

    runs_root = Path(args.root)
    print(f"[..] pilot-audit root={runs_root}")

    any_data = False
    for arch in ARCHS:
        run_dir = runs_root / f"v2_pilot_{arch}"
        if not run_dir.exists():
            print(f"[--] {arch}: no run dir at {run_dir}; skipping")
            continue
        any_data = True
        for label, dfile, efile in DATASETS:
            cp = run_dir / dfile; op = run_dir / efile
            if not cp.exists() or not op.exists():
                print(f"[--] {arch} {label}: missing {cp.name} or {efile}; skipping")
                continue
            rows = load_paired(cp, op)
            report_cell(f"{arch}  ::  {label}", rows)

    if not any_data:
        print("[!!] no pilot run dirs found. Run scripts/run_pilot_p0.sh first.")
        sys.exit(1)

    if args.with_nli:
        print("\n[..] --with-nli requested; SelfCheckGPT-NLI scoring not implemented "
              "in this revision (DeBERTa-v3-large-MNLI loader pending). "
              "Use the on-disk K=5 sample texts to compute offline.")


if __name__ == "__main__":
    main()
