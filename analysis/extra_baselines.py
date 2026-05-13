"""Better baselines than DRAGIN: HALT-aligned, Testoni-style, Correa OR,
SelfCheckGPT-trace-level approximations.

What's competitive with DRAGIN in the literature that we have NOT yet
benchmarked from existing data:

  1. **HALT** (Shapiro 2026): top-20 token logprobs as a time-series → GRU.
     We don't have raw top-20 logprobs but we DO have `mean_token_entropy`
     per step — the compressed time-series version of HALT's signal.
     Compute trajectory-shape features (slope, range, terminal-growth,
     monotonicity) which HALT's GRU would learn anyway.

  2. **Testoni** (cited as "Mind the Gap" in `contextual-citations.md`):
     behavioral text features ("Wait" tokens, self-questions, trace
     length). The paper argues hidden states are unnecessary — text-level
     behavior is sufficient. Extract from `step_text` via regex.

  3. **Correa Entropy-Guided Loop**: OR-rule combining perplexity + max
     token entropy + low-confidence-token-count. We approximate
     perplexity from mean(mean_token_entropy), max-entropy from
     max(mean_token_entropy), low-conf-count from how many steps have
     boundary_token_entropy > q70.

  4. **SelfCheckGPT-trace-level approximation** (Manakul 2023): K
     resamples + agreement. We have K=3 step-boundary resamples per
     trace; aggregate u_cons via MEAN (trace-level consistency) rather
     than MAX (any-boundary-disagreement). Mean is closer to
     SelfCheckGPT's trace-aggregated agreement.

  5. **Semantic Entropy-Light** (Farquhar 2024): semantically-clustered
     K samples. We don't have semantic clustering, but we can compute
     u_cons.std across boundaries as a rough analog ("how variable is
     the boundary agreement?").

What we can NOT approximate from existing data (and would need re-decode):
  - FLARE (lookahead generation)
  - Self-RAG / Open-RAG (trained reflection tokens)
  - GRIP (trained control tokens)
  - HALT proper (raw top-k logprobs, not just summary entropy)
  - Kossen SEP (trained semantic-entropy probe)

For each baseline: AUROC →lift+1 raw + length-residualized, plus the
paired bootstrap CI delta vs DRAGIN-RIND.max (the audit floor).
"""
from __future__ import annotations
import json
import math
import random
import re
import statistics
import sys
from pathlib import Path

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

# Testoni-style behavioral patterns. Compiled once.
TESTONI_PATTERNS = {
    "wait": re.compile(r"\b(wait|hold on|actually|hmm|hmmm|um|er)\b", re.I),
    "let_me": re.compile(r"\b(let me|let's)\b", re.I),
    "but": re.compile(r"\bbut\b", re.I),
    "however": re.compile(r"\b(however|nonetheless|nevertheless)\b", re.I),
    "self_q": re.compile(r"\?"),  # any question mark = self-question pattern
    "exclam": re.compile(r"!"),
    "reconsider": re.compile(r"\b(reconsider|reconsidering|rethink|on second thought|let me check|on reflection)\b", re.I),
    "not_sure": re.compile(r"\b(not sure|uncertain|unclear|i don't know|i'm not certain)\b", re.I),
    "correction": re.compile(r"\b(correction|i meant|sorry|i was wrong|actually,)\b", re.I),
}


def f(v):
    try: return float(v) if v is not None else None
    except Exception: return None


def step_vals(r, key):
    return [f(s.get(key)) for s in r.get("steps") or []]


def step_texts(r):
    return [str(s.get("step_text") or "") for s in r.get("steps") or []]


# ----- Baseline feature extractors -----

def halt_features(r):
    """Trajectory-shape features of mean_token_entropy series (HALT-aligned).
    Returns a dict of 7 features."""
    H = [v for v in step_vals(r, "mean_token_entropy") if v is not None]
    B = [v for v in step_vals(r, "boundary_token_entropy") if v is not None]
    out = {}
    if H:
        out["halt_H_max"] = max(H)
        out["halt_H_mean"] = sum(H) / len(H)
        out["halt_H_range"] = max(H) - min(H)
        out["halt_H_terminal_growth"] = (H[-1] - H[0]) if len(H) >= 2 else 0.0
        # slope via least squares
        n = len(H)
        if n >= 2:
            xm = (n - 1) / 2; ym = sum(H) / n
            num = sum((i - xm) * (H[i] - ym) for i in range(n))
            den = sum((i - xm) ** 2 for i in range(n))
            out["halt_H_slope"] = num / den if den > 0 else 0.0
        else:
            out["halt_H_slope"] = 0.0
        # monotonicity (fraction of adjacent increases)
        if n >= 2:
            inc = sum(1 for i in range(1, n) if H[i] > H[i-1])
            out["halt_H_monoton"] = inc / (n - 1)
        else:
            out["halt_H_monoton"] = 0.5
    else:
        for k in ("halt_H_max", "halt_H_mean", "halt_H_range",
                 "halt_H_terminal_growth", "halt_H_slope", "halt_H_monoton"):
            out[k] = 0.0
    out["halt_B_max"] = max(B) if B else 0.0
    return out


def testoni_features(r):
    """Trace-level totals of behavioral pattern hits."""
    texts = step_texts(r)
    full = " ".join(texts)
    out = {}
    for name, pat in TESTONI_PATTERNS.items():
        out[f"tst_{name}"] = float(len(pat.findall(full)))
    # Plus trace length (a la Testoni's length-as-feature)
    n_words = float(len(full.split()))
    out["tst_words"] = n_words
    # And question-density (self_q rate per step)
    n_steps = max(1, len(texts))
    out["tst_self_q_rate"] = out["tst_self_q"] / n_steps
    return out


def correa_features(r):
    """Correa Entropy-Guided Loop OR-rule signal."""
    H = [v for v in step_vals(r, "mean_token_entropy") if v is not None]
    out = {}
    if not H:
        return {"correa_perplex": 0.0, "correa_maxH": 0.0, "correa_lowconf_count": 0.0}
    mean_H = sum(H) / len(H)
    out["correa_perplex"] = math.exp(min(mean_H, 30))  # cap to avoid overflow
    out["correa_maxH"] = max(H)
    # low-confidence step count — number of steps above the trace's own median entropy
    med = sorted(H)[len(H)//2]
    out["correa_lowconf_count"] = float(sum(1 for h in H if h > med))
    return out


def selfcheckgpt_trace_features(r):
    """SelfCheckGPT-trace-level via u_cons MEAN aggregation (consistency
    across all boundaries), and Semantic-Entropy-Light via u_cons STD."""
    U = [v for v in step_vals(r, "u_cons") if v is not None]
    if not U:
        return {"scg_u_mean": 0.0, "scg_u_std": 0.0}
    return {
        "scg_u_mean": sum(U) / len(U),
        "scg_u_std": statistics.pstdev(U) if len(U) >= 2 else 0.0,
    }


def trace_max(r, key):
    vs = [v for v in step_vals(r, key) if v is not None]
    return max(vs) if vs else 0.0


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
        feats = {
            "id": qid, "type": c.get("type"),
            "closed_em": c_em, "lift_em": lift,
            "tokens": int(c.get("tokens_generated", 0)),
            "n_steps": int(c.get("n_steps", 0)),
            "log_tokens": math.log(max(1, int(c.get("tokens_generated", 0))) + 1),
            "dragin_max": trace_max(c, "dragin_rind_max"),  # the floor we compare against
            "u_cons_max": trace_max(c, "u_cons"),           # user's signal, for ref
        }
        feats.update(halt_features(c))
        feats.update(testoni_features(c))
        feats.update(correa_features(c))
        feats.update(selfcheckgpt_trace_features(c))
        rows.append(feats)
    return rows


def auroc_pos(rows, key, is_pos):
    p = [r[key] for r in rows if is_pos(r)]
    n = [r[key] for r in rows if not is_pos(r)]
    if not p or not n:
        return float("nan")
    return auroc(p, n)


def ols_residual(rows, key, x_key="log_tokens"):
    n = len(rows)
    xs = [r[x_key] for r in rows]
    ys = [r[key] for r in rows]
    mx = sum(xs) / n; my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    den = sum((xs[i] - mx) ** 2 for i in range(n))
    b = num / den if den > 0 else 0.0
    a = my - b * mx
    return [ys[i] - (a + b * xs[i]) for i in range(n)]


def boot_diff(rows, k_a, k_b, is_pos, B=2000, seed=11):
    rng = random.Random(seed); n = len(rows); diffs = []
    for _ in range(B):
        s = [rows[rng.randrange(n)] for _ in range(n)]
        a = auroc_pos(s, k_a, is_pos); b = auroc_pos(s, k_b, is_pos)
        if a == a and b == b: diffs.append(a - b)
    if not diffs:
        return float("nan"), float("nan"), float("nan"), float("nan")
    diffs.sort()
    return (statistics.mean(diffs),
            diffs[int(0.025 * (len(diffs)-1))],
            diffs[int(0.975 * (len(diffs)-1))],
            sum(1 for d in diffs if d > 0) / len(diffs))


def precision_at_k(rows, key, k_frac, is_pos):
    n = len(rows); n_pos = sum(1 for r in rows if is_pos(r))
    base = n_pos / n if n else 0.0
    pairs = sorted(rows, key=lambda r: r[key], reverse=True)
    nf = max(1, int(round(k_frac * n)))
    fired = pairs[:nf]
    return sum(1 for r in fired if is_pos(r)) / nf, base


# Baselines organized by family
BASELINES = [
    # (key, display, family)
    ("dragin_max",          "dragin.max",              "DRAGIN-RIND (floor)"),
    # HALT-aligned
    ("halt_H_max",          "halt: H_max",             "HALT-aligned"),
    ("halt_H_mean",         "halt: H_mean",            "HALT-aligned"),
    ("halt_H_range",        "halt: H_range",           "HALT-aligned"),
    ("halt_H_slope",        "halt: H_slope",           "HALT-aligned"),
    ("halt_H_terminal_growth", "halt: H_terminal_growth", "HALT-aligned"),
    ("halt_H_monoton",      "halt: H_monoton",         "HALT-aligned"),
    ("halt_B_max",          "halt: B_max (boundary H)","HALT-aligned"),
    # Testoni behavioral text
    ("tst_wait",            "testoni: wait/hmm count", "Testoni-text"),
    ("tst_let_me",          "testoni: let_me count",   "Testoni-text"),
    ("tst_but",             "testoni: but count",      "Testoni-text"),
    ("tst_self_q",          "testoni: self-Q count",   "Testoni-text"),
    ("tst_self_q_rate",     "testoni: self-Q rate",    "Testoni-text"),
    ("tst_reconsider",      "testoni: reconsider",     "Testoni-text"),
    ("tst_not_sure",        "testoni: not_sure",       "Testoni-text"),
    ("tst_words",           "testoni: word count",     "Testoni-text"),
    # Correa OR-rule components
    ("correa_perplex",      "correa: perplexity",      "Correa-OR"),
    ("correa_maxH",         "correa: max_H",           "Correa-OR"),
    ("correa_lowconf_count","correa: lowconf_count",   "Correa-OR"),
    # SelfCheckGPT / Semantic Entropy approximations
    ("scg_u_mean",          "SelfCheck (u_cons.mean)", "SelfCheckGPT-approx"),
    ("scg_u_std",           "SemEnt-light (u_cons.std)", "SemEnt-approx"),
    # Length floor
    ("tokens",              "tokens (length floor)",   "external floor"),
]


def main():
    for label, cp, op in DATASETS:
        rows = load_paired(cp, op)
        n = len(rows); n_help = sum(1 for r in rows if r["lift_em"] == +1)
        n_inc = sum(1 for r in rows if r["closed_em"] == 0)
        print(f"\n{'='*92}\n=== {label}   n={n}  incorrect={n_inc}  lift+1={n_help} ({n_help/n:.3f}) ===\n")

        is_help = lambda r: r["lift_em"] == +1
        base = n_help / n

        # Residualize every key (vs log_tokens)
        for key, _, _ in BASELINES:
            if key == "tokens": continue
            try:
                resids = ols_residual(rows, key, x_key="log_tokens")
                for r, v in zip(rows, resids):
                    r[f"{key}_resid"] = v
            except Exception:
                pass

        print(f"  {'family':<22} {'feature':<27} {'raw AUROC':>9} {'resid AUROC':>11} {'ΔvsDRAGIN':>10} {'95% CI':>20} {'P@10%':>7}")
        print(f"  {'-'*22} {'-'*27} {'-'*9} {'-'*11} {'-'*10} {'-'*20} {'-'*7}")

        for key, display, family in BASELINES:
            au_raw = auroc_pos(rows, key, is_help)
            resid_key = f"{key}_resid" if key != "tokens" else "tokens"
            au_res = auroc_pos(rows, resid_key, is_help) if key != "tokens" else au_raw
            # vs DRAGIN paired bootstrap
            if key != "dragin_max":
                m, lo, hi, p = boot_diff(rows, key, "dragin_max", is_help, B=1500)
                ci = f"[{lo:+.3f},{hi:+.3f}]"
                sig = "✓" if (lo > 0 or hi < 0) else " "
            else:
                ci = "(self)"
                sig = " "
            p10, _ = precision_at_k(rows, key, 0.10, is_help)
            sig_mark = "✓" if (key != "dragin_max" and au_raw > 0.65) else " "
            print(f"  {family:<22} {display:<27} {au_raw:>9.3f} {au_res:>11.3f} {(au_raw-0.732 if key != 'dragin_max' else 0):>+10.3f} {ci:>20} {p10:>7.3f}  {sig}")
        print(f"  {'BASE RATE':<22} {'':<27} {'':<9} {'':<11} {'':<10} {'':<20} {base:>7.3f}")


if __name__ == "__main__":
    main()
