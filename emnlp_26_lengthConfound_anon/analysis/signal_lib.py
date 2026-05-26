"""Signal-extraction and stats library shared across the analysis scripts.

This module hosts the per-trace signal extractors for the published
training-free dynamic-RAG trigger pool (DRAGIN, FLARE, SeaKR-proxy, ETC,
SelfCheckGPT-style u_cons, ...) plus the answer-trajectory aggregations
introduced in this paper (flip_rate, late_flips, min_majority — and FCT
lives in `moonshot_explore.py`). It also exposes the AUROC / paired-
bootstrap / OLS-residualization primitives consumed by every downstream
script.

All functions are pure and accept the raw `rq1-pilot-v1` schema trace
dictionaries (one per line in the JSONL files under `runs/v2_headline_*/`).

Imported by:
    - r1_paired_audit.py
    - moonshot_explore.py
    - moonshot_ensemble.py
    - e7_bonferroni_and_spearman.py
    - uncertainty_line_audit.py
    - llm_judge_audit.py
"""
from __future__ import annotations

import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _common import auroc as _auroc

ROOT = HERE.parent

ANSWER_RE = re.compile(r"\bAnswer\s*:\s*(.+?)(?:[\n.]|$)", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Per-step / per-trace scalar extractors
# --------------------------------------------------------------------------- #
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
    """ETC (Li et al. 2025): max absolute second-difference of step entropy.

    Δ²H_i = H_{i+1} − 2 H_i + H_{i−1}; we report max_i |Δ²H_i| over the
    trace.  Distinct from our novel FCT signal (which is a forward-looking
    first-commitment-time on the answer trajectory, not on the entropy
    series).  Naming-collision note in §2 of the paper.
    """
    H = [v for v in step_vals(r, "mean_token_entropy") if v is not None]
    if len(H) < 3:
        return 0.0
    d2 = [abs(H[i + 1] - 2 * H[i] + H[i - 1]) for i in range(1, len(H) - 1)]
    return max(d2) if d2 else 0.0


def hvr_max(r):
    vs = [v for v in step_vals(r, "hvr") if v is not None]
    return max(vs) if vs else 0.0


def flare_max_per_step(r):
    """FLARE: per step, 1 - min(per_token_max_logprob_alt) (probability, not
    log-prob).  Then max over steps.  Uses the pre-rep-penalty distribution
    (`_alt`) when available, falling back to `per_token_max_logprob`.
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
    """SeaKR K=5 proxy: Gram log-det of K=5 sample EOS hidden states per
    step, max over steps.  Returns None if no step has enough samples.
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
            G = M @ M.T              # K x K
            G = G + alpha * np.eye(G.shape[0])
            sign, logdet = np.linalg.slogdet(G)
            if sign > 0 and np.isfinite(logdet):
                out.append(float(logdet))
        except Exception:
            continue
    return max(out) if out else None


# --------------------------------------------------------------------------- #
# Answer-trajectory primitives
# --------------------------------------------------------------------------- #
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
    """flips in the last 50% of steps (post-boundary count)."""
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
    """1 - min(majority fraction across steps).  High = more uncertain."""
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
    """Paired bootstrap of Δ = AUROC(key_a) − AUROC(key_b).

    Returns (point estimate, 2.5%, 97.5%, P(Δ>0)).  The standard nominal-95%
    CI consumed by Table 1 in the paper.
    """
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
# Signal pool (display labels and family map)
# --------------------------------------------------------------------------- #
# Used by `r1_paired_audit.py` for the per-arch report.
SIGNALS = (
    # (key, display, family)
    ("tokens",         "tokens_generated",          "external/length"),
    ("wall_s",         "wall_clock_s",              "external/length"),
    ("n_steps",        "n_steps",                   "external/length"),
    ("ent_max",        "ent.max",                   "logit"),
    ("etc",            "etc (Δ²H max)",             "logit"),
    ("flare_max",      "flare.max (1-min p)",       "logit"),
    ("dragin_max",     "DRAGIN-RIND.max",           "attention"),
    ("drift_max",      "drift_cosine.max",          "hidden CRV"),
    ("var_max",        "var_l2_window.max",         "hidden CRV"),
    ("hidden_range",   "hidden_norm.range",         "hidden CRV"),
    ("hvr_max",        "hvr.max (SELFDOUBT-T1)",    "text"),
    ("u_cons_max",     "u_cons.max",                "K-sample-text"),
    ("seakr_k5_logdet","seakr_k5_proxy.logdet",     "K-sample-hidden"),
    ("flip_k5",        "flip_rate K=5",             "answer-trajectory"),
    ("flip_k1",        "flip_rate K=1",             "answer-trajectory"),
    ("late_flips",     "late_flips",                "answer-trajectory"),
    ("min_majority",   "min_majority",              "answer-trajectory"),
)
