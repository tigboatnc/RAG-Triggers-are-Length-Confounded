"""P0 free-signal extraction — 8 candidate triggers, all derivable from
existing rq1-pilot-v1 trace dumps without re-decoding.

Three reasons these were not in the v0 audit:
1. They're not in the published-trigger pool (DRAGIN/FLARE/SeaKR/etc.);
   we never thought to look.
2. They exploit modern-model behavior (chain-of-thought structure, K=5
   sample-text answer extraction, decode pacing) the older trigger
   literature predates.
3. They're cheap — derivable from fields the rq1-pilot-v1 capture
   already dumps.

Eight candidates audited against `closed_em=0` (incorrect) and
`lift_em=+1` (helpful):

  S1 K5_answer_agreement   — count distinct `Answer: X` extracted from K=5 sample texts
  S2 first_step_entropy    — mean_token_entropy[step 0] only
  S3 pacing_std            — std of n_tokens_in_step across the trace
  S4 answer_marker_step    — index of step that emitted `Answer:` (or -1)
  S5 repetition_5gram      — count of duplicate 5-grams in trace text
  S6 tok_per_s             — wall_clock_s / tokens_generated (model slowdown)
  S7 active_thinking_count — count of {wait, actually, hmm, but, on second thought, let me reconsider}
  S8 step_novelty_min      — min jaccard(5-grams(step_i), 5-grams(step_{i+1})) — how much each step
                             contributes new content vs. repeats prior; min = most-stuck step

Output: stdout table; per (arch x dataset) AUROC of each signal against
both targets, alongside the P0 v0 pool for direct comparison. CIs not
computed (n=25 per cell; ±0.20 by inspection).
"""
from __future__ import annotations
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _common import auroc

ROOT = HERE.parent
ARCHS = [("Phi", "phi-4-mini-instruct"), ("Gemma", "gemma-4-26b-a4b-it"), ("Qwen2.5", "qwen2.5-7b-instruct")]
DATASETS = [("Hot", "D1_closed_hotpot.jsonl", "E1_oracle_hotpot.jsonl"),
            ("2Wiki", "D2_closed_wiki.jsonl", "E2_oracle_wiki.jsonl")]

ANSWER_EXTRACT_RE = re.compile(r"\bAnswer\s*:\s*(.+?)(?:[\n.]|$)", re.IGNORECASE)
ACTIVE_THINKING_RE = re.compile(
    r"\b(wait|actually|hmm|but |on second thought|let me reconsider|"
    r"i think|let me think|reconsider|hold on|hmm,|wait,)\b", re.IGNORECASE)


def f(v):
    try: return float(v) if v is not None else None
    except: return None


def step_vals(r, k):
    return [f(s.get(k)) for s in r.get("steps") or []]


# --------------------------------------------------------------------- #
# 8 free-signal extractors
# --------------------------------------------------------------------- #
def s1_k5_answer_agreement(r):
    """Number of DISTINCT answers extracted from K=5 sample completions per
    step; trace-level = max over steps (most-divided step). Higher = more
    uncertain."""
    out = []
    for s in r.get("steps") or []:
        samples = s.get("samples") or []
        answers = []
        for samp in samples:
            text = samp.get("text", "")
            m = ANSWER_EXTRACT_RE.search(text)
            if m:
                ans = m.group(1).strip().lower()
                # Crude normalization: first 30 chars, alphanumeric only
                ans = re.sub(r"[^a-z0-9 ]+", "", ans)[:30].strip()
                if ans: answers.append(ans)
        if not answers: continue
        out.append(len(set(answers)) / len(answers))  # 1.0 = all distinct, 0.2 = all same
    return max(out) if out else None


def s2_first_step_entropy(r):
    H = [v for v in step_vals(r, "mean_token_entropy") if v is not None]
    return H[0] if H else None


def s3_pacing_std(r):
    """std of n_tokens_in_step across the trace."""
    n = [s.get("n_tokens_in_step") for s in r.get("steps") or [] if s.get("n_tokens_in_step")]
    if len(n) < 2: return 0.0
    m = sum(n) / len(n)
    return math.sqrt(sum((x - m) ** 2 for x in n) / (len(n) - 1))


def s4_answer_marker_step(r):
    """Which step emitted `Answer:`? (-1 if never). Late = higher number = more reasoning."""
    for i, s in enumerate(r.get("steps") or []):
        text = s.get("step_text", "") or ""
        if re.search(r"\bAnswer\s*:", text, re.IGNORECASE):
            return float(i)
    return -1.0  # never


def s5_repetition_5gram(r):
    """Count of duplicate 5-grams in the trace text."""
    parts = []
    for s in r.get("steps") or []:
        t = s.get("step_text", "") or ""
        parts.append(t)
    full = " ".join(parts).lower().split()
    if len(full) < 6: return 0
    grams = [tuple(full[i:i+5]) for i in range(len(full) - 4)]
    counts = defaultdict(int)
    for g in grams: counts[g] += 1
    return sum(c - 1 for c in counts.values() if c > 1)


def s6_tok_per_s(r):
    """Decode rate; lower = slower = more compute per token = uncertain.
    We *invert* (1 / tok_per_s) so that higher = more uncertain."""
    tok = r.get("tokens_generated", 0)
    wall = r.get("wall_clock_s", 0)
    if not tok or not wall: return None
    return wall / tok  # seconds per token; higher = slower


def s7_active_thinking_count(r):
    """Count of self-correction / active-thinking markers in trace text."""
    n = 0
    for s in r.get("steps") or []:
        t = s.get("step_text", "") or ""
        n += len(ACTIVE_THINKING_RE.findall(t))
    return float(n)


def s8_step_novelty_min(r):
    """Min Jaccard(5-grams(step_i), 5-grams(step_{i+1})) across the trace.
    Higher overlap = step repeats prior = stuck. Min=most-stuck step.
    We *invert* so higher final = more uncertain."""
    def fivegrams(text):
        toks = (text or "").lower().split()
        if len(toks) < 5: return set()
        return set(tuple(toks[i:i+5]) for i in range(len(toks) - 4))
    steps = r.get("steps") or []
    if len(steps) < 2: return None
    overlaps = []
    prev = fivegrams(steps[0].get("step_text", ""))
    for s in steps[1:]:
        curr = fivegrams(s.get("step_text", ""))
        if not (prev | curr):
            prev = curr; continue
        j = len(prev & curr) / max(1, len(prev | curr))
        overlaps.append(j)
        prev = curr
    if not overlaps: return None
    return max(overlaps)  # max overlap = most-stuck transition


# --------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------- #
def auroc_pos(rows, key, is_pos):
    p = [r[key] for r in rows if is_pos(r) and r.get(key) is not None]
    n = [r[key] for r in rows if not is_pos(r) and r.get(key) is not None]
    if not p or not n: return float("nan")
    return auroc(p, n)


def load_paired(d, e):
    closed = {r["id"]: r for r in (json.loads(l) for l in open(d))}
    oracle = {r["id"]: r for r in (json.loads(l) for l in open(e))}
    rows = []
    for qid, c in closed.items():
        if qid not in oracle: continue
        o = oracle[qid]
        c_em = int(bool(c.get("label_correct_em", c.get("em", False))))
        o_em = int(bool(o.get("label_correct_em", o.get("em", False))))
        lift = +1 if o_em > c_em else (-1 if o_em < c_em else 0)
        rows.append({
            "em": c_em, "lift": lift,
            "tokens": int(c.get("tokens_generated", 0)),
            # Existing v0 pool reference
            "dragin": (lambda v: max(v) if v else None)([f(x) for x in step_vals(c, "dragin_rind_max") if x is not None]),
            # 8 NEW free signals
            "s1_k5_ans_disagree": s1_k5_answer_agreement(c),
            "s2_first_ent":       s2_first_step_entropy(c),
            "s3_pacing_std":      s3_pacing_std(c),
            "s4_ans_marker_step": s4_answer_marker_step(c),
            "s5_rep_5gram":       s5_repetition_5gram(c),
            "s6_tok_per_s_inv":   s6_tok_per_s(c),
            "s7_active_think":    s7_active_thinking_count(c),
            "s8_novelty_overlap": s8_step_novelty_min(c),
        })
    return rows


def main():
    print("=" * 110)
    print(" P0 FREE-SIGNAL AUDIT — 8 candidates from existing rq1-pilot-v1 traces, n=25/cell, no re-decode")
    print("=" * 110)

    cells = []
    for atag, adir in ARCHS:
        for dtag, df, ef in DATASETS:
            d = ROOT / f"runs/v2_pilot_{adir}/{df}"
            e = ROOT / f"runs/v2_pilot_{adir}/{ef}"
            if not d.exists() or not e.exists(): continue
            rows = load_paired(d, e)
            cells.append((f"{atag} {dtag}", rows))

    SIGNALS = [
        ("tokens",              "tokens_generated"),
        ("dragin",              "DRAGIN.max (reference)"),
        ("s1_k5_ans_disagree",  "S1 K=5 answer-disagree"),
        ("s2_first_ent",        "S2 first-step entropy"),
        ("s3_pacing_std",       "S3 pacing std"),
        ("s4_ans_marker_step",  "S4 answer-marker step idx"),
        ("s5_rep_5gram",        "S5 5-gram repetition cnt"),
        ("s6_tok_per_s_inv",    "S6 sec/token (slowdown)"),
        ("s7_active_think",     "S7 active-thinking marker"),
        ("s8_novelty_overlap",  "S8 step-overlap (stuck)"),
    ]

    print("\n--- AUROC vs INCORRECT (n=25/cell, ±0.20 CIs) ---\n")
    header = f"{'cell':<14} " + " ".join(f"{label:>14}" for label, _ in [(s[1].split()[0] + ' ' + s[1].split()[1] if len(s[1].split())>1 else s[1], s[0]) for s in SIGNALS])
    # Simpler header:
    print(f"{'cell':<14} " + " ".join(f"{name[:7]:>9}" for name, _ in [(s[1].split()[0], s[0]) for s in SIGNALS]))
    for tag, rows in cells:
        n = len(rows); em = sum(r["em"] for r in rows)
        is_inc = lambda r: r["em"] == 0
        line = f"{tag:<14}"
        for k, _ in SIGNALS:
            v = auroc_pos(rows, k, is_inc)
            line += f" {v:>9.3f}" if v == v else f" {'n/a':>9}"
        print(line)

    print("\n--- AUROC vs lift_em=+1 (helpful) — only cells with >=2 helpful ---\n")
    print(f"{'cell':<14} {'#help':>6}  " + " ".join(f"{name[:7]:>9}" for name, _ in [(s[1].split()[0], s[0]) for s in SIGNALS]))
    for tag, rows in cells:
        n_help = sum(1 for r in rows if r["lift"] == +1)
        if n_help < 2:
            print(f"{tag:<14} {n_help:>6}  (skip, too few)"); continue
        is_help = lambda r: r["lift"] == +1
        line = f"{tag:<14} {n_help:>6}  "
        for k, _ in SIGNALS:
            v = auroc_pos(rows, k, is_help)
            line += f" {v:>9.3f}" if v == v else f" {'n/a':>9}"
        print(line)

    print("\n--- Population (how many traces had this signal extractable) ---")
    print(f"{'cell':<14} " + " ".join(f"{name[:7]:>9}" for name, _ in [(s[1].split()[0], s[0]) for s in SIGNALS]))
    for tag, rows in cells:
        n = len(rows)
        line = f"{tag:<14}"
        for k, _ in SIGNALS:
            pop = sum(1 for r in rows if r.get(k) is not None)
            line += f" {pop:>4}/{n:>3}"
        print(line)


if __name__ == "__main__":
    main()
