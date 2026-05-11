"""TRY 6 — Question-side prediction (PROMPT ONLY, zero generation features).

Hypothesis: closed-book correctness is largely determined by question structure,
not by generation dynamics. If true, the entire "when to retrieve during decode"
literature is solving the wrong problem.

Features computed from prompt only (NO step records, NO hidden states):
  - question length (chars and word count)
  - capitalized-word count (proxy for entity density)
  - presence of comparison markers (which / first / earlier / more / between / both / either)
  - presence of temporal markers (year / when / before / after / during)
  - level (easy/med/hard) one-hot
  - type (bridge/comparison) one-hot
  - n_supporting_facts (from gold)
"""
from __future__ import annotations
import re
import sys
import statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load_traces, label, auroc, ALL_TRACES

COMPARISON = re.compile(r"\b(which|first|earlier|more|between|both|either|or|fewer|less|greater|smaller|older|newer|after|before|same|different)\b", re.I)
TEMPORAL   = re.compile(r"\b(year|when|date|after|before|during|century|decade|month)\b", re.I)
WH         = re.compile(r"\b(who|what|where|when|why|how)\b", re.I)


def qfeatures(r: dict) -> dict:
    q = r.get("question", "") or ""
    feat = {
        "q_chars": len(q),
        "q_words": len(q.split()),
        "n_capitalized": len(re.findall(r"\b[A-Z][a-z]+", q)),
        "n_digits": len(re.findall(r"\d+", q)),
        "n_quoted": q.count('"') + q.count("'"),
        "has_comparison": int(bool(COMPARISON.search(q))),
        "has_temporal":   int(bool(TEMPORAL.search(q))),
        "n_wh":           len(WH.findall(q)),
        "is_easy":   int(r.get("level") == "easy"),
        "is_medium": int(r.get("level") == "medium"),
        "is_hard":   int(r.get("level") == "hard"),
        "is_bridge":     int(r.get("type") == "bridge"),
        "is_comparison": int(r.get("type") == "comparison"),
        "prompt_tokens": int(r.get("prompt_tokens", 0)),
    }
    return feat


ARCHS = ["phi", "gemma", "qwen", "phi_hotpot_v2", "phi_2wiki_v2"]
FEAT_KEYS = ["q_chars", "q_words", "n_capitalized", "n_digits", "n_quoted",
             "has_comparison", "has_temporal", "n_wh",
             "is_easy", "is_medium", "is_hard", "is_bridge", "is_comparison",
             "prompt_tokens"]

print("QUESTION-SIDE PREDICTION OF CLOSED-BOOK INCORRECTNESS\n")
print(f"AUROC for each PROMPT-ONLY feature → P(incorrect)")
print(f"{'feature':16} | " + "  ".join(f"{a:>14}" for a in ARCHS))
print("-" * (16 + 3 + 16 * len(ARCHS)))

best = {}
for arch in ARCHS:
    try:
        rows = load_traces(arch, min_steps=0)
    except (KeyError, FileNotFoundError):
        rows = []
    if not rows: continue
    labels = [label(r) for r in rows]
    feats = [qfeatures(r) for r in rows]
    best[arch] = ("", 0.5)
    for key in FEAT_KEYS:
        pos = [f[key] for f, l in zip(feats, labels) if not l]
        neg = [f[key] for f, l in zip(feats, labels) if l]
        au = auroc(pos, neg)
        if au == au and abs(au - 0.5) > abs(best[arch][1] - 0.5):
            best[arch] = (key, au)

# Print full table
for key in FEAT_KEYS:
    row = f"{key:16} | "
    for arch in ARCHS:
        try:
            rows = load_traces(arch, min_steps=0)
        except (KeyError, FileNotFoundError):
            row += f"{'—':>14}  "; continue
        if not rows:
            row += f"{'—':>14}  "; continue
        labels = [label(r) for r in rows]
        feats = [qfeatures(r) for r in rows]
        pos = [f[key] for f, l in zip(feats, labels) if not l]
        neg = [f[key] for f, l in zip(feats, labels) if l]
        au = auroc(pos, neg)
        row += f"{au:>14.3f}  " if au == au else f"{'n/a':>14}  "
    print(row)

print()
print("BEST PROMPT-ONLY FEATURE per arch (|AUROC − 0.5| maximised):")
for arch, (k, au) in best.items():
    print(f"  {arch:16}: {k:18} AUROC = {au:.3f}")

# Per-arch best vs the v1 CRV gates from try4 trace-level AUROC:
v1_drift = {"phi": 0.543, "gemma": 0.508, "qwen": 0.525,
            "phi_hotpot_v2": 0.549, "phi_2wiki_v2": 0.533}
v1_dragin = {"phi": 0.594, "gemma": 0.683, "qwen": 0.574,
             "phi_hotpot_v2": 0.572, "phi_2wiki_v2": 0.575}
print(f"\nCOMPARISON: prompt-best vs drift_cosine_trace_max vs dragin_trace_max")
print(f"  {'arch':16}  {'prompt-best':>15}  {'drift_trace':>12}  {'dragin_trace':>13}")
for arch in ARCHS:
    if arch not in best: continue
    k, au = best[arch]
    print(f"  {arch:16}  {f'{au:.3f} ({k})':>15}  {v1_drift.get(arch,0):>12.3f}  {v1_dragin.get(arch,0):>13.3f}")

# Simple sum-of-features greedy AUROC (toy logistic-style additive)
print("\nADDITIVE COMBO (sign-weighted greedy sum of top 5 prompt features per arch):")
for arch in ARCHS:
    try:
        rows = load_traces(arch, min_steps=0)
    except (KeyError, FileNotFoundError):
        rows = []
    if not rows: continue
    labels = [label(r) for r in rows]
    feats = [qfeatures(r) for r in rows]
    # rank features by |AUROC - 0.5|
    feature_scores = []
    for key in FEAT_KEYS:
        pos = [f[key] for f, l in zip(feats, labels) if not l]
        neg = [f[key] for f, l in zip(feats, labels) if l]
        au = auroc(pos, neg)
        if au == au:
            feature_scores.append((key, au, abs(au - 0.5)))
    feature_scores.sort(key=lambda x: -x[2])
    top5 = feature_scores[:5]
    # composite: weight = sign(au-0.5), score = sum
    composite_pos, composite_neg = [], []
    for f, l in zip(feats, labels):
        s = sum((1 if au > 0.5 else -1) * f[key] for key, au, _ in top5)
        (composite_pos if not l else composite_neg).append(s)
    au_comp = auroc(composite_pos, composite_neg)
    print(f"  {arch:16}  features={[k for k,_,_ in top5]}  composite AUROC = {au_comp:.3f}")
