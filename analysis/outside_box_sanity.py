"""Outside-the-box sanity sweep — five mini-analyses, one summary at end.

Skipped (require re-decode, not pure post-hoc on existing logs):
  - Idea 1: Activation steering (needs inference-time hooks)
  - Idea 3: Adaptive compute allocation (needs adaptive truncation at decode)

Run from existing on-disk data:

  IDEA 2 (approx). K=3 backtrack signal — does u_cons disagreement at
    step k correlate with subsequent trace failure? If yes, backtracking
    at high u_cons step could prevent failure. Sanity: per-step u_cons
    at trace's last step vs trace's first step, conditional on EM.

  IDEA 4. Failure-mode taxonomy — partition traces into 5 buckets
    (early-commit-wrong, long-rambling-wrong, short-correct, long-
    correct, ambiguous) and report mean signal values per bucket.
    Without sklearn we use hand-defined buckets, not k-means.

  IDEA 5. Architecture comparison — using V1 traces (Phi/Gemma/Qwen
    HotpotQA-200) which have drift/variance/dragin/ent/hidden_norm
    (but no u_cons). Compare mean signal × correct/incorrect per arch.

  IDEA 6. Multi-axis uncertainty decomposition — examine pairwise
    correlations of u_cons, drift, ent, u_verb on the v2 Phi traces.
    If correlations are < 0.3, the signals are quasi-orthogonal and
    decomposition is empirically supported.

  IDEA 7. Wu inverted-U empirically — EM rate as function of n_steps
    tertile, per arch and dataset. Show whether long traces are
    less accurate than short traces (the inverted-U claim).

Each section is independent and produces a small numerical summary.
The final ===SUMMARY=== block synthesises all five.
"""
from __future__ import annotations
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Iterable

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _common import auroc

ROOT = HERE.parent
V1_TRACES = {
    "phi":   ROOT / "runs/phase4_qa_phi-4-mini-instruct_mid_stack_dragin.jsonl",
    "gemma": ROOT / "runs/phase4_qa_gemma-4-26b-a4b-it_mid_stack_dragin.jsonl",
    "qwen":  ROOT / "runs/phase4_qa_qwen3.6-35b-a3b-instruct_mid_stack_dragin.jsonl",
}
V2_TRACES = {
    "phi_hot":  ROOT / "runs/v2_exp1_phi-4-mini-instruct/D1_closed_hotpot.jsonl",
    "phi_wiki": ROOT / "runs/v2_exp1_phi-4-mini-instruct/D2_closed_wiki.jsonl",
}


def f(v):
    try: return float(v) if v is not None else None
    except Exception: return None


def step_vals(r, key):
    return [f(s.get(key)) for s in r.get("steps") or []]


def load_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    for line in path.open():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def em_of(r):
    return int(bool(r.get("label_correct_em", r.get("em", False))))


def banner(s, ch="="):
    print(f"\n{ch*92}\n{s}\n{ch*92}")


# ----------------------------------------------------------------------- #
# IDEA 2 — K=3 backtrack signal: does u_cons at step k predict trace failure?
# Sanity: for each multi-step trace, find the step with max u_cons. Is that
# step earlier in correct traces and later in incorrect traces? If late→fail,
# that's the "trajectory has been locked too long" backtrack signal.
# ----------------------------------------------------------------------- #
def idea_2_backtrack(rows):
    rows = [r for r in rows if r.get("n_steps", 0) >= 3]
    if not rows:
        return {"n": 0, "msg": "no multistep traces"}
    correct = []; wrong = []
    for r in rows:
        us = [v for v in step_vals(r, "u_cons") if v is not None]
        if len(us) < 3: continue
        amax = us.index(max(us))
        norm_pos = amax / (len(us) - 1)
        (correct if em_of(r) == 1 else wrong).append(norm_pos)
    if not correct or not wrong:
        return {"n": len(rows), "msg": "no class diversity"}
    return {
        "n_correct": len(correct), "n_wrong": len(wrong),
        "mean_pos_correct": statistics.mean(correct),
        "mean_pos_wrong":   statistics.mean(wrong),
        "median_pos_correct": statistics.median(correct),
        "median_pos_wrong":   statistics.median(wrong),
        "auroc_pos_pred_wrong": auroc(wrong, correct),  # high pos→wrong predicted
    }


# ----------------------------------------------------------------------- #
# IDEA 4 — Failure-mode taxonomy via hand-defined buckets.
# Buckets:
#   early_commit_wrong:  n_steps == 1 AND em==0
#   short_correct:       n_steps <= 2 AND em==1
#   long_rambling_wrong: n_steps >= 5 AND em==0
#   long_correct:        n_steps >= 5 AND em==1
#   ambiguous:           the middle (3-4 steps)
# Report n + mean of {tokens, mean_token_entropy.max, u_cons.max, dragin.max,
# drift.max, hedge_total} per bucket.
# ----------------------------------------------------------------------- #
def idea_4_taxonomy(rows):
    def bucket(r):
        n = r.get("n_steps", 0); em = em_of(r)
        if n == 1 and em == 0: return "early_commit_wrong"
        if n <= 2 and em == 1: return "short_correct"
        if n >= 5 and em == 0: return "long_rambling_wrong"
        if n >= 5 and em == 1: return "long_correct"
        return "ambiguous"
    def feats(r):
        ent = [v for v in step_vals(r, "mean_token_entropy") if v is not None]
        uc  = [v for v in step_vals(r, "u_cons") if v is not None]
        dr  = [v for v in step_vals(r, "dragin_rind_max") if v is not None]
        dc  = [v for v in step_vals(r, "drift_cosine") if v is not None]
        hg  = [v for v in step_vals(r, "hedge_count") if v is not None]
        return {
            "tokens": float(r.get("tokens_generated", 0)),
            "ent_max": max(ent) if ent else 0.0,
            "u_cons_max": max(uc) if uc else 0.0,
            "dragin_max": max(dr) if dr else 0.0,
            "drift_max":  max(dc) if dc else 0.0,
            "hedge_total": sum(hg) if hg else 0.0,
        }
    by_b = {}
    for r in rows:
        b = bucket(r); by_b.setdefault(b, []).append(feats(r))
    out = {}
    for b, fl in by_b.items():
        if not fl: continue
        means = {k: statistics.mean(x[k] for x in fl) for k in fl[0].keys()}
        out[b] = {"n": len(fl), **means}
    return out


# ----------------------------------------------------------------------- #
# IDEA 5 — Architecture comparison from V1 corpus.
# Compute mean of each signal × correct/incorrect, per arch, on V1.
# Cross-arch: do they have different failure signatures?
# ----------------------------------------------------------------------- #
def idea_5_xarch_v1():
    results = {}
    for arch, path in V1_TRACES.items():
        rows = load_jsonl(path)
        if not rows: continue
        correct = [r for r in rows if em_of(r) == 1]
        wrong = [r for r in rows if em_of(r) == 0]
        def aggr(group):
            if not group: return None
            ents = []; drs = []; dcs = []; vrs = []; hns = []; etcs = []
            for r in group:
                e = [v for v in step_vals(r, "mean_token_entropy") if v is not None]
                d = [v for v in step_vals(r, "dragin_rind_max") if v is not None]
                c = [v for v in step_vals(r, "drift_cosine") if v is not None]
                w = [v for v in step_vals(r, "variance_l2_window") if v is not None]
                h = [v for v in step_vals(r, "hidden_norm") if v is not None]
                if e: ents.append(max(e))
                if d: drs.append(max(d))
                if c: dcs.append(max(c))
                if w: vrs.append(max(w))
                if h: hns.append(max(h))
            return {
                "n": len(group),
                "ent_max":   statistics.mean(ents) if ents else float("nan"),
                "dragin":    statistics.mean(drs) if drs else float("nan"),
                "drift":     statistics.mean(dcs) if dcs else float("nan"),
                "variance":  statistics.mean(vrs) if vrs else float("nan"),
                "hidden_n":  statistics.mean(hns) if hns else float("nan"),
                "em_rate":   sum(em_of(r) for r in group)/len(group),
                "tok_mean":  statistics.mean(int(r.get("tokens_generated", 0)) for r in group),
            }
        results[arch] = {
            "all": aggr(rows),
            "correct": aggr(correct),
            "wrong": aggr(wrong),
        }
    return results


# ----------------------------------------------------------------------- #
# IDEA 6 — Pairwise correlations of multi-signal stack.
# If u_cons, drift, ent, u_verb are pairwise correlated < 0.3, they
# capture different "uncertainty types". If > 0.7, they collapse to one axis.
# ----------------------------------------------------------------------- #
def pearson_corr(xs, ys):
    n = len(xs)
    if n < 2: return float("nan")
    mx = sum(xs)/n; my = sum(ys)/n
    num = sum((xs[i]-mx)*(ys[i]-my) for i in range(n))
    dx = math.sqrt(sum((xs[i]-mx)**2 for i in range(n)))
    dy = math.sqrt(sum((ys[i]-my)**2 for i in range(n)))
    return num / (dx*dy) if dx > 0 and dy > 0 else float("nan")


def idea_6_correlations(rows):
    SIGNALS = ("u_cons", "drift_cosine", "mean_token_entropy", "u_verb",
               "variance_l2_window", "boundary_token_entropy", "dragin_rind_max", "hvr")
    per_trace = {}
    for k in SIGNALS:
        per_trace[k] = []
    for r in rows:
        if r.get("n_steps", 0) < 2: continue
        valid = True
        per_trace_local = {}
        for k in SIGNALS:
            vs = [v for v in step_vals(r, k) if v is not None]
            if not vs:
                valid = False; break
            per_trace_local[k] = max(vs)
        if valid:
            for k in SIGNALS:
                per_trace[k].append(per_trace_local[k])
    n = len(per_trace["u_cons"])
    corrs = {}
    for i, ka in enumerate(SIGNALS):
        for kb in SIGNALS[i+1:]:
            corrs[f"{ka} ↔ {kb}"] = pearson_corr(per_trace[ka], per_trace[kb])
    return n, corrs


# ----------------------------------------------------------------------- #
# IDEA 7 — Wu inverted-U: EM rate by n_steps tertile, per arch + dataset.
# ----------------------------------------------------------------------- #
def idea_7_inverted_u(rows, name):
    if not rows: return {"name": name, "msg": "no rows"}
    nsteps = [r.get("n_steps", 0) for r in rows]
    if not nsteps: return {"name": name, "msg": "no n_steps"}
    s = sorted(nsteps)
    q33 = s[len(s)//3]; q66 = s[2*len(s)//3]
    buckets = {"short": [], "mid": [], "long": []}
    for r in rows:
        n = r.get("n_steps", 0)
        if n <= q33: buckets["short"].append(em_of(r))
        elif n >= q66: buckets["long"].append(em_of(r))
        else: buckets["mid"].append(em_of(r))
    out = {"name": name, "q33": q33, "q66": q66}
    for b, ems in buckets.items():
        out[f"{b}_n"] = len(ems)
        out[f"{b}_em"] = sum(ems)/len(ems) if ems else float("nan")
    return out


# ----------------------------------------------------------------------- #
# Run + print
# ----------------------------------------------------------------------- #
def main():
    phi_hot  = load_jsonl(V2_TRACES["phi_hot"])
    phi_wiki = load_jsonl(V2_TRACES["phi_wiki"])

    banner("IDEA 2 — K=3 backtrack signal (normalised position of u_cons.argmax | EM)")
    for name, rows in (("Phi v2 Hot", phi_hot), ("Phi v2 2Wiki", phi_wiki)):
        out = idea_2_backtrack(rows)
        print(f"  {name}: {out}")

    banner("IDEA 4 — Failure-mode taxonomy buckets (Phi v2 Hot + 2Wiki combined)")
    combined = phi_hot + phi_wiki
    tax = idea_4_taxonomy(combined)
    for b in ("early_commit_wrong", "short_correct", "long_rambling_wrong", "long_correct", "ambiguous"):
        if b in tax:
            d = tax[b]
            print(f"  {b:<22} n={d['n']:>4}  toks={d['tokens']:>6.0f}  ent={d['ent_max']:.2f}  "
                  f"u_cons={d['u_cons_max']:.3f}  dragin={d['dragin_max']:.2f}  drift={d['drift_max']:.3f}  hedge={d['hedge_total']:.1f}")

    banner("IDEA 5 — V1 cross-architecture failure signatures (HotpotQA-200)")
    xa = idea_5_xarch_v1()
    for arch, d in xa.items():
        if d["all"] is None:
            print(f"  {arch}: no data"); continue
        print(f"  {arch}  (n={d['all']['n']}, em={d['all']['em_rate']:.3f}, tok_mean={d['all']['tok_mean']:.0f})")
        for split in ("correct", "wrong"):
            s = d[split]
            if s is None: continue
            print(f"    {split:<8} n={s['n']:>4}  ent={s['ent_max']:.2f}  dragin={s['dragin']:.2f}  "
                  f"drift={s['drift']:.3f}  var={s['variance']:.3f}  hidden={s['hidden_n']:.2f}  toks={s['tok_mean']:.0f}")

    banner("IDEA 6 — Pairwise correlations of multi-signal uncertainty stack (Phi v2 Hot)")
    n6, corrs = idea_6_correlations(phi_hot)
    print(f"  n_traces with all 8 signals populated: {n6}")
    sorted_corrs = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)
    for pair, c in sorted_corrs:
        tag = "ORTHOG" if abs(c) < 0.30 else ("CORR" if abs(c) < 0.70 else "COLLAPSE")
        print(f"  {pair:<55} r={c:+.3f}  {tag}")

    banner("IDEA 7 — Wu inverted-U: EM rate by n_steps tertile")
    for name, rows in (("Phi v2 Hot", phi_hot), ("Phi v2 2Wiki", phi_wiki)):
        out = idea_7_inverted_u(rows, name)
        if "short_em" in out:
            print(f"  {name}  short≤{out['q33']} ({out['short_n']}) EM={out['short_em']:.3f}  "
                  f"|  mid ({out['mid_n']}) EM={out['mid_em']:.3f}  "
                  f"|  long≥{out['q66']} ({out['long_n']}) EM={out['long_em']:.3f}")
    for arch, path in V1_TRACES.items():
        v1 = load_jsonl(path)
        out = idea_7_inverted_u(v1, f"V1 {arch}")
        if "short_em" in out:
            print(f"  V1 {arch:<5}  short≤{out['q33']} ({out['short_n']}) EM={out['short_em']:.3f}  "
                  f"|  mid ({out['mid_n']}) EM={out['mid_em']:.3f}  "
                  f"|  long≥{out['q66']} ({out['long_n']}) EM={out['long_em']:.3f}")

    # ----- SUMMARY -----
    banner("===SUMMARY===", "#")
    # Idea 2 sanity
    b2_hot = idea_2_backtrack(phi_hot); b2_wiki = idea_2_backtrack(phi_wiki)
    print("[IDEA 2] u_cons.argmax_pos | correct vs wrong (Phi v2):")
    for name, b in (("Hot", b2_hot), ("Wiki", b2_wiki)):
        if "mean_pos_correct" in b:
            print(f"  {name}: correct={b['mean_pos_correct']:.2f}  wrong={b['mean_pos_wrong']:.2f}  Δ={b['mean_pos_wrong']-b['mean_pos_correct']:+.2f}  AUROC={b['auroc_pos_pred_wrong']:.3f}")
    print("[IDEA 4] Cluster signatures in `IDEA 4` table above — see if early_commit_wrong has different mean signals than long_rambling_wrong.")
    print("[IDEA 5] Cross-arch raw signals are NOT directly comparable (different scales) — but within-arch correct-vs-wrong differences ARE comparable. Look at the (wrong - correct) signal deltas in IDEA 5.")
    sorted_c = sorted(corrs.items(), key=lambda x: abs(x[1]))
    if sorted_c:
        most_orth_pair = sorted_c[0][0]; most_orth_r = sorted_c[0][1]
        most_corr_pair = sorted_c[-1][0]; most_corr_r = sorted_c[-1][1]
        print(f"[IDEA 6] Most-orthogonal signal pair: {most_orth_pair} (r={most_orth_r:+.3f}); most-collapsed pair: {most_corr_pair} (r={most_corr_r:+.3f}).")
    print("[IDEA 7] If long-tertile EM is meaningfully lower than short-tertile EM on multiple cells, Wu inverted-U holds in our data.")


if __name__ == "__main__":
    main()
