"""Length-confound audit — does length alone explain DRAGIN / u_cons / entropy
performance on multi-hop distractor QA?

Two analyses on Phi v2 paired data (Hot n=393, Wiki n=452):

  (A) LENGTH-CONDITIONAL AUROC. Stratify rows by n_steps tertile.
      Within each tertile (so length is approximately controlled),
      recompute AUROC of u_cons.max, dragin.max, ent.max, and the
      external `tokens_generated` baseline. If internal signals lose
      to within-tertile length, the field has been measuring length.

  (B) PER-QUESTION-TYPE AUROC. HotpotQA records carry `type` ∈
      {bridge, comparison}; 2Wiki uses {compositional, inference,
      bridge_comparison, comparison}. Compute AUROC of the same
      four signals per type. If length effect concentrates in one
      type (e.g. bridge), the dataset-feature is localised.

  (C) RESIDUAL TEST. Partial AUROC: rank u_cons / DRAGIN / ent
      WITHIN equal-length bins, weighted by bin frequency. This is
      the "after regressing length out, what's left" measurement.

Outputs are descriptive — no bootstrap CIs (too small per-cell at
n=120-180 per tertile).
"""
from __future__ import annotations
import json
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


def f(v):
    try: return float(v) if v is not None else None
    except Exception: return None


def step_vals(r, key):
    return [f(s.get(key)) for s in r.get("steps") or []]


def trace_max(r, key):
    vs = [v for v in step_vals(r, key) if v is not None]
    return max(vs) if vs else None


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
        u = trace_max(c, "u_cons"); d = trace_max(c, "dragin_rind_max"); e = trace_max(c, "mean_token_entropy")
        if u is None or d is None or e is None:
            continue
        rows.append({
            "id": qid, "type": c.get("type"), "level": c.get("level"),
            "closed_em": c_em, "lift_em": lift,
            "n_steps": int(c.get("n_steps", 0)),
            "tokens": int(c.get("tokens_generated", 0)),
            "wall_s": float(c.get("wall_clock_s") or 0),
            "u_cons_max": u, "dragin_max": d, "ent_max": e,
        })
    return rows


def auroc_pos(rows, key, is_pos):
    p = [r[key] for r in rows if is_pos(r)]
    n = [r[key] for r in rows if not is_pos(r)]
    if not p or not n:
        return float("nan")
    return auroc(p, n)


def tertile_bounds(rows, key):
    vals = sorted(r[key] for r in rows)
    if not vals:
        return 0, 0
    return vals[len(vals)//3], vals[2*len(vals)//3]


def in_tertile(r, key, lo, hi, which):
    v = r[key]
    if which == "short": return v <= lo
    if which == "long":  return v >= hi
    return lo < v < hi


def main():
    for label, cp, op in DATASETS:
        rows = load_paired(cp, op)
        n = len(rows); n_inc = sum(1 for r in rows if r["closed_em"] == 0); n_help = sum(1 for r in rows if r["lift_em"] == +1)
        print(f"\n{'='*92}\n=== {label}   n={n}  incorrect={n_inc}  lift+1={n_help} (base={n_help/n:.3f}) ===")

        is_help = lambda r: r["lift_em"] == +1
        is_inc = lambda r: r["closed_em"] == 0
        SIGNALS = (("tokens", "tokens (external)"),
                   ("n_steps", "n_steps (external)"),
                   ("wall_s", "wall_s (external)"),
                   ("u_cons_max", "u_cons (internal)"),
                   ("dragin_max", "dragin (internal)"),
                   ("ent_max", "ent (internal)"))

        # ----- (A) LENGTH-CONDITIONAL by n_steps tertile -----
        print("\n--- (A) AUROC WITHIN n_steps TERTILE  →lift+1 ---")
        lo, hi = tertile_bounds(rows, "n_steps")
        print(f"  tertile bounds: short ≤ {lo} steps  /  long ≥ {hi} steps  /  mid between")
        print(f"  {'signal':<22}  {'overall':>8}  {'short':>8}  {'mid':>8}  {'long':>8}  {'within-tertile':>14}")
        print(f"  {'-'*22}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*14}")
        # within-tertile weighted average
        for key, name in SIGNALS:
            overall = auroc_pos(rows, key, is_help)
            cells = {}
            for ter in ("short", "mid", "long"):
                sub = [r for r in rows if in_tertile(r, "n_steps", lo, hi, ter)]
                cells[ter] = (auroc_pos(sub, key, is_help), len(sub),
                              sum(1 for r in sub if is_help(r)))
            # weighted within-tertile AUROC (skip cells where helpful count == 0)
            num = 0.0; den = 0.0
            for ter in ("short", "mid", "long"):
                au, n_t, n_h = cells[ter]
                if au == au and n_h > 0:
                    num += au * n_t; den += n_t
            within = num / den if den > 0 else float("nan")
            cells_str = "  ".join(
                f"{cells[ter][0]:>8.3f}" if cells[ter][0] == cells[ter][0] else f"{'n/a':>8}"
                for ter in ("short", "mid", "long"))
            print(f"  {name:<22}  {overall:>8.3f}  {cells_str}  {within:>14.3f}")
            # also report counts
            cnt_str = "  ".join(f"({cells[ter][2]}/{cells[ter][1]})" for ter in ("short","mid","long"))
            print(f"  {'(n_help/n)':<22}  {'':<8}  {cnt_str}")

        # ----- (A2) LENGTH-CONDITIONAL by `tokens` tertile (continuous) -----
        print("\n--- (A2) AUROC WITHIN tokens_generated TERTILE  →lift+1 ---")
        lo, hi = tertile_bounds(rows, "tokens")
        print(f"  tertile bounds: short ≤ {lo} tok  /  long ≥ {hi} tok  /  mid between")
        print(f"  {'signal':<22}  {'overall':>8}  {'short':>8}  {'mid':>8}  {'long':>8}  {'within-tertile':>14}")
        print(f"  {'-'*22}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*14}")
        for key, name in SIGNALS:
            overall = auroc_pos(rows, key, is_help)
            cells = {}
            for ter in ("short", "mid", "long"):
                sub = [r for r in rows if in_tertile(r, "tokens", lo, hi, ter)]
                cells[ter] = (auroc_pos(sub, key, is_help), len(sub),
                              sum(1 for r in sub if is_help(r)))
            num = 0.0; den = 0.0
            for ter in ("short", "mid", "long"):
                au, n_t, n_h = cells[ter]
                if au == au and n_h > 0:
                    num += au * n_t; den += n_t
            within = num / den if den > 0 else float("nan")
            cells_str = "  ".join(
                f"{cells[ter][0]:>8.3f}" if cells[ter][0] == cells[ter][0] else f"{'n/a':>8}"
                for ter in ("short", "mid", "long"))
            print(f"  {name:<22}  {overall:>8.3f}  {cells_str}  {within:>14.3f}")
            cnt_str = "  ".join(f"({cells[ter][2]}/{cells[ter][1]})" for ter in ("short","mid","long"))
            print(f"  {'(n_help/n)':<22}  {'':<8}  {cnt_str}")

        # ----- (B) PER-QUESTION-TYPE AUROC -----
        print("\n--- (B) AUROC PER QUESTION TYPE  →lift+1 ---")
        types = sorted(set(r["type"] for r in rows if r["type"]))
        print(f"  types present: {types}")
        for t in types:
            sub = [r for r in rows if r["type"] == t]
            n_t = len(sub); n_h = sum(1 for r in sub if r["lift_em"] == +1)
            n_inc_t = sum(1 for r in sub if r["closed_em"] == 0)
            print(f"\n  type={t}  n={n_t}  incorrect={n_inc_t}  lift+1={n_h} (base={n_h/n_t:.3f})")
            print(f"    {'signal':<22}  {'AUROC →lift+1':>14}  {'AUROC →INCORRECT':>17}")
            for key, name in SIGNALS:
                au_h = auroc_pos(sub, key, is_help)
                au_i = auroc_pos(sub, key, is_inc)
                print(f"    {name:<22}  {au_h:>14.3f}  {au_i:>17.3f}")

        # ----- (C) DELTA AUROC: signal vs tokens within tertile -----
        print("\n--- (C) DOES INTERNAL SIGNAL BEAT LENGTH WITHIN-TERTILE? (Δ AUROC vs `tokens`) ---")
        lo, hi = tertile_bounds(rows, "tokens")
        for key, name in SIGNALS:
            if key == "tokens": continue
            print(f"  {name:<22}", end="")
            for ter in ("short", "mid", "long"):
                sub = [r for r in rows if in_tertile(r, "tokens", lo, hi, ter)]
                au_sig = auroc_pos(sub, key, is_help)
                au_tok = auroc_pos(sub, "tokens", is_help)
                delta = (au_sig - au_tok) if (au_sig == au_sig and au_tok == au_tok) else float("nan")
                tag = "+" if delta > 0.02 else ("-" if delta < -0.02 else " ")
                print(f"  {ter}=Δ{delta:+.3f}{tag}", end="")
            print()


if __name__ == "__main__":
    main()
