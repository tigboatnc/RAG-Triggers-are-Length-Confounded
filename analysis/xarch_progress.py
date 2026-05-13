"""Cross-arch progress / sanity dashboard.

Run this whenever the compute box pushes new JSONLs. It:

  1. Inventories every expected file across all 3 archs × (smoke, pilot,
     D1, D2, E1, E2) and prints what's missing vs present.
  2. For every present file, reports row count + EM rate + n_steps
     distribution + tokens_generated distribution + capture-field health
     (per-step fields populated? `u_cons` non-null fraction? `u_ent`
     missing as expected?).
  3. For any closed-book file with ≥ 25 rows, reports an early
     AUROC sanity number for `u_cons.max → INCORRECT`. If a paired
     oracle file is also ≥ 25 rows, reports the lift-conditional AUROC
     too. CI widths will be huge at small n — that's fine; the question
     is whether the AUROC point estimate is plausibly in [0.50, 0.70]
     range rather than chance or pathological.
  4. Cross-references against expected health signals (closed-book EM
     band from `_agent_v2.md` Carry-over B), flags surprises.

Run with system python (no GPU); no transformers needed.

Usage:
  python analysis/xarch_progress.py
  python analysis/xarch_progress.py --auroc-min-n 50   # stricter AUROC gate
"""
from __future__ import annotations
import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _common import auroc

ROOT = HERE.parent

ARCH_DIRS = {
    "phi-4-mini-instruct":      ROOT / "runs/v2_exp1_phi-4-mini-instruct",
    "gemma-4-26b-a4b-it":       ROOT / "runs/v2_xarch_gemma-4-26b-a4b-it",
    "qwen3.6-35b-a3b-instruct": ROOT / "runs/v2_xarch_qwen3.6-35b-a3b-instruct",
}

PHASE_FILES = [
    # (label, fname, target_n, has_step_capture)
    ("smoke ",  "B_smoke_hotpot.jsonl",   5,   True),
    ("pilot ",  "C_pilot_hotpot.jsonl",  50,   True),
    ("D1 hot",  "D1_closed_hotpot.jsonl", 200, True),
    ("D2 wiki", "D2_closed_wiki.jsonl",   200, True),
    ("E1 hot",  "E1_oracle_hotpot.jsonl", 200, False),  # oracle = EM-only, no per-step CRV
    ("E2 wiki", "E2_oracle_wiki.jsonl",   200, False),
]
# Phi (legacy v2_exp1_*) was run at N=500, not the xarch N=200. Override per-arch.
TARGET_N_OVERRIDE = {
    "phi-4-mini-instruct": 500,
}

# Expected closed-book EM bands per `_agent_v2.md` Carry-over B (HotpotQA-distractor)
EXPECTED_CB_EM = {
    "phi-4-mini-instruct":      (0.35, 0.55),
    "gemma-4-26b-a4b-it":       (0.45, 0.62),
    "qwen3.6-35b-a3b-instruct": (0.55, 0.85),
}

# Step-record fields we expect populated (u_ent is *expected* missing if rank_bm25 not installed)
EXPECTED_STEP_FIELDS = ("u_cons", "dragin_rind_max", "drift_cosine",
                        "variance_l2_window", "hidden_norm",
                        "mean_token_entropy", "boundary_token_entropy",
                        "hvr", "u_verb")


def f(v):
    try: return float(v) if v is not None else None
    except Exception: return None


def trace_max(r, key):
    vs = [f(s.get(key)) for s in r.get("steps") or []]
    vs = [v for v in vs if v is not None]
    return max(vs) if vs else None


def field_pop_rate(rows, key):
    total = 0; pop = 0
    for r in rows:
        for s in r.get("steps") or []:
            total += 1
            if s.get(key) is not None:
                pop += 1
    return pop / total if total else 0.0


def summarise_file(path: Path) -> dict | None:
    if not path.exists():
        return None
    rows = []
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    if not rows:
        return {"rows": 0, "_empty": True}
    em = [int(bool(r.get("label_correct_em", r.get("em", False)))) for r in rows]
    f1 = [float(r.get("f1") or 0.0) for r in rows]
    nsteps = [int(r.get("n_steps", 0)) for r in rows]
    toks = [int(r.get("tokens_generated", 0)) for r in rows]
    stop = {}
    for r in rows:
        s = r.get("early_stop_reason", "?")
        stop[s] = stop.get(s, 0) + 1
    field_health = {k: field_pop_rate(rows, k) for k in EXPECTED_STEP_FIELDS + ("u_ent",)}
    n_multi = sum(1 for n in nsteps if n >= 2)
    return {
        "rows": len(rows),
        "em_rate": sum(em) / len(em),
        "f1_mean": statistics.mean(f1),
        "n_steps_min": min(nsteps), "n_steps_med": sorted(nsteps)[len(nsteps)//2],
        "n_steps_max": max(nsteps), "n_steps_mean": statistics.mean(nsteps),
        "tok_med": sorted(toks)[len(toks)//2], "tok_min": min(toks), "tok_max": max(toks),
        "n_multi": n_multi, "n_multi_frac": n_multi / len(rows),
        "stop": stop,
        "field_health": field_health,
        "rows_data": rows,
    }


def closed_book_auroc_sanity(rows: list[dict]):
    """AUROC(u_cons.max, dragin.max, mean_token_entropy.max → INCORRECT)."""
    out = {}
    for key, alias in (("u_cons", "u_cons.max"), ("dragin_rind_max", "dragin.max"),
                       ("mean_token_entropy", "ent.max")):
        scores = []; labels = []
        for r in rows:
            v = trace_max(r, key)
            if v is None:
                continue
            scores.append(v)
            labels.append(int(bool(r.get("label_correct_em", r.get("em", False)))))
        pos = [s for s, c in zip(scores, labels) if c == 0]
        neg = [s for s, c in zip(scores, labels) if c == 1]
        if pos and neg:
            out[alias] = (auroc(pos, neg), len(pos), len(neg))
    return out


def lift_conditional_sanity(closed_rows: list[dict], oracle_rows: list[dict]):
    co = {r["id"]: r for r in closed_rows}
    op = {r["id"]: r for r in oracle_rows}
    merged = []
    for qid, c in co.items():
        if qid not in op:
            continue
        o = op[qid]
        c_em = int(bool(c.get("label_correct_em", c.get("em", False))))
        o_em = int(bool(o.get("label_correct_em", o.get("em", False))))
        lift = +1 if o_em > c_em else (-1 if o_em < c_em else 0)
        merged.append({"u": trace_max(c, "u_cons"),
                       "d": trace_max(c, "dragin_rind_max"),
                       "lift": lift, "c_em": c_em})
    merged = [r for r in merged if r["u"] is not None and r["d"] is not None]
    if not merged:
        return None
    out = {"n_paired": len(merged),
           "n_help": sum(1 for r in merged if r["lift"] == +1),
           "n_harm": sum(1 for r in merged if r["lift"] == -1)}
    pos_u = [r["u"] for r in merged if r["lift"] == +1]
    neg_u = [r["u"] for r in merged if r["lift"] != +1]
    pos_d = [r["d"] for r in merged if r["lift"] == +1]
    neg_d = [r["d"] for r in merged if r["lift"] != +1]
    if pos_u and neg_u:
        out["u_cons_AUROC_lift+1"] = auroc(pos_u, neg_u)
    if pos_d and neg_d:
        out["dragin_AUROC_lift+1"] = auroc(pos_d, neg_d)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--auroc-min-n", type=int, default=25)
    args = p.parse_args()

    for arch, arch_dir in ARCH_DIRS.items():
        print(f"\n{'='*88}\n=== {arch}    (dir: {arch_dir.relative_to(ROOT)}) ===")
        if not arch_dir.exists():
            print("  [NOT STARTED] dir missing")
            continue

        target_override = TARGET_N_OVERRIDE.get(arch)
        present = {}
        for label, fname, target_n, has_step_capture in PHASE_FILES:
            if target_override and fname not in ("B_smoke_hotpot.jsonl", "C_pilot_hotpot.jsonl"):
                target_n = target_override
            path = arch_dir / fname
            s = summarise_file(path)
            if s is None:
                print(f"  [missing] {label:8} {fname}")
                continue
            present[fname] = s
            if s.get("_empty"):
                print(f"  [empty  ] {label:8} {fname}")
                continue
            pct = 100 * s["rows"] / target_n
            stop_str = " ".join(f"{k}={v}" for k, v in s["stop"].items())
            print(f"  [ok {pct:5.1f}%] {label:8} n={s['rows']:>4}/{target_n}  "
                  f"EM={s['em_rate']:.3f}  F1={s['f1_mean']:.3f}  "
                  f"n_steps[min/med/mean/max]={s['n_steps_min']}/{s['n_steps_med']}/{s['n_steps_mean']:.1f}/{s['n_steps_max']}  "
                  f"multi={s['n_multi']}({s['n_multi_frac']*100:.0f}%)  "
                  f"tok[min/med/max]={s['tok_min']}/{s['tok_med']}/{s['tok_max']}  "
                  f"stop={{{stop_str}}}")
            if has_step_capture:
                unpop = [k for k, v in s["field_health"].items()
                         if k != "u_ent" and v < 0.95]
                if unpop:
                    # u_verb is naturally partial (only populated when probe succeeded) → quiet warn
                    unpop_critical = [k for k in unpop if k != "u_verb"]
                    if unpop_critical:
                        print(f"     ⚠ partially populated step fields: " +
                              ", ".join(f"{k}={s['field_health'][k]:.2f}" for k in unpop))
            if s["field_health"].get("u_ent", 0) > 0.05:
                print(f"     i u_ent populated at {s['field_health']['u_ent']:.2f} (rank_bm25 available on this box)")
            # EM-band check on closed-book HotpotQA pilot/D1
            if fname in ("C_pilot_hotpot.jsonl", "D1_closed_hotpot.jsonl") and arch in EXPECTED_CB_EM:
                lo, hi = EXPECTED_CB_EM[arch]
                tag = "✓" if lo <= s["em_rate"] <= hi else "⚠ outside expected band"
                if s["rows"] >= 30:  # only flag if we have a reasonable n
                    print(f"     {tag} EM {s['em_rate']:.3f} vs Carry-over-B expected {lo:.2f}-{hi:.2f}")

        # AUROC sanity on whichever closed-book file has the most rows
        cb_best = None
        for fname in ("D1_closed_hotpot.jsonl", "C_pilot_hotpot.jsonl"):
            if fname in present and present[fname].get("rows", 0) >= args.auroc_min_n:
                cb_best = (fname, present[fname]); break
        if cb_best:
            fname, s = cb_best
            au = closed_book_auroc_sanity(s["rows_data"])
            print(f"\n  closed-book AUROC sanity (→INCORRECT, n={s['rows']} from {fname}):")
            for k, (a, np_, nn) in au.items():
                tag = "✓" if 0.50 <= a <= 0.80 else ("⚠ chance" if abs(a-0.5) < 0.03 else "⚠ extreme")
                print(f"    {k:<12} AUROC={a:.3f}   (n_inc={np_} n_cor={nn})   {tag}")

            # 2Wiki sanity if D2 exists
            d2 = present.get("D2_closed_wiki.jsonl")
            if d2 and d2.get("rows", 0) >= args.auroc_min_n:
                au2 = closed_book_auroc_sanity(d2["rows_data"])
                print(f"\n  closed-book AUROC sanity (→INCORRECT, n={d2['rows']} from D2_closed_wiki.jsonl):")
                for k, (a, np_, nn) in au2.items():
                    tag = "✓" if 0.50 <= a <= 0.80 else ("⚠ chance" if abs(a-0.5) < 0.03 else "⚠ extreme")
                    print(f"    {k:<12} AUROC={a:.3f}   (n_inc={np_} n_cor={nn})   {tag}")

        # Lift-conditional sanity (D + E paired)
        for d_name, e_name, label in (
            ("D1_closed_hotpot.jsonl", "E1_oracle_hotpot.jsonl", "HotpotQA"),
            ("D2_closed_wiki.jsonl",   "E2_oracle_wiki.jsonl",   "2Wiki"),
        ):
            if d_name in present and e_name in present:
                d = present[d_name]; e = present[e_name]
                if d["rows"] >= args.auroc_min_n and e["rows"] >= args.auroc_min_n:
                    li = lift_conditional_sanity(d["rows_data"], e["rows_data"])
                    if li is None:
                        continue
                    print(f"\n  lift-conditional sanity ({label}, paired n={li['n_paired']}, lift+1={li['n_help']}, lift-1={li['n_harm']}):")
                    for k in ("u_cons_AUROC_lift+1", "dragin_AUROC_lift+1"):
                        if k in li:
                            print(f"    {k:<22} AUROC={li[k]:.3f}")

    print(f"\n{'='*88}\nDONE. Re-run whenever new JSONLs are pulled.")


if __name__ == "__main__":
    main()
