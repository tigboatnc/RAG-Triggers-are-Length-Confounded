"""Cross-arch u_cons.max AUROC analyzer for the smoke test.

Reads Gemma and Qwen 50-row smoke decode JSONLs (with u_cons populated by the
v2 capture pipeline), computes:
  - u_cons.max AUROC vs label_incorrect
  - bootstrap 90% AUROC band at n=50
  - paired AUROC vs dragin_rind_max.max (for sanity vs the v1 baseline)
  - go/no-go recommendation per pre-registered thresholds:
      AUROC ≥ 0.55 on at least one arch → green-light 200-row run
      AUROC < 0.50 on both              → paper stays single-arch
"""
from __future__ import annotations
import argparse
import json
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _common import auroc, sd


def f(v):
    try: return float(v) if v is not None else None
    except: return None


def step_vals(r, key):
    return [v for v in (f(s.get(key)) for s in r.get("steps") or []) if v is not None]


def umax(r):
    vs = step_vals(r, "u_cons")
    return max(vs) if vs else None


def dragin_max(r):
    vs = step_vals(r, "dragin_rind_max")
    return max(vs) if vs else None


def label(r):
    return bool(r.get("label_correct_em", r.get("em", False)))


def analyze(arch_name, path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows = [r for r in rows if r.get("n_steps", 0) >= 1]
    triples = []
    for r in rows:
        u = umax(r); d = dragin_max(r)
        if u is None: continue
        triples.append((u, d if d is not None else 0.0, label(r)))
    if not triples:
        return None
    n = len(triples)
    pos_u = [t[0] for t in triples if not t[2]]; neg_u = [t[0] for t in triples if t[2]]
    pos_d = [t[1] for t in triples if not t[2]]; neg_d = [t[1] for t in triples if t[2]]
    au_u = auroc(pos_u, neg_u)
    au_d = auroc(pos_d, neg_d)

    # Bootstrap 90% band
    aus = []; diffs = []
    for sd_ in range(1000):
        rng = random.Random(sd_)
        sample = [triples[rng.randrange(n)] for _ in range(n)]
        pu = [s[0] for s in sample if not s[2]]; nu = [s[0] for s in sample if s[2]]
        pd = [s[1] for s in sample if not s[2]]; nd = [s[1] for s in sample if s[2]]
        a_u = auroc(pu, nu); a_d = auroc(pd, nd)
        if a_u == a_u:
            aus.append(a_u)
        if a_u == a_u and a_d == a_d:
            diffs.append(a_u - a_d)
    aus.sort(); diffs.sort()
    band = (aus[int(0.05*(len(aus)-1))], aus[int(0.95*(len(aus)-1))]) if aus else (float("nan"),)*2
    diff_band = (diffs[int(0.025*(len(diffs)-1))], diffs[int(0.975*(len(diffs)-1))]) if diffs else (float("nan"),)*2

    return {
        "arch": arch_name, "n": n,
        "n_incorrect": sum(1 for t in triples if not t[2]),
        "em": sum(1 for t in triples if t[2]) / n,
        "auroc_ucons": au_u,
        "auroc_dragin": au_d,
        "auroc_band_90": band,
        "paired_diff_mean": statistics.mean(diffs) if diffs else float("nan"),
        "paired_diff_ci95": diff_band,
        "paired_p_pos": sum(1 for d in diffs if d > 0)/len(diffs) if diffs else float("nan"),
    }


def render(r):
    if r is None:
        return "(no usable traces)"
    lines = []
    lines.append(f"\n=== {r['arch']}  n={r['n']}  EM={r['em']*100:.1f}% (P_incorrect={r['n_incorrect']/r['n']:.3f}) ===")
    lines.append(f"  u_cons.max  AUROC = {r['auroc_ucons']:.3f}  (bootstrap 90% band [{r['auroc_band_90'][0]:.3f}, {r['auroc_band_90'][1]:.3f}])")
    lines.append(f"  dragin.max  AUROC = {r['auroc_dragin']:.3f}")
    lines.append(f"  paired u_cons − dragin: mean={r['paired_diff_mean']:+.3f}  95% CI [{r['paired_diff_ci95'][0]:+.3f}, {r['paired_diff_ci95'][1]:+.3f}]  P(u_cons > dragin) = {r['paired_p_pos']:.3f}")
    return "\n".join(lines)


def decision(reports):
    """Pre-registered go/no-go logic."""
    if not reports: return "ERROR: no reports"
    archs_above_055 = [r["arch"] for r in reports if r and r["auroc_ucons"] >= 0.55]
    archs_below_050 = [r["arch"] for r in reports if r and r["auroc_ucons"] < 0.50]
    if archs_above_055:
        return f"GO → run 200-row decode on: {', '.join(archs_above_055)}. Cross-arch story is supported."
    if len(archs_below_050) == len(reports):
        return f"NO-GO → u_cons signal is Phi-specific. Document cross-arch failure in paper, stay single-arch."
    return f"AMBIGUOUS → all archs landed in [0.50, 0.55]. Run a larger smoke (n=100-150) before committing 200-row compute."


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gemma", required=False)
    p.add_argument("--qwen",  required=False)
    p.add_argument("--phi-control", required=False,
                   help="Optional: Phi smoke at the same slice, for sanity replication")
    args = p.parse_args()
    reports = []
    for arch, path in (("gemma", args.gemma), ("qwen", args.qwen), ("phi", args.phi_control)):
        if not path: continue
        if not Path(path).exists():
            print(f"[skip] {arch}: {path} missing"); continue
        rep = analyze(arch, path)
        reports.append(rep)
        print(render(rep))

    print("\n" + "=" * 70)
    print(f"DECISION: {decision(reports)}")


if __name__ == "__main__":
    main()
