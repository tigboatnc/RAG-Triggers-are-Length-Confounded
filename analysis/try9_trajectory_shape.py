"""TRY 9 — Trajectory SHAPE features (Atkinson-aligned).

Atkinson et al. (2026) — hallucination is GEOMETRIC TRAJECTORY DIVERGENCE
in hidden-state space, not a pointwise event. v1's drift_cosine is pointwise.
Test whether the *shape* of the trajectory (slope, range, curvature, monotonicity,
runs of high values) predicts correctness better than max(score).

Shape features for each scalar series:
  - slope (linear fit on step index)
  - range (max - min)
  - n_local_maxima (turns)
  - monotonicity ratio (#increasing-pairs / #pairs)
  - end-to-start ratio (last/first)
  - cumulative AUC normalized by length
  - terminal slope (last-3 minus first-3 mean)
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load_traces, label, auroc

ARCHS = ["phi", "gemma", "qwen", "phi_hotpot_v2", "phi_2wiki_v2"]
SERIES = ["drift_cosine", "variance_l2_window", "hidden_norm",
          "mean_token_entropy", "dragin_rind_max"]


def f(v):
    try: return float(v) if v is not None else None
    except: return None


def step_series(r, key):
    return [f(s.get(key)) for s in (r.get("steps") or [])]


def slope(xs):
    """Linear fit slope on step-index domain."""
    pts = [(i, v) for i, v in enumerate(xs) if v is not None]
    if len(pts) < 2: return None
    n = len(pts)
    sx = sum(p[0] for p in pts); sy = sum(p[1] for p in pts)
    sxx = sum(p[0]**2 for p in pts); sxy = sum(p[0]*p[1] for p in pts)
    denom = n*sxx - sx*sx
    if denom == 0: return None
    return (n*sxy - sx*sy) / denom


def n_local_max(xs):
    vs = [v for v in xs if v is not None]
    if len(vs) < 3: return None
    return sum(1 for i in range(1, len(vs)-1) if vs[i] > vs[i-1] and vs[i] > vs[i+1])


def monotonicity(xs):
    vs = [v for v in xs if v is not None]
    if len(vs) < 2: return None
    inc = sum(1 for i in range(len(vs)-1) if vs[i+1] > vs[i])
    return inc / (len(vs) - 1)


def end_start_ratio(xs):
    vs = [v for v in xs if v is not None]
    if len(vs) < 2 or vs[0] == 0: return None
    return vs[-1] / (vs[0] + 1e-9)


def auc_norm(xs):
    vs = [v for v in xs if v is not None]
    if not vs: return None
    return sum(vs) / len(vs)


def terminal_minus_initial(xs):
    vs = [v for v in xs if v is not None]
    if len(vs) < 4: return None
    n = len(vs)
    return (sum(vs[-3:]) / 3) - (sum(vs[:3]) / 3)


def shape_range(xs):
    vs = [v for v in xs if v is not None]
    if not vs: return None
    return max(vs) - min(vs)


SHAPE_FNS = {
    "slope":           slope,
    "range":           shape_range,
    "n_local_max":     n_local_max,
    "monotonicity":    monotonicity,
    "end_start_ratio": end_start_ratio,
    "auc_norm":        auc_norm,
    "terminal_drift":  terminal_minus_initial,
}

print("TRAJECTORY SHAPE FEATURES — does the shape carry more signal than the max?\n")

all_results = []
for arch in ARCHS:
    try:
        rows = load_traces(arch, min_steps=4)
    except (KeyError, FileNotFoundError):
        continue
    if not rows: continue
    labels = [label(r) for r in rows]
    print(f"=== {arch}  (n={len(rows)}, requiring n_steps≥4 for shape) ===")
    print(f"{'series':22} {'max':>6} | " + " ".join(f"{s:>13}" for s in SHAPE_FNS))
    for series_key in SERIES:
        traces_series = [step_series(r, series_key) for r in rows]
        # baseline: max
        pos_max, neg_max = [], []
        for ts, l in zip(traces_series, labels):
            vs = [v for v in ts if v is not None]
            if not vs: continue
            (neg_max if l else pos_max).append(max(vs))
        au_max = auroc(pos_max, neg_max)
        row = f"{series_key:22} {au_max:>6.3f} | "
        for fn_name, fn in SHAPE_FNS.items():
            pos, neg = [], []
            for ts, l in zip(traces_series, labels):
                v = fn(ts)
                if v is None: continue
                (neg if l else pos).append(v)
            au = auroc(pos, neg)
            mark = " ★" if au == au and abs(au - 0.5) > abs(au_max - 0.5) + 0.02 else "  "
            row += f"{au:>11.3f}{mark} " if au == au else f"{'n/a':>13} "
            if au == au:
                all_results.append((arch, series_key, fn_name, au))
        print(row)
    print()

print("=" * 80)
print("TOP 15 (series, shape-feature, arch) by |AUROC − 0.5|:")
all_results.sort(key=lambda x: -abs(x[3] - 0.5))
for arch, ser, fn_name, au in all_results[:15]:
    print(f"  {arch:16}  {ser:24} → {fn_name:18}  AUROC = {au:.3f}")
