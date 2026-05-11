"""ITERATION 1 — Sign-only 5-feature combiner.

Combine the 5 strongest orthogonal signals into a single sign-weighted sum.
No learned weights — each feature contributes ±1 in the direction that gave it
AUROC > 0.5 on calibration. This is the simplest non-trivial combiner; if it
beats best-single by ≥0.04 AUROC, the features are genuinely orthogonal and
the combiner is a real contribution.

Five features picked from try6-10:
  F1: variance_l2_window monotonicity      (shape)
  F2: mean_token_entropy first-step value  (pre-decode)
  F3: u_cons max                           (v2 only)
  F4: dragin_rind_max max                  (v1 sanity)
  F5: tokens_generated                     (length)

Cross-fit:
  - Calibrate sign of each feature on a held-out half (split-half).
  - Compute combiner AUROC on the other half.
  - Repeat with halves swapped; report mean AUROC.
  - Compare to best single feature on the same eval half.
"""
from __future__ import annotations
import random
import sys
import statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load_traces, label, auroc, sd

ARCHS = ["phi", "gemma", "qwen", "phi_hotpot_v2", "phi_2wiki_v2"]


def f(v):
    try: return float(v) if v is not None else None
    except: return None


def step_vals(r, key):
    return [v for v in (f(s.get(key)) for s in r.get("steps") or []) if v is not None]


def feat_monotonicity_var(r):
    vs = step_vals(r, "variance_l2_window")
    if len(vs) < 2: return None
    return sum(1 for i in range(len(vs)-1) if vs[i+1] > vs[i]) / (len(vs)-1)


def feat_first_entropy(r):
    vs = step_vals(r, "mean_token_entropy")
    return vs[0] if vs else None


def feat_ucons_max(r):
    vs = step_vals(r, "u_cons")
    return max(vs) if vs else None


def feat_dragin_max(r):
    vs = step_vals(r, "dragin_rind_max")
    return max(vs) if vs else None


def feat_tokens(r):
    return float(r.get("tokens_generated", 0))


FEATS = [
    ("var_monotonic", feat_monotonicity_var),
    ("ent_first",     feat_first_entropy),
    ("ucons_max",     feat_ucons_max),
    ("dragin_max",    feat_dragin_max),
    ("tokens_gen",    feat_tokens),
]


def extract(rows):
    out = []
    for r in rows:
        feat = {name: fn(r) for name, fn in FEATS}
        out.append((feat, label(r)))
    return out


def single_auroc(records, name):
    pos = [rec[0][name] for rec in records if not rec[1] and rec[0][name] is not None]
    neg = [rec[0][name] for rec in records if rec[1] and rec[0][name] is not None]
    return auroc(pos, neg)


def combiner_score(rec, signs, valid_feats):
    s = 0.0; n = 0
    for name in valid_feats:
        v = rec[0][name]
        if v is None: continue
        s += signs[name] * v
        n += 1
    return s if n > 0 else None


def split_half_evaluate(records, seed=0):
    rng = random.Random(seed)
    idx = list(range(len(records))); rng.shuffle(idx)
    h1 = [records[i] for i in idx[:len(idx)//2]]
    h2 = [records[i] for i in idx[len(idx)//2:]]
    # Pass 1: calibrate on h1, eval on h2
    aurocs = {}
    valid = []
    signs = {}
    for name, _ in FEATS:
        au = single_auroc(h1, name)
        if au != au: continue
        valid.append(name)
        aurocs[name] = au
        signs[name] = 1 if au > 0.5 else -1
    # Single-best on h2 (using calibrated direction)
    best_name = max(valid, key=lambda n: abs(aurocs[n] - 0.5))
    pos = [rec[0][best_name] for rec in h2 if not rec[1] and rec[0][best_name] is not None]
    neg = [rec[0][best_name] for rec in h2 if rec[1] and rec[0][best_name] is not None]
    au_single = auroc(pos, neg)
    if au_single != au_single: au_single = float("nan")
    # Combiner on h2
    # Z-score normalize each feature on h1 first so they're commensurable
    means_stds = {}
    for name in valid:
        vs = [rec[0][name] for rec in h1 if rec[0][name] is not None]
        if not vs: continue
        m = statistics.mean(vs); s = statistics.pstdev(vs) if len(vs)>1 else 1.0
        means_stds[name] = (m, s if s > 0 else 1.0)
    def znorm(rec):
        s = 0; n = 0
        for name in valid:
            v = rec[0][name]
            if v is None or name not in means_stds: continue
            m, sd_ = means_stds[name]
            s += signs[name] * (v - m) / sd_; n += 1
        return s if n > 0 else None
    pos = [znorm(rec) for rec in h2 if not rec[1] and znorm(rec) is not None]
    neg = [znorm(rec) for rec in h2 if rec[1] and znorm(rec) is not None]
    au_combiner = auroc(pos, neg)
    # Drop-one ablation
    drop_one = {}
    for drop in valid:
        keep = [v for v in valid if v != drop]
        def zn(rec):
            s = 0; n = 0
            for name in keep:
                v = rec[0][name]
                if v is None or name not in means_stds: continue
                m, sd_ = means_stds[name]
                s += signs[name] * (v - m) / sd_; n += 1
            return s if n > 0 else None
        pos2 = [zn(rec) for rec in h2 if not rec[1] and zn(rec) is not None]
        neg2 = [zn(rec) for rec in h2 if rec[1] and zn(rec) is not None]
        drop_one[drop] = auroc(pos2, neg2)
    return {"valid": valid, "best_single_name": best_name, "best_single_auroc": au_single,
            "combiner_auroc": au_combiner, "drop_one": drop_one}


print("ITER 1 — 5-feature sign-only combiner (split-half, z-normed)\n")
print(f"{'arch':16}  {'valid feats':>30}  {'best-single':>22}  {'combiner':>10}  {'Δ':>7}")
print("-" * 100)

for arch in ARCHS:
    try: rows = load_traces(arch, min_steps=3)
    except: continue
    if not rows: continue
    recs = extract(rows)
    # 4 random split-half seeds, average
    results = [split_half_evaluate(recs, seed=s) for s in range(4)]
    valid = results[0]["valid"]
    best_singles = [r["best_single_auroc"] for r in results if r["best_single_auroc"] == r["best_single_auroc"]]
    combiners    = [r["combiner_auroc"]    for r in results if r["combiner_auroc"]    == r["combiner_auroc"]]
    bs = statistics.mean(best_singles) if best_singles else float("nan")
    cm = statistics.mean(combiners)    if combiners else float("nan")
    bs_name = max(set(r["best_single_name"] for r in results), key=lambda n: sum(1 for r in results if r["best_single_name"]==n))
    print(f"{arch:16}  {','.join(valid):>30}  {bs_name+f' ({bs:.3f})':>22}  {cm:>10.3f}  {cm-bs:>+7.3f}")

# Per-arch drop-one
print()
print("DROP-ONE ABLATION (avg AUROC over 4 split-halves; gap = combiner − drop-one):")
for arch in ARCHS:
    try: rows = load_traces(arch, min_steps=3)
    except: continue
    if not rows: continue
    recs = extract(rows)
    results = [split_half_evaluate(recs, seed=s) for s in range(4)]
    combiner = statistics.mean([r["combiner_auroc"] for r in results if r["combiner_auroc"]==r["combiner_auroc"]])
    print(f"  {arch:16}  full combiner AUROC = {combiner:.3f}")
    for name, _ in FEATS:
        drops = [r["drop_one"].get(name, float("nan")) for r in results]
        drops = [d for d in drops if d == d]
        if not drops: continue
        dm = statistics.mean(drops)
        print(f"    drop {name:14}  AUROC = {dm:.3f}  (Δ vs full = {dm - combiner:+.3f})")
