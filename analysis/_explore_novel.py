"""Deep exploratory analysis on P0 pilot data — 10 novel angles.
Seeks paper-worthy innovation directions from existing traces.
"""
import json
import math
import re
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parent.parent

def auroc(pos, neg):
    if not pos or not neg: return float('nan')
    n_pos, n_neg = len(pos), len(neg)
    all_items = [(v, 1) for v in pos] + [(v, 0) for v in neg]
    all_items.sort(key=lambda x: -x[0])
    tp = 0; auc = 0.0
    for _, label in all_items:
        if label == 1: tp += 1
        else: auc += tp
    return auc / (n_pos * n_neg) if (n_pos * n_neg) > 0 else 0.5

def step_vals(r, key):
    return [s.get(key) for s in r.get('steps', []) if s.get(key) is not None]

# Load ALL paired traces
all_traces = []
for arch in ['phi-4-mini-instruct', 'gemma-4-26b-a4b-it', 'qwen2.5-7b-instruct']:
    for ds, cf, of in [('hot', 'D1_closed_hotpot.jsonl', 'E1_oracle_hotpot.jsonl'),
                       ('wiki', 'D2_closed_wiki.jsonl', 'E2_oracle_wiki.jsonl')]:
        cp = ROOT / f'runs/v2_pilot_{arch}/{cf}'
        op = ROOT / f'runs/v2_pilot_{arch}/{of}'
        if not cp.exists() or not op.exists(): continue
        closed = {r['id']: r for r in (json.loads(l) for l in open(cp))}
        oracle = {r['id']: r for r in (json.loads(l) for l in open(op))}
        for qid, c in closed.items():
            if qid not in oracle: continue
            o = oracle[qid]
            c_em = int(bool(c.get('label_correct_em', c.get('em', False))))
            o_em = int(bool(o.get('label_correct_em', o.get('em', False))))
            lift = +1 if o_em > c_em else (-1 if o_em < c_em else 0)
            all_traces.append({
                'arch': arch.split('-')[0] if 'qwen' not in arch else 'qwen',
                'ds': ds, 'c': c, 'o': o, 'lift': lift, 'c_em': c_em,
                'cell': f"{arch.split('-')[0] if 'qwen' not in arch else 'qwen'}_{ds}"
            })

print(f"Loaded {len(all_traces)} paired traces across 6 cells\n")
is_inc = lambda t: t['c_em'] == 0
is_help = lambda t: t['lift'] == +1

# ===================================================================
# ANGLE 1: ENTROPY TRAJECTORY SHAPE FEATURES
# ===================================================================
print("=" * 80)
print("ANGLE 1: Entropy trajectory shape (monotone, spike, decline patterns)")
print("=" * 80)

def trajectory_features(vals):
    if not vals or len(vals) < 2:
        return {}
    vals = [float(v) for v in vals if v is not None]
    if len(vals) < 2:
        return {}
    n = len(vals)
    x = np.arange(n)
    slope = np.polyfit(x, vals, 1)[0] if n >= 2 else 0
    std = np.std(vals)
    spike = (max(vals) - np.mean(vals)) / std if std > 0 else 0
    argmax_frac = np.argmax(vals) / max(1, n-1)
    mid = n // 2
    if mid > 0 and (n - mid) > 0:
        half_ratio = np.mean(vals[mid:]) / max(1e-10, np.mean(vals[:mid]))
    else:
        half_ratio = 1.0
    mono_up = sum(1 for i in range(n-1) if vals[i+1] > vals[i]) / max(1, n-1)
    return {
        'slope': float(slope),
        'spike': float(spike),
        'argmax_frac': float(argmax_frac),
        'half_ratio': float(half_ratio),
        'mono_up': float(mono_up),
        'range': float(max(vals) - min(vals)),
        'cv': float(std / np.mean(vals)) if np.mean(vals) > 0 else 0,
    }

for t in all_traces:
    ent_vals = step_vals(t['c'], 'mean_token_entropy')
    t['ent_traj'] = trajectory_features(ent_vals)
    dragin_vals = step_vals(t['c'], 'dragin_rind_max')
    t['dragin_traj'] = trajectory_features(dragin_vals)

traj_signals = [
    ('ent_slope', lambda t: t['ent_traj'].get('slope')),
    ('ent_spike', lambda t: t['ent_traj'].get('spike')),
    ('ent_argmax_late', lambda t: t['ent_traj'].get('argmax_frac')),
    ('ent_half_ratio', lambda t: t['ent_traj'].get('half_ratio')),
    ('ent_mono_up', lambda t: t['ent_traj'].get('mono_up')),
    ('ent_cv', lambda t: t['ent_traj'].get('cv')),
    ('ent_range', lambda t: t['ent_traj'].get('range')),
    ('dragin_slope', lambda t: t['dragin_traj'].get('slope')),
    ('dragin_late_spike', lambda t: t['dragin_traj'].get('argmax_frac')),
    ('dragin_cv', lambda t: t['dragin_traj'].get('cv')),
]

print(f"\n{'signal':<22} {'AUROC_inc':>10} {'AUROC_help':>11}")
for name, fn in traj_signals:
    pos_inc = [fn(t) for t in all_traces if is_inc(t) and fn(t) is not None]
    neg_inc = [fn(t) for t in all_traces if not is_inc(t) and fn(t) is not None]
    pos_help = [fn(t) for t in all_traces if is_help(t) and fn(t) is not None]
    neg_help = [fn(t) for t in all_traces if not is_help(t) and fn(t) is not None]
    a_inc = auroc(pos_inc, neg_inc)
    a_help = auroc(pos_help, neg_help)
    flag = " ***" if (a_inc > 0.6 or a_inc < 0.4 or a_help > 0.6 or a_help < 0.4) else ""
    print(f"  {name:<20} {a_inc:>10.3f} {a_help:>11.3f}{flag}")

# ===================================================================
# ANGLE 2: PAIRED TRACE DIVERGENCE
# ===================================================================
print("\n" + "=" * 80)
print("ANGLE 2: Oracle vs Closed trajectory divergence")
print("=" * 80)

def text_similarity(t1, t2):
    w1 = set(t1.lower().split())
    w2 = set(t2.lower().split())
    if not w1 or not w2: return 0.0
    return len(w1 & w2) / len(w1 | w2)

helped_traces = [t for t in all_traces if t['lift'] == +1]
not_helped_incorrect = [t for t in all_traces if t['lift'] == 0 and t['c_em'] == 0]

print(f"\n  Helped: {len(helped_traces)}, NotHelped+Incorrect: {len(not_helped_incorrect)}")

# For helped: compare n_steps between closed and oracle
for group, label in [(helped_traces, "HELPED"), (not_helped_incorrect, "NOT_HELPED_INC")]:
    c_steps = [t['c'].get('n_steps', 0) for t in group]
    o_steps = [t['o'].get('n_steps', 0) for t in group]
    c_toks = [t['c'].get('tokens_generated', 0) for t in group]
    o_toks = [t['o'].get('tokens_generated', 0) for t in group]
    if c_steps:
        print(f"\n  {label} (n={len(group)}):")
        print(f"    Closed: mean_steps={np.mean(c_steps):.1f}, mean_toks={np.mean(c_toks):.0f}")
        print(f"    Oracle: mean_steps={np.mean(o_steps):.1f}, mean_toks={np.mean(o_toks):.0f}")
        print(f"    Oracle/Closed tok ratio: {np.mean(o_toks)/max(1,np.mean(c_toks)):.2f}")

# ===================================================================
# ANGLE 3: PER-TOKEN PROBABILITY TRAJECTORY FEATURES
# ===================================================================
print("\n" + "=" * 80)
print("ANGLE 3: Per-token probability — position of minimum, trajectory decline")
print("=" * 80)

def min_prob_features(trace):
    steps = trace.get('steps', [])
    all_probs = []
    for s in steps:
        probs = s.get('per_token_max_logprob', [])
        for p in (probs or []):
            if p is not None:
                all_probs.append(float(p))
    if not all_probs: return {}
    total = len(all_probs)
    min_idx = np.argmin(all_probs)
    min_frac = min_idx / max(1, total - 1)
    third = total // 3
    if third > 0:
        first_mean = np.mean(all_probs[:third])
        last_mean = np.mean(all_probs[-third:])
        prob_decline = first_mean - last_mean
    else:
        prob_decline = 0
    # Count low-confidence tokens (prob < 0.5)
    n_low = sum(1 for p in all_probs if p < 0.5)
    frac_low = n_low / total
    # Longest streak of low-confidence tokens
    max_streak = 0; cur_streak = 0
    for p in all_probs:
        if p < 0.5:
            cur_streak += 1
            max_streak = max(max_streak, cur_streak)
        else:
            cur_streak = 0
    return {
        'min_prob_frac': min_frac,
        'min_prob_val': min(all_probs),
        'prob_decline': prob_decline,
        'frac_low_conf': frac_low,
        'max_low_streak': max_streak,
        'mean_prob': np.mean(all_probs),
    }

for t in all_traces:
    t['prob_feats'] = min_prob_features(t['c'])

prob_signals = [
    ('min_prob_position', lambda t: t['prob_feats'].get('min_prob_frac')),
    ('prob_decline', lambda t: t['prob_feats'].get('prob_decline')),
    ('frac_low_conf', lambda t: t['prob_feats'].get('frac_low_conf')),
    ('max_low_streak', lambda t: t['prob_feats'].get('max_low_streak')),
    ('inv_mean_prob', lambda t: 1.0 - t['prob_feats'].get('mean_prob', 1.0) if t['prob_feats'].get('mean_prob') is not None else None),
]

print(f"\n{'signal':<22} {'AUROC_inc':>10} {'AUROC_help':>11}")
for name, fn in prob_signals:
    pos_inc = [fn(t) for t in all_traces if is_inc(t) and fn(t) is not None]
    neg_inc = [fn(t) for t in all_traces if not is_inc(t) and fn(t) is not None]
    pos_help = [fn(t) for t in all_traces if is_help(t) and fn(t) is not None]
    neg_help = [fn(t) for t in all_traces if not is_help(t) and fn(t) is not None]
    a_inc = auroc(pos_inc, neg_inc)
    a_help = auroc(pos_help, neg_help)
    flag = " ***" if (a_inc > 0.6 or a_inc < 0.4 or a_help > 0.6 or a_help < 0.4) else ""
    print(f"  {name:<20} {a_inc:>10.3f} {a_help:>11.3f}{flag}")

# ===================================================================
# ANGLE 4: SIGNAL INTERACTIONS
# ===================================================================
print("\n" + "=" * 80)
print("ANGLE 4: Signal interactions (product, residual, conditional)")
print("=" * 80)

def get_dragin(t):
    vs = [float(v) for v in step_vals(t['c'], 'dragin_rind_max') if v is not None]
    return max(vs) if vs else None

def get_flare(t):
    steps = t['c'].get('steps', [])
    vals = []
    for s in steps:
        probs = s.get('per_token_max_logprob', [])
        if probs:
            valid = [float(p) for p in probs if p is not None]
            if valid:
                vals.append(1.0 - min(valid))
    return max(vals) if vals else None

def get_ent(t):
    vs = [float(v) for v in step_vals(t['c'], 'mean_token_entropy') if v is not None]
    return max(vs) if vs else None

for t in all_traces:
    t['dragin'] = get_dragin(t)
    t['flare'] = get_flare(t)
    t['ent'] = get_ent(t)

combos = [
    ('dragin*flare', lambda t: t['dragin'] * t['flare'] if t['dragin'] and t['flare'] else None),
    ('dragin+ent', lambda t: t['dragin'] + t['ent'] if t['dragin'] and t['ent'] else None),
    ('dragin-log_tok', lambda t: t['dragin'] - 0.3*math.log(max(1,t['c'].get('tokens_generated',1))) 
        if t['dragin'] else None),
    ('flare/ent', lambda t: t['flare'] / max(0.01, t['ent']) if t['flare'] and t['ent'] else None),
    ('max(dragin,flare)', lambda t: max(t['dragin'], t['flare']) if t['dragin'] and t['flare'] else None),
    ('dragin*ent_cv', lambda t: t['dragin'] * t['ent_traj'].get('cv', 0) if t['dragin'] and t['ent_traj'].get('cv') else None),
]

print(f"\n{'combo':<22} {'AUROC_inc':>10} {'AUROC_help':>11}")
for name, fn in combos:
    pos_inc = [fn(t) for t in all_traces if is_inc(t) and fn(t) is not None]
    neg_inc = [fn(t) for t in all_traces if not is_inc(t) and fn(t) is not None]
    pos_help = [fn(t) for t in all_traces if is_help(t) and fn(t) is not None]
    neg_help = [fn(t) for t in all_traces if not is_help(t) and fn(t) is not None]
    a_inc = auroc(pos_inc, neg_inc)
    a_help = auroc(pos_help, neg_help)
    flag = " ***" if (a_inc > 0.6 or a_inc < 0.4 or a_help > 0.6 or a_help < 0.4) else ""
    print(f"  {name:<20} {a_inc:>10.3f} {a_help:>11.3f}{flag}")

print(f"\n  --- baselines ---")
for name, fn in [('dragin', lambda t: t.get('dragin')), ('flare', lambda t: t.get('flare')),
                  ('tokens', lambda t: float(t['c'].get('tokens_generated', 0)))]:
    pos_inc = [fn(t) for t in all_traces if is_inc(t) and fn(t) is not None]
    neg_inc = [fn(t) for t in all_traces if not is_inc(t) and fn(t) is not None]
    pos_help = [fn(t) for t in all_traces if is_help(t) and fn(t) is not None]
    neg_help = [fn(t) for t in all_traces if not is_help(t) and fn(t) is not None]
    print(f"  {name:<20} {auroc(pos_inc, neg_inc):>10.3f} {auroc(pos_help, neg_help):>11.3f}")

# ===================================================================
# ANGLE 5: K=5 SAMPLE ANSWER EXTRACTION — cross-step dynamics
# ===================================================================
print("\n" + "=" * 80)
print("ANGLE 5: K=5 sample answer stability over steps (answer flip detection)")
print("=" * 80)

ANSWER_RE = re.compile(r"\bAnswer\s*:\s*(.+?)(?:[\n.]|$)", re.IGNORECASE)

def sample_answer_trajectory(trace):
    """Track how K=5 samples' answers change step by step."""
    steps = trace.get('steps', [])
    per_step_answers = []
    for s in steps:
        samples = s.get('samples', [])
        step_answers = set()
        for samp in samples:
            text = samp.get('text', '')
            m = ANSWER_RE.search(text)
            if m:
                ans = re.sub(r'[^a-z0-9 ]', '', m.group(1).strip().lower())[:30].strip()
                if ans:
                    step_answers.add(ans)
        per_step_answers.append(step_answers)
    return per_step_answers

def answer_flip_count(trace):
    """Number of steps where the set of sample-answers changes from previous step."""
    traj = sample_answer_trajectory(trace)
    traj = [s for s in traj if s]  # Only steps with extractable answers
    if len(traj) < 2: return None
    flips = sum(1 for i in range(1, len(traj)) if traj[i] != traj[i-1])
    return flips

def answer_convergence(trace):
    """Does the answer set shrink over steps? (converging = good)
    Return: final_n_answers / max_n_answers. Low = converged."""
    traj = sample_answer_trajectory(trace)
    traj_sizes = [len(s) for s in traj if s]
    if not traj_sizes: return None
    if max(traj_sizes) == 0: return None
    return traj_sizes[-1] / max(traj_sizes)

for t in all_traces:
    t['ans_flips'] = answer_flip_count(t['c'])
    t['ans_convergence'] = answer_convergence(t['c'])

flip_signals = [
    ('ans_flip_count', lambda t: t.get('ans_flips')),
    ('ans_no_converge', lambda t: t.get('ans_convergence')),
]

print(f"\n{'signal':<22} {'AUROC_inc':>10} {'AUROC_help':>11} {'pop':>5}")
for name, fn in flip_signals:
    valid = [t for t in all_traces if fn(t) is not None]
    pos_inc = [fn(t) for t in valid if is_inc(t)]
    neg_inc = [fn(t) for t in valid if not is_inc(t)]
    pos_help = [fn(t) for t in valid if is_help(t)]
    neg_help = [fn(t) for t in valid if not is_help(t)]
    a_inc = auroc(pos_inc, neg_inc)
    a_help = auroc(pos_help, neg_help)
    flag = " ***" if (a_inc > 0.6 or a_inc < 0.4 or a_help > 0.6 or a_help < 0.4) else ""
    print(f"  {name:<20} {a_inc:>10.3f} {a_help:>11.3f} {len(valid):>5}{flag}")

# ===================================================================
# ANGLE 6: HIDDEN STATE GEOMETRY — SeaKR-K5 TRAJECTORY
# ===================================================================
print("\n" + "=" * 80)
print("ANGLE 6: SeaKR-K5 logdet trajectory (geometry change over reasoning)")
print("=" * 80)

def seakr_k5_per_step(trace, alpha=1e-3):
    steps = trace.get('steps', [])
    logdets = []
    for s in steps:
        samples = s.get('samples', [])
        if not samples: logdets.append(None); continue
        H = []
        for samp in samples:
            h = samp.get('eos_hidden')
            if h is not None:
                H.append(h)
        if len(H) < 2: logdets.append(None); continue
        arr = np.asarray(H, dtype=np.float64)
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        arr = arr / norms
        G = arr @ arr.T + alpha * np.eye(arr.shape[0])
        sign, logabsdet = np.linalg.slogdet(G)
        logdets.append(float(logabsdet) if sign > 0 else None)
    return logdets

# Compute SeaKR trajectory features
for t in all_traces:
    logdets = seakr_k5_per_step(t['c'])
    valid = [v for v in logdets if v is not None]
    t['seakr_traj'] = trajectory_features(valid) if len(valid) >= 2 else {}
    t['seakr_max'] = max(valid) if valid else None
    t['seakr_last'] = valid[-1] if valid else None
    t['seakr_first'] = valid[0] if valid else None

seakr_traj_signals = [
    ('seakr_max', lambda t: t.get('seakr_max')),
    ('seakr_last', lambda t: t.get('seakr_last')),
    ('seakr_slope', lambda t: t['seakr_traj'].get('slope')),
    ('seakr_range', lambda t: t['seakr_traj'].get('range')),
    ('seakr_argmax_late', lambda t: t['seakr_traj'].get('argmax_frac')),
    ('seakr_last-first', lambda t: t['seakr_last'] - t['seakr_first'] if t.get('seakr_last') and t.get('seakr_first') else None),
]

print(f"\n{'signal':<22} {'AUROC_inc':>10} {'AUROC_help':>11}")
for name, fn in seakr_traj_signals:
    pos_inc = [fn(t) for t in all_traces if is_inc(t) and fn(t) is not None]
    neg_inc = [fn(t) for t in all_traces if not is_inc(t) and fn(t) is not None]
    pos_help = [fn(t) for t in all_traces if is_help(t) and fn(t) is not None]
    neg_help = [fn(t) for t in all_traces if not is_help(t) and fn(t) is not None]
    a_inc = auroc(pos_inc, neg_inc)
    a_help = auroc(pos_help, neg_help)
    flag = " ***" if (a_inc > 0.6 or a_inc < 0.4 or a_help > 0.6 or a_help < 0.4) else ""
    print(f"  {name:<20} {a_inc:>10.3f} {a_help:>11.3f}{flag}")

# ===================================================================
# ANGLE 7: PER-CELL BREAKDOWN — which signals win WHERE
# ===================================================================
print("\n" + "=" * 80)
print("ANGLE 7: Per-cell signal ranking (arch x dataset specificity)")
print("=" * 80)

ALL_SIGNALS = [
    ('tokens', lambda t: float(t['c'].get('tokens_generated', 0))),
    ('dragin', lambda t: t.get('dragin')),
    ('flare', lambda t: t.get('flare')),
    ('ent_cv', lambda t: t['ent_traj'].get('cv')),
    ('ent_slope', lambda t: t['ent_traj'].get('slope')),
    ('prob_decline', lambda t: t['prob_feats'].get('prob_decline')),
    ('frac_low_conf', lambda t: t['prob_feats'].get('frac_low_conf')),
    ('seakr_max', lambda t: t.get('seakr_max')),
    ('dragin*flare', lambda t: t['dragin'] * t['flare'] if t['dragin'] and t['flare'] else None),
    ('dragin-log_tok', lambda t: t['dragin'] - 0.3*math.log(max(1,t['c'].get('tokens_generated',1))) 
        if t['dragin'] else None),
]

cells = defaultdict(list)
for t in all_traces:
    cells[t['cell']].append(t)

print(f"\n{'cell':<12} ", end="")
for name, _ in ALL_SIGNALS:
    print(f"{name[:8]:>9}", end="")
print()

for cell_name in sorted(cells.keys()):
    cell_rows = cells[cell_name]
    n_inc = sum(1 for t in cell_rows if is_inc(t))
    if n_inc < 3: continue
    print(f"{cell_name:<12} ", end="")
    for name, fn in ALL_SIGNALS:
        pos = [fn(t) for t in cell_rows if is_inc(t) and fn(t) is not None]
        neg = [fn(t) for t in cell_rows if not is_inc(t) and fn(t) is not None]
        a = auroc(pos, neg)
        s = f"{a:.2f}" if a == a else "n/a"
        print(f"{s:>9}", end="")
    print()

# ===================================================================
# ANGLE 8: STEP-LEVEL SIGNAL CORRELATION
# What moves together? What's orthogonal?
# ===================================================================
print("\n" + "=" * 80)
print("ANGLE 8: Step-level signal correlation matrix (which are redundant?)")
print("=" * 80)

step_features = []
for t in all_traces:
    for s in t['c'].get('steps', []):
        feats = {}
        feats['dragin'] = s.get('dragin_rind_max')
        feats['entropy'] = s.get('mean_token_entropy')
        feats['n_tokens'] = s.get('n_tokens_in_step')
        feats['hidden_norm'] = s.get('hidden_norm')
        feats['drift'] = s.get('drift_cosine')
        feats['hvr'] = s.get('hvr')
        probs = s.get('per_token_max_logprob', [])
        feats['flare_step'] = 1.0 - min(float(p) for p in probs if p is not None) if probs else None
        feats['u_cons'] = s.get('u_cons')
        if all(v is not None for v in feats.values()):
            step_features.append(feats)

if step_features:
    keys = sorted(step_features[0].keys())
    mat = np.array([[f[k] for k in keys] for f in step_features])
    corr = np.corrcoef(mat.T)
    print(f"\n  n_steps_with_all_features = {len(step_features)}")
    print(f"\n  {'':>12}", end="")
    for k in keys:
        print(f"{k[:8]:>9}", end="")
    print()
    for i, ki in enumerate(keys):
        print(f"  {ki[:12]:<12}", end="")
        for j in range(len(keys)):
            print(f"{corr[i,j]:>9.2f}", end="")
        print()

# ===================================================================
# ANGLE 9: ORACLE TRACE "RETRIEVAL BENEFIT SIGNATURE"
# Does the oracle trace show DIFFERENT internal signals? 
# ===================================================================
print("\n" + "=" * 80)
print("ANGLE 9: Oracle-trace signal shift (how retrieval changes model internals)")
print("=" * 80)

def trace_summary(r):
    dragin_vals = [float(v) for v in step_vals(r, 'dragin_rind_max') if v is not None]
    ent_vals = [float(v) for v in step_vals(r, 'mean_token_entropy') if v is not None]
    return {
        'dragin_max': max(dragin_vals) if dragin_vals else None,
        'dragin_mean': np.mean(dragin_vals) if dragin_vals else None,
        'ent_max': max(ent_vals) if ent_vals else None,
        'ent_mean': np.mean(ent_vals) if ent_vals else None,
        'n_steps': r.get('n_steps', 0),
        'tokens': r.get('tokens_generated', 0),
    }

for t in all_traces:
    t['c_sum'] = trace_summary(t['c'])
    t['o_sum'] = trace_summary(t['o'])

# Compare closed vs oracle signals for helped traces
print(f"\n  Signal shift: Oracle minus Closed (helped traces, n={len(helped_traces)})")
for key in ['dragin_max', 'dragin_mean', 'ent_max', 'ent_mean', 'n_steps', 'tokens']:
    c_vals = [t['c_sum'][key] for t in helped_traces if t['c_sum'].get(key) is not None and t['o_sum'].get(key) is not None]
    o_vals = [t['o_sum'][key] for t in helped_traces if t['c_sum'].get(key) is not None and t['o_sum'].get(key) is not None]
    if c_vals and o_vals:
        delta = np.mean(o_vals) - np.mean(c_vals)
        print(f"    {key:<15}: closed={np.mean(c_vals):.3f}  oracle={np.mean(o_vals):.3f}  Δ={delta:+.3f}")

# ===================================================================
# ANGLE 10: "EARLY RETRIEVAL NEED" — can we detect at step 1-2?
# ===================================================================
print("\n" + "=" * 80)
print("ANGLE 10: Early detection — can signals at steps 1-2 predict lift_em=+1?")
print("=" * 80)

def early_features(trace):
    steps = trace.get('steps', [])
    if not steps: return {}
    s0 = steps[0]
    s1 = steps[1] if len(steps) > 1 else None
    feats = {}
    feats['step0_dragin'] = s0.get('dragin_rind_max')
    feats['step0_entropy'] = s0.get('mean_token_entropy')
    feats['step0_n_tokens'] = s0.get('n_tokens_in_step')
    probs = s0.get('per_token_max_logprob', [])
    feats['step0_min_prob'] = min(float(p) for p in probs if p is not None) if probs else None
    if s1:
        feats['step1_dragin'] = s1.get('dragin_rind_max')
        feats['step1_entropy'] = s1.get('mean_token_entropy')
        p1 = s1.get('per_token_max_logprob', [])
        feats['step1_min_prob'] = min(float(p) for p in p1 if p is not None) if p1 else None
        # Step 0→1 change
        if feats['step0_entropy'] and feats['step1_entropy']:
            feats['ent_jump_01'] = feats['step1_entropy'] - feats['step0_entropy']
        if feats['step0_dragin'] and feats['step1_dragin']:
            feats['dragin_jump_01'] = feats['step1_dragin'] - feats['step0_dragin']
    return feats

for t in all_traces:
    t['early'] = early_features(t['c'])

early_signals = [
    ('step0_dragin', lambda t: t['early'].get('step0_dragin')),
    ('step0_entropy', lambda t: t['early'].get('step0_entropy')),
    ('step0_min_prob', lambda t: -(t['early'].get('step0_min_prob') or 1.0)),
    ('step1_dragin', lambda t: t['early'].get('step1_dragin')),
    ('step1_entropy', lambda t: t['early'].get('step1_entropy')),
    ('ent_jump_01', lambda t: t['early'].get('ent_jump_01')),
    ('dragin_jump_01', lambda t: t['early'].get('dragin_jump_01')),
]

print(f"\n{'signal':<22} {'AUROC_inc':>10} {'AUROC_help':>11}")
for name, fn in early_signals:
    pos_inc = [fn(t) for t in all_traces if is_inc(t) and fn(t) is not None]
    neg_inc = [fn(t) for t in all_traces if not is_inc(t) and fn(t) is not None]
    pos_help = [fn(t) for t in all_traces if is_help(t) and fn(t) is not None]
    neg_help = [fn(t) for t in all_traces if not is_help(t) and fn(t) is not None]
    a_inc = auroc(pos_inc, neg_inc)
    a_help = auroc(pos_help, neg_help)
    flag = " ***" if (a_inc > 0.6 or a_inc < 0.4 or a_help > 0.6 or a_help < 0.4) else ""
    print(f"  {name:<20} {a_inc:>10.3f} {a_help:>11.3f}{flag}")

print("\n\n" + "=" * 80)
print("SUMMARY: Top signals by AUROC (all angles pooled)")
print("=" * 80)

# Collect all signals tested and rank
all_signal_results = []
for name, fn in (traj_signals + prob_signals + combos + flip_signals + 
                  seakr_traj_signals + early_signals +
                  [('dragin', lambda t: t.get('dragin')), 
                   ('flare', lambda t: t.get('flare')),
                   ('tokens', lambda t: float(t['c'].get('tokens_generated', 0)))]):
    pos_inc = [fn(t) for t in all_traces if is_inc(t) and fn(t) is not None]
    neg_inc = [fn(t) for t in all_traces if not is_inc(t) and fn(t) is not None]
    pos_help = [fn(t) for t in all_traces if is_help(t) and fn(t) is not None]
    neg_help = [fn(t) for t in all_traces if not is_help(t) and fn(t) is not None]
    a_inc = auroc(pos_inc, neg_inc)
    a_help = auroc(pos_help, neg_help)
    all_signal_results.append((name, a_inc, a_help))

# Sort by AUROC for incorrect
print(f"\n  TOP by AUROC-incorrect:")
for name, a_inc, a_help in sorted(all_signal_results, key=lambda x: -x[1] if x[1]==x[1] else -99)[:15]:
    print(f"    {name:<22} inc={a_inc:.3f}  help={a_help:.3f}")

print(f"\n  TOP by AUROC-helpful:")
for name, a_inc, a_help in sorted(all_signal_results, key=lambda x: -x[2] if x[2]==x[2] else -99)[:15]:
    print(f"    {name:<22} inc={a_inc:.3f}  help={a_help:.3f}")
