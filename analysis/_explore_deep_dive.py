"""Deep dive on the top 3 novel signals from _explore_novel.py:
1. ans_flip_count (K=5 sample answer instability across steps)
2. max_low_streak (longest run of low-confidence tokens)
3. dragin_late_spike (WHERE in the trace DRAGIN peaks)

Questions:
- Do they survive length residualization?
- Which arch/dataset cells are they strongest on?
- What's the mechanism? (trace-level examples)
- Are they correlated with each other or independent?
- Can we define a practical online trigger from them?
"""
import json
import math
import re
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _common import auroc as _auroc

ROOT = HERE.parent

def auroc(pos, neg):
    if not pos or not neg: return float('nan')
    return _auroc(pos, neg)

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

print(f"Loaded {len(all_traces)} paired traces\n")

# ===================================================================
# EXTRACT THE 3 NOVEL SIGNALS + baselines for all traces
# ===================================================================
ANSWER_RE = re.compile(r"\bAnswer\s*:\s*(.+?)(?:[\n.]|$)", re.IGNORECASE)

def ans_flip_count(trace):
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
    # Count transitions where answer set changes
    answer_bearing = [(i, s) for i, s in enumerate(per_step_answers) if s]
    if len(answer_bearing) < 2: return None
    flips = sum(1 for i in range(1, len(answer_bearing)) 
                if answer_bearing[i][1] != answer_bearing[i-1][1])
    return float(flips)

def max_low_streak(trace):
    steps = trace.get('steps', [])
    all_probs = []
    for s in steps:
        probs = s.get('per_token_max_logprob', [])
        for p in (probs or []):
            if p is not None:
                all_probs.append(float(p))
    if not all_probs: return None
    max_s = 0; cur = 0
    for p in all_probs:
        if p < 0.5:
            cur += 1
            max_s = max(max_s, cur)
        else:
            cur = 0
    return float(max_s)

def dragin_late_spike(trace):
    vals = [float(v) for v in step_vals(trace, 'dragin_rind_max') if v is not None]
    if len(vals) < 2: return None
    return float(np.argmax(vals)) / max(1, len(vals) - 1)

def get_tokens(trace):
    return float(trace.get('tokens_generated', 0))

def get_dragin_max(trace):
    vals = [float(v) for v in step_vals(trace, 'dragin_rind_max') if v is not None]
    return max(vals) if vals else None

def get_flare_max(trace):
    steps = trace.get('steps', [])
    vals = []
    for s in steps:
        probs = s.get('per_token_max_logprob', [])
        if probs:
            valid = [float(p) for p in probs if p is not None]
            if valid:
                vals.append(1.0 - min(valid))
    return max(vals) if vals else None

# Extract all signals
for t in all_traces:
    t['ans_flips'] = ans_flip_count(t['c'])
    t['low_streak'] = max_low_streak(t['c'])
    t['dragin_late'] = dragin_late_spike(t['c'])
    t['tokens'] = get_tokens(t['c'])
    t['log_tokens'] = math.log(max(1, int(t['tokens'])) + 1)
    t['dragin_max'] = get_dragin_max(t['c'])
    t['flare_max'] = get_flare_max(t['c'])
    # N_steps (potential confounder for ans_flips)
    t['n_steps'] = float(t['c'].get('n_steps', 0))

is_inc = lambda t: t['c_em'] == 0
is_help = lambda t: t['lift'] == +1

# ===================================================================
# 1. LENGTH RESIDUALIZATION — do novel signals survive?
# ===================================================================
print("=" * 80)
print("1. LENGTH RESIDUALIZATION — do novel signals survive log(tokens)?")
print("=" * 80)

def ols_residualize(traces, key, x_key='log_tokens'):
    pts = [(t[x_key], t[key]) for t in traces if t.get(key) is not None and t.get(x_key) is not None]
    if len(pts) < 3: return
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    n = len(pts); mx = sum(xs)/n; my = sum(ys)/n
    num = sum((xs[i]-mx)*(ys[i]-my) for i in range(n))
    den = sum((xs[i]-mx)**2 for i in range(n))
    b = num/den if den > 0 else 0.0; a = my - b*mx
    for t in traces:
        if t.get(key) is not None:
            t[f'{key}_resid'] = t[key] - (a + b * t[x_key])
        else:
            t[f'{key}_resid'] = None

# Also residualize on n_steps (ans_flips could be step-count confounded)
def ols_residualize_2d(traces, key, x_keys=('log_tokens', 'n_steps')):
    """Residualize on two predictors."""
    valid = [t for t in traces if t.get(key) is not None and all(t.get(k) is not None for k in x_keys)]
    if len(valid) < 5: return
    Y = np.array([t[key] for t in valid])
    X = np.column_stack([np.ones(len(valid))] + [np.array([t[k] for t in valid]) for k in x_keys])
    try:
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    except:
        return
    for t in traces:
        if t.get(key) is not None and all(t.get(k) is not None for k in x_keys):
            pred = beta[0] + sum(beta[i+1] * t[x_keys[i]] for i in range(len(x_keys)))
            t[f'{key}_resid2d'] = t[key] - pred
        else:
            t[f'{key}_resid2d'] = None

signals_to_test = [
    ('ans_flips', 'K=5 answer flips'),
    ('low_streak', 'max low-conf streak'),
    ('dragin_late', 'DRAGIN peak position'),
    ('dragin_max', 'DRAGIN.max (baseline)'),
    ('flare_max', 'FLARE.max (baseline)'),
    ('tokens', 'tokens (reference)'),
]

for key, _ in signals_to_test:
    ols_residualize(all_traces, key)
    if key == 'ans_flips':
        ols_residualize_2d(all_traces, key)

print(f"\n  {'signal':<25} {'raw_inc':>8} {'resid_inc':>10} {'raw_help':>9} {'resid_help':>11}  {'Δ_inc':>7}")
for key, label in signals_to_test:
    if key == 'tokens': continue
    raw_inc = auroc([t[key] for t in all_traces if is_inc(t) and t.get(key) is not None],
                    [t[key] for t in all_traces if not is_inc(t) and t.get(key) is not None])
    res_inc = auroc([t[f'{key}_resid'] for t in all_traces if is_inc(t) and t.get(f'{key}_resid') is not None],
                    [t[f'{key}_resid'] for t in all_traces if not is_inc(t) and t.get(f'{key}_resid') is not None])
    raw_help = auroc([t[key] for t in all_traces if is_help(t) and t.get(key) is not None],
                     [t[key] for t in all_traces if not is_help(t) and t.get(key) is not None])
    res_help = auroc([t[f'{key}_resid'] for t in all_traces if is_help(t) and t.get(f'{key}_resid') is not None],
                     [t[f'{key}_resid'] for t in all_traces if not is_help(t) and t.get(f'{key}_resid') is not None])
    delta = res_inc - raw_inc if (res_inc == res_inc and raw_inc == raw_inc) else float('nan')
    print(f"  {label:<25} {raw_inc:>8.3f} {res_inc:>10.3f} {raw_help:>9.3f} {res_help:>11.3f}  {delta:>+7.3f}")

# Special: ans_flips residualized on BOTH tokens AND n_steps
key = 'ans_flips'
res2d_inc = auroc([t[f'{key}_resid2d'] for t in all_traces if is_inc(t) and t.get(f'{key}_resid2d') is not None],
                  [t[f'{key}_resid2d'] for t in all_traces if not is_inc(t) and t.get(f'{key}_resid2d') is not None])
res2d_help = auroc([t[f'{key}_resid2d'] for t in all_traces if is_help(t) and t.get(f'{key}_resid2d') is not None],
                   [t[f'{key}_resid2d'] for t in all_traces if not is_help(t) and t.get(f'{key}_resid2d') is not None])
print(f"\n  {'ans_flips resid(tok+steps)':<25} {'':>8} {res2d_inc:>10.3f} {'':>9} {res2d_help:>11.3f}  (double-residualized)")

# ===================================================================
# 2. PER-CELL BREAKDOWN with residualization
# ===================================================================
print("\n\n" + "=" * 80)
print("2. PER-CELL BREAKDOWN (raw + residualized)")
print("=" * 80)

cells = defaultdict(list)
for t in all_traces:
    cells[t['cell']].append(t)

novel_signals = [('ans_flips', 'flip'), ('low_streak', 'streak'), ('dragin_late', 'late'),
                 ('dragin_max', 'DRAG'), ('tokens', 'tok')]

print(f"\n  RAW AUROC vs incorrect:")
print(f"  {'cell':<12}", end="")
for _, short in novel_signals:
    print(f" {short:>7}", end="")
print()
for cell_name in sorted(cells.keys()):
    rows = cells[cell_name]
    n_inc = sum(1 for t in rows if is_inc(t))
    if n_inc < 3: continue
    print(f"  {cell_name:<12}", end="")
    for key, _ in novel_signals:
        pos = [t[key] for t in rows if is_inc(t) and t.get(key) is not None]
        neg = [t[key] for t in rows if not is_inc(t) and t.get(key) is not None]
        a = auroc(pos, neg)
        print(f" {a:>7.3f}" if a==a else f" {'n/a':>7}", end="")
    print()

# Per-cell residualization
print(f"\n  RESIDUALIZED (on log_tokens) AUROC vs incorrect:")
print(f"  {'cell':<12}", end="")
for _, short in novel_signals:
    if short == 'tok': continue
    print(f" {short:>7}", end="")
print()
for cell_name in sorted(cells.keys()):
    rows = cells[cell_name]
    n_inc = sum(1 for t in rows if is_inc(t))
    if n_inc < 3: continue
    # Per-cell OLS residualization
    for key, _ in novel_signals:
        if key == 'tokens': continue
        pts = [(t['log_tokens'], t[key]) for t in rows if t.get(key) is not None]
        if len(pts) < 3: continue
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        n = len(pts); mx = sum(xs)/n; my = sum(ys)/n
        num = sum((xs[i]-mx)*(ys[i]-my) for i in range(n))
        den = sum((xs[i]-mx)**2 for i in range(n))
        b = num/den if den > 0 else 0.0; a_ = my - b*mx
        for t in rows:
            if t.get(key) is not None:
                t[f'{key}_cell_resid'] = t[key] - (a_ + b * t['log_tokens'])
    
    print(f"  {cell_name:<12}", end="")
    for key, _ in novel_signals:
        if key == 'tokens': continue
        pos = [t[f'{key}_cell_resid'] for t in rows if is_inc(t) and t.get(f'{key}_cell_resid') is not None]
        neg = [t[f'{key}_cell_resid'] for t in rows if not is_inc(t) and t.get(f'{key}_cell_resid') is not None]
        a = auroc(pos, neg)
        print(f" {a:>7.3f}" if a==a else f" {'n/a':>7}", end="")
    print()

# ===================================================================
# 3. CORRELATION BETWEEN NOVEL SIGNALS
# ===================================================================
print("\n\n" + "=" * 80)
print("3. INTER-SIGNAL CORRELATION (are they measuring different things?)")
print("=" * 80)

valid = [t for t in all_traces if all(t.get(k) is not None for k in ['ans_flips', 'low_streak', 'dragin_late', 'dragin_max', 'flare_max', 'tokens'])]
if valid:
    keys = ['ans_flips', 'low_streak', 'dragin_late', 'dragin_max', 'flare_max', 'tokens', 'n_steps']
    mat = np.array([[t[k] for k in keys] for t in valid])
    corr = np.corrcoef(mat.T)
    print(f"\n  n={len(valid)}")
    print(f"  {'':>14}", end="")
    for k in keys:
        print(f" {k[:8]:>9}", end="")
    print()
    for i, ki in enumerate(keys):
        print(f"  {ki[:14]:<14}", end="")
        for j in range(len(keys)):
            print(f" {corr[i,j]:>9.2f}", end="")
        print()

# ===================================================================
# 4. MECHANISM: What do high-ans_flip traces look like?
# ===================================================================
print("\n\n" + "=" * 80)
print("4. MECHANISM: High vs Low ans_flip traces (what's different?)")
print("=" * 80)

valid_flips = [t for t in all_traces if t.get('ans_flips') is not None]
valid_flips.sort(key=lambda t: t['ans_flips'], reverse=True)

# Top 10 vs bottom 10
top = valid_flips[:15]
bot = valid_flips[-15:]

def describe_group(group, label):
    toks = [t['tokens'] for t in group]
    steps = [t['n_steps'] for t in group]
    em = [t['c_em'] for t in group]
    lift = [t['lift'] for t in group]
    dragin = [t['dragin_max'] for t in group if t['dragin_max'] is not None]
    print(f"\n  {label} (n={len(group)}):")
    print(f"    tokens: {np.mean(toks):.0f} ± {np.std(toks):.0f}")
    print(f"    n_steps: {np.mean(steps):.1f} ± {np.std(steps):.1f}")
    print(f"    correct: {sum(em)}/{len(em)} ({100*np.mean(em):.0f}%)")
    print(f"    lift+1: {sum(1 for l in lift if l==+1)}/{len(lift)}")
    print(f"    dragin_max: {np.mean(dragin):.3f}")
    print(f"    ans_flips: {np.mean([t['ans_flips'] for t in group]):.1f}")
    # Arch distribution
    archs = Counter(t['arch'] for t in group)
    print(f"    archs: {dict(archs)}")

describe_group(top, "HIGH ans_flips (top 15)")
describe_group(bot, "LOW ans_flips (bottom 15)")

# ===================================================================
# 5. PRACTICAL TRIGGER: Online-computable version of ans_flips
# ===================================================================
print("\n\n" + "=" * 80)
print("5. PRACTICAL TRIGGER DESIGN: Online ans_flip detection")
print("=" * 80)

# The key insight: ans_flips measures reasoning instability.
# For an online trigger: at step t, check if K=5 samples' answers
# differ from step t-1's samples' answers. Fire if they flip.
# This is computable at each boundary!

# Let's verify: per-step signal (1 if this step flipped from previous)
def per_step_flip_signal(trace):
    """Return list of 0/1 per step: did the K=5 answer set change?"""
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
    
    flips = [0.0] * len(steps)
    prev_answers = set()
    for i, answers in enumerate(per_step_answers):
        if not answers:
            continue
        if prev_answers and answers != prev_answers:
            flips[i] = 1.0
        prev_answers = answers
    return flips

# Statistics on flip timing
print("\n  Where in the trace do answer flips happen?")
flip_positions = []
for t in all_traces:
    flips = per_step_flip_signal(t['c'])
    n = len(flips)
    if n < 2: continue
    for i, f in enumerate(flips):
        if f > 0:
            flip_positions.append(i / max(1, n-1))

if flip_positions:
    print(f"    Total flips observed: {len(flip_positions)}")
    print(f"    Mean position (0=start, 1=end): {np.mean(flip_positions):.3f}")
    print(f"    Median position: {np.median(flip_positions):.3f}")
    print(f"    Std: {np.std(flip_positions):.3f}")
    # Quartile distribution
    q1 = sum(1 for p in flip_positions if p < 0.25)
    q2 = sum(1 for p in flip_positions if 0.25 <= p < 0.5)
    q3 = sum(1 for p in flip_positions if 0.5 <= p < 0.75)
    q4 = sum(1 for p in flip_positions if p >= 0.75)
    total = len(flip_positions)
    print(f"    Q1(<0.25): {q1} ({100*q1/total:.0f}%), Q2: {q2} ({100*q2/total:.0f}%), "
          f"Q3: {q3} ({100*q3/total:.0f}%), Q4(>0.75): {q4} ({100*q4/total:.0f}%)")

# ===================================================================
# 6. max_low_streak — is it length-driven or genuinely different?
# ===================================================================
print("\n\n" + "=" * 80)
print("6. max_low_streak: NORMALIZED by trace length")
print("=" * 80)

def max_low_streak_normalized(trace):
    steps = trace.get('steps', [])
    all_probs = []
    for s in steps:
        probs = s.get('per_token_max_logprob', [])
        for p in (probs or []):
            if p is not None:
                all_probs.append(float(p))
    if not all_probs: return None
    max_s = 0; cur = 0
    for p in all_probs:
        if p < 0.5:
            cur += 1
            max_s = max(max_s, cur)
        else:
            cur = 0
    return float(max_s) / len(all_probs)  # Normalize by total tokens

for t in all_traces:
    t['low_streak_norm'] = max_low_streak_normalized(t['c'])

# Test normalized version
for key, label in [('low_streak', 'raw streak'), ('low_streak_norm', 'normalized streak')]:
    inc_auroc = auroc([t[key] for t in all_traces if is_inc(t) and t.get(key) is not None],
                      [t[key] for t in all_traces if not is_inc(t) and t.get(key) is not None])
    help_auroc = auroc([t[key] for t in all_traces if is_help(t) and t.get(key) is not None],
                       [t[key] for t in all_traces if not is_help(t) and t.get(key) is not None])
    print(f"  {label:<20} inc={inc_auroc:.3f}  help={help_auroc:.3f}")

# ===================================================================
# 7. DRAGIN_LATE: What does it mean that late DRAGIN predicts help?
# ===================================================================
print("\n\n" + "=" * 80)
print("7. DRAGIN peak timing: mechanism (late uncertainty = retrieval need)")
print("=" * 80)

# If DRAGIN peaks late, it means the model's attention-entropy-content
# signal is HIGH at the END of reasoning. This suggests:
# "The model was uncertain at conclusion time" — not just uncertain somewhere.

# Combine: DRAGIN peaks late AND high?
def dragin_late_and_high(t):
    if t.get('dragin_late') is None or t.get('dragin_max') is None: return None
    return t['dragin_late'] * t['dragin_max']

for t in all_traces:
    t['dragin_late_x_high'] = dragin_late_and_high(t)

for key, label in [('dragin_max', 'DRAGIN max (where)'),
                   ('dragin_late', 'DRAGIN peak position'),
                   ('dragin_late_x_high', 'DRAGIN late×high')]:
    inc_a = auroc([t[key] for t in all_traces if is_inc(t) and t.get(key) is not None],
                  [t[key] for t in all_traces if not is_inc(t) and t.get(key) is not None])
    help_a = auroc([t[key] for t in all_traces if is_help(t) and t.get(key) is not None],
                   [t[key] for t in all_traces if not is_help(t) and t.get(key) is not None])
    print(f"  {label:<28} inc={inc_a:.3f}  help={help_a:.3f}")

# ===================================================================
# 8. THE BIG QUESTION: Can we build a single composite trigger that
#    dominates everything in the published pool?
# ===================================================================
print("\n\n" + "=" * 80)
print("8. COMPOSITE TRIGGER: rank-sum of top-3 novel signals")
print("=" * 80)

# Rank-based combination (non-parametric; robust)
def rank_normalize(traces, key):
    valid = [(i, t[key]) for i, t in enumerate(traces) if t.get(key) is not None]
    valid.sort(key=lambda x: x[1])
    for rank, (idx, _) in enumerate(valid):
        traces[idx][f'{key}_rank'] = rank / max(1, len(valid) - 1)

for key in ['ans_flips', 'low_streak', 'dragin_late']:
    rank_normalize(all_traces, key)

# Composite: mean rank of top-3
for t in all_traces:
    ranks = []
    for key in ['ans_flips', 'low_streak', 'dragin_late']:
        if t.get(f'{key}_rank') is not None:
            ranks.append(t[f'{key}_rank'])
    t['composite_top3'] = np.mean(ranks) if len(ranks) >= 2 else None

# Also: composite minus tokens-rank
rank_normalize(all_traces, 'tokens')
for t in all_traces:
    if t.get('composite_top3') is not None and t.get('tokens_rank') is not None:
        t['composite_resid'] = t['composite_top3'] - 0.5 * t['tokens_rank']
    else:
        t['composite_resid'] = None

for key, label in [('composite_top3', 'Composite (flip+streak+late)'),
                   ('composite_resid', 'Composite - 0.5*tok_rank'),
                   ('dragin_max', 'DRAGIN.max (baseline)'),
                   ('tokens', 'tokens (reference)')]:
    inc_a = auroc([t[key] for t in all_traces if is_inc(t) and t.get(key) is not None],
                  [t[key] for t in all_traces if not is_inc(t) and t.get(key) is not None])
    help_a = auroc([t[key] for t in all_traces if is_help(t) and t.get(key) is not None],
                   [t[key] for t in all_traces if not is_help(t) and t.get(key) is not None])
    print(f"  {label:<35} inc={inc_a:.3f}  help={help_a:.3f}")

# Per-cell composite
print(f"\n  Per-cell composite AUROC (vs incorrect):")
print(f"  {'cell':<12} {'composite':>9} {'DRAGIN':>9} {'tokens':>9}")
for cell_name in sorted(cells.keys()):
    rows = cells[cell_name]
    n_inc = sum(1 for t in rows if is_inc(t))
    if n_inc < 3: continue
    comp = auroc([t['composite_top3'] for t in rows if is_inc(t) and t.get('composite_top3') is not None],
                 [t['composite_top3'] for t in rows if not is_inc(t) and t.get('composite_top3') is not None])
    drag = auroc([t['dragin_max'] for t in rows if is_inc(t) and t.get('dragin_max') is not None],
                 [t['dragin_max'] for t in rows if not is_inc(t) and t.get('dragin_max') is not None])
    toks = auroc([t['tokens'] for t in rows if is_inc(t)],
                 [t['tokens'] for t in rows if not is_inc(t)])
    print(f"  {cell_name:<12} {comp:>9.3f} {drag:>9.3f} {toks:>9.3f}")

# ===================================================================
# 9. EXAMPLE TRACES — show what ans_flip looks like concretely
# ===================================================================
print("\n\n" + "=" * 80)
print("9. EXAMPLE: Highest ans_flip trace (mechanism illustration)")
print("=" * 80)

valid_flips_sorted = sorted([t for t in all_traces if t.get('ans_flips') is not None], 
                            key=lambda t: t['ans_flips'], reverse=True)
if valid_flips_sorted:
    example = valid_flips_sorted[0]
    c = example['c']
    print(f"\n  Question: {c.get('question', '')[:100]}")
    print(f"  Gold: {c.get('gold_answer')}")
    print(f"  Model answer: {c.get('model_answer_extracted')}")
    print(f"  Correct: {example['c_em']}  Lift: {example['lift']}")
    print(f"  Tokens: {c.get('tokens_generated')}  Steps: {c.get('n_steps')}  Flips: {example['ans_flips']}")
    print(f"  Arch: {example['arch']}  Dataset: {example['ds']}")
    print(f"\n  Step-by-step K=5 answers:")
    for i, s in enumerate(c.get('steps', [])):
        samples = s.get('samples', [])
        answers = []
        for samp in samples:
            text = samp.get('text', '')
            m = ANSWER_RE.search(text)
            if m:
                answers.append(m.group(1).strip()[:30])
        if answers:
            print(f"    step {i:>2}: {answers}")
