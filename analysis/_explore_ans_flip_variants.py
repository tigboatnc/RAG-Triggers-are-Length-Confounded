"""Targeted analysis on the ans_flip signal family.

Key question: If flips happen late (69% in Q4), can we build:
1. A "first flip" early-warning trigger (fire at the first flip)
2. A "late disagreement" trigger (only count flips in last 50% of trace)
3. A "flip intensity" trigger (max K=5 distinct answers at any step)
4. A semantic variant: not just set equality, but edit-distance between answers

Also: What if we look at AGREEMENT instead of flipping?
"All K=5 agree on an answer that differs from the current greedy path"
= strong retrieval signal (model knows what to say if asked directly)
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

ANSWER_RE = re.compile(r"\bAnswer\s*:\s*(.+?)(?:[\n.]|$)", re.IGNORECASE)

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
is_inc = lambda t: t['c_em'] == 0
is_help = lambda t: t['lift'] == +1

# ===================================================================
# Extract rich answer-stability features per trace
# ===================================================================
def extract_answer_features(trace):
    steps = trace.get('steps', [])
    n_steps = len(steps)
    if n_steps == 0:
        return {}
    
    per_step_answers = []
    per_step_n_distinct = []
    per_step_majority_frac = []
    
    for s in steps:
        samples = s.get('samples', [])
        answers = []
        for samp in samples:
            text = samp.get('text', '')
            m = ANSWER_RE.search(text)
            if m:
                ans = re.sub(r'[^a-z0-9 ]', '', m.group(1).strip().lower())[:30].strip()
                if ans:
                    answers.append(ans)
        per_step_answers.append(answers)
        if answers:
            per_step_n_distinct.append(len(set(answers)))
            counts = Counter(answers)
            majority = max(counts.values())
            per_step_majority_frac.append(majority / len(answers))
        else:
            per_step_n_distinct.append(None)
            per_step_majority_frac.append(None)
    
    # Answer-bearing steps
    answer_steps = [(i, set(a)) for i, a in enumerate(per_step_answers) if a]
    
    feats = {}
    
    # V1: Total flip count (original)
    if len(answer_steps) >= 2:
        flips = sum(1 for i in range(1, len(answer_steps)) 
                    if answer_steps[i][1] != answer_steps[i-1][1])
        feats['flip_count'] = float(flips)
    
    # V2: First flip position (as fraction of trace)
    if len(answer_steps) >= 2:
        first_flip = None
        for i in range(1, len(answer_steps)):
            if answer_steps[i][1] != answer_steps[i-1][1]:
                first_flip = answer_steps[i][0]
                break
        if first_flip is not None:
            feats['first_flip_frac'] = first_flip / max(1, n_steps - 1)
        else:
            feats['first_flip_frac'] = 1.0  # Never flips = "late" (no signal)
    
    # V3: Late-only flips (last 50% of steps)
    if len(answer_steps) >= 2:
        mid = n_steps // 2
        late_steps = [(i, s) for i, s in answer_steps if i >= mid]
        if len(late_steps) >= 2:
            late_flips = sum(1 for j in range(1, len(late_steps))
                           if late_steps[j][1] != late_steps[j-1][1])
            feats['late_flips'] = float(late_flips)
        else:
            feats['late_flips'] = 0.0
    
    # V4: Max distinct answers at any step
    valid_distinct = [d for d in per_step_n_distinct if d is not None]
    if valid_distinct:
        feats['max_distinct'] = float(max(valid_distinct))
    
    # V5: Min majority fraction (worst agreement step)
    valid_maj = [m for m in per_step_majority_frac if m is not None]
    if valid_maj:
        feats['min_majority'] = 1.0 - min(valid_maj)  # Invert: higher = more uncertain
    
    # V6: Answer diversity slope (are answers becoming MORE diverse?)
    if len(valid_distinct) >= 3:
        x = np.arange(len(valid_distinct))
        slope = np.polyfit(x, valid_distinct, 1)[0]
        feats['diversity_slope'] = float(slope)
    
    # V7: "Greedy vs samples" disagreement
    # Does the model's final answer agree with ANY sample answer at the last step?
    model_answer = trace.get('model_answer_extracted', '').lower().strip()
    if model_answer and answer_steps:
        last_answers = per_step_answers[answer_steps[-1][0]]
        model_norm = re.sub(r'[^a-z0-9 ]', '', model_answer)[:30].strip()
        agrees = any(model_norm in a or a in model_norm for a in last_answers)
        feats['greedy_vs_samples_disagree'] = 0.0 if agrees else 1.0
    
    # V8: Consecutive flip (two flips in a row = oscillation)
    if len(answer_steps) >= 3:
        consec = 0
        for i in range(2, len(answer_steps)):
            if (answer_steps[i][1] != answer_steps[i-1][1] and 
                answer_steps[i-1][1] != answer_steps[i-2][1]):
                consec += 1
        feats['consecutive_flips'] = float(consec)
    
    # V9: "Same answer but rephrased" — count steps where answers overlap but aren't identical
    if len(answer_steps) >= 2:
        partial_matches = 0
        for i in range(1, len(answer_steps)):
            curr = answer_steps[i][1]
            prev = answer_steps[i-1][1]
            if curr != prev:  # Not identical
                # Check if any word overlap
                curr_words = set(' '.join(curr).split())
                prev_words = set(' '.join(prev).split())
                if curr_words & prev_words:  # Some overlap
                    partial_matches += 1
        feats['partial_flips'] = float(partial_matches)
    
    # V10: Flip rate normalized by answer-bearing steps
    if len(answer_steps) >= 2:
        flips = feats.get('flip_count', 0)
        feats['flip_rate'] = flips / max(1, len(answer_steps) - 1)
    
    return feats

# Extract for all traces
for t in all_traces:
    t['af'] = extract_answer_features(t['c'])

# ===================================================================
# Test all variants
# ===================================================================
print("=" * 80)
print("ANS_FLIP FAMILY: 10 variants tested")
print("=" * 80)

variants = [
    ('flip_count', 'V1: total flip count'),
    ('first_flip_frac', 'V2: first flip position (inv)'),
    ('late_flips', 'V3: late-half flips only'),
    ('max_distinct', 'V4: max distinct ans at any step'),
    ('min_majority', 'V5: worst agreement (1-min_maj)'),
    ('diversity_slope', 'V6: diversity slope (rising=bad)'),
    ('greedy_vs_samples_disagree', 'V7: greedy≠samples at end'),
    ('consecutive_flips', 'V8: oscillation (consec flips)'),
    ('partial_flips', 'V9: partial overlap flips'),
    ('flip_rate', 'V10: flip rate (normalized)'),
]

print(f"\n{'variant':<38} {'AUROC_inc':>10} {'AUROC_help':>11} {'pop':>5}")
for key, label in variants:
    valid = [t for t in all_traces if t['af'].get(key) is not None]
    pos_inc = [t['af'][key] for t in valid if is_inc(t)]
    neg_inc = [t['af'][key] for t in valid if not is_inc(t)]
    pos_help = [t['af'][key] for t in valid if is_help(t)]
    neg_help = [t['af'][key] for t in valid if not is_help(t)]
    a_inc = auroc(pos_inc, neg_inc)
    a_help = auroc(pos_help, neg_help)
    flag = " <-- BEST" if (a_help > 0.65 or a_inc > 0.7) else ""
    flag2 = " ***" if (a_inc > 0.6 or a_help > 0.6) else ""
    print(f"  {label:<36} {a_inc:>10.3f} {a_help:>11.3f} {len(valid):>5}{flag or flag2}")

# ===================================================================
# Per-cell breakdown of best variants
# ===================================================================
print("\n\n" + "=" * 80)
print("PER-CELL: Best ans_flip variants + baselines")
print("=" * 80)

cells = defaultdict(list)
for t in all_traces:
    cells[t['cell']].append(t)

best_variants = ['flip_count', 'late_flips', 'min_majority', 'flip_rate', 'greedy_vs_samples_disagree']

# vs incorrect
print(f"\n  AUROC vs INCORRECT:")
print(f"  {'cell':<12}", end="")
for v in best_variants:
    print(f" {v[:8]:>9}", end="")
print(f" {'DRAGIN':>9} {'tokens':>9}")
for cell_name in sorted(cells.keys()):
    rows = cells[cell_name]
    n_inc = sum(1 for t in rows if is_inc(t))
    if n_inc < 3: continue
    print(f"  {cell_name:<12}", end="")
    for key in best_variants:
        valid = [t for t in rows if t['af'].get(key) is not None]
        pos = [t['af'][key] for t in valid if is_inc(t)]
        neg = [t['af'][key] for t in valid if not is_inc(t)]
        a = auroc(pos, neg)
        print(f" {a:>9.3f}" if a==a else f" {'n/a':>9}", end="")
    # Baselines
    dragin_vals = [max([float(v) for v in step_vals(t['c'], 'dragin_rind_max') if v is not None]) 
                   for t in rows if step_vals(t['c'], 'dragin_rind_max')]
    tok_vals = [float(t['c'].get('tokens_generated', 0)) for t in rows]
    d_p = [v for v, t in zip(dragin_vals, rows) if is_inc(t)]
    d_n = [v for v, t in zip(dragin_vals, rows) if not is_inc(t)]
    t_p = [v for v, t in zip(tok_vals, rows) if is_inc(t)]
    t_n = [v for v, t in zip(tok_vals, rows) if not is_inc(t)]
    print(f" {auroc(d_p, d_n):>9.3f} {auroc(t_p, t_n):>9.3f}")

# vs helpful
print(f"\n  AUROC vs HELPFUL (lift+1):")
print(f"  {'cell':<12}", end="")
for v in best_variants:
    print(f" {v[:8]:>9}", end="")
print(f" {'DRAGIN':>9}")
for cell_name in sorted(cells.keys()):
    rows = cells[cell_name]
    n_help = sum(1 for t in rows if is_help(t))
    if n_help < 2: continue
    print(f"  {cell_name:<12}", end="")
    for key in best_variants:
        valid = [t for t in rows if t['af'].get(key) is not None]
        pos = [t['af'][key] for t in valid if is_help(t)]
        neg = [t['af'][key] for t in valid if not is_help(t)]
        a = auroc(pos, neg)
        print(f" {a:>9.3f}" if a==a else f" {'n/a':>9}", end="")
    dragin_vals = []
    for t in rows:
        vs = [float(v) for v in step_vals(t['c'], 'dragin_rind_max') if v is not None]
        dragin_vals.append(max(vs) if vs else None)
    d_p = [v for v, t in zip(dragin_vals, rows) if is_help(t) and v is not None]
    d_n = [v for v, t in zip(dragin_vals, rows) if not is_help(t) and v is not None]
    print(f" {auroc(d_p, d_n):>9.3f}")

# ===================================================================
# KEY INSIGHT: What about the ORACLE trace's answer flips?
# If retrieval helps, does the oracle trace show FEWER flips?
# ===================================================================
print("\n\n" + "=" * 80)
print("ORACLE COMPARISON: Do helped traces show fewer flips after retrieval?")
print("=" * 80)

for t in all_traces:
    t['o_af'] = extract_answer_features(t['o'])

helped = [t for t in all_traces if is_help(t)]
print(f"\n  Helped traces (n={len(helped)}):")
c_flips = [t['af'].get('flip_count', 0) for t in helped if t['af'].get('flip_count') is not None]
o_flips = [t['o_af'].get('flip_count', 0) for t in helped if t['o_af'].get('flip_count') is not None]
c_distinct = [t['af'].get('max_distinct', 0) for t in helped if t['af'].get('max_distinct') is not None]
o_distinct = [t['o_af'].get('max_distinct', 0) for t in helped if t['o_af'].get('max_distinct') is not None]

if c_flips and o_flips:
    print(f"    Closed flips: mean={np.mean(c_flips):.2f}, Oracle flips: mean={np.mean(o_flips):.2f}")
    print(f"    Closed max_distinct: mean={np.mean(c_distinct):.2f}, Oracle: mean={np.mean(o_distinct):.2f}")
    
not_helped_wrong = [t for t in all_traces if t['lift'] == 0 and t['c_em'] == 0]
print(f"\n  Not-helped incorrect (n={len(not_helped_wrong)}):")
c_flips2 = [t['af'].get('flip_count', 0) for t in not_helped_wrong if t['af'].get('flip_count') is not None]
o_flips2 = [t['o_af'].get('flip_count', 0) for t in not_helped_wrong if t['o_af'].get('flip_count') is not None]
if c_flips2 and o_flips2:
    print(f"    Closed flips: mean={np.mean(c_flips2):.2f}, Oracle flips: mean={np.mean(o_flips2):.2f}")

# ===================================================================
# BONUS: Combine with DRAGIN — the "DRAGIN fires late AND samples disagree"
# ===================================================================
print("\n\n" + "=" * 80)
print("COMBINED: DRAGIN-late AND ans_flip (fire when BOTH are high)")
print("=" * 80)

for t in all_traces:
    dragin_vals = [float(v) for v in step_vals(t['c'], 'dragin_rind_max') if v is not None]
    if not dragin_vals or not t['af'].get('flip_count'):
        t['combined_late_flip'] = None
        continue
    n = len(dragin_vals)
    # Position of DRAGIN peak (1.0 = last step)
    peak_pos = np.argmax(dragin_vals) / max(1, n-1)
    # Normalized flip
    flip_rate = t['af'].get('flip_rate', 0)
    t['combined_late_flip'] = peak_pos * flip_rate * max(dragin_vals)

# Also: per-step joint trigger (step has BOTH high DRAGIN AND flip)
def joint_step_trigger(trace):
    """Fire at first step where DRAGIN > median AND answer flipped from previous."""
    steps = trace.get('steps', [])
    if not steps: return None
    
    dragin_vals = [s.get('dragin_rind_max') for s in steps]
    valid_dragin = [float(v) for v in dragin_vals if v is not None]
    if not valid_dragin: return None
    median_d = np.median(valid_dragin)
    
    prev_answers = set()
    for i, s in enumerate(steps):
        samples = s.get('samples', [])
        curr_answers = set()
        for samp in samples:
            text = samp.get('text', '')
            m = ANSWER_RE.search(text)
            if m:
                ans = re.sub(r'[^a-z0-9 ]', '', m.group(1).strip().lower())[:30].strip()
                if ans: curr_answers.add(ans)
        
        d = dragin_vals[i]
        if d is not None and float(d) > median_d and prev_answers and curr_answers and curr_answers != prev_answers:
            return float(i) / max(1, len(steps) - 1)
        
        if curr_answers:
            prev_answers = curr_answers
    
    return None  # Never fires

for t in all_traces:
    t['joint_trigger_pos'] = joint_step_trigger(t['c'])
    # "Did joint trigger fire at all?" (binary)
    t['joint_fires'] = 1.0 if t['joint_trigger_pos'] is not None else 0.0

combined_signals = [
    ('combined_late_flip', lambda t: t.get('combined_late_flip')),
    ('joint_fires', lambda t: t.get('joint_fires')),
]

print(f"\n{'signal':<30} {'AUROC_inc':>10} {'AUROC_help':>11} {'pop':>5}")
for name, fn in combined_signals:
    valid = [t for t in all_traces if fn(t) is not None]
    pos_inc = [fn(t) for t in valid if is_inc(t)]
    neg_inc = [fn(t) for t in valid if not is_inc(t)]
    pos_help = [fn(t) for t in valid if is_help(t)]
    neg_help = [fn(t) for t in valid if not is_help(t)]
    a_inc = auroc(pos_inc, neg_inc)
    a_help = auroc(pos_help, neg_help)
    print(f"  {name:<28} {a_inc:>10.3f} {a_help:>11.3f} {len(valid):>5}")

# Fire rate of joint trigger
fires = sum(1 for t in all_traces if t.get('joint_fires', 0) > 0)
print(f"\n  Joint trigger fires on {fires}/{len(all_traces)} traces ({100*fires/len(all_traces):.0f}%)")
# Of those that fire: what fraction are incorrect / helped?
fired = [t for t in all_traces if t.get('joint_fires', 0) > 0]
if fired:
    inc_in_fired = sum(1 for t in fired if is_inc(t))
    help_in_fired = sum(1 for t in fired if is_help(t))
    print(f"  Of fired: {inc_in_fired}/{len(fired)} incorrect ({100*inc_in_fired/len(fired):.0f}%), "
          f"{help_in_fired}/{len(fired)} helped ({100*help_in_fired/len(fired):.0f}%)")
    # Precision vs base rates
    base_inc = sum(1 for t in all_traces if is_inc(t)) / len(all_traces)
    base_help = sum(1 for t in all_traces if is_help(t)) / len(all_traces)
    print(f"  Base rates: incorrect={100*base_inc:.0f}%, helped={100*base_help:.0f}%")
    print(f"  Precision lift: incorrect={100*(inc_in_fired/len(fired) - base_inc):.0f}pp, "
          f"helped={100*(help_in_fired/len(fired) - base_help):.0f}pp")
