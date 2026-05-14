import json, statistics
from collections import Counter

E1 = "runs/v2_headline_gemma-4-26b-a4b-it/E1_oracle_hotpot.jsonl"
D1 = "runs/v2_headline_gemma-4-26b-a4b-it/D1_closed_hotpot.jsonl"

rows = [json.loads(l) for l in open(E1)]
print(f"n_rows={len(rows)}")
print(f"top-level keys (row0): {sorted(rows[0].keys())}")

ems = [r.get("em", 0) for r in rows]
f1s = [r.get("f1", 0.0) for r in rows]
toks = [r.get("tokens_generated", 0) for r in rows]
print(f"EM rate: {sum(ems)}/{len(ems)} = {sum(ems)/len(ems):.3f}")
print(f"meanF1: {sum(f1s)/len(f1s):.3f}")
ts = sorted(toks)
print(f"tokens: min={min(toks)} p10={ts[len(ts)//10]} median={statistics.median(toks):.0f} mean={sum(toks)/len(toks):.1f} p90={ts[-len(ts)//10]} max={max(toks)}")

ids = [r.get("id") for r in rows]
print(f"unique ids: {len(set(ids))} / {len(ids)}")

d1_rows = [json.loads(l) for l in open(D1)]
d1_ids = [r.get("id") for r in d1_rows]
d1_set = set(d1_ids)
overlap = set(ids) & d1_set
print(f"D1 ids: {len(d1_ids)}, E1 ∩ D1: {len(overlap)}, E1-only: {len(set(ids)-d1_set)}, D1-only: {len(d1_set-set(ids))}")
print(f"ID order matches D1: {ids == d1_ids}")

empty_ans = sum(1 for r in rows if not (r.get("model_answer_extracted") or "").strip())
print(f"empty extracted answers: {empty_ans}/{len(rows)}")
print(f"models seen: {set(r.get('model') for r in rows)}")
print(f"splits: {set(r.get('split') for r in rows)}")
print(f"types: {sorted(set(r.get('type') for r in rows))}")

fr = [r.get("early_stop_reason") for r in rows]
print(f"early_stop_reason: {Counter(fr).most_common()}")

# Compute lift_em on paired
d1_by_id = {r.get("id"): r for r in d1_rows}
helped = 0
hurt = 0
same = 0
both_em1 = 0
both_em0 = 0
for r in rows:
    qid = r.get("id")
    if qid not in d1_by_id:
        continue
    e = r.get("em", 0)
    c = d1_by_id[qid].get("em", 0)
    if c == 0 and e == 1: helped += 1
    elif c == 1 and e == 0: hurt += 1
    elif c == 0 and e == 0: both_em0 += 1
    elif c == 1 and e == 1: both_em1 += 1
print(f"\nPAIRED LIFT (closed→oracle):")
print(f"  helped (closed=0 → oracle=1): {helped}")
print(f"  hurt   (closed=1 → oracle=0): {hurt}")
print(f"  both correct: {both_em1}")
print(f"  both wrong  : {both_em0}")
print(f"  base rate helped: {helped}/200 = {helped/200:.3f}")
print(f"  closed EM: {sum(c.get('em',0) for c in d1_rows)}/200")
print(f"  oracle EM: {sum(ems)}/200")

# Sample rows
for i in (0, 100, 199):
    r = rows[i]
    print(f"\nrow {i}: id={r.get('id')} em={r.get('em')} f1={r.get('f1')} tok={r.get('tokens_generated')} type={r.get('type')}")
    print(f"  gold: {str(r.get('gold_answer'))[:80]!r}")
    print(f"  ans:  {(r.get('model_answer_extracted') or '')[:80]!r}")
