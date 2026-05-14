import json

E1 = "runs/v2_headline_gemma-4-26b-a4b-it/E1_oracle_hotpot.jsonl"
D1 = "runs/v2_headline_gemma-4-26b-a4b-it/D1_closed_hotpot.jsonl"
SLICE = "data/hotpotqa_distractor_500_v2test.jsonl"

e1_ids = [json.loads(l)["id"] for l in open(E1)]
d1_ids = [json.loads(l)["id"] for l in open(D1)]
slice_rows = [json.loads(l) for l in open(SLICE)]
slice_ids = [r.get("id") or r.get("_id") for r in slice_rows]

print(f"slice len = {len(slice_ids)}")
print(f"slice first-200 == D1 ids (ordered)? {slice_ids[:200] == d1_ids}")
print(f"slice first-200 == E1 ids (ordered)? {slice_ids[:200] == e1_ids}")
print(f"slice first-200 set == E1 set? {set(slice_ids[:200]) == set(e1_ids)}")

# Compare which slice positions each came from
d1_set = set(d1_ids)
e1_set = set(e1_ids)

print(f"\nE1-only ids (in E1, not in D1) — slice positions:")
for qid in sorted(e1_set - d1_set):
    pos = slice_ids.index(qid) if qid in slice_ids else -1
    print(f"  {qid}  slice_pos={pos}")
print(f"\nD1-only ids (in D1, not in E1) — slice positions:")
for qid in sorted(d1_set - e1_set):
    pos = slice_ids.index(qid) if qid in slice_ids else -1
    print(f"  {qid}  slice_pos={pos}")

# Check first 10 ids of E1 vs slice
print(f"\nE1 first 10 ids: {e1_ids[:10]}")
print(f"slice first 10 ids: {slice_ids[:10]}")
print(f"D1 first 10 ids: {d1_ids[:10]}")
