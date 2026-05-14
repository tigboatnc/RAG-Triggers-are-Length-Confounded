import json
SLICE_500 = [json.loads(l) for l in open("data/hotpotqa_distractor_500_v2test.jsonl")]
slice_ids = [r.get("id") or r.get("_id") for r in SLICE_500]
slice_first200 = set(slice_ids[:200])
slice_500_set = set(slice_ids)

for arch in ("phi-4-mini-instruct", "qwen2.5-7b-instruct", "gemma-4-26b-a4b-it"):
    for phase, suffix in (("D1", "D1_closed_hotpot.jsonl"), ("E1", "E1_oracle_hotpot.jsonl")):
        path = f"runs/v2_headline_{arch}/{suffix}"
        try:
            ids = [json.loads(l)["id"] for l in open(path)]
        except FileNotFoundError:
            print(f"{arch:30s} {phase}: (missing)")
            continue
        s = set(ids)
        match_first200 = s == slice_first200
        in_500 = len(s & slice_500_set)
        outside_500 = len(s - slice_500_set)
        print(f"{arch:30s} {phase}: n={len(ids):3d}  matches_first200={match_first200}  in_canonical_500={in_500}  outside_500={outside_500}")
