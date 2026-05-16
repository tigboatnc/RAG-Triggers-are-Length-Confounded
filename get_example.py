import json

oracle_file = "runs/v2_headline_qwen2.5-7b-instruct/E1_oracle_hotpot.jsonl"
closed_file = "runs/v2_headline_qwen2.5-7b-instruct/D1_closed_hotpot.jsonl"

oracle_data = {}
with open(oracle_file, "r") as f:
    for line in f:
        data = json.loads(line)
        oracle_data[data["id"]] = data

with open(closed_file, "r") as f:
    for line in f:
        data = json.loads(line)
        qid = data["id"]
        if qid in oracle_data:
            oracle_em = oracle_data[qid].get("em", 0)
            closed_em = data.get("em", 0)
            if oracle_em == 1 and closed_em == 0:
                out = {"qid": qid, "closed": data, "oracle": oracle_data[qid]}
                with open("example_trace.json", "w") as out_f:
                    json.dump(out, out_f, indent=2)
                print(f"Found match: {qid}")
                break
