import json
with open("runs/phase4_method_variants.json") as f:
    d = json.load(f)
qwen = d["per_arch"]["qwen3.6-35b-a3b-instruct"]
for split in ["validation", "test"]:
    print(f"--- {split.upper()} ---")
    for gate, res in qwen[split]["per_gate"].items():
        if gate.startswith("latent"):
            print(f"{gate}: point={res.get('point')} ci=[{res.get('ci_lo')}, {res.get('ci_hi')}]")
    print("Margins vs DRAGIN:")
    for m, res in qwen[split]["margins"].items():
        if "dragin" in m:
            print(f"{m}: point={res.get('point')} ci=[{res.get('ci_lo')}, {res.get('ci_hi')}] P(>0)={res.get('p_gt_zero')}")
