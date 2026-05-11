import json
from collections import defaultdict

counts = defaultdict(lambda: {"TP": 0, "FP": 0, "TN": 0, "FN": 0})

with open("runs/phase4_qa_gemma-4-26b-a4b-it_mid_stack.jsonl") as f:
    for line in f:
        row = json.loads(line)
        if "error" in row:
            continue
        
        is_wrong = row["f1"] < 0.5
        
        # We need to know if the gate fired AT ANY STEP during the trace
        # gate_decisions is a list of {step_idx, token_idx, gate_name, fired, ...}
        gate_fired = defaultdict(bool)
        for dec in row.get("gate_decisions", []):
            if dec["fired"]:
                gate_fired[dec["gate_name"]] = True
        
        # Evaluate each gate
        for gate in ["latent", "realm_sim", "grip_sim", "random"]:
            fired = gate_fired[gate]
            if fired and is_wrong:
                counts[gate]["TP"] += 1
            elif fired and not is_wrong:
                counts[gate]["FP"] += 1
            elif not fired and not is_wrong:
                counts[gate]["TN"] += 1
            elif not fired and is_wrong:
                counts[gate]["FN"] += 1

print("=== Gemma 4 26B Mid-Stack Gate Evaluation (Trace-Level) ===")
print("Ground truth: Trace is WRONG (F1 < 0.5)")
for gate, c in counts.items():
    tp, fp, tn, fn = c["TP"], c["FP"], c["TN"], c["FN"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    print(f"\nGate: {gate}")
    print(f"  TP: {tp:<3}  FP: {fp:<3}")
    print(f"  FN: {fn:<3}  TN: {tn:<3}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  Gate-F1:   {f1:.3f}")

