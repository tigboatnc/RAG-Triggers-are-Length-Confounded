import json
import numpy as np

def calculate_auroc(y_true, y_score):
    # Simple AUROC implementation
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = np.array(y_score)[desc_score_indices]
    y_true = np.array(y_true)[desc_score_indices]
    
    tps = np.cumsum(y_true)
    fps = np.cumsum(1 - y_true)
    
    tpr = tps / tps[-1] if tps[-1] > 0 else tps
    fpr = fps / fps[-1] if fps[-1] > 0 else fps
    
    # prepend 0,0
    tpr = np.r_[0, tpr]
    fpr = np.r_[0, fpr]
    
    return np.trapz(tpr, fpr)

y_true = []  # 1 if wrong (F1 < 0.5)
max_drift = []
max_variance = []
max_realm = []
max_grip = []

with open("runs/phase4_qa_gemma-4-26b-a4b-it_mid_stack.jsonl") as f:
    for line in f:
        row = json.loads(line)
        if "error" in row: continue
        
        is_wrong = 1 if row["f1"] < 0.5 else 0
        y_true.append(is_wrong)
        
        # Aggregate max signal over all steps in the trace
        drifts = [s["drift_cosine"] for s in row["steps"]]
        variances = [s["variance_l2_window"] for s in row["steps"]]
        entropies = [s["boundary_token_entropy"] for s in row["steps"]]
        mean_entropies = [s["mean_token_entropy"] for s in row["steps"]]
        
        max_drift.append(max(drifts) if drifts else 0)
        max_variance.append(max(variances) if variances else 0)
        max_realm.append(max(mean_entropies) if mean_entropies else 0)
        max_grip.append(max(entropies) if entropies else 0)

print("=== Gemma 4 26B Mid-Stack Signal AUROC ===")
print("(Predicting if the trace is WRONG)")
print(f"Latent (Max Drift):    {calculate_auroc(y_true, max_drift):.3f}")
print(f"Latent (Max Variance): {calculate_auroc(y_true, max_variance):.3f}")
print(f"ReaLM Sim (Mean H):    {calculate_auroc(y_true, max_realm):.3f}")
print(f"GRIP Sim (Boundary H): {calculate_auroc(y_true, max_grip):.3f}")

