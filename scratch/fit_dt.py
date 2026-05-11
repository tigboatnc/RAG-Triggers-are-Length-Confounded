import sys
import importlib
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier, export_text

sys.path.insert(0, "scripts")
mod = importlib.import_module("04k_method_variants")

calib, val, test = mod._load_split(Path("runs/phase4_qa_qwen3.6-35b-a3b-instruct_mid_stack_dragin.jsonl"), 3)
Xc, yc, _ = mod._extract_step_matrix(calib, mod.GEOM_FEATURES)

dt = DecisionTreeClassifier(max_depth=2, class_weight="balanced", random_state=20260511)
dt.fit(Xc, yc)

print(export_text(dt, feature_names=mod.GEOM_FEATURES))

Xv, yv, _ = mod._extract_step_matrix(val, mod.GEOM_FEATURES)
print("Val accuracy:", dt.score(Xv, yv))
