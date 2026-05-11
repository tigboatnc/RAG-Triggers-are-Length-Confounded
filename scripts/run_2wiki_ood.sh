#!/usr/bin/env bash
# Phase 4 / #21 — single-command compute-box wrapper for the 2WikiMultiHopQA
# OOD transfer test on Qwen3.6 (Narrative A).
#
# Chains Steps 4 -> 5 -> 6 of the handoff runbook in _AGENT.md #21
# §compute-box-handoff. Aborts at the smoke gate if the 5-question smoke
# fails any of the four pre-registered criteria. Reads no flags;
# everything is pinned to #21 defaults.
#
# Prerequisites (handled separately; this script does not run them):
#   - Step 1: git pull
#   - Step 2: Qwen3.6 weights under models/qwen3.6-35b-a3b-instruct/
#   - Step 3: data/2wikimultihop_distractor_200.jsonl + manifest produced
#             by `python scripts/04_dataset_2wiki.py`
#
# Usage:
#   conda activate gzn
#   bash scripts/run_2wiki_ood.sh
#
# Logs:
#   runs/04f_qwen_2wiki_smoke.log     (Step 4 driver output)
#   runs/04i_qwen_2wiki_smoke.log     (smoke validator)
#   runs/04f_qwen_2wiki_ood.log       (Step 5 full sweep)
#   runs/04i_qwen_2wiki_ood.log       (Step 6 OOD analyzer)

set -euo pipefail

MODEL="qwen3.6-35b-a3b-instruct"
DATASET="data/2wikimultihop_distractor_200.jsonl"
MANIFEST="data/2wikimultihop_distractor_200.manifest.json"
SMOKE_SUFFIX="_2wiki_smoke"
OOD_SUFFIX="_2wiki_ood"

SMOKE_TRACES="runs/phase4_qa_${MODEL}_mid_stack${SMOKE_SUFFIX}.jsonl"
OOD_TRACES="runs/phase4_qa_${MODEL}_mid_stack${OOD_SUFFIX}.jsonl"
BOOTSTRAP_JSON="runs/phase4_bootstrap_dragin.json"
OUT_JSON="runs/phase4_ood_transfer_2wiki.json"

echo "[..] 2Wiki OOD transfer runbook (Qwen3.6 only, #21 defaults)"
echo "     model     = $MODEL"
echo "     dataset   = $DATASET"
echo "     bootstrap = $BOOTSTRAP_JSON (frozen Qwen config from #20)"
echo

if [ ! -f "$DATASET" ]; then
    echo "[FAIL] missing $DATASET. Run scripts/04_dataset_2wiki.py first (Step 3)."
    exit 2
fi
if [ ! -f "$BOOTSTRAP_JSON" ]; then
    echo "[FAIL] missing $BOOTSTRAP_JSON. The #20 bootstrap must be on disk."
    exit 2
fi
if [ ! -d "models/$MODEL" ]; then
    echo "[FAIL] missing models/$MODEL. Run Step 2 (Qwen download) first."
    exit 2
fi

# --- Step 4: 5-question smoke ---
echo "[..] Step 4: 5-question smoke"
python scripts/04f_run_qa_sota.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --manifest "$MANIFEST" \
    --split all --limit 5 \
    --out-suffix "$SMOKE_SUFFIX" \
    2>&1 | tee "runs/04f_qwen_2wiki_smoke.log"

# --- Smoke validator (gate) ---
echo
echo "[..] Step 4b: smoke validator"
if ! python scripts/04i_2wiki_smoke_validator.py --traces "$SMOKE_TRACES" \
        2>&1 | tee "runs/04i_qwen_2wiki_smoke.log"; then
    echo
    echo "[FAIL] smoke validator returned non-zero. Aborting before the 200-q sweep."
    echo "       Copy runs/04i_qwen_2wiki_smoke.log into _AGENT.md under #21 §smoke-fail."
    exit 1
fi

# --- Step 5: full 200-q sweep ---
echo
echo "[..] Step 5: full 200-q sweep (~1.5-2 h)"
python scripts/04f_run_qa_sota.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --manifest "$MANIFEST" \
    --split all \
    --layer-mode mid_stack \
    --out-suffix "$OOD_SUFFIX" \
    2>&1 | tee "runs/04f_qwen_2wiki_ood.log"

# --- Step 6: OOD analyzer ---
echo
echo "[..] Step 6: OOD transfer analysis"
python scripts/04i_ood_transfer.py \
    --bootstrap-json "$BOOTSTRAP_JSON" \
    --traces "$OOD_TRACES" \
    --bootstrap 10000 \
    --out "$OUT_JSON" \
    2>&1 | tee "runs/04i_qwen_2wiki_ood.log"

echo
echo "[ok] done. Headline numbers in $OUT_JSON; full log in runs/04i_qwen_2wiki_ood.log."
echo "     Next: append the verdict + numbers to _AGENT.md as entry #22, referencing #21."
