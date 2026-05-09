#!/usr/bin/env bash
# Phase 3 full sweep: 3 models x 2 layer modes x 3 prompts x 5000 tokens.
# Sequential per (model, layer_mode) session so each model loads only twice.
# Per-session log lands in runs/phase3_<model>_<layermode>.log

set -uo pipefail

source /opt/miniforge3/etc/profile.d/conda.sh
conda activate recollection

cd "$(dirname "$0")/.."

SESSIONS=(
  "phi-4-mini-instruct       last_layer"
  "phi-4-mini-instruct       mid_stack"
  "gemma-4-26b-a4b-it        last_layer"
  "gemma-4-26b-a4b-it        mid_stack"
  "qwen3.6-35b-a3b-instruct  last_layer"
  "qwen3.6-35b-a3b-instruct  mid_stack"
)

mkdir -p runs

START_ALL=$(date +%s)
echo "[sweep] start $(date -Iseconds)" | tee runs/phase3_sweep.log

for entry in "${SESSIONS[@]}"; do
  read -r MODEL LAYER <<<"$entry"
  TAG="${MODEL}_${LAYER}"
  LOG="runs/phase3_${TAG}.log"
  T0=$(date +%s)
  echo "" | tee -a runs/phase3_sweep.log
  echo "[sweep] === ${TAG} === start $(date -Iseconds)" | tee -a runs/phase3_sweep.log
  python scripts/03_baselines.py \
    --model "$MODEL" \
    --layer-mode "$LAYER" \
    --max-new-tokens 5000 \
    --prompt-class all \
    --progress-every 1000 \
    > "$LOG" 2>&1
  RC=$?
  T1=$(date +%s)
  ELAPSED=$((T1 - T0))
  if [[ $RC -eq 0 ]]; then
    echo "[sweep] ${TAG} OK in ${ELAPSED}s -> $LOG" | tee -a runs/phase3_sweep.log
  else
    echo "[sweep] ${TAG} FAILED rc=$RC after ${ELAPSED}s -> $LOG" | tee -a runs/phase3_sweep.log
  fi
done

END_ALL=$(date +%s)
echo "" | tee -a runs/phase3_sweep.log
echo "[sweep] all done in $((END_ALL - START_ALL))s" | tee -a runs/phase3_sweep.log
