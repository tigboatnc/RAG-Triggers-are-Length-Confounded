#!/usr/bin/env bash
# Phase 4.1 sweep: 200-question HotpotQA-distractor x 3 architectures, mid_stack CRV.
# Estimated wall: Phi ~35 min, Gemma ~2.0 h, Qwen ~2.1 h, plus loads = ~4.7 h total.

set -uo pipefail

source /opt/miniforge3/etc/profile.d/conda.sh
conda activate recollection
cd "$(dirname "$0")/.."

MODELS=(
  "phi-4-mini-instruct"
  "gemma-4-26b-a4b-it"
  "qwen3.6-35b-a3b-instruct"
)
LAYER="mid_stack"

mkdir -p runs
START_ALL=$(date +%s)
echo "[sweep] start $(date -Iseconds)" | tee runs/phase4_sweep.log

for MODEL in "${MODELS[@]}"; do
  TAG="${MODEL}_${LAYER}"
  LOG="runs/phase4_qa_${TAG}.log"
  T0=$(date +%s)
  echo "" | tee -a runs/phase4_sweep.log
  echo "[sweep] === ${TAG} === start $(date -Iseconds)" | tee -a runs/phase4_sweep.log
  python scripts/04_run_qa.py \
    --model "$MODEL" \
    --layer-mode "$LAYER" \
    --max-new-tokens 512 \
    --progress-every 20 \
    > "$LOG" 2>&1
  RC=$?
  T1=$(date +%s)
  ELAPSED=$((T1 - T0))
  if [[ $RC -eq 0 ]]; then
    echo "[sweep] ${TAG} OK in ${ELAPSED}s -> $LOG" | tee -a runs/phase4_sweep.log
  else
    echo "[sweep] ${TAG} FAILED rc=$RC after ${ELAPSED}s -> $LOG" | tee -a runs/phase4_sweep.log
  fi
done

END_ALL=$(date +%s)
echo "" | tee -a runs/phase4_sweep.log
echo "[sweep] all done in $((END_ALL - START_ALL))s" | tee -a runs/phase4_sweep.log
