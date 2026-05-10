#!/usr/bin/env bash
# Phase 4.1b sweep: 200-question HotpotQA-distractor x 3 architectures, mid_stack CRV,
# Option A protocol (no thinking, system role, per-arch decoding hygiene).
# See _AGENT.md 2026-05-10 #11 for protocol derivation.
#
# Always run scripts/04b_protocol_smoke.py first on a new compute box. The smoke
# verifies each model's chat-template render matches the live model card -- our
# training cutoff is well before these 2026 releases, so we cannot trust
# memorised defaults. Sweep refuses to start if smoke didn't pass recently.
#
# Estimated wall (mirrors the 4.1 sweep on A800; 1024 vs 512 max_new_tokens may
# stretch tail latencies modestly, but Option A's early-stop on Answer: should
# offset it): Phi ~35 min, Gemma ~2 h, Qwen ~2 h, total ~4.5 h.

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
SUFFIX="_v2"  # output lands at runs/phase4_qa_<model>_<layer>_v2.jsonl

mkdir -p runs
START_ALL=$(date +%s)
LOG_MAIN="runs/phase4_sweep_v2.log"
echo "[sweep-v2] start $(date -Iseconds)" | tee "$LOG_MAIN"

# Soft check: most recent smoke artifact within last 24h.
LATEST_SMOKE=$(ls -t runs/04b_protocol_smoke_*.json 2>/dev/null | head -1 || true)
if [[ -z "$LATEST_SMOKE" ]]; then
  echo "[sweep-v2] WARN: no protocol smoke artifact found; running smoke now" | tee -a "$LOG_MAIN"
  python scripts/04b_protocol_smoke.py 2>&1 | tee -a "$LOG_MAIN"
  RC=${PIPESTATUS[0]}
  if [[ $RC -ne 0 ]]; then
    echo "[sweep-v2] ABORT: protocol smoke failed (rc=$RC)" | tee -a "$LOG_MAIN"
    exit $RC
  fi
else
  echo "[sweep-v2] using existing smoke artifact: $LATEST_SMOKE" | tee -a "$LOG_MAIN"
fi

for MODEL in "${MODELS[@]}"; do
  TAG="${MODEL}_${LAYER}${SUFFIX}"
  LOG="runs/phase4_qa_${TAG}.log"
  T0=$(date +%s)
  echo "" | tee -a "$LOG_MAIN"
  echo "[sweep-v2] === ${TAG} === start $(date -Iseconds)" | tee -a "$LOG_MAIN"
  python scripts/04_run_qa.py \
    --model "$MODEL" \
    --layer-mode "$LAYER" \
    --max-new-tokens 1024 \
    --out-suffix "$SUFFIX" \
    --progress-every 20 \
    > "$LOG" 2>&1
  RC=$?
  T1=$(date +%s)
  ELAPSED=$((T1 - T0))
  if [[ $RC -eq 0 ]]; then
    echo "[sweep-v2] ${TAG} OK in ${ELAPSED}s -> $LOG" | tee -a "$LOG_MAIN"
  else
    echo "[sweep-v2] ${TAG} FAILED rc=$RC after ${ELAPSED}s -> $LOG" | tee -a "$LOG_MAIN"
  fi
done

END_ALL=$(date +%s)
echo "" | tee -a "$LOG_MAIN"
echo "[sweep-v2] all done in $((END_ALL - START_ALL))s" | tee -a "$LOG_MAIN"
