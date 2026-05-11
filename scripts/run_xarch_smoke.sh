#!/usr/bin/env bash
# Cross-arch smoke test for u_cons step-boundary consistency signal.
#
# Goal: at n=50, test whether u_cons.max AUROC > 0.55 generalizes to Gemma
# and Qwen. v1 traces don't have u_cons populated (older capture code); this
# script re-decodes 50 questions per arch with the v2 capture pipeline that
# produces u_cons.
#
# Decision rule (per _agent_v2.md):
#   - AUROC ≥ 0.55 on at least one of {gemma, qwen}: go for 200-row run
#   - AUROC < 0.50 on both: paper stays single-arch (Phi only), document negative
#
# Time estimate (A800):
#   - Gemma  ~12 tok/s × ~250 tok × 50 questions + K=3 boundary overhead ~10%
#            → ~25-30 min wall
#   - Qwen   ~10 tok/s × ~250 tok × 50 questions + K=3 overhead
#            → ~30-35 min wall
#   - Total  ~60-65 min on a single A800 (serial)
#
# Idempotent: re-running with the same SLICE skips already-decoded rows
# (04f_run_qa_sota.py uses output JSONL line-count for resume).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

SLICE="${SLICE:-data/hotpotqa_distractor_smoke_50.jsonl}"
OUT_DIR="${OUT_DIR:-runs/v2_smoke_xarch}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "${LOG_DIR}/00_master.log"; }

log "=== cross-arch u_cons smoke test ==="
log "REPO=${REPO_ROOT}"
log "SLICE=${SLICE}"
log "OUT_DIR=${OUT_DIR}"
log "git rev: $(git rev-parse --short HEAD)"
log "nvidia-smi: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)"

# Phase A — build 50-q slice (HotpotQA, seed disjoint from v1/v2 cohorts)
if [[ ! -f "${SLICE}" ]]; then
    log "A: building ${SLICE}"
    python scripts/04_dataset.py \
        --n 50 --seed 20260512 --split-seed 20260512 \
        --exclude-ids-from data/hotpotqa_distractor_200.jsonl \
        --exclude-ids-from data/hotpotqa_distractor_500_v2test.jsonl \
        --all-test --name hotpotqa_distractor_smoke_50 \
        2>&1 | tee "${LOG_DIR}/A_slice.log"
else
    log "A: ${SLICE} already exists, skipping"
fi

# Phase B — decode on each arch with consistency sampler K=3
# Skip any arch whose model weights are not on disk; warn loudly so the user
# knows to download them (scripts/download_models.sh phase1.4).
AVAILABLE_ARCHS=()
for ARCH_PAIR in \
    "phi-4-mini-instruct models/phi-4-mini-instruct" \
    "gemma-4-26b-a4b-it models/gemma-4-26b-a4b-it" \
    "qwen3.6-35b-a3b-instruct models/qwen3.6-35b-a3b-instruct"
do
    MODEL_NAME=$(echo "${ARCH_PAIR}" | cut -d' ' -f1)
    MODEL_DIR=$(echo "${ARCH_PAIR}" | cut -d' ' -f2)
    OUT="${OUT_DIR}/${MODEL_NAME}_smoke.jsonl"
    LOG="${LOG_DIR}/B_${MODEL_NAME}.log"

    # Availability check: a model is considered "available" if its directory
    # exists AND contains at least one *.safetensors file (cheap proxy for
    # "weights are on disk", avoids picking up empty stub dirs).
    if [[ ! -d "${MODEL_DIR}" ]] || ! ls "${MODEL_DIR}"/*.safetensors >/dev/null 2>&1; then
        log "B: ${MODEL_NAME} NOT available (missing ${MODEL_DIR} or no safetensors). To fetch: bash scripts/download_models.sh phase1.4. Skipping."
        continue
    fi
    AVAILABLE_ARCHS+=("${MODEL_NAME}")

    if [[ -f "${OUT}" ]] && [[ $(wc -l < "${OUT}") -ge 50 ]]; then
        log "B: ${MODEL_NAME} already done (n=$(wc -l < "${OUT}")), skipping decode"
        continue
    fi
    log "B: decoding ${MODEL_NAME} → ${OUT}  (VRAM: Phi ~8GB, Gemma ~50GB, Qwen ~70GB)"
    python scripts/04f_run_qa_sota.py \
        --model "${MODEL_NAME}" \
        --model-dir "${MODEL_DIR}" \
        --dataset "${SLICE}" \
        --layer-mode mid_stack \
        --output "${OUT}" \
        2>&1 | tee "${LOG}"
done

if [[ ${#AVAILABLE_ARCHS[@]} -eq 0 ]]; then
    log "FATAL: no models available on this box. Run: bash scripts/download_models.sh phase1.4"
    exit 2
fi
log "B: available archs this run: ${AVAILABLE_ARCHS[*]}"

# Phase C — analyze u_cons.max AUROC for whichever archs decoded successfully.
log "C: analyzing"
ANALYZE_ARGS=()
[[ -f "${OUT_DIR}/phi-4-mini-instruct_smoke.jsonl" ]] && ANALYZE_ARGS+=(--phi-control "${OUT_DIR}/phi-4-mini-instruct_smoke.jsonl")
[[ -f "${OUT_DIR}/gemma-4-26b-a4b-it_smoke.jsonl" ]] && ANALYZE_ARGS+=(--gemma "${OUT_DIR}/gemma-4-26b-a4b-it_smoke.jsonl")
[[ -f "${OUT_DIR}/qwen3.6-35b-a3b-instruct_smoke.jsonl" ]] && ANALYZE_ARGS+=(--qwen "${OUT_DIR}/qwen3.6-35b-a3b-instruct_smoke.jsonl")
python analysis/smoke_xarch_analyze.py "${ANALYZE_ARGS[@]}" 2>&1 | tee "${LOG_DIR}/C_analysis.log"

log "=== smoke test complete ==="
log "See ${LOG_DIR}/C_analysis.log for the go/no-go decision."
