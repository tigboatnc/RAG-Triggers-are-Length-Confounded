#!/usr/bin/env bash
# Local analysis runner. Runs OFF the compute box, on whatever machine has
# the analysis env. Reads the v2 outputs from runs/v2_exp1_phi-4-mini-instruct/
# and produces:
#
#   1. The pre-registered u_cons hypothesis check (analysis/iter5_preregistration.py).
#      This is the PRIMARY analysis for the paper.
#
#   2. The legacy v1-gate analyzer (scripts/04o_exp_analyzer.py) for honest
#      comparison. Reports per-gate Δfrac, paired bootstrap, McNemar.
#
# Both are pure Python stats — no GPU, no transformers required. Conda env
# `gzn` (or any stdlib python 3.10+) suffices.
#
# Usage:
#
#     conda activate gzn        # or any stdlib-only python ≥3.10
#     bash scripts/run_local_analysis.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

RUN_DIR="${RUN_DIR:-runs/v2_exp1_phi-4-mini-instruct}"
ANALYSIS_OUT="${ANALYSIS_OUT:-analysis/final_n500.out}"

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "[$(ts)] $*"; }

log "=== local analysis on ${RUN_DIR} ==="

# Sanity: required inputs present?
need=(
    "${RUN_DIR}/D1_closed_hotpot.jsonl"
    "${RUN_DIR}/D2_closed_wiki.jsonl"
    "${RUN_DIR}/C2_calibration.json"
)
optional_oracle=(
    "${RUN_DIR}/E1_oracle_hotpot.jsonl"
    "${RUN_DIR}/E2_oracle_wiki.jsonl"
)
for f in "${need[@]}"; do
    [[ -f "$f" ]] || { log "FATAL: missing required input $f"; exit 1; }
done
for f in "${need[@]}"; do log "  ✓ $f  ($(wc -l < "$f") rows)"; done

ORACLE_READY=1
for f in "${optional_oracle[@]}"; do
    if [[ ! -f "$f" ]] || [[ $(wc -l < "$f") -lt 50 ]]; then
        log "  ⚠ $f  not ready (oracle phase E may still be running) — legacy F-analysis will be skipped"
        ORACLE_READY=0
    else
        log "  ✓ $f  ($(wc -l < "$f") rows)"
    fi
done

# ---------------------------------------------------------------------
# Primary: u_cons hypothesis (the paper headline)
# ---------------------------------------------------------------------
log ""
log "=== PRIMARY: u_cons step-boundary consistency hypothesis (Experiment 2) ==="
python analysis/iter5_preregistration.py 2>&1 | tee "${ANALYSIS_OUT}"
log "Wrote ${ANALYSIS_OUT}"

# ---------------------------------------------------------------------
# Secondary: legacy v1-gate analysis (only if oracle ready)
# ---------------------------------------------------------------------
if [[ "${ORACLE_READY}" -eq 1 ]]; then
    log ""
    log "=== SECONDARY: legacy v1-gate analyzer (DRAGIN/ETC/SELFDOUBT/random) ==="

    log "F.1: HotpotQA-500 in-domain"
    python scripts/04o_exp_analyzer.py \
        --closed-book "${RUN_DIR}/D1_closed_hotpot.jsonl" \
        --oracle      "${RUN_DIR}/E1_oracle_hotpot.jsonl" \
        --thresholds  "${RUN_DIR}/C2_calibration.json" \
        --out         "${RUN_DIR}/F1_analysis_hotpot.json" \
        --label       "phi_hotpot500_indomain" \
        --min-steps   3 \
        -B 10000 \
        2>&1 | tee "analysis/F1_legacy.out"

    log "F.2: 2WikiMHQA-500 OOD"
    python scripts/04o_exp_analyzer.py \
        --closed-book "${RUN_DIR}/D2_closed_wiki.jsonl" \
        --oracle      "${RUN_DIR}/E2_oracle_wiki.jsonl" \
        --thresholds  "${RUN_DIR}/C2_calibration.json" \
        --out         "${RUN_DIR}/F2_analysis_wiki.json" \
        --label       "phi_wiki500_ood" \
        --min-steps   3 \
        -B 10000 \
        2>&1 | tee "analysis/F2_legacy.out"
fi

log ""
log "=== done ==="
log "Primary results:   ${ANALYSIS_OUT}"
if [[ "${ORACLE_READY}" -eq 1 ]]; then
    log "Legacy results:    ${RUN_DIR}/F1_analysis_hotpot.json  ${RUN_DIR}/F2_analysis_wiki.json"
    log "                    analysis/F1_legacy.out  analysis/F2_legacy.out"
else
    log "Oracle (E) not yet complete on this pull; legacy F-analysis skipped."
    log "Re-run this script once E1/E2 are >50 rows each."
fi
