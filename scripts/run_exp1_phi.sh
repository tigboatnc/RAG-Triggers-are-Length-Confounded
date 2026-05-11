#!/usr/bin/env bash
# Experiment 1 — Phi-4-mini on HotpotQA-500 (in-domain test) + 2WikiMHQA-500 (zero-shot OOD).
#
# Calibrates on the v1 200-q HotpotQA cohort (ALL three splits used as one calibration pool).
# Decodes on fresh HotpotQA-500 disjoint from v1, and on 2WikiMHQA-500 OOD.
# Oracle-retrieves on both test slices. Analyses everything with rich logging.
#
# Run from project root on the compute box. ~estimated wall clock on a single
# H100 with Phi-4-mini-instruct: 90-120 minutes total (decode dominates).

set -euo pipefail
shopt -s inherit_errexit

# Defensive: ensure the project conda env is on PATH even from non-interactive
# shells that don't source ~/.bashrc. `/venv/recollection` is a conda env (no
# `activate` script), so we just prepend its bin dir.
if [[ -x /venv/recollection/bin/python ]]; then
  export PATH="/venv/recollection/bin:${PATH}"
fi

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
MODEL_NAME="${MODEL_NAME:-phi-4-mini-instruct}"
MODEL_DIR="${MODEL_DIR:-models/${MODEL_NAME}}"

V1_SLICE="${V1_SLICE:-data/hotpotqa_distractor_200.jsonl}"          # v1 calibration cohort
HOTPOT_TEST_SLICE="data/hotpotqa_distractor_500_v2test.jsonl"
WIKI_TEST_SLICE="data/2wikimultihop_distractor_500_v2test.jsonl"

RUN_DIR="${RUN_DIR:-runs/v2_exp1_${MODEL_NAME}}"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${RUN_DIR}" "${LOG_DIR}"

TS() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log()  { echo "[$(TS)] $*" | tee -a "${LOG_DIR}/00_master.log"; }
step() { log "==== STEP $* ===="; }

trap 'log "ABORT @ line $LINENO (last cmd: $BASH_COMMAND)"' ERR

log "exp1 starting"
log "MODEL_NAME=${MODEL_NAME}"
log "MODEL_DIR=${MODEL_DIR}"
log "V1_SLICE=${V1_SLICE}"
log "HOTPOT_TEST_SLICE=${HOTPOT_TEST_SLICE}"
log "WIKI_TEST_SLICE=${WIKI_TEST_SLICE}"
log "RUN_DIR=${RUN_DIR}"
log "git rev: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
log "python:  $(python -V 2>&1)"
log "nvidia-smi: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1)"

# ---------------------------------------------------------------------
# Phase A — Build fresh slices (disjoint from v1)
# ---------------------------------------------------------------------
step "A.1: HotpotQA-500 test slice (disjoint from v1, all rows tagged split=test)"
if [[ ! -f "${HOTPOT_TEST_SLICE}" ]]; then
  python scripts/04_dataset.py \
    --n 500 \
    --seed 20260601 \
    --split-seed 20260601 \
    --exclude-ids-from "${V1_SLICE}" \
    --all-test \
    --name "hotpotqa_distractor_500_v2test" \
    2>&1 | tee "${LOG_DIR}/A1_hotpot_slice.log"
else
  log "A.1: slice already exists, skipping"
fi
ls -la "${HOTPOT_TEST_SLICE}" 2>&1 | tee -a "${LOG_DIR}/A1_hotpot_slice.log"

step "A.2: 2WikiMHQA-500 test slice (OOD)"
if [[ ! -f "${WIKI_TEST_SLICE}" ]]; then
  python scripts/04_dataset_2wiki.py \
    --n 500 \
    --seed 20260601 \
    --name "2wikimultihop_distractor_500_v2test" \
    2>&1 | tee "${LOG_DIR}/A2_wiki_slice.log"
else
  log "A.2: slice already exists, skipping"
fi
ls -la "${WIKI_TEST_SLICE}" 2>&1 | tee -a "${LOG_DIR}/A2_wiki_slice.log"

# ---------------------------------------------------------------------
# Phase B — Smoke test (5 questions) before the long decode
# ---------------------------------------------------------------------
step "B: smoke decode (Phi on 5 HotpotQA v2-test questions)"
python scripts/04f_run_qa_sota.py \
  --model "${MODEL_NAME}" \
  --model-dir "${MODEL_DIR}" \
  --dataset "${HOTPOT_TEST_SLICE}" \
  --layer-mode mid_stack \
  --limit 5 \
  --output "${RUN_DIR}/B_smoke_hotpot.jsonl" \
  2>&1 | tee "${LOG_DIR}/B_smoke.log"

# Sanity check: at least 4/5 should have n_steps >= 3 and a non-empty answer.
python - <<EOF | tee -a "${LOG_DIR}/B_smoke.log"
import json
rows = [json.loads(l) for l in open("${RUN_DIR}/B_smoke.jsonl".replace("B_smoke.jsonl","B_smoke_hotpot.jsonl")) if l.strip()]
nsteps = [r.get("n_steps", 0) for r in rows]
ans_lens = [len(r.get("answer", "") or "") for r in rows]
print(f"[B-check] n_traces={len(rows)}  n_steps={nsteps}  ans_len={ans_lens}")
ok = sum(1 for r in rows if r.get("n_steps", 0) >= 3 and (r.get("answer") or "").strip())
print(f"[B-check] n_usable={ok}/{len(rows)}")
if ok < 3:
    raise SystemExit("smoke test FAILED: < 3/5 usable traces; investigate before launching long decode")
print("[B-check] PASS")
EOF

# ---------------------------------------------------------------------
# Phase C — Calibration on v1 cohort (uses pre-existing v1 traces if available;
#           otherwise decodes v1 200 first).
# ---------------------------------------------------------------------
V1_TRACES="runs/phase4_qa_${MODEL_NAME}_mid_stack_dragin.jsonl"
step "C.1: ensure v1 calibration traces exist at ${V1_TRACES}"
if [[ ! -f "${V1_TRACES}" ]]; then
  log "C.1: v1 trace file missing; re-decoding on v1 200-q slice"
  python scripts/04f_run_qa_sota.py \
    --model "${MODEL_NAME}" \
    --model-dir "${MODEL_DIR}" \
    --dataset "${V1_SLICE}" \
    --layer-mode mid_stack \
    --output "${V1_TRACES}" \
    2>&1 | tee "${LOG_DIR}/C1_v1_decode.log"
else
  log "C.1: v1 trace file present, skipping"
fi

step "C.2: calibrate (5-fold CV on all 200 v1 rows, trace-level iso-0.30)"
python scripts/04p_calibrate.py \
  --closed-book "${V1_TRACES}" \
  --splits calibration validation test \
  --target-trace-fire-rate 0.30 \
  --kfold 5 \
  --min-steps 3 \
  --out "${RUN_DIR}/C2_calibration.json" \
  2>&1 | tee "${LOG_DIR}/C2_calibration.log"

# ---------------------------------------------------------------------
# Phase D — Closed-book decode on both test slices
# ---------------------------------------------------------------------
step "D.1: closed-book decode on HotpotQA-500 v2-test"
python scripts/04f_run_qa_sota.py \
  --model "${MODEL_NAME}" \
  --model-dir "${MODEL_DIR}" \
  --dataset "${HOTPOT_TEST_SLICE}" \
  --layer-mode mid_stack \
  --output "${RUN_DIR}/D1_closed_hotpot.jsonl" \
  2>&1 | tee "${LOG_DIR}/D1_decode_hotpot.log"

step "D.2: closed-book decode on 2WikiMHQA-500 v2-test (OOD)"
python scripts/04f_run_qa_sota.py \
  --model "${MODEL_NAME}" \
  --model-dir "${MODEL_DIR}" \
  --dataset "${WIKI_TEST_SLICE}" \
  --layer-mode mid_stack \
  --output "${RUN_DIR}/D2_closed_wiki.jsonl" \
  2>&1 | tee "${LOG_DIR}/D2_decode_wiki.log"

# ---------------------------------------------------------------------
# Phase E — Oracle-retrieve decode on both test slices
# ---------------------------------------------------------------------
step "E.1: oracle-retrieve decode on HotpotQA-500 v2-test"
python scripts/04l_oracle_retrieve_decode.py \
  --model "${MODEL_NAME}" \
  --dataset "${HOTPOT_TEST_SLICE}" \
  --output "${RUN_DIR}/E1_oracle_hotpot.jsonl" \
  2>&1 | tee "${LOG_DIR}/E1_oracle_hotpot.log"

step "E.2: oracle-retrieve decode on 2WikiMHQA-500 v2-test (OOD)"
python scripts/04l_oracle_retrieve_decode.py \
  --model "${MODEL_NAME}" \
  --dataset "${WIKI_TEST_SLICE}" \
  --output "${RUN_DIR}/E2_oracle_wiki.jsonl" \
  2>&1 | tee "${LOG_DIR}/E2_oracle_wiki.log"

# ---------------------------------------------------------------------
# Phase F — Analysis (frozen thresholds from Phase C)
# ---------------------------------------------------------------------
step "F.1: analyze HotpotQA-500 v2-test (in-domain)"
python scripts/04o_exp_analyzer.py \
  --closed-book "${RUN_DIR}/D1_closed_hotpot.jsonl" \
  --oracle      "${RUN_DIR}/E1_oracle_hotpot.jsonl" \
  --thresholds  "${RUN_DIR}/C2_calibration.json" \
  --out         "${RUN_DIR}/F1_analysis_hotpot.json" \
  --label       "phi_hotpot500_indomain" \
  --min-steps   3 \
  -B 10000 \
  2>&1 | tee "${LOG_DIR}/F1_analyze_hotpot.log"

step "F.2: analyze 2WikiMHQA-500 v2-test (zero-shot OOD, frozen thresholds)"
python scripts/04o_exp_analyzer.py \
  --closed-book "${RUN_DIR}/D2_closed_wiki.jsonl" \
  --oracle      "${RUN_DIR}/E2_oracle_wiki.jsonl" \
  --thresholds  "${RUN_DIR}/C2_calibration.json" \
  --out         "${RUN_DIR}/F2_analysis_wiki.json" \
  --label       "phi_wiki500_ood" \
  --min-steps   3 \
  -B 10000 \
  2>&1 | tee "${LOG_DIR}/F2_analyze_wiki.log"

# ---------------------------------------------------------------------
# Phase G — Wrap up
# ---------------------------------------------------------------------
step "G: summary"
log "All artifacts under ${RUN_DIR}"
ls -la "${RUN_DIR}"
log "Logs under ${LOG_DIR}"
ls -la "${LOG_DIR}"
log "exp1 done"
