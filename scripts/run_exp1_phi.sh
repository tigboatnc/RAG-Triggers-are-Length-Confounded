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
rows = [json.loads(l) for l in open("${RUN_DIR}/B_smoke_hotpot.jsonl") if l.strip()]
nsteps = [r.get("n_steps", 0) for r in rows]
# trace schema uses model_answer_extracted, not "answer" (which was a v1-era alias)
ans_lens = [len((r.get("model_answer_extracted") or "")) for r in rows]
print(f"[B-check] n_traces={len(rows)}  n_steps={nsteps}  ans_len={ans_lens}")
ok = sum(1 for r in rows if r.get("n_steps", 0) >= 3 and (r.get("model_answer_extracted") or "").strip())
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
step "D.1 + D.2: closed-book decodes in PARALLEL (HotpotQA-500 + 2WikiMHQA-500)"
log "D.1 + D.2 share GPU; Phi-4-mini @ bf16 ~8 GB/proc; expect ~15-30 GB total VRAM"
(
  python scripts/04f_run_qa_sota.py \
    --model "${MODEL_NAME}" \
    --model-dir "${MODEL_DIR}" \
    --dataset "${HOTPOT_TEST_SLICE}" \
    --layer-mode mid_stack \
    --output "${RUN_DIR}/D1_closed_hotpot.jsonl" \
    2>&1 | tee "${LOG_DIR}/D1_decode_hotpot.log"
) &
PID_D1=$!
log "D.1 launched as pid=${PID_D1}"
(
  python scripts/04f_run_qa_sota.py \
    --model "${MODEL_NAME}" \
    --model-dir "${MODEL_DIR}" \
    --dataset "${WIKI_TEST_SLICE}" \
    --layer-mode mid_stack \
    --output "${RUN_DIR}/D2_closed_wiki.jsonl" \
    2>&1 | tee "${LOG_DIR}/D2_decode_wiki.log"
) &
PID_D2=$!
log "D.2 launched as pid=${PID_D2}"
log "waiting for D.1 (pid=${PID_D1}) and D.2 (pid=${PID_D2})..."
D_FAIL=0
wait "${PID_D1}" || { log "D.1 FAILED (pid=${PID_D1}, exit=$?)"; D_FAIL=1; }
wait "${PID_D2}" || { log "D.2 FAILED (pid=${PID_D2}, exit=$?)"; D_FAIL=1; }
[[ ${D_FAIL} -eq 0 ]] || { log "ABORT after D parallel block: one or both decodes failed"; exit 1; }
log "D.1 + D.2 done"

# ---------------------------------------------------------------------
# Phase E — Oracle-retrieve decode on both test slices
# ---------------------------------------------------------------------
step "E.1 + E.2: oracle-retrieve decodes in PARALLEL"
(
  python scripts/04l_oracle_retrieve_decode.py \
    --model "${MODEL_NAME}" \
    --dataset "${HOTPOT_TEST_SLICE}" \
    --output "${RUN_DIR}/E1_oracle_hotpot.jsonl" \
    2>&1 | tee "${LOG_DIR}/E1_oracle_hotpot.log"
) &
PID_E1=$!
log "E.1 launched as pid=${PID_E1}"
(
  python scripts/04l_oracle_retrieve_decode.py \
    --model "${MODEL_NAME}" \
    --dataset "${WIKI_TEST_SLICE}" \
    --output "${RUN_DIR}/E2_oracle_wiki.jsonl" \
    2>&1 | tee "${LOG_DIR}/E2_oracle_wiki.log"
) &
PID_E2=$!
log "E.2 launched as pid=${PID_E2}"
log "waiting for E.1 (pid=${PID_E1}) and E.2 (pid=${PID_E2})..."
E_FAIL=0
wait "${PID_E1}" || { log "E.1 FAILED (pid=${PID_E1}, exit=$?)"; E_FAIL=1; }
wait "${PID_E2}" || { log "E.2 FAILED (pid=${PID_E2}, exit=$?)"; E_FAIL=1; }
[[ ${E_FAIL} -eq 0 ]] || { log "ABORT after E parallel block: one or both oracles failed"; exit 1; }
log "E.1 + E.2 done"

# ---------------------------------------------------------------------
# Phases F (analysis) and G (summary) intentionally REMOVED from the
# compute-box wrapper. This box runs LLM compute only (A → B → C → D → E).
#
# Analysis happens on the analysis box via `scripts/run_local_analysis.sh`
# after the compute box's outputs are pulled (`git pull`). The legacy v1-gate
# analyzer (`04o_exp_analyzer.py`) and the new u_cons hypothesis analyzer
# (`analysis/iter5_preregistration.py`) both run there; both are pure Python
# stats and do not need a GPU or transformers.
# ---------------------------------------------------------------------

step "compute-side run finished. Outputs:"
log "  D1 (closed-book Hot):   $(wc -l < "${RUN_DIR}/D1_closed_hotpot.jsonl")   rows"
log "  D2 (closed-book 2Wiki): $(wc -l < "${RUN_DIR}/D2_closed_wiki.jsonl")    rows"
log "  E1 (oracle Hot):        $(wc -l < "${RUN_DIR}/E1_oracle_hotpot.jsonl") rows"
log "  E2 (oracle 2Wiki):      $(wc -l < "${RUN_DIR}/E2_oracle_wiki.jsonl")   rows"
log "Commit + push the JSONLs; analysis happens on the analysis box."
log "exp1 (compute portion) done"
