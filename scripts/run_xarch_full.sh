#!/usr/bin/env bash
# Cross-architecture full-pipeline replication of v2 Phi Experiment 2 on
# Gemma-4-26B-A4B-it and Qwen3.6-35B-A3B-Instruct.
#
# Goal: produce closed-book + oracle traces (n=200 per dataset) on Gemma and
# Qwen so the headline u_cons.max / mean_token_entropy.argmax_pos AUROCs and
# the lift-conditional deployment table can be reported as a 3-arch grid.
#
# Phase sequence (per arch, serial because Gemma=50GB and Qwen=68GB don't
# co-fit on an 80GB A800 with each other or with parallel Phi):
#   A. Slices (reuse existing 500-row v2test slices, --limit 200 at decode).
#   B. Smoke (5 Q on HotpotQA; hard-fail < 3/5 usable).
#   C. Pilot (50 Q on HotpotQA; sanity-check tokens_generated + base EM).
#   D1. Full closed-book HotpotQA  n=200.
#   D2. Full closed-book 2Wiki     n=200.
#   E1. Full oracle HotpotQA       n=200.
#   E2. Full oracle 2Wiki          n=200.
#
# Idempotent: re-running with the same n skips already-decoded rows
# (`04f_run_qa_sota.py` and `04l_oracle_retrieve_decode.py` both resume from
# the output JSONL's line count).
#
# Wall-clock estimate on a single A800 PCIe 80 GB:
#   Gemma  ~13 tok/s × 250 tok × (5+50+200+200+200+200) = ~655 questions
#          + K=3 boundary overhead (~10%) + oracle re-decode (no K=3, cheaper)
#          → closed-book ~3.5-4 h, oracle ~1.5-2 h, total ~5-6 h
#   Qwen   ~10-12 tok/s, slightly larger active params
#          → closed-book ~4-5 h, oracle ~2-2.5 h, total ~6-7 h
#   Pipeline total (serial): ~11-13 h on one card.
#
# Usage (compute box):
#   bash scripts/run_xarch_full.sh                    # both archs, hot+2wiki
#   ARCHS="gemma-4-26b-a4b-it" bash scripts/run_xarch_full.sh  # gemma only
#   N_FULL=100 bash scripts/run_xarch_full.sh         # smaller cohort
#
# Logs and outputs:
#   runs/v2_xarch_<model>/                            (per-arch run dir)
#     ├── B_smoke_hotpot.jsonl                        (Phase B)
#     ├── C_pilot_hotpot.jsonl                        (Phase C)
#     ├── D1_closed_hotpot.jsonl                      (Phase D, n=N_FULL)
#     ├── D2_closed_wiki.jsonl
#     ├── E1_oracle_hotpot.jsonl                      (Phase E)
#     ├── E2_oracle_wiki.jsonl
#     └── logs/                                       (one log per phase)

set -euo pipefail
shopt -s inherit_errexit

if [[ -x /venv/recollection/bin/python ]]; then
  export PATH="/venv/recollection/bin:${PATH}"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
ARCHS="${ARCHS:-gemma-4-26b-a4b-it qwen3.6-35b-a3b-instruct}"
N_SMOKE="${N_SMOKE:-5}"
N_PILOT="${N_PILOT:-50}"
N_FULL="${N_FULL:-200}"

HOTPOT_TEST_SLICE="${HOTPOT_TEST_SLICE:-data/hotpotqa_distractor_500_v2test.jsonl}"
WIKI_TEST_SLICE="${WIKI_TEST_SLICE:-data/2wikimultihop_distractor_500_v2test.jsonl}"

TS() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

# ---------------------------------------------------------------------
# Pre-flight (slices + GPU + python)
# ---------------------------------------------------------------------
PREFLIGHT_LOG="runs/v2_xarch_preflight.log"
mkdir -p runs
log_pre() { echo "[$(TS)] $*" | tee -a "${PREFLIGHT_LOG}"; }

log_pre "=== xarch full pipeline starting ==="
log_pre "git rev: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
log_pre "python:  $(python -V 2>&1)"
log_pre "nvidia-smi: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1)"
log_pre "ARCHS=${ARCHS}"
log_pre "N_SMOKE=${N_SMOKE}  N_PILOT=${N_PILOT}  N_FULL=${N_FULL}"
log_pre "HOTPOT_TEST_SLICE=${HOTPOT_TEST_SLICE}"
log_pre "WIKI_TEST_SLICE=${WIKI_TEST_SLICE}"

if [[ ! -f "${HOTPOT_TEST_SLICE}" ]]; then
  log_pre "FATAL: ${HOTPOT_TEST_SLICE} missing. Build it via scripts/run_exp1_phi.sh phase A, or:"
  log_pre "  python scripts/04_dataset.py --n 500 --seed 20260601 --split-seed 20260601 \\"
  log_pre "      --exclude-ids-from data/hotpotqa_distractor_200.jsonl --all-test \\"
  log_pre "      --name hotpotqa_distractor_500_v2test"
  exit 2
fi
if [[ ! -f "${WIKI_TEST_SLICE}" ]]; then
  log_pre "FATAL: ${WIKI_TEST_SLICE} missing. Build it via:"
  log_pre "  python scripts/04_dataset_2wiki.py --n 500 --seed 20260601 \\"
  log_pre "      --name 2wikimultihop_distractor_500_v2test"
  exit 2
fi

# ---------------------------------------------------------------------
# Per-arch pipeline
# ---------------------------------------------------------------------
for MODEL_NAME in ${ARCHS}; do
  MODEL_DIR="models/${MODEL_NAME}"
  RUN_DIR="runs/v2_xarch_${MODEL_NAME}"
  LOG_DIR="${RUN_DIR}/logs"
  mkdir -p "${RUN_DIR}" "${LOG_DIR}"

  log() { echo "[$(TS)] [${MODEL_NAME}] $*" | tee -a "${LOG_DIR}/00_master.log"; }
  step() { log "==== $* ===="; }
  trap 'log "ABORT @ line $LINENO (last cmd: $BASH_COMMAND)"' ERR

  step "PRE: ${MODEL_NAME}"
  log "MODEL_DIR=${MODEL_DIR}"
  log "RUN_DIR=${RUN_DIR}"

  if [[ ! -d "${MODEL_DIR}" ]] || ! ls "${MODEL_DIR}"/*.safetensors >/dev/null 2>&1; then
    log "MODEL WEIGHTS MISSING — run: bash scripts/download_models.sh phase1.4"
    log "Skipping ${MODEL_NAME}."
    continue
  fi

  # -------- Phase B: smoke (5 questions on HotpotQA) --------
  SMOKE_OUT="${RUN_DIR}/B_smoke_hotpot.jsonl"
  if [[ -f "${SMOKE_OUT}" ]] && [[ $(wc -l < "${SMOKE_OUT}") -ge ${N_SMOKE} ]]; then
    log "B: smoke already done (n=$(wc -l < "${SMOKE_OUT}")), skipping"
  else
    step "B: smoke decode (n=${N_SMOKE} on HotpotQA)"
    python scripts/04f_run_qa_sota.py \
      --model "${MODEL_NAME}" \
      --model-dir "${MODEL_DIR}" \
      --dataset "${HOTPOT_TEST_SLICE}" \
      --layer-mode mid_stack \
      --limit "${N_SMOKE}" \
      --output "${SMOKE_OUT}" \
      2>&1 | tee "${LOG_DIR}/B_smoke.log"

    # Sanity check — must have >= 3/5 usable (n_steps>=2 AND non-empty answer).
    # n_steps>=2 (not >=3) is the right bar here because pivot to Idea-1 is
    # only meaningful on multi-step traces; an arch that produces 5/5 1-step
    # traces still passes smoke but won't carry Idea-1 signal — flag, not fail.
    python - <<PYEOF | tee -a "${LOG_DIR}/B_smoke.log"
import json
rows = [json.loads(l) for l in open("${SMOKE_OUT}") if l.strip()]
nsteps = [r.get("n_steps", 0) for r in rows]
ans_lens = [len((r.get("model_answer_extracted") or "")) for r in rows]
toks = [r.get("tokens_generated", 0) for r in rows]
ok = sum(1 for r in rows if r.get("n_steps", 0) >= 2 and (r.get("model_answer_extracted") or "").strip())
print(f"[B-check] n_traces={len(rows)} n_steps={nsteps} ans_len={ans_lens} tok_gen={toks}")
print(f"[B-check] n_usable_multistep={ok}/{len(rows)}")
n_short = sum(1 for r in rows if r.get("n_steps", 0) == 1)
if n_short > len(rows) // 2:
    print(f"[B-WARN] {n_short}/{len(rows)} traces are 1-step; Idea-1 spike-position will be undefined for those.")
if ok == 0:
    raise SystemExit("smoke FAILED: 0 usable multi-step traces")
if ok < 2:
    raise SystemExit(f"smoke FAILED: {ok} usable < 2; investigate before pilot")
print("[B-check] PASS")
PYEOF
  fi

  # -------- Phase C: pilot (50 questions on HotpotQA) --------
  PILOT_OUT="${RUN_DIR}/C_pilot_hotpot.jsonl"
  if [[ -f "${PILOT_OUT}" ]] && [[ $(wc -l < "${PILOT_OUT}") -ge ${N_PILOT} ]]; then
    log "C: pilot already done (n=$(wc -l < "${PILOT_OUT}")), skipping"
  else
    step "C: pilot decode (n=${N_PILOT} on HotpotQA)"
    python scripts/04f_run_qa_sota.py \
      --model "${MODEL_NAME}" \
      --model-dir "${MODEL_DIR}" \
      --dataset "${HOTPOT_TEST_SLICE}" \
      --layer-mode mid_stack \
      --limit "${N_PILOT}" \
      --output "${PILOT_OUT}" \
      2>&1 | tee "${LOG_DIR}/C_pilot.log"

    python - <<PYEOF | tee -a "${LOG_DIR}/C_pilot.log"
import json
rows = [json.loads(l) for l in open("${PILOT_OUT}") if l.strip()]
em = [int(bool(r.get("label_correct_em", r.get("em", False)))) for r in rows]
nsteps = [r.get("n_steps", 0) for r in rows]
n_multi = sum(1 for n in nsteps if n >= 2)
print(f"[C-pilot] n={len(rows)} EM_rate={sum(em)/max(1,len(em)):.3f} n_multistep={n_multi}/{len(rows)}")
print(f"[C-pilot] n_steps distribution: min={min(nsteps)} median={sorted(nsteps)[len(nsteps)//2]} max={max(nsteps)}")
PYEOF
  fi

  # -------- Phase D1+D2: full closed-book on both datasets --------
  for PAIR in "D1_closed_hotpot:${HOTPOT_TEST_SLICE}" "D2_closed_wiki:${WIKI_TEST_SLICE}"; do
    TAG="${PAIR%%:*}"; SLICE="${PAIR##*:}"
    OUT="${RUN_DIR}/${TAG}.jsonl"
    if [[ -f "${OUT}" ]] && [[ $(wc -l < "${OUT}") -ge ${N_FULL} ]]; then
      log "D: ${TAG} already done (n=$(wc -l < "${OUT}")), skipping"
      continue
    fi
    step "D: ${TAG} closed-book decode (n=${N_FULL} from ${SLICE})"
    python scripts/04f_run_qa_sota.py \
      --model "${MODEL_NAME}" \
      --model-dir "${MODEL_DIR}" \
      --dataset "${SLICE}" \
      --layer-mode mid_stack \
      --limit "${N_FULL}" \
      --output "${OUT}" \
      2>&1 | tee "${LOG_DIR}/${TAG}.log"
  done

  # -------- Phase E1+E2: oracle re-decode (no K=3 needed; cheaper) --------
  for PAIR in "E1_oracle_hotpot:${HOTPOT_TEST_SLICE}" "E2_oracle_wiki:${WIKI_TEST_SLICE}"; do
    TAG="${PAIR%%:*}"; SLICE="${PAIR##*:}"
    OUT="${RUN_DIR}/${TAG}.jsonl"
    if [[ -f "${OUT}" ]] && [[ $(wc -l < "${OUT}") -ge ${N_FULL} ]]; then
      log "E: ${TAG} already done (n=$(wc -l < "${OUT}")), skipping"
      continue
    fi
    step "E: ${TAG} oracle decode (n=${N_FULL} from ${SLICE})"
    python scripts/04l_oracle_retrieve_decode.py \
      --model "${MODEL_NAME}" \
      --model-dir "${MODEL_DIR}" \
      --dataset "${SLICE}" \
      --limit "${N_FULL}" \
      --out "${OUT}" \
      2>&1 | tee "${LOG_DIR}/${TAG}.log"
  done

  step "${MODEL_NAME} done. Outputs:"
  for f in D1_closed_hotpot D2_closed_wiki E1_oracle_hotpot E2_oracle_wiki; do
    if [[ -f "${RUN_DIR}/${f}.jsonl" ]]; then
      log "  ${f}: $(wc -l < "${RUN_DIR}/${f}.jsonl") rows"
    fi
  done

  trap - ERR
done

log_pre "=== xarch full pipeline complete ==="
log_pre "Commit + push runs/v2_xarch_*/D{1,2}_closed_*.jsonl and runs/v2_xarch_*/E{1,2}_oracle_*.jsonl"
log_pre "Then on analysis box: python analysis/xarch_summary.py"
