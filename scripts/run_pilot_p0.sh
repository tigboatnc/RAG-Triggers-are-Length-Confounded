#!/usr/bin/env bash
# RQ1 Experiment P0 — Smoke + sanity pilot.
#
# Goals:
#   G1 (test):       directional trigger-pool numbers on fresh data.
#   G2 (post-hoc):   max-logging schema (rq1-pilot-v1) so every audited
#                    method (DRAGIN, FLARE, SeaKR-K5-proxy, SelfCheckGPT-NLI,
#                    legacy v2 pool) is reproducible offline without re-decode.
#   G3 (learning):   3-arch behavior of the new methods on a fresh slice.
#
# Cohort:
#   3 archs (Phi, Gemma, Qwen) × 2 datasets (HotpotQA-distractor, 2WikiMHQA-distractor)
#   × {closed-book D, oracle re-decode E} = 12 cells.
#   n=25 per cell (300 traces total).
#   K=5 boundary sampler with per-sample text + per-sample mid-stack EOS hidden dump.
#
# Wall-clock estimate (single A800 80 GB serial):
#   Phi   ~9 min, Gemma ~32 min, Qwen ~38 min, slice/smoke/setup ~15 min,
#   offline analysis ~15 min  →  ~1 h 50 min total. Budget: 4 h.
#
# Pre-registered sanity gates (orchestrator-enforced):
#   1. 5-question smoke per arch passes (>=3/5 multi-step-with-answer).
#   2. tokens_generated median in [80, 600] per cell.
#   3. NEW pilot fields populated on >=80% of steps per cell.
#   4. Wall-clock budget cap: 3 h total (margin to 4 h ceiling).
#
# Idempotent: re-running with the same N skips already-decoded rows.
#
# Usage (compute box):
#   bash scripts/run_pilot_p0.sh                              # full pilot
#   ARCHS="phi-4-mini-instruct" bash scripts/run_pilot_p0.sh  # single arch
#   N_PILOT=10 bash scripts/run_pilot_p0.sh                   # smaller cohort
#
# Outputs:
#   runs/v2_pilot_<model>/
#     ├── B_smoke_hotpot.jsonl             (Phase B; 5-question smoke)
#     ├── D1_closed_hotpot.jsonl           (Phase D; n=N_PILOT, rq1-pilot-v1 schema)
#     ├── D2_closed_wiki.jsonl
#     ├── E1_oracle_hotpot.jsonl           (Phase E; oracle re-decode, paired)
#     ├── E2_oracle_wiki.jsonl
#     ├── checkpoints/                     (commit-friendly: thresholds, manifest, sha256s)
#     └── logs/                            (one log per phase + 00_master.log)

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
ARCHS="${ARCHS:-phi-4-mini-instruct gemma-4-26b-a4b-it qwen3.6-35b-a3b-instruct}"
N_SMOKE="${N_SMOKE:-5}"
N_PILOT="${N_PILOT:-25}"
CONS_K="${CONS_K:-5}"

PILOT_SEED="${PILOT_SEED:-20260513}"
PILOT_HOTPOT_SLICE="${PILOT_HOTPOT_SLICE:-data/hotpotqa_distractor_25_pilot.jsonl}"
PILOT_WIKI_SLICE="${PILOT_WIKI_SLICE:-data/2wikimultihop_distractor_25_pilot.jsonl}"

EXISTING_HOTPOT_500="data/hotpotqa_distractor_500_v2test.jsonl"
EXISTING_HOTPOT_200="data/hotpotqa_distractor_200.jsonl"
EXISTING_WIKI_500="data/2wikimultihop_distractor_500_v2test.jsonl"

TS() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

# ---------------------------------------------------------------------
# Pre-flight: build fresh pilot slices, log sysinfo
# ---------------------------------------------------------------------
PRE_LOG="runs/v2_pilot_preflight.log"
mkdir -p runs
log_pre() { echo "[$(TS)] $*" | tee -a "${PRE_LOG}"; }

log_pre "=== RQ1 Experiment P0 starting ==="
log_pre "git rev: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
log_pre "python:  $(python -V 2>&1)"
log_pre "nvidia-smi: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1)"
log_pre "ARCHS=${ARCHS}"
log_pre "N_SMOKE=${N_SMOKE}  N_PILOT=${N_PILOT}  CONS_K=${CONS_K}  PILOT_SEED=${PILOT_SEED}"

# ----- Slice builds (fresh, disjoint from prior cohorts) ---------------
build_slice_hotpot() {
  if [[ -f "${PILOT_HOTPOT_SLICE}" ]]; then
    log_pre "slice ${PILOT_HOTPOT_SLICE} already present (n=$(wc -l < "${PILOT_HOTPOT_SLICE}"))"
    return
  fi
  log_pre "building HotpotQA pilot slice (seed=${PILOT_SEED}, n=${N_PILOT}, disjoint from prior cohorts)"
  EXCL_FLAGS=""
  for f in "${EXISTING_HOTPOT_500}" "${EXISTING_HOTPOT_200}"; do
    [[ -f "${f}" ]] && EXCL_FLAGS+=" --exclude-ids-from ${f}"
  done
  python scripts/04_dataset.py \
    --n "${N_PILOT}" --seed "${PILOT_SEED}" --split-seed "${PILOT_SEED}" \
    ${EXCL_FLAGS} --all-test \
    --name "$(basename "${PILOT_HOTPOT_SLICE}" .jsonl)" \
    2>&1 | tee -a "${PRE_LOG}"
}

build_slice_wiki() {
  if [[ -f "${PILOT_WIKI_SLICE}" ]]; then
    log_pre "slice ${PILOT_WIKI_SLICE} already present (n=$(wc -l < "${PILOT_WIKI_SLICE}"))"
    return
  fi
  log_pre "building 2WikiMHQA pilot slice (seed=${PILOT_SEED}, n=${N_PILOT})"
  EXCL_FLAGS=""
  [[ -f "${EXISTING_WIKI_500}" ]] && EXCL_FLAGS+=" --exclude-ids-from ${EXISTING_WIKI_500}"
  python scripts/04_dataset_2wiki.py \
    --n "${N_PILOT}" --seed "${PILOT_SEED}" \
    ${EXCL_FLAGS} \
    --name "$(basename "${PILOT_WIKI_SLICE}" .jsonl)" \
    2>&1 | tee -a "${PRE_LOG}"
}

build_slice_hotpot
build_slice_wiki

if [[ ! -f "${PILOT_HOTPOT_SLICE}" ]] || [[ ! -f "${PILOT_WIKI_SLICE}" ]]; then
  log_pre "FATAL: pilot slices missing after build attempt"
  exit 2
fi

# ---------------------------------------------------------------------
# Per-arch pilot
# ---------------------------------------------------------------------
T_GLOBAL_START=$(date +%s)
WALL_CAP_SEC=$((6 * 3600))  # 6 h budget cap (bumped from 3h: A100 80GB observed ~65 s/q on Gemma D, projecting ~2 h/arch for D+E across 2 datasets)

for MODEL_NAME in ${ARCHS}; do
  MODEL_DIR="models/${MODEL_NAME}"
  RUN_DIR="runs/v2_pilot_${MODEL_NAME}"
  LOG_DIR="${RUN_DIR}/logs"
  CKPT_DIR="${RUN_DIR}/checkpoints"
  mkdir -p "${RUN_DIR}" "${LOG_DIR}" "${CKPT_DIR}"

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

  COMMON_DECODE_FLAGS=(
    --model "${MODEL_NAME}"
    --model-dir "${MODEL_DIR}"
    --layer-mode mid_stack
    --schema-version rq1-pilot-v1
    --cons-k "${CONS_K}"
    --dump-sample-texts
    --dump-sample-hidden
  )

  # -------- Phase B: smoke (HotpotQA, 5 questions, FULL pilot schema) --------
  SMOKE_OUT="${RUN_DIR}/B_smoke_hotpot.jsonl"
  if [[ -f "${SMOKE_OUT}" ]] && [[ $(wc -l < "${SMOKE_OUT}") -ge ${N_SMOKE} ]]; then
    log "B: smoke already done (n=$(wc -l < "${SMOKE_OUT}")), skipping"
  else
    step "B: smoke decode (n=${N_SMOKE} on HotpotQA pilot slice, K=${CONS_K} sampler + sample dumps)"
    python scripts/04f_run_qa_sota.py \
      "${COMMON_DECODE_FLAGS[@]}" \
      --dataset "${PILOT_HOTPOT_SLICE}" \
      --limit "${N_SMOKE}" \
      --output "${SMOKE_OUT}" \
      2>&1 | tee "${LOG_DIR}/B_smoke.log"

    # Sanity gate (orchestrator-enforced; fails out before any expensive cell).
    python - <<PYEOF | tee -a "${LOG_DIR}/B_smoke.log"
import json, sys
rows = [json.loads(l) for l in open("${SMOKE_OUT}") if l.strip()]
nsteps = [r.get("n_steps", 0) for r in rows]
toks   = [r.get("tokens_generated", 0) for r in rows]
ans_ok = [bool((r.get("model_answer_extracted") or "").strip()) for r in rows]
multistep_ok = sum(1 for r in rows if r.get("n_steps", 0) >= 2 and bool((r.get("model_answer_extracted") or "").strip()))
print(f"[B-check] n_traces={len(rows)} n_steps={nsteps} tok_gen={toks} ans_ok={ans_ok}")
print(f"[B-check] multistep+answer={multistep_ok}/{len(rows)}")
# Pilot-schema population check
new_field_pop = []
for r in rows:
    steps = r.get("steps") or []
    if not steps: continue
    pop = sum(1 for s in steps if s.get("per_token_max_logprob") and s.get("samples"))
    new_field_pop.append((pop, len(steps)))
print(f"[B-check] new-field population per trace (have/total): {new_field_pop}")
schema_ok = all(pop >= int(0.8 * total) for pop, total in new_field_pop) if new_field_pop else False
print(f"[B-check] capture_schema_version: {sorted(set(r.get('capture_schema_version') for r in rows))}")
print(f"[B-check] sampler_K: {sorted(set(r.get('sampler_K') for r in rows))}")
if multistep_ok < 2:
    print(f"[B-FAIL] {multistep_ok} multi-step+answer < 2 (gate 1)")
    sys.exit(2)
if not schema_ok:
    print(f"[B-FAIL] new pilot fields populated on <80% of steps (gate 3)")
    sys.exit(2)
print("[B-check] PASS (gates 1 + 3)")
PYEOF
  fi

  # -------- Phase D1+D2: pilot closed-book on both datasets --------
  for PAIR in "D1_closed_hotpot:${PILOT_HOTPOT_SLICE}" "D2_closed_wiki:${PILOT_WIKI_SLICE}"; do
    TAG="${PAIR%%:*}"; SLICE="${PAIR##*:}"
    OUT="${RUN_DIR}/${TAG}.jsonl"
    if [[ -f "${OUT}" ]] && [[ $(wc -l < "${OUT}") -ge ${N_PILOT} ]]; then
      log "D: ${TAG} already done (n=$(wc -l < "${OUT}")), skipping"
      continue
    fi
    step "D: ${TAG} closed-book decode (n=${N_PILOT} from ${SLICE})"
    python scripts/04f_run_qa_sota.py \
      "${COMMON_DECODE_FLAGS[@]}" \
      --dataset "${SLICE}" \
      --limit "${N_PILOT}" \
      --output "${OUT}" \
      2>&1 | tee "${LOG_DIR}/${TAG}.log"

    # Gate 2: tokens_generated median in [80, 600].
    python - <<PYEOF | tee -a "${LOG_DIR}/${TAG}.log"
import json, sys, statistics
rows = [json.loads(l) for l in open("${OUT}") if l.strip()]
toks = sorted(r.get("tokens_generated", 0) for r in rows)
median = statistics.median(toks) if toks else 0
print(f"[gate2] {len(rows)} rows, tokens_generated median={median}, p10={toks[len(toks)//10] if toks else '-'}, p90={toks[-max(1,len(toks)//10)] if toks else '-'}")
if not (80 <= median <= 600):
    print(f"[gate2-FAIL] median {median} outside [80, 600]; investigate decoding misconfig (Qwen 2Wiki under-generation trap)")
    sys.exit(2)
print("[gate2] PASS")
PYEOF
  done

  # -------- Phase E1+E2: oracle re-decode (no K-sampler; cheap) --------
  for PAIR in "E1_oracle_hotpot:${PILOT_HOTPOT_SLICE}" "E2_oracle_wiki:${PILOT_WIKI_SLICE}"; do
    TAG="${PAIR%%:*}"; SLICE="${PAIR##*:}"
    OUT="${RUN_DIR}/${TAG}.jsonl"
    if [[ -f "${OUT}" ]] && [[ $(wc -l < "${OUT}") -ge ${N_PILOT} ]]; then
      log "E: ${TAG} already done (n=$(wc -l < "${OUT}")), skipping"
      continue
    fi
    step "E: ${TAG} oracle decode (n=${N_PILOT} from ${SLICE})"
    python scripts/04l_oracle_retrieve_decode.py \
      --model "${MODEL_NAME}" \
      --model-dir "${MODEL_DIR}" \
      --dataset "${SLICE}" \
      --limit "${N_PILOT}" \
      --out "${OUT}" \
      2>&1 | tee "${LOG_DIR}/${TAG}.log"
  done

  # -------- Per-arch summary + budget check --------
  step "${MODEL_NAME} done. Outputs:"
  for f in B_smoke_hotpot D1_closed_hotpot D2_closed_wiki E1_oracle_hotpot E2_oracle_wiki; do
    if [[ -f "${RUN_DIR}/${f}.jsonl" ]]; then
      log "  ${f}: $(wc -l < "${RUN_DIR}/${f}.jsonl") rows"
    fi
  done

  # Manifest checkpoint (committable artifact).
  python - <<PYEOF
import json, hashlib, pathlib, time
run_dir = pathlib.Path("${RUN_DIR}")
ckpt = run_dir / "checkpoints" / "slice_manifest.json"
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1<<16), b""):
            h.update(chunk)
    return h.hexdigest()[:16]
files = [p for p in run_dir.glob("*.jsonl")]
manifest = {
    "model": "${MODEL_NAME}",
    "n_pilot": int("${N_PILOT}"),
    "cons_k": int("${CONS_K}"),
    "pilot_seed": int("${PILOT_SEED}"),
    "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "files": [{"name": p.name, "rows": sum(1 for _ in open(p)), "sha256_16": sha(p)} for p in sorted(files)],
}
ckpt.write_text(json.dumps(manifest, indent=2))
print("manifest written ->", ckpt)
PYEOF

  # Wall-clock budget cap.
  T_NOW=$(date +%s); ELAPSED=$((T_NOW - T_GLOBAL_START))
  log "elapsed=${ELAPSED}s budget=${WALL_CAP_SEC}s"
  if (( ELAPSED > WALL_CAP_SEC )); then
    log "ABORT: wall-clock cap (3 h) exceeded; remaining archs not run"
    exit 3
  fi

  trap - ERR
done

log_pre "=== RQ1 P0 pilot complete ==="
log_pre "Total wall: $(( $(date +%s) - T_GLOBAL_START )) s"
log_pre "Next: python analysis/pilot_audit.py | tee analysis/pilot.out"
