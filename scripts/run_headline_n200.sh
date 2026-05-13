#!/usr/bin/env bash
# RQ1 Run 1 — Headline cross-arch experiment (n=200 × 3 archs × 2 datasets).
#
# Goals:
#   G1: Paired-bootstrap significance on AID flip_rate vs DRAGIN vs tokens.
#   G2: Full rq1-pilot-v1 schema capture → all signals post-hoc derivable.
#   G3: Architecture-heterogeneity confirmation at publishable power.
#
# Cohort:
#   3 archs × 2 datasets × {closed D, oracle E} = 12 cells.
#   n=200 per cell. K=5 boundary sampler + sample text + EOS hidden dump.
#   Slices: existing 500-question v2test sets (first 200 used via --limit).
#
# Arch ordering (fastest first):
#   1. phi-4-mini-instruct      (~42 tok/s, ~2.5 h for D+E across 2 datasets)
#   2. qwen2.5-7b-instruct      (~25 tok/s, ~5 h)
#   3. gemma-4-26b-a4b-it        (~12 tok/s, ~10 h)
#   Total serial: ~18-22 h.
#
# Wall-clock budget: 30 h (generous margin for retry/resume).
#
# Usage:
#   bash scripts/run_headline_n200.sh                              # all 3 archs
#   ARCHS="phi-4-mini-instruct" bash scripts/run_headline_n200.sh  # single arch
#   N=100 bash scripts/run_headline_n200.sh                        # smaller cohort
#
# Outputs:
#   runs/v2_headline_<model>/
#     ├── B_smoke_hotpot.jsonl
#     ├── D1_closed_hotpot.jsonl       (n=200, rq1-pilot-v1 schema)
#     ├── D2_closed_wiki.jsonl
#     ├── E1_oracle_hotpot.jsonl       (oracle re-decode, paired by qid)
#     ├── E2_oracle_wiki.jsonl
#     ├── checkpoints/
#     └── logs/

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
ARCHS="${ARCHS:-phi-4-mini-instruct qwen2.5-7b-instruct gemma-4-26b-a4b-it}"
N="${N:-200}"
N_SMOKE="${N_SMOKE:-5}"
CONS_K="${CONS_K:-5}"

HOTPOT_SLICE="${HOTPOT_SLICE:-data/hotpotqa_distractor_500_v2test.jsonl}"
WIKI_SLICE="${WIKI_SLICE:-data/2wikimultihop_distractor_500_v2test.jsonl}"

TS() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

# ---------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------
PRE_LOG="runs/v2_headline_preflight.log"
mkdir -p runs
log_pre() { echo "[$(TS)] $*" | tee -a "${PRE_LOG}"; }

log_pre "=== RQ1 Run 1 — Headline n=${N} starting ==="
log_pre "git rev: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
log_pre "python:  $(python -V 2>&1)"
log_pre "nvidia-smi: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1)"
log_pre "ARCHS=${ARCHS}"
log_pre "N=${N}  N_SMOKE=${N_SMOKE}  CONS_K=${CONS_K}"

# Verify slices exist
for f in "${HOTPOT_SLICE}" "${WIKI_SLICE}"; do
  if [[ ! -f "${f}" ]]; then
    log_pre "FATAL: data slice missing: ${f}"
    log_pre "Run the dataset builders first:"
    log_pre "  python scripts/04_dataset.py --n 500 --seed 20260601 --split-seed 20260601 --exclude-ids-from data/hotpotqa_distractor_200.jsonl --all-test --name hotpotqa_distractor_500_v2test"
    log_pre "  python scripts/04_dataset_2wiki.py --n 500 --seed 20260601 --name 2wikimultihop_distractor_500_v2test"
    exit 2
  fi
  log_pre "slice ${f}: $(wc -l < "${f}") rows"
done

# ---------------------------------------------------------------------
# Per-arch headline decode
# ---------------------------------------------------------------------
T_GLOBAL_START=$(date +%s)
WALL_CAP_SEC=$((30 * 3600))  # 30 h cap

for MODEL_NAME in ${ARCHS}; do
  MODEL_DIR="models/${MODEL_NAME}"
  RUN_DIR="runs/v2_headline_${MODEL_NAME}"
  LOG_DIR="${RUN_DIR}/logs"
  CKPT_DIR="${RUN_DIR}/checkpoints"
  mkdir -p "${RUN_DIR}" "${LOG_DIR}" "${CKPT_DIR}"

  log() { echo "[$(TS)] [${MODEL_NAME}] $*" | tee -a "${LOG_DIR}/00_master.log"; }
  step() { log "==== $* ===="; }
  trap 'log "ABORT @ line $LINENO (last cmd: $BASH_COMMAND)"' ERR

  step "START: ${MODEL_NAME} (n=${N}, K=${CONS_K}, schema=rq1-pilot-v1)"
  log "MODEL_DIR=${MODEL_DIR}"

  if [[ ! -d "${MODEL_DIR}" ]] || ! ls "${MODEL_DIR}"/*.safetensors >/dev/null 2>&1; then
    log "MODEL WEIGHTS MISSING at ${MODEL_DIR}"
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

  # -------- Phase B: 5-question smoke (validates model loads + schema) --------
  SMOKE_OUT="${RUN_DIR}/B_smoke_hotpot.jsonl"
  if [[ -f "${SMOKE_OUT}" ]] && [[ $(wc -l < "${SMOKE_OUT}") -ge ${N_SMOKE} ]]; then
    log "B: smoke present (n=$(wc -l < "${SMOKE_OUT}")), skipping"
  else
    step "B: smoke decode (n=${N_SMOKE}, HotpotQA)"
    python scripts/04f_run_qa_sota.py \
      "${COMMON_DECODE_FLAGS[@]}" \
      --dataset "${HOTPOT_SLICE}" \
      --limit "${N_SMOKE}" \
      --output "${SMOKE_OUT}" \
      2>&1 | tee "${LOG_DIR}/B_smoke.log"

    python - <<PYEOF | tee -a "${LOG_DIR}/B_smoke.log"
import json, sys
rows = [json.loads(l) for l in open("${SMOKE_OUT}") if l.strip()]
multistep_ok = sum(1 for r in rows if r.get("n_steps", 0) >= 2 and bool((r.get("model_answer_extracted") or "").strip()))
new_field_pop = []
for r in rows:
    steps = r.get("steps") or []
    if not steps: continue
    pop = sum(1 for s in steps if s.get("per_token_max_logprob") and s.get("samples"))
    new_field_pop.append((pop, len(steps)))
schema_ok = all(pop >= int(0.8 * total) for pop, total in new_field_pop) if new_field_pop else False
print(f"[B-check] n={len(rows)} multistep+answer={multistep_ok} schema_pop={new_field_pop}")
print(f"[B-check] schema_version={sorted(set(r.get('capture_schema_version') for r in rows))}")
if multistep_ok < 2:
    print(f"[B-FAIL] {multistep_ok} multi-step+answer < 2"); sys.exit(2)
if not schema_ok:
    print(f"[B-FAIL] pilot fields under-populated"); sys.exit(2)
print("[B-check] PASS")
PYEOF
  fi

  # -------- Phase D: closed-book decode (the main compute) --------
  for PAIR in "D1_closed_hotpot:${HOTPOT_SLICE}" "D2_closed_wiki:${WIKI_SLICE}"; do
    TAG="${PAIR%%:*}"; SLICE="${PAIR##*:}"
    OUT="${RUN_DIR}/${TAG}.jsonl"
    EXISTING_N=0
    [[ -f "${OUT}" ]] && EXISTING_N=$(wc -l < "${OUT}")

    if [[ ${EXISTING_N} -ge ${N} ]]; then
      log "D: ${TAG} already done (n=${EXISTING_N}), skipping"
      continue
    fi

    step "D: ${TAG} closed-book (n=${N} from ${SLICE}, resuming from ${EXISTING_N})"
    python scripts/04f_run_qa_sota.py \
      "${COMMON_DECODE_FLAGS[@]}" \
      --dataset "${SLICE}" \
      --limit "${N}" \
      --output "${OUT}" \
      2>&1 | tee "${LOG_DIR}/${TAG}.log"

    # Gate: tokens median in [50, 700] (wider than pilot for robustness at n=200)
    python - <<PYEOF | tee -a "${LOG_DIR}/${TAG}.log"
import json, sys, statistics
rows = [json.loads(l) for l in open("${OUT}") if l.strip()]
toks = sorted(r.get("tokens_generated", 0) for r in rows)
median = statistics.median(toks) if toks else 0
print(f"[gate] n={len(rows)} tok_median={median:.0f} p10={toks[len(toks)//10]:.0f} p90={toks[-max(1,len(toks)//10)]:.0f}")
if not (50 <= median <= 700):
    print(f"[gate-WARN] median {median} outside [50, 700]; investigate but continuing")
print("[gate] OK")
PYEOF
  done

  # -------- Phase E: oracle re-decode (no K-sampler; fast) --------
  for PAIR in "E1_oracle_hotpot:${HOTPOT_SLICE}" "E2_oracle_wiki:${WIKI_SLICE}"; do
    TAG="${PAIR%%:*}"; SLICE="${PAIR##*:}"
    OUT="${RUN_DIR}/${TAG}.jsonl"
    EXISTING_N=0
    [[ -f "${OUT}" ]] && EXISTING_N=$(wc -l < "${OUT}")

    if [[ ${EXISTING_N} -ge ${N} ]]; then
      log "E: ${TAG} already done (n=${EXISTING_N}), skipping"
      continue
    fi

    step "E: ${TAG} oracle decode (n=${N} from ${SLICE})"
    python scripts/04l_oracle_retrieve_decode.py \
      --model "${MODEL_NAME}" \
      --model-dir "${MODEL_DIR}" \
      --dataset "${SLICE}" \
      --limit "${N}" \
      --out "${OUT}" \
      2>&1 | tee "${LOG_DIR}/${TAG}.log"
  done

  # -------- Per-arch summary --------
  step "${MODEL_NAME} complete"
  for f in B_smoke_hotpot D1_closed_hotpot D2_closed_wiki E1_oracle_hotpot E2_oracle_wiki; do
    [[ -f "${RUN_DIR}/${f}.jsonl" ]] && log "  ${f}: $(wc -l < "${RUN_DIR}/${f}.jsonl") rows"
  done

  # Manifest checkpoint
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
files = sorted(run_dir.glob("*.jsonl"))
manifest = {
    "model": "${MODEL_NAME}",
    "n": int("${N}"),
    "cons_k": int("${CONS_K}"),
    "schema": "rq1-pilot-v1",
    "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "files": [{"name": p.name, "rows": sum(1 for _ in open(p)), "sha256_16": sha(p)} for p in files],
}
ckpt.write_text(json.dumps(manifest, indent=2))
print(f"[ckpt] manifest -> {ckpt}")
PYEOF

  # Budget check
  T_NOW=$(date +%s); ELAPSED=$((T_NOW - T_GLOBAL_START))
  log "elapsed=${ELAPSED}s ($(( ELAPSED / 3600 ))h $(( (ELAPSED % 3600) / 60 ))m)"
  if (( ELAPSED > WALL_CAP_SEC )); then
    log "ABORT: wall-clock cap (30 h) exceeded"
    exit 3
  fi

  trap - ERR
done

log_pre "=== RQ1 Run 1 complete ==="
log_pre "Total wall: $(( $(date +%s) - T_GLOBAL_START )) s"
log_pre "Next step:"
log_pre "  python analysis/pilot_audit.py --root runs --prefix v2_headline | tee analysis/headline_n200.out"
