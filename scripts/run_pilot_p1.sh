#!/usr/bin/env bash
# RQ1 Experiment P1 — extended-schema pilot with rq1-pilot-v2 captures.
#
# Adds 4 NOVEL trigger captures on top of P0 (rq1-pilot-v1):
#   N1 moe_routing_entropy       (Gemma only; dense models report None)
#   N2 prefix_answer_now         (KV-fork "answer NOW" probe per step)
#   N3 multi_layer_dragin        (DRAGIN-RIND at first/mid/last full-attn)
#   N4 attention_sink_frac       (StreamingLLM-style sink fraction)
#
# Per parent's plan: parallel split — Phi + Qwen2.5 on one box (small),
# Gemma on the warm box. Same pilot slices as P0 (n=25, seed=20260513);
# fresh outputs land at runs/v2_pilot_v2_<arch>/ to avoid clobbering P0.
#
# Usage:
#   bash scripts/run_pilot_p1.sh                          # all 3 archs serial
#   ARCHS="phi-4-mini-instruct qwen2.5-7b-instruct" \
#       bash scripts/run_pilot_p1.sh                      # Box A (small models)
#   ARCHS="gemma-4-26b-a4b-it" bash scripts/run_pilot_p1.sh  # Box B (warm)
#
# Compute estimate (per-arch, A800; K=5 sampler + 4 N captures):
#   Phi      ~10 min wall  (small + low n_steps)
#   Qwen2.5  ~30 min wall  (medium pacing)
#   Gemma    ~3 h wall     (high n_steps, prefix-probe overhead non-trivial)
#
# All cells: 25 traces × {closed, oracle} × {Hot, 2Wiki} per arch.

set -euo pipefail
shopt -s inherit_errexit

if [[ -x /venv/recollection/bin/python ]]; then
  export PATH="/venv/recollection/bin:${PATH}"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

ARCHS="${ARCHS:-phi-4-mini-instruct qwen2.5-7b-instruct gemma-4-26b-a4b-it}"
N_SMOKE="${N_SMOKE:-5}"
N_PILOT="${N_PILOT:-25}"
CONS_K="${CONS_K:-5}"
PREFIX_PROBE_TOKENS="${PREFIX_PROBE_TOKENS:-10}"

PILOT_HOTPOT_SLICE="data/hotpotqa_distractor_25_pilot.jsonl"
PILOT_WIKI_SLICE="data/2wikimultihop_distractor_25_pilot.jsonl"

TS() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

PRE_LOG="runs/v2_pilot_v2_preflight.log"
mkdir -p runs
log_pre() { echo "[$(TS)] $*" | tee -a "${PRE_LOG}"; }

log_pre "=== RQ1 P1 (rq1-pilot-v2 schema) starting ==="
log_pre "git rev: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
log_pre "ARCHS=${ARCHS}  N_PILOT=${N_PILOT}  CONS_K=${CONS_K}  PREFIX_PROBE_TOKENS=${PREFIX_PROBE_TOKENS}"

if [[ ! -f "${PILOT_HOTPOT_SLICE}" ]] || [[ ! -f "${PILOT_WIKI_SLICE}" ]]; then
  log_pre "FATAL: pilot slices missing (run scripts/run_pilot_p0.sh first to build them)."
  exit 2
fi

T_GLOBAL_START=$(date +%s)

for MODEL_NAME in ${ARCHS}; do
  MODEL_DIR="models/${MODEL_NAME}"
  RUN_DIR="runs/v2_pilot_v2_${MODEL_NAME}"
  LOG_DIR="${RUN_DIR}/logs"
  CKPT_DIR="${RUN_DIR}/checkpoints"
  mkdir -p "${RUN_DIR}" "${LOG_DIR}" "${CKPT_DIR}"

  log() { echo "[$(TS)] [${MODEL_NAME}] $*" | tee -a "${LOG_DIR}/00_master.log"; }
  step() { log "==== $* ===="; }
  trap 'log "ABORT @ line $LINENO (last cmd: $BASH_COMMAND)"' ERR

  step "PRE: ${MODEL_NAME} (rq1-pilot-v2)"
  if [[ ! -d "${MODEL_DIR}" ]] || ! ls "${MODEL_DIR}"/*.safetensors >/dev/null 2>&1; then
    log "MODEL WEIGHTS MISSING — skipping ${MODEL_NAME}"
    continue
  fi

  COMMON_FLAGS=(
    --model "${MODEL_NAME}"
    --model-dir "${MODEL_DIR}"
    --layer-mode mid_stack
    --schema-version rq1-pilot-v2
    --cons-k "${CONS_K}"
    --dump-sample-texts
    --dump-sample-hidden
    --prefix-probe-tokens "${PREFIX_PROBE_TOKENS}"
    --multi-layer-dragin auto
  )

  # Phase B: smoke (5 questions on HotpotQA)
  SMOKE_OUT="${RUN_DIR}/B_smoke_hotpot.jsonl"
  if [[ -f "${SMOKE_OUT}" ]] && [[ $(wc -l < "${SMOKE_OUT}") -ge ${N_SMOKE} ]]; then
    log "B: smoke already done, skipping"
  else
    step "B: rq1-pilot-v2 smoke (n=${N_SMOKE})"
    python scripts/04f_run_qa_sota.py "${COMMON_FLAGS[@]}" \
      --dataset "${PILOT_HOTPOT_SLICE}" \
      --limit "${N_SMOKE}" \
      --output "${SMOKE_OUT}" 2>&1 | tee "${LOG_DIR}/B_smoke.log"

    # Sanity: confirm v2 fields populated.
    python - <<PYEOF | tee -a "${LOG_DIR}/B_smoke.log"
import json, sys
rows = [json.loads(l) for l in open("${SMOKE_OUT}") if l.strip()]
print(f"[v2-check] n_traces={len(rows)} schema={sorted({r.get('capture_schema_version') for r in rows})}")
v2_pop = []
for r in rows:
    steps = r.get("steps") or []
    if not steps: continue
    n_with_v2 = sum(1 for s in steps if "attention_sink_frac" in s and s["attention_sink_frac"] is not None)
    v2_pop.append((n_with_v2, len(steps)))
print(f"[v2-check] attention_sink_frac population: {v2_pop}")
n_with_prefix = sum(1 for r in rows for s in (r.get("steps") or []) if s.get("prefix_answer_now"))
n_total_steps = sum(len(r.get("steps") or []) for r in rows)
print(f"[v2-check] prefix_answer_now populated on {n_with_prefix}/{n_total_steps} steps")
moe = [s.get("moe_routing_entropy") for r in rows for s in (r.get("steps") or [])]
n_moe = sum(1 for v in moe if v is not None)
print(f"[v2-check] moe_routing_entropy populated on {n_moe}/{len(moe)} steps (None expected on dense)")
ml = [list(s.get("multi_layer_dragin", {}).keys()) for r in rows for s in (r.get("steps") or []) if s.get("multi_layer_dragin")]
print(f"[v2-check] multi_layer_dragin layer indices observed: {sorted({l for ls in ml for l in ls})}")
print("[v2-check] PASS" if v2_pop and v2_pop[0][0] >= 1 else "[v2-WARN] no v2 fields populated")
PYEOF
  fi

  # Phase D1+D2: closed-book on both datasets
  for PAIR in "D1_closed_hotpot:${PILOT_HOTPOT_SLICE}" "D2_closed_wiki:${PILOT_WIKI_SLICE}"; do
    TAG="${PAIR%%:*}"; SLICE="${PAIR##*:}"
    OUT="${RUN_DIR}/${TAG}.jsonl"
    if [[ -f "${OUT}" ]] && [[ $(wc -l < "${OUT}") -ge ${N_PILOT} ]]; then
      log "D: ${TAG} already done, skipping"; continue
    fi
    step "D: ${TAG} closed-book v2 (n=${N_PILOT})"
    python scripts/04f_run_qa_sota.py "${COMMON_FLAGS[@]}" \
      --dataset "${SLICE}" --limit "${N_PILOT}" \
      --output "${OUT}" 2>&1 | tee "${LOG_DIR}/${TAG}.log"
  done

  # Phase E1+E2: oracle re-decode (no v2 captures needed; just lift labels).
  for PAIR in "E1_oracle_hotpot:${PILOT_HOTPOT_SLICE}" "E2_oracle_wiki:${PILOT_WIKI_SLICE}"; do
    TAG="${PAIR%%:*}"; SLICE="${PAIR##*:}"
    OUT="${RUN_DIR}/${TAG}.jsonl"
    if [[ -f "${OUT}" ]] && [[ $(wc -l < "${OUT}") -ge ${N_PILOT} ]]; then
      log "E: ${TAG} already done, skipping"; continue
    fi
    step "E: ${TAG} oracle decode (n=${N_PILOT})"
    python scripts/04l_oracle_retrieve_decode.py \
      --model "${MODEL_NAME}" --model-dir "${MODEL_DIR}" \
      --dataset "${SLICE}" --limit "${N_PILOT}" \
      --out "${OUT}" 2>&1 | tee "${LOG_DIR}/${TAG}.log"
  done

  # Per-arch manifest.
  python - <<PYEOF
import json, hashlib, pathlib, time
run_dir = pathlib.Path("${RUN_DIR}")
ckpt = run_dir / "checkpoints" / "slice_manifest.json"
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1<<16), b""): h.update(chunk)
    return h.hexdigest()[:16]
files = list(run_dir.glob("*.jsonl"))
ckpt.write_text(json.dumps({
    "model": "${MODEL_NAME}",
    "schema_version": "rq1-pilot-v2",
    "n_pilot": int("${N_PILOT}"),
    "cons_k": int("${CONS_K}"),
    "prefix_probe_tokens": int("${PREFIX_PROBE_TOKENS}"),
    "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "files": [{"name": p.name, "rows": sum(1 for _ in open(p)), "sha256_16": sha(p)} for p in sorted(files)],
}, indent=2))
print("manifest written ->", ckpt)
PYEOF

  trap - ERR
done

log_pre "=== P1 complete ==="
log_pre "Total wall: $(( $(date +%s) - T_GLOBAL_START )) s"
log_pre "Next: extend analysis/pilot_audit.py for v2 fields, then run on runs/v2_pilot_v2_*"
