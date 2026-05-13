#!/usr/bin/env bash
# v3 model download. HF safetensors snapshots (NOT GGUF) — quantization
# happens at load time via bitsandbytes.NF4 OR runs at bf16 native. See
# _AGENT.md 2026-05-09 #5 for the Phase 1.4 architectural-class generator
# pool and why we pull all four.
#
# Models (Phase 1.4 three-way + optional fourth):
#   - microsoft/Phi-4-mini-instruct       (Dense SLM, 32 layers, ~7 GB bf16)
#   - google/gemma-4-26B-A4B-it           (Standard MoE, sliding-window+global, ~52 GB bf16)
#   - Qwen/Qwen3.6-35B-A3B                (Hybrid Linear MoE, 3:1 Gated DeltaNet : full, ~69 GB bf16)
#   - google/gemma-4-31B-it    (optional) (Scale-matched Dense ablation, ~63 GB bf16)
#
# Revisions: pinned (SHAs captured 2026-05-09). Bump by clearing + rerunning
# this script; resolved SHAs get re-written to models/<name>.revision on each
# successful pull.
#
# Usage:
#   bash scripts/download_models.sh             # phi-4-mini only (small, smoke first)
#   bash scripts/download_models.sh all         # phi + qwen3.6 (Phase 0/1 baseline pair)
#   bash scripts/download_models.sh phase1.4    # phi + gemma-4-26B-A4B + qwen3.6 (canonical 3-way)
#   bash scripts/download_models.sh phase1.4-full   # adds gemma-4-31B Dense (optional 4th row)

set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

ENV_NAME="recollection"

# Pinned SHAs captured 2026-05-09. Bump by clearing + rerunning.
REV_QWEN="995ad96eacd98c81ed38be0c5b274b04031597b0"
REV_PHI4MINI="cfbefacb99257ffa30c83adab238a50856ac3083"
# Gemma 4 family (released ~2026-05-05, transformers 5.8.0 supports `gemma4`):
REV_GEMMA4_26B_A4B="462a98a12e28e2cbcfccaf78fe41e3e50235e6ae"
REV_GEMMA4_31B="ba74f5b6c647c0911554e50278d6f6f4477f9010"
# RQ1 reference backbone (added 2026-05-13). Apache-2.0, ungated.
REV_QWEN25_7B="a09a35458c702b33eeacc393d103063234e8bc28"

# The bare Qwen3.6-35B-A3B repo IS the instruct variant (chat_template.jinja
# is bundled). There is no "-Instruct" suffix repo. Architecture tag: qwen3_5_moe.
REPO_QWEN="Qwen/Qwen3.6-35B-A3B"
REPO_PHI4MINI="microsoft/Phi-4-mini-instruct"
# Gemma 4: `Gemma4ForConditionalGeneration` arch class, model_type=gemma4.
# Multimodal head (Image-Text-to-Text) but text-only path works via
# AutoModelForCausalLM. Sliding-window + sparse global attention; 256K context.
REPO_GEMMA4_26B_A4B="google/gemma-4-26B-A4B-it"
REPO_GEMMA4_31B="google/gemma-4-31B-it"
REPO_QWEN25_7B="Qwen/Qwen2.5-7B-Instruct"

green() { printf "\033[32m[ok]\033[0m %s\n" "$1"; }
yellow() { printf "\033[33m[..]\033[0m %s\n" "$1"; }
red() { printf "\033[31m[!!]\033[0m %s\n" "$1" >&2; }

# conda probe (matches setup.sh)
if ! command -v conda >/dev/null 2>&1; then
  if [[ -x "$HOME/miniconda3/bin/conda" ]]; then
    export PATH="$HOME/miniconda3/bin:$PATH"
  elif [[ -x "$HOME/miniforge3/bin/conda" ]]; then
    export PATH="$HOME/miniforge3/bin:$PATH"
  elif [[ -x "/opt/miniforge3/bin/conda" ]]; then
    export PATH="/opt/miniforge3/bin:$PATH"
  fi
fi
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

if ! command -v hf >/dev/null 2>&1; then
  red "hf CLI missing in env '$ENV_NAME'. Run: bash scripts/setup.sh"
  exit 1
fi

mkdir -p models

# fetch_snapshot REPO LOCAL_NAME [REVISION]
fetch_snapshot() {
  local repo="$1"
  local local_name="$2"
  local revision="${3:-}"
  local target="models/${local_name}"
  local rev_flag=()
  if [[ -n "$revision" ]]; then
    rev_flag=(--revision "$revision")
    yellow "downloading $repo @ $revision -> $target"
  else
    yellow "downloading $repo @ main -> $target  (revision NOT pinned — pin after first good run)"
  fi
  hf download "$repo" \
    --local-dir "$target" \
    "${rev_flag[@]}"
  # Capture resolved SHA so we can pin it.
  local sha_file="${target}/.gitattributes"  # always present in a HF snapshot dir
  local resolved_sha
  if [[ -f "$target/.cache/huggingface/download/.git_sha" ]]; then
    resolved_sha="$(cat "$target/.cache/huggingface/download/.git_sha")"
  else
    # Fallback: ask HF API for the commit SHA of the ref we resolved.
    resolved_sha="$(python - <<PY
from huggingface_hub import HfApi
api = HfApi()
ref = "${revision:-main}"
info = api.repo_info(repo_id="${repo}", revision=ref)
print(info.sha)
PY
)"
  fi
  echo "$resolved_sha" > "models/${local_name}.revision"
  green "${local_name} downloaded (sha=${resolved_sha})"
  [[ -f "$sha_file" ]] || true  # silence unused
}

# Default = small model only (fast smoke). Subcommands add larger sets.
fetch_snapshot "$REPO_PHI4MINI" "phi-4-mini-instruct" "$REV_PHI4MINI"

case "${1:-}" in
  all)
    fetch_snapshot "$REPO_QWEN" "qwen3.6-35b-a3b-instruct" "$REV_QWEN"
    ;;
  phase1.4)
    fetch_snapshot "$REPO_GEMMA4_26B_A4B" "gemma-4-26b-a4b-it" "$REV_GEMMA4_26B_A4B"
    fetch_snapshot "$REPO_QWEN" "qwen3.6-35b-a3b-instruct" "$REV_QWEN"
    ;;
  phase1.4-full)
    fetch_snapshot "$REPO_GEMMA4_26B_A4B" "gemma-4-26b-a4b-it" "$REV_GEMMA4_26B_A4B"
    fetch_snapshot "$REPO_GEMMA4_31B" "gemma-4-31b-it" "$REV_GEMMA4_31B"
    fetch_snapshot "$REPO_QWEN" "qwen3.6-35b-a3b-instruct" "$REV_QWEN"
    ;;
  rq1)
    # RQ1 final pool (2026-05-13): Phi + Gemma + Qwen2.5-7B.
    # Drops Qwen3.6 (chain-of-thought collapses under matched-protocol
    # greedy + enable_thinking=False; documented limitation).
    fetch_snapshot "$REPO_GEMMA4_26B_A4B" "gemma-4-26b-a4b-it" "$REV_GEMMA4_26B_A4B"
    fetch_snapshot "$REPO_QWEN25_7B" "qwen2.5-7b-instruct" "$REV_QWEN25_7B"
    ;;
  "")
    : # phi-only default
    ;;
  *)
    red "unknown subcommand '$1'. Try: '', all, phase1.4, phase1.4-full"
    exit 1
    ;;
esac

green "downloads complete"
echo
echo "Resolved revisions (pin these into REV_QWEN / REV_PHI4MINI after a clean run):"
for f in models/*.revision; do
  [[ -f "$f" ]] && printf "  %s = %s\n" "$(basename "$f" .revision)" "$(cat "$f")"
done
echo
du -sh models/* 2>/dev/null || true
