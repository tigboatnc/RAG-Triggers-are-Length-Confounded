#!/usr/bin/env bash
# v3 model download. HF safetensors snapshots (NOT GGUF) — quantization
# happens at load time via bitsandbytes.NF4. See _AGENT.md "Engine pivot"
# for why we eat the ~70 GB fp16 download instead of pre-quantized AWQ/GPTQ.
#
# Models:
#   - Qwen/Qwen3.6-35B-A3B-Instruct  (primary generator, MoE ~3B active/token)
#   - microsoft/Phi-4-mini-instruct  (lightweight reflector / baseline)
#
# Revisions: not pinned by default. On first successful download we capture the
# resolved commit SHA into models/<name>.revision. After verifying the run is
# good, hardcode the SHA into the REV_QWEN / REV_PHI4MINI vars below and commit.
#
# Usage:
#   bash scripts/download_models.sh         # phi-4-mini only (small, smoke first)
#   bash scripts/download_models.sh all     # both models

set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

ENV_NAME="recollection"

# Pinned SHAs captured 2026-05-09. Bump by clearing + rerunning.
REV_QWEN="995ad96eacd98c81ed38be0c5b274b04031597b0"
REV_PHI4MINI="cfbefacb99257ffa30c83adab238a50856ac3083"

# The bare Qwen3.6-35B-A3B repo IS the instruct variant (chat_template.jinja
# is bundled). There is no "-Instruct" suffix repo. Architecture tag: qwen3_5_moe.
REPO_QWEN="Qwen/Qwen3.6-35B-A3B"
REPO_PHI4MINI="microsoft/Phi-4-mini-instruct"

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

# Default = small model only (fast smoke). `all` adds the 35B MoE.
fetch_snapshot "$REPO_PHI4MINI" "phi-4-mini-instruct" "$REV_PHI4MINI"

if [[ "${1:-}" == "all" ]]; then
  fetch_snapshot "$REPO_QWEN" "qwen3.6-35b-a3b-instruct" "$REV_QWEN"
fi

green "downloads complete"
echo
echo "Resolved revisions (pin these into REV_QWEN / REV_PHI4MINI after a clean run):"
for f in models/*.revision; do
  [[ -f "$f" ]] && printf "  %s = %s\n" "$(basename "$f" .revision)" "$(cat "$f")"
done
echo
du -sh models/* 2>/dev/null || true
