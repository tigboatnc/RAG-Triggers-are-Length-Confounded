#!/usr/bin/env bash
# Download GGUF models via huggingface-cli into ./models/.
# Reproducible: explicit repo + filename. Run inside the conda env.
#
# Usage:
#   bash scripts/download_models.sh         # phase 0 only (small model)
#   bash scripts/download_models.sh all     # phase 0 + phase 3 models

set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

ENV_NAME="recollection"

if ! command -v conda >/dev/null 2>&1; then
  if [[ -x "$HOME/miniconda3/bin/conda" ]]; then
    export PATH="$HOME/miniconda3/bin:$PATH"
  fi
fi
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "[!!] huggingface-cli missing in env '$ENV_NAME'. Run: bash scripts/setup.sh" >&2
  exit 1
fi

mkdir -p models

green() { printf "\033[32m[ok]\033[0m %s\n" "$1"; }
yellow() { printf "\033[33m[..]\033[0m %s\n" "$1"; }

fetch() {
  local repo="$1"
  local filename="$2"
  yellow "downloading $repo :: $filename"
  huggingface-cli download "$repo" "$filename" \
    --local-dir "models/" \
    --local-dir-use-symlinks False
  green "models/$filename"
}

# Phase 0: smallest model first (~2.4 GB, Q4_K_M).
fetch "bartowski/Phi-3.5-mini-instruct-GGUF" "Phi-3.5-mini-instruct-Q4_K_M.gguf"

if [[ "${1:-}" == "all" ]]; then
  fetch "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF" "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf"
  yellow "Phi-4-mini-instruct GGUF: not yet pinned. Update this script when a stable repo+filename is chosen."
fi

green "downloads complete"
ls -lh models/
