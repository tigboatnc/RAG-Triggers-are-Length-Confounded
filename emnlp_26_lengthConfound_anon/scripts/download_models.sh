#!/usr/bin/env bash
# Download the two headline-pool model weights as HF safetensors snapshots.
# Quantisation happens at load time via bitsandbytes.NF4 or runs at bf16
# native (see run_headline_n200.sh for the decode wrapper).
#
# Models:
#   - Qwen/Qwen2.5-7B-Instruct          (dense Transformer, ~15 GB bf16)
#   - google/gemma-4-26B-A4B-it         (MoE-hybrid sliding-window + global,
#                                        ~52 GB bf16, 3.8B active / 25.2B total)
#
# Revisions are pinned by SHA so the paper's numbers can be reproduced
# byte-for-byte.  Bump by clearing models/ and rerunning; the resolved SHA
# is written to models/<name>.revision on each successful pull.
#
# Usage:
#   bash scripts/download_models.sh

set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

ENV_NAME="recollection"

# Pinned SHAs (captured 2026-05-13).
REV_QWEN25_7B="a09a35458c702b33eeacc393d103063234e8bc28"
REV_GEMMA4_26B_A4B="462a98a12e28e2cbcfccaf78fe41e3e50235e6ae"

REPO_QWEN25_7B="Qwen/Qwen2.5-7B-Instruct"
REPO_GEMMA4_26B_A4B="google/gemma-4-26B-A4B-it"

green()  { printf "\033[32m[ok]\033[0m %s\n" "$1"; }
yellow() { printf "\033[33m[..]\033[0m %s\n" "$1"; }
red()    { printf "\033[31m[!!]\033[0m %s\n" "$1" >&2; }

# conda probe (matches setup.sh)
if ! command -v conda >/dev/null 2>&1; then
  if   [[ -x "$HOME/miniconda3/bin/conda" ]]; then export PATH="$HOME/miniconda3/bin:$PATH"
  elif [[ -x "$HOME/miniforge3/bin/conda" ]]; then export PATH="$HOME/miniforge3/bin:$PATH"
  elif [[ -x "/opt/miniforge3/bin/conda"  ]]; then export PATH="/opt/miniforge3/bin:$PATH"
  fi
fi
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

if ! command -v hf >/dev/null 2>&1; then
  red "hf CLI missing in env '$ENV_NAME'.  Run: bash scripts/setup.sh"
  exit 1
fi

mkdir -p models

# fetch_snapshot REPO LOCAL_NAME REVISION
fetch_snapshot() {
  local repo="$1" local_name="$2" revision="$3"
  local target="models/${local_name}"
  yellow "downloading $repo @ $revision -> $target"
  hf download "$repo" --local-dir "$target" --revision "$revision"
  # Capture resolved SHA so we can pin it.
  local resolved_sha
  if [[ -f "$target/.cache/huggingface/download/.git_sha" ]]; then
    resolved_sha="$(cat "$target/.cache/huggingface/download/.git_sha")"
  else
    resolved_sha="$(python - <<PY
from huggingface_hub import HfApi
api = HfApi()
info = api.repo_info(repo_id="${repo}", revision="${revision}")
print(info.sha)
PY
)"
  fi
  echo "$resolved_sha" > "models/${local_name}.revision"
  green "${local_name} downloaded (sha=${resolved_sha})"
}

fetch_snapshot "$REPO_QWEN25_7B"     "qwen2.5-7b-instruct" "$REV_QWEN25_7B"
fetch_snapshot "$REPO_GEMMA4_26B_A4B" "gemma-4-26b-a4b-it" "$REV_GEMMA4_26B_A4B"

green "downloads complete"
echo
echo "Resolved revisions:"
for f in models/*.revision; do
  [[ -f "$f" ]] && printf "  %s = %s\n" "$(basename "$f" .revision)" "$(cat "$f")"
done
echo
du -sh models/* 2>/dev/null || true
