#!/usr/bin/env bash
# Bootstrap script. Detects OS + accelerator, builds the conda env from
# environment.yml, and installs llama-cpp-python with the correct backend.
#
# Strategy:
#   - Mac (Apple Silicon): install the prebuilt Metal wheel (no compile, ~seconds).
#   - Linux + NVIDIA: install the prebuilt CUDA wheel from abetlen's index.
#   - Otherwise: pip's plain wheel (CPU).
#
# Reproducible: pinned versions in environment.yml; llama-cpp-python pinned here.
# Re-runnable: env is updated in place if it already exists.
#
# Usage:
#   bash scripts/setup.sh             # phase-0 minimal deps
#   bash scripts/setup.sh --phase4    # adds sentence-transformers, lancedb, etc.

set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

ENV_NAME="recollection"
LLAMA_CPP_VERSION="0.3.22"   # bumped from 0.3.5 (Phase 3.0): older wheel lacks DeepSeek-R1 pre-tokenizer support
PHASE4=0
for arg in "$@"; do
  case "$arg" in
    --phase4) PHASE4=1 ;;
  esac
done

green() { printf "\033[32m[ok]\033[0m %s\n" "$1"; }
yellow() { printf "\033[33m[..]\033[0m %s\n" "$1"; }
red() { printf "\033[31m[!!]\033[0m %s\n" "$1" >&2; }

# 1) conda on PATH.
if ! command -v conda >/dev/null 2>&1; then
  if [[ -x "$HOME/miniconda3/bin/conda" ]]; then
    export PATH="$HOME/miniconda3/bin:$PATH"
  elif [[ -x "$HOME/miniforge3/bin/conda" ]]; then
    export PATH="$HOME/miniforge3/bin:$PATH"
  else
    red "conda not found. Install Miniconda/Miniforge and re-run."
    exit 1
  fi
fi
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
green "conda $(conda --version | awk '{print $2}') (base=$CONDA_BASE)"

# 2) Detect platform.
OS_NAME="$(uname -s)"
ARCH_NAME="$(uname -m)"
BACKEND="cpu"
LCPP_INDEX=""

case "$OS_NAME" in
  Darwin)
    if [[ "$ARCH_NAME" == "arm64" ]]; then
      BACKEND="metal"
      LCPP_INDEX="https://abetlen.github.io/llama-cpp-python/whl/metal"
    else
      yellow "Intel Mac detected; using plain CPU wheel."
    fi
    ;;
  Linux)
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
      BACKEND="cuda"
      # Pick a CUDA index. Default to cu124; override with LCPP_CUDA= envvar if needed.
      LCPP_INDEX="https://abetlen.github.io/llama-cpp-python/whl/${LCPP_CUDA:-cu124}"
    else
      yellow "No working nvidia-smi; using plain CPU wheel."
    fi
    ;;
  *)
    red "Unsupported OS: $OS_NAME"
    exit 1
    ;;
esac
green "OS=$OS_NAME arch=$ARCH_NAME backend=$BACKEND"

# 3) Create or update the conda env.
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  yellow "env '$ENV_NAME' exists; updating from environment.yml"
  conda env update -n "$ENV_NAME" -f environment.yml --prune
else
  yellow "creating env '$ENV_NAME' from environment.yml"
  conda env create -f environment.yml
fi
conda activate "$ENV_NAME"
green "env active: $CONDA_PREFIX (python=$(python -c 'import sys; print("%d.%d.%d"%sys.version_info[:3])'))"

# 4) llama-cpp-python via prebuilt wheel.
INSTALL_LCPP=1
if pip show llama-cpp-python >/dev/null 2>&1; then
  CURRENT_VERSION="$(pip show llama-cpp-python | awk '/^Version:/ {print $2}')"
  if [[ "$CURRENT_VERSION" == "$LLAMA_CPP_VERSION" ]]; then
    green "llama-cpp-python==$CURRENT_VERSION already installed; skipping"
    INSTALL_LCPP=0
  else
    yellow "llama-cpp-python $CURRENT_VERSION -> $LLAMA_CPP_VERSION"
    pip uninstall -y llama-cpp-python
  fi
fi
if [[ "$INSTALL_LCPP" -eq 1 ]]; then
  yellow "Installing llama-cpp-python==$LLAMA_CPP_VERSION (backend=$BACKEND, prebuilt wheel)"
  if [[ -n "$LCPP_INDEX" ]]; then
    pip install --upgrade --force-reinstall --no-cache-dir \
      "llama-cpp-python==$LLAMA_CPP_VERSION" \
      --extra-index-url "$LCPP_INDEX"
  else
    pip install --upgrade --no-cache-dir "llama-cpp-python==$LLAMA_CPP_VERSION"
  fi
fi
green "llama-cpp-python ready"

# 5) Optional Phase-4 stack.
if [[ "$PHASE4" -eq 1 ]]; then
  yellow "Installing Phase-4 stack (sentence-transformers, lancedb, pandas, tantivy)"
  pip install --no-cache-dir \
    "pandas==2.2.3" \
    "sentence-transformers==3.3.1" \
    "lancedb==0.18.0" \
    "tantivy==0.22.2"
  green "Phase-4 stack ready"
fi

# 6) Smoke import.
python - <<'PY'
import llama_cpp, platform, sys
print(f"[ok] llama_cpp {llama_cpp.__version__} on {platform.platform()} (py {sys.version.split()[0]})")
PY

# 7) Project dirs.
mkdir -p models data runs
green "models/, data/, runs/ ready"

green "setup complete."
echo "    Activate with: conda activate $ENV_NAME"
