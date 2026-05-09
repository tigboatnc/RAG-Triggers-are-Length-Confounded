#!/usr/bin/env bash
# v3 bootstrap. Single supported path: Linux + NVIDIA CUDA.
#
# Strategy:
#   - Verify Linux + nvidia-smi. Mac/Metal deferred (see _PLAN.md).
#   - Build / update the conda env from environment.yml.
#   - Install torch from the cu124 wheel index (kept here, not in environment.yml,
#     because the index URL is platform-specific).
#   - Smoke-import torch + transformers + bitsandbytes; confirm CUDA visible.
#
# Usage:
#   bash scripts/setup.sh

set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

ENV_NAME="recollection"
# torch pinned: pip's "latest" was 2.11.0+cu130 which needs driver >=13.0.
# The A40 box runs driver 565.57.01 (CUDA 12.7). 2.6.x has real cu124 wheels
# that run fine against driver 12.7 (CUDA is forward-compatible at runtime).
TORCH_VERSION="${TORCH_VERSION:-2.6.0}"
TORCH_CUDA="${TORCH_CUDA:-cu124}"
TORCH_INDEX="https://download.pytorch.org/whl/${TORCH_CUDA}"

green() { printf "\033[32m[ok]\033[0m %s\n" "$1"; }
yellow() { printf "\033[33m[..]\033[0m %s\n" "$1"; }
red() { printf "\033[31m[!!]\033[0m %s\n" "$1" >&2; }

# 1) Platform gate. v3 is CUDA-only by design; fail loudly otherwise.
OS_NAME="$(uname -s)"
ARCH_NAME="$(uname -m)"
if [[ "$OS_NAME" != "Linux" ]]; then
  red "v3 supports Linux + NVIDIA only (got OS=$OS_NAME). Mac/Metal deferred — see _PLAN.md."
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  red "nvidia-smi missing or non-functional. v3 requires a CUDA GPU."
  exit 1
fi
green "OS=$OS_NAME arch=$ARCH_NAME backend=cuda"

# 2) conda on PATH. Probe order: PATH, ~/miniconda3, ~/miniforge3, /opt/miniforge3
#    (last is for vast.ai-style boxes where miniforge ships in /opt).
if ! command -v conda >/dev/null 2>&1; then
  if [[ -x "$HOME/miniconda3/bin/conda" ]]; then
    export PATH="$HOME/miniconda3/bin:$PATH"
  elif [[ -x "$HOME/miniforge3/bin/conda" ]]; then
    export PATH="$HOME/miniforge3/bin:$PATH"
  elif [[ -x "/opt/miniforge3/bin/conda" ]]; then
    export PATH="/opt/miniforge3/bin:$PATH"
  else
    red "conda not found. Install Miniconda/Miniforge and re-run."
    exit 1
  fi
fi
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
green "conda $(conda --version | awk '{print $2}') (base=$CONDA_BASE)"

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

# 4) torch from the cu124 wheel index. Kept out of environment.yml because
#    the index URL is platform-specific.
NEED_INSTALL=1
if pip show torch >/dev/null 2>&1; then
  CURRENT_TORCH="$(pip show torch | awk '/^Version:/ {print $2}')"
  if [[ "$CURRENT_TORCH" == "${TORCH_VERSION}+${TORCH_CUDA}" || "$CURRENT_TORCH" == "$TORCH_VERSION" ]]; then
    green "torch $CURRENT_TORCH already installed"
    NEED_INSTALL=0
  else
    yellow "torch $CURRENT_TORCH -> ${TORCH_VERSION}+${TORCH_CUDA} (driver-compat reinstall)"
  fi
fi
if [[ "$NEED_INSTALL" -eq 1 ]]; then
  yellow "Installing torch==$TORCH_VERSION (${TORCH_CUDA} wheel)"
  pip install --no-cache-dir --force-reinstall \
    "torch==$TORCH_VERSION" \
    --index-url "$TORCH_INDEX"
fi

# 5) Smoke import: torch sees CUDA; transformers + bitsandbytes import clean.
python - <<'PY'
import platform, sys
import torch, transformers, accelerate
print(f"[ok] torch {torch.__version__} cuda_available={torch.cuda.is_available()} "
      f"device_count={torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"[ok] device 0: {torch.cuda.get_device_name(0)}")
print(f"[ok] transformers {transformers.__version__} / accelerate {accelerate.__version__}")
try:
    import bitsandbytes as bnb
    print(f"[ok] bitsandbytes {bnb.__version__}")
except Exception as e:
    print(f"[!!] bitsandbytes import failed: {e!r}")
    sys.exit(1)
print(f"[ok] platform={platform.platform()} py={sys.version.split()[0]}")
PY

# 6) Project dirs.
mkdir -p models data runs figs
green "models/, data/, runs/, figs/ ready"

# 7) Lockfile capture for reproducibility.
pip freeze > runs/env_lockfile.txt
green "runs/env_lockfile.txt written ($(wc -l < runs/env_lockfile.txt) packages)"

green "setup complete."
echo "    Activate with: conda activate $ENV_NAME"
