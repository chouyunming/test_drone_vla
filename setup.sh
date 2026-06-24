#!/usr/bin/env bash
# =============================================================================
# setup.sh — One-shot VLA environment installer
#
# Usage:
#   bash setup.sh                    # SmolVLA only (default)
#   bash setup.sh --model smolvla    # SmolVLA only
#   bash setup.sh --model groot      # GR00T-N1.6 only
#
# Environments created:
#   smolvla → conda env  : smolvla   (Python 3.11, CUDA 12.1)
#   groot   → conda env  : Isaac-GR00T     (Python 3.10, CUDA 12.8)
#             pip-cloned : ./Isaac-GR00T/      (NVIDIA/Isaac-GR00T @ n1.6-release)
#
# Optional overrides:
#   CONDA_DIR=/opt/my-conda bash setup.sh --model smolvla
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Argument parsing
# ---------------------------------------------------------------------------
MODEL="smolvla"   # default

for arg in "$@"; do
  case "$arg" in
    --model=*) MODEL="${arg#--model=}" ;;
    --model)   echo "ERROR: --model requires a value (smolvla|groot)"; exit 1 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "Unknown arg: $arg  (try --help)" >&2
      exit 1
      ;;
  esac
done

case "$MODEL" in
  smolvla|groot) ;;
  *) echo "ERROR: --model must be smolvla or groot (got: $MODEL)" >&2; exit 1 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_DIR="${CONDA_DIR:-$SCRIPT_DIR/miniconda3}"

# ---------------------------------------------------------------------------
# 1. Shared helper: ensure conda is available
# ---------------------------------------------------------------------------
_ensure_conda() {
  if command -v conda >/dev/null 2>&1; then
    echo ">>> conda found on PATH, skipping Miniconda install"
    source "$(conda info --base)/etc/profile.d/conda.sh"
  else
    if [ ! -x "$CONDA_DIR/bin/conda" ]; then
      echo ">>> Installing Miniconda into $CONDA_DIR"
      url="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
      tmp_sh="$(mktemp --suffix=.sh)"
      if command -v wget >/dev/null 2>&1; then
        wget --show-progress "$url" -O "$tmp_sh"
      elif command -v curl >/dev/null 2>&1; then
        curl -fL --progress-bar "$url" -o "$tmp_sh"
      else
        echo "ERROR: need wget or curl to download Miniconda." >&2
        exit 1
      fi
      bash "$tmp_sh" -b -p "$CONDA_DIR"
      rm -f "$tmp_sh"
    else
      echo ">>> Found existing Miniconda at $CONDA_DIR"
    fi
    source "$CONDA_DIR/etc/profile.d/conda.sh"
  fi
}

# ---------------------------------------------------------------------------
# 2. SmolVLA environment (conda, Python 3.11, CUDA 12.1)
# ---------------------------------------------------------------------------
_setup_smolvla() {
  echo ""
  echo "============================================================"
  echo " Setting up SmolVLA environment (smolvla)"
  echo "============================================================"

  # 0. System headers check (evdev needs linux/input.h)
  if [ ! -f /usr/include/linux/input.h ]; then
    echo "ERROR: Kernel headers missing. Run first:" >&2
    echo "  sudo apt-get install -y linux-libc-dev build-essential" >&2
    exit 1
  fi

  _ensure_conda

  local ENV_NAME="smolvla"
  if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo ">>> Env '$ENV_NAME' already exists, skipping create"
  else
    echo ">>> Creating env '$ENV_NAME' from environment.yaml"
    conda env create -f "$SCRIPT_DIR/environment.yaml"
  fi

  set +u; conda activate "$ENV_NAME"; set -u

  cat <<DONE

============================================================
SmolVLA environment ready: $ENV_NAME

Activate in a new terminal:
  source "${CONDA_DIR}/etc/profile.d/conda.sh"
  conda activate $ENV_NAME

Run the SmolVLA test:
  python test.py --model smolvla
============================================================
DONE
}

# ---------------------------------------------------------------------------
# 3. GR00T-N1.6 environment (conda, Python 3.10, CUDA 12.8)
# ---------------------------------------------------------------------------
_setup_groot() {
  echo ""
  echo "============================================================"
  echo " Setting up GR00T-N1.6 environment (Isaac-GR00T)"
  echo "============================================================"

  # --- 3a. System deps ---
  echo ">>> Checking / installing system dependencies"
  local MISSING_PKGS=""
  for pkg in ffmpeg git-lfs; do
    if ! command -v "$pkg" >/dev/null 2>&1; then
      MISSING_PKGS="$MISSING_PKGS $pkg"
    fi
  done
  if ! dpkg -s libaio-dev &>/dev/null 2>&1; then
    MISSING_PKGS="$MISSING_PKGS libaio-dev"
  fi
  if [ -n "$MISSING_PKGS" ]; then
    echo ">>> Installing:$MISSING_PKGS"
    sudo apt-get install -y --no-install-recommends $MISSING_PKGS
  else
    echo ">>> System deps already present"
  fi

  git lfs install --skip-repo 2>/dev/null || true

  # --- 3b. Conda env ---
  _ensure_conda

  local ENV_NAME="Isaac-GR00T"
  if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo ">>> Env '$ENV_NAME' already exists, skipping create"
  else
    echo ">>> Creating env '$ENV_NAME' from environment_groot.yaml"
    conda env create -f "$SCRIPT_DIR/environment_groot.yaml"
  fi

  set +u; conda activate "$ENV_NAME"; set -u

  # --- 3c. PyTorch with CUDA 12.8 support ---
  # GR00T requires torch==2.7.1 built against CUDA 12.8.
  # Install from the pytorch-cu128 index before gr00t so that pip sees the
  # CUDA-enabled wheel and won't pull the CPU-only build from PyPI later.
  if python -c "import torch; assert torch.__version__.startswith('2.7')" 2>/dev/null; then
    echo ">>> torch 2.7.x already installed, skipping"
  else
    echo ">>> Installing torch 2.7.1+cu128 / torchvision 0.22.1+cu128"
    pip install \
      "torch==2.7.1+cu128" \
      "torchvision==0.22.1+cu128" \
      --index-url https://download.pytorch.org/whl/cu128
  fi

  # --- 3d. flash-attn prebuilt wheel (Python 3.10 + CUDA 12 + torch 2.7) ---
  if python -c "import flash_attn" 2>/dev/null; then
    echo ">>> flash-attn already installed, skipping"
  else
    echo ">>> Installing flash-attn 2.7.4.post1 (prebuilt wheel)"
    FLASH_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
    pip install "$FLASH_WHEEL"
  fi

  # --- 3e. Clone Isaac-GR00T at n1.6-release ---
  local GROOT_DIR="$SCRIPT_DIR/Isaac-GR00T"
  if [ -d "$GROOT_DIR/.git" ]; then
    echo ">>> Isaac-GR00T already cloned at $GROOT_DIR"
  else
    echo ">>> Cloning NVIDIA/Isaac-GR00T @ n1.6-release into $GROOT_DIR"
    # --depth 1 keeps it fast; submodules bring in groot_infra
    git clone \
      --branch n1.6-release \
      --depth 1 \
      --recurse-submodules \
      --shallow-submodules \
      https://github.com/NVIDIA/Isaac-GR00T.git \
      "$GROOT_DIR"
  fi

  # --- 3f. Install gr00t package + remaining dependencies ---
  # DS_BUILD_OPS=0 prevents deepspeed from trying to JIT-compile CUDA ops at
  # install time (the user's nvcc may differ from the runtime CUDA version).
  # --extra-index-url makes nvidia-pypi and pytorch-cu128 available for
  # any dependency that needs them (e.g. tensorrt from nvidia-pypi).
  if python -c "import gr00t" 2>/dev/null; then
    echo ">>> gr00t already installed, skipping"
  else
    echo ">>> Installing gr00t and remaining dependencies (this may take 10–20 min)"
    DS_BUILD_OPS=0 pip install \
      -e "$GROOT_DIR" \
      --extra-index-url https://download.pytorch.org/whl/cu128 \
      --extra-index-url https://pypi.nvidia.com
  fi

  cat <<DONE

============================================================
GR00T-N1.6 environment ready: $ENV_NAME
Isaac-GR00T cloned to    : $GROOT_DIR

Activate in a new terminal:
  source "${CONDA_DIR}/etc/profile.d/conda.sh"
  conda activate $ENV_NAME

Run the GR00T smoke test (downloads ~12 GB on first run):
  python test.py --model groot

Override checkpoint / embodiment:
  python test.py --model groot \\
      --model-id nvidia/GR00T-N1.6-3B \\
      --embodiment-tag gr1

NOTE: The GR00T model is released under the NVIDIA OneWay
Non-Commercial License. Accept it on HuggingFace before
downloading:  https://huggingface.co/nvidia/GR00T-N1.6-DROID
============================================================
DONE
}

# ---------------------------------------------------------------------------
# 4. Dispatch
# ---------------------------------------------------------------------------
case "$MODEL" in
  smolvla)
    _setup_smolvla
    ;;
  groot)
    _setup_groot
    ;;
esac