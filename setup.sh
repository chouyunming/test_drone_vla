#!/usr/bin/env bash
# =============================================================================
# setup.sh — One-shot SmolVLA environment installer
#
# Usage:
#   bash setup.sh
#
# Environment created:
#   smolvla → conda env : smolvla (Python 3.10, CUDA 12.8)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_DIR="${CONDA_DIR:-$SCRIPT_DIR/miniconda3}"

# ---------------------------------------------------------------------------
# 1. Ensure conda is available
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
# 2. SmolVLA environment (conda, Python 3.10, CUDA 12.8)
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " Setting up SmolVLA environment (smolvla)"
echo "============================================================"

# System headers check (evdev needs linux/input.h)
if [ ! -f /usr/include/linux/input.h ]; then
  echo ">>> Installing kernel headers and build tools"
  sudo apt-get install -y --no-install-recommends linux-libc-dev build-essential
fi

_ensure_conda

ENV_NAME="smolvla"
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
