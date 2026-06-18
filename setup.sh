#!/usr/bin/env bash
# Usage:
#   bash setup.sh            # SmolVLA only
#   bash setup.sh --groot    # also build flash-attn for GR00T

set -euo pipefail

CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
ENV_NAME="lerobot-vla-test"
INSTALL_FLASH_ATTN=0

for arg in "$@"; do
  case "$arg" in
    --groot) INSTALL_FLASH_ATTN=1 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown arg: $arg (try --help)"; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. Miniconda
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

# 2. Conda env
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo ">>> Env '$ENV_NAME' already exists, skipping create"
else
  echo ">>> Creating env '$ENV_NAME' from environment.yaml"
  conda env create -f "$SCRIPT_DIR/environment.yaml"
fi

conda activate "$ENV_NAME"

# 3. flash-attn (GR00T only)
if [ "$INSTALL_FLASH_ATTN" -eq 1 ]; then
  echo ">>> Installing flash-attn"
  pip install "flash-attn>=2.5.9,<3.0.0" --no-build-isolation
  python -c "import flash_attn; print('flash-attn', flash_attn.__version__, 'OK')"
fi

# Done
cat <<DONE

============================================================
Environment ready: $ENV_NAME

To use it in a NEW terminal, run:
  source "$CONDA_DIR/etc/profile.d/conda.sh"
  conda activate $ENV_NAME

Run the SmolVLA test:
  python test.py --model smolvla
DONE

if [ "$INSTALL_FLASH_ATTN" -eq 1 ]; then
  cat <<DONE

For GR00T (CUDA only):
  hf auth login
  python test.py --model groot --device cuda
============================================================
DONE
else
  cat <<DONE

For GR00T, re-run with:
  bash setup.sh --groot
============================================================
DONE
fi