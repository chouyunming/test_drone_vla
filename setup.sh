#!/usr/bin/env bash
# Usage:
#   bash setup.sh            # SmolVLA only

set -euo pipefail

ENV_NAME="lerobot_vla_test"

for arg in "$@"; do
  case "$arg" in
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown arg: $arg (try --help)"; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_DIR="${CONDA_DIR:-$SCRIPT_DIR/miniconda3}"

# 0. System deps check (evdev needs linux/input.h to build from source)
if [ ! -f /usr/include/linux/input.h ]; then
  echo "ERROR: Kernel headers missing. Run first:" >&2
  echo "  sudo apt-get install -y linux-libc-dev" >&2
  exit 1
fi

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

set +u; conda activate "$ENV_NAME"; set -u

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

cat <<DONE
============================================================
DONE