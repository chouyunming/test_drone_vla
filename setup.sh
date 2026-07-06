#!/usr/bin/env bash
# =============================================================================
# setup.sh — One-shot SmolVLA environment installer
#
# Usage:
#   bash setup.sh
#
# Creates smolvla_venv/ in the project directory.
# Prerequisites (must be present before running):
#   - python3 >= 3.12
#   - ffmpeg  (apt install ffmpeg)
#   - NVIDIA driver >= 570.26  (for CUDA 12.8, if using GPU)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/smolvla_venv"

# ---------------------------------------------------------------------------
# 1. Sanity checks
# ---------------------------------------------------------------------------
if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found." >&2; exit 1
fi

PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3,12) else 1)'; then
  echo ">>> Python $PY_VER OK"
else
  echo "ERROR: Python >= 3.12 required (found $PY_VER)." >&2; exit 1
fi

# ---------------------------------------------------------------------------
# 2. Create virtual environment
# ---------------------------------------------------------------------------
if [ ! -d "$VENV_DIR" ]; then
  echo ">>> Creating virtual environment at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
else
  echo ">>> Virtual environment already exists at $VENV_DIR"
fi

PIP="$VENV_DIR/bin/pip"

# ---------------------------------------------------------------------------
# 3. Install requirements
# ---------------------------------------------------------------------------
echo ">>> Upgrading pip"
"$PIP" install --upgrade pip

echo ">>> Installing requirements"
"$PIP" install -r "$SCRIPT_DIR/requirements-smolvla-edge-inference.txt"

echo ">>> Installing lerobot (no-deps: all dependencies already pinned above)"
"$PIP" install --no-deps 'lerobot[smolvla] @ git+https://github.com/huggingface/lerobot.git@main'
"$PIP" install --no-deps 'lerobot[dataset] @ git+https://github.com/huggingface/lerobot.git@main'

# ---------------------------------------------------------------------------
# 4. Done
# ---------------------------------------------------------------------------
cat <<DONE

============================================================
SmolVLA environment ready.

Activate:
  source $VENV_DIR/bin/activate

Run the SmolVLA test:
  python test.py               # auto-detects GPU
  python test.py --device cuda
  python test.py --device cpu
============================================================
DONE
