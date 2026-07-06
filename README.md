# VLA Inference Test — SmolVLA

## Model Card

| Model | Checkpoint (auto-download) | Dataset / Obs | Device |
| --- | --- | --- | --- |
| **SmolVLA** | `lerobot/smolvla_base` (~0.5B) | `lerobot/libero` (real frame) | CPU / CUDA |

---

## Tested Hardware

| Component | Details |
| --- | --- |
| **OS** | Ubuntu 24.04.4 LTS (Noble Numbat), x86\_64 |
| **GPU** | NVIDIA RTX PRO 6000 Blackwell, 96 GB VRAM |
| **NVIDIA Driver** | 580.159.03 |
| **CUDA (driver-reported)** | 13.0 |
| **GCC** | 11.5.0 (Ubuntu 11.5.0-1ubuntu1\~24.04.1) |

> A GPU with ≥ 8 GB VRAM is sufficient. A GPU is recommended but not required — SmolVLA can run on CPU.

---

## Software Versions

| | SmolVLA |
| --- | --- |
| **Python** | 3.12 |
| **PyTorch** | 2.11.0+cu128 (or CPU-only) |
| **torchvision** | 0.26.0+cu128 (or CPU-only) |
| **CUDA toolkit** | 12.8 (optional, GPU only) |
| **lerobot** | 0.5.2 (`lerobot[smolvla]`) |

> **Note:** you do not need to install any of these manually.
> `setup.sh` creates an isolated Python virtual environment and installs the correct versions automatically.

---

## Prerequisites

- **Ubuntu 24.04 LTS, x86_64.**
- **Python ≥ 3.12** must already be installed (`python3 --version`).
- **ffmpeg** must be installed: `sudo apt install ffmpeg`
- **NVIDIA driver ≥ 570.26** if using GPU (CUDA 12.8 compatibility).
- `wget` or `curl`, plus an internet connection (first run downloads several GB).

---

## Step 1 — One-shot setup

Run from the folder containing `setup.sh`:

```bash
git clone https://github.com/chouyunming/test_drone_vla.git
cd test_drone_vla

bash setup.sh
```

**What setup does:**

| Step | Action |
| --- | --- |
| Sanity check | Verifies Python ≥ 3.12 |
| Virtual env | Creates `smolvla_venv/` via `python3 -m venv` |
| Dependencies | `pip install -r requirements-smolvla-edge-inference.txt` (CPU-only torch by default; CUDA 12.8 torch available) |
| lerobot | `pip install --no-deps lerobot[smolvla]==0.5.2` |

> **First-time download is large.** Expect ~10–20 min depending on connection speed.

---

## Step 2 — Activate the environment

```bash
source smolvla_venv/bin/activate
```

---

## Step 3 — Run the test

```bash
# Auto-detects GPU; falls back to CPU if unavailable
python test.py --model smolvla

# Explicit device selection
python test.py --model smolvla --device cuda
python test.py --model smolvla --device cpu
```

### Expected output

```
[1/4] Loading SmolVLA policy: lerobot/smolvla_base → cuda
[2/4] Building pre/post processors
[3/4] Loading dataset: lerobot/libero (first frame)
[4/4] Running select_action

✓ INFERENCE OK
   model        : smolvla (lerobot/smolvla_base)
   device       : cuda
   action shape : (1, N)
   latency      : ... ms
```

A non-zero exit code with `✗ INFERENCE FAILED` means inference could not run on that setup.

---

## How isolation works (your system stays untouched)

- `setup.sh` creates `smolvla_venv/` inside the project directory — nothing is installed system-wide.
- To remove everything, delete the venv:
  ```bash
  rm -rf smolvla_venv
  ```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `python3: command not found` | Python 3.12 not installed | `sudo apt install python3.12 python3.12-venv` |
| `ffmpeg: command not found` | ffmpeg missing | `sudo apt install ffmpeg` |
| CUDA errors at runtime | Driver / toolkit mismatch | Ensure NVIDIA driver ≥ 570.26; run with `--device cpu` to verify CPU path works |
| Partial `smolvla_venv` blocks reinstall | Interrupted previous run | `rm -rf smolvla_venv` then re-run `bash setup.sh` |

---

## Files in this folder

| File | Purpose |
| --- | --- |
| `setup.sh` | One-shot installer: creates `smolvla_venv` and installs all dependencies |
| `requirements-smolvla-edge-inference.txt` | Pinned pip deps (CPU-only torch by default, CUDA 12.8 torch optional + lerobot transitive deps) |
| `test.py` | Inference smoke test (`--model smolvla [--device cpu\|cuda]`) |

---

## References

- LeRobot: <https://github.com/huggingface/lerobot>
- SmolVLA model card: <https://huggingface.co/lerobot/smolvla_base>
