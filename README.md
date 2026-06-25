# VLA Inference Test — SmolVLA & GR00T-N1.6
## Model Card
| Model | Conda env | Checkpoint (auto-download) | Dataset / Obs | Device |
| --- | --- | --- | --- | --- |
| **SmolVLA** | `smolvla` | `lerobot/smolvla_base` (~0.5B) | `lerobot/libero` (real frame) | CPU / CUDA |
| **GR00T-N1.6** | `Isaac-GR00T` | `nvidia/GR00T-N1.6-DROID` (~3B) | synthetic tensor | CUDA required |

---

## Tested Hardware

The configuration below was used to develop and validate this repo:

| Component | Details |
| --- | --- |
| **OS** | Ubuntu 24.04.4 LTS (Noble Numbat), x86\_64 |
| **GPU** | NVIDIA RTX PRO 6000 Blackwell, 96 GB VRAM |
| **NVIDIA Driver** | 580.159.03 |
| **CUDA (driver-reported)** | 13.0 |
| **GCC** | 11.5.0 (Ubuntu 11.5.0-1ubuntu1\~24.04.1) |

> A GPU with ≥ 8 GB VRAM is sufficient for both models.

---

## Software Version

These are the exact versions pinned per environment — useful if you hit a
dependency conflict and need to know what "known-good" looks like:

| | SmolVLA (`smolvla` env) | GR00T-N1.6 (`Isaac-GR00T` env) |
| --- | --- | --- |
| **Python** | 3.10 | 3.10 |
| **PyTorch** | ≥ 2.2.1, < 2.11.0 | 2.7.1+cu128 |
| **torchvision** | ≥ 0.21.0, < 0.26.0 | 0.22.1+cu128 |
| **torchcodec** | ≥ 0.2.1, < 0.11.0 | not used |
| **CUDA toolkit** | 12.8 (via PyTorch cu128 wheel) | 12.8 (via PyTorch cu128 wheel) |
| **flash-attn** | not used | 2.7.4.post1 (prebuilt wheel, Python 3.10 + CUDA 12.8) |
| **lerobot** | 0.4.4 (`lerobot[smolvla]`) | not used |
| **Isaac-GR00T** | not used | `n1.6-release` branch (gr00t 0.1.0) |

> **Note:** you do not need to install any of these manually.
> `setup.sh` creates isolated Conda environments and installs the correct
> versions for you automatically.

---

## Prerequisites

- **Ubuntu 24.04 LTS, x86_64.**
- A GPU is recommended but not required for SmolVLA. **GR00T-N1.6 requires a
  CUDA GPU** — it will refuse to run on CPU.
- `wget` or `curl`, plus an internet connection (first run downloads several GB).
- **Python is NOT required up front** — `setup.sh` installs it via Miniconda.

---

## Step 1 — One-shot setup

Run from the folder containing `setup.sh`. Pass `--model` to choose which
environment to install:

```bash
# Clone this repo
git clone https://github.com/chouyunming/test_drone_vla.git
cd test_drone_vla

# SmolVLA only (default)
bash setup.sh

# GR00T-N1.6 only
bash setup.sh --model groot
```

**What each mode does:**

| Step | SmolVLA | GR00T-N1.6 |
| --- | --- | --- |
| Miniconda | installs into `./miniconda3` if absent | same |
| Conda env | creates `smolvla` from `environment.yaml` (Python 3.10, CUDA 12.8) | creates `Isaac-GR00T` from `environment_groot.yaml` (Python 3.10) |
| PyTorch | installed by conda | `torch==2.7.1+cu128` from pytorch-cu128 index |
| flash-attn | n/a | prebuilt wheel (Python 3.10 + CUDA 12) |
| GR00T package | n/a | clones `NVIDIA/Isaac-GR00T@n1.6-release`, installs with `pip install -e` |

> **First-time downloads are large.** SmolVLA: ~10–20 min. GR00T: ~12 GB on
> first run, allow 20–40 min depending on connection speed.

---

## Step 2 — Activate the environment

`setup.sh` does **not** modify `~/.bashrc`. Activate manually in each new
terminal:

```bash
# SmolVLA
source miniconda3/etc/profile.d/conda.sh
conda activate smolvla

# GR00T-N1.6
source miniconda3/etc/profile.d/conda.sh
conda activate Isaac-GR00T
```

---

## Step 3 — Run the test

```bash
# SmolVLA (CPU works; GPU is faster)
python test.py --model smolvla

# GR00T-N1.6 (downloads ~12 GB on first run)
python test.py --model groot
```

### Expected output — SmolVLA

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

### Expected output — GR00T-N1.6

```
[1/3] Loading GR00T policy: nvidia/GR00T-N1.6-DROID  embodiment=oxe_droid → cuda
[2/3] Building synthetic observation
[3/3] Running get_action

✓ INFERENCE OK
   model          : groot (nvidia/GR00T-N1.6-DROID)
   embodiment     : oxe_droid
   device         : cuda
   action shapes  : {'joint_position': (1, T, 6), 'gripper_position': (1, T, 1)}
   latency        : ... ms
```

A non-zero exit code with `✗ INFERENCE FAILED` means inference could not run
on that setup.

### Test on CPU (SmolVLA-Only)

```bash
# SmolVLA — force CPU (GR00T-N1.6 does not support CPU)
python test.py --model smolvla --device cpu
```

## How isolation works (your system stays untouched)

- Miniconda installs into `./miniconda3` inside the project directory by
  default (or your `CONDA_DIR`).
- `setup.sh` does **not** run `conda init`, so `~/.bashrc` is never modified.
- To remove everything, delete the Miniconda directory and the Isaac-GR00T
  clone:
  ```bash
  rm -rf miniconda3 Isaac-GR00T
  ```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `ERROR: GR00T-N1.6 requires a CUDA-capable GPU` | No NVIDIA GPU / driver | Install the NVIDIA driver; GR00T-N1.6 does not support CPU |
| Partial `./miniconda3` blocks reinstall | Interrupted previous run | `rm -rf ./miniconda3` then re-run |

---

## Files in this folder

| File | Purpose |
| --- | --- |
| `setup.sh` | One-shot installer: Miniconda + conda env (`--model smolvla\|groot`) |
| `environment.yaml` | Conda env for SmolVLA (Python 3.10, CUDA 12.8, lerobot) |
| `environment_groot.yaml` | Conda env base for GR00T-N1.6 (Python 3.10; torch/flash-attn/gr00t added by setup.sh) |
| `test.py` | Inference smoke test (`--model smolvla\|groot`) |

---

## References

- LeRobot: <https://github.com/huggingface/lerobot>
- SmolVLA model card: <https://huggingface.co/lerobot/smolvla_base>
- Isaac-GR00T repo: <https://github.com/NVIDIA/Isaac-GR00T>
- GR00T-N1.6-DROID model card: <https://huggingface.co/nvidia/GR00T-N1.6-DROID>
