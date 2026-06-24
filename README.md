# VLA Inference Smoke Test — SmolVLA & GR00T-N1.6

This repo lets **anyone on Ubuntu 24.04 LTS**, starting from a machine with
**nothing installed**, verify that a Vision-Language-Action (VLA) policy can
run inference. The test loads pretrained weights and a dataset, runs a single
forward pass, and prints `INFERENCE OK`. No robot hardware required.

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

## Software Versions at a Glance

These are the exact versions pinned per environment — useful if you hit a
dependency conflict and need to know what "known-good" looks like:

| | SmolVLA (`smolvla` env) | GR00T-N1.6 (`Isaac-GR00T` env) |
| --- | --- | --- |
| **Python** | 3.11 | 3.10 |
| **PyTorch** | ≥ 2.2.1, < 2.6.0 | 2.7.1+cu128 |
| **torchvision** | ≥ 0.17.0, < 0.21.0 | bundled with torch |
| **CUDA toolkit** | 12.1 (via conda) | 12.8 (via PyTorch cu128 wheel) |
| **flash-attn** | not used | prebuilt wheel (Python 3.10 + CUDA 12) |
| **lerobot** | latest (`lerobot[smolvla]`) | not used |
| **Isaac-GR00T** | not used | `n1.6-release` branch |

> **Beginner tip:** you do not need to install any of these manually.
> `setup.sh` creates isolated Conda environments and installs the correct
> versions for you automatically.

---

## Prerequisites

- **Ubuntu 24.04 LTS, x86_64.**
- A GPU is recommended but not required for SmolVLA. **GR00T-N1.6 requires a
  CUDA GPU** — it will refuse to run on CPU.
- `wget` or `curl`, plus an internet connection (first run downloads several GB).
- **Python is NOT required up front** — `setup.sh` installs it via Miniconda.

### GR00T — NVIDIA license

GR00T-N1.6 is released under the **NVIDIA OneWay Non-Commercial License**.
Accept it on HuggingFace before the first run:
<https://huggingface.co/nvidia/GR00T-N1.6-DROID>

---

## Step 1 — One-shot setup

Run from the folder containing `setup.sh`. Pass `--model` to choose which
environment to install:

```bash
# SmolVLA only (default)
bash setup.sh

# GR00T-N1.6 only
bash setup.sh --model groot
```

**What each mode does:**

| Step | SmolVLA | GR00T-N1.6 |
| --- | --- | --- |
| Miniconda | installs into `./miniconda3` if absent | same |
| Conda env | creates `smolvla` from `environment.yaml` (Python 3.11, CUDA 12.1) | creates `Isaac-GR00T` from `environment_groot.yaml` (Python 3.10) |
| PyTorch | installed by conda | `torch==2.7.1+cu128` from pytorch-cu128 index |
| flash-attn | n/a | prebuilt wheel (Python 3.10 + CUDA 12) |
| GR00T package | n/a | clones `NVIDIA/Isaac-GR00T@n1.6-release`, installs with `pip install -e` |

> **First-time downloads are large.** SmolVLA: ~10–20 min. GR00T: ~12 GB on
> first run, allow 20–40 min depending on connection speed.

Optional: install Miniconda into a custom location:

```bash
CONDA_DIR=/opt/my-conda bash setup.sh --model smolvla
```

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

If you used a custom `CONDA_DIR`, replace `miniconda3` with that path.

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

### Useful overrides

```bash
# SmolVLA — test a finetuned checkpoint
python test.py --model smolvla --model-id <user>/smolvla_libero_10

# SmolVLA — force CPU (GR00T-N1.6 does not support CPU)
python test.py --model smolvla --device cpu

# GR00T — use the base model + gr1 embodiment (CUDA required)
python test.py --model groot \
    --model-id nvidia/GR00T-N1.6-3B \
    --embodiment-tag gr1
```

---

## What the test actually checks

**SmolVLA** runs the standard LeRobot inference path:

1. `SmolVLAPolicy.from_pretrained(model_id)` — downloads and loads weights.
2. `make_pre_post_processors(...)` — builds the normalize/tokenize pipeline.
3. `LeRobotDataset(dataset_id)` — pulls one real frame from a built-in dataset.
4. `policy.select_action(batch)` — single forward pass returning an action tensor.

**GR00T-N1.6** runs the Isaac-GR00T inference path:

1. `Gr00tPolicy(model_path, embodiment_tag, ...)` — downloads and loads weights.
2. Synthetic observation dict is built to match the checkpoint's modality config
   (video: `(B, T, 224, 224, 3)` uint8; state: `(B, T, D)` float32; language string).
3. `policy.get_action(obs)` — single forward pass returning an action dict.

---

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
| `PackagesNotFoundError: torchvision` | Wrong version constraint in environment.yaml | Already fixed — update to the latest `environment.yaml` |
| `MKL_INTERFACE_LAYER: unbound variable` during activate | MKL activation script + `set -u` | Already fixed in `setup.sh` — update to the latest version |
| `ERROR: GR00T-N1.6 requires a CUDA-capable GPU` | No NVIDIA GPU / driver | Install the NVIDIA driver; GR00T-N1.6 does not support CPU |
| `401 Unauthorized` on GR00T download | NVIDIA license not accepted | Accept at <https://huggingface.co/nvidia/GR00T-N1.6-DROID> |
| Partial `./miniconda3` blocks reinstall | Interrupted previous run | `rm -rf ./miniconda3` then re-run |

---

## Files in this folder

| File | Purpose |
| --- | --- |
| `setup.sh` | One-shot installer: Miniconda + conda env (`--model smolvla\|groot`) |
| `environment.yaml` | Conda env for SmolVLA (Python 3.11, CUDA 12.1, lerobot) |
| `environment_groot.yaml` | Conda env base for GR00T-N1.6 (Python 3.10; torch/flash-attn/gr00t added by setup.sh) |
| `test.py` | Inference smoke test (`--model smolvla\|groot`) |

---

## References

- LeRobot: <https://github.com/huggingface/lerobot>
- SmolVLA model card: <https://huggingface.co/lerobot/smolvla_base>
- Isaac-GR00T repo: <https://github.com/NVIDIA/Isaac-GR00T>
- GR00T-N1.6-DROID model card: <https://huggingface.co/nvidia/GR00T-N1.6-DROID>
