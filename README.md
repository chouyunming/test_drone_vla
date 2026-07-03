# VLA Inference Test — SmolVLA
## Model Card
| Model | Conda env | Checkpoint (auto-download) | Dataset / Obs | Device |
| --- | --- | --- | --- | --- |
| **SmolVLA** | `smolvla` | `lerobot/smolvla_base` (~0.5B) | `lerobot/libero` (real frame) | CPU / CUDA |

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

> A GPU with ≥ 8 GB VRAM is sufficient. A GPU is recommended but not required — SmolVLA can run on CPU.

---

## Software Version

| | SmolVLA (`smolvla` env) |
| --- | --- |
| **Python** | 3.10 |
| **PyTorch** | 2.10.0 |
| **torchvision** | 0.25.0 |
| **torchcodec** | 0.10.0 |
| **CUDA toolkit** | 12.1 (via PyTorch cu121 wheel) |
| **lerobot** | 0.4.4 (`lerobot[smolvla]`) |

> **Note:** you do not need to install any of these manually.
> `setup.sh` creates an isolated Conda environment and installs the correct
> versions for you automatically.

---

## Prerequisites

- **Ubuntu 24.04 LTS, x86_64.**
- A GPU is recommended but not required.
- `wget` or `curl`, plus an internet connection (first run downloads several GB).
- **Python is NOT required up front** — `setup.sh` installs it via Miniconda.

---

## Step 1 — One-shot setup

Run from the folder containing `setup.sh`:

```bash
# Clone this repo
git clone https://github.com/chouyunming/test_drone_vla.git
cd test_drone_vla

bash setup.sh
```

**What setup does:**

| Step | Action |
| --- | --- |
| Miniconda | installs into `./miniconda3` if absent |
| Conda env | creates `smolvla` from `environment.yaml` (Python 3.10, CUDA 12.1) |
| PyTorch | installed by conda |

> **First-time download is large.** SmolVLA: ~10–20 min depending on connection speed.

---

## Step 2 — Activate the environment

`setup.sh` does **not** modify `~/.bashrc`. Activate manually in each new terminal:

```bash
source miniconda3/etc/profile.d/conda.sh
conda activate smolvla
```

---

## Step 3 — Run the test

```bash
# SmolVLA (CPU works; GPU is faster)
python test.py --model smolvla

# Force CPU
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

## How isolation works (your system stays untouched)

- Miniconda installs into `./miniconda3` inside the project directory by
  default (or your `CONDA_DIR`).
- `setup.sh` does **not** run `conda init`, so `~/.bashrc` is never modified.
- To remove everything, delete the Miniconda directory:
  ```bash
  rm -rf miniconda3
  ```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Partial `./miniconda3` blocks reinstall | Interrupted previous run | `rm -rf ./miniconda3` then re-run |

---

## Files in this folder

| File | Purpose |
| --- | --- |
| `setup.sh` | One-shot installer: Miniconda + `smolvla` conda env |
| `environment.yaml` | Conda env for SmolVLA (Python 3.10, CUDA 12.8, lerobot) |
| `test.py` | Inference smoke test (`--model smolvla`) |
| `requirements-smolvla-edge-inference.txt` | Pinned pip deps for edge/CPU inference deployment |

---

## References

- LeRobot: <https://github.com/huggingface/lerobot>
- SmolVLA model card: <https://huggingface.co/lerobot/smolvla_base>
