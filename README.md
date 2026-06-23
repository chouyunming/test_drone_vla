# Quick Start — SmolVLA Inference Smoke Test

This guide lets **anyone on Ubuntu 24.04 LTS**, starting from a machine with
**nothing installed**, verify that the SmolVLA Vision-Language-Action (VLA)
policy can run inference. The test loads the model's **open pretrained weights**
and an **open built-in dataset**, runs a single `select_action` call, and prints
`INFERENCE OK`. No robot hardware is needed.

| Model | Checkpoint (auto-download) | Dataset (auto-download) | Device |
| --- | --- | --- | --- |
| **SmolVLA** | `lerobot/smolvla_base` (~0.5B, fully open) | `lerobot/libero` | CPU / CUDA |

---

## Prerequisites

- **Ubuntu 24.04 LTS, x86_64.**
- A GPU is recommended but not required — SmolVLA also runs on CPU.
- `wget` or `curl`, plus an internet connection (first run downloads several GB).
- **Python is NOT required up front** — `setup.sh` installs it via Miniconda.

---

## Step 1 — Install system packages

These are needed to compile the `evdev` Python package (a lerobot dependency):

```bash
sudo apt-get install -y linux-libc-dev build-essential
```

> **Why?** `evdev` is a Linux input-device library pulled in by lerobot. It
> compiles a C extension that needs Linux kernel headers (`linux/input.h`) and
> a C compiler (`gcc`), neither of which ship in a minimal Ubuntu install.

---

## Step 2 — One-shot setup

Run from the folder containing `setup.sh`:

```bash
bash setup.sh
```

**What it does:**
1. Checks for the required system headers and exits early with a clear message
   if they are missing.
2. Checks for Miniconda; downloads and installs into `./miniconda3` if missing
   (~156 MB).
3. Creates the `lerobot_vla_test` conda env from `environment.yaml` — installs
   Python 3.11, PyTorch (CUDA), torchvision, ffmpeg, and lerobot (~10–20 min
   depending on network speed).

> **First-time downloads are large.** PyTorch alone is several hundred MB.
> Allow 20–30 minutes on a typical connection.

Optional: install Miniconda somewhere else (e.g. a shared `/opt` location):

```bash
CONDA_DIR=/opt/lerobot-conda bash setup.sh
```

---

## Step 3 — Activate the environment

`setup.sh` does **not** modify your shell config (`~/.bashrc`), so activate
manually in each new terminal:

```bash
# Run from inside the project directory:
source miniconda3/etc/profile.d/conda.sh
conda activate lerobot_vla_test
```

If you used a custom `CONDA_DIR`, replace `miniconda3` with that path.

---

## Step 4 — Run the test

```bash
# SmolVLA (CPU works; GPU is faster)
python test.py --model smolvla
```

### Expected output

```
[1/4] Loading policy: lerobot/smolvla_base -> cuda
[2/4] Building pre/post processors
[3/4] Loading built-in dataset: lerobot/libero (first frame)
[4/4] Running select_action

 INFERENCE OK
   model        : smolvla (lerobot/smolvla_base)
   device       : cuda
   action shape : (1, N)
   latency      : ... ms
```

A non-zero exit code with `INFERENCE FAILED` means inference could not run on
that setup.

### Useful overrides

```bash
# Test a finetuned checkpoint instead of the base
python test.py --model smolvla --model-id <user>/smolvla_libero_10

# Force CPU
python test.py --model smolvla --device cpu
```

---

## What the test actually checks

The script runs the standard LeRobot inference path:

1. `Policy.from_pretrained(model_id)` — downloads and loads the weights.
2. `make_pre_post_processors(...)` — builds the normalize/tokenize pipeline.
3. `LeRobotDataset(dataset_id)` — pulls one frame of a built-in dataset (a real
   multi-view observation + state, so the input is not faked).
4. `policy.select_action(batch)` — a single forward pass that returns an action.

If step 4 returns an action tensor, inference works end-to-end.

---

## How isolation works (your system stays untouched)

- Miniconda is installed into `./miniconda3` inside the project directory by
  default (or your `CONDA_DIR`).
- `setup.sh` does **not** run `conda init`, so your `~/.bashrc` or shell
  startup files are **never modified**.
- To remove everything, just delete that one directory:
  ```bash
  rm -rf miniconda3
  ```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `linux/input.h: No such file` | Kernel headers missing | `sudo apt-get install -y linux-libc-dev` |
| `gcc: command not found` | C compiler missing | `sudo apt-get install -y build-essential` |
| `MKL_INTERFACE_LAYER: unbound variable` during activate | MKL activation script + `set -u` | Already fixed in `setup.sh` — update to the latest version if you see this |

---

## Files in this folder

| File | Purpose |
| --- | --- |
| `setup.sh` | One-shot installer: Miniconda + conda env |
| `environment.yaml` | Conda environment definition (Python, PyTorch, lerobot[smolvla]) |
| `test.py` | Inference smoke test (`--model smolvla`) |

---

## References

- LeRobot: <https://github.com/huggingface/lerobot>
- SmolVLA model card: <https://huggingface.co/lerobot/smolvla_base>
