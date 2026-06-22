# Quick Start — Inference Smoke Test (SmolVLA & GR00T N1.5)

This guide lets **anyone on Ubuntu 24.04 LTS**, starting from a machine with
**nothing installed**, verify that two LeRobot Vision-Language-Action (VLA)
policies can run inference. The test loads each model's **open pretrained
weights** and an **open built-in dataset**, runs a single `select_action` call,
and prints `INFERENCE OK`. No robot hardware is needed.

| Model | Checkpoint (auto-download) | Dataset (auto-download) | Device |
| --- | --- | --- | --- |
| **SmolVLA** | `lerobot/smolvla_base` (~0.5B, fully open) | `lerobot/libero` | CPU / CUDA |
| **GR00T N1.5** | `nvidia/GR00T-N1.5-3B` (3B, gated*) | `HuggingFaceVLA/libero` | CUDA only |

\* GR00T's weights use NVIDIA's **non-commercial** license. They are public but
gated: accept the terms on the model page and log in once with `hf auth login`
before the first run.

---

## Prerequisites

- **Ubuntu 24.04 LTS, x86_64.**
- A GPU is recommended. **GR00T requires an NVIDIA (CUDA) GPU.** SmolVLA also
  runs on CPU.
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
# SmolVLA only:
bash setup.sh

# Include GR00T (also builds flash-attn, takes longer):
bash setup.sh --groot
```

**What it does:**
1. Checks for the required system headers and exits early with a clear message
   if they are missing.
2. Checks for Miniconda; downloads and installs into `./miniconda3` if missing
   (~156 MB).
3. Creates the `lerobot_vla_test` conda env from `environment.yaml` — installs
   Python 3.11, PyTorch (CUDA), torchvision, ffmpeg, and lerobot (~10–20 min
   depending on network speed).
4. *(With `--groot` only)* Builds flash-attn with `--no-build-isolation`
   (~5–10 min), then installs `lerobot[groot]`.

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

For **GR00T**, first accept the license at
<https://huggingface.co/nvidia/GR00T-N1.5-3B>, then:

```bash
hf auth login                              # one-time login
python test.py --model groot --device cuda
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
python test.py --model groot --model-id <user>/groot_libero_10

# Force CPU (SmolVLA only)
python test.py --model smolvla --device cpu
```

---

## What the test actually checks

For each model the script runs the standard LeRobot inference path:

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
| `ModuleNotFoundError: No module named 'torch'` during flash-attn build | flash-attn built in pip's isolated env without torch visible | Use `bash setup.sh --groot` — do not add flash-attn manually to `environment.yaml` |
| `MKL_INTERFACE_LAYER: unbound variable` during activate | MKL activation script + `set -u` | Already fixed in `setup.sh` — update to the latest version if you see this |
| `401 / 403` loading GR00T weights | License not accepted or not logged in | Accept terms at huggingface.co/nvidia/GR00T-N1.5-3B, then `hf auth login` |
| `flash-attn` build fails on `--groot` | CUDA compiler (`nvcc`) missing | `conda install -c nvidia cuda-nvcc=12.1` then retry |

---

## Notes & caveats

- **GR00T is CUDA-only.** The flash-attn dependency has no CPU fallback.
- **GR00T weights are gated.** A 401/403 on load means the license was not
  accepted or `hf auth login` has not been run.
- **First runs download several GB** (model weights + dataset shards). Allow
  time and disk space (~5–10 GB total for both models).
- If loading the GR00T base checkpoint via `GrootPolicy.from_pretrained` is
  rejected on your LeRobot version, use a LeRobot-format finetuned checkpoint
  with `--model-id`, or the CLI eval path:
  ```bash
  lerobot-eval \
    --policy.path=<user>/groot_libero_10 \
    --env.type=libero --env.task=libero_10 \
    --eval.n_episodes=1 --policy.n_action_steps=50
  ```

---

## Files in this folder

| File | Purpose |
| --- | --- |
| `setup.sh` | One-shot installer: Miniconda + conda env + optional flash-attn |
| `environment.yaml` | Conda environment definition (Python, PyTorch, lerobot[smolvla]) |
| `test.py` | Inference smoke test (`--model {smolvla,groot}`) |
| `setup_challenges.txt` | Log of every error hit during initial setup and how each was fixed |

---

## References

- LeRobot: <https://github.com/huggingface/lerobot>
- SmolVLA model card: <https://huggingface.co/lerobot/smolvla_base>
- GR00T N1.5 in LeRobot (docs): <https://huggingface.co/docs/lerobot/en/groot>
- GR00T N1.5 base model: <https://huggingface.co/nvidia/GR00T-N1.5-3B>
