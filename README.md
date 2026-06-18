# Quick Start — Inference Smoke Test (SmolVLA & GR00T N1.5)

This guide lets **anyone on Linux**, starting from a machine with **nothing
installed**, verify that two LeRobot Vision-Language-Action (VLA) policies can run
inference successfully. The test loads each model's **open pretrained weights** and
an **open built-in dataset**, runs a single `select_action` call, and prints
`INFERENCE OK`. No robot hardware is needed.

| Model | Checkpoint (auto-download) | Dataset (auto-download) | Device |
| --- | --- | --- | --- |
| **SmolVLA** | `lerobot/smolvla_base` (~0.5B, fully open) | `lerobot/libero` | CPU / CUDA |
| **GR00T N1.5** | `nvidia/GR00T-N1.5-3B` (3B, gated*) | `HuggingFaceVLA/libero` | CUDA only |

\* GR00T's base weights use NVIDIA's **non-commercial** license. They are public
but gated: accept the terms on the model page and log in once with `hf auth login`
before the first run.

---

## Prerequisites

- **Linux x86_64.**
- A GPU machine is recommended. **GR00T requires an NVIDIA (CUDA) GPU** because it
  depends on flash-attention. **SmolVLA also runs on CPU.**
- `wget` or `curl`, plus an internet connection (first run downloads several GB).
- **Python is NOT required up front** — `setup.sh` installs it via Miniconda.

> On [Google Colab](https://colab.research.google.com) with a GPU runtime, conda
> and Python already exist; `setup.sh` will detect conda and skip the Miniconda
> install automatically.

---

## How isolation works (so it won't disturb your system)

`setup.sh` is designed to leave your environment untouched:

- Miniconda is installed into a **self-contained directory** (default `~/miniconda3`).
- It **does not run `conda init`**, so your `~/.bashrc` / shell startup is **not modified**.
  conda is activated only inside the script (and in any shell where you `source` it).
- Everything lives under that one directory; to remove all of it, just delete it.

This is the right isolation model. (You do **not** put conda inside a `venv` — venv
needs Python first, and conda is itself the environment manager.)

---

## Step 1 — One-shot setup

From the folder containing `setup.sh` and `environment.yml`:

```bash
# SmolVLA only (or to set up first, add GR00T later):
bash setup.sh

# Include GR00T (also builds flash-attn, takes longer):
bash setup.sh --groot
```

**What it does:**
- Checks for Miniconda; downloads and installs if missing (~90 MB, may take a minute).
- Creates the `lerobot-vla-test` conda env from `environment.yml` (~10-15 min on GPU, pulls PyTorch + LeRobot).
- (with `--groot`) Builds flash-attn (~5-10 min).

**If it hangs or reports network errors:**

```bash
# Try verbose mode to see what's happening
bash setup.sh -v

# Or set a longer timeout (for slow networks)
export CONDA_PKGS_DIRS=/tmp/conda-pkgs  # temporary cache
bash setup.sh --groot
```

Optional: install Miniconda somewhere else:

```bash
CONDA_DIR=/opt/lerobot-conda bash setup.sh --groot
```

---

## Step 2 — Activate the env in your shell

The script does not touch your shell config, so activate manually:

```bash
source ~/miniconda3/etc/profile.d/conda.sh   # or your CONDA_DIR
conda activate lerobot-vla-test
```

---

## Step 3 — Run the test

```bash
# SmolVLA (CPU works, GPU is faster)
python test.py --model smolvla
```

For **GR00T**, accept the license at
<https://huggingface.co/nvidia/GR00T-N1.5-3B>, log in once, then run:

```bash
hf auth login
python test.py --model groot --device cuda
```

Expected output:

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
# Test a LeRobot-format finetuned checkpoint instead of the base
python test.py --model groot --model-id <user>/groot_libero_10

# Force CPU (SmolVLA only)
python test.py --model smolvla --device cpu
```

---

## What the test actually checks

For each model the script runs the standard LeRobot inference path:

1. `Policy.from_pretrained(model_id)` — downloads + loads the open weights.
2. `make_pre_post_processors(...)` — builds the normalize/tokenize pipeline.
3. `LeRobotDataset(dataset_id)` — pulls one frame of a built-in dataset (a real
   multi-view observation + state, so the input is not faked).
4. `policy.select_action(batch)` — a single forward pass that returns an action.

If step 4 returns an action tensor, inference works end-to-end.

---

## Notes & caveats

- **GR00T is CUDA-only** in LeRobot today (flash-attn dependency); no CPU fallback.
- **GR00T weights are gated.** A 401/403 on load means you have not accepted the
  license or run `hf auth login`.
- **flash-attn build** needs a CUDA compiler (`nvcc`). Cloud GPU images and Colab
  have it. If `bash setup.sh --groot` fails on the flash-attn step, install a
  compiler into the env and retry:
  ```bash
  conda activate lerobot-vla-test
  conda install -c nvidia cuda-nvcc=12.1
  pip install "flash-attn>=2.5.9,<3.0.0" --no-build-isolation
  ```
- The GR00T **base** checkpoint is in NVIDIA's native format. If loading it via
  `GrootPolicy.from_pretrained` is rejected on your LeRobot version, pass a
  **LeRobot-format** checkpoint with `--model-id`, or use the CLI eval path:
  ```bash
  lerobot-eval \
    --policy.path=<user>/groot_libero_10 \
    --env.type=libero --env.task=libero_10 \
    --eval.n_episodes=1 --policy.n_action_steps=50
  ```
- First runs download several GB (weights + dataset shards); allow time and disk.

---

## Files in this folder

- `setup.sh` — one-shot installer (Miniconda + conda env).
- `environment.yml` — conda environment definition.
- `test.py` — the smoke-test script (`--model {smolvla, groot}`).

## References

- LeRobot: <https://github.com/huggingface/lerobot>
- SmolVLA model card: <https://huggingface.co/lerobot/smolvla_base>
- GR00T N1.5 in LeRobot (docs): <https://huggingface.co/docs/lerobot/en/groot>
- GR00T N1.5 base model: <https://huggingface.co/nvidia/GR00T-N1.5-3B>