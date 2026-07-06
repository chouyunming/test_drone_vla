#!/usr/bin/env python
"""Inference smoke test for SmolVLA.

Loads a pretrained policy, builds an observation from a real dataset frame,
runs a single forward pass, and reports whether inference succeeded.
No robot hardware required.

Usage:
    conda activate smolvla
    python test.py

    # Override checkpoint or dataset
    python test.py --model smolvla --model-id lerobot/smolvla_base
    python test.py --model smolvla --dataset lerobot/libero

    # Explicit device selection
    python test.py --model smolvla --device cuda
    python test.py --model smolvla --device cpu
"""

import argparse
import logging
import os
import sys
import time
import warnings

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

warnings.filterwarnings("ignore")

logging.disable(logging.WARNING)

import torch

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODELS: dict[str, dict] = {
    "smolvla": {
        "model_id": "lerobot/smolvla_base",
        "dataset":  "lerobot/libero",
    },
}


# ---------------------------------------------------------------------------
# SmolVLA
# ---------------------------------------------------------------------------
def _load_smolvla(model_id: str):
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    return SmolVLAPolicy.from_pretrained(model_id)


def _run_smolvla(args, device: torch.device) -> int:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors

    cfg      = MODELS["smolvla"]
    model_id = args.model_id or cfg["model_id"]
    dataset_id = args.dataset or cfg["dataset"]

    print(f"[1/4] Loading SmolVLA policy: {model_id} → {device}")
    policy = _load_smolvla(model_id).to(device).eval()

    print("[2/4] Building pre/post processors")
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        model_id,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    print(f"[3/4] Loading dataset: {dataset_id} (first frame)")
    dataset = LeRobotDataset(dataset_id)
    frame   = dict(dataset[0])

    # Remap image keys if policy checkpoint uses different names than dataset
    if hasattr(policy.config, "input_features"):
        policy_img_keys = [
            k for k, v in policy.config.input_features.items()
            if hasattr(v, "type") and "VISUAL" in str(v.type)
        ]
        frame_img_keys = [k for k in frame if k.startswith("observation.images.")]
        for i, dst in enumerate(policy_img_keys):
            if dst not in frame and frame_img_keys:
                frame[dst] = frame[frame_img_keys[i % len(frame_img_keys)]]

    print("[4/4] Running select_action")
    batch = preprocess(frame)
    t0 = time.perf_counter()
    with torch.inference_mode():
        action = policy.select_action(batch)
        action = postprocess(action)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    if isinstance(action, dict):
        action = next(iter(action.values()))

    print("\n✓ INFERENCE OK")
    print(f"   model        : smolvla ({model_id})")
    print(f"   device       : {device}")
    print(f"   action shape : {tuple(action.shape)}")
    print(f"   latency      : {latency_ms:.1f} ms")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model",
        choices=list(MODELS),
        default='smolvla',
        help="Policy to smoke-test (smolvla)",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Override the default HuggingFace checkpoint",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Override the built-in LeRobot dataset repo id",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda | cpu | mps  (default: cuda if available)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    try:
        return _run_smolvla(args, device)
    except Exception as exc:          # noqa: BLE001  — smoke test wants a clean summary
        print(f"\n✗ INFERENCE FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
