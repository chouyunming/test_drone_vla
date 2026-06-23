#!/usr/bin/env python
"""Inference smoke test for LeRobot VLA policies (SmolVLA).

Loads an open pretrained policy + an open built-in LeRobotDataset, runs a single
forward pass through `select_action`, and reports whether inference ran.
No robot hardware required.

Usage:
    python test.py --model smolvla
"""

import argparse
import logging
import sys
import time
import warnings

warnings.filterwarnings("ignore", message="`torch_dtype` is deprecated")

logging.getLogger("transformers").addFilter(
    lambda r: "`torch_dtype` is deprecated" not in r.getMessage()
)

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors

# Open / built-in checkpoints and datasets that download automatically on first run.
MODELS = {
    "smolvla": {
        "model_id": "lerobot/smolvla_base",   # open, ~0.5B params
        "dataset": "lerobot/libero",          # open, built-in
    },
}


def load_policy(name: str, model_id: str):
    if name == "smolvla":
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        return SmolVLAPolicy.from_pretrained(model_id)
    raise ValueError(f"Unknown model: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(MODELS), required=True, default='smolvla', 
                        help="Which policy to smoke-test")
    parser.add_argument("--model-id", default=None,
                        help="Override the default checkpoint (e.g. a LeRobot-format finetune)")
    parser.add_argument("--dataset", default=None,
                        help="Override the default built-in dataset repo id")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="cuda | cpu | mps")
    args = parser.parse_args()

    cfg = MODELS[args.model]
    model_id = args.model_id or cfg["model_id"]
    dataset_id = args.dataset or cfg["dataset"]
    device = torch.device(args.device)

    try:
        print(f"[1/4] Loading policy: {model_id} -> {device}")
        policy = load_policy(args.model, model_id).to(device).eval()

        print("[2/4] Building pre/post processors")
        preprocess, postprocess = make_pre_post_processors(
            policy.config,
            model_id,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )

        print(f"[3/4] Loading built-in dataset: {dataset_id} (first frame)")
        dataset = LeRobotDataset(dataset_id)
        frame = dict(dataset[0])

        # Remap dataset image keys to whatever names the policy checkpoint expects.
        # E.g. smolvla_base uses camera1/2/3 but lerobot/libero uses image/image2.
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

        print("\n INFERENCE OK")
        print(f"   model        : {args.model} ({model_id})")
        print(f"   device       : {device}")
        print(f"   action shape : {tuple(action.shape)}")
        print(f"   latency      : {latency_ms:.1f} ms")
        return 0

    except Exception as exc:  # noqa: BLE001 - smoke test wants a clean summary
        print(f"\n INFERENCE FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())