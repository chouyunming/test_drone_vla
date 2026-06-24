#!/usr/bin/env python
"""Inference smoke test for VLA policies (SmolVLA and GR00T-N1.6).

Loads a pretrained policy, builds an observation (real dataset frame for
SmolVLA, synthetic tensor for GR00T), runs a single forward pass, and
reports whether inference succeeded.  No robot hardware required.

Usage:
    # SmolVLA — uses lerobot/smolvla_base + lerobot/libero dataset
    conda activate lerobot_vla_test
    python test.py --model smolvla

    # GR00T-N1.6 — uses nvidia/GR00T-N1.6-DROID + synthetic obs
    conda activate groot_vla_test
    python test.py --model groot

    # Override checkpoint / embodiment for GR00T
    python test.py --model groot \\
        --model-id nvidia/GR00T-N1.6-3B \\
        --embodiment-tag gr1

    # Force CPU
    python test.py --model smolvla --device cpu
    python test.py --model groot   --device cpu
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

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODELS: dict[str, dict] = {
    "smolvla": {
        # Open ~0.5 B param VLA from HuggingFace LeRobot
        "model_id": "lerobot/smolvla_base",
        "dataset":  "lerobot/libero",
    },
    "groot": {
        # GR00T-N1.6 fine-tuned on DROID (Franka arm, 6-DOF + gripper).
        # Override with --model-id nvidia/GR00T-N1.6-3B for the base model.
        "model_id":       "nvidia/GR00T-N1.6-DROID",
        "embodiment_tag": "oxe_droid",
        # Expected state dims for oxe_droid embodiment:
        #   joint_position   : (B, T, 6)   — 6-DOF relative joint positions
        #   gripper_position : (B, T, 1)   — gripper open/close
        # Video keys (224×224 RGB, uint8): exterior_image_1_left, wrist_image_left
        # Language key: annotation.language.language_instruction
        "_state_dims": {
            "joint_position":   6,
            "gripper_position": 1,
        },
    },
}


# ---------------------------------------------------------------------------
# SmolVLA helpers
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
# GR00T-N1.6 helpers
# ---------------------------------------------------------------------------
def _load_groot(model_id: str, embodiment_tag: str, device: str):
    """Load Gr00tPolicy from a HuggingFace checkpoint or local path."""
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.data.embodiment_tags import EmbodimentTag

    # EmbodimentTag values are lower-case strings like "oxe_droid", "gr1", …
    tag = EmbodimentTag(embodiment_tag.lower())

    policy = Gr00tPolicy(
        model_path=model_id,
        embodiment_tag=tag,
        device=device,
        strict=True,   # keep strict=True; synthetic obs matches expected shapes
    )
    return policy


def _make_groot_obs(policy, state_dims: dict[str, int]) -> dict:
    """Build a synthetic observation dict that matches the policy's modality config.

    Args:
        policy    : Loaded Gr00tPolicy (used to read modality config).
        state_dims: Dict mapping each state key → its feature dimension D.

    Returns:
        Nested observation dict with keys "video", "state", "language".
    """
    import numpy as np

    modcfg  = policy.modality_configs           # loaded from checkpoint
    B       = 1                                  # batch size
    T_vid   = len(modcfg["video"].delta_indices)
    T_state = len(modcfg["state"].delta_indices)

    # --- video ---
    obs_video = {
        key: np.zeros((B, T_vid, 224, 224, 3), dtype=np.uint8)
        for key in modcfg["video"].modality_keys
    }

    # --- state ---
    # Verify every required state key has a known dimension.
    for key in modcfg["state"].modality_keys:
        if key not in state_dims:
            raise ValueError(
                f"State key '{key}' is not in the state_dims mapping.  "
                f"Pass --model groot and check MODELS['groot']['_state_dims'], "
                f"or override via --model-id with a matching checkpoint."
            )
    obs_state = {
        key: np.zeros((B, T_state, state_dims[key]), dtype=np.float32)
        for key in modcfg["state"].modality_keys
    }

    # --- language ---
    lang_key = modcfg["language"].modality_keys[0]
    obs_lang = {lang_key: [["pick up the object and place it on the plate"]]}

    return {"video": obs_video, "state": obs_state, "language": obs_lang}


def _run_groot(args, device: torch.device) -> int:
    import numpy as np

    cfg            = MODELS["groot"]
    model_id       = args.model_id       or cfg["model_id"]
    embodiment_tag = args.embodiment_tag or cfg["embodiment_tag"]
    state_dims: dict[str, int] = cfg["_state_dims"]

    print(f"[1/3] Loading GR00T policy: {model_id}  embodiment={embodiment_tag}  → {device}")
    policy = _load_groot(model_id, embodiment_tag, str(device))

    print("[2/3] Building synthetic observation")
    obs = _make_groot_obs(policy, state_dims)

    print("[3/3] Running get_action")
    t0 = time.perf_counter()
    with torch.inference_mode():
        action, _info = policy.get_action(obs)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    # action is dict[str, np.ndarray(B, T_action, D)]
    action_summary = {k: v.shape for k, v in action.items()}

    print("\n✓ INFERENCE OK")
    print(f"   model          : groot ({model_id})")
    print(f"   embodiment     : {embodiment_tag}")
    print(f"   device         : {device}")
    print(f"   action shapes  : {action_summary}")
    print(f"   latency        : {latency_ms:.1f} ms")
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
        required=True,
        help="Which policy to smoke-test: smolvla | groot",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Override the default HuggingFace checkpoint (e.g. a fine-tuned variant)",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="[smolvla only] Override the built-in LeRobot dataset repo id",
    )
    parser.add_argument(
        "--embodiment-tag",
        default=None,
        dest="embodiment_tag",
        help=(
            "[groot only] GR00T embodiment tag, e.g. oxe_droid, gr1, libero_panda.  "
            "Must match the checkpoint you pass via --model-id."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda | cpu | mps  (default: cuda if available)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    try:
        if args.model == "smolvla":
            return _run_smolvla(args, device)
        elif args.model == "groot":
            return _run_groot(args, device)
        else:
            print(f"ERROR: unknown model '{args.model}'", file=sys.stderr)
            return 1
    except Exception as exc:          # noqa: BLE001  — smoke test wants a clean summary
        print(f"\n✗ INFERENCE FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())