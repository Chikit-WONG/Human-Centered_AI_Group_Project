#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from samga_lora.model import SAMGATaskModel, load_clip_provider  # noqa: E402
from samga_lora.utils import atomic_write_json, hash_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify a SAMGA checkpoint can be reconstructed exactly")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    task = SAMGATaskModel(
        layer_ids=config["layer_ids"], prior_center=int(config["prior_center"])
    )
    load_result = task.load_state_dict(checkpoint["task_state_dict"], strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(f"Task reload failed: {load_result}")
    adapter_keys = 0
    if config["vision_mode"] == "lora":
        provider, _ = load_clip_provider(
            model_path=config["clip_path"],
            layer_ids=config["layer_ids"],
            vision_mode="lora",
            lora_rank=int(config["lora_rank"]),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        adapter_state = checkpoint["vision_adapter_state_dict"]
        adapter_keys = len(adapter_state)
        adapter_result = set_peft_model_state_dict(provider.backbone, adapter_state)
        if getattr(adapter_result, "unexpected_keys", []):
            raise RuntimeError(f"Adapter reload failed: {adapter_result.unexpected_keys}")
        reloaded = get_peft_model_state_dict(provider.backbone)
        if set(reloaded) != set(adapter_state):
            raise RuntimeError("Reloaded adapter keys differ from checkpoint keys")
        for key, expected in adapter_state.items():
            if not torch.equal(reloaded[key].cpu(), expected.cpu()):
                raise RuntimeError(f"Reloaded adapter tensor differs for {key}")
    report = {
        "schema_version": 1,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": hash_file(args.checkpoint),
        "epoch": int(checkpoint["epoch"]),
        "vision_mode": config["vision_mode"],
        "task_keys": len(checkpoint["task_state_dict"]),
        "adapter_keys": adapter_keys,
        "passed": True,
    }
    atomic_write_json(args.output, report)


if __name__ == "__main__":
    main()
