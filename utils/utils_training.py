import os
from typing import Optional

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from torch import nn

logger = get_logger(__name__)


def backward_and_clip_gradients(
    *,
    accelerator: Accelerator,
    model: nn.Module,
    loss: torch.Tensor,
    max_grad_norm: float,
) -> None:
    accelerator.backward(loss)
    if accelerator.sync_gradients:
        params_to_clip = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)


def rotate_checkpoints(output_dir: str, save_total_limit: Optional[int]) -> None:
    if (
        save_total_limit is None
        or save_total_limit <= 0
        or not os.path.isdir(output_dir)
    ):
        return
    checkpoints = []
    for name in os.listdir(output_dir):
        if name.startswith("step_") or name.startswith("epoch_"):
            path = os.path.join(output_dir, name)
            if os.path.isdir(path):
                checkpoints.append((os.path.getmtime(path), path))
    checkpoints.sort()
    while len(checkpoints) > save_total_limit:
        _, path = checkpoints.pop(0)
        logger.info(f"Deleting old checkpoint: {path}")
        import shutil

        shutil.rmtree(path, ignore_errors=True)
