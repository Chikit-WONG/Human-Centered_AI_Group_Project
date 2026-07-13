import os
from typing import Optional

from accelerate.logging import get_logger

logger = get_logger(__name__)


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
