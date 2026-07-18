from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


POSTERIOR_CHANNELS = (
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2",
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stable_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def hash_file(path: str | Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def hash_jsonable(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return stable_digest(payload)


def hash_state_dict(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def atomic_write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        json.dump(value, tmp, indent=2, ensure_ascii=False, sort_keys=True)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def append_jsonl(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def git_revision(root: str | Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def parameter_count(parameters: Iterable[torch.nn.Parameter]) -> int:
    return sum(parameter.numel() for parameter in parameters if parameter.requires_grad)


def retrieval_metrics(
    eeg_features: torch.Tensor,
    image_features: torch.Tensor,
    query_image_ids: list[str],
    gallery_image_ids: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if eeg_features.ndim != 2 or image_features.ndim != 2:
        raise ValueError("Retrieval features must be two-dimensional")
    if eeg_features.shape[0] != len(query_image_ids):
        raise ValueError("Query feature/ID count mismatch")
    if image_features.shape[0] != len(gallery_image_ids):
        raise ValueError("Gallery feature/ID count mismatch")
    if len(set(gallery_image_ids)) != len(gallery_image_ids):
        raise ValueError("Gallery image IDs must be unique")
    gallery_lookup = {image_id: index for index, image_id in enumerate(gallery_image_ids)}
    missing = [image_id for image_id in query_image_ids if image_id not in gallery_lookup]
    if missing:
        raise ValueError(f"Queries missing from gallery: {missing[:5]}")
    eeg = torch.nn.functional.normalize(eeg_features.float(), dim=-1)
    image = torch.nn.functional.normalize(image_features.float(), dim=-1)
    similarity = eeg @ image.T
    order = torch.argsort(similarity, dim=1, descending=True, stable=True)
    targets = torch.tensor([gallery_lookup[value] for value in query_image_ids], device=order.device)
    positions = torch.argsort(order, dim=1, stable=True)
    ranks = positions[torch.arange(len(targets), device=order.device), targets] + 1
    top1 = int((ranks == 1).sum().item())
    top5 = int((ranks <= min(5, image.shape[0])).sum().item())
    predictions: list[dict[str, Any]] = []
    order_cpu = order.cpu()
    ranks_cpu = ranks.cpu()
    similarity_cpu = similarity.cpu()
    for row, query_id in enumerate(query_image_ids):
        top_indices = order_cpu[row, : min(5, order_cpu.shape[1])].tolist()
        predictions.append(
            {
                "query_index": row,
                "query_image_id": query_id,
                "target_gallery_index": int(targets[row].item()),
                "predicted_image_id": gallery_image_ids[top_indices[0]],
                "target_rank": int(ranks_cpu[row].item()),
                "top5_image_ids": [gallery_image_ids[index] for index in top_indices],
                "top5_scores": [float(similarity_cpu[row, index].item()) for index in top_indices],
            }
        )
    total = len(query_image_ids)
    return (
        {
            "num_queries": total,
            "num_gallery": len(gallery_image_ids),
            "top1_correct": top1,
            "top5_correct": top5,
            "top1": top1 / total,
            "top5": top5 / total,
            "protocol": "standard_independent_exact_image",
        },
        predictions,
    )
