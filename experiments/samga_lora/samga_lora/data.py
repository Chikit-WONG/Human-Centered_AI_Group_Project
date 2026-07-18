from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .utils import POSTERIOR_CHANNELS, atomic_write_json, hash_file, hash_jsonable, read_json, stable_digest


Subset = Literal["pilot_train", "pilot_validation", "formal_train", "test"]


def _first_image_paths(raw: Any, expected: int) -> list[str]:
    values = np.asarray(raw)
    if values.ndim >= 2:
        values = values.reshape(values.shape[0], -1)[:, 0]
    values = values.reshape(-1)
    if len(values) != expected:
        raise ValueError(f"EEG/image row mismatch: {expected} vs {len(values)}")
    return [str(value) for value in values.tolist()]


def _flatten_eeg_rows(eeg: np.ndarray) -> np.ndarray:
    if eeg.ndim == 5:
        return eeg.reshape(eeg.shape[0] * eeg.shape[1], *eeg.shape[2:])
    if eeg.ndim == 4:
        return eeg
    raise ValueError(f"Expected EEG [N,R,C,T] or [concept,image,R,C,T], got {eeg.shape}")


def resolve_image_path(things_root: str | Path, recorded_path: str) -> Path:
    root = Path(things_root)
    recorded = Path(recorded_path)
    candidates = [recorded] if recorded.is_absolute() else [root / recorded]
    normalized = recorded_path.replace("\\", "/")
    if normalized.startswith("train_images/"):
        candidates.append(root / normalized.replace("train_images/", "training_images/", 1))
    elif normalized.startswith("training_images/"):
        candidates.append(root / normalized.replace("training_images/", "train_images/", 1))
    if normalized.startswith("Image_set_Resize/"):
        candidates.append(root / normalized.split("Image_set_Resize/", 1)[1])
    candidates.extend([root / "training_images" / recorded.name, root / "test_images" / recorded.name])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    matches = list(root.glob(f"**/{recorded.name}"))
    matches = [match for match in matches if match.is_file()]
    if len(matches) == 1:
        return matches[0].resolve()
    raise FileNotFoundError(f"Cannot resolve image path {recorded_path!r} below {root}")


def build_subject_manifest(
    *,
    things_root: str | Path,
    brain_root: str | Path,
    subject_id: int,
    split: Literal["train", "test"],
    output_path: str | Path,
    validation_concepts: int = 200,
) -> dict[str, Any]:
    pt_path = Path(brain_root) / f"sub-{subject_id:02d}" / f"{split}.pt"
    loaded = torch.load(pt_path, map_location="cpu", weights_only=False)
    eeg = _flatten_eeg_rows(np.asarray(loaded["eeg"]))
    image_paths = _first_image_paths(loaded["img"], eeg.shape[0])
    records: list[dict[str, Any]] = []
    grouped: dict[str, list[int]] = defaultdict(list)
    for row_index, raw_path in enumerate(image_paths):
        path = resolve_image_path(things_root, raw_path)
        concept_id = path.parent.name
        record = {
            "row_index": row_index,
            "image_id": path.stem,
            "concept_id": concept_id,
            "image_path": str(path),
            "validation_query": False,
        }
        grouped[concept_id].append(len(records))
        records.append(record)

    validation_ids: list[str] = []
    if split == "train":
        concepts = sorted(grouped, key=lambda value: stable_digest(f"samga-lora-val-v1:{value}"))
        if len(concepts) != 1654:
            raise ValueError(f"Expected 1654 train concepts, found {len(concepts)}")
        validation_ids = concepts[:validation_concepts]
        for concept_id in validation_ids:
            positions = sorted(grouped[concept_id], key=lambda i: records[i]["image_id"])
            selector = int(stable_digest(f"samga-lora-val-image-v1:{concept_id}"), 16)
            records[positions[selector % len(positions)]]["validation_query"] = True
            if len(positions) != 10:
                raise ValueError(f"Expected ten images for {concept_id}, found {len(positions)}")
    else:
        if len(records) != 200 or len(grouped) != 200:
            raise ValueError(f"Expected 200 unique test images/concepts, found {len(records)}/{len(grouped)}")
        for record in records:
            record["validation_query"] = True

    ch_names = [str(name) for name in loaded.get("ch_names", [])]
    if len(ch_names) != eeg.shape[-2]:
        raise ValueError(f"Channel metadata mismatch: {len(ch_names)} vs {eeg.shape[-2]}")
    manifest = {
        "schema_version": 1,
        "subject_id": subject_id,
        "split": split,
        "source_pt": str(pt_path.resolve()),
        "eeg_shape": list(eeg.shape),
        "eeg_dtype": str(eeg.dtype),
        "ch_names": ch_names,
        "validation_salt": "samga-lora-val-v1",
        "validation_concepts": validation_ids,
        "records": records,
    }
    manifest["records_sha256"] = hash_jsonable(records)
    atomic_write_json(output_path, manifest)
    return manifest


def load_manifest(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported manifest schema in {path}")
    expected = manifest.get("records_sha256")
    if expected != hash_jsonable(manifest["records"]):
        raise ValueError(f"Manifest record hash mismatch in {path}")
    return manifest


def feature_cache_metadata_path(cache_path: str | Path) -> Path:
    cache_path = Path(cache_path)
    return cache_path.with_suffix(cache_path.suffix + ".meta.json")


def validate_feature_cache(
    cache_path: str | Path,
    manifest: dict[str, Any],
    *,
    expected_layer_ids: Sequence[int] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    cache_path = Path(cache_path)
    metadata_path = feature_cache_metadata_path(cache_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Feature-cache metadata is missing: {metadata_path}")
    metadata = read_json(metadata_path)
    if metadata.get("schema_version") != 1:
        raise ValueError(f"Unsupported feature-cache schema in {metadata_path}")
    if metadata.get("records_sha256") != manifest.get("records_sha256"):
        raise ValueError("Feature cache and manifest use different image rows/order")
    if expected_layer_ids is not None:
        actual_layers = [int(value) for value in metadata.get("layer_ids", [])]
        if actual_layers != [int(value) for value in expected_layer_ids]:
            raise ValueError(f"Feature-cache layers {actual_layers} do not match {expected_layer_ids}")
    expected_hash = metadata.get("cache_sha256")
    if expected_hash and expected_hash != hash_file(cache_path):
        raise ValueError(f"Feature-cache hash mismatch for {cache_path}")
    cache = np.load(cache_path, mmap_mode="r")
    expected_shape = tuple(int(value) for value in metadata.get("shape", []))
    if tuple(cache.shape) != expected_shape:
        raise ValueError(f"Feature-cache shape {cache.shape} does not match metadata {expected_shape}")
    if cache.shape[0] != len(manifest["records"]):
        raise ValueError(
            f"Feature rows {cache.shape[0]} do not match manifest rows {len(manifest['records'])}"
        )
    return cache, metadata


def _moving_average(signal: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    time_len = signal.shape[-1]
    left = np.maximum(0, np.arange(time_len) - kernel_size // 2)
    right = np.minimum(time_len, np.arange(time_len) + kernel_size // 2 + 1)
    cumulative = np.pad(np.cumsum(signal, axis=-1, dtype=np.float32), ((0, 0), (1, 0)))
    return (cumulative[:, right] - cumulative[:, left]) / (right - left)[None, :]


class ThingsEEGSubjectDataset(Dataset):
    def __init__(
        self,
        *,
        manifest_path: str | Path,
        subset: Subset,
        seed: int,
        selected_channels: Sequence[str] = POSTERIOR_CHANNELS,
        feature_cache: str | Path | None = None,
        expected_layer_ids: Sequence[int] | None = None,
        smooth_probability: float = 0.3,
    ) -> None:
        self.manifest = load_manifest(manifest_path)
        self.subset = subset
        self.seed = int(seed)
        self.smooth_probability = float(smooth_probability)
        loaded = torch.load(self.manifest["source_pt"], map_location="cpu", weights_only=False)
        self.eeg = _flatten_eeg_rows(np.asarray(loaded["eeg"]))
        all_channels = [str(name) for name in loaded["ch_names"]]
        missing = [name for name in selected_channels if name not in all_channels]
        if missing:
            raise KeyError(f"Missing channels: {missing}")
        self.channel_indices = np.asarray([all_channels.index(name) for name in selected_channels])
        validation = set(self.manifest.get("validation_concepts", []))
        records = self.manifest["records"]
        if subset == "pilot_train":
            self.records = [record for record in records if record["concept_id"] not in validation]
        elif subset == "pilot_validation":
            self.records = [record for record in records if record["validation_query"]]
        elif subset in ("formal_train", "test"):
            self.records = list(records)
        else:
            raise ValueError(f"Unknown subset {subset}")
        self.training = subset in ("pilot_train", "formal_train")
        self.feature_cache = None
        self.feature_cache_metadata = None
        if feature_cache is not None:
            self.feature_cache, self.feature_cache_metadata = validate_feature_cache(
                feature_cache,
                self.manifest,
                expected_layer_ids=expected_layer_ids,
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, item: int) -> dict[str, Any]:
        record = self.records[item]
        row = np.asarray(self.eeg[record["row_index"]])
        if row.ndim != 3:
            raise ValueError(f"Expected repetitions x channels x time, got {row.shape}")
        eeg = row[:, self.channel_indices, :].mean(axis=0, dtype=np.float32)
        if self.training and self.smooth_probability > 0:
            rng_seed = int(stable_digest(f"smooth:{self.seed}:{record['row_index']}")[:16], 16)
            mask = np.random.default_rng(rng_seed).random(eeg.shape[0]) < self.smooth_probability
            if mask.any():
                eeg = eeg.copy()
                eeg[mask] = _moving_average(eeg[mask])
        output: dict[str, Any] = {
            "eeg": torch.from_numpy(np.ascontiguousarray(eeg)).float(),
            "subject_id": int(self.manifest["subject_id"]),
            "image_id": record["image_id"],
            "concept_id": record["concept_id"],
            "image_path": record["image_path"],
            "row_index": int(record["row_index"]),
        }
        if self.feature_cache is not None:
            output["layer_features"] = torch.from_numpy(
                np.asarray(self.feature_cache[record["row_index"]], dtype=np.float32).copy()
            )
        return output


class SAMGACollator:
    def __init__(self, image_processor: Any | None = None) -> None:
        self.image_processor = image_processor

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        batch: dict[str, Any] = {
            "eeg": torch.stack([example["eeg"] for example in examples]),
            "subject_ids": torch.tensor([example["subject_id"] for example in examples], dtype=torch.long),
            "image_ids": [example["image_id"] for example in examples],
            "concept_ids": [example["concept_id"] for example in examples],
            "row_indices": torch.tensor([example["row_index"] for example in examples], dtype=torch.long),
        }
        if "layer_features" in examples[0]:
            batch["layer_features"] = torch.stack([example["layer_features"] for example in examples])
        else:
            if self.image_processor is None:
                raise RuntimeError("An image processor is required for online vision batches")
            images = [Image.open(example["image_path"]).convert("RGB") for example in examples]
            batch["pixel_values"] = self.image_processor(images=images, return_tensors="pt").pixel_values
        return batch
