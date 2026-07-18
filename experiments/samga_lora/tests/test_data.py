from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from samga_lora.data import ThingsEEGSubjectDataset, feature_cache_metadata_path
from samga_lora.utils import hash_file, hash_jsonable


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "train.pt"
    eeg = np.arange(3 * 2 * 3 * 4, dtype=np.float32).reshape(3, 2, 3, 4)
    torch.save({"eeg": eeg, "ch_names": ["C0", "C1", "C2"]}, source)
    records = [
        {"row_index": 0, "image_id": "a", "concept_id": "c1", "image_path": "/a", "validation_query": False},
        {"row_index": 1, "image_id": "b", "concept_id": "c1", "image_path": "/b", "validation_query": False},
        {"row_index": 2, "image_id": "c", "concept_id": "c2", "image_path": "/c", "validation_query": True},
    ]
    manifest = {
        "schema_version": 1,
        "subject_id": 1,
        "split": "train",
        "source_pt": str(source),
        "validation_concepts": ["c2"],
        "records": records,
        "records_sha256": hash_jsonable(records),
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    cache_path = tmp_path / "cache.npy"
    np.save(cache_path, np.arange(12, dtype=np.float16).reshape(3, 1, 4))
    metadata = {
        "schema_version": 1,
        "records_sha256": manifest["records_sha256"],
        "shape": [3, 1, 4],
        "dtype": "float16",
        "layer_ids": [1],
        "cache_sha256": hash_file(cache_path),
    }
    feature_cache_metadata_path(cache_path).write_text(json.dumps(metadata), encoding="utf-8")
    return manifest_path, cache_path


def test_concept_disjoint_subsets_and_cache_alignment(tmp_path: Path) -> None:
    manifest, cache = _write_fixture(tmp_path)
    train = ThingsEEGSubjectDataset(
        manifest_path=manifest,
        subset="pilot_train",
        seed=42,
        selected_channels=("C0", "C1"),
        feature_cache=cache,
        expected_layer_ids=(1,),
        smooth_probability=0.0,
    )
    validation = ThingsEEGSubjectDataset(
        manifest_path=manifest,
        subset="pilot_validation",
        seed=42,
        selected_channels=("C0", "C1"),
        feature_cache=cache,
        expected_layer_ids=(1,),
        smooth_probability=0.0,
    )
    assert len(train) == 2
    assert {train[index]["concept_id"] for index in range(len(train))} == {"c1"}
    assert len(validation) == 1
    assert validation[0]["concept_id"] == "c2"
    assert validation[0]["layer_features"].shape == (1, 4)
    expected = torch.from_numpy(np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)[:, :2].mean(0))
    assert torch.equal(train[0]["eeg"], expected)


def test_feature_cache_rejects_wrong_layer_ids(tmp_path: Path) -> None:
    manifest, cache = _write_fixture(tmp_path)
    with pytest.raises(ValueError, match="layers"):
        ThingsEEGSubjectDataset(
            manifest_path=manifest,
            subset="formal_train",
            seed=42,
            selected_channels=("C0", "C1"),
            feature_cache=cache,
            expected_layer_ids=(2,),
        )
