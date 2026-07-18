from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from exploratory_internvit.entry import add_image_dim_to_parsed_args
from exploratory_internvit.extract_features import collect_hidden_state_cls
from exploratory_internvit import merge_feature_shards
from samga_lora.data import feature_cache_metadata_path
from samga_lora.utils import atomic_write_json, hash_file, hash_jsonable, read_json


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


class AddOne(nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + 1


class FakeInternViT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings = nn.Identity()
        self.encoder = SimpleNamespace(layers=nn.ModuleList([AddOne() for _ in range(4)]))


def test_collect_hidden_state_cls_uses_embedding_zero_indexing() -> None:
    pixels = torch.zeros(2, 3, 5)
    result = collect_hidden_state_cls(FakeInternViT(), pixels, [1, 3])
    assert result.shape == (2, 2, 5)
    assert torch.equal(result[:, 0], torch.ones(2, 5))
    assert torch.equal(result[:, 1], torch.full((2, 5), 3.0))


def test_entry_persists_wrapper_image_dimension() -> None:
    parsed = add_image_dim_to_parsed_args(lambda: SimpleNamespace(seed=2025), 3200)
    assert parsed.seed == 2025
    assert parsed.image_dim == 3200


@pytest.mark.parametrize("layers", [[], [0], [1, 1], [5]])
def test_collect_hidden_state_cls_rejects_invalid_layers(layers: list[int]) -> None:
    with pytest.raises(ValueError):
        collect_hidden_state_cls(FakeInternViT(), torch.zeros(1, 2, 3), layers)


def test_merge_feature_shards_preserves_global_row_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = [{"row_index": index} for index in range(4)]
    manifest = tmp_path / "manifest.json"
    atomic_write_json(
        manifest,
        {
            "schema_version": 1,
            "subject_id": 1,
            "split": "train",
            "records": records,
            "records_sha256": hash_jsonable(records),
        },
    )
    shards = []
    for start, end in ((0, 2), (2, 4)):
        path = tmp_path / f"rows_{start}_{end}.npy"
        values = np.arange(start * 6, end * 6, dtype=np.float16).reshape(end - start, 2, 3)
        np.save(path, values)
        metadata = {
            "schema_version": 1,
            "exploratory": True,
            "inferred_model": True,
            "model_repo": "test/model",
            "model_revision": "revision",
            "model_path": "/model",
            "model_config_sha256": "config",
            "model_weight_sha256": {"shard": "weight"},
            "layer_semantics": "embedding_is_zero",
            "records_sha256": hash_jsonable(records),
            "split": "train",
            "dtype": "float16",
            "layer_ids": [1, 2],
            "row_start": start,
            "row_end": end,
            "partial_rows": True,
            "shape": list(values.shape),
            "cache_sha256": hash_file(path),
            "complete": True,
        }
        atomic_write_json(feature_cache_metadata_path(path), metadata)
        shards.append(path)
    output = tmp_path / "merged.npy"
    monkeypatch.setattr(
        "sys.argv",
        [
            "merge_feature_shards.py",
            "--manifest",
            str(manifest),
            "--output",
            str(output),
            "--shards",
            *(str(path) for path in shards),
        ],
    )
    merge_feature_shards.main()
    expected = np.concatenate([np.load(path) for path in shards], axis=0)
    assert np.array_equal(np.load(output), expected)
    metadata = read_json(feature_cache_metadata_path(output))
    assert metadata["row_start"] == 0
    assert metadata["row_end"] == 4
    assert metadata["partial_rows"] is False
    assert metadata["cache_sha256"] == hash_file(output)


@pytest.mark.parametrize(
    "filename", ["extract.slurm", "extract_sharded.slurm", "train_array.slurm"]
)
def test_slurm_wrappers_resolve_project_from_submit_directory(filename: str) -> None:
    text = (EXPERIMENT_ROOT / "exploratory_internvit" / filename).read_text(
        encoding="utf-8"
    )
    assert "SLURM_SUBMIT_DIR" in text
    assert 'dirname "$0"' not in text
