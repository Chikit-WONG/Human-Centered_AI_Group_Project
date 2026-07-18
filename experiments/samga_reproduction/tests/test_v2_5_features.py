from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn


REPRODUCTION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRODUCTION_ROOT))

LOGICAL_LAYER_IDS = [20, 24, 28, 32, 36]
CAPTURED_BLOCK_OUTPUTS = [20, 21, 24, 25, 28, 29, 32, 33, 36, 37]
MODEL_WEIGHT_SHA256 = {
    "model-00001-of-00003.safetensors": (
        "9818659d13d932da8bc0c3b8ee15f5b5d68d8c94d66eb525be566066630111da"
    ),
    "model-00002-of-00003.safetensors": (
        "4f0c10e72d6f6513f421baa6ec843d5508657435059c1d18b6b5fd7789f9d5b7"
    ),
    "model-00003-of-00003.safetensors": (
        "d21c4fe0bc4af1425cfae1a59a8f5fbb00fde9d8e2888325a60913ac61b0494d"
    ),
}
MODEL_SMALL_FILE_SHA256 = {
    "config.json": "4fc4a1187b20575c0da8d27df2ad17f5ad6e8ac1c8b2af707bc8b263bd40c0a2",
    "configuration_intern_vit.py": "e620864fe9f2ef0104b39ea496cb844e1b363caaf8208e6f0bef1a72f31f00a3",
    "flash_attention.py": "d84f36949763545b58039d28669f9dc46fcace6c94b796e3f91a92553f5f5cad",
    "model.safetensors.index.json": "94d376c898c00585a38a588df9ff354fa965eafa9a1d56f69c1c8bad7ad08502",
    "modeling_intern_vit.py": "56220ba82cb511d51d5f2fa71eebd728b330fbccad9dfb128088a8fcc8f7d260",
    "preprocessor_config.json": "0658115064c561026539aeeead9ed3b1a8e0cc90967df8c142849199f955d2b4",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_digest(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_manifest(
    path: Path,
    row_count: int,
    split: str = "train",
    subject_id: int = 1,
) -> dict[str, Any]:
    records = [
        {"row_index": row, "image_path": f"/unused/image_{row}.jpg"}
        for row in range(row_count)
    ]
    manifest = {
        "schema_version": 1,
        "subject_id": subject_id,
        "split": split,
        "records": records,
        "records_sha256": json_digest(records),
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest["_manifest_sha256"] = sha256_file(path)
    return manifest


def allow_synthetic_manifest(
    monkeypatch: pytest.MonkeyPatch, manifest: dict[str, Any]
) -> None:
    contract = importlib.import_module("v2_5_feature_contract")
    monkeypatch.setitem(
        contract.CANONICAL_RECORDS_SHA256,
        manifest["split"],
        manifest["records_sha256"],
    )


def logical_layer_routes() -> dict[str, list[dict[str, int]]]:
    return {
        "idx0": [
            {
                "logical_layer_id": logical,
                "captured_block_output": logical,
                "source_axis_index": CAPTURED_BLOCK_OUTPUTS.index(logical),
            }
            for logical in LOGICAL_LAYER_IDS
        ],
        "idx_plus_1": [
            {
                "logical_layer_id": logical,
                "captured_block_output": logical + 1,
                "source_axis_index": CAPTURED_BLOCK_OUTPUTS.index(logical + 1),
            }
            for logical in LOGICAL_LAYER_IDS
        ],
    }


def write_shard(
    root: Path,
    manifest: dict[str, Any],
    *,
    shard_index: int,
    shard_count: int,
    row_start: int,
    row_end: int,
    hidden_size: int = 3200,
    full_shard_row_end: int | None = None,
    debug_partial_shard: bool = False,
    value_offset: float = 0.0,
) -> Path:
    shard_dir = root / f"shard_{shard_index:02d}"
    shard_dir.mkdir()
    shape = (row_end - row_start, len(CAPTURED_BLOCK_OUTPUTS), hidden_size)
    base = np.arange(np.prod(shape), dtype=np.float16).reshape(shape)
    raw_cls = base + np.float16(1000 * shard_index + value_offset)
    patch_mean = base + np.float16(100 * shard_index + value_offset)
    np.save(shard_dir / "raw_cls.npy", raw_cls, allow_pickle=False)
    np.save(shard_dir / "patch_mean.npy", patch_mean, allow_pickle=False)
    full_end = row_end if full_shard_row_end is None else full_shard_row_end
    metadata = {
        "schema_version": 2,
        "artifact_kind": "internvit_v2_5_feature_shard",
        "complete": True,
        "model_repo": "OpenGVLab/InternViT-6B-448px-V2_5",
        "model_revision": "9d1a4344077479c93d42584b6941c64d795d508d",
        "model_path": "/fixed/model",
        "model_config_sha256": "config-sha256",
        "preprocessor_config_sha256": "processor-sha256",
        "model_weight_sha256": MODEL_WEIGHT_SHA256,
        "model_small_file_sha256": MODEL_SMALL_FILE_SHA256,
        "manifest": "/fixed/manifest.json",
        "manifest_sha256": manifest["_manifest_sha256"],
        "subject_id": 1,
        "records_sha256": manifest["records_sha256"],
        "split": manifest["split"],
        "row_start": row_start,
        "row_end": row_end,
        "full_shard_row_end": full_end,
        "debug_partial_shard": debug_partial_shard,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "seed": 0,
        "hidden_size": hidden_size,
        "logical_layer_ids": LOGICAL_LAYER_IDS,
        "captured_block_outputs": CAPTURED_BLOCK_OUTPUTS,
        "logical_layer_routes": logical_layer_routes(),
        "pooling_files": {
            "raw_cls": {
                "filename": "raw_cls.npy",
                "shape": list(raw_cls.shape),
                "dtype": "float16",
                "sha256": sha256_file(shard_dir / "raw_cls.npy"),
            },
            "patch_mean": {
                "filename": "patch_mean.npy",
                "shape": list(patch_mean.shape),
                "dtype": "float16",
                "sha256": sha256_file(shard_dir / "patch_mean.npy"),
            },
        },
    }
    metadata["shard_payload_sha256"] = json_digest(
        {
            pooling: entry["sha256"]
            for pooling, entry in metadata["pooling_files"].items()
        }
    )
    (shard_dir / "metadata.json").write_text(
        json.dumps(metadata, sort_keys=True), encoding="utf-8"
    )
    return shard_dir


class AddOne(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return values + 1


class FakeInternViT(nn.Module):
    def __init__(self, block_count: int = 5) -> None:
        super().__init__()
        self.embeddings = nn.Identity()
        self.encoder = SimpleNamespace(
            layers=nn.ModuleList([AddOne() for _ in range(block_count)])
        )


def test_collect_block_poolings_uses_actual_outputs_and_excludes_cls() -> None:
    extractor = importlib.import_module("extract_v2_5_features")
    model = FakeInternViT()
    pixels = torch.tensor(
        [
            [[100.0, 200.0], [1.0, 3.0], [5.0, 7.0]],
            [[300.0, 400.0], [9.0, 11.0], [13.0, 15.0]],
        ]
    )

    raw_cls, patch_mean = extractor.collect_block_poolings(
        model, pixels, captured_block_outputs=[1, 3]
    )

    assert raw_cls.shape == (2, 2, 2)
    assert patch_mean.shape == (2, 2, 2)
    assert torch.equal(raw_cls[0], torch.tensor([[101.0, 201.0], [103.0, 203.0]]))
    assert torch.equal(patch_mean[0], torch.tensor([[4.0, 6.0], [6.0, 8.0]]))
    assert [layer.calls for layer in model.encoder.layers] == [1, 1, 1, 0, 0]


@pytest.mark.parametrize("captured", [[], [0], [1, 1], [6]])
def test_collect_block_poolings_rejects_invalid_actual_outputs(
    captured: list[int],
) -> None:
    extractor = importlib.import_module("extract_v2_5_features")
    with pytest.raises(ValueError):
        extractor.collect_block_poolings(
            FakeInternViT(), torch.zeros(1, 2, 3), captured
        )


def test_merge_rejects_missing_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    merger = importlib.import_module("merge_v2_5_features")
    manifest_path = tmp_path / "manifest.json"
    manifest = write_manifest(manifest_path, row_count=8)
    allow_synthetic_manifest(monkeypatch, manifest)
    first = write_shard(
        tmp_path,
        manifest,
        shard_index=0,
        shard_count=8,
        row_start=0,
        row_end=1,
    )

    with pytest.raises(ValueError, match="shard|cover"):
        merger.merge_feature_shards(
            manifest_path=manifest_path,
            shard_directories=[first],
            output_directory=tmp_path / "merged",
        )

    assert not (tmp_path / "merged").exists()


def test_merge_rejects_corrupt_shard_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    merger = importlib.import_module("merge_v2_5_features")
    manifest_path = tmp_path / "manifest.json"
    manifest = write_manifest(manifest_path, row_count=8)
    allow_synthetic_manifest(monkeypatch, manifest)
    shards = [
        write_shard(
            tmp_path,
            manifest,
            shard_index=index,
            shard_count=8,
            row_start=index,
            row_end=index + 1,
        )
        for index in range(8)
    ]
    corrupt = np.load(shards[1] / "raw_cls.npy")
    corrupt[0, 0, 0] += np.float16(1)
    np.save(shards[1] / "raw_cls.npy", corrupt, allow_pickle=False)

    with pytest.raises(ValueError, match="hash mismatch"):
        merger.merge_feature_shards(
            manifest_path=manifest_path,
            shard_directories=shards,
            output_directory=tmp_path / "merged",
        )


def test_merge_preserves_shapes_and_writes_unambiguous_layer_routes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    merger = importlib.import_module("merge_v2_5_features")
    verifier = importlib.import_module("verify_v2_5_features")
    manifest_path = tmp_path / "manifest.json"
    manifest = write_manifest(manifest_path, row_count=8)
    allow_synthetic_manifest(monkeypatch, manifest)
    shards = [
        write_shard(
            tmp_path,
            manifest,
            shard_index=index,
            shard_count=8,
            row_start=index,
            row_end=index + 1,
        )
        for index in range(8)
    ]
    output = tmp_path / "merged"

    merger.merge_feature_shards(
        manifest_path=manifest_path,
        shard_directories=list(reversed(shards)),
        output_directory=output,
    )
    metadata = verifier.verify_feature_directory(
        manifest_path=manifest_path,
        feature_directory=output,
        expected_artifact_kind="internvit_v2_5_feature_merged",
    )

    assert np.load(output / "raw_cls.npy").shape == (8, 10, 3200)
    assert np.load(output / "patch_mean.npy").shape == (8, 10, 3200)
    assert metadata["logical_layer_ids"] == LOGICAL_LAYER_IDS
    assert metadata["captured_block_outputs"] == CAPTURED_BLOCK_OUTPUTS
    assert [
        route["captured_block_output"]
        for route in metadata["logical_layer_routes"]["idx0"]
    ] == [20, 24, 28, 32, 36]
    assert [
        route["captured_block_output"]
        for route in metadata["logical_layer_routes"]["idx_plus_1"]
    ] == [21, 25, 29, 33, 37]
    assert [
        route["logical_layer_id"]
        for route in metadata["logical_layer_routes"]["idx_plus_1"]
    ] == LOGICAL_LAYER_IDS
    assert metadata["model_revision"] == "9d1a4344077479c93d42584b6941c64d795d508d"
    assert metadata["model_weight_sha256"] == MODEL_WEIGHT_SHA256


def test_manifest_rejects_non_sub_01_even_with_same_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = importlib.import_module("v2_5_feature_contract")
    manifest_path = tmp_path / "sub-02_train.json"
    manifest = write_manifest(manifest_path, row_count=8, subject_id=2)
    monkeypatch.setitem(
        contract.CANONICAL_RECORDS_SHA256, "train", manifest["records_sha256"]
    )
    with pytest.raises(ValueError, match="sub-01"):
        contract.load_manifest(manifest_path)


def test_feature_validation_rejects_non_3200_width(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = importlib.import_module("v2_5_feature_contract")
    manifest_path = tmp_path / "manifest.json"
    manifest = write_manifest(manifest_path, row_count=8)
    allow_synthetic_manifest(monkeypatch, manifest)
    shard = write_shard(
        tmp_path,
        manifest,
        shard_index=0,
        shard_count=8,
        row_start=0,
        row_end=1,
        hidden_size=3,
    )
    loaded = contract.load_manifest(manifest_path)
    with pytest.raises(ValueError, match="3200|shape"):
        contract.validate_feature_directory(loaded, shard)


def test_merge_rejects_noncanonical_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    merger = importlib.import_module("merge_v2_5_features")
    manifest_path = tmp_path / "manifest.json"
    manifest = write_manifest(manifest_path, row_count=8)
    allow_synthetic_manifest(monkeypatch, manifest)
    shards = [
        write_shard(
            tmp_path,
            manifest,
            shard_index=index,
            shard_count=8,
            row_start=0 if index == 1 else index,
            row_end=1 if index == 1 else index + 1,
        )
        for index in range(8)
    ]
    with pytest.raises(ValueError, match="canonical|overlap|gap"):
        merger.merge_feature_shards(
            manifest_path=manifest_path,
            shard_directories=shards,
            output_directory=tmp_path / "merged",
        )


def test_verifier_rejects_single_train_shard_as_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verifier = importlib.import_module("verify_v2_5_features")
    manifest_path = tmp_path / "manifest.json"
    manifest = write_manifest(manifest_path, row_count=8)
    allow_synthetic_manifest(monkeypatch, manifest)
    shard = write_shard(
        tmp_path,
        manifest,
        shard_index=0,
        shard_count=8,
        row_start=0,
        row_end=1,
    )
    with pytest.raises(ValueError, match="8|shard"):
        verifier.verify_feature_set(
            manifest_path=manifest_path,
            feature_directories=[shard],
            expected_artifact_kind="internvit_v2_5_feature_shard",
        )


def test_partial_smoke_shard_cannot_satisfy_full_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    extractor = importlib.import_module("extract_v2_5_features")
    manifest_path = tmp_path / "manifest.json"
    manifest = write_manifest(manifest_path, row_count=16)
    allow_synthetic_manifest(monkeypatch, manifest)
    shard = write_shard(
        tmp_path,
        manifest,
        shard_index=0,
        shard_count=8,
        row_start=0,
        row_end=1,
        full_shard_row_end=2,
        debug_partial_shard=True,
    )
    loaded = importlib.import_module("v2_5_feature_contract").load_manifest(
        manifest_path
    )
    model_identity = {
        "model_repo": "OpenGVLab/InternViT-6B-448px-V2_5",
        "model_revision": "9d1a4344077479c93d42584b6941c64d795d508d",
        "model_path": "/fixed/model",
        "model_config_sha256": "config-sha256",
        "preprocessor_config_sha256": "processor-sha256",
        "model_weight_sha256": MODEL_WEIGHT_SHA256,
        "model_small_file_sha256": MODEL_SMALL_FILE_SHA256,
    }
    with pytest.raises(FileExistsError, match="incompatible"):
        extractor._validate_existing_output(
            loaded,
            shard,
            shard_index=0,
            shard_count=8,
            row_start=0,
            row_end=2,
            full_shard_row_end=2,
            seed=0,
            model_identity=model_identity,
        )


def write_fake_model(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Any, Path]:
    extractor = importlib.import_module("extract_v2_5_features")
    root.mkdir()
    contents = {
        "config.json": json.dumps(
            {
                "hidden_size": 3200,
                "num_hidden_layers": 45,
                "image_size": 448,
                "patch_size": 14,
            }
        ).encode(),
        "preprocessor_config.json": json.dumps(
            {
                "crop_size": 448,
                "size": 448,
                "image_mean": [0.485, 0.456, 0.406],
                "image_std": [0.229, 0.224, 0.225],
            }
        ).encode(),
        "configuration_intern_vit.py": b"config-code",
        "flash_attention.py": b"flash-code",
        "model.safetensors.index.json": b"{}",
        "modeling_intern_vit.py": b"model-code",
    }
    for filename, content in contents.items():
        (root / filename).write_bytes(content)
    weights = {"weight-a.safetensors": b"weight-a", "weight-b.safetensors": b"weight-b"}
    for filename, content in weights.items():
        (root / filename).write_bytes(content)
    small_hashes = {
        filename: sha256_file(root / filename) for filename in contents
    }
    weight_hashes = {
        filename: sha256_file(root / filename) for filename in weights
    }
    monkeypatch.setattr(extractor, "MODEL_SMALL_FILE_SHA256", small_hashes)
    monkeypatch.setattr(extractor, "MODEL_WEIGHT_SHA256", weight_hashes)
    monkeypatch.setattr(
        extractor,
        "MODEL_WEIGHT_BYTES",
        {filename: len(content) for filename, content in weights.items()},
    )
    (root / "model_provenance.json").write_text(
        json.dumps(
            {
                "complete": True,
                "model_repo": extractor.MODEL_REPO,
                "model_revision": extractor.MODEL_REVISION,
                "model_weight_sha256": weight_hashes,
                "small_file_sha256": small_hashes,
            }
        ),
        encoding="utf-8",
    )
    return extractor, root


def test_model_verification_hashes_current_weight_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    extractor, model = write_fake_model(tmp_path / "model", monkeypatch)
    extractor.verify_model_directory(model)
    weight = model / "weight-a.safetensors"
    weight.write_bytes(b"Weight-a")
    with pytest.raises(ValueError, match="weight hash mismatch"):
        extractor.verify_model_directory(model)


def test_model_verification_hashes_current_remote_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    extractor, model = write_fake_model(tmp_path / "model", monkeypatch)
    code = model / "modeling_intern_vit.py"
    code.write_bytes(b"Model-code")
    with pytest.raises(ValueError, match="small-file hash mismatch"):
        extractor.verify_model_directory(model)


def test_symlinked_parent_is_rejected_before_resolution(tmp_path: Path) -> None:
    contract = importlib.import_module("v2_5_feature_contract")
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "redirect"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        contract.resolve_without_symlinks(link / "new-cache", "Output path")


def test_existing_variant_is_bound_to_requested_merged_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    merger = importlib.import_module("merge_v2_5_features")
    selector = importlib.import_module("materialize_v2_5_variant")
    manifest_path = tmp_path / "manifest.json"
    manifest = write_manifest(manifest_path, row_count=8)
    allow_synthetic_manifest(monkeypatch, manifest)

    def build_source(name: str, value_offset: float) -> Path:
        shard_root = tmp_path / f"{name}_shards"
        shard_root.mkdir()
        shards = [
            write_shard(
                shard_root,
                manifest,
                shard_index=index,
                shard_count=8,
                row_start=index,
                row_end=index + 1,
                value_offset=value_offset,
            )
            for index in range(8)
        ]
        merged = tmp_path / f"{name}_merged"
        merger.merge_feature_shards(
            manifest_path=manifest_path,
            shard_directories=shards,
            output_directory=merged,
        )
        return merged

    first_source = build_source("first", 0.0)
    second_source = build_source("second", 7.0)
    output = tmp_path / "selected"
    selector.materialize_variant(
        manifest_path=manifest_path,
        merged_directory=first_source,
        output_directory=output,
        pooling="raw_cls",
        indexing_variant="idx0",
    )
    with pytest.raises(FileExistsError, match="invalid selected variant"):
        selector.materialize_variant(
            manifest_path=manifest_path,
            merged_directory=second_source,
            output_directory=output,
            pooling="raw_cls",
            indexing_variant="idx0",
        )


def test_existing_merge_revalidates_supplied_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    merger = importlib.import_module("merge_v2_5_features")
    manifest_path = tmp_path / "manifest.json"
    manifest = write_manifest(manifest_path, row_count=8)
    allow_synthetic_manifest(monkeypatch, manifest)
    shards = [
        write_shard(
            tmp_path,
            manifest,
            shard_index=index,
            shard_count=8,
            row_start=index,
            row_end=index + 1,
        )
        for index in range(8)
    ]
    output = tmp_path / "merged"
    merger.merge_feature_shards(
        manifest_path=manifest_path,
        shard_directories=shards,
        output_directory=output,
    )

    other_root = tmp_path / "other"
    other_root.mkdir()
    other_shards = [
        write_shard(
            other_root,
            manifest,
            shard_index=index,
            shard_count=8,
            row_start=index,
            row_end=index + 1,
            value_offset=7.0,
        )
        for index in range(8)
    ]
    with pytest.raises(FileExistsError, match="invalid merged"):
        merger.merge_feature_shards(
            manifest_path=manifest_path,
            shard_directories=other_shards,
            output_directory=output,
        )

    corrupt = np.load(shards[3] / "raw_cls.npy")
    corrupt[0, 0, 0] += np.float16(64)
    np.save(shards[3] / "raw_cls.npy", corrupt, allow_pickle=False)

    with pytest.raises(ValueError, match="hash mismatch"):
        merger.merge_feature_shards(
            manifest_path=manifest_path,
            shard_directories=shards,
            output_directory=output,
        )


def test_merged_validation_rejects_rehashed_malformed_source_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    merger = importlib.import_module("merge_v2_5_features")
    contract = importlib.import_module("v2_5_feature_contract")
    manifest_path = tmp_path / "manifest.json"
    manifest = write_manifest(manifest_path, row_count=8)
    allow_synthetic_manifest(monkeypatch, manifest)
    shards = [
        write_shard(
            tmp_path,
            manifest,
            shard_index=index,
            shard_count=8,
            row_start=index,
            row_end=index + 1,
        )
        for index in range(8)
    ]
    output = tmp_path / "merged"
    merger.merge_feature_shards(
        manifest_path=manifest_path,
        shard_directories=shards,
        output_directory=output,
    )
    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    del metadata["source_shards"][2]["pooling_sha256"]["patch_mean"]
    metadata["source_shards_sha256"] = json_digest(metadata["source_shards"])
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    loaded_manifest = contract.load_manifest(manifest_path)

    with pytest.raises(ValueError, match="source-shard"):
        contract.validate_feature_directory(
            loaded_manifest,
            output,
            expected_artifact_kind="internvit_v2_5_feature_merged",
        )


def test_existing_variant_requires_semantic_shape_and_regular_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    merger = importlib.import_module("merge_v2_5_features")
    selector = importlib.import_module("materialize_v2_5_variant")
    manifest_path = tmp_path / "manifest.json"
    manifest = write_manifest(manifest_path, row_count=8)
    allow_synthetic_manifest(monkeypatch, manifest)
    shards = [
        write_shard(
            tmp_path,
            manifest,
            shard_index=index,
            shard_count=8,
            row_start=index,
            row_end=index + 1,
        )
        for index in range(8)
    ]
    merged = tmp_path / "merged"
    merger.merge_feature_shards(
        manifest_path=manifest_path,
        shard_directories=shards,
        output_directory=merged,
    )

    malformed = tmp_path / "selected-malformed"
    selector.materialize_variant(
        manifest_path=manifest_path,
        merged_directory=merged,
        output_directory=malformed,
        pooling="raw_cls",
        indexing_variant="idx0",
    )
    malformed_feature = np.zeros((8, 5, 3), dtype=np.float16)
    np.save(
        malformed / selector.FEATURE_FILENAME,
        malformed_feature,
        allow_pickle=False,
    )
    malformed_metadata_path = malformed / "metadata.json"
    malformed_metadata = json.loads(
        malformed_metadata_path.read_text(encoding="utf-8")
    )
    malformed_metadata["shape"] = list(malformed_feature.shape)
    malformed_metadata["feature_sha256"] = sha256_file(
        malformed / selector.FEATURE_FILENAME
    )
    malformed_metadata_path.write_text(
        json.dumps(malformed_metadata), encoding="utf-8"
    )
    with pytest.raises(FileExistsError, match="invalid selected variant"):
        selector.materialize_variant(
            manifest_path=manifest_path,
            merged_directory=merged,
            output_directory=malformed,
            pooling="raw_cls",
            indexing_variant="idx0",
        )

    symlinked = tmp_path / "selected-symlinked"
    selector.materialize_variant(
        manifest_path=manifest_path,
        merged_directory=merged,
        output_directory=symlinked,
        pooling="raw_cls",
        indexing_variant="idx0",
    )
    metadata_path = symlinked / "metadata.json"
    external_metadata = tmp_path / "external-metadata.json"
    metadata_path.rename(external_metadata)
    metadata_path.symlink_to(external_metadata)
    with pytest.raises(FileExistsError, match="invalid selected variant"):
        selector.materialize_variant(
            manifest_path=manifest_path,
            merged_directory=merged,
            output_directory=symlinked,
            pooling="raw_cls",
            indexing_variant="idx0",
        )
