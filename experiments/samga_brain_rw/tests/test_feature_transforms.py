from __future__ import annotations

import copy
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from samga_brain_rw.feature_transforms import (
    LayerNormTransform,
    SeparateImagePreProjector,
    SharedImagePreProjector,
    Stage2PreprocessingRegistry,
    TrainWhitening,
    load_stage2_preprocessing_candidates,
    validate_stage2_runner_config,
)
from samga_brain_rw.hashing import sha256_json


def _features() -> np.ndarray:
    rng = np.random.default_rng(20260719)
    return rng.normal(size=(8, 5, 3)).astype(np.float32)


def _manual_zca(
    features: np.ndarray, rows: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    selected = features[np.asarray(rows, dtype=np.int64)].astype(np.float64)
    means = selected.mean(axis=0)
    matrices = []
    for layer in range(5):
        centered = selected[:, layer, :] - means[layer]
        covariance = centered.T @ centered / (len(rows) - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        inverse_sqrt = (np.maximum(eigenvalues, 0.0) + 1e-5) ** -0.5
        matrices.append((eigenvectors * inverse_sqrt) @ eigenvectors.T)
    return means.astype(np.float32), np.stack(matrices).astype(np.float32)


def test_layernorm_is_per_sample_per_layer_float32_then_cast_back() -> None:
    torch.manual_seed(19)
    values = torch.randn(2, 5, 3200, dtype=torch.float16)
    transform = LayerNormTransform(enabled=True)

    actual = transform(values)
    expected = F.layer_norm(
        values.float(), (3200,), weight=None, bias=None, eps=1e-6
    ).to(values.dtype)

    assert torch.equal(actual, expected)
    assert actual.shape == values.shape
    assert actual.dtype == values.dtype
    assert list(transform.parameters()) == []


def test_layernorm_disabled_is_validated_identity() -> None:
    values = torch.randn(1, 5, 3200)
    transform = LayerNormTransform(enabled=False)
    assert transform(values) is values
    with pytest.raises(ValueError, match="shape|3200"):
        transform(torch.randn(1, 5, 3199))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"eps": 1e-5},
        {"affine": True},
        {"compute_dtype": torch.float64},
    ],
)
def test_layernorm_rejects_unlocked_contract(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="locked"):
        LayerNormTransform(enabled=True, **kwargs)


@pytest.mark.parametrize(
    "values",
    [
        torch.randn(5, 3200),
        torch.randn(1, 4, 3200),
        torch.randn(1, 5, 3199),
        torch.empty(0, 5, 3200),
        torch.ones(1, 5, 3200, dtype=torch.int64),
    ],
)
def test_layernorm_rejects_bad_dtype_or_shape(values: torch.Tensor) -> None:
    with pytest.raises(
        (TypeError, ValueError), match="floating|shape|batch|layers|3200"
    ):
        LayerNormTransform(enabled=True)(values)


def test_layernorm_rejects_nonfinite_input() -> None:
    values = torch.zeros(1, 5, 3200)
    values[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        LayerNormTransform(enabled=True)(values)


def test_whitening_fit_uses_float64_n_minus_one_eigh_zca() -> None:
    features = _features()
    rows = (0, 2, 4, 7)
    expected_mean, expected_matrix = _manual_zca(features, rows)

    whitening = TrainWhitening.fit(
        features,
        rows,
        input_provenance_sha256="1" * 64,
        cache_provenance_sha256="2" * 64,
    )

    assert whitening.source_scope == "train"
    assert whitening.canonical_train_rows == rows
    assert whitening.eps == 1e-5
    assert whitening.mean.dtype == torch.float32
    assert whitening.matrix.dtype == torch.float32
    np.testing.assert_array_equal(whitening.mean.numpy(), expected_mean)
    np.testing.assert_array_equal(whitening.matrix.numpy(), expected_matrix)


def test_whitening_accepts_real_float16_canonical_cache_dtype() -> None:
    features = _features().astype(np.float16)
    rows = (0, 2, 4, 7)
    expected_mean, expected_matrix = _manual_zca(features, rows)

    whitening = TrainWhitening.fit(features, rows)

    np.testing.assert_array_equal(whitening.mean.numpy(), expected_mean)
    np.testing.assert_array_equal(whitening.matrix.numpy(), expected_matrix)
    assert (
        whitening.input_provenance_sha256
        != TrainWhitening.fit(features.astype(np.float32), rows).input_provenance_sha256
    )


def test_whitening_never_reads_or_hashes_unselected_rows() -> None:
    rows = (0, 2, 4, 7)
    clean = _features()
    inaccessible = clean.copy()
    inaccessible[[1, 3, 5, 6]] = np.nan

    first = TrainWhitening.fit(clean, rows)
    second = TrainWhitening.fit(inaccessible, rows)

    assert second.to_payload() == first.to_payload()


@pytest.mark.parametrize(
    "source_scope", ["val-dev", "val-confirm", "test", "formal-test", "train+val-dev"]
)
def test_whitening_rejects_nontrain_source(source_scope: str) -> None:
    with pytest.raises(ValueError, match="train"):
        TrainWhitening.fit(_features(), (0, 2), source_scope=source_scope)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ((0, 0, 2), "unique"),
        ((2, 0, 4), "sorted"),
        ((-1, 2), "range"),
        ((0, 8), "range"),
        ((False, 2), "integer"),
        ((0,), "two"),
    ],
)
def test_whitening_rejects_invalid_canonical_train_rows(
    rows: tuple[object, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        TrainWhitening.fit(_features(), rows)


@pytest.mark.parametrize(
    "features",
    [
        _features().astype(np.float64),
        _features()[:, :, ::-1],
        _features()[:, 0, :],
        _features()[:, :4, :],
        np.empty((0, 5, 3), dtype=np.float32),
        np.empty((8, 5, 0), dtype=np.float32),
    ],
)
def test_whitening_rejects_bad_fit_dtype_or_shape(features: np.ndarray) -> None:
    with pytest.raises(
        (TypeError, ValueError),
        match="float16|float32|C-contiguous|shape|rows|dimension",
    ):
        TrainWhitening.fit(features, (0, 2))


def test_whitening_rejects_nonfinite_selected_train_data() -> None:
    features = _features()
    features[2, 1, 1] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        TrainWhitening.fit(features, (0, 2, 4))


def test_whitening_serialization_binds_rows_provenance_and_float32_arrays() -> None:
    whitening = TrainWhitening.fit(
        _features(),
        (0, 2, 4, 7),
        input_provenance_sha256="a" * 64,
        cache_provenance_sha256="b" * 64,
    )
    payload = whitening.to_payload()

    assert payload["canonical_train_rows"] == [0, 2, 4, 7]
    assert payload["row_list_sha256"] == sha256_json([0, 2, 4, 7])
    assert payload["input_provenance_sha256"] == "a" * 64
    assert payload["cache_provenance_sha256"] == "b" * 64
    assert payload["mean"]["dtype"] == "float32"
    assert payload["matrix"]["dtype"] == "float32"
    assert payload["payload_sha256"] == whitening.payload_sha256

    restored = TrainWhitening.from_payload(payload)
    assert restored.to_payload() == payload
    assert torch.equal(restored.mean, whitening.mean)
    assert torch.equal(restored.matrix, whitening.matrix)


@pytest.mark.parametrize("target", ["payload", "rows", "mean", "matrix"])
def test_whitening_rejects_serialization_hash_tampering(target: str) -> None:
    payload = TrainWhitening.fit(_features(), (0, 2, 4, 7)).to_payload()
    tampered = copy.deepcopy(payload)
    if target == "payload":
        tampered["payload_sha256"] = "0" * 64
    elif target == "rows":
        tampered["canonical_train_rows"][0] = 1
    else:
        encoded = tampered[target]["data_base64"]
        tampered[target]["data_base64"] = ("A" if encoded[0] != "A" else "B") + encoded[
            1:
        ]

    with pytest.raises(ValueError, match="hash|SHA|payload|base64"):
        TrainWhitening.from_payload(tampered)


def test_whitening_transform_computes_float32_then_casts_back() -> None:
    whitening = TrainWhitening.fit(_features(), (0, 2, 4, 7))
    values = torch.tensor(_features()[:2], dtype=torch.float16)

    actual = whitening.transform(values)
    expected = torch.einsum(
        "bld,lde->ble",
        values.float() - whitening.mean,
        whitening.matrix,
    ).to(values.dtype)

    assert torch.equal(actual, expected)
    assert actual.shape == values.shape
    assert actual.dtype == values.dtype
    assert actual.device == values.device
    assert torch.equal(whitening(values), actual)


@pytest.mark.parametrize(
    "values",
    [
        torch.randn(5, 3),
        torch.randn(1, 4, 3),
        torch.randn(1, 5, 4),
        torch.empty(0, 5, 3),
        torch.ones(1, 5, 3, dtype=torch.int64),
    ],
)
def test_whitening_transform_rejects_bad_dtype_or_shape(values: torch.Tensor) -> None:
    whitening = TrainWhitening.fit(_features(), (0, 2, 4, 7))
    with pytest.raises(
        (TypeError, ValueError), match="floating|shape|batch|layers|dimension"
    ):
        whitening(values)


def test_whitening_transform_rejects_nonfinite_input() -> None:
    whitening = TrainWhitening.fit(_features(), (0, 2, 4, 7))
    values = torch.zeros(1, 5, 3)
    values[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        whitening(values)


def test_preprojectors_have_locked_structure_and_exact_parameter_counts() -> None:
    shared = SharedImagePreProjector()
    separate = SeparateImagePreProjector()
    one_preprojector = 3200 * 1024 + 1024
    downstream = 5 * (1024 * 512 + 512)

    assert sum(parameter.numel() for parameter in shared.parameters()) == (
        one_preprojector + downstream
    )
    assert sum(parameter.numel() for parameter in separate.parameters()) == (
        5 * one_preprojector + downstream
    )
    assert isinstance(shared.image_pre_projector, torch.nn.Linear)
    assert len(shared.image_projectors) == 5
    assert len(separate.image_pre_projectors) == 5
    assert len(separate.image_projectors) == 5
    assert (
        len({layer.weight.data_ptr() for layer in separate.image_pre_projectors}) == 5
    )


def test_preprojectors_preserve_five_layer_output_interface() -> None:
    torch.manual_seed(29)
    shared = SharedImagePreProjector()
    separate = SeparateImagePreProjector()
    with torch.no_grad():
        for layer in separate.image_pre_projectors:
            layer.load_state_dict(shared.image_pre_projector.state_dict())
        for separate_layer, shared_layer in zip(
            separate.image_projectors, shared.image_projectors, strict=True
        ):
            separate_layer.load_state_dict(shared_layer.state_dict())

    values = torch.randn(2, 5, 3200)
    assert shared.preproject(values).shape == (2, 5, 1024)
    assert separate.preproject(values).shape == (2, 5, 1024)
    assert shared(values).shape == (2, 5, 512)
    assert separate(values).shape == (2, 5, 512)
    assert torch.equal(shared.preproject(values), separate.preproject(values))
    assert torch.equal(shared(values), separate(values))


@pytest.mark.parametrize(
    "projector_type", [SharedImagePreProjector, SeparateImagePreProjector]
)
@pytest.mark.parametrize(
    "values",
    [
        torch.randn(5, 3200),
        torch.randn(1, 4, 3200),
        torch.randn(1, 5, 3199),
        torch.empty(0, 5, 3200),
        torch.ones(1, 5, 3200, dtype=torch.int64),
    ],
)
def test_preprojectors_reject_bad_dtype_or_shape(
    projector_type: type[torch.nn.Module], values: torch.Tensor
) -> None:
    with pytest.raises(
        (TypeError, ValueError), match="floating|shape|batch|layers|3200"
    ):
        projector_type()(values)


@pytest.mark.parametrize(
    "projector_type", [SharedImagePreProjector, SeparateImagePreProjector]
)
def test_preprojectors_reject_nonfinite_input(
    projector_type: type[torch.nn.Module],
) -> None:
    values = torch.zeros(1, 5, 3200)
    values[0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        projector_type()(values)


def test_stage2_registry_loads_only_locked_preprocessing_ids(configs_dir: Path) -> None:
    registry = load_stage2_preprocessing_candidates(
        configs_dir / "stage2_candidates_v1.json"
    )

    assert isinstance(registry, Stage2PreprocessingRegistry)
    assert registry.layernorm_ids == ("s2-layernorm-off", "s2-layernorm-on")
    assert registry.whitening_ids == ("s2-whitening-off", "s2-whitening-on")
    assert registry.preprojector_ids == ("s2-preproj-shared", "s2-preproj-separate")


def test_stage2_registry_rejects_modified_candidate_ids(
    configs_dir: Path, tmp_path: Path
) -> None:
    payload = json.loads(
        (configs_dir / "stage2_candidates_v1.json").read_text(encoding="utf-8")
    )
    payload["layernorm"][1]["config_id"] = "s2-layernorm-surprise"
    path = tmp_path / "stage2-modified.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="stage2|layernorm|config_id|exact"):
        load_stage2_preprocessing_candidates(path)


def test_stage2_runner_allows_baseline_or_exactly_one_active_factor(
    configs_dir: Path,
) -> None:
    registry = load_stage2_preprocessing_candidates(
        configs_dir / "stage2_candidates_v1.json"
    )
    baseline = {
        "layernorm_config_id": "s2-layernorm-off",
        "whitening_config_id": "s2-whitening-off",
        "preprojector_config_id": "s2-preproj-shared",
    }

    assert validate_stage2_runner_config(registry, **baseline).active_factor is None
    for argument, config_id, factor in (
        ("layernorm_config_id", "s2-layernorm-on", "layernorm"),
        ("whitening_config_id", "s2-whitening-on", "whitening"),
        ("preprojector_config_id", "s2-preproj-separate", "preprojector"),
    ):
        selected = dict(baseline)
        selected[argument] = config_id
        assert (
            validate_stage2_runner_config(registry, **selected).active_factor == factor
        )


def test_stage2_runner_rejects_more_than_one_active_factor(
    configs_dir: Path,
) -> None:
    registry = load_stage2_preprocessing_candidates(
        configs_dir / "stage2_candidates_v1.json"
    )
    active = {
        "layernorm_config_id": "s2-layernorm-on",
        "whitening_config_id": "s2-whitening-on",
        "preprojector_config_id": "s2-preproj-separate",
    }
    baseline = {
        "layernorm_config_id": "s2-layernorm-off",
        "whitening_config_id": "s2-whitening-off",
        "preprojector_config_id": "s2-preproj-shared",
    }
    for keys in (*combinations(active, 2), tuple(active)):
        selected = dict(baseline)
        selected.update({key: active[key] for key in keys})
        with pytest.raises(ValueError, match="one.*factor|more than one"):
            validate_stage2_runner_config(registry, **selected)


def test_stage2_runner_rejects_unknown_preprocessing_id(configs_dir: Path) -> None:
    registry = load_stage2_preprocessing_candidates(
        configs_dir / "stage2_candidates_v1.json"
    )
    with pytest.raises(ValueError, match="config_id"):
        validate_stage2_runner_config(
            registry,
            layernorm_config_id="s2-layernorm-unknown",
            whitening_config_id="s2-whitening-off",
            preprojector_config_id="s2-preproj-shared",
        )
