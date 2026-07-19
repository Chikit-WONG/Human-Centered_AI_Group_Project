"""Locked Stage 2 cached-feature preprocessing candidates.

Whitening deliberately accepts canonical cache rows rather than split arrays so
that the fitted artifact records, and can later prove, the exact train-only row
selection used to estimate its statistics.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import SemanticConfig
from .hashing import sha256_json


_LAYERS = 5
_IMAGE_DIM = 3200
_IMAGE_MID_DIM = 1024
_FEATURE_DIM = 512
_LAYERNORM_EPS = 1e-6
_WHITENING_EPS = 1e-5
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_LAYERNORM_IDS = ("s2-layernorm-off", "s2-layernorm-on")
_WHITENING_IDS = ("s2-whitening-off", "s2-whitening-on")
_PREPROJECTOR_IDS = ("s2-preproj-shared", "s2-preproj-separate")


def _locked_float(value: object, expected: float, context: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{context} must be the locked finite value {expected}")
    parsed = float(value)
    if parsed != expected:
        raise ValueError(f"{context} must be the locked value {expected}")
    return parsed


def _sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256 hash")
    return value


def _validate_tensor_features(
    features: torch.Tensor,
    *,
    feature_dim: int,
    context: str,
) -> None:
    if not isinstance(features, torch.Tensor):
        raise TypeError(f"{context} must be a torch.Tensor")
    if features.ndim != 3:
        raise ValueError(
            f"{context} must have shape [batch,{_LAYERS},{feature_dim}], "
            f"got {tuple(features.shape)}"
        )
    if features.shape[0] <= 0:
        raise ValueError(f"{context} batch dimension must be non-empty")
    if features.shape[1] != _LAYERS or features.shape[2] != feature_dim:
        raise ValueError(
            f"{context} must have shape [batch,{_LAYERS},{feature_dim}], "
            f"got {tuple(features.shape)}"
        )
    if not features.dtype.is_floating_point:
        raise TypeError(f"{context} must have a floating-point dtype")
    if not bool(torch.isfinite(features).all().item()):
        raise ValueError(f"{context} must contain only finite values")


class LayerNormTransform(nn.Module):
    """Optional non-affine LayerNorm under the preregistered Stage 2 contract."""

    def __init__(
        self,
        enabled: bool,
        eps: float = _LAYERNORM_EPS,
        affine: bool = False,
        compute_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if type(enabled) is not bool:
            raise ValueError("enabled must be a boolean")
        self.eps = _locked_float(eps, _LAYERNORM_EPS, "LayerNorm eps")
        if affine is not False:
            raise ValueError("LayerNorm affine must be the locked value False")
        if compute_dtype is not torch.float32:
            raise ValueError("LayerNorm compute_dtype must be locked to torch.float32")
        self.enabled = enabled
        self.affine = False
        self.compute_dtype = torch.float32

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        _validate_tensor_features(
            features, feature_dim=_IMAGE_DIM, context="LayerNorm features"
        )
        if not self.enabled:
            return features
        normalized = F.layer_norm(
            features.to(dtype=self.compute_dtype),
            (_IMAGE_DIM,),
            weight=None,
            bias=None,
            eps=self.eps,
        )
        return normalized.to(dtype=features.dtype)


def _validate_canonical_rows(
    rows: Sequence[int],
    *,
    row_count: int,
) -> tuple[int, ...]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise ValueError("canonical_train_rows must be a sequence of integers")
    values: list[int] = []
    for index, value in enumerate(rows):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"canonical_train_rows[{index}] must be an integer")
        values.append(int(value))
    canonical = tuple(values)
    if len(canonical) < 2:
        raise ValueError("canonical_train_rows must contain at least two rows")
    if len(set(canonical)) != len(canonical):
        raise ValueError("canonical_train_rows must be unique")
    if tuple(sorted(canonical)) != canonical:
        raise ValueError("canonical_train_rows must be sorted")
    if any(row < 0 or row >= row_count for row in canonical):
        raise ValueError(
            "canonical_train_rows values must be in canonical cache row range"
        )
    return canonical


def _array_payload(tensor: torch.Tensor) -> dict[str, object]:
    array = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=np.dtype("<f4"))
    raw = array.tobytes(order="C")
    return {
        "dtype": "float32",
        "shape": list(array.shape),
        "data_base64": base64.b64encode(raw).decode("ascii"),
        "data_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _object(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{context} keys must be strings")
    return dict(value)


def _exact_keys(value: Mapping[str, object], expected: set[str], context: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(
            f"{context} payload keys do not match the sealed schema; "
            f"missing={missing}, unknown={unknown}"
        )


def _decode_array(value: object, context: str) -> np.ndarray:
    payload = _object(value, context)
    _exact_keys(
        payload,
        {"dtype", "shape", "data_base64", "data_sha256"},
        context,
    )
    if payload["dtype"] != "float32":
        raise ValueError(f"{context}.dtype must be float32")
    shape_value = payload["shape"]
    if not isinstance(shape_value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in shape_value
    ):
        raise ValueError(f"{context}.shape must contain positive integers")
    encoded = payload["data_base64"]
    if not isinstance(encoded, str):
        raise ValueError(f"{context}.data_base64 must be a base64 string")
    try:
        raw = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (UnicodeEncodeError, binascii.Error) as exc:
        raise ValueError(f"{context}.data_base64 is invalid base64") from exc
    expected_hash = _sha256(payload["data_sha256"], f"{context}.data_sha256")
    if hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError(f"{context} data SHA-256 hash does not match")
    expected_size = math.prod(shape_value) * np.dtype("<f4").itemsize
    if len(raw) != expected_size:
        raise ValueError(f"{context} byte size does not match its shape")
    return np.frombuffer(raw, dtype=np.dtype("<f4")).reshape(shape_value).copy()


class TrainWhitening(nn.Module):
    """Independent per-layer train-only ZCA whitening with a sealed payload."""

    _ARTIFACT_TYPE = "samga_brain_rw_train_whitening"
    _ALGORITHM = "per_layer_zca_float64_n_minus_one_eigh"
    _BODY_KEYS = {
        "artifact_type",
        "schema_version",
        "algorithm",
        "source_scope",
        "eps",
        "canonical_row_count",
        "canonical_train_rows",
        "row_list_sha256",
        "input_provenance_sha256",
        "cache_provenance_sha256",
        "mean",
        "matrix",
    }

    def __init__(
        self,
        *,
        mean: np.ndarray,
        matrix: np.ndarray,
        canonical_train_rows: tuple[int, ...],
        canonical_row_count: int,
        input_provenance_sha256: str,
        cache_provenance_sha256: str,
        payload_sha256: str | None = None,
    ) -> None:
        super().__init__()
        mean_array = np.ascontiguousarray(mean, dtype=np.float32)
        matrix_array = np.ascontiguousarray(matrix, dtype=np.float32)
        if mean_array.ndim != 2 or mean_array.shape[0] != _LAYERS:
            raise ValueError("whitening mean must have shape [5,D]")
        dimension = int(mean_array.shape[1])
        if dimension <= 0 or matrix_array.shape != (
            _LAYERS,
            dimension,
            dimension,
        ):
            raise ValueError("whitening matrix must have shape [5,D,D]")
        if not np.isfinite(mean_array).all() or not np.isfinite(matrix_array).all():
            raise ValueError("whitening statistics must contain only finite values")
        self.register_buffer("mean", torch.from_numpy(mean_array.copy()))
        self.register_buffer("matrix", torch.from_numpy(matrix_array.copy()))
        self.canonical_train_rows = canonical_train_rows
        self.canonical_row_count = canonical_row_count
        self.source_scope = "train"
        self.eps = _WHITENING_EPS
        self.input_provenance_sha256 = _sha256(
            input_provenance_sha256, "input_provenance_sha256"
        )
        self.cache_provenance_sha256 = _sha256(
            cache_provenance_sha256, "cache_provenance_sha256"
        )
        calculated = sha256_json(self._payload_body())
        if payload_sha256 is not None and (
            _sha256(payload_sha256, "payload_sha256") != calculated
        ):
            raise ValueError("whitening payload SHA-256 hash does not match")
        self._payload_sha256 = calculated

    @classmethod
    def fit(
        cls,
        canonical_features: np.ndarray,
        canonical_train_rows: Sequence[int],
        eps: float = _WHITENING_EPS,
        *,
        source_scope: str = "train",
        input_provenance_sha256: str | None = None,
        cache_provenance_sha256: str | None = None,
    ) -> "TrainWhitening":
        if source_scope != "train":
            raise ValueError("whitening statistics may be fit from train rows only")
        _locked_float(eps, _WHITENING_EPS, "whitening eps")
        if not isinstance(canonical_features, np.ndarray):
            raise TypeError("canonical_features must be a numpy.ndarray")
        if canonical_features.dtype not in (
            np.dtype(np.float16),
            np.dtype(np.float32),
        ):
            raise TypeError(
                "canonical_features dtype must be exactly float16 or float32"
            )
        if not canonical_features.flags.c_contiguous:
            raise ValueError("canonical_features must be C-contiguous")
        if canonical_features.ndim != 3:
            raise ValueError("canonical_features must have shape [rows,5,dimension]")
        row_count, layers, dimension = canonical_features.shape
        if row_count <= 0 or layers != _LAYERS or dimension <= 0:
            raise ValueError(
                "canonical_features must have non-empty shape [rows,5,dimension]"
            )
        rows = _validate_canonical_rows(canonical_train_rows, row_count=int(row_count))

        # Fancy indexing materializes only the selected canonical train rows.
        # In particular, no whole-cache finite check or content hash is allowed.
        row_indices = np.asarray(rows, dtype=np.int64)
        selected = np.ascontiguousarray(canonical_features[row_indices, :, :])
        if not np.isfinite(selected).all():
            raise ValueError("selected canonical train rows must be finite")

        input_dtype = str(canonical_features.dtype)
        serialized_dtype = np.dtype("<f2" if input_dtype == "float16" else "<f4")
        selected_bytes = np.ascontiguousarray(selected, dtype=serialized_dtype).tobytes(
            order="C"
        )
        selected_sha256 = hashlib.sha256(selected_bytes).hexdigest()
        row_list_sha256 = sha256_json(list(rows))
        if input_provenance_sha256 is None:
            input_provenance_sha256 = sha256_json(
                {
                    "artifact": "selected_canonical_train_features",
                    "canonical_cache_shape": list(canonical_features.shape),
                    "dtype": input_dtype,
                    "canonical_train_rows_sha256": row_list_sha256,
                    "selected_data_sha256": selected_sha256,
                }
            )
        else:
            input_provenance_sha256 = _sha256(
                input_provenance_sha256, "input_provenance_sha256"
            )
        if cache_provenance_sha256 is None:
            cache_provenance_sha256 = sha256_json(
                {
                    "artifact": "train_whitening_input_cache_view",
                    "canonical_cache_shape": list(canonical_features.shape),
                    "dtype": input_dtype,
                    "input_provenance_sha256": input_provenance_sha256,
                    "canonical_train_rows_sha256": row_list_sha256,
                    "selected_data_sha256": selected_sha256,
                }
            )
        else:
            cache_provenance_sha256 = _sha256(
                cache_provenance_sha256, "cache_provenance_sha256"
            )

        selected64 = selected.astype(np.float64)
        mean64 = selected64.mean(axis=0)
        matrices: list[np.ndarray] = []
        for layer in range(_LAYERS):
            centered = selected64[:, layer, :] - mean64[layer]
            covariance = centered.T @ centered / (len(rows) - 1)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            inverse_sqrt = (np.maximum(eigenvalues, 0.0) + _WHITENING_EPS) ** -0.5
            matrices.append((eigenvectors * inverse_sqrt) @ eigenvectors.T)

        return cls(
            mean=mean64.astype(np.float32),
            matrix=np.stack(matrices).astype(np.float32),
            canonical_train_rows=rows,
            canonical_row_count=int(row_count),
            input_provenance_sha256=input_provenance_sha256,
            cache_provenance_sha256=cache_provenance_sha256,
        )

    @property
    def payload_sha256(self) -> str:
        return self._payload_sha256

    def _payload_body(self) -> dict[str, object]:
        rows = list(self.canonical_train_rows)
        return {
            "artifact_type": self._ARTIFACT_TYPE,
            "schema_version": 1,
            "algorithm": self._ALGORITHM,
            "source_scope": self.source_scope,
            "eps": self.eps,
            "canonical_row_count": self.canonical_row_count,
            "canonical_train_rows": rows,
            "row_list_sha256": sha256_json(rows),
            "input_provenance_sha256": self.input_provenance_sha256,
            "cache_provenance_sha256": self.cache_provenance_sha256,
            "mean": _array_payload(self.mean),
            "matrix": _array_payload(self.matrix),
        }

    def to_payload(self) -> dict[str, object]:
        body = self._payload_body()
        calculated = sha256_json(body)
        if calculated != self._payload_sha256:
            raise ValueError(
                "whitening statistics or metadata changed after payload sealing"
            )
        return {**body, "payload_sha256": calculated}

    @classmethod
    def from_payload(cls, value: object) -> "TrainWhitening":
        payload = _object(value, "whitening")
        _exact_keys(payload, cls._BODY_KEYS | {"payload_sha256"}, "whitening")
        claimed_hash = _sha256(payload["payload_sha256"], "payload_sha256")
        body = {key: payload[key] for key in cls._BODY_KEYS}
        try:
            calculated_hash = sha256_json(body)
        except (TypeError, ValueError) as exc:
            raise ValueError("whitening payload is not canonical JSON") from exc
        if calculated_hash != claimed_hash:
            raise ValueError("whitening payload SHA-256 hash does not match")
        if payload["artifact_type"] != cls._ARTIFACT_TYPE:
            raise ValueError("whitening artifact_type does not match")
        if payload["schema_version"] != 1:
            raise ValueError("unsupported whitening schema_version")
        if payload["algorithm"] != cls._ALGORITHM:
            raise ValueError("whitening algorithm does not match")
        if payload["source_scope"] != "train":
            raise ValueError("whitening source_scope must be train only")
        _locked_float(payload["eps"], _WHITENING_EPS, "whitening eps")

        row_count_value = payload["canonical_row_count"]
        if (
            isinstance(row_count_value, bool)
            or not isinstance(row_count_value, int)
            or row_count_value <= 0
        ):
            raise ValueError("canonical_row_count must be a positive integer")
        raw_rows = payload["canonical_train_rows"]
        if not isinstance(raw_rows, list):
            raise ValueError("canonical_train_rows must be an array")
        rows = _validate_canonical_rows(raw_rows, row_count=row_count_value)
        row_hash = _sha256(payload["row_list_sha256"], "row_list_sha256")
        if sha256_json(list(rows)) != row_hash:
            raise ValueError("canonical train row-list SHA-256 hash does not match")
        input_hash = _sha256(
            payload["input_provenance_sha256"], "input_provenance_sha256"
        )
        cache_hash = _sha256(
            payload["cache_provenance_sha256"], "cache_provenance_sha256"
        )
        mean = _decode_array(payload["mean"], "mean")
        matrix = _decode_array(payload["matrix"], "matrix")
        return cls(
            mean=mean,
            matrix=matrix,
            canonical_train_rows=rows,
            canonical_row_count=row_count_value,
            input_provenance_sha256=input_hash,
            cache_provenance_sha256=cache_hash,
            payload_sha256=claimed_hash,
        )

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        dimension = int(self.mean.shape[1])
        _validate_tensor_features(
            features, feature_dim=dimension, context="whitening features"
        )
        centered = features.to(dtype=torch.float32) - self.mean.to(
            device=features.device, dtype=torch.float32
        )
        transformed = torch.einsum(
            "bld,lde->ble",
            centered,
            self.matrix.to(device=features.device, dtype=torch.float32),
        )
        return transformed.to(dtype=features.dtype)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.transform(features)


def _project_layers(hidden: torch.Tensor, projectors: nn.ModuleList) -> torch.Tensor:
    return torch.stack(
        [projector(hidden[:, layer, :]) for layer, projector in enumerate(projectors)],
        dim=1,
    )


class SharedImagePreProjector(nn.Module):
    """The baseline: one 3200-to-1024 preprojector shared by five layers."""

    def __init__(self) -> None:
        super().__init__()
        self.image_pre_projector = nn.Linear(_IMAGE_DIM, _IMAGE_MID_DIM)
        self.image_projectors = nn.ModuleList(
            nn.Linear(_IMAGE_MID_DIM, _FEATURE_DIM) for _ in range(_LAYERS)
        )

    def preproject(self, features: torch.Tensor) -> torch.Tensor:
        _validate_tensor_features(
            features, feature_dim=_IMAGE_DIM, context="image features"
        )
        return self.image_pre_projector(features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return _project_layers(self.preproject(features), self.image_projectors)


class SeparateImagePreProjector(nn.Module):
    """The candidate: five independent 3200-to-1024 preprojectors."""

    def __init__(self) -> None:
        super().__init__()
        self.image_pre_projectors = nn.ModuleList(
            nn.Linear(_IMAGE_DIM, _IMAGE_MID_DIM) for _ in range(_LAYERS)
        )
        self.image_projectors = nn.ModuleList(
            nn.Linear(_IMAGE_MID_DIM, _FEATURE_DIM) for _ in range(_LAYERS)
        )

    def preproject(self, features: torch.Tensor) -> torch.Tensor:
        _validate_tensor_features(
            features, feature_dim=_IMAGE_DIM, context="image features"
        )
        return torch.stack(
            [
                projector(features[:, layer, :])
                for layer, projector in enumerate(self.image_pre_projectors)
            ],
            dim=1,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return _project_layers(self.preproject(features), self.image_projectors)


@dataclass(frozen=True)
class Stage2PreprocessingRegistry:
    """Preprocessing subset of the exact, sealed Stage 2 candidate registry."""

    layernorm_ids: tuple[str, str]
    whitening_ids: tuple[str, str]
    preprojector_ids: tuple[str, str]
    combination_policy: str
    config_sha256: str


@dataclass(frozen=True)
class Stage2RunnerSelection:
    layernorm_config_id: str
    whitening_config_id: str
    preprojector_config_id: str
    active_factor: str | None


def load_stage2_preprocessing_candidates(
    path: Path | str,
) -> Stage2PreprocessingRegistry:
    config = SemanticConfig.from_path(Path(path))
    payload = config.canonical_payload()
    layernorm_ids = tuple(
        entry["config_id"]
        for entry in payload["layernorm"]  # type: ignore[index]
    )
    whitening_ids = tuple(
        entry["config_id"]
        for entry in payload["whitening"]  # type: ignore[index]
    )
    preprojector_ids = tuple(
        entry["config_id"]
        for entry in payload["preprojectors"]  # type: ignore[index]
    )
    if (
        layernorm_ids != _LAYERNORM_IDS
        or whitening_ids != _WHITENING_IDS
        or preprojector_ids != _PREPROJECTOR_IDS
        or payload["combination_policy"] != "one_factor_only_no_post_hoc_combinations"
    ):
        raise ValueError(
            "stage2 preprocessing config_ids must match the exact locked registry"
        )
    return Stage2PreprocessingRegistry(
        layernorm_ids=_LAYERNORM_IDS,
        whitening_ids=_WHITENING_IDS,
        preprojector_ids=_PREPROJECTOR_IDS,
        combination_policy="one_factor_only_no_post_hoc_combinations",
        config_sha256=config.sha256,
    )


def validate_stage2_runner_config(
    registry: Stage2PreprocessingRegistry,
    *,
    layernorm_config_id: str,
    whitening_config_id: str,
    preprojector_config_id: str,
) -> Stage2RunnerSelection:
    if not isinstance(registry, Stage2PreprocessingRegistry):
        raise TypeError("registry must be a Stage2PreprocessingRegistry")
    selected = (
        ("layernorm", layernorm_config_id, registry.layernorm_ids),
        ("whitening", whitening_config_id, registry.whitening_ids),
        ("preprojector", preprojector_config_id, registry.preprojector_ids),
    )
    for factor, config_id, allowed in selected:
        if not isinstance(config_id, str) or config_id not in allowed:
            raise ValueError(
                f"unknown Stage 2 {factor} config_id: {config_id!r}; "
                f"expected one of {allowed}"
            )
    active = tuple(
        factor
        for factor, config_id, _ in selected
        if config_id in {"s2-layernorm-on", "s2-whitening-on", "s2-preproj-separate"}
    )
    if len(active) > 1:
        raise ValueError(
            "Stage 2 runner allows one active factor only; more than one was enabled"
        )
    return Stage2RunnerSelection(
        layernorm_config_id=layernorm_config_id,
        whitening_config_id=whitening_config_id,
        preprojector_config_id=preprojector_config_id,
        active_factor=active[0] if active else None,
    )
