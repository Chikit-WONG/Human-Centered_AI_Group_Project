"""Strict paired-cell gates for development-only SAMGA experiments."""

from __future__ import annotations

import json
import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import numpy as np

from .config import BootstrapConfig
from .hashing import canonical_json_bytes, sha256_json


CELL_MATRIX_TYPE = "samga_brain_rw.cell_matrix"
PILOT_SUBJECTS = (1, 5, 8)
PILOT_SEEDS = (42, 43)
CONFIRMATION_SUBJECTS = tuple(range(1, 11))
CONFIRMATION_SEEDS = tuple(range(42, 47))
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260719
BOOTSTRAP_RESAMPLING = (
    "independent_subject_and_seed_indices_with_replacement_cartesian_mean"
)
BOOTSTRAP_QUANTILE_METHOD = "linear"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SUBJECT_TEST_RE = re.compile(r"^sub-\d{2}_test\.json$", re.IGNORECASE)
_FORMAL_TEST_RECORD_SHA256 = (
    "02d7e33b3fe8e5a571f8db232ca5fa86abb0c16981876ec84feae7ba64636f1a"
)
_FORBIDDEN_PATH_COMPONENTS = frozenset(
    {"formal", "formal-test", "formal_test", "test", "test_images"}
)
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)

_CELL_KEYS = frozenset(
    {
        "architecture_id",
        "architecture_specific_initialization_sha256",
        "data_order_sha256",
        "effective_batch_size",
        "full_task_initialization_sha256",
        "optimization_steps",
        "scope",
        "seed",
        "shared_parameter_intersection_name",
        "shared_parameter_intersection_sha256",
        "split_role",
        "subject",
        "top1",
        "top5",
    }
)
_MATRIX_KEYS = frozenset(
    {
        "artifact_type",
        "cells",
        "component_sha256s",
        "config_id",
        "config_sha256",
        "hyperparameters_sha256",
        "role",
        "schedule_sha256",
        "schema_version",
    }
)


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a 64-character lowercase SHA-256")
    return value


def _optional_sha256(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _require_sha256(value, field)


def _require_safe_id(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or _SAFE_ID_RE.fullmatch(value) is None
        or value in {".", ".."}
    ):
        raise ValueError(f"{field} must be a safe nonempty identifier")
    return value


def _require_nonempty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty string")
    if "\r" in value or "\n" in value:
        raise ValueError(f"{field} must not contain line breaks")
    return value


def _require_integer(value: object, field: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _require_rate(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be a finite number in [0, 1]")
    return result


def _require_exact_keys(
    value: Mapping[str, object],
    expected: frozenset[str],
    context: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"{context} keys mismatch: missing={missing}, extra={extra}"
        )


@dataclass(frozen=True)
class Cell:
    """One fully bound subject/seed metric cell."""

    subject: int
    seed: int
    top1: float
    top5: float
    effective_batch_size: int
    optimization_steps: int
    scope: str
    split_role: str
    data_order_sha256: str
    architecture_id: str
    full_task_initialization_sha256: str | None
    shared_parameter_intersection_name: str | None = None
    shared_parameter_intersection_sha256: str | None = None
    architecture_specific_initialization_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "subject",
            _require_integer(self.subject, "cell.subject"),
        )
        object.__setattr__(
            self,
            "seed",
            _require_integer(self.seed, "cell.seed"),
        )
        top1 = _require_rate(self.top1, "cell.top1")
        top5 = _require_rate(self.top5, "cell.top5")
        if top5 < top1:
            raise ValueError("cell.top5 must be >= cell.top1")
        object.__setattr__(self, "top1", top1)
        object.__setattr__(self, "top5", top5)
        object.__setattr__(
            self,
            "effective_batch_size",
            _require_integer(
                self.effective_batch_size,
                "cell.effective_batch_size",
            ),
        )
        object.__setattr__(
            self,
            "optimization_steps",
            _require_integer(
                self.optimization_steps,
                "cell.optimization_steps",
            ),
        )
        scope = _require_nonempty(self.scope, "cell.scope")
        split_role = _require_nonempty(self.split_role, "cell.split_role")
        if scope not in {"val-dev", "val-confirm"}:
            raise ValueError("cell scope must be val-dev or val-confirm")
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "split_role", split_role)
        object.__setattr__(
            self,
            "data_order_sha256",
            _require_sha256(
                self.data_order_sha256,
                "cell.data_order_sha256",
            ),
        )
        object.__setattr__(
            self,
            "architecture_id",
            _require_safe_id(self.architecture_id, "cell.architecture_id"),
        )
        object.__setattr__(
            self,
            "full_task_initialization_sha256",
            _optional_sha256(
                self.full_task_initialization_sha256,
                "cell.full_task_initialization_sha256",
            ),
        )
        if self.shared_parameter_intersection_name is not None:
            object.__setattr__(
                self,
                "shared_parameter_intersection_name",
                _require_nonempty(
                    self.shared_parameter_intersection_name,
                    "cell.shared_parameter_intersection_name",
                ),
            )
        object.__setattr__(
            self,
            "shared_parameter_intersection_sha256",
            _optional_sha256(
                self.shared_parameter_intersection_sha256,
                "cell.shared_parameter_intersection_sha256",
            ),
        )
        object.__setattr__(
            self,
            "architecture_specific_initialization_sha256",
            _optional_sha256(
                self.architecture_specific_initialization_sha256,
                "cell.architecture_specific_initialization_sha256",
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "architecture_id": self.architecture_id,
            "architecture_specific_initialization_sha256": (
                self.architecture_specific_initialization_sha256
            ),
            "data_order_sha256": self.data_order_sha256,
            "effective_batch_size": self.effective_batch_size,
            "full_task_initialization_sha256": (
                self.full_task_initialization_sha256
            ),
            "optimization_steps": self.optimization_steps,
            "scope": self.scope,
            "seed": self.seed,
            "shared_parameter_intersection_name": (
                self.shared_parameter_intersection_name
            ),
            "shared_parameter_intersection_sha256": (
                self.shared_parameter_intersection_sha256
            ),
            "split_role": self.split_role,
            "subject": self.subject,
            "top1": self.top1,
            "top5": self.top5,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "Cell":
        _require_exact_keys(payload, _CELL_KEYS, "cell")
        return cls(
            subject=payload["subject"],  # type: ignore[arg-type]
            seed=payload["seed"],  # type: ignore[arg-type]
            top1=payload["top1"],  # type: ignore[arg-type]
            top5=payload["top5"],  # type: ignore[arg-type]
            effective_batch_size=payload["effective_batch_size"],  # type: ignore[arg-type]
            optimization_steps=payload["optimization_steps"],  # type: ignore[arg-type]
            scope=payload["scope"],  # type: ignore[arg-type]
            split_role=payload["split_role"],  # type: ignore[arg-type]
            data_order_sha256=payload["data_order_sha256"],  # type: ignore[arg-type]
            architecture_id=payload["architecture_id"],  # type: ignore[arg-type]
            full_task_initialization_sha256=payload[
                "full_task_initialization_sha256"
            ],  # type: ignore[arg-type]
            shared_parameter_intersection_name=payload[
                "shared_parameter_intersection_name"
            ],  # type: ignore[arg-type]
            shared_parameter_intersection_sha256=payload[
                "shared_parameter_intersection_sha256"
            ],  # type: ignore[arg-type]
            architecture_specific_initialization_sha256=payload[
                "architecture_specific_initialization_sha256"
            ],  # type: ignore[arg-type]
        )


# Descriptive aliases are intentionally kept for downstream callers.
CellMetric = Cell
CellRecord = Cell


@dataclass(frozen=True)
class CellMatrix:
    """A typed compact matrix; never a raw score or prediction fallback."""

    role: Literal["candidate", "control"]
    config_id: str
    config_sha256: str
    hyperparameters_sha256: str
    schedule_sha256: str
    component_sha256s: tuple[str, ...]
    cells: tuple[Cell, ...]

    def __post_init__(self) -> None:
        if self.role not in {"candidate", "control"}:
            raise ValueError("matrix role must be candidate or control")
        object.__setattr__(
            self,
            "config_id",
            _require_safe_id(self.config_id, "matrix.config_id"),
        )
        for field in (
            "config_sha256",
            "hyperparameters_sha256",
            "schedule_sha256",
        ):
            object.__setattr__(
                self,
                field,
                _require_sha256(getattr(self, field), f"matrix.{field}"),
            )
        if not isinstance(self.component_sha256s, tuple):
            object.__setattr__(
                self,
                "component_sha256s",
                tuple(self.component_sha256s),
            )
        components = tuple(
            _require_sha256(value, f"matrix.component_sha256s[{index}]")
            for index, value in enumerate(self.component_sha256s)
        )
        if not components:
            raise ValueError("matrix.component_sha256s must be nonempty")
        if len(set(components)) != len(components):
            raise ValueError("matrix.component_sha256s contains duplicates")
        object.__setattr__(self, "component_sha256s", tuple(sorted(components)))
        if not isinstance(self.cells, tuple):
            object.__setattr__(self, "cells", tuple(self.cells))
        if not self.cells or any(not isinstance(cell, Cell) for cell in self.cells):
            raise ValueError("matrix.cells must be a nonempty sequence of Cell")
        coordinates = [(cell.subject, cell.seed) for cell in self.cells]
        if len(set(coordinates)) != len(coordinates):
            raise ValueError("duplicate subject-seed cell in matrix")
        object.__setattr__(
            self,
            "cells",
            tuple(sorted(self.cells, key=lambda cell: (cell.subject, cell.seed))),
        )

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_document())

    @property
    def scope(self) -> str:
        scopes = {cell.scope for cell in self.cells}
        if len(scopes) != 1:
            raise ValueError("matrix has mixed scopes")
        return next(iter(scopes))

    def to_document(self) -> dict[str, object]:
        return {
            "artifact_type": CELL_MATRIX_TYPE,
            "cells": [cell.to_payload() for cell in self.cells],
            "component_sha256s": list(self.component_sha256s),
            "config_id": self.config_id,
            "config_sha256": self.config_sha256,
            "hyperparameters_sha256": self.hyperparameters_sha256,
            "role": self.role,
            "schedule_sha256": self.schedule_sha256,
            "schema_version": 1,
        }

    @classmethod
    def from_document(cls, document: Mapping[str, object]) -> "CellMatrix":
        _require_exact_keys(document, _MATRIX_KEYS, "cell matrix")
        if document["schema_version"] != 1:
            raise ValueError("cell matrix schema_version must be 1")
        if document["artifact_type"] != CELL_MATRIX_TYPE:
            raise ValueError(
                f"cell matrix artifact_type must be {CELL_MATRIX_TYPE}"
            )
        raw_cells = document["cells"]
        if (
            not isinstance(raw_cells, Sequence)
            or isinstance(raw_cells, (str, bytes, bytearray))
        ):
            raise ValueError("cell matrix cells must be a sequence")
        cells: list[Cell] = []
        for index, value in enumerate(raw_cells):
            if not isinstance(value, Mapping):
                raise ValueError(f"cell matrix cells[{index}] must be an object")
            cells.append(Cell.from_payload(value))
        raw_components = document["component_sha256s"]
        if (
            not isinstance(raw_components, Sequence)
            or isinstance(raw_components, (str, bytes, bytearray))
        ):
            raise ValueError("component_sha256s must be a sequence")
        return cls(
            role=document["role"],  # type: ignore[arg-type]
            config_id=document["config_id"],  # type: ignore[arg-type]
            config_sha256=document["config_sha256"],  # type: ignore[arg-type]
            hyperparameters_sha256=document[
                "hyperparameters_sha256"
            ],  # type: ignore[arg-type]
            schedule_sha256=document["schedule_sha256"],  # type: ignore[arg-type]
            component_sha256s=tuple(raw_components),  # type: ignore[arg-type]
            cells=tuple(cells),
        )


@dataclass(frozen=True)
class InitializationEvidence:
    subject: int
    seed: int
    mode: Literal["full-task", "shared-intersection"]
    candidate_architecture_id: str
    control_architecture_id: str
    full_task_initialization_sha256: str | None = None
    shared_parameter_intersection_name: str | None = None
    shared_parameter_intersection_sha256: str | None = None
    candidate_specific_initialization_sha256: str | None = None
    control_specific_initialization_sha256: str | None = None

    def to_payload(self) -> dict[str, object]:
        return {
            "candidate_architecture_id": self.candidate_architecture_id,
            "candidate_specific_initialization_sha256": (
                self.candidate_specific_initialization_sha256
            ),
            "control_architecture_id": self.control_architecture_id,
            "control_specific_initialization_sha256": (
                self.control_specific_initialization_sha256
            ),
            "full_task_initialization_sha256": (
                self.full_task_initialization_sha256
            ),
            "mode": self.mode,
            "seed": self.seed,
            "shared_parameter_intersection_name": (
                self.shared_parameter_intersection_name
            ),
            "shared_parameter_intersection_sha256": (
                self.shared_parameter_intersection_sha256
            ),
            "subject": self.subject,
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
    ) -> "InitializationEvidence":
        expected = frozenset(cls(  # type: ignore[call-arg]
            subject=1,
            seed=1,
            mode="full-task",
            candidate_architecture_id="x",
            control_architecture_id="x",
        ).to_payload())
        _require_exact_keys(payload, expected, "initialization evidence")
        mode = payload["mode"]
        if mode not in {"full-task", "shared-intersection"}:
            raise ValueError("invalid initialization evidence mode")
        return cls(
            subject=_require_integer(payload["subject"], "evidence.subject"),
            seed=_require_integer(payload["seed"], "evidence.seed"),
            mode=mode,
            candidate_architecture_id=_require_safe_id(
                payload["candidate_architecture_id"],
                "evidence.candidate_architecture_id",
            ),
            control_architecture_id=_require_safe_id(
                payload["control_architecture_id"],
                "evidence.control_architecture_id",
            ),
            full_task_initialization_sha256=_optional_sha256(
                payload["full_task_initialization_sha256"],
                "evidence.full_task_initialization_sha256",
            ),
            shared_parameter_intersection_name=(
                None
                if payload["shared_parameter_intersection_name"] is None
                else _require_nonempty(
                    payload["shared_parameter_intersection_name"],
                    "evidence.shared_parameter_intersection_name",
                )
            ),
            shared_parameter_intersection_sha256=_optional_sha256(
                payload["shared_parameter_intersection_sha256"],
                "evidence.shared_parameter_intersection_sha256",
            ),
            candidate_specific_initialization_sha256=_optional_sha256(
                payload["candidate_specific_initialization_sha256"],
                "evidence.candidate_specific_initialization_sha256",
            ),
            control_specific_initialization_sha256=_optional_sha256(
                payload["control_specific_initialization_sha256"],
                "evidence.control_specific_initialization_sha256",
            ),
        )


@dataclass(frozen=True)
class GateDecision:
    gate_kind: Literal["pilot", "confirmation"]
    stage: int | None
    passed: bool
    mean_top1_delta: float
    mean_top5_delta: float
    ci95: tuple[float, float] | None
    positive_cells: int
    positive_subjects: int
    worst_subject_top1_delta: float
    subject_mean_top1_deltas: tuple[tuple[int, float], ...]
    criteria: Mapping[str, bool]
    initialization_evidence: tuple[InitializationEvidence, ...]

    def __post_init__(self) -> None:
        if self.gate_kind not in {"pilot", "confirmation"}:
            raise ValueError("gate_kind must be pilot or confirmation")
        if self.stage is not None:
            _require_integer(self.stage, "gate.stage")
        if not isinstance(self.passed, bool):
            raise ValueError("gate.passed must be boolean")
        for field in (
            "mean_top1_delta",
            "mean_top5_delta",
            "worst_subject_top1_delta",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"gate.{field} must be finite")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"gate.{field} must be finite")
            object.__setattr__(self, field, value)
        if self.ci95 is not None:
            if len(self.ci95) != 2:
                raise ValueError("gate.ci95 must contain two values")
            ci = tuple(float(value) for value in self.ci95)
            if not all(math.isfinite(value) for value in ci) or ci[0] > ci[1]:
                raise ValueError("gate.ci95 must be finite and ordered")
            object.__setattr__(self, "ci95", ci)
        for field in ("positive_cells", "positive_subjects"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"gate.{field} must be a nonnegative integer")
        subject_means = tuple(
            (
                _require_integer(subject, "gate subject"),
                float(value),
            )
            for subject, value in self.subject_mean_top1_deltas
        )
        if (
            len({subject for subject, _ in subject_means}) != len(subject_means)
            or any(not math.isfinite(value) for _, value in subject_means)
        ):
            raise ValueError("invalid or duplicate gate subject means")
        object.__setattr__(
            self,
            "subject_mean_top1_deltas",
            tuple(sorted(subject_means)),
        )
        if not isinstance(self.criteria, Mapping) or not self.criteria:
            raise ValueError("gate.criteria must be a nonempty mapping")
        criteria: dict[str, bool] = {}
        for key, value in self.criteria.items():
            if not isinstance(key, str) or not key or not isinstance(value, bool):
                raise ValueError("gate criteria must map names to booleans")
            criteria[key] = value
        if self.passed != all(criteria.values()):
            raise ValueError("gate passed flag must equal all criteria")
        object.__setattr__(
            self,
            "criteria",
            MappingProxyType(dict(sorted(criteria.items()))),
        )
        if not isinstance(self.initialization_evidence, tuple):
            object.__setattr__(
                self,
                "initialization_evidence",
                tuple(self.initialization_evidence),
            )
        if any(
            not isinstance(value, InitializationEvidence)
            for value in self.initialization_evidence
        ):
            raise ValueError("invalid initialization evidence")

    def to_payload(self) -> dict[str, object]:
        return {
            "ci95": None if self.ci95 is None else list(self.ci95),
            "criteria": dict(self.criteria),
            "gate_kind": self.gate_kind,
            "initialization_evidence": [
                value.to_payload() for value in self.initialization_evidence
            ],
            "mean_top1_delta": self.mean_top1_delta,
            "mean_top5_delta": self.mean_top5_delta,
            "passed": self.passed,
            "positive_cells": self.positive_cells,
            "positive_subjects": self.positive_subjects,
            "stage": self.stage,
            "subject_mean_top1_deltas": [
                {"mean_top1_delta": value, "subject": subject}
                for subject, value in self.subject_mean_top1_deltas
            ],
            "worst_subject_top1_delta": self.worst_subject_top1_delta,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "GateDecision":
        expected = frozenset(
            {
                "ci95",
                "criteria",
                "gate_kind",
                "initialization_evidence",
                "mean_top1_delta",
                "mean_top5_delta",
                "passed",
                "positive_cells",
                "positive_subjects",
                "stage",
                "subject_mean_top1_deltas",
                "worst_subject_top1_delta",
            }
        )
        _require_exact_keys(payload, expected, "gate decision")
        criteria = payload["criteria"]
        if not isinstance(criteria, Mapping):
            raise ValueError("gate criteria must be an object")
        raw_evidence = payload["initialization_evidence"]
        if (
            not isinstance(raw_evidence, Sequence)
            or isinstance(raw_evidence, (str, bytes, bytearray))
        ):
            raise ValueError("initialization_evidence must be a sequence")
        evidence: list[InitializationEvidence] = []
        for value in raw_evidence:
            if not isinstance(value, Mapping):
                raise ValueError("initialization evidence must be an object")
            evidence.append(InitializationEvidence.from_payload(value))
        raw_subjects = payload["subject_mean_top1_deltas"]
        if (
            not isinstance(raw_subjects, Sequence)
            or isinstance(raw_subjects, (str, bytes, bytearray))
        ):
            raise ValueError("subject_mean_top1_deltas must be a sequence")
        subject_means: list[tuple[int, float]] = []
        for value in raw_subjects:
            if not isinstance(value, Mapping):
                raise ValueError("subject mean must be an object")
            _require_exact_keys(
                value,
                frozenset({"mean_top1_delta", "subject"}),
                "subject mean",
            )
            subject_means.append(
                (
                    value["subject"],  # type: ignore[arg-type]
                    value["mean_top1_delta"],  # type: ignore[arg-type]
                )
            )
        raw_ci = payload["ci95"]
        if raw_ci is None:
            ci95 = None
        elif (
            isinstance(raw_ci, Sequence)
            and not isinstance(raw_ci, (str, bytes, bytearray))
            and len(raw_ci) == 2
        ):
            ci95 = (raw_ci[0], raw_ci[1])  # type: ignore[assignment]
        else:
            raise ValueError("gate ci95 must be null or a pair")
        return cls(
            gate_kind=payload["gate_kind"],  # type: ignore[arg-type]
            stage=payload["stage"],  # type: ignore[arg-type]
            passed=payload["passed"],  # type: ignore[arg-type]
            mean_top1_delta=payload["mean_top1_delta"],  # type: ignore[arg-type]
            mean_top5_delta=payload["mean_top5_delta"],  # type: ignore[arg-type]
            ci95=ci95,  # type: ignore[arg-type]
            positive_cells=payload["positive_cells"],  # type: ignore[arg-type]
            positive_subjects=payload["positive_subjects"],  # type: ignore[arg-type]
            worst_subject_top1_delta=payload[
                "worst_subject_top1_delta"
            ],  # type: ignore[arg-type]
            subject_mean_top1_deltas=tuple(subject_means),
            criteria=criteria,  # type: ignore[arg-type]
            initialization_evidence=tuple(evidence),
        )


def _validate_bootstrap_config(config: BootstrapConfig) -> None:
    if not isinstance(config, BootstrapConfig):
        raise ValueError("bootstrap config must be a BootstrapConfig")
    expected = {
        "samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
        "resampling": BOOTSTRAP_RESAMPLING,
        "quantile_method": BOOTSTRAP_QUANTILE_METHOD,
    }
    if config.canonical_payload() != expected:
        raise ValueError("bootstrap config differs from the locked protocol")


def two_way_cluster_bootstrap(
    delta: np.ndarray,
    config: BootstrapConfig,
) -> tuple[float, float]:
    """Return the locked draw-by-draw subject/seed Cartesian bootstrap CI."""

    _validate_bootstrap_config(config)
    values = np.asarray(delta)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("delta must be a nonempty two-dimensional matrix")
    if values.dtype.kind not in {"f", "i", "u"}:
        raise ValueError("delta must be a numeric matrix")
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("delta must contain only finite values")

    subject_count, seed_count = values.shape
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    for draw_index in range(BOOTSTRAP_SAMPLES):
        subject_indices = rng.integers(
            0,
            subject_count,
            size=subject_count,
        )
        seed_indices = rng.integers(0, seed_count, size=seed_count)
        draws[draw_index] = values[
            np.ix_(subject_indices, seed_indices)
        ].mean(dtype=np.float64)
    low, high = np.quantile(
        draws,
        (0.025, 0.975),
        method=BOOTSTRAP_QUANTILE_METHOD,
    )
    return float(low), float(high)


def _validate_grid(
    matrix: CellMatrix,
    *,
    expected_role: Literal["candidate", "control"],
    expected_scope: Literal["val-dev", "val-confirm"],
    expected_subjects: tuple[int, ...],
    expected_seeds: tuple[int, ...],
) -> dict[tuple[int, int], Cell]:
    if not isinstance(matrix, CellMatrix):
        raise ValueError(f"{expected_role} must be a CellMatrix")
    if matrix.role != expected_role:
        raise ValueError(f"matrix role must be {expected_role}")
    coordinates = [(cell.subject, cell.seed) for cell in matrix.cells]
    if len(set(coordinates)) != len(coordinates):
        raise ValueError("duplicate subject-seed cell")
    actual_subjects = {subject for subject, _ in coordinates}
    actual_seeds = {seed for _, seed in coordinates}
    if actual_subjects != set(expected_subjects):
        raise ValueError(
            "matrix subject set mismatch: "
            f"expected={list(expected_subjects)}, actual={sorted(actual_subjects)}"
        )
    if actual_seeds != set(expected_seeds):
        raise ValueError(
            "matrix seed set mismatch: "
            f"expected={list(expected_seeds)}, actual={sorted(actual_seeds)}"
        )
    expected_coordinates = {
        (subject, seed)
        for subject in expected_subjects
        for seed in expected_seeds
    }
    actual_coordinates = set(coordinates)
    if actual_coordinates != expected_coordinates:
        missing = sorted(expected_coordinates - actual_coordinates)
        extra = sorted(actual_coordinates - expected_coordinates)
        raise ValueError(
            f"matrix has missing/extra cells: missing={missing}, extra={extra}"
        )
    scopes = {cell.scope for cell in matrix.cells}
    if scopes != {expected_scope}:
        raise ValueError(
            f"matrix scope mismatch: expected {expected_scope}, actual={sorted(scopes)}"
        )
    split_roles = {cell.split_role for cell in matrix.cells}
    if split_roles != {expected_scope}:
        raise ValueError(
            "matrix has mixed or incorrect split roles: "
            f"expected {expected_scope}, actual={sorted(split_roles)}"
        )
    architectures = {cell.architecture_id for cell in matrix.cells}
    if len(architectures) != 1:
        raise ValueError("matrix must name one architecture")
    return {
        (cell.subject, cell.seed): cell
        for cell in matrix.cells
    }


def _validate_pairing(
    candidate: CellMatrix,
    control: CellMatrix,
    *,
    expected_scope: Literal["val-dev", "val-confirm"],
    expected_subjects: tuple[int, ...],
    expected_seeds: tuple[int, ...],
) -> tuple[
    tuple[tuple[Cell, Cell], ...],
    tuple[InitializationEvidence, ...],
]:
    candidate_cells = _validate_grid(
        candidate,
        expected_role="candidate",
        expected_scope=expected_scope,
        expected_subjects=expected_subjects,
        expected_seeds=expected_seeds,
    )
    control_cells = _validate_grid(
        control,
        expected_role="control",
        expected_scope=expected_scope,
        expected_subjects=expected_subjects,
        expected_seeds=expected_seeds,
    )
    if candidate.schedule_sha256 != control.schedule_sha256:
        raise ValueError(
            "candidate/control schedule SHA-256 mismatch"
        )
    pairs: list[tuple[Cell, Cell]] = []
    evidence: list[InitializationEvidence] = []
    for subject in expected_subjects:
        for seed in expected_seeds:
            coordinate = (subject, seed)
            candidate_cell = candidate_cells[coordinate]
            control_cell = control_cells[coordinate]
            if (
                candidate_cell.effective_batch_size
                != control_cell.effective_batch_size
            ):
                raise ValueError(
                    f"effective batch mismatch at subject={subject}, seed={seed}"
                )
            if candidate_cell.optimization_steps != control_cell.optimization_steps:
                raise ValueError(
                    f"optimization steps mismatch at subject={subject}, seed={seed}"
                )
            if candidate_cell.scope != control_cell.scope:
                raise ValueError(
                    f"scope mismatch at subject={subject}, seed={seed}"
                )
            if candidate_cell.split_role != control_cell.split_role:
                raise ValueError(
                    f"split role mismatch at subject={subject}, seed={seed}"
                )
            if candidate_cell.data_order_sha256 != control_cell.data_order_sha256:
                raise ValueError(
                    f"data-order hash mismatch at subject={subject}, seed={seed}"
                )

            if candidate_cell.architecture_id == control_cell.architecture_id:
                candidate_hash = candidate_cell.full_task_initialization_sha256
                control_hash = control_cell.full_task_initialization_sha256
                if candidate_hash is None or control_hash is None:
                    raise ValueError(
                        "same architecture requires full task initialization hashes"
                    )
                if candidate_hash != control_hash:
                    raise ValueError(
                        "full task initialization hash mismatch at "
                        f"subject={subject}, seed={seed}"
                    )
                evidence.append(
                    InitializationEvidence(
                        subject=subject,
                        seed=seed,
                        mode="full-task",
                        candidate_architecture_id=candidate_cell.architecture_id,
                        control_architecture_id=control_cell.architecture_id,
                        full_task_initialization_sha256=candidate_hash,
                    )
                )
            else:
                candidate_name = (
                    candidate_cell.shared_parameter_intersection_name
                )
                control_name = control_cell.shared_parameter_intersection_name
                if not candidate_name or not control_name or candidate_name != control_name:
                    raise ValueError(
                        "different architectures require the same named "
                        "shared-parameter intersection"
                    )
                candidate_shared = (
                    candidate_cell.shared_parameter_intersection_sha256
                )
                control_shared = (
                    control_cell.shared_parameter_intersection_sha256
                )
                if (
                    candidate_shared is None
                    or control_shared is None
                    or candidate_shared != control_shared
                ):
                    raise ValueError(
                        "shared-parameter intersection hash mismatch at "
                        f"subject={subject}, seed={seed}"
                    )
                candidate_specific = (
                    candidate_cell.architecture_specific_initialization_sha256
                )
                control_specific = (
                    control_cell.architecture_specific_initialization_sha256
                )
                if candidate_specific is None or control_specific is None:
                    raise ValueError(
                        "different architectures require separate specific "
                        "initialization hashes"
                    )
                evidence.append(
                    InitializationEvidence(
                        subject=subject,
                        seed=seed,
                        mode="shared-intersection",
                        candidate_architecture_id=candidate_cell.architecture_id,
                        control_architecture_id=control_cell.architecture_id,
                        shared_parameter_intersection_name=candidate_name,
                        shared_parameter_intersection_sha256=candidate_shared,
                        candidate_specific_initialization_sha256=(
                            candidate_specific
                        ),
                        control_specific_initialization_sha256=control_specific,
                    )
                )
            pairs.append((candidate_cell, control_cell))
    return tuple(pairs), tuple(evidence)


def _decimal_delta(candidate: float, control: float) -> Decimal:
    return Decimal(str(candidate)) - Decimal(str(control))


def _paired_summary(
    pairs: tuple[tuple[Cell, Cell], ...],
    subjects: tuple[int, ...],
    seeds: tuple[int, ...],
) -> tuple[
    np.ndarray,
    np.ndarray,
    Decimal,
    Decimal,
    int,
    tuple[tuple[int, Decimal], ...],
]:
    top1 = np.empty((len(subjects), len(seeds)), dtype=np.float64)
    top5 = np.empty((len(subjects), len(seeds)), dtype=np.float64)
    top1_decimal: list[Decimal] = []
    top5_decimal: list[Decimal] = []
    positive_cells = 0
    index = 0
    for subject_index, _subject in enumerate(subjects):
        for seed_index, _seed in enumerate(seeds):
            candidate, control = pairs[index]
            delta_top1 = _decimal_delta(candidate.top1, control.top1)
            delta_top5 = _decimal_delta(candidate.top5, control.top5)
            top1[subject_index, seed_index] = float(delta_top1)
            top5[subject_index, seed_index] = float(delta_top5)
            top1_decimal.append(delta_top1)
            top5_decimal.append(delta_top5)
            if delta_top1 > 0:
                positive_cells += 1
            index += 1
    cell_count = Decimal(len(top1_decimal))
    mean_top1 = sum(top1_decimal, Decimal(0)) / cell_count
    mean_top5 = sum(top5_decimal, Decimal(0)) / cell_count
    subject_means: list[tuple[int, Decimal]] = []
    for subject_index, subject in enumerate(subjects):
        values = [
            Decimal(str(top1[subject_index, seed_index]))
            for seed_index in range(len(seeds))
        ]
        subject_means.append(
            (
                subject,
                sum(values, Decimal(0)) / Decimal(len(values)),
            )
        )
    return (
        top1,
        top5,
        mean_top1,
        mean_top5,
        positive_cells,
        tuple(subject_means),
    )


def pilot_gate(
    candidate: CellMatrix,
    control: CellMatrix,
    stage: int,
) -> GateDecision:
    """Apply the exact six-cell Stage 1/other-stage pilot gate."""

    stage = _require_integer(stage, "stage")
    if stage > 5:
        raise ValueError("stage must be in 1..5")
    pairs, evidence = _validate_pairing(
        candidate,
        control,
        expected_scope="val-dev",
        expected_subjects=PILOT_SUBJECTS,
        expected_seeds=PILOT_SEEDS,
    )
    (
        _top1,
        _top5,
        mean_top1,
        mean_top5,
        positive_cells,
        subject_means,
    ) = _paired_summary(pairs, PILOT_SUBJECTS, PILOT_SEEDS)
    minimum_top1 = Decimal("0.003" if stage == 1 else "0.005")
    worst_subject = min(value for _, value in subject_means)
    criteria = {
        "mean_top1_delta": mean_top1 >= minimum_top1,
        "mean_top5_delta": mean_top5 >= Decimal("-0.002"),
        "positive_cells": positive_cells >= 4,
        "subject_floor": worst_subject >= Decimal("-0.02"),
    }
    positive_subjects = sum(value > 0 for _, value in subject_means)
    return GateDecision(
        gate_kind="pilot",
        stage=stage,
        passed=all(criteria.values()),
        mean_top1_delta=float(mean_top1),
        mean_top5_delta=float(mean_top5),
        ci95=None,
        positive_cells=positive_cells,
        positive_subjects=positive_subjects,
        worst_subject_top1_delta=float(worst_subject),
        subject_mean_top1_deltas=tuple(
            (subject, float(value)) for subject, value in subject_means
        ),
        criteria=criteria,
        initialization_evidence=evidence,
    )


def confirmation_gate(
    candidate: CellMatrix,
    control: CellMatrix,
    bootstrap: BootstrapConfig,
) -> GateDecision:
    """Apply the exact complete 10-by-5 val-confirm gate."""

    pairs, evidence = _validate_pairing(
        candidate,
        control,
        expected_scope="val-confirm",
        expected_subjects=CONFIRMATION_SUBJECTS,
        expected_seeds=CONFIRMATION_SEEDS,
    )
    (
        top1,
        _top5,
        mean_top1,
        mean_top5,
        positive_cells,
        subject_means,
    ) = _paired_summary(
        pairs,
        CONFIRMATION_SUBJECTS,
        CONFIRMATION_SEEDS,
    )
    ci95 = two_way_cluster_bootstrap(top1, bootstrap)
    worst_subject = min(value for _, value in subject_means)
    positive_subjects = sum(value > 0 for _, value in subject_means)
    criteria = {
        "ci95_lower": ci95[0] > 0.0,
        "mean_top1_delta": mean_top1 >= Decimal("0.005"),
        "mean_top5_delta": mean_top5 >= Decimal("-0.002"),
        "positive_subjects": positive_subjects >= 8,
        "subject_floor": worst_subject >= Decimal("-0.02"),
    }
    return GateDecision(
        gate_kind="confirmation",
        stage=None,
        passed=all(criteria.values()),
        mean_top1_delta=float(mean_top1),
        mean_top5_delta=float(mean_top5),
        ci95=ci95,
        positive_cells=positive_cells,
        positive_subjects=positive_subjects,
        worst_subject_top1_delta=float(worst_subject),
        subject_mean_top1_deltas=tuple(
            (subject, float(value)) for subject, value in subject_means
        ),
        criteria=criteria,
        initialization_evidence=evidence,
    )


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_nonfinite_json(value: str) -> object:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


def validate_development_path(
    path: Path,
    *,
    allowed_suffixes: frozenset[str],
) -> Path:
    raw = os.fspath(path)
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise ValueError("development artifact path is invalid")
    absolute = Path(os.path.abspath(os.path.normpath(raw)))
    lowered = raw.lower()
    components: list[str] = []
    for component in (*Path(raw).parts, *absolute.parts):
        components.extend(
            part for part in re.split(r"[\\/]+", component) if part
        )
    if (
        _FORMAL_TEST_RECORD_SHA256 in lowered
        or any(_SUBJECT_TEST_RE.fullmatch(value) for value in components)
        or any(
            value.lower() in _FORBIDDEN_PATH_COMPONENTS
            for value in components
        )
    ):
        raise ValueError("formal/test artifact path is forbidden")
    if absolute.suffix.lower() not in allowed_suffixes:
        raise ValueError(
            "typed compact input/output has an unsupported file suffix"
        )
    if not absolute.name or absolute == absolute.parent:
        raise ValueError("development artifact path must name a file")
    return absolute


def _open_directory_nofollow(path: Path, *, create: bool) -> int:
    absolute = Path(os.path.abspath(os.fspath(path)))
    flags = os.O_RDONLY | _O_DIRECTORY | _O_CLOEXEC | _O_NOFOLLOW
    descriptor = os.open(absolute.anchor, flags)
    try:
        for component in absolute.parts[1:]:
            try:
                next_descriptor = os.open(
                    component,
                    flags,
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(component, mode=0o700, dir_fd=descriptor)
                os.fsync(descriptor)
                next_descriptor = os.open(
                    component,
                    flags,
                    dir_fd=descriptor,
                )
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def read_development_bytes(
    path: Path,
    *,
    allowed_suffixes: frozenset[str],
    maximum_bytes: int = 32 * 1024 * 1024,
) -> bytes:
    absolute = validate_development_path(
        path,
        allowed_suffixes=allowed_suffixes,
    )
    parent_fd = _open_directory_nofollow(absolute.parent, create=False)
    descriptor = -1
    try:
        descriptor = os.open(
            absolute.name,
            os.O_RDONLY | _O_CLOEXEC | _O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("development input must be a regular file")
        if metadata.st_size > maximum_bytes:
            raise ValueError("development input exceeds the byte limit")
        chunks: list[bytes] = []
        remaining = maximum_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        after = os.fstat(descriptor)
        if (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError("development input changed during read")
        if len(data) != metadata.st_size:
            raise ValueError("development input byte count changed during read")
        return data
    except OSError as exc:
        raise ValueError(
            f"cannot read development input without symlinks: {absolute}"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)


def load_strict_json_object(path: Path) -> dict[str, object]:
    data = read_development_bytes(
        path,
        allowed_suffixes=frozenset({".json"}),
    )
    try:
        decoded = data.decode("utf-8")
        payload = json.loads(
            decoded,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("typed compact input is not strict UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("typed compact input root must be an object")
    return payload


def load_cell_matrix(
    path: Path,
    *,
    expected_role: Literal["candidate", "control"],
) -> CellMatrix:
    matrix = CellMatrix.from_document(load_strict_json_object(path))
    if matrix.role != expected_role:
        raise ValueError(f"matrix role must be {expected_role}")
    return matrix


def write_development_json_exclusive(path: Path, payload: object) -> None:
    absolute = validate_development_path(
        path,
        allowed_suffixes=frozenset({".json"}),
    )
    data = canonical_json_bytes(payload) + b"\n"
    parent_fd = _open_directory_nofollow(absolute.parent, create=True)
    descriptor = -1
    created = False
    try:
        descriptor = os.open(
            absolute.name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | _O_CLOEXEC
            | _O_NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
        created = True
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.fsync(parent_fd)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        if created:
            try:
                os.unlink(absolute.name, dir_fd=parent_fd)
                os.fsync(parent_fd)
            except OSError:
                pass
        raise
    finally:
        os.close(parent_fd)


__all__ = [
    "BOOTSTRAP_QUANTILE_METHOD",
    "BOOTSTRAP_RESAMPLING",
    "BOOTSTRAP_SAMPLES",
    "BOOTSTRAP_SEED",
    "CELL_MATRIX_TYPE",
    "CONFIRMATION_SEEDS",
    "CONFIRMATION_SUBJECTS",
    "Cell",
    "CellMatrix",
    "CellMetric",
    "CellRecord",
    "GateDecision",
    "InitializationEvidence",
    "PILOT_SEEDS",
    "PILOT_SUBJECTS",
    "confirmation_gate",
    "load_cell_matrix",
    "load_strict_json_object",
    "pilot_gate",
    "read_development_bytes",
    "two_way_cluster_bootstrap",
    "validate_development_path",
    "write_development_json_exclusive",
]
