"""Strict, immutable configuration and semantic run identity.

Only JSON-compatible values enter a semantic payload.  Canonical hashes use
sorted, compact UTF-8 JSON.  Generated artifact/log/result paths are parsed
strictly but deliberately excluded from semantic identity.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .contracts import stage1_v1_payload, stage2_v1_payload


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top-level JSON value must be an object")
    return value


def _object(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{context} keys must be strings")
        result[key] = item
    return result


def _keys(
    value: Mapping[str, object],
    required: set[str],
    context: str,
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    actual = set(value)
    unknown = actual - required - optional
    missing = required - actual
    if unknown:
        raise ValueError(f"{context} has unknown keys: {sorted(unknown)}")
    if missing:
        raise ValueError(f"{context} is missing keys: {sorted(missing)}")


def _string(value: object, context: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a string")
    return value


def _boolean(value: object, context: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{context} must be a boolean")
    return value


def _integer(value: object, context: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{context} must be an integer")
    return value


def _number(value: object, context: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{context} must be a finite number")
    return float(value)


def _sequence(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be an array")
    return value


def _string_tuple(value: object, context: str) -> tuple[str, ...]:
    return tuple(
        _string(item, f"{context}[{index}]")
        for index, item in enumerate(_sequence(value, context))
    )


def _integer_tuple(value: object, context: str) -> tuple[int, ...]:
    return tuple(
        _integer(item, f"{context}[{index}]")
        for index, item in enumerate(_sequence(value, context))
    )


def _sha256(value: object, context: str) -> str:
    digest = _string(value, context)
    if not _SHA256_RE.fullmatch(digest):
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return digest


def _git_revision(value: object, context: str) -> str:
    revision = _string(value, context)
    if not _GIT_REVISION_RE.fullmatch(revision):
        raise ValueError(f"{context} must be a lowercase 40-hex git revision")
    return revision


def _identifier(value: object, context: str) -> str:
    identifier = _string(value, context)
    if not _ID_RE.fullmatch(identifier):
        raise ValueError(f"{context} is not a safe identifier")
    return identifier


def _jsonable(value: object, context: str = "value") -> object:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{context} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        return {
            _string(key, f"{context} key"): _jsonable(item, f"{context}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _jsonable(item, f"{context}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ValueError(f"{context} is not JSON-compatible")


def _canonical_json(value: object) -> str:
    normalized = _jsonable(value)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _hash_payload(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SplitSizes:
    train: int
    val_dev: int
    val_confirm: int

    def canonical_payload(self) -> dict[str, int]:
        return {
            "train": self.train,
            "val-dev": self.val_dev,
            "val-confirm": self.val_confirm,
        }


@dataclass(frozen=True)
class PilotGate:
    stage1_min_top1_delta: float
    other_min_top1_delta: float
    minimum_positive_cells: int
    minimum_top5_delta: float
    minimum_subject_mean_top1_delta: float

    def canonical_payload(self) -> dict[str, object]:
        return {
            "stage1_min_top1_delta": self.stage1_min_top1_delta,
            "other_min_top1_delta": self.other_min_top1_delta,
            "minimum_positive_cells": self.minimum_positive_cells,
            "minimum_top5_delta": self.minimum_top5_delta,
            "minimum_subject_mean_top1_delta": (
                self.minimum_subject_mean_top1_delta
            ),
        }


@dataclass(frozen=True)
class ConfirmationGate:
    minimum_top1_delta: float
    ci95_lower_must_exceed: float
    minimum_top5_delta: float
    minimum_positive_subjects: int
    minimum_subject_mean_top1_delta: float

    def canonical_payload(self) -> dict[str, object]:
        return {
            "minimum_top1_delta": self.minimum_top1_delta,
            "ci95_lower_must_exceed": self.ci95_lower_must_exceed,
            "minimum_top5_delta": self.minimum_top5_delta,
            "minimum_positive_subjects": self.minimum_positive_subjects,
            "minimum_subject_mean_top1_delta": (
                self.minimum_subject_mean_top1_delta
            ),
        }


@dataclass(frozen=True)
class BootstrapConfig:
    samples: int
    seed: int
    resampling: str
    quantile_method: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "samples": self.samples,
            "seed": self.seed,
            "resampling": self.resampling,
            "quantile_method": self.quantile_method,
        }


@dataclass(frozen=True)
class RetrievalConfig:
    method: str
    similarity: str
    assignment: str
    hungarian: bool

    def canonical_payload(self) -> dict[str, object]:
        return {
            "method": self.method,
            "similarity": self.similarity,
            "assignment": self.assignment,
            "hungarian": self.hungarian,
        }


@dataclass(frozen=True)
class OutputPaths:
    artifacts: str
    logs: str
    results: str

    def canonical_payload(self) -> dict[str, str]:
        return {
            "artifacts": self.artifacts,
            "logs": self.logs,
            "results": self.results,
        }


@dataclass(frozen=True)
class ProtocolConfig:
    schema_version: int
    split_salt: str
    stimulus_salt: str
    expected_non_test_concepts: int
    split_sizes: SplitSizes
    pilot_subjects: tuple[int, ...]
    pilot_seeds: tuple[int, ...]
    confirmation_subjects: tuple[int, ...]
    confirmation_seeds: tuple[int, ...]
    historical_top1: float
    historical_top5: float
    paper_top1: float
    paper_top5: float
    pilot_gate: PilotGate
    confirmation_gate: ConfirmationGate
    bootstrap: BootstrapConfig
    retrieval: RetrievalConfig
    output_paths: OutputPaths

    @classmethod
    def from_path(cls, path: Path) -> "ProtocolConfig":
        payload = _load_json_object(Path(path))
        _keys(
            payload,
            {
                "schema_version",
                "split_salt",
                "stimulus_salt",
                "expected_non_test_concepts",
                "split_sizes",
                "pilot_subjects",
                "pilot_seeds",
                "confirmation_subjects",
                "confirmation_seeds",
                "historical_top1",
                "historical_top5",
                "paper_top1",
                "paper_top5",
                "pilot_gate",
                "confirmation_gate",
                "bootstrap",
                "retrieval",
                "output_paths",
            },
            "protocol",
        )
        schema_version = _integer(payload["schema_version"], "schema_version")
        if schema_version != 1:
            raise ValueError(f"unsupported protocol schema_version: {schema_version}")

        split_sizes = _object(payload["split_sizes"], "split_sizes")
        _keys(split_sizes, {"train", "val-dev", "val-confirm"}, "split_sizes")
        pilot_gate = _object(payload["pilot_gate"], "pilot_gate")
        _keys(
            pilot_gate,
            {
                "stage1_min_top1_delta",
                "other_min_top1_delta",
                "minimum_positive_cells",
                "minimum_top5_delta",
                "minimum_subject_mean_top1_delta",
            },
            "pilot_gate",
        )
        confirmation_gate = _object(
            payload["confirmation_gate"], "confirmation_gate"
        )
        _keys(
            confirmation_gate,
            {
                "minimum_top1_delta",
                "ci95_lower_must_exceed",
                "minimum_top5_delta",
                "minimum_positive_subjects",
                "minimum_subject_mean_top1_delta",
            },
            "confirmation_gate",
        )
        bootstrap = _object(payload["bootstrap"], "bootstrap")
        _keys(
            bootstrap,
            {"samples", "seed", "resampling", "quantile_method"},
            "bootstrap",
        )
        retrieval = _object(payload["retrieval"], "retrieval")
        _keys(
            retrieval,
            {"method", "similarity", "assignment", "hungarian"},
            "retrieval",
        )
        output_paths = _object(payload["output_paths"], "output_paths")
        _keys(
            output_paths,
            {"artifacts", "logs", "results"},
            "output_paths",
        )

        return cls(
            schema_version=schema_version,
            split_salt=_string(payload["split_salt"], "split_salt"),
            stimulus_salt=_string(payload["stimulus_salt"], "stimulus_salt"),
            expected_non_test_concepts=_integer(
                payload["expected_non_test_concepts"],
                "expected_non_test_concepts",
            ),
            split_sizes=SplitSizes(
                train=_integer(split_sizes["train"], "split_sizes.train"),
                val_dev=_integer(
                    split_sizes["val-dev"], "split_sizes.val-dev"
                ),
                val_confirm=_integer(
                    split_sizes["val-confirm"], "split_sizes.val-confirm"
                ),
            ),
            pilot_subjects=_integer_tuple(
                payload["pilot_subjects"], "pilot_subjects"
            ),
            pilot_seeds=_integer_tuple(payload["pilot_seeds"], "pilot_seeds"),
            confirmation_subjects=_integer_tuple(
                payload["confirmation_subjects"], "confirmation_subjects"
            ),
            confirmation_seeds=_integer_tuple(
                payload["confirmation_seeds"], "confirmation_seeds"
            ),
            historical_top1=_number(
                payload["historical_top1"], "historical_top1"
            ),
            historical_top5=_number(
                payload["historical_top5"], "historical_top5"
            ),
            paper_top1=_number(payload["paper_top1"], "paper_top1"),
            paper_top5=_number(payload["paper_top5"], "paper_top5"),
            pilot_gate=PilotGate(
                stage1_min_top1_delta=_number(
                    pilot_gate["stage1_min_top1_delta"],
                    "pilot_gate.stage1_min_top1_delta",
                ),
                other_min_top1_delta=_number(
                    pilot_gate["other_min_top1_delta"],
                    "pilot_gate.other_min_top1_delta",
                ),
                minimum_positive_cells=_integer(
                    pilot_gate["minimum_positive_cells"],
                    "pilot_gate.minimum_positive_cells",
                ),
                minimum_top5_delta=_number(
                    pilot_gate["minimum_top5_delta"],
                    "pilot_gate.minimum_top5_delta",
                ),
                minimum_subject_mean_top1_delta=_number(
                    pilot_gate["minimum_subject_mean_top1_delta"],
                    "pilot_gate.minimum_subject_mean_top1_delta",
                ),
            ),
            confirmation_gate=ConfirmationGate(
                minimum_top1_delta=_number(
                    confirmation_gate["minimum_top1_delta"],
                    "confirmation_gate.minimum_top1_delta",
                ),
                ci95_lower_must_exceed=_number(
                    confirmation_gate["ci95_lower_must_exceed"],
                    "confirmation_gate.ci95_lower_must_exceed",
                ),
                minimum_top5_delta=_number(
                    confirmation_gate["minimum_top5_delta"],
                    "confirmation_gate.minimum_top5_delta",
                ),
                minimum_positive_subjects=_integer(
                    confirmation_gate["minimum_positive_subjects"],
                    "confirmation_gate.minimum_positive_subjects",
                ),
                minimum_subject_mean_top1_delta=_number(
                    confirmation_gate["minimum_subject_mean_top1_delta"],
                    "confirmation_gate.minimum_subject_mean_top1_delta",
                ),
            ),
            bootstrap=BootstrapConfig(
                samples=_integer(bootstrap["samples"], "bootstrap.samples"),
                seed=_integer(bootstrap["seed"], "bootstrap.seed"),
                resampling=_string(
                    bootstrap["resampling"], "bootstrap.resampling"
                ),
                quantile_method=_string(
                    bootstrap["quantile_method"], "bootstrap.quantile_method"
                ),
            ),
            retrieval=RetrievalConfig(
                method=_string(retrieval["method"], "retrieval.method"),
                similarity=_string(
                    retrieval["similarity"], "retrieval.similarity"
                ),
                assignment=_string(
                    retrieval["assignment"], "retrieval.assignment"
                ),
                hungarian=_boolean(
                    retrieval["hungarian"], "retrieval.hungarian"
                ),
            ),
            output_paths=OutputPaths(
                artifacts=_string(
                    output_paths["artifacts"], "output_paths.artifacts"
                ),
                logs=_string(output_paths["logs"], "output_paths.logs"),
                results=_string(
                    output_paths["results"], "output_paths.results"
                ),
            ),
        )

    def canonical_payload(self) -> dict[str, object]:
        """Return semantic protocol data, excluding generated output paths."""
        return {
            "schema_version": self.schema_version,
            "split_salt": self.split_salt,
            "stimulus_salt": self.stimulus_salt,
            "expected_non_test_concepts": self.expected_non_test_concepts,
            "split_sizes": self.split_sizes.canonical_payload(),
            "pilot_subjects": list(self.pilot_subjects),
            "pilot_seeds": list(self.pilot_seeds),
            "confirmation_subjects": list(self.confirmation_subjects),
            "confirmation_seeds": list(self.confirmation_seeds),
            "historical_top1": self.historical_top1,
            "historical_top5": self.historical_top5,
            "paper_top1": self.paper_top1,
            "paper_top5": self.paper_top5,
            "pilot_gate": self.pilot_gate.canonical_payload(),
            "confirmation_gate": self.confirmation_gate.canonical_payload(),
            "bootstrap": self.bootstrap.canonical_payload(),
            "retrieval": self.retrieval.canonical_payload(),
        }

    @property
    def sha256(self) -> str:
        return _hash_payload(self.canonical_payload())


def _validate_sha_mapping(value: object, context: str) -> None:
    mapping = _object(value, context)
    if not mapping:
        raise ValueError(f"{context} must not be empty")
    for key, digest in mapping.items():
        _string(key, f"{context} key")
        _sha256(digest, f"{context}.{key}")


def _validate_internvit(payload: dict[str, object]) -> None:
    _keys(
        payload,
        {"schema_version", "config_type", "config_id", "upstream", "model", "cache", "task"},
        "internvit config",
    )
    upstream = _object(payload["upstream"], "upstream")
    _keys(upstream, {"path", "git_commit"}, "upstream")
    _string(upstream["path"], "upstream.path")
    _git_revision(upstream["git_commit"], "upstream.git_commit")

    model = _object(payload["model"], "model")
    _keys(
        model,
        {
            "repo",
            "revision",
            "path",
            "config_sha256",
            "preprocessor_sha256",
            "weight_sha256",
        },
        "model",
    )
    for key in ("repo", "path"):
        _string(model[key], f"model.{key}")
    _git_revision(model["revision"], "model.revision")
    _sha256(model["config_sha256"], "model.config_sha256")
    _sha256(model["preprocessor_sha256"], "model.preprocessor_sha256")
    weights = _object(model["weight_sha256"], "model.weight_sha256")
    _keys(
        weights,
        {
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
        },
        "model.weight_sha256",
    )
    _validate_sha_mapping(weights, "model.weight_sha256")

    cache = _object(payload["cache"], "cache")
    _keys(
        cache,
        {
            "path",
            "sha256",
            "generator_git_revision",
            "canonical_train_manifest_sha256",
            "shape",
            "dtype",
            "layer_route",
            "pooling",
            "normalization",
        },
        "cache",
    )
    for key in ("path", "dtype", "layer_route", "pooling", "normalization"):
        _string(cache[key], f"cache.{key}")
    for key in (
        "sha256",
        "canonical_train_manifest_sha256",
    ):
        _sha256(cache[key], f"cache.{key}")
    _git_revision(cache["generator_git_revision"], "cache.generator_git_revision")
    _integer_tuple(cache["shape"], "cache.shape")

    task = _object(payload["task"], "task")
    _keys(
        task,
        {
            "layer_ids",
            "image_dim",
            "prior_center",
            "router_eval_mode",
            "force_global",
            "channels",
            "trial_averaging",
            "smooth_probability",
            "batch_size",
            "epochs",
            "stage1_epochs",
            "stage1_learning_rate",
            "stage2_learning_rate",
            "mmd_start",
            "mmd_end",
            "image_l2_normalization",
            "eeg_l2_normalization",
        },
        "task",
    )
    _integer_tuple(task["layer_ids"], "task.layer_ids")
    _string_tuple(task["channels"], "task.channels")
    for key in (
        "image_dim",
        "prior_center",
        "trial_averaging",
        "batch_size",
        "epochs",
        "stage1_epochs",
    ):
        _integer(task[key], f"task.{key}")
    for key in (
        "smooth_probability",
        "stage1_learning_rate",
        "stage2_learning_rate",
        "mmd_start",
        "mmd_end",
    ):
        _number(task[key], f"task.{key}")
    _string(task["router_eval_mode"], "task.router_eval_mode")
    _boolean(task["force_global"], "task.force_global")
    _boolean(task["image_l2_normalization"], "task.image_l2_normalization")
    _boolean(task["eeg_l2_normalization"], "task.eeg_l2_normalization")


def _validate_brainrw(payload: dict[str, object]) -> None:
    _keys(
        payload,
        {
            "schema_version",
            "config_type",
            "config_id",
            "clip",
            "brain_mlp",
            "lora",
            "optimizer",
            "training",
        },
        "brainrw config",
    )
    schemas = {
        "clip": {"model_id", "path", "config_sha256", "weights_sha256"},
        "brain_mlp": {"dropout"},
        "lora": {"targets", "rank", "alpha", "dropout"},
        "optimizer": {
            "name",
            "brain_learning_rate",
            "visual_learning_rate",
            "weight_decay",
            "schedule",
        },
        "training": {
            "epochs",
            "epoch_policy",
            "gradient_checkpointing",
            "precision",
            "batch_size",
            "trial_averaging",
            "channels",
        },
    }
    sections = {
        name: _object(payload[name], name)
        for name in schemas
    }
    for name, expected_keys in schemas.items():
        _keys(sections[name], expected_keys, name)
    for key in ("model_id", "path"):
        _string(sections["clip"][key], f"clip.{key}")
    _sha256(sections["clip"]["config_sha256"], "clip.config_sha256")
    _sha256(sections["clip"]["weights_sha256"], "clip.weights_sha256")
    _number(sections["brain_mlp"]["dropout"], "brain_mlp.dropout")
    _string_tuple(sections["lora"]["targets"], "lora.targets")
    _integer(sections["lora"]["rank"], "lora.rank")
    _integer(sections["lora"]["alpha"], "lora.alpha")
    _number(sections["lora"]["dropout"], "lora.dropout")
    for key in ("name", "schedule"):
        _string(sections["optimizer"][key], f"optimizer.{key}")
    for key in (
        "brain_learning_rate",
        "visual_learning_rate",
        "weight_decay",
    ):
        _number(sections["optimizer"][key], f"optimizer.{key}")
    _boolean(
        sections["training"]["gradient_checkpointing"],
        "training.gradient_checkpointing",
    )
    for key in ("epochs", "batch_size", "trial_averaging"):
        _integer(sections["training"][key], f"training.{key}")
    for key in ("epoch_policy", "precision"):
        _string(sections["training"][key], f"training.{key}")
    _string_tuple(sections["training"]["channels"], "training.channels")


def _validate_stage1_structure(payload: dict[str, object]) -> None:
    _keys(
        payload,
        {
            "schema_version",
            "config_type",
            "config_id",
            "selection",
            "formulas",
            "candidates",
        },
        "stage1 config",
    )
    selection = _object(payload["selection"], "selection")
    _keys(
        selection,
        {
            "scope",
            "retrieval",
            "zscore_variance",
            "constant_row",
            "temperature_softmax",
            "branch_score_tie_break",
            "final_score_tie_break",
            "metric_tie_break",
        },
        "selection",
    )
    for key in (
        "scope",
        "retrieval",
        "zscore_variance",
        "constant_row",
        "branch_score_tie_break",
        "final_score_tie_break",
    ):
        _string(selection[key], f"selection.{key}")
    _boolean(selection["temperature_softmax"], "selection.temperature_softmax")
    _string_tuple(selection["metric_tie_break"], "selection.metric_tie_break")
    formulas = _object(payload["formulas"], "formulas")
    _keys(formulas, {"zscore_convex", "temperature_convex", "rrf"}, "formulas")
    for key, value in formulas.items():
        _string(value, f"formulas.{key}")

    candidates = _sequence(payload["candidates"], "candidates")
    identifiers: set[str] = set()
    for index, raw in enumerate(candidates):
        context = f"candidates[{index}]"
        entry = _object(raw, context)
        family = _string(entry.get("family"), f"{context}.family")
        common = {"config_id", "family", "formula"}
        if family == "zscore_convex":
            expected = common | {"alpha"}
        elif family == "temperature_convex":
            expected = common | {
                "internvit_temperature",
                "clip_temperature",
                "alpha",
            }
        elif family == "rrf":
            expected = common | {
                "k",
                "internvit_weight",
                "rank_origin",
                "score_tie_break",
                "final_tie_break",
            }
        else:
            raise ValueError(f"{context}.family is unsupported: {family}")
        _keys(entry, expected, context)
        identifier = _identifier(entry["config_id"], f"{context}.config_id")
        if identifier in identifiers:
            raise ValueError(f"duplicate candidate config_id: {identifier}")
        identifiers.add(identifier)
        _string(entry["formula"], f"{context}.formula")
        if family == "zscore_convex":
            _number(entry["alpha"], f"{context}.alpha")
        elif family == "temperature_convex":
            for key in ("internvit_temperature", "clip_temperature", "alpha"):
                _number(entry[key], f"{context}.{key}")
        else:
            _integer(entry["k"], f"{context}.k")
            _number(entry["internvit_weight"], f"{context}.internvit_weight")
            _integer(entry["rank_origin"], f"{context}.rank_origin")
            _string(entry["score_tie_break"], f"{context}.score_tie_break")
            _string(entry["final_tie_break"], f"{context}.final_tie_break")


def _validate_entry_list(
    value: object,
    context: str,
    required: set[str],
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for index, raw in enumerate(_sequence(value, context)):
        entry = _object(raw, f"{context}[{index}]")
        _keys(entry, required, f"{context}[{index}]")
        _identifier(entry["config_id"], f"{context}[{index}].config_id")
        entries.append(entry)
    return entries


def _validate_stage2_structure(payload: dict[str, object]) -> None:
    _keys(
        payload,
        {
            "schema_version",
            "config_type",
            "config_id",
            "combination_policy",
            "layernorm",
            "whitening",
            "preprojectors",
            "checkpoint_averaging",
            "feature_adapter",
        },
        "stage2 config",
    )
    _string(payload["combination_policy"], "combination_policy")
    for context in ("layernorm", "whitening"):
        entries = _validate_entry_list(
            payload[context],
            context,
            {"config_id", "enabled", "role", "fit_scope"},
        )
        for index, entry in enumerate(entries):
            _boolean(entry["enabled"], f"{context}[{index}].enabled")
            _string(entry["role"], f"{context}[{index}].role")
            _string(entry["fit_scope"], f"{context}[{index}].fit_scope")
    preprojectors = _validate_entry_list(
        payload["preprojectors"],
        "preprojectors",
        {"config_id", "mode", "role"},
    )
    for index, entry in enumerate(preprojectors):
        _string(entry["mode"], f"preprojectors[{index}].mode")
        _string(entry["role"], f"preprojectors[{index}].role")
    checkpoints = _validate_entry_list(
        payload["checkpoint_averaging"],
        "checkpoint_averaging",
        {"config_id", "method", "window", "epochs", "role"},
    )
    for index, entry in enumerate(checkpoints):
        _string(entry["method"], f"checkpoint_averaging[{index}].method")
        _integer(entry["window"], f"checkpoint_averaging[{index}].window")
        _integer_tuple(entry["epochs"], f"checkpoint_averaging[{index}].epochs")
        _string(entry["role"], f"checkpoint_averaging[{index}].role")

    adapter = _object(payload["feature_adapter"], "feature_adapter")
    _keys(
        adapter,
        {
            "label",
            "formula",
            "zero_initialized",
            "gamma_initial",
            "ranks",
            "learning_rate_ratios",
            "candidates",
            "controls",
            "tie_break",
        },
        "feature_adapter",
    )
    _string(adapter["label"], "feature_adapter.label")
    _string(adapter["formula"], "feature_adapter.formula")
    _string(adapter["zero_initialized"], "feature_adapter.zero_initialized")
    _number(adapter["gamma_initial"], "feature_adapter.gamma_initial")
    _integer_tuple(adapter["ranks"], "feature_adapter.ranks")
    for index, ratio in enumerate(
        _sequence(adapter["learning_rate_ratios"], "feature_adapter.learning_rate_ratios")
    ):
        _number(ratio, f"feature_adapter.learning_rate_ratios[{index}]")
    candidates = _validate_entry_list(
        adapter["candidates"],
        "feature_adapter.candidates",
        {"config_id", "rank", "learning_rate_ratio"},
    )
    for index, entry in enumerate(candidates):
        _integer(entry["rank"], f"feature_adapter.candidates[{index}].rank")
        _number(
            entry["learning_rate_ratio"],
            f"feature_adapter.candidates[{index}].learning_rate_ratio",
        )
    controls = _validate_entry_list(
        adapter["controls"],
        "feature_adapter.controls",
        {"config_id", "kind", "parameter_match_tolerance"},
    )
    for index, entry in enumerate(controls):
        _string(entry["kind"], f"feature_adapter.controls[{index}].kind")
        _number(
            entry["parameter_match_tolerance"],
            f"feature_adapter.controls[{index}].parameter_match_tolerance",
        )
    _string_tuple(adapter["tie_break"], "feature_adapter.tie_break")


def _validate_exact_shape(actual: object, expected: object, context: str) -> None:
    """Reject missing/unknown nested keys before exact semantic comparison."""
    if isinstance(expected, dict):
        actual_object = _object(actual, context)
        expected_keys = set(expected)
        actual_keys = set(actual_object)
        unknown = actual_keys - expected_keys
        missing = expected_keys - actual_keys
        if unknown:
            raise ValueError(f"{context} has unknown keys: {sorted(unknown)}")
        if missing:
            raise ValueError(f"{context} is missing keys: {sorted(missing)}")
        for key, expected_value in expected.items():
            _validate_exact_shape(
                actual_object[key], expected_value, f"{context}.{key}"
            )
        return
    if isinstance(expected, list):
        actual_sequence = _sequence(actual, context)
        for index, (actual_item, expected_item) in enumerate(
            zip(actual_sequence, expected)
        ):
            _validate_exact_shape(
                actual_item, expected_item, f"{context}[{index}]"
            )


def _validate_stage2_unique_registry_ids(payload: dict[str, object]) -> None:
    adapter = _object(payload["feature_adapter"], "feature_adapter")
    registries = (
        ("layernorm", payload["layernorm"]),
        ("whitening", payload["whitening"]),
        ("preprojectors", payload["preprojectors"]),
        ("checkpoint_averaging", payload["checkpoint_averaging"]),
        ("feature_adapter.candidates", adapter["candidates"]),
        ("feature_adapter.controls", adapter["controls"]),
    )
    seen: set[str] = set()
    for context, raw_entries in registries:
        for index, raw_entry in enumerate(_sequence(raw_entries, context)):
            entry = _object(raw_entry, f"{context}[{index}]")
            config_id = _identifier(
                entry.get("config_id"), f"{context}[{index}].config_id"
            )
            if config_id in seen:
                raise ValueError(
                    "stage2 config must match the exact preregistered Stage 2 v1 "
                    f"contract: duplicate Stage 2 config_id: {config_id}"
                )
            seen.add(config_id)


def _validate_stage1(payload: dict[str, object]) -> None:
    _validate_stage1_structure(payload)
    expected = stage1_v1_payload()
    if _canonical_json(payload) != _canonical_json(expected):
        raise ValueError(
            "stage1 config must match the exact preregistered Stage 1 v1 contract"
        )


def _validate_stage2(payload: dict[str, object]) -> None:
    expected = stage2_v1_payload()
    _validate_exact_shape(payload, expected, "stage2 config")
    _validate_stage2_unique_registry_ids(payload)
    if _canonical_json(payload) != _canonical_json(expected):
        raise ValueError(
            "stage2 config must match the exact preregistered Stage 2 v1 contract"
        )


@dataclass(frozen=True)
class SemanticConfig:
    """A deeply immutable semantic JSON document."""

    _canonical: str

    @classmethod
    def from_path(cls, path: Path) -> "SemanticConfig":
        payload = _load_json_object(Path(path))
        schema_version = _integer(
            payload.get("schema_version"), "schema_version"
        )
        if schema_version != 1:
            raise ValueError(f"unsupported semantic schema_version: {schema_version}")
        config_type = _string(payload.get("config_type"), "config_type")
        _identifier(payload.get("config_id"), "config_id")
        validators = {
            "internvit_baseline": _validate_internvit,
            "brainrw_clip_lora": _validate_brainrw,
            "stage1_fusion": _validate_stage1,
            "stage2_candidates": _validate_stage2,
        }
        try:
            validator = validators[config_type]
        except KeyError as exc:
            raise ValueError(f"unsupported config_type: {config_type}") from exc
        validator(payload)
        return cls(_canonical=_canonical_json(payload))

    def canonical_payload(self) -> dict[str, object]:
        payload = json.loads(self._canonical)
        if not isinstance(payload, dict):  # pragma: no cover - construction invariant
            raise AssertionError("semantic config is not an object")
        return payload

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self._canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class OutputLocations:
    artifact_dir: str
    log_path: str
    result_path: str


@dataclass(frozen=True)
class ResolvedRunConfig:
    stage: str
    config_id: str
    subject: int
    seed: int
    semantic_config_sha256: str
    input_bundle_sha256: str
    run_key: str
    output_locations: OutputLocations | None
    _canonical: str
    input_hashes: tuple[tuple[str, str], ...]

    def canonical_payload(self) -> dict[str, object]:
        payload = json.loads(self._canonical)
        if not isinstance(payload, dict):  # pragma: no cover - construction invariant
            raise AssertionError("resolved config is not an object")
        return payload


def make_run_key(
    stage: str,
    config_id: str,
    subject: int,
    seed: int,
    semantic_config_sha256: str,
    input_bundle_sha256: str,
) -> str:
    stage = _identifier(stage, "stage")
    config_id = _identifier(config_id, "config_id")
    subject = _integer(subject, "subject")
    seed = _integer(seed, "seed")
    if subject <= 0:
        raise ValueError("subject must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    semantic_config_sha256 = _sha256(
        semantic_config_sha256, "semantic_config_sha256"
    )
    input_bundle_sha256 = _sha256(
        input_bundle_sha256, "input_bundle_sha256"
    )
    return (
        f"{stage}__{config_id}__sub-{subject:02d}__seed-{seed}__"
        f"config-{semantic_config_sha256}__inputs-{input_bundle_sha256}"
    )


def resolve_run_config(
    protocol: ProtocolConfig,
    candidate: Mapping[str, object],
    input_hashes: Mapping[str, str],
) -> ResolvedRunConfig:
    if not isinstance(protocol, ProtocolConfig):
        raise TypeError("protocol must be a ProtocolConfig")
    candidate_object = _object(candidate, "candidate")
    _keys(
        candidate_object,
        {
            "schema_version",
            "stage",
            "config_id",
            "subject",
            "seed",
            "semantics",
            "runtime",
        },
        "candidate",
        optional={"outputs"},
    )
    schema_version = _integer(
        candidate_object["schema_version"], "candidate.schema_version"
    )
    if schema_version != 1:
        raise ValueError(
            f"unsupported candidate schema_version: {schema_version}"
        )
    stage = _identifier(candidate_object["stage"], "candidate.stage")
    config_id = _identifier(
        candidate_object["config_id"], "candidate.config_id"
    )
    subject = _integer(candidate_object["subject"], "candidate.subject")
    seed = _integer(candidate_object["seed"], "candidate.seed")
    if subject <= 0:
        raise ValueError("candidate.subject must be positive")
    if seed < 0:
        raise ValueError("candidate.seed must be non-negative")
    semantics = _jsonable(
        _object(candidate_object["semantics"], "candidate.semantics"),
        "candidate.semantics",
    )
    runtime = _jsonable(
        _object(candidate_object["runtime"], "candidate.runtime"),
        "candidate.runtime",
    )

    outputs: OutputLocations | None = None
    if "outputs" in candidate_object:
        output_object = _object(candidate_object["outputs"], "candidate.outputs")
        _keys(
            output_object,
            {"artifact_dir", "log_path", "result_path"},
            "candidate.outputs",
        )
        outputs = OutputLocations(
            artifact_dir=_string(
                output_object["artifact_dir"], "candidate.outputs.artifact_dir"
            ),
            log_path=_string(
                output_object["log_path"], "candidate.outputs.log_path"
            ),
            result_path=_string(
                output_object["result_path"], "candidate.outputs.result_path"
            ),
        )

    inputs = _object(input_hashes, "input_hashes")
    if not inputs:
        raise ValueError("input_hashes must not be empty")
    required_input_hashes = {
        "model_sha256",
        "cache_sha256",
        "checkpoint_sha256",
        "manifest_sha256",
    }
    missing_input_hashes = required_input_hashes - set(inputs)
    if missing_input_hashes:
        raise ValueError(
            "input_hashes missing required provenance SHA-256 keys: "
            + ", ".join(sorted(missing_input_hashes))
        )
    normalized_inputs: dict[str, str] = {}
    for key, digest in inputs.items():
        safe_key = _identifier(key, "input_hashes key")
        normalized_inputs[safe_key] = _sha256(
            digest, f"input_hashes.{safe_key}"
        )
    normalized_inputs = dict(sorted(normalized_inputs.items()))
    input_bundle_sha256 = _hash_payload(normalized_inputs)

    candidate_semantics = {
        "schema_version": schema_version,
        "stage": stage,
        "config_id": config_id,
        "subject": subject,
        "seed": seed,
        "semantics": semantics,
        "runtime": runtime,
    }
    semantic_payload = {
        "schema_version": 1,
        "protocol": protocol.canonical_payload(),
        "candidate": candidate_semantics,
        "input_hashes": normalized_inputs,
    }
    canonical = _canonical_json(semantic_payload)
    semantic_config_sha256 = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    run_key = make_run_key(
        stage,
        config_id,
        subject,
        seed,
        semantic_config_sha256,
        input_bundle_sha256,
    )
    return ResolvedRunConfig(
        stage=stage,
        config_id=config_id,
        subject=subject,
        seed=seed,
        semantic_config_sha256=semantic_config_sha256,
        input_bundle_sha256=input_bundle_sha256,
        run_key=run_key,
        output_locations=outputs,
        _canonical=canonical,
        input_hashes=tuple(normalized_inputs.items()),
    )
