#!/usr/bin/env python3
"""Build the canonical identity/model inputs for the Stage 1 cost job."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from samga_brain_rw.config import SemanticConfig
from samga_brain_rw.cost_capability import (
    build_stage1_cost_model_manifest,
    build_stage1_cost_score_input_manifest,
    stable_regular_file_sha256,
)
from samga_brain_rw.hashing import canonical_json_bytes, sha256_json


_COORDINATES = tuple(
    (subject, seed)
    for subject in (1, 5, 8)
    for seed in (42, 43)
)
_BRANCH_IDS = ("internvit", "brainrw")
_INPUT_ROOT_RELATIVE = Path(
    "artifacts/samga_brain_rw/stage-1-cost-inputs"
)
_CONFIG_ROOT_RELATIVE = Path("experiments/samga_brain_rw/configs")
_PACKAGE_ROOT_RELATIVE = Path(
    "experiments/samga_brain_rw/samga_brain_rw"
)
_INTERNVIT_FILE_ROLES = frozenset(
    {
        "internvit_config",
        "internvit_configuration_code",
        "internvit_flash_attention_code",
        "internvit_feature_contract_code",
        "internvit_feature_extractor_code",
        "internvit_modeling_code",
        "internvit_preprocessor_config",
        "internvit_weight_index",
        "internvit_weight_shard_1",
        "internvit_weight_shard_2",
        "internvit_weight_shard_3",
        "samga_adapters_code",
        "samga_checkpoint",
        "samga_checkpoint_identity_code",
        "samga_checkpoint_io_code",
        "samga_checkpoint_loader_code",
        "samga_checkpoint_sidecar",
        "samga_checkpoints_code",
        "samga_feature_transforms_code",
        "samga_model_code",
        "samga_trainer_code",
        "samga_upstream_loader_code",
        "semantic_config",
        "upstream_eeg_encoder_code",
        "upstream_loss_code",
        "upstream_projector_code",
    }
)
_BRAINRW_FILE_ROLES = frozenset(
    {
        "brainrw_access_code",
        "brainrw_artifacts_code",
        "brainrw_checkpoint",
        "brainrw_checkpoint_sidecar",
        "brainrw_config_code",
        "brainrw_data_code",
        "brainrw_factory_code",
        "brainrw_hashing_code",
        "brainrw_runtime_contract_code",
        "clip_config",
        "clip_preprocessor_config",
        "clip_weights",
    }
)
_INTERNVIT_FOUNDATION_FILES = {
    "internvit_config": "config.json",
    "internvit_configuration_code": "configuration_intern_vit.py",
    "internvit_flash_attention_code": "flash_attention.py",
    "internvit_modeling_code": "modeling_intern_vit.py",
    "internvit_preprocessor_config": "preprocessor_config.json",
    "internvit_weight_index": "model.safetensors.index.json",
    "internvit_weight_shard_1": "model-00001-of-00003.safetensors",
    "internvit_weight_shard_2": "model-00002-of-00003.safetensors",
    "internvit_weight_shard_3": "model-00003-of-00003.safetensors",
}
_INTERNVIT_PROJECT_FILES = {
    "internvit_feature_contract_code": (
        "experiments/samga_reproduction/v2_5_feature_contract.py"
    ),
    "internvit_feature_extractor_code": (
        "experiments/samga_reproduction/extract_v2_5_features.py"
    ),
    "samga_adapters_code": (
        "experiments/samga_brain_rw/samga_brain_rw/adapters.py"
    ),
    "samga_checkpoint_identity_code": (
        "experiments/samga_brain_rw/samga_brain_rw/checkpoint_identity.py"
    ),
    "samga_checkpoint_io_code": (
        "experiments/samga_brain_rw/samga_brain_rw/checkpoint_io.py"
    ),
    "samga_checkpoint_loader_code": "experiments/samga_brain_rw/train.py",
    "samga_checkpoints_code": (
        "experiments/samga_brain_rw/samga_brain_rw/checkpoints.py"
    ),
    "samga_feature_transforms_code": (
        "experiments/samga_brain_rw/samga_brain_rw/feature_transforms.py"
    ),
    "samga_model_code": (
        "experiments/samga_brain_rw/samga_brain_rw/model.py"
    ),
    "samga_trainer_code": (
        "experiments/samga_brain_rw/samga_brain_rw/trainer.py"
    ),
    "samga_upstream_loader_code": (
        "experiments/samga_brain_rw/samga_brain_rw/upstream_samga.py"
    ),
}
_BRAINRW_PROJECT_FILES = {
    "brainrw_access_code": (
        "experiments/samga_brain_rw/samga_brain_rw/access.py"
    ),
    "brainrw_artifacts_code": (
        "experiments/samga_brain_rw/samga_brain_rw/artifacts.py"
    ),
    "brainrw_config_code": (
        "experiments/samga_brain_rw/samga_brain_rw/config.py"
    ),
    "brainrw_data_code": (
        "experiments/samga_brain_rw/samga_brain_rw/data.py"
    ),
    "brainrw_factory_code": (
        "experiments/samga_brain_rw/samga_brain_rw/brainrw.py"
    ),
    "brainrw_hashing_code": (
        "experiments/samga_brain_rw/samga_brain_rw/hashing.py"
    ),
    "brainrw_runtime_contract_code": (
        "experiments/samga_brain_rw/samga_brain_rw/runtime_contract.py"
    ),
}
_INTERNVIT_CODE_ROLES = frozenset(
    role
    for role in _INTERNVIT_FILE_ROLES
    if role.endswith("_code")
)
_INTERNVIT_CONFIG_ROLES = frozenset(
    {
        "internvit_config",
        "internvit_preprocessor_config",
        "internvit_weight_index",
        "samga_checkpoint_sidecar",
        "semantic_config",
    }
)
_INTERNVIT_WEIGHT_ROLES = (
    _INTERNVIT_FILE_ROLES
    - _INTERNVIT_CODE_ROLES
    - _INTERNVIT_CONFIG_ROLES
)
_BRAINRW_CODE_ROLES = frozenset(
    role for role in _BRAINRW_FILE_ROLES if role.endswith("_code")
)
_BRAINRW_CONFIG_ROLES = frozenset(
    {
        "brainrw_checkpoint_sidecar",
        "clip_config",
        "clip_preprocessor_config",
    }
)
_BRAINRW_WEIGHT_ROLES = (
    _BRAINRW_FILE_ROLES - _BRAINRW_CODE_ROLES - _BRAINRW_CONFIG_ROLES
)


def _object_mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise ValueError(f"{context} must be a string-keyed object")
    return dict(value)


def _build_score_input_manifest(
    cells: Sequence[object],
) -> dict[str, object]:
    """Derive identity-only cost inputs from six completion-bound cells."""

    values = tuple(cells)
    if len(values) != len(_COORDINATES):
        raise ValueError("cost inputs require exactly six Stage 1 cells")
    score_inputs: list[dict[str, object]] = []
    raw_cells: list[dict[str, object]] = []
    for index, (cell, coordinate) in enumerate(
        zip(values, _COORDINATES, strict=True)
    ):
        subject = getattr(cell, "subject", None)
        seed = getattr(cell, "seed", None)
        if (subject, seed) != coordinate:
            raise ValueError(
                f"cost cell[{index}] is not in canonical pilot order"
            )
        cell_id = f"{subject:02d}/{seed}"
        alignment_sha256 = getattr(cell, "alignment_sha256", None)
        raw_branches: dict[str, object] = {}
        score_branches: dict[str, object] = {}
        shared_query_sha256: str | None = None
        shared_gallery_sha256: str | None = None
        query_count: int | None = None
        gallery_count: int | None = None
        for branch_id in _BRANCH_IDS:
            binding = getattr(cell, branch_id, None)
            proof = getattr(binding, "run_proof", None)
            identity_loader = getattr(proof, "identity_payload", None)
            if not callable(identity_loader):
                raise TypeError(
                    f"{branch_id} cell binding lacks a validated run proof"
                )
            identity = _object_mapping(
                identity_loader(),
                f"{branch_id} component proof identity",
            )
            score = getattr(binding, "score", None)
            query_ids = getattr(score, "query_ids", None)
            gallery_ids = getattr(score, "gallery_ids", None)
            if (
                not isinstance(query_ids, Sequence)
                or isinstance(query_ids, (str, bytes, bytearray))
                or not isinstance(gallery_ids, Sequence)
                or isinstance(gallery_ids, (str, bytes, bytearray))
            ):
                raise TypeError(
                    f"{branch_id} component score lacks ordered IDs"
                )
            current_query_sha256 = getattr(
                binding,
                "query_ids_sha256",
                None,
            )
            current_gallery_sha256 = getattr(
                binding,
                "gallery_ids_sha256",
                None,
            )
            if shared_query_sha256 is None:
                shared_query_sha256 = current_query_sha256
                shared_gallery_sha256 = current_gallery_sha256
                query_count = len(query_ids)
                gallery_count = len(gallery_ids)
            elif (
                shared_query_sha256 != current_query_sha256
                or shared_gallery_sha256 != current_gallery_sha256
                or query_count != len(query_ids)
                or gallery_count != len(gallery_ids)
            ):
                raise ValueError("cost component score identities do not align")
            if (
                getattr(binding, "alignment_sha256", None)
                != alignment_sha256
            ):
                raise ValueError("cost component alignment identity mismatch")
            raw_branches[branch_id] = {
                field_name: identity[field_name]
                for field_name in (
                    "checkpoint_sha256",
                    "input_bundle_sha256",
                    "resolved_config_sha256",
                    "run_key",
                    "run_manifest_sha256",
                    "score_envelope_sha256",
                    "score_payload_sha256",
                    "source_payload_sha256",
                )
            }
            score_branches[branch_id] = {
                "binding_sha256": getattr(
                    binding,
                    "binding_sha256",
                    None,
                ),
                "checkpoint_sha256": getattr(
                    binding,
                    "checkpoint_sha256",
                    None,
                ),
                "resolved_config_sha256": getattr(
                    binding,
                    "resolved_config_sha256",
                    None,
                ),
                "run_proof_sha256": getattr(
                    proof,
                    "proof_sha256",
                    None,
                ),
                "score_envelope_sha256": getattr(
                    binding,
                    "score_envelope_sha256",
                    None,
                ),
                "score_payload_sha256": getattr(
                    binding,
                    "score_payload_sha256",
                    None,
                ),
            }
        raw_cells.append(
            {
                "alignment_sha256": alignment_sha256,
                "branches": raw_branches,
                "cell_id": cell_id,
                "gallery_ids_sha256": shared_gallery_sha256,
                "query_ids_sha256": shared_query_sha256,
                "seed": seed,
                "subject": subject,
            }
        )
        score_inputs.append(
            {
                "alignment_sha256": alignment_sha256,
                "brainrw": score_branches["brainrw"],
                "cell_id": cell_id,
                "gallery_count": gallery_count,
                "gallery_ids_sha256": shared_gallery_sha256,
                "internvit": score_branches["internvit"],
                "query_count": query_count,
                "query_ids_sha256": shared_query_sha256,
                "seed": seed,
                "subject": subject,
            }
        )
    raw_input_reference = {
        "cells": raw_cells,
        "provenance_scope": "val-dev-identities-only",
        "schema_version": 1,
    }
    return build_stage1_cost_score_input_manifest(
        score_inputs=score_inputs,
        raw_input_reference=raw_input_reference,
    )


def _canonical_project_root(path: Path) -> Path:
    raw = os.fspath(path)
    absolute = Path(os.path.abspath(os.path.normpath(raw)))
    if not Path(raw).is_absolute() or Path(raw) != absolute:
        raise ValueError("project_root must be absolute and normalized")
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as exc:
        raise ValueError("project_root is unavailable") from exc
    if resolved != absolute or not absolute.is_dir():
        raise ValueError("project_root must be a canonical directory")
    git_entry = absolute / ".git"
    try:
        git_stat = git_entry.lstat()
    except OSError as exc:
        raise ValueError("project_root is not a repository root") from exc
    if not stat.S_ISDIR(git_stat.st_mode):
        raise ValueError("project_root .git must be a real directory")
    return absolute


def _file_entries(paths: Mapping[str, Path]) -> list[dict[str, object]]:
    return [
        {
            "path": str(paths[role]),
            "role": role,
            "sha256": stable_regular_file_sha256(paths[role]),
        }
        for role in sorted(paths)
    ]


def _aggregate_roles(
    files: Sequence[Mapping[str, object]],
    roles: frozenset[str],
) -> str:
    selected = [
        {
            "role": value["role"],
            "sha256": value["sha256"],
        }
        for value in files
        if value["role"] in roles
    ]
    if {str(value["role"]) for value in selected} != set(roles):
        raise ValueError("model aggregate is missing a required file role")
    return sha256_json(selected)


def _representative_checkpoint(
    cells: Sequence[object],
    *,
    branch_id: str,
    checkpoint_name: str,
) -> tuple[object, Path]:
    representative = cells[0]
    if (
        getattr(representative, "subject", None) != 1
        or getattr(representative, "seed", None) != 42
    ):
        raise ValueError("representative cell must be sub-01/seed-42")
    binding = getattr(representative, branch_id, None)
    score = getattr(binding, "score", None)
    score_directory = Path(str(getattr(score, "directory", "")))
    checkpoint = score_directory.parent / checkpoint_name
    checkpoint_sha256 = stable_regular_file_sha256(checkpoint)
    if checkpoint_sha256 != getattr(binding, "checkpoint_sha256", None):
        raise ValueError(
            f"{branch_id} representative checkpoint differs from its "
            "completion-bound score identity"
        )
    return binding, checkpoint


def _build_model_manifest(
    project_root: Path,
    cells: Sequence[object],
) -> dict[str, object]:
    intern_config_path = (
        project_root
        / _CONFIG_ROOT_RELATIVE
        / "internvit_baseline_v1.json"
    )
    brain_config_path = (
        project_root
        / _CONFIG_ROOT_RELATIVE
        / "brainrw_clip_lora_v1.json"
    )
    intern_config = SemanticConfig.from_path(intern_config_path)
    brain_config = SemanticConfig.from_path(brain_config_path)
    intern_payload = intern_config.canonical_payload()
    brain_payload = brain_config.canonical_payload()
    intern_model = _object_mapping(
        intern_payload["model"],
        "InternViT model config",
    )
    upstream_config = _object_mapping(
        intern_payload["upstream"],
        "InternViT upstream config",
    )
    clip_config = _object_mapping(
        brain_payload["clip"],
        "BrainRW CLIP config",
    )
    foundation = Path(str(intern_model["path"]))
    upstream = Path(str(upstream_config["path"]))
    clip = Path(str(clip_config["path"]))
    intern_binding, intern_checkpoint = _representative_checkpoint(
        cells,
        branch_id="internvit",
        checkpoint_name="checkpoint_epoch060.pt",
    )
    brain_binding, brain_checkpoint = _representative_checkpoint(
        cells,
        branch_id="brainrw",
        checkpoint_name="checkpoint.pt",
    )

    intern_paths = {
        role: foundation / filename
        for role, filename in _INTERNVIT_FOUNDATION_FILES.items()
    }
    intern_paths.update(
        {
            role: project_root / relative
            for role, relative in _INTERNVIT_PROJECT_FILES.items()
        }
    )
    intern_paths.update(
        {
            "samga_checkpoint": intern_checkpoint,
            "samga_checkpoint_sidecar": intern_checkpoint.with_suffix(
                intern_checkpoint.suffix + ".meta.json"
            ),
            "semantic_config": intern_config_path,
            "upstream_eeg_encoder_code": (
                upstream / "module/eeg_encoder/model.py"
            ),
            "upstream_loss_code": upstream / "module/loss.py",
            "upstream_projector_code": upstream / "module/projector.py",
        }
    )
    brain_paths = {
        role: project_root / relative
        for role, relative in _BRAINRW_PROJECT_FILES.items()
    }
    brain_paths.update(
        {
            "brainrw_checkpoint": brain_checkpoint,
            "brainrw_checkpoint_sidecar": brain_checkpoint.with_suffix(
                brain_checkpoint.suffix + ".meta.json"
            ),
            "clip_config": clip / "config.json",
            "clip_preprocessor_config": clip / "preprocessor_config.json",
            "clip_weights": clip / "model.safetensors",
        }
    )
    if set(intern_paths) != set(_INTERNVIT_FILE_ROLES):
        raise AssertionError("InternViT cost model role table is incomplete")
    if set(brain_paths) != set(_BRAINRW_FILE_ROLES):
        raise AssertionError("BrainRW cost model role table is incomplete")
    intern_files = _file_entries(intern_paths)
    brain_files = _file_entries(brain_paths)
    branches = {
        "internvit": {
            "factory": "internvit_v2_5_plus_samga",
            "files": intern_files,
            "parameters": {
                "checkpoint_path": str(intern_checkpoint),
                "foundation_model_path": str(foundation),
                "representative_seed": 42,
                "representative_subject": 1,
                "semantic_config_path": str(intern_config_path),
            },
        },
        "brainrw": {
            "factory": "brainrw_clip_lora",
            "files": brain_files,
            "parameters": {
                "checkpoint_path": str(brain_checkpoint),
                "representative_seed": 42,
                "representative_subject": 1,
            },
        },
    }
    raw_model_reference = {
        "branches": {
            "internvit": {
                "checkpoint_sha256": getattr(
                    intern_binding,
                    "checkpoint_sha256",
                ),
                "model_code_sha256": _aggregate_roles(
                    intern_files,
                    _INTERNVIT_CODE_ROLES,
                ),
                "model_config_sha256": _aggregate_roles(
                    intern_files,
                    _INTERNVIT_CONFIG_ROLES,
                ),
                "model_id": "internvit_v2_5_samga_stage0_sub01_seed42",
                "parameter_dtypes": {
                    "foundation": "bfloat16",
                    "task": "float32",
                },
                "weights_sha256": _aggregate_roles(
                    intern_files,
                    _INTERNVIT_WEIGHT_ROLES,
                ),
            },
            "brainrw": {
                "checkpoint_sha256": getattr(
                    brain_binding,
                    "checkpoint_sha256",
                ),
                "model_code_sha256": _aggregate_roles(
                    brain_files,
                    _BRAINRW_CODE_ROLES,
                ),
                "model_config_sha256": _aggregate_roles(
                    brain_files,
                    _BRAINRW_CONFIG_ROLES,
                ),
                "model_id": "brainrw_clip_lora_sub01_seed42",
                "parameter_dtypes": {"model": "bfloat16"},
                "weights_sha256": _aggregate_roles(
                    brain_files,
                    _BRAINRW_WEIGHT_ROLES,
                ),
            },
        },
        "schema_version": 1,
    }
    return build_stage1_cost_model_manifest(
        branches=branches,
        raw_model_reference=raw_model_reference,
    )


def _directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _open_directory_path_nofollow(path: Path) -> int:
    checked = Path(path)
    if not checked.is_absolute():
        raise ValueError(f"directory path must be absolute: {checked}")
    current_fd = os.open("/", _directory_open_flags())
    try:
        for component in checked.parts[1:]:
            next_fd = os.open(
                component,
                _directory_open_flags(),
                dir_fd=current_fd,
            )
            os.close(current_fd)
            current_fd = next_fd
    except BaseException:
        os.close(current_fd)
        raise
    return current_fd


def _open_or_create_directory_at(
    parent_fd: int,
    name: str,
    path: Path,
) -> int:
    created = False
    try:
        os.mkdir(name, 0o755, dir_fd=parent_fd)
        created = True
    except FileExistsError:
        pass
    try:
        child_fd = os.open(
            name,
            _directory_open_flags(),
            dir_fd=parent_fd,
        )
    except OSError as exc:
        raise ValueError(
            f"cost output directory is symbolic or unavailable: {path}"
        ) from exc
    if created:
        os.fsync(parent_fd)
    return child_fd


def _same_identity(first: os.stat_result, second: os.stat_result) -> bool:
    return (first.st_dev, first.st_ino) == (second.st_dev, second.st_ino)


def _require_directory_path_identity(path: Path, directory_fd: int) -> None:
    try:
        observed_fd = _open_directory_path_nofollow(path)
    except OSError as exc:
        raise ValueError(
            f"cost output directory path became symbolic or changed: {path}"
        ) from exc
    try:
        if not _same_identity(os.fstat(directory_fd), os.fstat(observed_fd)):
            raise ValueError(f"cost output directory identity changed: {path}")
    finally:
        os.close(observed_fd)


def _open_fixed_output_roots(
    project_root: Path,
) -> tuple[Path, int, Path, int]:
    root_fd = _open_directory_path_nofollow(project_root)
    opened = [root_fd]
    try:
        if not _same_identity(
            os.fstat(root_fd),
            project_root.stat(follow_symlinks=False),
        ):
            raise ValueError("project_root identity changed while opening")
        artifacts_path = project_root / "artifacts"
        artifacts_fd = _open_or_create_directory_at(
            root_fd,
            "artifacts",
            artifacts_path,
        )
        opened.append(artifacts_fd)
        experiment_path = artifacts_path / "samga_brain_rw"
        experiment_fd = _open_or_create_directory_at(
            artifacts_fd,
            "samga_brain_rw",
            experiment_path,
        )
        opened.append(experiment_fd)
        output_root = experiment_path / "stage-1-cost-inputs"
        output_fd = _open_or_create_directory_at(
            experiment_fd,
            "stage-1-cost-inputs",
            output_root,
        )
        opened.append(output_fd)
        benchmark_root = experiment_path / "stage-1-cost-benchmark"
        benchmark_fd = _open_or_create_directory_at(
            experiment_fd,
            "stage-1-cost-benchmark",
            benchmark_root,
        )
        opened.append(benchmark_fd)
        opened.remove(output_fd)
        opened.remove(benchmark_fd)
        return output_root, output_fd, benchmark_root, benchmark_fd
    finally:
        for descriptor in reversed(opened):
            os.close(descriptor)


def _strict_document_bytes(data: bytes, path: Path) -> dict[str, object]:
    def strict_object(
        pairs: Sequence[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"non-finite JSON number in {path}: {value}")

    try:
        document = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=strict_object,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid canonical cost input: {path}") from exc
    if (
        not isinstance(document, dict)
        or canonical_json_bytes(document) != data
    ):
        raise ValueError(f"cost input is not canonical JSON: {path}")
    return document


def _read_regular_file_at(
    directory_fd: int,
    name: str,
    path: Path,
) -> bytes:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise ValueError(f"cannot read canonical cost input: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"cost input is not a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        data = b"".join(chunks)
        after = os.fstat(descriptor)
        if (
            not _same_identity(before, after)
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ctime_ns != after.st_ctime_ns
            or len(data) != after.st_size
        ):
            raise ValueError(f"cost input changed while reading: {path}")
        return data
    finally:
        os.close(descriptor)


def _verify_existing_or_missing_at(
    directory_fd: int,
    name: str,
    path: Path,
    document: Mapping[str, object],
) -> bool:
    try:
        data = _read_regular_file_at(directory_fd, name, path)
    except FileNotFoundError:
        return False
    current = _strict_document_bytes(data, path)
    if current != dict(document):
        raise ValueError(
            f"existing canonical cost input differs from requested bytes: {path}"
        )
    return True


def _unlink_same_identity_at(
    directory_fd: int,
    name: str,
    expected: os.stat_result,
) -> None:
    try:
        observed = os.stat(
            name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    if _same_identity(observed, expected):
        os.unlink(name, dir_fd=directory_fd)
        os.fsync(directory_fd)


def _create_or_verify_identical_at(
    directory_fd: int,
    name: str,
    path: Path,
    document: Mapping[str, object],
) -> None:
    data = canonical_json_bytes(document)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(name, flags, 0o644, dir_fd=directory_fd)
    except FileExistsError:
        if not _verify_existing_or_missing_at(
            directory_fd,
            name,
            path,
            document,
        ):
            raise AssertionError("existing cost input disappeared")
        return
    created_identity = os.fstat(descriptor)
    try:
        offset = 0
        while offset < len(data):
            written = os.write(descriptor, data[offset:])
            if written <= 0:
                raise OSError("short write while publishing cost input")
            offset += written
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        _unlink_same_identity_at(
            directory_fd,
            name,
            created_identity,
        )
        raise
    else:
        os.close(descriptor)
        os.fsync(directory_fd)


def build_stage1_cost_inputs(
    *,
    project_root: Path,
    component_loader: Callable[[Path, SemanticConfig], Sequence[object]]
    | None = None,
) -> dict[str, object]:
    """Build or byte-verify the two fixed Stage 1 cost input manifests."""

    root = _canonical_project_root(project_root)
    fusion_config = SemanticConfig.from_path(
        root / _CONFIG_ROOT_RELATIVE / "stage1_fusion_v1.json"
    )
    if component_loader is None:
        from samga_brain_rw.component_proofs import (
            load_stage1_composition_cells,
        )

        component_loader = load_stage1_composition_cells
    cells = tuple(component_loader(root, fusion_config))
    score_document = _build_score_input_manifest(cells)
    model_document = _build_model_manifest(root, cells)
    output_root, output_fd, benchmark_root, benchmark_fd = (
        _open_fixed_output_roots(root)
    )
    try:
        score_path = output_root / "score-inputs.json"
        model_path = output_root / "model-manifest.json"
        for name, path, document in (
            ("score-inputs.json", score_path, score_document),
            ("model-manifest.json", model_path, model_document),
        ):
            if not _verify_existing_or_missing_at(
                output_fd,
                name,
                path,
                document,
            ):
                _create_or_verify_identical_at(
                    output_fd,
                    name,
                    path,
                    document,
                )
        _require_directory_path_identity(output_root, output_fd)
        _require_directory_path_identity(benchmark_root, benchmark_fd)
        score_file_sha256 = hashlib.sha256(
            canonical_json_bytes(score_document)
        ).hexdigest()
        model_file_sha256 = hashlib.sha256(
            canonical_json_bytes(model_document)
        ).hexdigest()
    finally:
        os.close(benchmark_fd)
        os.close(output_fd)
    return {
        "model_manifest_file_sha256": model_file_sha256,
        "model_manifest_path": str(model_path),
        "score_inputs_file_sha256": score_file_sha256,
        "score_inputs_path": str(score_path),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = build_stage1_cost_inputs(project_root=arguments.project_root)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
