"""Trusted Stage 1 cost capability issued from one sealed A40 job completion.

``inference_cost.RawStage1CostRecord`` is deliberately self-attested.  This
module does not upgrade such a record by itself.  It requires the current
completion capability for the exact cost-benchmark job-map row and rechecks
every completed output plus every protocol, score-identity, model, and runtime
file bound by that row.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from .fusion import enumerate_stage1_configs
from .hashing import canonical_json_bytes, sha256_json
from .inference_cost import (
    CostProtocol,
    RawStage1CostRecord,
    validate_self_attested_input_reference,
    validate_self_attested_model_reference,
    validate_self_attested_runtime_reference,
)
from .stage1 import (
    VALIDATED_COST_CAPABILITY_TYPE,
    ValidatedStage1CostCapability,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SLURM_ARRAY_JOB_RE = re.compile(
    r"^(?P<array_job_id>[1-9][0-9]*)_(?P<array_task_id>0|[1-9][0-9]*)$"
)
_COORDINATES = (
    (1, 42),
    (1, 43),
    (5, 42),
    (5, 43),
    (8, 42),
    (8, 43),
)
_BRANCH_IDS = ("internvit", "brainrw")
_OUTPUT_HASH_NAMES = frozenset(
    {
        "raw_record_file_sha256",
        "run_manifest_file_sha256",
        "runtime_manifest_file_sha256",
    }
)
_COMPLETION_DOCUMENT_KEYS = frozenset(
    {"payload", "payload_sha256", "payload_type", "schema_version"}
)
_COMPLETION_PAYLOAD_KEYS = frozenset(
    {
        "array_index",
        "claim_sha256",
        "generation",
        "job_map_sha256",
        "output_hashes",
        "row_sha256",
    }
)
_AUTHORITY_EXECUTION_IDENTITY_KEYS = frozenset(
    {
        "array_index",
        "attempt_payload_sha256",
        "attempt_record_sha256",
        "claim_sha256",
        "generation",
        "job_map_sha256",
        "path",
        "payload_sha256",
        "row_sha256",
        "scheduler_job_id",
        "sha256",
    }
)
_RAW_RECORD_NAME_RE = re.compile(r"^stage1-cost-attempt-(\d{4})\.json$")
_SCORE_DOCUMENT_KEYS = frozenset(
    {
        "artifact_type",
        "raw_input_reference",
        "raw_input_reference_sha256",
        "schema_version",
        "scope",
        "score_inputs",
        "score_inputs_sha256",
    }
)
_SCORE_INPUT_KEYS = frozenset(
    {
        "alignment_sha256",
        "brainrw",
        "cell_id",
        "gallery_count",
        "gallery_ids_sha256",
        "internvit",
        "query_count",
        "query_ids_sha256",
        "seed",
        "subject",
    }
)
_SCORE_BRANCH_KEYS = frozenset(
    {
        "binding_sha256",
        "checkpoint_sha256",
        "resolved_config_sha256",
        "run_proof_sha256",
        "score_envelope_sha256",
        "score_payload_sha256",
    }
)
_MODEL_DOCUMENT_KEYS = frozenset(
    {
        "artifact_type",
        "branches",
        "raw_model_reference",
        "raw_model_reference_sha256",
        "schema_version",
        "scope",
    }
)
_MODEL_BRANCH_KEYS = frozenset({"factory", "files", "parameters"})
_MODEL_FILE_KEYS = frozenset({"path", "role", "sha256"})
_MODEL_FILE_ROLES = {
    "internvit": frozenset(
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
    ),
    "brainrw": frozenset(
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
    ),
}
_INTERNVIT_MODEL_PARAMETER_KEYS = frozenset(
    {
        "checkpoint_path",
        "foundation_model_path",
        "representative_seed",
        "representative_subject",
        "semantic_config_path",
    }
)
_BRAINRW_MODEL_PARAMETER_KEYS = frozenset(
    {
        "checkpoint_path",
        "representative_seed",
        "representative_subject",
    }
)
_INTERNVIT_CODE_ROLES = frozenset(
    role for role in _MODEL_FILE_ROLES["internvit"] if role.endswith("_code")
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
_BRAINRW_CODE_ROLES = frozenset(
    role for role in _MODEL_FILE_ROLES["brainrw"] if role.endswith("_code")
)
_BRAINRW_CONFIG_ROLES = frozenset(
    {
        "brainrw_checkpoint_sidecar",
        "clip_config",
        "clip_preprocessor_config",
    }
)
_INTERNVIT_MODEL_REPO = "OpenGVLab/InternViT-6B-448px-V2_5"
_INTERNVIT_MODEL_REVISION = "9d1a4344077479c93d42584b6941c64d795d508d"
_INTERNVIT_WEIGHT_ROLE_BY_FILENAME = {
    "model-00001-of-00003.safetensors": "internvit_weight_shard_1",
    "model-00002-of-00003.safetensors": "internvit_weight_shard_2",
    "model-00003-of-00003.safetensors": "internvit_weight_shard_3",
}
_MODEL_AGGREGATE_ROLES = {
    "internvit": {
        "model_code_sha256": _INTERNVIT_CODE_ROLES,
        "model_config_sha256": _INTERNVIT_CONFIG_ROLES,
        "weights_sha256": (
            _MODEL_FILE_ROLES["internvit"]
            - _INTERNVIT_CODE_ROLES
            - _INTERNVIT_CONFIG_ROLES
        ),
    },
    "brainrw": {
        "model_code_sha256": _BRAINRW_CODE_ROLES,
        "model_config_sha256": _BRAINRW_CONFIG_ROLES,
        "weights_sha256": (
            _MODEL_FILE_ROLES["brainrw"]
            - _BRAINRW_CODE_ROLES
            - _BRAINRW_CONFIG_ROLES
        ),
    },
}
_RUNTIME_DOCUMENT_KEYS = frozenset(
    {
        "artifact_type",
        "execution_config_file_sha256",
        "execution_config_sha256",
        "runtime_evidence",
        "runtime_evidence_sha256",
        "runtime_reference",
        "runtime_reference_sha256",
        "schema_version",
        "scope",
    }
)
_RUN_MANIFEST_KEYS = frozenset(
    {
        "artifact_type",
        "authority_execution_file_sha256",
        "authority_execution_path",
        "authority_execution_payload_sha256",
        "execution_config_file_sha256",
        "execution_config_path",
        "execution_config_sha256",
        "input_bundle_sha256",
        "model_manifest_file_sha256",
        "model_manifest_path",
        "protocol_file_sha256",
        "protocol_path",
        "protocol_sha256",
        "raw_record_file_sha256",
        "raw_record_path",
        "raw_record_sha256",
        "runner_file_sha256",
        "runner_path",
        "runtime_evidence_sha256",
        "runtime_manifest_file_sha256",
        "runtime_manifest_path",
        "schema_version",
        "scope",
        "score_inputs_file_sha256",
        "score_inputs_path",
    }
)
_RUNNER_FLAGS = frozenset(
    {
        "--config",
        "--config-id",
        "--device",
        "--execution-config",
        "--expected-config-sha256",
        "--expected-execution-config-sha256",
        "--expected-input-bundle-sha256",
        "--model-manifest",
        "--output-dir",
        "--project-root",
        "--run-key",
        "--score-inputs",
        "--seed",
        "--subject",
    }
)
_CONSTRUCTION_TOKEN = object()
_EXECUTION_PLAN_CONSTRUCTION_TOKEN = object()
_EXPECTED_EXECUTION_PLAN = {
    "schema_version": 1,
    "config_type": "stage1_cost_execution",
    "config_id": "stage1_cost_execution_v1",
    "scope": "stage1-cost",
    "runtime": {
        "accelerator": "NVIDIA A40",
        "accelerator_count": 1,
        "device": "cuda:0",
        "process_mode": "single_process",
        "inference_mode": True,
        "synchronize": "protocol_before_and_after_branch_callable",
    },
    "coverage": {
        "seed": 20260720,
        "query_count": 200,
        "gallery_count": 200,
        "full_query_coverage": True,
        "full_gallery_coverage": True,
        "synthetic_generation_device": "cpu",
        "synthetic_distribution": "standard_normal_float32_then_cast",
        "synthetic_tensor_order": [
            "internvit.eeg",
            "internvit.image",
            "brainrw.eeg",
            "brainrw.image",
        ],
        "similarity_shape": [200, 200],
    },
    "observations": {
        "labels_present": False,
        "metrics_computed": False,
        "predictions_persisted": False,
        "test_data_access": False,
    },
    "branch_order": ["internvit", "brainrw"],
    "branches": {
        "internvit": {
            "factory": "internvit_v2_5_plus_samga",
            "representative_subject": 1,
            "representative_seed": 42,
            "image_shape": [200, 3, 448, 448],
            "image_dtype": "bfloat16",
            "image_microbatch_size": 16,
            "image_chunk_order": "ascending_contiguous_no_shuffle",
            "eeg_shape": [200, 17, 250],
            "eeg_dtype": "float32",
            "eeg_microbatch_size": 200,
            "eeg_chunk_order": "ascending_contiguous_no_shuffle",
            "foundation_parameter_dtype": "bfloat16",
            "task_parameter_dtype": "float32",
            "foundation_autocast": "cuda_bfloat16_enabled",
            "task_autocast": "disabled",
            "feature_layer_ids": [20, 24, 28, 32, 36],
            "feature_pooling": "patch_mean",
            "feature_dtype_at_task_boundary": "float32",
            "embedding_normalization": "l2_float32",
            "similarity_dtype": "float32",
        },
        "brainrw": {
            "factory": "brainrw_clip_lora",
            "representative_subject": 1,
            "representative_seed": 42,
            "image_shape": [200, 3, 224, 224],
            "image_dtype": "bfloat16",
            "image_microbatch_size": 64,
            "image_chunk_order": "ascending_contiguous_no_shuffle",
            "eeg_shape": [200, 17, 250],
            "eeg_dtype": "bfloat16",
            "eeg_microbatch_size": 200,
            "eeg_chunk_order": "ascending_contiguous_no_shuffle",
            "model_parameter_dtype": "bfloat16",
            "autocast": "disabled",
            "embedding_normalization": "model_l2_then_float32",
            "similarity_dtype": "float32",
        },
    },
}


def _load_job_map_module() -> object:
    """Load the existing job-map authority from the scripts PYTHONPATH."""

    return importlib.import_module("build_job_map")


def _first_difference(
    actual: object,
    expected: object,
    path: str = "execution plan",
) -> str | None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return path
        if set(actual) != set(expected):
            return f"{path}.schema"
        for key in expected:
            difference = _first_difference(
                actual[key],
                expected[key],
                f"{path}.{key}",
            )
            if difference is not None:
                return difference
        return None
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return path
        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected, strict=True)
        ):
            difference = _first_difference(
                actual_item,
                expected_item,
                f"{path}[{index}]",
            )
            if difference is not None:
                return difference
        return None
    return None if actual == expected and type(actual) is type(expected) else path


@dataclass(frozen=True, init=False)
class Stage1CostExecutionPlan:
    """Controlled semantic view of the prospective full-model timing plan."""

    _payload: Mapping[str, object]
    _construction_token: object

    def __new__(cls, *_: object, **__: object) -> "Stage1CostExecutionPlan":
        raise TypeError(
            "Stage1CostExecutionPlan requires controlled file construction"
        )

    @classmethod
    def _from_payload(cls, value: object) -> "Stage1CostExecutionPlan":
        payload = _mapping(value, "Stage 1 cost execution plan")
        difference = _first_difference(payload, _EXPECTED_EXECUTION_PLAN)
        if difference is not None:
            raise ValueError(
                "Stage 1 cost execution plan semantic mismatch at "
                f"{difference}; the seal requires one NVIDIA A40, exact "
                "200x200 coverage, locked microbatch sizes, and no labels "
                "or metrics"
            )
        instance = object.__new__(cls)
        object.__setattr__(instance, "_payload", _deep_freeze(payload))
        object.__setattr__(
            instance,
            "_construction_token",
            _EXECUTION_PLAN_CONSTRUCTION_TOKEN,
        )
        return instance

    @property
    def config_id(self) -> str:
        return "stage1_cost_execution_v1"

    @property
    def seed(self) -> int:
        return 20260720

    @property
    def query_count(self) -> int:
        return 200

    @property
    def gallery_count(self) -> int:
        return 200

    @property
    def branch_order(self) -> tuple[str, str]:
        return ("internvit", "brainrw")

    @property
    def labels_present(self) -> bool:
        return False

    @property
    def metrics_computed(self) -> bool:
        return False

    @property
    def synchronize(self) -> str:
        return "protocol_before_and_after_branch_callable"

    def image_microbatch_size(self, branch_id: str) -> int:
        if branch_id not in _BRANCH_IDS:
            raise ValueError("unknown cost execution branch")
        branches = self._payload["branches"]
        assert isinstance(branches, Mapping)
        branch = branches[branch_id]
        assert isinstance(branch, Mapping)
        return int(branch["image_microbatch_size"])

    def eeg_microbatch_size(self, branch_id: str) -> int:
        if branch_id not in _BRANCH_IDS:
            raise ValueError("unknown cost execution branch")
        branches = self._payload["branches"]
        assert isinstance(branches, Mapping)
        branch = branches[branch_id]
        assert isinstance(branch, Mapping)
        return int(branch["eeg_microbatch_size"])

    def branch_payload(self, branch_id: str) -> dict[str, object]:
        if branch_id not in _BRANCH_IDS:
            raise ValueError("unknown cost execution branch")
        payload = self.to_payload()
        branches = payload["branches"]
        assert isinstance(branches, dict)
        branch = branches[branch_id]
        assert isinstance(branch, dict)
        return branch

    def to_payload(self) -> dict[str, object]:
        payload = _deep_thaw(self._payload)
        if not isinstance(payload, dict):
            raise AssertionError("execution plan payload must be an object")
        return payload

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_payload())


def load_stage1_cost_execution_plan(path: Path) -> Stage1CostExecutionPlan:
    """Load a duplicate-key-safe plan and require every preregistered value."""

    document, _ = _load_json_file(
        Path(path),
        "Stage 1 cost execution plan",
        require_canonical=False,
    )
    return Stage1CostExecutionPlan._from_payload(document)


def _sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise ValueError(f"{context} must be a string-keyed object")
    return {str(key): copy.deepcopy(child) for key, child in value.items()}


def _exact_keys(
    value: Mapping[str, object],
    expected: frozenset[str],
    context: str,
) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ValueError(
            f"{context} schema mismatch: missing={missing}, extra={extra}"
        )


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _open_nofollow_path(path: Path, *, directory: bool, context: str) -> int:
    raw = os.fspath(path)
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise ValueError(f"{context} path is invalid")
    declared = Path(raw)
    if not declared.is_absolute() or declared != Path(
        os.path.abspath(os.path.normpath(raw))
    ):
        raise ValueError(f"{context} path must be absolute and normalized")
    parts = declared.parts
    if len(parts) < 2:
        raise ValueError(f"{context} path cannot be the filesystem root")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(declared.anchor, directory_flags)
    try:
        for component in parts[1:-1]:
            child = os.open(component, directory_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        leaf_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
            os, "O_NOFOLLOW", 0
        )
        if directory:
            leaf_flags |= getattr(os, "O_DIRECTORY", 0)
        leaf = os.open(parts[-1], leaf_flags, dir_fd=descriptor)
    except OSError as exc:
        raise ValueError(f"{context} is unavailable or symbolic") from exc
    finally:
        os.close(descriptor)
    return leaf


def _stream_regular_file(
    path: Path,
    context: str,
    *,
    collect_bytes: bool,
) -> tuple[bytes | None, str]:
    try:
        descriptor = _open_nofollow_path(
            path,
            directory=False,
            context=context,
        )
    except OSError as exc:
        raise ValueError(f"{context} file is unavailable") from exc
    digest = hashlib.sha256()
    chunks: list[bytes] | None = [] if collect_bytes else None
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{context} must be a regular non-symlink file")
        total = 0
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
            if chunks is not None:
                chunks.append(chunk)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise ValueError(f"{context} could not be read stably") from exc
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after or total != after.st_size:
        raise ValueError(f"{context} changed while it was read")
    return (
        None if chunks is None else b"".join(chunks),
        digest.hexdigest(),
    )


def _read_regular_file(path: Path, context: str) -> tuple[bytes, str]:
    data, digest = _stream_regular_file(
        path,
        context,
        collect_bytes=True,
    )
    if data is None:
        raise AssertionError("regular-file byte collection was lost")
    return data, digest


def stable_regular_file_sha256(path: Path) -> str:
    """Hash one absolute regular file through stable, no-follow descriptors."""

    _, digest = _stream_regular_file(
        Path(path),
        "sealed input file",
        collect_bytes=False,
    )
    return digest


def _model_file_aggregate_sha256(
    files: Sequence[Mapping[str, object]],
    roles: frozenset[str],
) -> str:
    selected = sorted(
        (
            {
                "role": str(entry["role"]),
                "sha256": str(entry["sha256"]),
            }
            for entry in files
            if entry["role"] in roles
        ),
        key=lambda entry: entry["role"],
    )
    if {entry["role"] for entry in selected} != set(roles):
        raise ValueError("model aggregate is missing a required file role")
    return sha256_json(selected)


def _require_nofollow_directory(path: Path, context: str) -> None:
    descriptor = _open_nofollow_path(
        path,
        directory=True,
        context=context,
    )
    try:
        value = os.fstat(descriptor)
        if not stat.S_ISDIR(value.st_mode):
            raise ValueError(f"{context} must be a non-symlink directory")
    finally:
        os.close(descriptor)


def _load_json_file(
    path: Path,
    context: str,
    *,
    require_canonical: bool,
) -> tuple[dict[str, object], str]:
    data, digest = _read_regular_file(path, context)
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{context} must contain an object")
    canonical = canonical_json_bytes(value)
    if require_canonical and data not in {canonical, canonical + b"\n"}:
        raise ValueError(f"{context} must use canonical JSON bytes")
    return value, digest


def load_canonical_json_document(
    path: Path,
) -> tuple[dict[str, object], str]:
    """Read one canonical JSON document through stable no-follow descriptors."""

    return _load_json_file(
        Path(path),
        "sealed JSON document",
        require_canonical=True,
    )


def _validate_internvit_foundation_pins(
    files: Sequence[Mapping[str, object]],
    parameters: Mapping[str, object],
) -> None:
    """Cross-bind live foundation bytes to the training semantic config."""

    by_role = {
        str(entry["role"]): entry
        for entry in files
    }
    semantic_path = _path(
        parameters["semantic_config_path"],
        "InternViT semantic config",
    )
    semantic, _ = _load_json_file(
        semantic_path,
        "InternViT semantic config",
        require_canonical=False,
    )
    model = _mapping(
        semantic.get("model"),
        "InternViT semantic foundation model",
    )
    foundation_path = _path(
        parameters["foundation_model_path"],
        "InternViT foundation model",
    )
    if (
        model.get("repo") != _INTERNVIT_MODEL_REPO
        or model.get("revision") != _INTERNVIT_MODEL_REVISION
        or model.get("path") != str(foundation_path)
    ):
        raise ValueError(
            "InternViT semantic foundation identity differs from the "
            "pinned repository/revision/path"
        )
    pinned_config_sha256 = _sha256(
        model.get("config_sha256"),
        "InternViT pinned config SHA-256",
    )
    pinned_preprocessor_sha256 = _sha256(
        model.get("preprocessor_sha256"),
        "InternViT pinned preprocessor SHA-256",
    )
    if (
        by_role["internvit_config"]["sha256"] != pinned_config_sha256
        or by_role["internvit_preprocessor_config"]["sha256"]
        != pinned_preprocessor_sha256
    ):
        raise ValueError(
            "InternViT live config/preprocessor bytes differ from semantic "
            "foundation pins"
        )
    weights = _mapping(
        model.get("weight_sha256"),
        "InternViT pinned foundation weights",
    )
    if set(weights) != set(_INTERNVIT_WEIGHT_ROLE_BY_FILENAME):
        raise ValueError(
            "InternViT semantic foundation weight-shard schema mismatch"
        )
    for filename, role in _INTERNVIT_WEIGHT_ROLE_BY_FILENAME.items():
        pinned_sha256 = _sha256(
            weights[filename],
            f"InternViT pinned weight {filename}",
        )
        if by_role[role]["sha256"] != pinned_sha256:
            raise ValueError(
                f"InternViT live weight differs from semantic foundation "
                f"pin: {filename}"
            )


def _path(value: object, context: str) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"{context} path must be text")
    candidate = Path(value)
    if not candidate.is_absolute() or candidate != Path(
        os.path.abspath(os.path.normpath(value))
    ):
        raise ValueError(f"{context} path must be absolute and normalized")
    return candidate


def _runner_values(row: Mapping[str, object]) -> dict[str, str]:
    argv = row.get("argv")
    if (
        not isinstance(argv, list)
        or len(argv) != 2 + 2 * len(_RUNNER_FLAGS)
        or any(not isinstance(item, str) or not item for item in argv)
        or argv[0] != "python"
    ):
        raise ValueError("cost benchmark row argv is not the sealed runner form")
    tokens = argv[2:]
    flags = tokens[::2]
    values = tokens[1::2]
    if (
        set(flags) != _RUNNER_FLAGS
        or len(flags) != len(set(flags))
        or any(value.startswith("--") for value in values)
    ):
        raise ValueError("cost benchmark runner flags differ from the seal")
    parsed = dict(zip(flags, values, strict=True))
    project_root = _path(parsed["--project-root"], "project root")
    expected_runner = (
        project_root
        / "experiments/samga_brain_rw/scripts/run_stage1_cost.py"
    )
    if _path(argv[1], "cost benchmark runner") != expected_runner:
        raise ValueError("cost benchmark row does not use the exact runner")
    expected_protocol = (
        project_root
        / "experiments/samga_brain_rw/configs/stage1_cost_v1.json"
    )
    expected_execution = (
        project_root
        / "experiments/samga_brain_rw/configs/stage1_cost_execution_v1.json"
    )
    if (
        _path(parsed["--config"], "cost protocol") != expected_protocol
        or _path(parsed["--execution-config"], "cost execution plan")
        != expected_execution
    ):
        raise ValueError("cost benchmark row config paths differ from the seal")
    if (
        parsed["--subject"] != "1"
        or parsed["--seed"] != "20260720"
        or parsed["--config-id"] != "stage1_cost_v1"
        or parsed["--device"] != "cuda"
    ):
        raise ValueError("cost benchmark fixed runner identity mismatch")
    return parsed


def _validate_score_document(
    value: object,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    document = _mapping(value, "Stage 1 cost score-input manifest")
    _exact_keys(document, _SCORE_DOCUMENT_KEYS, "cost score-input manifest")
    if (
        document["artifact_type"]
        != "samga_brain_rw.stage1_cost_score_inputs"
        or document["schema_version"] != 1
        or document["scope"] != "val-dev"
    ):
        raise ValueError("cost score-input manifest identity/scope mismatch")
    raw_inputs = validate_self_attested_input_reference(
        document["raw_input_reference"]
    )
    if document["raw_input_reference_sha256"] != sha256_json(raw_inputs):
        raise ValueError("cost raw input-reference SHA-256 mismatch")
    raw_scores = document["score_inputs"]
    if not isinstance(raw_scores, list) or len(raw_scores) != 6:
        raise ValueError("cost score-input manifest requires exactly six cells")
    score_inputs: list[dict[str, object]] = []
    for index, (raw_score, raw_input) in enumerate(
        zip(raw_scores, raw_inputs["cells"], strict=True)
    ):
        score = _mapping(raw_score, f"cost score_inputs[{index}]")
        _exact_keys(score, _SCORE_INPUT_KEYS, f"cost score_inputs[{index}]")
        subject, seed = _COORDINATES[index]
        if (
            score["subject"] != subject
            or score["seed"] != seed
            or score["cell_id"] != f"{subject:02d}/{seed}"
            or score["query_count"] != 200
            or score["gallery_count"] != 200
        ):
            raise ValueError("cost score-input coordinate/count mismatch")
        checked_input = _mapping(raw_input, f"raw input cell[{index}]")
        for field in (
            "alignment_sha256",
            "cell_id",
            "gallery_ids_sha256",
            "query_ids_sha256",
            "seed",
            "subject",
        ):
            if score[field] != checked_input[field]:
                raise ValueError(f"cost score-input {field} cross-binding mismatch")
        for field in (
            "alignment_sha256",
            "gallery_ids_sha256",
            "query_ids_sha256",
        ):
            _sha256(score[field], f"cost score-input {field}")
        raw_branches = _mapping(
            checked_input["branches"],
            f"raw input branches[{index}]",
        )
        for branch_id in _BRANCH_IDS:
            branch = _mapping(
                score[branch_id],
                f"cost score_inputs[{index}].{branch_id}",
            )
            _exact_keys(
                branch,
                _SCORE_BRANCH_KEYS,
                f"cost score_inputs[{index}].{branch_id}",
            )
            for field in _SCORE_BRANCH_KEYS:
                _sha256(
                    branch[field],
                    f"cost score_inputs[{index}].{branch_id}.{field}",
                )
            raw_branch = _mapping(
                raw_branches[branch_id],
                f"raw input branch {branch_id}[{index}]",
            )
            for field in (
                "checkpoint_sha256",
                "resolved_config_sha256",
                "score_envelope_sha256",
                "score_payload_sha256",
            ):
                if branch[field] != raw_branch[field]:
                    raise ValueError(
                        f"cost score-input {branch_id} {field} cross-binding mismatch"
                    )
            score[branch_id] = branch
        score_inputs.append(score)
    if document["score_inputs_sha256"] != sha256_json(score_inputs):
        raise ValueError("cost score-input manifest SHA-256 mismatch")
    return score_inputs, raw_inputs


def build_stage1_cost_score_input_manifest(
    *,
    score_inputs: object,
    raw_input_reference: object,
) -> dict[str, object]:
    """Build the strict identity-only bridge for all twelve component bindings.

    ``score_inputs`` is the exact six-cell identity shape consumed by
    :func:`samga_brain_rw.stage1.compose_stage1`; ``raw_input_reference`` is
    the observation-free shape embedded in the raw benchmark record.  The
    validator cross-binds every shared component and alignment digest.
    """

    raw_inputs = validate_self_attested_input_reference(raw_input_reference)
    if not isinstance(score_inputs, list):
        raise ValueError("cost score_inputs must be a list")
    document = {
        "artifact_type": "samga_brain_rw.stage1_cost_score_inputs",
        "raw_input_reference": raw_inputs,
        "raw_input_reference_sha256": sha256_json(raw_inputs),
        "schema_version": 1,
        "scope": "val-dev",
        "score_inputs": copy.deepcopy(score_inputs),
        "score_inputs_sha256": sha256_json(score_inputs),
    }
    validated_scores, validated_raw = _validate_score_document(document)
    return {
        **document,
        "raw_input_reference": validated_raw,
        "raw_input_reference_sha256": sha256_json(validated_raw),
        "score_inputs": validated_scores,
        "score_inputs_sha256": sha256_json(validated_scores),
    }


def load_stage1_cost_score_input_manifest(
    path: Path,
) -> tuple[dict[str, object], str]:
    """Load, strictly validate, and file-hash one score-identity manifest."""

    document, file_sha256 = _load_json_file(
        Path(path),
        "cost score-input manifest",
        require_canonical=True,
    )
    scores, raw_inputs = _validate_score_document(document)
    return (
        {
            **document,
            "raw_input_reference": raw_inputs,
            "score_inputs": scores,
        },
        file_sha256,
    )


def _validate_model_document(value: object) -> dict[str, object]:
    document = _mapping(value, "Stage 1 cost model manifest")
    _exact_keys(document, _MODEL_DOCUMENT_KEYS, "cost model manifest")
    if (
        document["artifact_type"]
        != "samga_brain_rw.stage1_cost_model_manifest"
        or document["schema_version"] != 1
        or document["scope"] != "stage1-cost"
    ):
        raise ValueError("cost model manifest identity/scope mismatch")
    raw_reference = validate_self_attested_model_reference(
        document["raw_model_reference"],
    )
    if document["raw_model_reference_sha256"] != sha256_json(raw_reference):
        raise ValueError("raw cost model reference SHA-256 mismatch")
    branches = _mapping(document["branches"], "cost model branches")
    if set(branches) != set(_BRANCH_IDS):
        raise ValueError("cost model manifest requires the exact two branches")
    expected_factories = {
        "internvit": "internvit_v2_5_plus_samga",
        "brainrw": "brainrw_clip_lora",
    }
    raw_reference_branches = _mapping(
        raw_reference["branches"],
        "raw cost model branches",
    )
    normalized: dict[str, object] = {}
    for branch_id in _BRANCH_IDS:
        branch = _mapping(branches[branch_id], f"{branch_id} cost model")
        _exact_keys(branch, _MODEL_BRANCH_KEYS, f"{branch_id} cost model")
        if branch["factory"] != expected_factories[branch_id]:
            raise ValueError(f"{branch_id} cost model factory mismatch")
        raw_files = branch["files"]
        if not isinstance(raw_files, list) or not raw_files:
            raise ValueError(f"{branch_id} cost model requires bound files")
        files: list[dict[str, object]] = []
        seen: set[str] = set()
        seen_roles: set[str] = set()
        for index, raw_file in enumerate(raw_files):
            file_entry = _mapping(
                raw_file,
                f"{branch_id} cost model file[{index}]",
            )
            _exact_keys(
                file_entry,
                _MODEL_FILE_KEYS,
                f"{branch_id} cost model file[{index}]",
            )
            file_path = _path(
                file_entry["path"],
                f"{branch_id} cost model file[{index}]",
            )
            role = file_entry["role"]
            if (
                not isinstance(role, str)
                or role not in _MODEL_FILE_ROLES[branch_id]
                or role in seen_roles
            ):
                raise ValueError(
                    f"{branch_id} cost model file role is unknown or duplicated"
                )
            actual_sha256 = stable_regular_file_sha256(file_path)
            claimed = _sha256(
                file_entry["sha256"],
                f"{branch_id} cost model file[{index}] SHA-256",
            )
            if actual_sha256 != claimed:
                raise ValueError(f"{branch_id} model file SHA-256 mismatch")
            if str(file_path) in seen:
                raise ValueError(f"{branch_id} cost model file is duplicated")
            seen.add(str(file_path))
            seen_roles.add(role)
            files.append(
                {
                    "path": str(file_path),
                    "role": role,
                    "sha256": claimed,
                }
            )
        if seen_roles != set(_MODEL_FILE_ROLES[branch_id]):
            missing = sorted(_MODEL_FILE_ROLES[branch_id] - seen_roles)
            extra = sorted(seen_roles - _MODEL_FILE_ROLES[branch_id])
            raise ValueError(
                f"{branch_id} required model file-role mismatch: "
                f"missing={missing}, extra={extra}"
            )
        normalized[branch_id] = {
            "factory": branch["factory"],
            "files": files,
        }
        parameters = _mapping(
            branch["parameters"],
            f"{branch_id} cost model parameters",
        )
        expected_parameter_keys = (
            _INTERNVIT_MODEL_PARAMETER_KEYS
            if branch_id == "internvit"
            else _BRAINRW_MODEL_PARAMETER_KEYS
        )
        _exact_keys(
            parameters,
            expected_parameter_keys,
            f"{branch_id} cost model parameters",
        )
        coordinate = (
            parameters["representative_subject"],
            parameters["representative_seed"],
        )
        if coordinate != (1, 42):
            raise ValueError(
                f"{branch_id} representative cell must be fixed at sub-01/seed-42"
            )
        role_paths = {
            str(entry["role"]): str(entry["path"])
            for entry in files
        }
        checkpoint_path = _path(
            parameters["checkpoint_path"],
            f"{branch_id} representative checkpoint",
        )
        checkpoint_role = (
            "samga_checkpoint"
            if branch_id == "internvit"
            else "brainrw_checkpoint"
        )
        if role_paths[checkpoint_role] != str(checkpoint_path):
            raise ValueError(
                f"{branch_id} representative checkpoint role/path mismatch"
            )
        raw_branch_reference = _mapping(
            raw_reference_branches[branch_id],
            f"raw {branch_id} model reference",
        )
        checkpoint_entry = next(
            entry
            for entry in files
            if entry["role"] == checkpoint_role
        )
        if (
            raw_branch_reference["checkpoint_sha256"]
            != checkpoint_entry["sha256"]
        ):
            raise ValueError(
                f"{branch_id} raw checkpoint differs from its bound file"
            )
        for field_name, roles in _MODEL_AGGREGATE_ROLES[branch_id].items():
            expected_aggregate = _model_file_aggregate_sha256(files, roles)
            if raw_branch_reference[field_name] != expected_aggregate:
                raise ValueError(
                    f"{branch_id} raw model {field_name} aggregate mismatch"
                )
        sidecar_role = (
            "samga_checkpoint_sidecar"
            if branch_id == "internvit"
            else "brainrw_checkpoint_sidecar"
        )
        expected_sidecar = checkpoint_path.with_suffix(
            checkpoint_path.suffix + ".meta.json"
        )
        if role_paths[sidecar_role] != str(expected_sidecar):
            raise ValueError(f"{branch_id} checkpoint sidecar role/path mismatch")
        normalized_parameters: dict[str, object] = {
            "checkpoint_path": str(checkpoint_path),
            "representative_seed": coordinate[1],
            "representative_subject": coordinate[0],
        }
        if branch_id == "internvit":
            semantic_config_path = _path(
                parameters["semantic_config_path"],
                "InternViT semantic config",
            )
            if role_paths["semantic_config"] != str(semantic_config_path):
                raise ValueError(
                    "InternViT semantic config role/path mismatch"
                )
            foundation_model_path = _path(
                parameters["foundation_model_path"],
                "InternViT foundation model",
            )
            _require_nofollow_directory(
                foundation_model_path,
                "InternViT foundation model",
            )
            expected_foundation_roles = {
                "internvit_config": "config.json",
                "internvit_configuration_code": "configuration_intern_vit.py",
                "internvit_flash_attention_code": "flash_attention.py",
                "internvit_modeling_code": "modeling_intern_vit.py",
                "internvit_preprocessor_config": "preprocessor_config.json",
                "internvit_weight_index": "model.safetensors.index.json",
                "internvit_weight_shard_1": (
                    "model-00001-of-00003.safetensors"
                ),
                "internvit_weight_shard_2": (
                    "model-00002-of-00003.safetensors"
                ),
                "internvit_weight_shard_3": (
                    "model-00003-of-00003.safetensors"
                ),
            }
            for role, filename in expected_foundation_roles.items():
                if role_paths[role] != str(foundation_model_path / filename):
                    raise ValueError(
                        f"InternViT foundation file role/path mismatch: {role}"
                    )
            _validate_internvit_foundation_pins(files, parameters)
            normalized_parameters.update(
                {
                    "foundation_model_path": str(foundation_model_path),
                    "semantic_config_path": str(semantic_config_path),
                }
            )
        normalized[branch_id] = {
            **normalized[branch_id],  # type: ignore[arg-type]
            "parameters": normalized_parameters,
        }
    document["branches"] = normalized
    document["raw_model_reference"] = raw_reference
    return document


def load_stage1_cost_model_manifest(
    path: Path,
) -> tuple[dict[str, object], str]:
    """Load a strict model manifest and rehash every bound model file."""

    document, file_sha256 = _load_json_file(
        Path(path),
        "cost model manifest",
        require_canonical=True,
    )
    return _validate_model_document(document), file_sha256


def build_stage1_cost_model_manifest(
    *,
    branches: object,
    raw_model_reference: object,
) -> dict[str, object]:
    """Build and fully rehash the exact two-branch raw-model manifest."""

    raw_reference = _mapping(
        raw_model_reference,
        "raw cost model reference",
    )
    document = {
        "artifact_type": "samga_brain_rw.stage1_cost_model_manifest",
        "branches": _mapping(branches, "cost model branches"),
        "raw_model_reference": raw_reference,
        "raw_model_reference_sha256": sha256_json(raw_reference),
        "schema_version": 1,
        "scope": "stage1-cost",
    }
    return _validate_model_document(document)


def _validate_runtime_document(
    value: object,
    *,
    execution_config_file_sha256: str,
    execution_config_sha256: str,
) -> tuple[dict[str, object], str]:
    document = _mapping(value, "Stage 1 cost runtime manifest")
    _exact_keys(document, _RUNTIME_DOCUMENT_KEYS, "cost runtime manifest")
    if (
        document["artifact_type"]
        != "samga_brain_rw.stage1_cost_runtime_manifest"
        or document["schema_version"] != 1
        or document["scope"] != "stage1-cost"
    ):
        raise ValueError("cost runtime manifest identity/scope mismatch")
    if (
        document["execution_config_file_sha256"]
        != execution_config_file_sha256
        or document["execution_config_sha256"] != execution_config_sha256
    ):
        raise ValueError("cost runtime/execution-plan binding mismatch")
    reference = validate_self_attested_runtime_reference(
        document["runtime_reference"]
    )
    if document["runtime_reference_sha256"] != sha256_json(reference):
        raise ValueError("cost runtime-reference SHA-256 mismatch")
    evidence = _mapping(
        document["runtime_evidence"],
        "cost runtime evidence",
    )
    expected_evidence = reference["declared_runtime_observation"]
    if evidence != expected_evidence:
        raise ValueError("cost runtime evidence/reference mismatch")
    evidence_sha256 = _sha256(
        document["runtime_evidence_sha256"],
        "cost runtime evidence SHA-256",
    )
    if evidence_sha256 != sha256_json(evidence):
        raise ValueError("cost runtime evidence SHA-256 mismatch")
    return reference, evidence_sha256


def _completion_claim_identity(
    completion: object,
    *,
    job_map_sha256: str,
    row: Mapping[str, object],
    output_hashes: Mapping[str, str],
) -> tuple[int, str]:
    raw_document = _deep_thaw(getattr(completion, "document", None))
    if not isinstance(raw_document, dict):
        raise ValueError("cost completion lacks its sealed document")
    _exact_keys(
        raw_document,
        _COMPLETION_DOCUMENT_KEYS,
        "cost completion document",
    )
    schema = row.get("expected_completion_schema")
    if (
        not isinstance(schema, Mapping)
        or raw_document["schema_version"] != 1
        or raw_document["payload_type"] != schema.get("payload_type")
    ):
        raise ValueError("cost completion document identity mismatch")
    payload = _mapping(
        raw_document["payload"],
        "cost completion payload",
    )
    _exact_keys(
        payload,
        _COMPLETION_PAYLOAD_KEYS,
        "cost completion payload",
    )
    if raw_document["payload_sha256"] != sha256_json(payload):
        raise ValueError("cost completion payload SHA-256 mismatch")
    if (
        payload["job_map_sha256"] != job_map_sha256
        or payload["row_sha256"] != sha256_json(row)
        or payload["array_index"] != row.get("array_index")
        or payload["output_hashes"] != dict(output_hashes)
    ):
        raise ValueError(
            "cost completion row/output-hash binding mismatch"
        )
    generation = payload["generation"]
    if type(generation) is not int or generation <= 0:
        raise ValueError("cost completion generation must be positive")
    claim_sha256 = _sha256(
        payload["claim_sha256"],
        "cost completion claim SHA-256",
    )
    return generation, claim_sha256


def _validated_authority_execution_identity(
    value: object,
    *,
    job_map_sha256: str,
    row: Mapping[str, object],
    generation: int,
    claim_sha256: str,
) -> dict[str, object]:
    authority = _mapping(value, "cost execution authority identity")
    _exact_keys(
        authority,
        _AUTHORITY_EXECUTION_IDENTITY_KEYS,
        "cost execution authority identity",
    )
    scheduler_job_id = authority["scheduler_job_id"]
    scheduler_match = (
        _SLURM_ARRAY_JOB_RE.fullmatch(scheduler_job_id)
        if isinstance(scheduler_job_id, str)
        else None
    )
    if (
        authority["job_map_sha256"] != job_map_sha256
        or authority["row_sha256"] != sha256_json(row)
        or authority["array_index"] != row.get("array_index")
        or authority["generation"] != generation
        or authority["claim_sha256"] != claim_sha256
        or scheduler_match is None
        or int(scheduler_match.group("array_task_id"))
        != int(row["array_index"])
    ):
        raise ValueError(
            "cost execution authority does not match the completion claim"
        )
    execution_path = _path(
        authority["path"],
        "cost execution authority record",
    )
    if (
        execution_path.name != "execution.json"
        or execution_path.parent.name
        != f"generation-{generation:06d}"
    ):
        raise ValueError("cost execution authority path/generation mismatch")
    authority["path"] = str(execution_path)
    authority["sha256"] = _sha256(
        authority["sha256"],
        "cost execution authority file SHA-256",
    )
    authority["payload_sha256"] = _sha256(
        authority["payload_sha256"],
        "cost execution authority payload SHA-256",
    )
    if generation == 1:
        if (
            authority["attempt_record_sha256"] is not None
            or authority["attempt_payload_sha256"] is not None
        ):
            raise ValueError(
                "first cost execution cannot bind a recovery attempt"
            )
    else:
        authority["attempt_record_sha256"] = _sha256(
            authority["attempt_record_sha256"],
            "cost recovery attempt file SHA-256",
        )
        authority["attempt_payload_sha256"] = _sha256(
            authority["attempt_payload_sha256"],
            "cost recovery attempt payload SHA-256",
        )
    return authority


def _validate_representative_model_score_binding(
    score_inputs: list[dict[str, object]],
    model: Mapping[str, object],
) -> None:
    if len(score_inputs) != 6:
        raise ValueError("representative binding requires six score cells")
    representative = _mapping(
        score_inputs[0],
        "representative score input",
    )
    if (
        representative.get("subject") != 1
        or representative.get("seed") != 42
    ):
        raise ValueError("representative score input must be sub-01/seed-42")
    raw_reference = _mapping(
        model.get("raw_model_reference"),
        "raw cost model reference",
    )
    raw_branches = _mapping(
        raw_reference.get("branches"),
        "raw cost model branches",
    )
    for branch_id in _BRANCH_IDS:
        score_branch = _mapping(
            representative.get(branch_id),
            f"representative {branch_id} score binding",
        )
        model_branch = _mapping(
            raw_branches.get(branch_id),
            f"representative {branch_id} model binding",
        )
        if (
            score_branch.get("checkpoint_sha256")
            != model_branch.get("checkpoint_sha256")
        ):
            raise ValueError(
                f"{branch_id} representative checkpoint/score mismatch"
            )


@dataclass(frozen=True)
class _Snapshot:
    identity: dict[str, object]
    proof_sha256: str
    branch_costs: dict[str, float]
    operator_keys: dict[str, tuple[int, int, int]]
    completion: object


def _load_snapshot(
    job_map_path: Path,
    expected_job_map_sha256: str,
) -> _Snapshot:
    expected_map_sha256 = _sha256(
        expected_job_map_sha256,
        "expected cost job-map SHA-256",
    )
    job_maps = _load_job_map_module()
    load_map = getattr(job_maps, "load_job_map", None)
    load_completion = getattr(job_maps, "load_job_completion", None)
    load_execution = getattr(
        job_maps,
        "load_cost_execution_authority",
        None,
    )
    if (
        not callable(load_map)
        or not callable(load_completion)
        or not callable(load_execution)
    ):
        raise RuntimeError(
            "job-map authority lacks completion/execution validation"
        )
    job_map = load_map(
        Path(job_map_path),
        expected_sha256=expected_map_sha256,
    )
    if (
        not isinstance(job_map, Mapping)
        or job_map.get("payload_sha256") != expected_map_sha256
        or job_map.get("stage") != "stage-1-cost-benchmark"
        or job_map.get("array_bounds") != [0, 0]
        or job_map.get("row_count") != 1
    ):
        raise ValueError("cost capability requires the exact single-row job map")
    rows = job_map.get("rows")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(
        rows[0], Mapping
    ):
        raise ValueError("cost capability job map has an invalid row")
    row = rows[0]
    if (
        row.get("array_index") != 0
        or row.get("stage") != "stage-1-cost-benchmark"
        or row.get("role") != "cost-benchmark"
        or row.get("config_id") != "stage1_cost_v1"
        or row.get("partition") != "i64m1tga40u"
        or row.get("gres") != "gpu:a40:1"
    ):
        raise ValueError("cost benchmark row identity/resource mismatch")
    values = _runner_values(row)
    completion = load_completion(job_map, row)
    if completion is None or not callable(getattr(completion, "revalidate", None)):
        raise ValueError("cost capability requires a sealed job completion")
    completion.revalidate()
    output_hashes = getattr(completion, "output_hashes", None)
    if not isinstance(output_hashes, Mapping) or set(output_hashes) != set(
        _OUTPUT_HASH_NAMES
    ):
        raise ValueError("cost completion output-hash schema mismatch")
    checked_output_hashes = {
        name: _sha256(output_hashes[name], f"cost completion {name}")
        for name in _OUTPUT_HASH_NAMES
    }
    completion_generation, completion_claim_sha256 = (
        _completion_claim_identity(
            completion,
            job_map_sha256=expected_map_sha256,
            row=row,
            output_hashes=checked_output_hashes,
        )
    )
    authority_execution = _validated_authority_execution_identity(
        load_execution(
            job_map,
            row,
            expected_generation=completion_generation,
            expected_claim_sha256=completion_claim_sha256,
        ),
        job_map_sha256=expected_map_sha256,
        row=row,
        generation=completion_generation,
        claim_sha256=completion_claim_sha256,
    )

    output_dir = _path(values["--output-dir"], "cost output directory")
    run_path = output_dir / "run-manifest.json"
    runtime_path = output_dir / "runtime-manifest.json"
    run_document, run_file_sha256 = _load_json_file(
        run_path,
        "cost run manifest",
        require_canonical=True,
    )
    runtime_document, runtime_file_sha256 = _load_json_file(
        runtime_path,
        "cost runtime manifest",
        require_canonical=True,
    )
    run_manifest = _mapping(run_document, "cost run manifest")
    _exact_keys(run_manifest, _RUN_MANIFEST_KEYS, "cost run manifest")
    if (
        run_manifest["artifact_type"]
        != "samga_brain_rw.stage1_cost_run_manifest"
        or run_manifest["schema_version"] != 1
        or run_manifest["scope"] != "stage1-cost"
    ):
        raise ValueError("cost run manifest identity/scope mismatch")
    raw_path = _path(
        run_manifest["raw_record_path"],
        "raw Stage 1 cost record",
    )
    expected_attempt_index = completion_generation - 1
    expected_raw_name = (
        f"stage1-cost-attempt-{expected_attempt_index:04d}.json"
    )
    match = _RAW_RECORD_NAME_RE.fullmatch(raw_path.name)
    if (
        raw_path.parent != output_dir
        or match is None
        or raw_path.name != expected_raw_name
    ):
        raise ValueError(
            "raw cost record path does not match the current completion "
            "generation"
        )
    raw_document, raw_file_sha256 = _load_json_file(
        raw_path,
        "raw Stage 1 cost record",
        require_canonical=True,
    )
    actual_outputs = {
        "raw_record_file_sha256": raw_file_sha256,
        "run_manifest_file_sha256": run_file_sha256,
        "runtime_manifest_file_sha256": runtime_file_sha256,
    }
    if checked_output_hashes != actual_outputs:
        raise ValueError("cost completion output hashes differ from current files")

    declared_paths = {
        "authority_execution_path": _path(
            authority_execution["path"],
            "cost execution authority record",
        ),
        "protocol_path": _path(values["--config"], "cost protocol"),
        "execution_config_path": _path(
            values["--execution-config"],
            "cost execution plan",
        ),
        "score_inputs_path": _path(
            values["--score-inputs"],
            "cost score-input manifest",
        ),
        "model_manifest_path": _path(
            values["--model-manifest"],
            "cost model manifest",
        ),
        "runner_path": _path(
            row["argv"][1],  # type: ignore[index]
            "cost runner",
        ),
        "raw_record_path": raw_path,
        "runtime_manifest_path": runtime_path,
    }
    for field_name, expected_path in declared_paths.items():
        if run_manifest[field_name] != str(expected_path):
            raise ValueError(f"cost run manifest {field_name} mismatch")

    protocol_path = declared_paths["protocol_path"]
    execution_path = declared_paths["execution_config_path"]
    score_path = declared_paths["score_inputs_path"]
    model_path = declared_paths["model_manifest_path"]
    runner_path = declared_paths["runner_path"]
    _, protocol_file_sha256 = _read_regular_file(
        protocol_path,
        "cost protocol",
    )
    _, execution_file_sha256 = _read_regular_file(
        execution_path,
        "cost execution plan",
    )
    score_document, score_file_sha256 = _load_json_file(
        score_path,
        "cost score-input manifest",
        require_canonical=True,
    )
    model_document, model_file_sha256 = _load_json_file(
        model_path,
        "cost model manifest",
        require_canonical=True,
    )
    _, runner_file_sha256 = _read_regular_file(
        runner_path,
        "cost runner",
    )
    current_hashes = {
        "authority_execution_file_sha256": authority_execution["sha256"],
        "protocol_file_sha256": protocol_file_sha256,
        "execution_config_file_sha256": execution_file_sha256,
        "score_inputs_file_sha256": score_file_sha256,
        "model_manifest_file_sha256": model_file_sha256,
        "runner_file_sha256": runner_file_sha256,
        "raw_record_file_sha256": raw_file_sha256,
        "runtime_manifest_file_sha256": runtime_file_sha256,
    }
    for field_name, actual_sha256 in current_hashes.items():
        if run_manifest[field_name] != actual_sha256:
            raise ValueError(f"cost run manifest {field_name} mismatch")
    if (
        run_manifest["authority_execution_payload_sha256"]
        != authority_execution["payload_sha256"]
    ):
        raise ValueError(
            "cost run manifest authority execution payload mismatch"
        )

    protocol_document, checked_protocol_file_sha256 = _load_json_file(
        protocol_path,
        "cost measurement protocol",
        require_canonical=False,
    )
    if checked_protocol_file_sha256 != protocol_file_sha256:
        raise ValueError("cost protocol changed between stable reads")
    protocol = CostProtocol.from_payload(protocol_document)
    execution = load_stage1_cost_execution_plan(execution_path)
    if (
        run_manifest["protocol_sha256"] != protocol.sha256
        or row.get("config_sha256") != protocol.sha256
        or values["--expected-config-sha256"] != protocol.sha256
    ):
        raise ValueError("cost measurement protocol SHA-256 mismatch")
    if (
        run_manifest["execution_config_sha256"] != execution.sha256
        or values["--expected-execution-config-sha256"]
        != execution.sha256
    ):
        raise ValueError("cost execution-plan SHA-256 mismatch")
    score_inputs, raw_inputs = _validate_score_document(score_document)
    model = _validate_model_document(model_document)
    _validate_representative_model_score_binding(score_inputs, model)
    runtime_reference, runtime_evidence_sha256 = _validate_runtime_document(
        runtime_document,
        execution_config_file_sha256=execution_file_sha256,
        execution_config_sha256=execution.sha256,
    )
    input_bundle_sha256 = sha256_json(
        {
            "execution_config_sha256": execution.sha256,
            "model_manifest_file_sha256": model_file_sha256,
            "runner_file_sha256": runner_file_sha256,
            "score_inputs_file_sha256": score_file_sha256,
        }
    )
    if (
        run_manifest["input_bundle_sha256"] != input_bundle_sha256
        or row.get("input_bundle_sha256") != input_bundle_sha256
        or values["--expected-input-bundle-sha256"] != input_bundle_sha256
    ):
        raise ValueError("cost input-bundle SHA-256 mismatch")

    raw_record = RawStage1CostRecord.from_document(raw_document)
    raw_payload = raw_record.to_payload()
    raw_claim = _mapping(
        raw_payload["self_attested_job_claim_reference"],
        "raw cost job claim",
    )
    if (
        run_manifest["raw_record_sha256"] != raw_record.record_sha256
        or raw_payload["measurement_protocol_sha256"] != protocol.sha256
        or raw_payload["self_attested_input_reference"] != raw_inputs
        or raw_payload["self_attested_model_reference"]
        != model["raw_model_reference"]
        or raw_payload["self_attested_runtime_reference"] != runtime_reference
    ):
        raise ValueError("raw cost record differs from the sealed inputs")
    if (
        raw_claim["unverified_claim_sha256"]
        != completion_claim_sha256
        or raw_claim["attempt_index"] != expected_attempt_index
        or raw_claim["claim_id"] != row.get("run_key")
        or raw_claim["claim_id"] != values["--run-key"]
        or raw_claim["slurm_partition"] != row.get("partition")
        or raw_claim["slurm_job_id"]
        != authority_execution["scheduler_job_id"]
        or raw_claim["authority_execution_file_sha256"]
        != authority_execution["sha256"]
        or raw_claim["authority_execution_payload_sha256"]
        != authority_execution["payload_sha256"]
    ):
        raise ValueError(
            "raw cost claim does not match the current scheduler execution "
            "and sealed completion"
        )
    if (
        run_manifest["runtime_evidence_sha256"]
        != runtime_evidence_sha256
    ):
        raise ValueError("cost run/runtime evidence SHA-256 mismatch")

    raw_benchmark = _mapping(
        raw_payload["branch_benchmark"],
        "raw branch benchmark",
    )
    raw_branches = _mapping(
        raw_benchmark["branches"],
        "raw branch benchmark branches",
    )
    branch_costs: dict[str, float] = {}
    for branch_id in _BRANCH_IDS:
        raw_branch = _mapping(
            raw_branches[branch_id],
            f"{branch_id} raw branch benchmark",
        )
        cost = raw_branch["milliseconds_per_query"]
        if (
            not isinstance(cost, (int, float))
            or isinstance(cost, bool)
            or float(cost) <= 0.0
        ):
            raise ValueError(f"{branch_id} measured branch cost is invalid")
        branch_costs[branch_id] = float(cost)

    operator_entries: list[dict[str, object]] = []
    operator_keys: dict[str, tuple[int, int, int]] = {}
    for config in enumerate_stage1_configs():
        key = protocol.operator_complexity[config.config_id]
        operator_keys[config.config_id] = key
        operator_entries.append(
            {
                "config_id": config.config_id,
                "operator_complexity_key": list(key),
            }
        )
    identity = {
        "artifact_type": VALIDATED_COST_CAPABILITY_TYPE,
        "branch_measured_ms_per_query": branch_costs,
        "measurement_protocol_sha256": protocol.sha256,
        "operator_complexity_keys": operator_entries,
        "raw_record_sha256": raw_record.record_sha256,
        "runtime_evidence_sha256": runtime_evidence_sha256,
        "schema_version": 1,
        "scope": "val-dev",
        "score_inputs": score_inputs,
        "score_inputs_sha256": sha256_json(score_inputs),
        "two_encoder_count": 2,
    }
    final_authority_execution = _validated_authority_execution_identity(
        load_execution(
            job_map,
            row,
            expected_generation=completion_generation,
            expected_claim_sha256=completion_claim_sha256,
        ),
        job_map_sha256=expected_map_sha256,
        row=row,
        generation=completion_generation,
        claim_sha256=completion_claim_sha256,
    )
    if final_authority_execution != authority_execution:
        raise ValueError("cost execution authority changed during validation")
    completion.revalidate()
    return _Snapshot(
        identity=identity,
        proof_sha256=sha256_json(identity),
        branch_costs=branch_costs,
        operator_keys=operator_keys,
        completion=completion,
    )


def _deep_freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _deep_freeze(child)
                for key, child in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(child) for child in value)
    return value


def _deep_thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _deep_thaw(child)
            for key, child in value.items()
        }
    if isinstance(value, tuple):
        return [_deep_thaw(child) for child in value]
    return value


@dataclass(frozen=True, init=False)
class SealedStage1CostCapability(ValidatedStage1CostCapability):
    """Nominal cost capability backed by a current sealed job completion."""

    _job_map_path: Path
    _expected_job_map_sha256: str
    _identity: Mapping[str, object]
    _proof_sha256: str
    _branch_costs: Mapping[str, float]
    _operator_keys: Mapping[str, tuple[int, int, int]]
    _construction_token: object

    def __new__(cls, *_: object, **__: object) -> "SealedStage1CostCapability":
        raise TypeError(
            "SealedStage1CostCapability requires controlled construction from "
            "a sealed A40 job completion"
        )

    @classmethod
    def _issue(
        cls,
        *,
        job_map_path: Path,
        expected_job_map_sha256: str,
        snapshot: _Snapshot,
        token: object,
    ) -> "SealedStage1CostCapability":
        if token is not _CONSTRUCTION_TOKEN:
            raise TypeError("cost capability construction is controlled")
        instance = object.__new__(cls)
        object.__setattr__(instance, "_job_map_path", Path(job_map_path))
        object.__setattr__(
            instance,
            "_expected_job_map_sha256",
            expected_job_map_sha256,
        )
        object.__setattr__(
            instance,
            "_identity",
            _deep_freeze(snapshot.identity),
        )
        object.__setattr__(instance, "_proof_sha256", snapshot.proof_sha256)
        object.__setattr__(
            instance,
            "_branch_costs",
            MappingProxyType(dict(snapshot.branch_costs)),
        )
        object.__setattr__(
            instance,
            "_operator_keys",
            MappingProxyType(dict(snapshot.operator_keys)),
        )
        object.__setattr__(instance, "_construction_token", token)
        return instance

    def revalidate(self) -> None:
        if self._construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("cost capability construction marker is invalid")
        snapshot = _load_snapshot(
            self._job_map_path,
            self._expected_job_map_sha256,
        )
        if (
            snapshot.identity != _deep_thaw(self._identity)
            or snapshot.proof_sha256 != self._proof_sha256
            or snapshot.branch_costs != dict(self._branch_costs)
            or snapshot.operator_keys != dict(self._operator_keys)
        ):
            raise ValueError("validated Stage 1 cost capability changed")

    def identity_payload(self) -> dict[str, object]:
        value = _deep_thaw(self._identity)
        if not isinstance(value, dict):
            raise AssertionError("cost capability identity must be an object")
        return value

    @property
    def proof_sha256(self) -> str:
        return self._proof_sha256

    def measured_branch_cost(self, branch_id: str) -> float:
        if branch_id not in self._branch_costs:
            raise ValueError("unknown Stage 1 cost branch")
        return float(self._branch_costs[branch_id])

    def operator_complexity_key(
        self,
        config_id: str,
    ) -> tuple[int, int, int]:
        if config_id not in self._operator_keys:
            raise ValueError("unknown Stage 1 fusion config")
        return tuple(self._operator_keys[config_id])


def load_validated_stage1_cost_capability(
    job_map_path: Path,
    expected_job_map_sha256: str,
) -> SealedStage1CostCapability:
    """Issue only from the exact current sealed cost-benchmark completion."""

    snapshot = _load_snapshot(
        Path(job_map_path),
        expected_job_map_sha256,
    )
    return SealedStage1CostCapability._issue(
        job_map_path=Path(job_map_path),
        expected_job_map_sha256=expected_job_map_sha256,
        snapshot=snapshot,
        token=_CONSTRUCTION_TOKEN,
    )


__all__ = [
    "SealedStage1CostCapability",
    "Stage1CostExecutionPlan",
    "build_stage1_cost_model_manifest",
    "build_stage1_cost_score_input_manifest",
    "load_canonical_json_document",
    "load_stage1_cost_execution_plan",
    "load_stage1_cost_model_manifest",
    "load_stage1_cost_score_input_manifest",
    "load_validated_stage1_cost_capability",
    "stable_regular_file_sha256",
]
