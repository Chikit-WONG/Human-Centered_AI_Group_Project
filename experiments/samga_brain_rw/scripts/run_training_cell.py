#!/usr/bin/env python3
"""Run one sealed Stage 0/2 development training cell.

Smoke mode is intentionally partial and never invokes the epoch-60 evaluator.
Full mode performs the three independent checkpoint emissions and four-way
parity check before it can publish a sealed job-map completion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from samga_brain_rw.checkpoint_identity import (
    validate_epoch_checkpoint_identity,
)
from samga_brain_rw.checkpoint_io import load_typed_torch_checkpoint
from samga_brain_rw.checkpoints import CHECKPOINT_PAYLOAD_TYPE
from samga_brain_rw.config import make_run_key
from samga_brain_rw.hashing import canonical_json_bytes, sha256_json
from samga_brain_rw.runtime_contract import (
    validate_environment_binding,
)
from samga_brain_rw.scores import ScoreArtifact


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_RUN_MANIFEST_BASE_KEYS = frozenset(
    {
        "schema_version",
        "payload_type",
        "stage",
        "subject",
        "seed",
        "config_id",
        "config_sha256",
        "protocol_sha256",
        "cache_sha256",
        "git_sha",
        "upstream_sha",
        "data_order_sha256",
        "candidate_spec_sha256",
        "run_key",
        "run_manifest_sha256",
    }
)
_RUN_SUMMARY_EXTRA_KEYS = frozenset(
    {
        "completed",
        "global_step",
        "final_checkpoint",
        "final_checkpoint_sha256",
        "checkpoint_hashes",
        "in_loop_score_directory",
        "max_train_steps",
        "resume_source_checkpoint_sha256",
        "environment",
        "runtime_contract",
        "runtime_contract_sha256",
        "semantic_environment_sha256",
        "runtime_evidence",
        "top1_rate",
        "top5_rate",
    }
)
_RUNTIME_EVIDENCE_KEYS = frozenset(
    {
        "accelerator_name",
        "attention_evidence_scope",
        "autocast",
        "compute_capability",
        "compute_dtype",
        "cudnn_sdp_enabled",
        "cuda_available",
        "cuda_matmul_tf32",
        "cudnn_tf32",
        "device_count",
        "flash_sdp_enabled",
        "math_sdp_enabled",
        "mem_efficient_sdp_enabled",
        "torch_sdpa_canary_passed",
        "torch_sdpa_policy",
    }
)
_SEALED_COMPONENTS = frozenset(
    {
        "formal",
        "formal_input",
        "formal_refit",
        "formal_test",
        "test",
        "test_images",
        "val_confirm",
    }
)
_FORMAL_TEST_RECORD_SHA256 = (
    "02d7e33b3fe8e5a571f8db232ca5fa86"
    "abb0c16981876ec84feae7ba64636f1a"
)
_EVALUATION_DIRECTORIES = (
    "saved_checkpoint",
    "repeat_emission",
    "reload_evaluation",
)
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


@dataclass(frozen=True)
class TrainingOutputs:
    """Hashes and paths proven after the trainer exits."""

    run_manifest_path: Path
    run_manifest_sha256: str
    final_checkpoint_path: Path
    final_checkpoint_sha256: str
    in_loop_metadata_path: Path
    in_loop_metadata_sha256: str


SubprocessRunner = Callable[..., Any]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("smoke", "full"))
    parser.add_argument("--stage", required=True, type=int, choices=(0, 2))
    parser.add_argument("--role", required=True)
    parser.add_argument("--subject", required=True, type=int, choices=range(1, 11))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--resume", required=True)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--feature-cache", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-input-bundle-sha256", required=True)
    parser.add_argument("--run-key", required=True)
    parser.add_argument("--stage2-config", type=Path)
    parser.add_argument("--candidate-id")
    parser.add_argument("--adapter-rank", type=int)
    parser.add_argument("--adapter-lr-ratio", type=float)
    parser.add_argument("--whitening-artifact", type=Path)
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--device", default="cuda", choices=("cuda",))
    return parser


def parse_arguments(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.seed < 0:
        parser.error("--seed must be non-negative")
    if not arguments.resume:
        parser.error("--resume must be explicit")
    for name in ("role", "config_id"):
        value = getattr(arguments, name)
        if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
            parser.error(f"--{name.replace('_', '-')} must be a safe identifier")
    for name in ("expected_config_sha256", "expected_input_bundle_sha256"):
        if _SHA256_RE.fullmatch(getattr(arguments, name)) is None:
            parser.error(f"--{name.replace('_', '-')} must be lowercase SHA-256")
    expected_run_key = make_run_key(
        f"stage{arguments.stage}",
        arguments.config_id,
        arguments.subject,
        arguments.seed,
        arguments.expected_config_sha256,
        arguments.expected_input_bundle_sha256,
    )
    if arguments.run_key != expected_run_key:
        parser.error("--run-key does not bind the declared cell identities")
    if arguments.output_dir.name != arguments.run_key:
        parser.error("--output-dir basename must equal --run-key")
    if arguments.mode == "smoke":
        if (
            type(arguments.max_train_steps) is not int
            or arguments.max_train_steps <= 0
        ):
            parser.error("smoke mode requires positive --max-train-steps")
        if arguments.resume != "none":
            parser.error("smoke mode requires --resume none")
    elif arguments.max_train_steps is not None:
        parser.error("full mode forbids --max-train-steps")
    if arguments.stage == 0:
        if (
            arguments.stage2_config is not None
            or arguments.candidate_id is not None
            or arguments.adapter_rank is not None
            or arguments.adapter_lr_ratio is not None
            or arguments.whitening_artifact is not None
        ):
            parser.error("Stage 0 forbids every Stage 2 candidate argument")
    else:
        if arguments.stage2_config is None or arguments.candidate_id is None:
            parser.error("Stage 2 requires --stage2-config and --candidate-id")
        if arguments.candidate_id != arguments.config_id:
            parser.error("--candidate-id must equal --config-id")
        if (arguments.adapter_rank is None) != (
            arguments.adapter_lr_ratio is None
        ):
            parser.error("adapter rank and LR ratio must be supplied together")
    return arguments


def _semantic_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _development_path(path: Path, context: str) -> Path:
    raw = os.fspath(path)
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise ValueError(f"{context} must be a non-empty text path")
    if _FORMAL_TEST_RECORD_SHA256 in raw.lower():
        raise PermissionError(f"{context} contains the formal-test record hash")
    absolute = Path(os.path.abspath(os.path.normpath(raw)))
    for component in absolute.parts:
        token = _semantic_token(component)
        if token in _SEALED_COMPONENTS or re.fullmatch(
            r"sub_?\d+_test(?:_.*)?",
            token,
        ):
            raise PermissionError(f"{context} is outside development scope")
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            break
        except OSError as exc:
            raise ValueError(f"{context} cannot be inspected safely") from exc
        if stat.S_ISLNK(mode):
            raise ValueError(f"{context} contains a symlink component")
    return absolute


def _stable_regular_bytes(path: Path, context: str) -> bytes:
    flags = os.O_RDONLY | _O_CLOEXEC | _O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"{context} cannot be opened safely") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{context} must be a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)

        def identity(value: os.stat_result) -> tuple[int, ...]:
            return (
                value.st_dev,
                value.st_ino,
                value.st_size,
                value.st_mtime_ns,
                value.st_ctime_ns,
            )

        if identity(before) != identity(after):
            raise ValueError(f"{context} changed while it was read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _sha256_file(path: Path, context: str) -> str:
    return hashlib.sha256(_stable_regular_bytes(path, context)).hexdigest()


def _canonical_json_line(path: Path, context: str) -> dict[str, object]:
    raw = _stable_regular_bytes(path, context)
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise ValueError(f"{context} must be one canonical JSON line")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{context} must contain an object")
    if canonical_json_bytes(value) + b"\n" != raw:
        raise ValueError(f"{context} is not canonical JSON")
    return value


def _mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise ValueError(f"{context} must be a string-keyed mapping")
    return dict(value)


def _sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be lowercase SHA-256")
    return value


def _validate_runtime_manifest_metadata(
    value: Mapping[str, object],
) -> None:
    environment = validate_environment_binding(value["environment"])
    runtime_contract = _mapping(
        value["runtime_contract"],
        "run manifest runtime_contract",
    )
    if runtime_contract != environment["runtime_contract"]:
        raise ValueError(
            "run manifest runtime_contract differs from environment"
        )
    runtime_contract_sha256 = _sha256(
        value["runtime_contract_sha256"],
        "run manifest runtime_contract_sha256",
    )
    if runtime_contract_sha256 != environment["runtime_contract_sha256"]:
        raise ValueError("run manifest runtime contract hash mismatch")
    semantic_environment_sha256 = _sha256(
        value["semantic_environment_sha256"],
        "run manifest semantic_environment_sha256",
    )
    if (
        semantic_environment_sha256
        != environment["semantic_environment_sha256"]
    ):
        raise ValueError("run manifest semantic environment hash mismatch")

    evidence = _mapping(
        value["runtime_evidence"],
        "run manifest runtime_evidence",
    )
    if set(evidence) != _RUNTIME_EVIDENCE_KEYS:
        raise ValueError(
            "run manifest runtime_evidence keys differ from the locked schema"
        )
    if evidence["cuda_available"] is not True:
        raise ValueError("run manifest runtime evidence requires CUDA")
    if (
        type(evidence["device_count"]) is not int
        or evidence["device_count"] < 1
    ):
        raise ValueError(
            "run manifest runtime evidence device_count must be positive"
        )
    for key in _RUNTIME_EVIDENCE_KEYS - {
        "cuda_available",
        "device_count",
    }:
        if evidence[key] != runtime_contract[key]:
            raise ValueError(
                f"run manifest runtime evidence {key} mismatch"
            )


def _validate_run_manifest(
    value: Mapping[str, object],
    arguments: argparse.Namespace,
) -> dict[str, object]:
    expected_keys = _RUN_MANIFEST_BASE_KEYS | _RUN_SUMMARY_EXTRA_KEYS
    if set(value) != expected_keys:
        raise ValueError("run_manifest.json keys differ from the locked schema")
    if (
        value["schema_version"] != 1
        or value["payload_type"] != "samga_brain_rw.development_run"
    ):
        raise ValueError("run manifest identity mismatch")
    expected = {
        "stage": arguments.stage,
        "subject": arguments.subject,
        "seed": arguments.seed,
        "config_id": arguments.config_id,
        "config_sha256": arguments.expected_config_sha256,
        "run_key": arguments.run_key,
    }
    for key, expected_value in expected.items():
        if value[key] != expected_value:
            raise ValueError(f"run manifest {key} mismatch")
    manifest_body = {
        key: value[key]
        for key in _RUN_MANIFEST_BASE_KEYS
        if key != "run_manifest_sha256"
    }
    if value["run_manifest_sha256"] != sha256_json(manifest_body):
        raise ValueError("run manifest semantic hash mismatch")
    if value["in_loop_score_directory"] != "in_loop":
        raise ValueError("run manifest in-loop directory mismatch")
    if type(value["global_step"]) is not int or value["global_step"] <= 0:
        raise ValueError("run manifest global_step must be positive")
    resume_source = value["resume_source_checkpoint_sha256"]
    if resume_source is not None:
        _sha256(
            resume_source,
            "run manifest resume_source_checkpoint_sha256",
        )
    _validate_runtime_manifest_metadata(value)
    if arguments.mode == "smoke":
        if value["completed"] is not False:
            raise ValueError("smoke run must remain partial")
        if value["max_train_steps"] != arguments.max_train_steps:
            raise ValueError("smoke max_train_steps mismatch")
        if value["global_step"] != arguments.max_train_steps:
            raise ValueError("smoke global_step mismatch")
    else:
        if value["completed"] is not True:
            raise ValueError("full run must be completed")
        if value["max_train_steps"] is not None:
            raise ValueError("full run cannot declare max_train_steps")
    return dict(value)


def _validate_checkpoint(
    path: Path,
    run_manifest: Mapping[str, object],
    arguments: argparse.Namespace,
) -> tuple[Mapping[str, object], str]:
    checkpoint_sha256 = _sha256_file(path, "final checkpoint")
    if checkpoint_sha256 != _sha256(
        run_manifest["final_checkpoint_sha256"],
        "run manifest final checkpoint hash",
    ):
        raise ValueError("final checkpoint file hash mismatch")
    loaded = load_typed_torch_checkpoint(
        path,
        payload_type=CHECKPOINT_PAYLOAD_TYPE,
        requested_scope="train",
    )
    if loaded.sha256 != checkpoint_sha256:
        raise ValueError("typed checkpoint hash mismatch")
    payload = _mapping(loaded.payload, "checkpoint payload")
    validate_epoch_checkpoint_identity(
        payload,
        loaded.envelope,
    )
    expected = {
        "payload_type": CHECKPOINT_PAYLOAD_TYPE,
        "subject": arguments.subject,
        "seed": arguments.seed,
        "config_sha256": arguments.expected_config_sha256,
        "global_step": run_manifest["global_step"],
    }
    for key, expected_value in expected.items():
        if payload.get(key) != expected_value:
            raise ValueError(f"checkpoint {key} mismatch")
    checkpoint_environment = validate_environment_binding(
        payload.get("environment")
    )
    if checkpoint_environment != run_manifest["environment"]:
        raise ValueError(
            "checkpoint environment differs from run manifest"
        )
    candidate = _mapping(payload.get("candidate_spec"), "checkpoint candidate_spec")
    candidate_expected = {
        "config_id": arguments.config_id,
        "input_bundle_sha256": arguments.expected_input_bundle_sha256,
        "run_key": arguments.run_key,
    }
    for key, expected_value in candidate_expected.items():
        if candidate.get(key) != expected_value:
            raise ValueError(f"checkpoint candidate {key} mismatch")
    input_hashes = _mapping(payload.get("input_hashes"), "checkpoint input_hashes")
    if sha256_json(dict(sorted(input_hashes.items()))) != (
        arguments.expected_input_bundle_sha256
    ):
        raise ValueError("checkpoint input bundle mismatch")
    nested_manifest = _mapping(
        payload.get("run_manifest"),
        "checkpoint run_manifest",
    )
    expected_manifest = {
        key: run_manifest[key] for key in _RUN_MANIFEST_BASE_KEYS
    }
    if nested_manifest != expected_manifest:
        raise ValueError("checkpoint run manifest mismatch")
    runtime = _mapping(payload.get("runtime_state"), "checkpoint runtime_state")
    epoch = payload.get("epoch")
    if type(epoch) is not int or epoch <= 0:
        raise ValueError("checkpoint epoch must be positive")
    epoch_complete = runtime.get("epoch_complete")
    if type(epoch_complete) is not bool:
        raise ValueError("checkpoint epoch_complete must be boolean")
    if runtime.get("resume_source_checkpoint_sha256") != (
        run_manifest["resume_source_checkpoint_sha256"]
    ):
        raise ValueError(
            "checkpoint resume source differs from run manifest"
        )
    if arguments.mode == "smoke":
        if epoch_complete:
            raise ValueError("smoke checkpoint must be partial (epoch_complete=false)")
    elif epoch != 60 or not epoch_complete:
        raise ValueError("full checkpoint must be epoch-60 complete")
    return MappingProxyType(payload), checkpoint_sha256


def _validate_in_loop_score_artifact(
    score_artifact: ScoreArtifact,
    *,
    run_manifest: Mapping[str, object],
    checkpoint_payload: Mapping[str, object],
    final_checkpoint_sha256: str,
    arguments: argparse.Namespace,
) -> None:
    metadata = _mapping(
        getattr(score_artifact, "metadata", None),
        "in-loop score metadata",
    )
    expected_stage = (
        "training_smoke/in_loop"
        if arguments.mode == "smoke"
        else f"stage{arguments.stage}"
    )
    expected_identity = {
        "checkpoint_sha256": final_checkpoint_sha256,
        "config_sha256": arguments.expected_config_sha256,
        "git_sha": run_manifest["git_sha"],
        "protocol_sha256": run_manifest["protocol_sha256"],
        "seed": arguments.seed,
        "stage": expected_stage,
        "subject": arguments.subject,
    }
    for key, expected in expected_identity.items():
        if metadata.get(key) != expected:
            raise ValueError(f"in-loop score {key} mismatch")
    if metadata.get("split_role") != "val-dev":
        raise ValueError("in-loop score split_role must be val-dev")

    if arguments.mode == "smoke":
        if metadata.get("training_complete") is not False:
            raise ValueError(
                "smoke in-loop score training_complete must be false"
            )
        global_step = metadata.get("global_step")
        if (
            type(global_step) is not int
            or global_step != run_manifest["global_step"]
            or global_step != arguments.max_train_steps
        ):
            raise ValueError(
                "smoke in-loop score global_step mismatch"
            )
        planned_steps = metadata.get("planned_steps")
        if (
            type(planned_steps) is not int
            or planned_steps <= global_step
        ):
            raise ValueError(
                "smoke in-loop score planned_steps must exceed global_step"
            )

    source_records = metadata.get("source_records")
    if not isinstance(source_records, (list, tuple)) or len(source_records) != 1:
        raise ValueError(
            "in-loop score must bind exactly one source record"
        )
    source = _mapping(
        source_records[0],
        "in-loop score source record",
    )
    if source.get("role") != "val-dev":
        raise ValueError("in-loop score source record role mismatch")
    if source.get("run_key") != arguments.run_key:
        raise ValueError("in-loop score source record run_key mismatch")

    input_hashes = _mapping(
        checkpoint_payload.get("input_hashes"),
        "checkpoint input_hashes",
    )
    source_input_bindings = {
        "manifest_sha256": "manifest_sha256",
        "records_sha256": "records_sha256",
        "role_payload_sha256": "val_dev_role_sha256",
        "source_manifest_sha256": "source_manifest_sha256",
        "source_payload_sha256": "source_payload_sha256",
    }
    for source_key, input_key in source_input_bindings.items():
        if (
            input_key in input_hashes
            and source.get(source_key) != input_hashes[input_key]
        ):
            raise ValueError(
                "in-loop score source record "
                f"{source_key} input hash mismatch"
            )
    derived_input_bindings = {
        "source_payload_byte_count": "source_payload_byte_count_sha256",
        "source_payload_path": "source_payload_path_sha256",
    }
    for source_key, input_key in derived_input_bindings.items():
        if (
            input_key in input_hashes
            and sha256_json(source.get(source_key)) != input_hashes[input_key]
        ):
            raise ValueError(
                "in-loop score source record "
                f"{source_key} input hash mismatch"
            )
    if (
        "input_bundle_sha256" in source
        and source["input_bundle_sha256"]
        != arguments.expected_input_bundle_sha256
    ):
        raise ValueError(
            "in-loop score source record input_bundle_sha256 mismatch"
        )


def validate_training_command_outputs(
    argv: Sequence[str],
    *,
    expected_mode: str | None = None,
) -> TrainingOutputs:
    """Reparse one sealed runner command and validate its on-disk outputs."""

    if (
        isinstance(argv, (str, bytes, bytearray))
        or len(argv) < 3
        or any(not isinstance(value, str) or not value for value in argv)
    ):
        raise ValueError(
            "sealed training runner argv must be a string sequence"
        )
    command = list(argv)
    if command[0] != "python":
        raise ValueError("sealed training runner argv prefix mismatch")
    try:
        arguments = parse_arguments(command[2:])
    except SystemExit as exc:
        raise ValueError(
            "sealed training runner argument parsing failed"
        ) from exc
    if expected_mode not in (None, "smoke", "full"):
        raise ValueError("expected_mode must be smoke, full, or None")
    if expected_mode is not None and arguments.mode != expected_mode:
        raise ValueError(
            "sealed training runner mode differs from expected_mode"
        )
    project_root = _development_path(
        arguments.project_root,
        "sealed training command project root",
    )
    expected_runner = (
        project_root
        / "experiments/samga_brain_rw/scripts/run_training_cell.py"
    )
    if Path(command[1]) != expected_runner:
        raise ValueError(
            "sealed training runner path differs from project-root binding"
        )
    return validate_training_outputs(arguments)


def validate_training_outputs(arguments: argparse.Namespace) -> TrainingOutputs:
    """Strictly validate the trainer outputs for smoke or full mode."""

    output_dir = _development_path(arguments.output_dir, "training output")
    if output_dir.name != arguments.run_key or not output_dir.is_dir():
        raise ValueError("training output/run_key mismatch")
    run_manifest_path = output_dir / "run_manifest.json"
    run_manifest = _validate_run_manifest(
        _canonical_json_line(run_manifest_path, "run manifest"),
        arguments,
    )
    final_name = run_manifest["final_checkpoint"]
    if not isinstance(final_name, str) or Path(final_name).name != final_name:
        raise ValueError("final checkpoint must be a single filename")
    checkpoint_hashes = _mapping(
        run_manifest["checkpoint_hashes"],
        "checkpoint hashes",
    )
    if final_name not in checkpoint_hashes:
        raise ValueError("final checkpoint is missing from checkpoint_hashes")
    final_checkpoint_path = output_dir / final_name
    checkpoint_payload, final_checkpoint_sha256 = _validate_checkpoint(
        final_checkpoint_path,
        run_manifest,
        arguments,
    )
    if checkpoint_hashes[final_name] != final_checkpoint_sha256:
        raise ValueError("checkpoint_hashes final entry mismatch")
    in_loop_score = ScoreArtifact.load(
        output_dir / "in_loop",
        {"val-dev"},
    )
    _validate_in_loop_score_artifact(
        in_loop_score,
        run_manifest=run_manifest,
        checkpoint_payload=checkpoint_payload,
        final_checkpoint_sha256=final_checkpoint_sha256,
        arguments=arguments,
    )
    in_loop_metadata_path = output_dir / "in_loop" / "metadata.json"
    return TrainingOutputs(
        run_manifest_path=run_manifest_path,
        run_manifest_sha256=_sha256_file(run_manifest_path, "run manifest"),
        final_checkpoint_path=final_checkpoint_path,
        final_checkpoint_sha256=final_checkpoint_sha256,
        in_loop_metadata_path=in_loop_metadata_path,
        in_loop_metadata_sha256=_sha256_file(
            in_loop_metadata_path,
            "in-loop metadata",
        ),
    )


def _project_file(arguments: argparse.Namespace, relative: str) -> Path:
    return _development_path(
        arguments.project_root / relative,
        f"project file {relative}",
    )


def _environment(arguments: argparse.Namespace) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(
                _project_file(arguments, "experiments/samga_brain_rw")
            ),
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
        }
    )
    return environment


def _train_command(arguments: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(_project_file(arguments, "experiments/samga_brain_rw/train.py")),
        "--scope",
        "train",
        "--validation-scope",
        "val-dev",
        "--stage",
        str(arguments.stage),
        "--subject",
        str(arguments.subject),
        "--seed",
        str(arguments.seed),
        "--resume",
        arguments.resume,
        "--config",
        str(arguments.config),
        "--manifest",
        str(arguments.manifest),
        "--feature-cache",
        str(arguments.feature_cache),
        "--output-dir",
        str(arguments.output_dir),
        "--device",
        "cuda",
    ]
    if arguments.stage == 2:
        command.extend(
            [
                "--stage2-config",
                str(arguments.stage2_config),
                "--candidate-id",
                arguments.candidate_id,
            ]
        )
        if arguments.adapter_rank is not None:
            command.extend(
                [
                    "--adapter-rank",
                    str(arguments.adapter_rank),
                    "--adapter-lr-ratio",
                    str(arguments.adapter_lr_ratio),
                ]
            )
        if arguments.whitening_artifact is not None:
            command.extend(
                [
                    "--whitening-artifact",
                    str(arguments.whitening_artifact),
                ]
            )
    if arguments.mode == "smoke":
        command.extend(
            ["--max-train-steps", str(arguments.max_train_steps)]
        )
    return command


def _evaluation_command(
    arguments: argparse.Namespace,
    outputs: TrainingOutputs,
    directory: str,
) -> list[str]:
    return [
        sys.executable,
        str(_project_file(arguments, "experiments/samga_brain_rw/evaluate.py")),
        "--scope",
        "val-dev",
        "--subject",
        str(arguments.subject),
        "--seed",
        str(arguments.seed),
        "--config",
        str(arguments.config),
        "--manifest",
        str(arguments.manifest),
        "--feature-cache",
        str(arguments.feature_cache),
        "--checkpoint",
        str(outputs.final_checkpoint_path),
        "--output-dir",
        str(arguments.output_dir / directory),
        "--device",
        "cuda",
    ]


def _parity_command(arguments: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(
            _project_file(
                arguments,
                "experiments/samga_brain_rw/scripts/check_baseline_parity.py",
            )
        ),
        "--run-dir",
        str(arguments.output_dir),
        "--scope",
        "val-dev",
        "--output",
        str(arguments.output_dir / "baseline_parity.json"),
    ]


def validate_full_outputs(
    arguments: argparse.Namespace,
    outputs: TrainingOutputs,
) -> tuple[Path, str]:
    """Validate all three evaluator bundles and the four-way parity report."""

    del outputs
    output_dir = _development_path(arguments.output_dir, "full output")
    for directory in _EVALUATION_DIRECTORIES:
        ScoreArtifact.load(output_dir / directory, {"val-dev"})
    parity_path = output_dir / "baseline_parity.json"
    report = _canonical_json_line(parity_path, "baseline parity report")
    if (
        report.get("schema_version") != 1
        or report.get("report_type") != "samga_brain_rw.baseline_parity"
        or report.get("scope") != "val-dev"
        or report.get("passed") is not True
        or report.get("run_directory") != str(output_dir)
    ):
        raise ValueError("baseline parity report identity mismatch")
    return parity_path, _sha256_file(parity_path, "baseline parity report")


def _complete_command(
    arguments: argparse.Namespace,
    output_hashes: Mapping[str, str],
) -> list[str]:
    return [
        sys.executable,
        str(
            _project_file(
                arguments,
                "experiments/samga_brain_rw/scripts/build_job_map.py",
            )
        ),
        "complete-env",
        "--output-hashes",
        canonical_json_bytes(dict(sorted(output_hashes.items()))).decode("utf-8"),
    ]


def _preflight_inputs(arguments: argparse.Namespace) -> None:
    project_root = _development_path(arguments.project_root, "project root")
    if not project_root.is_dir():
        raise ValueError("project root must be an existing directory")
    if arguments.output_dir.name != arguments.run_key:
        raise ValueError("output directory does not match run_key")
    for name in ("config", "manifest", "feature_cache"):
        _development_path(getattr(arguments, name), name.replace("_", " "))
    if arguments.resume != "none":
        _development_path(Path(arguments.resume), "resume checkpoint")
    if arguments.stage2_config is not None:
        _development_path(arguments.stage2_config, "Stage 2 config")
    if arguments.whitening_artifact is not None:
        _development_path(
            arguments.whitening_artifact,
            "whitening artifact",
        )
    for relative in (
        "experiments/samga_brain_rw/train.py",
        "experiments/samga_brain_rw/evaluate.py",
        "experiments/samga_brain_rw/scripts/check_baseline_parity.py",
        "experiments/samga_brain_rw/scripts/build_job_map.py",
    ):
        path = _project_file(arguments, relative)
        if not path.is_file():
            raise ValueError(f"required project entry point is missing: {relative}")


def run_cell(
    arguments: argparse.Namespace,
    *,
    subprocess_runner: SubprocessRunner = subprocess.run,
) -> int:
    """Execute one validated development cell and optionally complete its row."""

    _preflight_inputs(arguments)
    environment = _environment(arguments)
    subprocess_runner(
        _train_command(arguments),
        check=True,
        env=environment,
    )
    outputs = validate_training_outputs(arguments)
    completion_hashes = {
        "final_checkpoint_sha256": outputs.final_checkpoint_sha256,
        "run_manifest_sha256": outputs.run_manifest_sha256,
    }
    if arguments.mode == "smoke":
        completion_hashes["in_loop_metadata_sha256"] = (
            outputs.in_loop_metadata_sha256
        )
    else:
        for directory in _EVALUATION_DIRECTORIES:
            subprocess_runner(
                _evaluation_command(arguments, outputs, directory),
                check=True,
                env=environment,
            )
        subprocess_runner(
            _parity_command(arguments),
            check=True,
            env=environment,
        )
        _, parity_sha256 = validate_full_outputs(arguments, outputs)
        completion_hashes["parity_sha256"] = parity_sha256
    if os.environ.get("SAMGA_JOB_MAP"):
        subprocess_runner(
            _complete_command(arguments, completion_hashes),
            check=True,
            env=environment,
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    try:
        arguments = parse_arguments(argv)
        return run_cell(arguments)
    except SystemExit:
        raise
    except (
        FileExistsError,
        OSError,
        PermissionError,
        RuntimeError,
        subprocess.CalledProcessError,
        TypeError,
        ValueError,
    ) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
