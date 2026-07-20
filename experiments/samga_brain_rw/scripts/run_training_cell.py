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
import math
import os
import re
import stat
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field as dataclass_field, replace
from itertools import combinations
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from samga_brain_rw.checkpoints import (
    CHECKPOINT_PAYLOAD_TYPE,
    VerifiedEpochCheckpoint,
    verify_epoch_checkpoint,
)
from samga_brain_rw.config import SemanticConfig, make_run_key
from samga_brain_rw.hashing import (
    canonical_json_bytes,
    ordered_ids_sha256,
    sha256_json,
)
from samga_brain_rw.runtime_contract import (
    validate_environment_binding,
)
from samga_brain_rw.scores import ScoreArtifact
from samga_brain_rw.trainer import SCHEDULE_SHA256


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
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_FULL_RETAINED_CHECKPOINT_NAMES = tuple(
    f"checkpoint_epoch{epoch:03d}.pt"
    for epoch in range(51, 61)
)
_LEGACY_ALL_CHECKPOINT_NAMES = tuple(
    f"checkpoint_epoch{epoch:03d}.pt"
    for epoch in range(1, 61)
)
_LEGACY_STAGE0_ALL_CHECKPOINTS_GIT_SHA = (
    "aed25e2e5756cc1f08a859d385ffb116364fa2f9"
)
_LEGACY_STAGE0_RUN_MANIFEST_FILE_SHA256 = {
    (
        "stage0__internvit_baseline_v1__sub-01__seed-42__"
        "config-6a7bdeb6994c6d3475cc6eedf56a9d82fc8892cb6170d76c75610d09697b61b5__"
        "inputs-89c0a111da7f96d4f7083432549d1031f7f22efce6e332e801f36c770a41b910"
    ): "0589e81bd8e9154277aa0798a6cc07e02bc52493a48daa3fa577a5d4fd96588d",
    (
        "stage0__internvit_baseline_v1__sub-01__seed-43__"
        "config-c2c1d2f5176ead7ac718e423f2c4c61308c1511041a61069ae324080e36f3e1e__"
        "inputs-89c0a111da7f96d4f7083432549d1031f7f22efce6e332e801f36c770a41b910"
    ): "ce6c95e7d04e73e1989d1045cccea2b4d80c4036ddf3e8dd87284a2d571b5b91",
    (
        "stage0__internvit_baseline_v1__sub-05__seed-42__"
        "config-a51faacdceb44218412c05fd759b7da5df0971b6f5615af81e9842311153ce5f__"
        "inputs-909cc2246a737c56febc52c8dc1fd914ff766d290921cf266cbb335cf3aed86c"
    ): "af010b11e19bf995f00c064e8947e638476b961efdb4bf0e1d40bbf369e4e161",
    (
        "stage0__internvit_baseline_v1__sub-05__seed-43__"
        "config-4943994fee90926122546bab2d52b3f16e82cc28f7af9a0dda4e63febac76f23__"
        "inputs-909cc2246a737c56febc52c8dc1fd914ff766d290921cf266cbb335cf3aed86c"
    ): "cabb097e9f540baf7a61022c59028132950f833c036bffa57d8bb2eee43267de",
    (
        "stage0__internvit_baseline_v1__sub-08__seed-42__"
        "config-b4a4cc8125592793399c9aa7d27e8d88dcc41be8bc2f63c421446dc2870bb4e9__"
        "inputs-0a7f7de9ba6843f213462ac4ab56456839beea9bd22f223032b0b4501c3f37c3"
    ): "a857355d61a0b776bf2d0422c4e7923eb950bf2349373267b32a4ad8c9a179ba",
    (
        "stage0__internvit_baseline_v1__sub-08__seed-43__"
        "config-1f95a5a666b5473dde4676ffb8529189c3c19c0b9f4ebc28602bc4418573ca4e__"
        "inputs-0a7f7de9ba6843f213462ac4ab56456839beea9bd22f223032b0b4501c3f37c3"
    ): "a1616a977610ce78506f2ba50023118d9f088f17899d61d9548b7714c420dbb3",
}
_CHECKPOINT_NAME_RE = re.compile(
    r"^checkpoint_epoch(?P<epoch>\d{3})"
    r"(?P<partial>_step\d{8})?\.pt$"
)


@dataclass(frozen=True)
class TrainingOutputs:
    """Hashes and paths proven after the trainer exits."""

    run_manifest_path: Path
    run_manifest_sha256: str
    final_checkpoint_path: Path
    final_checkpoint_sha256: str
    in_loop_metadata_path: Path
    in_loop_metadata_sha256: str


@dataclass(frozen=True)
class ValidatedTrainingRunProof:
    """Frozen, typed evidence for one sealed SAMGA development run."""

    outputs: TrainingOutputs
    checkpoint: VerifiedEpochCheckpoint
    run_manifest: Mapping[str, object]
    run_manifest_bytes: bytes
    in_loop_score: ScoreArtifact
    static_config_sha256: str
    resolved_config_sha256: str
    candidate_spec_sha256: str
    schedule_sha256: str
    epochs: int
    run_key: str
    input_bundle_sha256: str
    subject: int
    seed: int
    stage: str
    scope: str
    split_role: str
    protocol_sha256: str
    manifest_sha256: str
    records_sha256: str
    role_payload_sha256: str
    source_manifest_sha256: str
    source_payload_sha256: str
    source_records_sha256: str
    alignment_sha256: str
    query_ids_sha256: str
    gallery_ids_sha256: str
    git_sha: str
    semantic_environment_sha256: str
    config_path: Path
    manifest_path: Path
    output_dir: Path
    sealed_argv: tuple[str, ...] = ()
    completion_output_hashes: Mapping[str, str] = dataclass_field(
        default_factory=lambda: MappingProxyType({})
    )
    terminal_score: ScoreArtifact | None = None
    parity_artifacts: Mapping[str, ScoreArtifact] = dataclass_field(
        default_factory=lambda: MappingProxyType({})
    )
    parity_report: Mapping[str, object] | None = None
    parity_report_bytes: bytes | None = None
    parity_sha256: str | None = None


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


def _canonical_json_line_bytes(
    path: Path,
    context: str,
) -> tuple[dict[str, object], bytes]:
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
    return value, raw


def _canonical_json_line(path: Path, context: str) -> dict[str, object]:
    return _canonical_json_line_bytes(path, context)[0]


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


def _is_allowed_legacy_stage0_manifest(
    value: Mapping[str, object],
    arguments: argparse.Namespace,
    run_manifest_file_sha256: str | None,
) -> bool:
    if run_manifest_file_sha256 is None:
        return False
    actual_file_sha256 = hashlib.sha256(
        canonical_json_bytes(value) + b"\n"
    ).hexdigest()
    expected_file_sha256 = _LEGACY_STAGE0_RUN_MANIFEST_FILE_SHA256.get(
        arguments.run_key
    )
    return (
        actual_file_sha256 == run_manifest_file_sha256
        and expected_file_sha256 == run_manifest_file_sha256
        and arguments.mode == "full"
        and arguments.stage == 0
        and arguments.config_id == "internvit_baseline_v1"
        and value.get("schema_version") == 1
        and value.get("payload_type")
        == "samga_brain_rw.development_run"
        and value.get("git_sha")
        == _LEGACY_STAGE0_ALL_CHECKPOINTS_GIT_SHA
        and value.get("run_key") == arguments.run_key
    )


def _validate_checkpoint_retention_manifest(
    value: Mapping[str, object],
    arguments: argparse.Namespace,
    *,
    run_manifest_file_sha256: str | None = None,
) -> dict[str, str]:
    final_name = value.get("final_checkpoint")
    if (
        not isinstance(final_name, str)
        or Path(final_name).name != final_name
        or _CHECKPOINT_NAME_RE.fullmatch(final_name) is None
    ):
        raise ValueError("final checkpoint retention name is invalid")
    raw_hashes = _mapping(
        value.get("checkpoint_hashes"),
        "checkpoint retention hashes",
    )
    hashes = {
        name: _sha256(digest, f"checkpoint retention hash {name}")
        for name, digest in raw_hashes.items()
    }
    if final_name not in hashes:
        raise ValueError(
            "final checkpoint retention name is absent from hashes"
        )
    if any(
        Path(name).name != name
        or _CHECKPOINT_NAME_RE.fullmatch(name) is None
        for name in hashes
    ):
        raise ValueError("checkpoint retention contains an invalid name")
    legacy_stage0 = _is_allowed_legacy_stage0_manifest(
        value,
        arguments,
        run_manifest_file_sha256,
    )
    if legacy_stage0:
        if (
            tuple(sorted(hashes)) != _LEGACY_ALL_CHECKPOINT_NAMES
            or final_name != _LEGACY_ALL_CHECKPOINT_NAMES[-1]
        ):
            raise ValueError(
                "sealed legacy Stage 0 checkpoint retention must contain "
                "exact epochs 1 through 60"
            )
    elif arguments.mode == "full":
        if (
            tuple(sorted(hashes)) != _FULL_RETAINED_CHECKPOINT_NAMES
            or final_name != _FULL_RETAINED_CHECKPOINT_NAMES[-1]
        ):
            raise ValueError(
                "full checkpoint retention must contain exact epochs 51 "
                "through 60"
            )
    else:
        durable_names = tuple(
            name
            for name in _FULL_RETAINED_CHECKPOINT_NAMES
            if name in hashes
        )
        if (
            durable_names
            != _FULL_RETAINED_CHECKPOINT_NAMES[: len(durable_names)]
        ):
            raise ValueError(
                "smoke durable checkpoint retention must be a contiguous "
                "prefix beginning at epoch 51"
            )
        transient_names = set(hashes).difference(durable_names)
        if not durable_names:
            if transient_names != {final_name}:
                raise ValueError(
                    "smoke checkpoint retention before epoch 51 must "
                    "contain only its latest transient"
                )
            match = _CHECKPOINT_NAME_RE.fullmatch(final_name)
            assert match is not None
            if int(match.group("epoch")) > 51:
                raise ValueError(
                    "smoke checkpoint retention is missing its durable "
                    "prefix"
                )
        elif not transient_names:
            if final_name != durable_names[-1]:
                raise ValueError(
                    "smoke checkpoint retention boundary final checkpoint "
                    "mismatch"
                )
        elif len(transient_names) == 1:
            transient_name = next(iter(transient_names))
            match = _CHECKPOINT_NAME_RE.fullmatch(transient_name)
            assert match is not None
            expected_epoch = 51 + len(durable_names)
            if (
                transient_name != final_name
                or match.group("partial") is None
                or int(match.group("epoch")) != expected_epoch
                or expected_epoch > 60
            ):
                raise ValueError(
                    "smoke late-partial checkpoint must immediately follow "
                    "its durable retention prefix"
                )
        else:
            raise ValueError(
                "smoke checkpoint retention may contain at most one "
                "transient"
            )
    final_sha256 = _sha256(
        value.get("final_checkpoint_sha256"),
        "final checkpoint retention hash",
    )
    if hashes[final_name] != final_sha256:
        raise ValueError(
            "final checkpoint retention hash differs from checkpoint_hashes"
        )
    return hashes


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
    *,
    run_manifest_file_sha256: str | None = None,
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
    _validate_checkpoint_retention_manifest(
        value,
        arguments,
        run_manifest_file_sha256=run_manifest_file_sha256,
    )
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


def _fd_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        stat.S_IFMT(value.st_mode),
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _stable_regular_bytes_at(
    directory_fd: int,
    name: str,
    *,
    context: str,
) -> bytes:
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | _O_CLOEXEC | _O_NOFOLLOW,
            dir_fd=directory_fd,
        )
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
        if _fd_identity(before) != _fd_identity(after):
            raise ValueError(f"{context} changed while read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _sha256_regular_at(
    directory_fd: int,
    name: str,
    *,
    context: str,
) -> str:
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | _O_CLOEXEC | _O_NOFOLLOW,
            dir_fd=directory_fd,
        )
    except OSError as exc:
        raise ValueError(f"{context} cannot be opened safely") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{context} must be a regular file")
        try:
            named_before = os.stat(
                name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise ValueError(f"{context} changed before read") from exc
        if _fd_identity(named_before) != _fd_identity(before):
            raise ValueError(f"{context} path identity mismatch")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        try:
            named_after = os.stat(
                name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise ValueError(f"{context} changed during read") from exc
        if (
            _fd_identity(before) != _fd_identity(after)
            or _fd_identity(after) != _fd_identity(named_after)
        ):
            raise ValueError(f"{context} changed while read")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _canonical_json_document(
    raw: bytes,
    *,
    context: str,
) -> dict[str, object]:
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise ValueError(f"{context} must be one canonical JSON line")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context} is invalid JSON") from exc
    if (
        not isinstance(value, dict)
        or canonical_json_bytes(value) + b"\n" != raw
    ):
        raise ValueError(f"{context} is not canonical JSON")
    return value


def _crossbind_verified_checkpoint(
    verified: VerifiedEpochCheckpoint,
    *,
    path: Path,
    name: str,
    expected_sha256: str,
    run_manifest: Mapping[str, object],
    arguments: argparse.Namespace,
) -> None:
    normalized_path = Path(
        os.path.abspath(os.path.normpath(os.fspath(path)))
    )
    if verified.path != normalized_path:
        raise ValueError(f"retained checkpoint {name} path mismatch")
    if verified.sha256 != expected_sha256:
        raise ValueError(
            f"typed checkpoint hash mismatch for retained checkpoint {name}"
        )
    match = _CHECKPOINT_NAME_RE.fullmatch(name)
    if match is None:
        raise ValueError(f"retained checkpoint {name} name mismatch")
    expected_epoch = int(match.group("epoch"))
    expected_identity = {
        "epoch": expected_epoch,
        "subject": arguments.subject,
        "seed": arguments.seed,
        "config_id": arguments.config_id,
        "config_sha256": arguments.expected_config_sha256,
        "schedule_sha256": SCHEDULE_SHA256,
        "data_order_sha256": run_manifest["data_order_sha256"],
        "candidate_spec_sha256": run_manifest["candidate_spec_sha256"],
        "input_bundle_sha256": arguments.expected_input_bundle_sha256,
        "run_key": arguments.run_key,
    }
    for field, expected in expected_identity.items():
        if getattr(verified, field) != expected:
            raise ValueError(
                f"retained checkpoint {name} {field} mismatch"
            )
    if expected_epoch >= 51 and verified.optimizer_stage != "stage2":
        raise ValueError(
            f"retained checkpoint {name} optimizer stage mismatch"
        )
    if (
        validate_environment_binding(verified.environment)
        != run_manifest["environment"]
    ):
        raise ValueError(
            f"retained checkpoint {name} environment mismatch"
        )
    expected_manifest = {
        key: run_manifest[key]
        for key in _RUN_MANIFEST_BASE_KEYS
    }
    if dict(verified.run_manifest) != expected_manifest:
        raise ValueError(
            f"retained checkpoint {name} run_key/run manifest mismatch"
        )
    candidate_expected = {
        "candidate_spec_sha256": run_manifest[
            "candidate_spec_sha256"
        ],
        "config_id": arguments.config_id,
        "data_order_sha256": run_manifest["data_order_sha256"],
        "input_bundle_sha256": arguments.expected_input_bundle_sha256,
        "run_key": arguments.run_key,
        "trajectory_sha256": verified.trajectory_sha256,
    }
    for field, expected in candidate_expected.items():
        if verified.candidate_spec.get(field) != expected:
            raise ValueError(
                f"retained checkpoint {name} candidate {field} mismatch"
            )
    epoch_complete = verified.runtime_state.get("epoch_complete")
    if type(epoch_complete) is not bool:
        raise ValueError(
            f"retained checkpoint {name} epoch_complete mismatch"
        )
    expected_epoch_complete = match.group("partial") is None
    if epoch_complete is not expected_epoch_complete:
        raise ValueError(
            f"retained checkpoint {name} epoch_complete/filename mismatch"
        )
    if verified.runtime_state.get(
        "resume_source_checkpoint_sha256"
    ) != run_manifest["resume_source_checkpoint_sha256"]:
        raise ValueError(
            f"retained checkpoint {name} resume source mismatch"
        )
    retention = verified.retention
    if name in _FULL_RETAINED_CHECKPOINT_NAMES:
        if (
            retention.get("policy")
            != "retain_exact_epochs_51_through_60"
            or retention.get("required_epochs")
            != list(range(51, 61))
            or retention.get("retain_for_averaging") is not True
        ):
            raise ValueError(
                f"retained checkpoint {name} epoch/retention mismatch"
            )
    elif retention.get("retain_for_averaging") is not False:
        raise ValueError(
            f"transient checkpoint {name} retention mismatch"
        )


def _validate_retained_checkpoint_outputs(
    output_dir: Path,
    run_manifest: Mapping[str, object],
    arguments: argparse.Namespace,
    *,
    run_manifest_file_sha256: str | None = None,
) -> tuple[dict[str, str], VerifiedEpochCheckpoint]:
    output = _development_path(output_dir, "checkpoint retention output")
    hashes = _validate_checkpoint_retention_manifest(
        run_manifest,
        arguments,
        run_manifest_file_sha256=run_manifest_file_sha256,
    )
    legacy_stage0 = _is_allowed_legacy_stage0_manifest(
        run_manifest,
        arguments,
        run_manifest_file_sha256,
    )
    expected_names = set(hashes)
    expected_names.update(f"{name}.meta.json" for name in hashes)
    try:
        directory_fd = os.open(
            output,
            os.O_RDONLY | _O_CLOEXEC | _O_NOFOLLOW | _O_DIRECTORY,
        )
    except OSError as exc:
        raise ValueError(
            "checkpoint retention output cannot be opened safely"
        ) from exc
    try:
        before = os.fstat(directory_fd)
        if not stat.S_ISDIR(before.st_mode):
            raise ValueError(
                "checkpoint retention output must be a directory"
            )
        try:
            path_before = os.stat(output, follow_symlinks=False)
            names = os.listdir(directory_fd)
        except OSError as exc:
            raise ValueError(
                "checkpoint retention output cannot be inspected safely"
            ) from exc
        if _fd_identity(path_before) != _fd_identity(before):
            raise ValueError(
                "checkpoint retention output path identity mismatch"
            )
        observed_names = {
            name
            for name in names
            if name.lstrip(".").startswith("checkpoint_epoch")
        }
        if observed_names != expected_names:
            raise ValueError(
                "checkpoint retention output does not contain the exact "
                "expected bundles"
            )
        final_verified: VerifiedEpochCheckpoint | None = None
        for name, expected_sha256 in sorted(hashes.items()):
            if legacy_stage0 and name not in _FULL_RETAINED_CHECKPOINT_NAMES:
                if (
                    _sha256_regular_at(
                        directory_fd,
                        name,
                        context=f"retained checkpoint {name}",
                    )
                    != expected_sha256
                ):
                    raise ValueError(
                        f"retained checkpoint {name} hash mismatch"
                    )
            else:
                verified = verify_epoch_checkpoint(output / name)
                _crossbind_verified_checkpoint(
                    verified,
                    path=output / name,
                    name=name,
                    expected_sha256=expected_sha256,
                    run_manifest=run_manifest,
                    arguments=arguments,
                )
                if name == run_manifest["final_checkpoint"]:
                    final_verified = verified
            sidecar_name = f"{name}.meta.json"
            sidecar = _canonical_json_document(
                _stable_regular_bytes_at(
                    directory_fd,
                    sidecar_name,
                    context=f"retained checkpoint sidecar {sidecar_name}",
                ),
                context=f"retained checkpoint sidecar {sidecar_name}",
            )
            if (
                sidecar.get("schema_version") != 1
                or sidecar.get("payload_type") != CHECKPOINT_PAYLOAD_TYPE
                or sidecar.get("scope") != "train"
                or sidecar.get("payload_sha256") != expected_sha256
            ):
                raise ValueError(
                    f"retained checkpoint sidecar {sidecar_name} "
                    "binding mismatch"
                )
        after = os.fstat(directory_fd)
        try:
            path_after = os.stat(output, follow_symlinks=False)
        except OSError as exc:
            raise ValueError(
                "checkpoint retention output changed during validation"
            ) from exc
        if (
            _fd_identity(before) != _fd_identity(after)
            or _fd_identity(after) != _fd_identity(path_after)
        ):
            raise ValueError(
                "checkpoint retention output changed during validation"
            )
    finally:
        os.close(directory_fd)
    if final_verified is None:
        raise ValueError("final checkpoint did not receive full verification")
    return hashes, final_verified


def _validate_checkpoint(
    path: Path,
    run_manifest: Mapping[str, object],
    arguments: argparse.Namespace,
    *,
    verified: VerifiedEpochCheckpoint | None = None,
) -> tuple[Mapping[str, object], str]:
    expected_sha256 = _sha256(
        run_manifest["final_checkpoint_sha256"],
        "run manifest final checkpoint hash",
    )
    checkpoint = verified or verify_epoch_checkpoint(path)
    _crossbind_verified_checkpoint(
        checkpoint,
        path=path,
        name=path.name,
        expected_sha256=expected_sha256,
        run_manifest=run_manifest,
        arguments=arguments,
    )
    if checkpoint.global_step != run_manifest["global_step"]:
        raise ValueError("checkpoint global_step mismatch")
    epoch_complete = checkpoint.runtime_state["epoch_complete"]
    if arguments.mode == "smoke":
        expected_complete = "_step" not in path.name
        if epoch_complete is not expected_complete:
            raise ValueError(
                "smoke checkpoint completion differs from filename"
            )
    elif checkpoint.epoch != 60 or not epoch_complete:
        raise ValueError("full checkpoint must be epoch-60 complete")
    return checkpoint.payload, checkpoint.sha256


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


def _parse_sealed_training_command(
    argv: Sequence[str],
    *,
    expected_mode: str | None,
) -> argparse.Namespace:
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
    for field, context in (
        ("config", "sealed training command config"),
        ("manifest", "sealed training command manifest"),
        ("feature_cache", "sealed training command feature cache"),
        ("output_dir", "sealed training command output"),
    ):
        _development_path(getattr(arguments, field), context)
    if arguments.stage2_config is not None:
        _development_path(
            arguments.stage2_config,
            "sealed training command Stage 2 config",
        )
    if arguments.whitening_artifact is not None:
        _development_path(
            arguments.whitening_artifact,
            "sealed training command whitening artifact",
        )
    if arguments.resume != "none":
        _development_path(
            Path(arguments.resume),
            "sealed training command resume checkpoint",
        )
    return arguments


def validate_training_command_proof(
    argv: Sequence[str],
    *,
    expected_mode: str = "full",
) -> ValidatedTrainingRunProof:
    """Reparse one sealed command and return its immutable typed proof."""

    arguments = _parse_sealed_training_command(
        argv,
        expected_mode=expected_mode,
    )
    proof = _capture_training_run_proof(
        arguments,
        verify_static_config=True,
        sealed_argv=tuple(argv),
    )
    if arguments.mode == "full":
        proof = _validate_full_training_proof(arguments, proof)
    if not isinstance(proof, ValidatedTrainingRunProof):
        raise TypeError(
            "training command validation did not return a typed run proof"
        )
    if not isinstance(proof.checkpoint, VerifiedEpochCheckpoint):
        raise TypeError(
            "training command proof lacks a typed checkpoint"
        )
    if not isinstance(proof.in_loop_score, ScoreArtifact):
        raise TypeError(
            "training command proof lacks a typed in-loop ScoreArtifact"
        )
    if not isinstance(proof.terminal_score, ScoreArtifact):
        raise TypeError(
            "training command proof lacks a typed terminal ScoreArtifact"
        )
    expected_hash_names = (
        {
            "final_checkpoint_sha256",
            "in_loop_metadata_sha256",
            "run_manifest_sha256",
        }
        if arguments.mode == "smoke"
        else {
            "final_checkpoint_sha256",
            "parity_sha256",
            "run_manifest_sha256",
        }
    )
    if set(proof.completion_output_hashes) != expected_hash_names:
        raise ValueError(
            "training proof completion output names mismatch"
        )
    expected_common = {
        "final_checkpoint_sha256": (
            proof.outputs.final_checkpoint_sha256
        ),
        "run_manifest_sha256": proof.outputs.run_manifest_sha256,
    }
    for name, expected_value in expected_common.items():
        if proof.completion_output_hashes.get(name) != expected_value:
            raise ValueError(
                f"training proof completion {name} mismatch"
            )
    for name, digest in proof.completion_output_hashes.items():
        _sha256(digest, f"training proof completion {name}")
    return proof


def validate_training_command_outputs(
    argv: Sequence[str],
    *,
    expected_mode: str | None = None,
) -> TrainingOutputs:
    """Reparse one sealed runner command and validate its on-disk outputs."""

    arguments = _parse_sealed_training_command(
        argv,
        expected_mode=expected_mode,
    )
    return validate_training_outputs(arguments)


def _capture_training_run_proof(
    arguments: argparse.Namespace,
    *,
    verify_static_config: bool,
    sealed_argv: tuple[str, ...] = (),
) -> ValidatedTrainingRunProof:
    """Capture each training-side artifact once into one frozen proof."""

    output_dir = _development_path(arguments.output_dir, "training output")
    if output_dir.name != arguments.run_key or not output_dir.is_dir():
        raise ValueError("training output/run_key mismatch")
    run_manifest_path = output_dir / "run_manifest.json"
    run_manifest_document, run_manifest_bytes = _canonical_json_line_bytes(
        run_manifest_path,
        "run manifest",
    )
    run_manifest_file_sha256 = hashlib.sha256(
        run_manifest_bytes
    ).hexdigest()
    run_manifest = _validate_run_manifest(
        run_manifest_document,
        arguments,
        run_manifest_file_sha256=run_manifest_file_sha256,
    )
    final_name = run_manifest["final_checkpoint"]
    if not isinstance(final_name, str) or Path(final_name).name != final_name:
        raise ValueError("final checkpoint must be a single filename")
    checkpoint_hashes, verified_final = _validate_retained_checkpoint_outputs(
        output_dir,
        run_manifest,
        arguments,
        run_manifest_file_sha256=run_manifest_file_sha256,
    )
    if final_name not in checkpoint_hashes:
        raise ValueError("final checkpoint is missing from checkpoint_hashes")
    final_checkpoint_path = output_dir / final_name
    checkpoint_payload, final_checkpoint_sha256 = _validate_checkpoint(
        final_checkpoint_path,
        run_manifest,
        arguments,
        verified=verified_final,
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
    if isinstance(in_loop_score, ScoreArtifact):
        in_loop_metadata_sha256 = (
            in_loop_score.verified.envelope_sha256
        )
    else:
        if verify_static_config:
            raise TypeError(
                "training proof requires a typed in-loop ScoreArtifact"
            )
        in_loop_metadata_sha256 = _sha256_file(
            in_loop_metadata_path,
            "in-loop metadata",
        )
    outputs = TrainingOutputs(
        run_manifest_path=run_manifest_path,
        run_manifest_sha256=run_manifest_file_sha256,
        final_checkpoint_path=final_checkpoint_path,
        final_checkpoint_sha256=final_checkpoint_sha256,
        in_loop_metadata_path=in_loop_metadata_path,
        in_loop_metadata_sha256=in_loop_metadata_sha256,
    )

    candidate_spec = _mapping(
        checkpoint_payload.get("candidate_spec"),
        "checkpoint candidate_spec",
    )
    static_config_value = candidate_spec.get(
        "baseline_config_sha256"
    )
    if static_config_value is None and not verify_static_config:
        static_config_sha256 = arguments.expected_config_sha256
    else:
        static_config_sha256 = _sha256(
            static_config_value,
            "checkpoint baseline config",
        )
    if verify_static_config:
        config_path = _development_path(
            arguments.config,
            "training proof static config",
        )
        static_config = SemanticConfig.from_path(config_path)
        if static_config.sha256 != static_config_sha256:
            raise ValueError(
                "static config semantic hash differs from the checkpoint"
            )
        if arguments.resume != "none":
            raise ValueError(
                "public training proof requires null resume provenance"
            )
        if (
            run_manifest["resume_source_checkpoint_sha256"] is not None
            or verified_final.runtime_state.get(
                "resume_source_checkpoint_sha256"
            )
            is not None
        ):
            raise ValueError(
                "public training proof requires null resume parent"
            )

    metadata = _mapping(
        getattr(in_loop_score, "metadata", None),
        "in-loop score metadata",
    )
    source_records = metadata.get("source_records")
    if (
        not isinstance(source_records, (list, tuple))
        or len(source_records) != 1
    ):
        raise ValueError(
            "training proof requires exactly one val-dev source record"
        )
    source = _mapping(
        source_records[0],
        "training proof source record",
    )
    required_source_fields = {
        "manifest_sha256",
        "records_sha256",
        "role_payload_sha256",
        "source_manifest_sha256",
        "source_payload_sha256",
    }
    if verify_static_config and not required_source_fields.issubset(source):
        raise ValueError(
            "training proof source record lacks required provenance"
        )

    query_ids = tuple(getattr(in_loop_score, "query_ids", ()))
    gallery_ids = tuple(getattr(in_loop_score, "gallery_ids", ()))
    if verify_static_config and (
        not query_ids
        or not gallery_ids
        or any(not isinstance(value, str) or not value for value in query_ids)
        or any(not isinstance(value, str) or not value for value in gallery_ids)
    ):
        raise ValueError(
            "training proof requires typed ordered query/gallery IDs"
        )
    query_hash = metadata.get("query_ids_sha256")
    gallery_hash = metadata.get("gallery_ids_sha256")
    if query_hash is None:
        query_hash = ordered_ids_sha256(query_ids)
    if gallery_hash is None:
        gallery_hash = ordered_ids_sha256(gallery_ids)
    query_ids_sha256 = _sha256(query_hash, "score query IDs")
    gallery_ids_sha256 = _sha256(
        gallery_hash,
        "score gallery IDs",
    )
    if verify_static_config and (
        ordered_ids_sha256(query_ids) != query_ids_sha256
        or ordered_ids_sha256(gallery_ids) != gallery_ids_sha256
    ):
        raise ValueError(
            "training proof ordered-ID hashes do not match the score"
        )
    normalized_source_records = [
        dict(record)
        for record in source_records
        if isinstance(record, Mapping)
    ]
    source_records_sha256 = sha256_json(normalized_source_records)
    if (
        metadata.get(
            "source_records_sha256",
            source_records_sha256,
        )
        != source_records_sha256
    ):
        raise ValueError(
            "training proof source record hash mismatch"
        )
    completion_hashes = {
        "final_checkpoint_sha256": outputs.final_checkpoint_sha256,
        "run_manifest_sha256": outputs.run_manifest_sha256,
    }
    terminal_score: ScoreArtifact | None = None
    if arguments.mode == "smoke":
        completion_hashes["in_loop_metadata_sha256"] = (
            outputs.in_loop_metadata_sha256
        )
        if isinstance(in_loop_score, ScoreArtifact):
            terminal_score = in_loop_score
    return ValidatedTrainingRunProof(
        outputs=outputs,
        checkpoint=verified_final,
        run_manifest=MappingProxyType(dict(run_manifest)),
        run_manifest_bytes=bytes(run_manifest_bytes),
        in_loop_score=in_loop_score,
        static_config_sha256=static_config_sha256,
        resolved_config_sha256=arguments.expected_config_sha256,
        candidate_spec_sha256=_sha256(
            run_manifest["candidate_spec_sha256"],
            "candidate spec",
        ),
        schedule_sha256=SCHEDULE_SHA256,
        epochs=60,
        run_key=arguments.run_key,
        input_bundle_sha256=arguments.expected_input_bundle_sha256,
        subject=arguments.subject,
        seed=arguments.seed,
        stage=f"stage{arguments.stage}",
        scope="val-dev",
        split_role="val-dev",
        protocol_sha256=_sha256(
            run_manifest["protocol_sha256"],
            "run protocol",
        ),
        manifest_sha256=_sha256(
            source.get("manifest_sha256", _h_missing()),
            "source manifest",
        ),
        records_sha256=_sha256(
            source.get("records_sha256", _h_missing()),
            "source records",
        ),
        role_payload_sha256=_sha256(
            source.get("role_payload_sha256", _h_missing()),
            "source role payload",
        ),
        source_manifest_sha256=_sha256(
            source.get("source_manifest_sha256", _h_missing()),
            "source source-manifest",
        ),
        source_payload_sha256=_sha256(
            source.get("source_payload_sha256", _h_missing()),
            "source payload",
        ),
        source_records_sha256=source_records_sha256,
        alignment_sha256=ordered_ids_sha256(
            [*query_ids, *gallery_ids]
        ),
        query_ids_sha256=query_ids_sha256,
        gallery_ids_sha256=gallery_ids_sha256,
        git_sha=str(run_manifest["git_sha"]),
        semantic_environment_sha256=_sha256(
            run_manifest["semantic_environment_sha256"],
            "semantic environment",
        ),
        config_path=_development_path(
            arguments.config,
            "training proof config path",
        ),
        manifest_path=_development_path(
            arguments.manifest,
            "training proof manifest path",
        ),
        output_dir=output_dir,
        sealed_argv=sealed_argv,
        completion_output_hashes=MappingProxyType(completion_hashes),
        terminal_score=terminal_score,
    )


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _freeze_json(child)
                for key, child in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(child) for child in value)
    return value


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _thaw_json(child)
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_thaw_json(child) for child in value]
    return value


def _validate_terminal_score_against_proof(
    score: ScoreArtifact,
    *,
    arguments: argparse.Namespace,
    proof: ValidatedTrainingRunProof,
) -> None:
    if not isinstance(score, ScoreArtifact):
        raise TypeError(
            "SAMGA terminal proof requires a typed ScoreArtifact"
        )
    if not isinstance(proof.in_loop_score, ScoreArtifact):
        raise TypeError(
            "SAMGA proof requires a typed in-loop ScoreArtifact"
        )
    _validate_in_loop_score_artifact(
        score,
        run_manifest=proof.run_manifest,
        checkpoint_payload=proof.checkpoint.payload,
        final_checkpoint_sha256=proof.checkpoint.sha256,
        arguments=arguments,
    )
    if (
        tuple(score.query_ids) != tuple(proof.in_loop_score.query_ids)
        or tuple(score.gallery_ids)
        != tuple(proof.in_loop_score.gallery_ids)
        or score.query_ids_sha256 != proof.query_ids_sha256
        or score.gallery_ids_sha256 != proof.gallery_ids_sha256
        or ordered_ids_sha256(
            [*score.query_ids, *score.gallery_ids]
        )
        != proof.alignment_sha256
    ):
        raise ValueError(
            "terminal score ordered IDs differ from the training proof"
        )
    metadata = _mapping(score.metadata, "terminal score metadata")
    provenance = _mapping(
        score.provenance,
        "terminal score provenance",
    )
    source_records = metadata.get("source_records")
    plain_source_records = _thaw_json(source_records)
    if (
        not isinstance(source_records, (list, tuple))
        or not isinstance(plain_source_records, list)
        or sha256_json(plain_source_records)
        != proof.source_records_sha256
        or metadata.get("source_records_sha256")
        != proof.source_records_sha256
        or provenance.get("source_records_sha256")
        != proof.source_records_sha256
    ):
        raise ValueError(
            "terminal score source records differ from the training proof"
        )
    expected = {
        "checkpoint_sha256": proof.checkpoint.sha256,
        "config_sha256": proof.resolved_config_sha256,
        "git_sha": proof.git_sha,
        "protocol_sha256": proof.protocol_sha256,
        "seed": proof.seed,
        "split_role": proof.split_role,
        "stage": proof.stage,
        "subject": proof.subject,
        "query_ids_sha256": proof.query_ids_sha256,
        "gallery_ids_sha256": proof.gallery_ids_sha256,
        "source_records_sha256": proof.source_records_sha256,
    }
    for field_name, expected_value in expected.items():
        if (
            metadata.get(field_name) != expected_value
            or provenance.get(field_name) != expected_value
        ):
            raise ValueError(
                f"terminal score {field_name} differs from proof"
            )
    if dict(score.provenance) != dict(proof.in_loop_score.provenance):
        raise ValueError(
            "terminal score provenance differs from in-loop proof"
        )


def _retrieval_metrics_payload(score: ScoreArtifact) -> dict[str, object]:
    metrics = score.metrics
    return {
        "gallery_count": metrics.gallery_count,
        "query_count": metrics.query_count,
        "top1_count": metrics.top1_count,
        "top1_rate": metrics.top1_rate,
        "top5_count": metrics.top5_count,
        "top5_rate": metrics.top5_rate,
    }


def _parity_prediction_payload(
    score: ScoreArtifact,
) -> list[dict[str, object]]:
    return [
        {
            "predicted_gallery_id": item.predicted_gallery_id,
            "query_id": item.query_id,
            "query_index": item.query_index,
            "target_gallery_id": item.target_gallery_id,
            "target_rank": item.target_rank,
            "top1": item.top1,
            "top5": item.top5,
        }
        for item in score.metrics.predictions
    ]


def _parity_file_descriptor(
    path: Path,
    *,
    expected_sha256: str,
    context: str,
) -> dict[str, object]:
    raw = _stable_regular_bytes(path, context)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"{context} hash differs from typed artifact")
    return {"sha256": digest, "size": len(raw)}


def _maximum_absolute_score_difference(
    left: ScoreArtifact,
    right: ScoreArtifact,
) -> float:
    if left.similarity.shape != right.similarity.shape:
        raise ValueError("baseline parity score shapes differ")
    difference = np.abs(
        left.similarity.astype(np.longdouble, copy=False)
        - right.similarity.astype(np.longdouble, copy=False)
    )
    maximum = float(np.max(difference))
    if not math.isfinite(maximum) or maximum < 0.0:
        raise ValueError(
            "baseline parity score difference must be finite and non-negative"
        )
    return maximum


def _validate_parity_report_against_artifacts(
    report: Mapping[str, object],
    *,
    output_dir: Path,
    artifacts: Mapping[str, ScoreArtifact],
) -> None:
    role_directories = (
        ("in_loop", "in_loop"),
        ("saved_checkpoint", "saved_checkpoint"),
        ("repeat_emission", "repeat_emission"),
        ("reload_evaluation", "reload_evaluation"),
    )
    expected_roles = {role for role, _ in role_directories}
    if set(artifacts) != expected_roles:
        raise ValueError("baseline parity typed artifact roles mismatch")
    output = _development_path(output_dir, "baseline parity run directory")
    if not output.is_dir():
        raise ValueError("baseline parity run directory is missing")
    run_stat = os.lstat(output)
    if not stat.S_ISDIR(run_stat.st_mode):
        raise ValueError("baseline parity run path is not a directory")
    run_path_identity = (
        run_stat.st_dev,
        run_stat.st_ino,
        stat.S_IFMT(run_stat.st_mode),
    )

    expected_keys = {
        "artifacts",
        "comparisons",
        "passed",
        "report_type",
        "run_directory",
        "run_directory_identity",
        "schema_version",
        "scope",
        "summary",
        "tolerance",
    }
    if set(report) != expected_keys:
        raise ValueError("baseline parity report schema mismatch")
    if (
        report["schema_version"] != 1
        or report["report_type"]
        != "samga_brain_rw.baseline_parity"
        or report["scope"] != "val-dev"
        or report["passed"] is not True
        or report["run_directory"] != str(output)
        or type(report["tolerance"]) is not float
        or report["tolerance"] != 1e-6
    ):
        raise ValueError("baseline parity report identity mismatch")
    reported_run_identity = _mapping(
        report["run_directory_identity"],
        "baseline parity run directory identity",
    )
    if reported_run_identity != {
        "device": run_stat.st_dev,
        "inode": run_stat.st_ino,
    }:
        raise ValueError(
            "baseline parity run directory identity mismatch"
        )
    artifact_reports = _mapping(
        report["artifacts"],
        "baseline parity artifacts",
    )
    if set(artifact_reports) != expected_roles:
        raise ValueError("baseline parity artifact roles mismatch")

    for role, directory_name in role_directories:
        score = artifacts[role]
        if not isinstance(score, ScoreArtifact):
            raise TypeError(
                f"baseline parity {role} is not a typed ScoreArtifact"
            )
        score.verified.revalidate()
        score.verified.revalidate_envelope()
        expected_directory = output / directory_name
        if (
            score.directory != expected_directory
            or score.scope != "val-dev"
        ):
            raise ValueError(
                f"baseline parity {role} directory/scope mismatch"
            )
        if (
            not bool(np.all(np.isfinite(score.similarity)))
            or score.query_ids_sha256
            != ordered_ids_sha256(score.query_ids)
            or score.gallery_ids_sha256
            != ordered_ids_sha256(score.gallery_ids)
        ):
            raise ValueError(
                f"baseline parity {role} score/ID identity mismatch"
            )
        files = {
            "metadata.json": _parity_file_descriptor(
                expected_directory / "metadata.json",
                expected_sha256=score.verified.envelope_sha256,
                context=f"baseline parity {role} metadata",
            ),
            "predictions.csv": _parity_file_descriptor(
                expected_directory / "predictions.csv",
                expected_sha256=str(
                    score.metadata["predictions_sha256"]
                ),
                context=f"baseline parity {role} predictions",
            ),
            "similarity.npy": _parity_file_descriptor(
                expected_directory / "similarity.npy",
                expected_sha256=score.verified.payload_sha256,
                context=f"baseline parity {role} similarity",
            ),
        }
        score.verified.revalidate()
        score.verified.revalidate_envelope()
        expected_report = {
            "directory": directory_name,
            "files": files,
            "gallery_ids_sha256": score.gallery_ids_sha256,
            "metrics": _retrieval_metrics_payload(score),
            "ordered_ids_sha256": ordered_ids_sha256(
                [*score.query_ids, *score.gallery_ids]
            ),
            "prediction_semantics_sha256": sha256_json(
                _parity_prediction_payload(score)
            ),
            "provenance": _thaw_json(score.provenance),
            "query_ids_sha256": score.query_ids_sha256,
            "similarity_dtype": str(score.similarity.dtype),
            "similarity_shape": [
                int(value) for value in score.similarity.shape
            ],
        }
        actual_report = _mapping(
            artifact_reports[role],
            f"baseline parity {role} report",
        )
        if actual_report != expected_report:
            raise ValueError(
                f"baseline parity {role} artifact report mismatch"
            )

    comparisons = report["comparisons"]
    if not isinstance(comparisons, list) or len(comparisons) != 6:
        raise ValueError("baseline parity comparison count mismatch")
    expected_pairs = list(combinations((role for role, _ in role_directories), 2))
    recomputed_maxima: list[float] = []
    for comparison, (left_role, right_role) in zip(
        comparisons,
        expected_pairs,
        strict=True,
    ):
        actual = _mapping(
            comparison,
            "baseline parity comparison",
        )
        left = artifacts[left_role]
        right = artifacts[right_role]
        ids_identical = (
            left.query_ids == right.query_ids
            and left.gallery_ids == right.gallery_ids
        )
        metrics_identical = left.metrics == right.metrics
        predictions_identical = (
            left.metrics.predictions == right.metrics.predictions
        )
        provenance_identical = left.provenance == right.provenance
        maximum = _maximum_absolute_score_difference(left, right)
        recomputed_maxima.append(maximum)
        if not (
            ids_identical
            and metrics_identical
            and predictions_identical
            and provenance_identical
            and maximum <= 1e-6
        ):
            raise ValueError(
                "baseline parity actual artifact comparison failed"
            )
        expected_comparison = {
            "left": left_role,
            "max_absolute_score_difference": maximum,
            "metrics_identical": True,
            "ordered_ids_identical": True,
            "predictions_identical": True,
            "provenance_identical": True,
            "right": right_role,
            "within_tolerance": True,
        }
        if actual != expected_comparison:
            raise ValueError(
                "baseline parity comparison identity/value mismatch"
            )

    summary = _mapping(report["summary"], "baseline parity summary")
    expected_summary = {
        "artifact_count": 4,
        "comparison_count": 6,
        "maximum_absolute_score_difference": max(recomputed_maxima),
        "shared_provenance_sha256": sha256_json(
            _thaw_json(artifacts["in_loop"].provenance)
        ),
    }
    if summary != expected_summary:
        raise ValueError("baseline parity summary mismatch")
    try:
        final_run_stat = os.lstat(output)
    except OSError as exc:
        raise ValueError(
            "baseline parity run directory changed during validation"
        ) from exc
    final_run_identity = (
        final_run_stat.st_dev,
        final_run_stat.st_ino,
        stat.S_IFMT(final_run_stat.st_mode),
    )
    if final_run_identity != run_path_identity:
        raise ValueError(
            "baseline parity run directory identity changed during validation"
        )


def _validate_full_training_proof(
    arguments: argparse.Namespace,
    proof: ValidatedTrainingRunProof,
) -> ValidatedTrainingRunProof:
    """Attach one terminal score and one parity document without reloading."""

    if arguments.mode != "full":
        raise ValueError("full training proof requires full mode")
    if not isinstance(proof, ValidatedTrainingRunProof):
        raise TypeError("full training proof must be typed")
    role_directories = (
        ("in_loop", "in_loop"),
        ("saved_checkpoint", "saved_checkpoint"),
        ("repeat_emission", "repeat_emission"),
        ("reload_evaluation", "reload_evaluation"),
    )
    artifacts = {
        "in_loop": proof.in_loop_score,
        **{
        role: ScoreArtifact.load(
            proof.output_dir / directory,
            {"val-dev"},
        )
        for role, directory in role_directories
        if role != "in_loop"
        },
    }
    terminal = artifacts["saved_checkpoint"]
    _validate_terminal_score_against_proof(
        terminal,
        arguments=arguments,
        proof=proof,
    )
    parity_path = proof.output_dir / "baseline_parity.json"
    parity_document, parity_bytes = _canonical_json_line_bytes(
        parity_path,
        "baseline parity report",
    )
    _validate_parity_report_against_artifacts(
        parity_document,
        output_dir=proof.output_dir,
        artifacts=artifacts,
    )
    parity_sha256 = hashlib.sha256(parity_bytes).hexdigest()
    completion_hashes = MappingProxyType(
        {
            "final_checkpoint_sha256": (
                proof.outputs.final_checkpoint_sha256
            ),
            "parity_sha256": parity_sha256,
            "run_manifest_sha256": proof.outputs.run_manifest_sha256,
        }
    )
    return replace(
        proof,
        completion_output_hashes=completion_hashes,
        terminal_score=terminal,
        parity_artifacts=MappingProxyType(dict(artifacts)),
        parity_report=_freeze_json(parity_document),
        parity_report_bytes=bytes(parity_bytes),
        parity_sha256=parity_sha256,
    )


def _h_missing() -> str:
    """Compatibility-only placeholder for legacy unit doubles."""

    return hashlib.sha256(b"missing-unit-double-field").hexdigest()


def validate_training_outputs(arguments: argparse.Namespace) -> TrainingOutputs:
    """Strictly validate the trainer outputs for smoke or full mode."""

    return _capture_training_run_proof(
        arguments,
        verify_static_config=False,
    ).outputs


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
