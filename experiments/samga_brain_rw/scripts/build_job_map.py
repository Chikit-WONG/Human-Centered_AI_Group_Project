#!/usr/bin/env python3
"""Build and enforce immutable SLURM job maps for SAMGA BrainRW runs."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import pwd
import re
import shlex
import stat
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field as dataclass_field
from datetime import datetime
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from samga_brain_rw.config import make_run_key
from samga_brain_rw.hashing import canonical_json_bytes, sha256_json


SCHEMA_VERSION = 1
JOB_MAP_TYPE = "samga_brain_rw.job_map"
CLAIM_TYPE = "samga_brain_rw.job_claim"
RECOVERY_TYPE = "samga_brain_rw.job_claim_recovery"
ATTEMPT_TYPE = "samga_brain_rw.job_attempt"
COST_EXECUTION_TYPE = "samga_brain_rw.cost_execution_authority"
SLURM_RECOVERY_AUDIT_TYPE = "samga_brain_rw.slurm_recovery_audit"
TEST_ONLY_RECOVERY_AUDIT_TYPE = "samga_brain_rw.test_only_opaque_audit"
LOG_ROOT = PurePosixPath("logs/samga_brain_rw")
DEVELOPMENT_OUTPUT_ROOTS = (
    PurePosixPath("artifacts/samga_brain_rw"),
    PurePosixPath("results/samga_brain_rw"),
)
ALLOWED_A40_PARTITIONS = {
    "debug",
    "i64m1tga40u",
    "i64m1tga40ue",
    "emergency_gpua40",
}
ROW_KEYS = {
    "array_index",
    "stage",
    "role",
    "config_id",
    "config_sha256",
    "input_bundle_sha256",
    "run_key",
    "subject",
    "seed",
    "argv",
    "partition",
    "gres",
    "cpus",
    "memory",
    "time",
    "stdout_path",
    "stderr_path",
    "completion_path",
    "expected_completion_schema",
}
RAW_ROW_KEYS = ROW_KEYS - {"array_index"}
MAP_KEYS = {
    "schema_version",
    "payload_type",
    "stage",
    "array_bounds",
    "row_count",
    "rows",
    "payload_sha256",
}
DOCUMENT_KEYS = {
    "schema_version",
    "payload_type",
    "payload",
    "payload_sha256",
}
CLAIM_PAYLOAD_KEYS = {
    "job_map_sha256",
    "row_sha256",
    "array_index",
    "generation",
    "recovered_from_claim_sha256",
    "recovery_record_sha256",
}
RECOVERY_PAYLOAD_KEYS = {
    "claim_sha256",
    "next_generation",
    "recovery_audit_sha256",
    # Deliberately required: older opaque recovery records fail closed.
    "recovery_audit_type",
    "attempt_record_sha256",
    "quarantine",
    "restart_mode",
}
ATTEMPT_PAYLOAD_KEYS = {
    "job_map_sha256",
    "row_sha256",
    "array_index",
    "generation",
    "claim_sha256",
    "scheduler_job_id",
}
COST_EXECUTION_PAYLOAD_KEYS = {
    "job_map_sha256",
    "row_sha256",
    "array_index",
    "generation",
    "claim_sha256",
    "scheduler_job_id",
    "attempt_record_sha256",
    "attempt_payload_sha256",
}
QUARANTINE_KEYS = {
    "file_count",
    "quarantine_path",
    "source_output_dir",
    "tree_sha256",
}
COMPLETION_PAYLOAD_KEYS = {
    "job_map_sha256",
    "row_sha256",
    "array_index",
    "generation",
    "claim_sha256",
    "output_hashes",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_TIME_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2})$")
_MEMORY_RE = re.compile(r"^[1-9]\d*[KMGTP]$")
_GENERATION_RE = re.compile(r"^generation-(\d{6})$")
_DEVELOPMENT_STAGE_RE = re.compile(
    r"^stage-(?P<stage>[02])-(?P<phase>smoke|pilot|full)$"
)
_BRAINRW_STAGE_RE = re.compile(r"^stage-1-brainrw-(?P<phase>smoke|pilot)$")
_BRAINRW_TOPOLOGIES = {
    "stage-1-brainrw-smoke": ((8, 42),),
    "stage-1-brainrw-pilot": tuple(
        (subject, seed) for subject in (1, 5, 8) for seed in (42, 43)
    ),
}
_BRAINRW_PILOT_PARTITIONS = frozenset(
    {
        "i64m1tga40u",
        "i64m1tga40ue",
        "emergency_gpua40",
    }
)
_SLURM_ARRAY_JOB_RE = re.compile(
    r"^(?P<array_job_id>[1-9][0-9]*)_(?P<array_task_id>0|[1-9][0-9]*)$"
)
_SLURM_JOB_ID_RE = re.compile(r"^[1-9][0-9]*$")
SLURM_RECOVERY_SACCT_FIELDS = (
    "JobIDRaw",
    "JobID",
    "State",
    "ExitCode",
    "DerivedExitCode",
    "Submit",
    "Eligible",
    "Start",
    "End",
    "ElapsedRaw",
    "Partition",
    "Account",
    "QOS",
    "UID",
    "User",
    "JobName",
    "NodeList",
    "AllocTRES",
    "ReqTRES",
    "TimelimitRaw",
    "WorkDir",
    "SubmitLine",
    "Cluster",
)
_FAILED_SLURM_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
}
_SLURM_AUDIT_PAYLOAD_KEYS = {
    "binding_mode",
    "job_map",
    "row",
    "claim",
    "attempt",
    "scheduler",
    "sacct",
    "squeue",
    "submission",
    "logs",
}


@dataclass(frozen=True)
class JobClaim:
    """An immutable claim on one sealed job-map row."""

    path: Path
    generation: int
    sha256: str
    document: dict[str, object]

    @property
    def recovery_path(self) -> Path:
        return self.path.with_name("recovery.json")

    @property
    def attempt_path(self) -> Path:
        return self.path.with_name("attempt.json")

    @property
    def cost_execution_path(self) -> Path:
        return self.path.with_name("execution.json")

    @property
    def slurm_recovery_audit_path(self) -> Path:
        return self.path.with_name("slurm-recovery-audit.json")


@dataclass(frozen=True)
class JobCompletion:
    """An immutable completion for one sealed job-map row."""

    path: Path
    sha256: str
    document: Mapping[str, object]
    output_hashes: Mapping[str, str]
    _job_map_snapshot: bytes = dataclass_field(
        repr=False,
        compare=False,
    )
    _array_index: int = dataclass_field(
        repr=False,
        compare=False,
    )

    def revalidate(self) -> None:
        """Revalidate this snapshot through map/row/current-claim state."""

        _revalidate_job_completion(self)


def _require_exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"{label} keys mismatch: missing={missing}, extra={extra}"
        )


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _require_safe_id(value: object, label: str) -> str:
    text = _require_nonempty_string(value, label)
    if _SAFE_ID_RE.fullmatch(text) is None:
        raise ValueError(f"{label} contains unsafe characters")
    return text


def _require_int(
    value: object,
    label: str,
    *,
    minimum: int,
) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _time_seconds(value: object) -> int:
    text = _require_nonempty_string(value, "time")
    match = _TIME_RE.fullmatch(text)
    if match is None:
        raise ValueError("time must use HH:MM:SS")
    hours, minutes, seconds = (int(part) for part in match.groups())
    if minutes >= 60 or seconds >= 60:
        raise ValueError("time minutes and seconds must be below 60")
    return hours * 3600 + minutes * 60 + seconds


def _is_forbidden_path_token(value: str) -> bool:
    lower = value.replace("\\", "/").lower()
    if "formal-test" in lower or "formal_test" in lower:
        return True
    if re.search(r"(^|/)test(/|$)", lower):
        return True
    if re.search(r"sub-\d+_test(?:[./_-]|$)", lower):
        return True
    return False


def _validate_slurm_log_pattern(value: object, label: str, suffix: str) -> str:
    text = _validate_log_path(value, label, suffix)
    if text.count("%A") != 1 or text.count("%a") != 1:
        raise ValueError(f"{label} must contain exactly one %A and one %a")
    remainder = text.replace("%A", "").replace("%a", "")
    if "%" in remainder:
        raise ValueError(f"{label} contains an unsupported SLURM log pattern")
    return text


def _has_flag(argv: Sequence[str], flag: str) -> bool:
    return flag in argv


def _forbid_flag(argv: Sequence[str], flag: str) -> None:
    if _has_flag(argv, flag):
        raise ValueError(f"sealed argv forbids {flag}")


def _validate_log_path(value: object, label: str, suffix: str) -> str:
    text = _require_nonempty_string(value, label)
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must remain below logs/samga_brain_rw")
    try:
        path.relative_to(LOG_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"{label} must remain below logs/samga_brain_rw"
        ) from exc
    if path.suffix != suffix:
        raise ValueError(f"{label} must end in {suffix}")
    return text


def _validate_canonical_absolute_path(
    value: object,
    label: str,
) -> Path:
    text = _require_nonempty_string(value, label)
    pure = PurePosixPath(text)
    if not pure.is_absolute() or ".." in pure.parts or pure.as_posix() != text:
        raise ValueError(f"{label} must be an absolute normalized path")
    path = Path(text)
    try:
        resolved = path.resolve(strict=False)
    except OSError as exc:
        raise ValueError(f"{label} cannot be inspected safely") from exc
    if resolved != path:
        raise ValueError(
            f"{label} contains a symlink component or is not normalized"
        )
    return path


def _flag_value(argv: Sequence[str], flag: str) -> str:
    positions = [index for index, value in enumerate(argv) if value == flag]
    if len(positions) != 1:
        raise ValueError(f"sealed argv must contain {flag} exactly once")
    position = positions[0]
    if position + 1 >= len(argv) or argv[position + 1].startswith("--"):
        raise ValueError(f"sealed argv must bind a value to {flag}")
    return argv[position + 1]


def _validate_training_runner_argv(
    row: Mapping[str, object],
    argv: Sequence[str],
) -> None:
    if len(argv) < 2:
        raise ValueError("sealed training runner argv is incomplete")
    executable = PurePosixPath(argv[1]).as_posix()
    match = _DEVELOPMENT_STAGE_RE.fullmatch(str(row["stage"]))
    expected_suffix = "experiments/samga_brain_rw/scripts/run_training_cell.py"
    if not executable.endswith(expected_suffix):
        if match is not None:
            raise ValueError(
                "development training stage requires the unified runner"
            )
        return
    if match is None:
        raise ValueError(
            "unified training runner requires a development training stage"
        )
    flags = [value for value in argv if value.startswith("--")]
    if len(flags) != len(set(flags)):
        raise ValueError(
            "sealed training runner argv contains a duplicate flag"
        )
    required = (
        "--mode",
        "--stage",
        "--role",
        "--subject",
        "--seed",
        "--resume",
        "--config",
        "--manifest",
        "--feature-cache",
        "--output-dir",
        "--project-root",
        "--config-id",
        "--expected-config-sha256",
        "--expected-input-bundle-sha256",
        "--run-key",
        "--device",
    )
    values = {flag: _flag_value(argv, flag) for flag in required}
    stage = match.group("stage")
    phase = match.group("phase")
    expected_mode = "smoke" if phase == "smoke" else "full"
    expected_run_key = make_run_key(
        f"stage{stage}",
        str(row["config_id"]),
        int(row["subject"]),
        int(row["seed"]),
        str(row["config_sha256"]),
        str(row["input_bundle_sha256"]),
    )
    if row["run_key"] != expected_run_key:
        raise ValueError("sealed row run_key is not canonical")
    expected = {
        "--mode": expected_mode,
        "--stage": stage,
        "--role": str(row["role"]),
        "--subject": str(row["subject"]),
        "--seed": str(row["seed"]),
        "--config-id": str(row["config_id"]),
        "--expected-config-sha256": str(row["config_sha256"]),
        "--expected-input-bundle-sha256": str(row["input_bundle_sha256"]),
        "--run-key": str(row["run_key"]),
        "--device": "cuda",
    }
    for flag, expected_value in expected.items():
        if values[flag] != expected_value:
            label = flag.removeprefix("--").replace("-", "_")
            raise ValueError(f"sealed argv {label} does not match the row")
    if values["--resume"] == "":
        raise ValueError("sealed argv resume must be explicit")
    for flag in (
        "--config",
        "--manifest",
        "--feature-cache",
        "--project-root",
    ):
        _require_nonempty_string(values[flag], f"argv {flag}")
    project_root = _validate_canonical_absolute_path(
        values["--project-root"],
        "sealed argv project-root",
    )
    expected_runner = (project_root / expected_suffix).as_posix()
    if executable != expected_runner:
        raise ValueError(
            "unified runner path does not match the declared project-root"
        )
    output_dir = _validate_canonical_absolute_path(
        values["--output-dir"],
        "sealed argv output directory",
    )
    completion_path = _validate_canonical_absolute_path(
        row["completion_path"],
        "completion_path",
    )
    if output_dir.name != row["run_key"]:
        raise ValueError("sealed argv output directory does not match run_key")
    if completion_path.parent != output_dir:
        raise ValueError(
            "completion_path parent does not match sealed argv output directory"
        )
    for relative_root in DEVELOPMENT_OUTPUT_ROOTS:
        development_root = project_root / relative_root.as_posix()
        try:
            relative_output = output_dir.relative_to(development_root)
        except ValueError:
            continue
        if relative_output.parts:
            break
    else:
        raise ValueError(
            "sealed argv output directory must remain below the project-root "
            "artifacts/samga_brain_rw or results/samga_brain_rw"
        )
    role = str(row["role"])
    if (stage == "0" and role != "baseline") or (
        stage == "2" and role not in {"baseline", "candidate", "control"}
    ):
        raise ValueError("training row role is invalid for its stage")
    if stage == "2":
        for flag in ("--stage2-config", "--candidate-id"):
            value = _flag_value(argv, flag)
            if not value:
                raise ValueError(f"sealed argv {flag} must be nonempty")
        if _flag_value(argv, "--candidate-id") != row["config_id"]:
            raise ValueError("sealed argv candidate does not match config_id")
    else:
        for flag in (
            "--stage2-config",
            "--candidate-id",
            "--adapter-rank",
            "--adapter-lr-ratio",
            "--whitening-artifact",
        ):
            _forbid_flag(argv, flag)
    if expected_mode == "smoke":
        text = _flag_value(argv, "--max-train-steps")
        if (
            not text.isascii()
            or not text.isdecimal()
            or int(text) <= 0
            or str(int(text)) != text
        ):
            raise ValueError(
                "smoke training runner requires positive --max-train-steps"
            )
        required_hashes = [
            "final_checkpoint_sha256",
            "in_loop_metadata_sha256",
            "run_manifest_sha256",
        ]
    else:
        _forbid_flag(argv, "--max-train-steps")
        required_hashes = [
            "final_checkpoint_sha256",
            "parity_sha256",
            "run_manifest_sha256",
        ]
    schema = row["expected_completion_schema"]
    if (
        not isinstance(schema, Mapping)
        or schema.get("required_output_hashes") != required_hashes
    ):
        raise ValueError(
            "completion schema does not match the training runner mode"
        )


def _validate_brainrw_runner_argv(
    row: Mapping[str, object],
    argv: Sequence[str],
) -> None:
    if len(argv) < 2:
        raise ValueError("sealed BrainRW runner argv is incomplete")
    executable = PurePosixPath(argv[1]).as_posix()
    match = _BRAINRW_STAGE_RE.fullmatch(str(row["stage"]))
    expected_suffix = "experiments/samga_brain_rw/scripts/run_brainrw_cell.py"
    uses_runner = executable.endswith(expected_suffix)
    if match is not None and not uses_runner:
        raise ValueError("Stage 1 BrainRW stage requires the BrainRW runner")
    if uses_runner and match is None:
        raise ValueError(
            "BrainRW runner requires stage-1-brainrw-smoke or pilot"
        )
    if match is None:
        return
    if argv[0] != "python":
        raise ValueError("sealed BrainRW runner must use the python prefix")
    argument_tokens = argv[2:]
    if (
        len(argument_tokens) % 2 != 0
        or any(not value.startswith("--") for value in argument_tokens[::2])
        or any(value.startswith("--") for value in argument_tokens[1::2])
    ):
        raise ValueError(
            "sealed BrainRW runner argv must contain only flag/value pairs"
        )
    flags = list(argument_tokens[::2])
    if len(flags) != len(set(flags)):
        raise ValueError(
            "sealed BrainRW runner argv contains a duplicate flag"
        )
    required = {
        "--mode",
        "--subject",
        "--seed",
        "--resume",
        "--config",
        "--manifest",
        "--clip-path",
        "--output-dir",
        "--project-root",
        "--config-id",
        "--expected-config-sha256",
        "--expected-input-bundle-sha256",
        "--expected-semantic-environment-sha256",
        "--run-key",
        "--device",
    }
    phase = match.group("phase")
    allowed = set(required)
    if phase == "smoke":
        allowed.add("--max-train-steps")
    if set(flags) != allowed:
        missing = sorted(allowed - set(flags))
        extra = sorted(set(flags) - allowed)
        raise ValueError(
            f"sealed BrainRW runner flags mismatch: missing={missing}, extra={extra}"
        )
    values = {flag: _flag_value(argv, flag) for flag in required}
    expected_mode = "smoke" if phase == "smoke" else "full"
    expected_run_key = make_run_key(
        "brainrw-clip-lora",
        str(row["config_id"]),
        int(row["subject"]),
        int(row["seed"]),
        str(row["config_sha256"]),
        str(row["input_bundle_sha256"]),
    )
    if row["run_key"] != expected_run_key:
        raise ValueError("sealed BrainRW row run_key is not canonical")
    expected = {
        "--mode": expected_mode,
        "--subject": str(row["subject"]),
        "--seed": str(row["seed"]),
        "--resume": "none",
        "--config-id": str(row["config_id"]),
        "--expected-config-sha256": str(row["config_sha256"]),
        "--expected-input-bundle-sha256": str(row["input_bundle_sha256"]),
        "--run-key": str(row["run_key"]),
        "--device": "cuda",
    }
    for flag, expected_value in expected.items():
        if values[flag] != expected_value:
            label = flag.removeprefix("--").replace("-", "_")
            raise ValueError(
                f"sealed BrainRW argv {label} does not match the row"
            )
    _require_sha256(
        values["--expected-semantic-environment-sha256"],
        "BrainRW semantic environment",
    )
    if row["role"] != "clip-branch":
        raise ValueError("Stage 1 BrainRW role must be clip-branch")
    if row["config_id"] != "brainrw_clip_lora_v1":
        raise ValueError(
            "Stage 1 BrainRW config_id must be brainrw_clip_lora_v1"
        )
    project_root = _validate_canonical_absolute_path(
        values["--project-root"],
        "sealed BrainRW project-root",
    )
    expected_runner = (project_root / expected_suffix).as_posix()
    if executable != expected_runner:
        raise ValueError(
            "BrainRW runner path does not match the declared project-root"
        )
    config_path = _validate_canonical_absolute_path(
        values["--config"],
        "sealed BrainRW config",
    )
    expected_config = (
        project_root
        / "experiments/samga_brain_rw/configs/brainrw_clip_lora_v1.json"
    )
    if config_path != expected_config:
        raise ValueError("sealed BrainRW config path is not the locked config")
    manifest_path = _validate_canonical_absolute_path(
        values["--manifest"],
        "sealed BrainRW manifest",
    )
    expected_manifest = (
        project_root
        / "artifacts/samga_brain_rw/protocol/manifests"
        / f"sub-{int(row['subject']):02d}_protocol.json"
    )
    if manifest_path != expected_manifest:
        raise ValueError(
            "sealed BrainRW manifest is not the subject protocol manifest"
        )
    _validate_canonical_absolute_path(
        values["--clip-path"],
        "sealed BrainRW CLIP path",
    )
    output_dir = _validate_canonical_absolute_path(
        values["--output-dir"],
        "sealed BrainRW output directory",
    )
    completion_path = _validate_canonical_absolute_path(
        row["completion_path"],
        "completion_path",
    )
    expected_output_parent = (
        project_root / "artifacts/samga_brain_rw" / str(row["stage"])
    )
    if (
        output_dir.parent != expected_output_parent
        or output_dir.name != row["run_key"]
        or completion_path != output_dir / "completion.json"
    ):
        raise ValueError(
            "sealed BrainRW output/completion does not match stage and run_key"
        )
    if phase == "smoke":
        if _flag_value(argv, "--max-train-steps") != "1":
            raise ValueError(
                "Stage 1 BrainRW smoke requires exactly one training step"
            )
        expected_resource = {
            "partition": "debug",
            "gres": "gpu:a40:1",
            "cpus": 8,
            "memory": "64G",
            "time": "00:30:00",
            "stdout_path": ("logs/samga_brain_rw/stage1_brainrw_%A_%a.out"),
            "stderr_path": ("logs/samga_brain_rw/stage1_brainrw_%A_%a.err"),
        }
        payload_type = "samga_brain_rw.brainrw_smoke_completion"
        required_hashes = [
            "final_checkpoint_sha256",
            "in_loop_metadata_sha256",
            "run_manifest_sha256",
        ]
    else:
        partition = _require_nonempty_string(row["partition"], "partition")
        if partition not in _BRAINRW_PILOT_PARTITIONS:
            raise ValueError(
                "Stage 1 BrainRW pilot resource partition mismatch"
            )
        expected_resource = {
            "gres": "gpu:a40:1",
            "cpus": 8,
            "memory": "64G",
            "time": "02:00:00",
            "stdout_path": ("logs/samga_brain_rw/stage1_brainrw_%A_%a.out"),
            "stderr_path": ("logs/samga_brain_rw/stage1_brainrw_%A_%a.err"),
        }
        payload_type = "samga_brain_rw.brainrw_full_completion"
        required_hashes = [
            "final_checkpoint_sha256",
            "run_manifest_sha256",
            "score_envelope_sha256",
            "score_payload_sha256",
        ]
    for field, expected_value in expected_resource.items():
        if row[field] != expected_value:
            raise ValueError(
                f"Stage 1 BrainRW {phase} resource {field} mismatch"
            )
    schema = row["expected_completion_schema"]
    if (
        not isinstance(schema, Mapping)
        or schema.get("schema_version") != 1
        or schema.get("payload_type") != payload_type
        or schema.get("required_output_hashes") != required_hashes
    ):
        raise ValueError(
            "Stage 1 BrainRW completion schema does not match the phase"
        )


def _validate_brainrw_map_topology(
    stage: str,
    rows: Sequence[Mapping[str, object]],
) -> None:
    """Validate the sealed Stage 1 grid and its cross-row identity."""

    expected_cells = _BRAINRW_TOPOLOGIES.get(stage)
    if expected_cells is None:
        return
    cells = sorted((int(row["subject"]), int(row["seed"])) for row in rows)
    if cells != list(expected_cells):
        raise ValueError(
            "Stage 1 BrainRW map topology differs from the sealed cell grid"
        )
    identity_fields = {
        "config_id": lambda row, argv: str(row["config_id"]),
        "config_sha256": lambda row, argv: str(row["config_sha256"]),
        "semantic_environment": lambda row, argv: _flag_value(
            argv,
            "--expected-semantic-environment-sha256",
        ),
        "project_root": lambda row, argv: _flag_value(
            argv,
            "--project-root",
        ),
        "runner": lambda row, argv: str(argv[1]),
        "config_path": lambda row, argv: _flag_value(
            argv,
            "--config",
        ),
        "clip_path": lambda row, argv: _flag_value(
            argv,
            "--clip-path",
        ),
    }
    for field, extractor in identity_fields.items():
        values: set[str] = set()
        for row in rows:
            argv = row["argv"]
            if not isinstance(argv, list):
                raise ValueError("Stage 1 BrainRW argv is invalid")
            values.add(extractor(row, argv))
        if len(values) != 1:
            raise ValueError(
                f"Stage 1 BrainRW map identity is inconsistent for {field}"
            )
    for subject in sorted({int(row["subject"]) for row in rows}):
        subject_rows = [row for row in rows if int(row["subject"]) == subject]
        bundles = {str(row["input_bundle_sha256"]) for row in subject_rows}
        manifests = {
            _flag_value(row["argv"], "--manifest")  # type: ignore[arg-type]
            for row in subject_rows
        }
        if len(bundles) != 1 or len(manifests) != 1:
            raise ValueError(
                "Stage 1 BrainRW subject manifest/input bundle "
                "identity is inconsistent across seeds"
            )


def brainrw_map_identity(
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Return the comparable sealed identity of one Stage 1 map."""

    stage = payload.get("stage")
    rows = payload.get("rows")
    if (
        not isinstance(stage, str)
        or stage not in _BRAINRW_TOPOLOGIES
        or not isinstance(rows, list)
        or not rows
        or any(not isinstance(row, Mapping) for row in rows)
    ):
        raise ValueError("payload is not a sealed Stage 1 BrainRW map")
    _validate_brainrw_map_topology(stage, rows)
    first = rows[0]
    argv = first["argv"]
    if not isinstance(argv, list):
        raise ValueError("Stage 1 BrainRW argv is invalid")
    sub08 = [row for row in rows if row["subject"] == 8]
    manifests = {
        _flag_value(row["argv"], "--manifest")  # type: ignore[arg-type]
        for row in sub08
    }
    bundles = {str(row["input_bundle_sha256"]) for row in sub08}
    if len(manifests) != 1 or len(bundles) != 1:
        raise ValueError(
            "Stage 1 sub08 manifest/input identity is inconsistent"
        )
    return {
        "project_root": _flag_value(argv, "--project-root"),
        "runner": str(argv[1]),
        "config_id": str(first["config_id"]),
        "config_sha256": str(first["config_sha256"]),
        "config_path": _flag_value(argv, "--config"),
        "semantic_environment_sha256": _flag_value(
            argv,
            "--expected-semantic-environment-sha256",
        ),
        "clip_path": _flag_value(argv, "--clip-path"),
        "sub08_manifest_path": next(iter(manifests)),
        "sub08_input_bundle_sha256": next(iter(bundles)),
    }


def _validate_completion_schema(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("expected_completion_schema must be an object")
    _require_exact_keys(
        value,
        {"schema_version", "payload_type", "required_output_hashes"},
        "expected_completion_schema",
    )
    if value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("expected completion schema_version must be 1")
    _require_safe_id(value["payload_type"], "completion payload_type")
    names = value["required_output_hashes"]
    if not isinstance(names, list) or not names:
        raise ValueError("required_output_hashes must be a non-empty list")
    if any(
        not isinstance(name, str)
        or not name.endswith("_sha256")
        or _SAFE_ID_RE.fullmatch(name) is None
        for name in names
    ):
        raise ValueError("required output hashes must be safe *_sha256 names")
    if len(set(names)) != len(names) or names != sorted(names):
        raise ValueError("required_output_hashes must be unique and sorted")
    return value


def _validate_row(
    row: Mapping[str, object],
    *,
    expect_index: bool,
) -> dict[str, object]:
    expected_keys = ROW_KEYS if expect_index else RAW_ROW_KEYS
    _require_exact_keys(row, expected_keys, "job-map row")
    if expect_index:
        _require_int(row["array_index"], "array_index", minimum=0)
    _require_safe_id(row["stage"], "stage")
    _require_safe_id(row["role"], "role")
    _require_safe_id(row["config_id"], "config_id")
    _require_sha256(row["config_sha256"], "config_sha256")
    _require_sha256(row["input_bundle_sha256"], "input_bundle_sha256")
    _require_safe_id(row["run_key"], "run_key")
    subject = _require_int(row["subject"], "subject", minimum=1)
    seed = _require_int(row["seed"], "seed", minimum=0)

    argv = row["argv"]
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(value, str) or not value for value in argv)
    ):
        raise ValueError("sealed argv must be a non-empty list of strings")
    if any(_is_forbidden_path_token(value) for value in argv):
        raise ValueError("sealed argv contains a forbidden test/formal path")
    if _flag_value(argv, "--subject") != str(subject):
        raise ValueError("sealed argv subject does not match the job-map row")
    if _flag_value(argv, "--seed") != str(seed):
        raise ValueError("sealed argv seed does not match the job-map row")
    _require_nonempty_string(_flag_value(argv, "--config"), "argv config")
    _validate_training_runner_argv(row, argv)
    _validate_brainrw_runner_argv(row, argv)

    partition = _require_nonempty_string(row["partition"], "partition")
    if partition not in ALLOWED_A40_PARTITIONS:
        raise ValueError(f"unsupported A40 partition: {partition}")
    if row["gres"] != "gpu:a40:1":
        raise ValueError("gres must be exactly gpu:a40:1")
    _require_int(row["cpus"], "cpus", minimum=1)
    memory = _require_nonempty_string(row["memory"], "memory")
    if _MEMORY_RE.fullmatch(memory) is None:
        raise ValueError("memory must be a positive SLURM size such as 64G")
    seconds = _time_seconds(row["time"])
    if seconds <= 0:
        raise ValueError("time must be positive")
    if partition == "debug" and seconds > 30 * 60:
        raise ValueError("debug resource jobs must finish within 30 minutes")

    _validate_slurm_log_pattern(row["stdout_path"], "stdout_path", ".out")
    _validate_slurm_log_pattern(row["stderr_path"], "stderr_path", ".err")
    completion_path = _validate_canonical_absolute_path(
        row["completion_path"],
        "completion_path",
    )
    if _is_forbidden_path_token(str(completion_path)):
        raise ValueError(
            "completion_path contains a forbidden test/formal path"
        )
    if completion_path.suffix != ".json":
        raise ValueError("completion_path must end in .json")
    _validate_completion_schema(row["expected_completion_schema"])
    return dict(row)


def job_row_sort_key(row: Mapping[str, object]) -> tuple[object, ...]:
    """Return the one canonical ordering key for job-map rows."""

    return (
        row["stage"],
        row["role"],
        row["config_id"],
        row["subject"],
        row["seed"],
        row["config_sha256"],
        row["run_key"],
        row["input_bundle_sha256"],
        tuple(row["argv"]),  # type: ignore[arg-type]
    )


def _resource_key(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        row["partition"],
        row["gres"],
        row["cpus"],
        row["memory"],
        row["time"],
        row["stdout_path"],
        row["stderr_path"],
    )


def _logical_run_key(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        row["stage"],
        row["role"],
        row["config_id"],
        row["subject"],
        row["seed"],
        row["run_key"],
    )


def _json_copy(value: object) -> Any:
    try:
        return json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            "job map must contain only canonical JSON values"
        ) from exc


def build_job_map(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Validate, sort, index, and hash one homogeneous stage/resource map."""

    if (
        isinstance(rows, (str, bytes))
        or not isinstance(rows, Sequence)
        or not rows
    ):
        raise ValueError("job map requires at least one row")
    validated = [
        _validate_row(_json_copy(row), expect_index=False) for row in rows
    ]
    stages = {str(row["stage"]) for row in validated}
    if len(stages) != 1:
        raise ValueError("job map must contain one homogeneous stage")
    stage = next(iter(stages))
    _validate_brainrw_map_topology(stage, validated)
    resources = {_resource_key(row) for row in validated}
    if len(resources) != 1:
        raise ValueError("job map must contain one homogeneous resource class")
    logical_keys = [_logical_run_key(row) for row in validated]
    if len(set(logical_keys)) != len(logical_keys):
        raise ValueError("duplicate logical job-map row")
    for field in ("stdout_path", "stderr_path"):
        values = {str(row[field]) for row in validated}
        if len(values) != 1:
            raise ValueError(
                f"job map must use one shared sealed {field} pattern"
            )
    completion_paths = [str(row["completion_path"]) for row in validated]
    if len(set(completion_paths)) != len(completion_paths):
        raise ValueError("duplicate completion_path in job map")

    ordered = sorted(validated, key=job_row_sort_key)
    indexed = [
        {"array_index": index, **row} for index, row in enumerate(ordered)
    ]
    body: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "payload_type": JOB_MAP_TYPE,
        "stage": stage,
        "array_bounds": [0, len(indexed) - 1],
        "row_count": len(indexed),
        "rows": indexed,
    }
    return {**body, "payload_sha256": sha256_json(body)}


def validate_job_map(payload: Mapping[str, object]) -> dict[str, object]:
    """Validate every byte-bound invariant of a job map."""

    if not isinstance(payload, dict):
        raise ValueError("job map must be an object")
    _require_exact_keys(payload, MAP_KEYS, "job map")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported job-map schema_version")
    if payload["payload_type"] != JOB_MAP_TYPE:
        raise ValueError("unexpected job-map payload_type")
    stage = _require_safe_id(payload["stage"], "job-map stage")
    rows = payload["rows"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("job-map rows must be a non-empty list")
    count = _require_int(payload["row_count"], "row_count", minimum=1)
    if count != len(rows):
        raise ValueError("job-map row count mismatch")
    bounds = payload["array_bounds"]
    if (
        not isinstance(bounds, list)
        or len(bounds) != 2
        or type(bounds[0]) is not int
        or type(bounds[1]) is not int
        or bounds != [0, count - 1]
    ):
        raise ValueError("job-map array bounds mismatch")

    validated = [
        _validate_row(_json_copy(row), expect_index=True) for row in rows
    ]
    if any(row["stage"] != stage for row in validated):
        raise ValueError("job-map row stage mismatch")
    _validate_brainrw_map_topology(stage, validated)
    indices = [row["array_index"] for row in validated]
    if indices != list(range(count)):
        raise ValueError("job-map array indices must be unique and contiguous")
    if validated != sorted(validated, key=job_row_sort_key):
        raise ValueError("job-map rows are not canonically sorted")
    if len({_resource_key(row) for row in validated}) != 1:
        raise ValueError("job map must contain one homogeneous resource class")
    logical_keys = [_logical_run_key(row) for row in validated]
    if len(set(logical_keys)) != len(logical_keys):
        raise ValueError("duplicate logical job-map row")
    for field in ("stdout_path", "stderr_path"):
        values = {str(row[field]) for row in validated}
        if len(values) != 1:
            raise ValueError(
                f"job map must use one shared sealed {field} pattern"
            )
    completion_paths = [str(row["completion_path"]) for row in validated]
    if len(set(completion_paths)) != len(completion_paths):
        raise ValueError("duplicate completion_path in job map")

    claimed = _require_sha256(payload["payload_sha256"], "job-map hash")
    body = {
        key: value for key, value in payload.items() if key != "payload_sha256"
    }
    if sha256_json(body) != claimed:
        raise ValueError("job-map hash mismatch")
    return dict(payload)


def _reject_duplicate_pairs(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _read_regular_file(path: Path) -> bytes:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"cannot read sealed regular file: {path}") from exc
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError(f"sealed path is not a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


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
        fd = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise ValueError(f"cannot read sealed regular file: {path}") from exc
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError(f"sealed path is not a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def _strict_json_bytes(
    data: bytes,
    path: Path,
) -> dict[str, object]:
    try:
        result = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid sealed JSON: {path}") from exc
    if not isinstance(result, dict):
        raise ValueError(f"sealed JSON must contain an object: {path}")
    if canonical_json_bytes(result) != data:
        raise ValueError(f"sealed JSON bytes are not canonical: {path}")
    return result


def _strict_load_at(
    directory_fd: int,
    name: str,
    path: Path,
) -> dict[str, object]:
    return _strict_json_bytes(
        _read_regular_file_at(directory_fd, name, path),
        path,
    )


def _strict_load(path: Path) -> dict[str, object]:
    return _strict_json_bytes(_read_regular_file(path), path)


def _exclusive_publish(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_path, path)
        directory_fd = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)


def _exclusive_publish_at(
    directory_fd: int,
    name: str,
    data: bytes,
) -> None:
    if not name or name in {".", ".."} or PurePosixPath(name).name != name:
        raise ValueError("sealed record name must be one safe path component")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    temporary_name: str | None = None
    temporary_fd = -1
    for _ in range(128):
        candidate = f".{name}.{os.urandom(16).hex()}.tmp"
        try:
            temporary_fd = os.open(
                candidate,
                flags,
                0o600,
                dir_fd=directory_fd,
            )
        except FileExistsError:
            continue
        temporary_name = candidate
        break
    if temporary_name is None:
        raise RuntimeError("cannot allocate exclusive state-record temporary")
    try:
        with os.fdopen(temporary_fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(
            temporary_name,
            name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
        os.fsync(directory_fd)
    finally:
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass


def write_job_map(
    rows: Sequence[Mapping[str, object]],
    path: Path,
) -> dict[str, object]:
    """Publish a canonical job map without ever replacing an existing path."""

    payload = build_job_map(rows)
    _exclusive_publish(Path(path), canonical_json_bytes(payload))
    return payload


def load_job_map(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, object]:
    """Read and validate a canonical job map and optional expected map hash."""

    payload = validate_job_map(_strict_load(Path(path)))
    if expected_sha256 is not None and payload[
        "payload_sha256"
    ] != _require_sha256(expected_sha256, "expected job-map hash"):
        raise ValueError("job-map hash does not match the submitted hash")
    return payload


def select_job_row(
    payload: Mapping[str, object],
    *,
    expected_sha256: str,
    array_index: int,
    array_min: int,
    array_max: int,
) -> dict[str, object]:
    """Select a row only when map hash, scheduler bounds, and index all agree."""

    validated = validate_job_map(payload)
    expected = _require_sha256(expected_sha256, "expected job-map hash")
    if validated["payload_sha256"] != expected:
        raise ValueError("submitted job-map hash mismatch")
    bounds = validated["array_bounds"]
    if [array_min, array_max] != bounds:
        raise ValueError("SLURM array bounds do not match the job map")
    if (
        type(array_index) is not int
        or array_index < array_min
        or array_index > array_max
    ):
        raise ValueError("SLURM array index is out of range")
    row = validated["rows"][array_index]  # type: ignore[index]
    if row["array_index"] != array_index:  # type: ignore[index]
        raise ValueError("selected job-map row/index mismatch")
    return dict(row)  # type: ignore[arg-type]


def _row_context(
    payload: Mapping[str, object],
    row: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], str, str]:
    validated = validate_job_map(payload)
    checked_row = _validate_row(_json_copy(row), expect_index=True)
    index = checked_row["array_index"]
    if (
        type(index) is not int
        or index
        >= validated[  # type: ignore[operator]
            "row_count"
        ]
    ):
        raise ValueError("job row index is outside the map")
    sealed_row = validated["rows"][index]  # type: ignore[index]
    if checked_row != sealed_row:
        raise ValueError("job row does not exactly match its sealed map row")
    return (
        validated,
        checked_row,
        str(validated["payload_sha256"]),
        sha256_json(checked_row),
    )


def _output_dir(row: Mapping[str, object]) -> Path:
    completion = Path(str(row["completion_path"]))
    output_dir = completion.parent
    if output_dir == output_dir.parent:
        raise ValueError(
            "completion output directory cannot be a filesystem root"
        )
    return output_dir


def _state_dir(row: Mapping[str, object]) -> Path:
    output_dir = _output_dir(row)
    return (
        output_dir.parent
        / ".job-claims"
        / f"{output_dir.name}-array-{int(row['array_index']):06d}"
    )


def _directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _open_directory_path_nofollow(
    path: Path,
    *,
    create: bool,
) -> int:
    checked = Path(path)
    if not checked.is_absolute():
        raise ValueError(f"directory path must be absolute: {checked}")
    current_fd = os.open("/", _directory_open_flags())
    try:
        for component in checked.parts[1:]:
            if create:
                try:
                    os.mkdir(component, 0o777, dir_fd=current_fd)
                except FileExistsError:
                    pass
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


def _read_regular_file_path_nofollow(path: Path) -> bytes:
    checked = Path(path)
    if not checked.is_absolute():
        raise ValueError(f"sealed file path must be absolute: {checked}")
    try:
        directory_fd = _open_directory_path_nofollow(
            checked.parent,
            create=False,
        )
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"sealed file parent contains a symbolic link: {checked}"
        ) from exc
    try:
        return _read_regular_file_at(
            directory_fd,
            checked.name,
            checked,
        )
    finally:
        os.close(directory_fd)


@contextlib.contextmanager
def _open_state_directory(
    directory: Path,
    *,
    create: bool,
) -> Iterator[int | None]:
    try:
        state_fd = _open_directory_path_nofollow(
            directory,
            create=create,
        )
    except FileNotFoundError:
        if not create:
            yield None
            return
        raise
    except OSError as exc:
        raise ValueError(
            "state directory contains a symbolic link or is not a real "
            f"directory: {directory}"
        ) from exc
    try:
        yield state_fd
    finally:
        os.close(state_fd)


@contextlib.contextmanager
def _open_child_directory(
    parent_fd: int,
    name: str,
    path: Path,
    *,
    create: bool = False,
    exclusive: bool = False,
) -> Iterator[int]:
    if create:
        try:
            os.mkdir(name, 0o777, dir_fd=parent_fd)
        except FileExistsError as exc:
            if exclusive:
                raise ValueError(
                    f"state generation already exists or is unsafe: {path}"
                ) from exc
    try:
        child_fd = os.open(
            name,
            _directory_open_flags(),
            dir_fd=parent_fd,
        )
    except OSError as exc:
        raise ValueError(
            f"directory entry is symbolic or not a real directory: {path}"
        ) from exc
    try:
        yield child_fd
    finally:
        os.close(child_fd)


def _entry_lexists_at(
    directory_fd: int,
    name: str,
    path: Path,
) -> bool:
    try:
        os.stat(
            name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ValueError(f"cannot inspect sealed state entry: {path}") from exc
    return True


def _generation_name(generation: int) -> str:
    return f"generation-{generation:06d}"


@contextlib.contextmanager
def _open_generation_directory(
    state_fd: int,
    base: Path,
    generation: int,
    *,
    create: bool = False,
) -> Iterator[int]:
    name = _generation_name(generation)
    with _open_child_directory(
        state_fd,
        name,
        base / name,
        create=create,
        exclusive=create,
    ) as generation_fd:
        yield generation_fd


def _path_lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _directory_tree_identity(root: Path) -> dict[str, object]:
    try:
        root_mode = root.lstat().st_mode
    except OSError as exc:
        raise ValueError(f"cannot inspect output directory: {root}") from exc
    if stat.S_ISLNK(root_mode) or not stat.S_ISDIR(root_mode):
        raise ValueError(f"output path is not a real directory: {root}")

    directories: list[str] = []
    files: list[dict[str, object]] = []

    def visit(directory: Path, relative_parts: tuple[str, ...]) -> None:
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
        except OSError as exc:
            raise ValueError(
                f"cannot enumerate output directory: {directory}"
            ) from exc
        for entry in entries:
            relative = PurePosixPath(*relative_parts, entry.name).as_posix()
            try:
                mode = entry.stat(follow_symlinks=False).st_mode
            except OSError as exc:
                raise ValueError(
                    f"cannot inspect output tree entry: {entry.path}"
                ) from exc
            entry_path = Path(entry.path)
            if stat.S_ISLNK(mode):
                raise ValueError(
                    f"output tree contains a symbolic link: {entry_path}"
                )
            if stat.S_ISDIR(mode):
                directories.append(relative)
                visit(entry_path, (*relative_parts, entry.name))
                continue
            if not stat.S_ISREG(mode):
                raise ValueError(
                    f"output tree contains a non-regular entry: {entry_path}"
                )
            data = _read_regular_file(entry_path)
            files.append(
                {
                    "byte_count": len(data),
                    "path": relative,
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )

    visit(root, ())
    tree = {
        "directories": directories,
        "files": files,
    }
    return {
        "file_count": len(files),
        "tree_sha256": sha256_json(tree),
    }


def _directory_tree_identity_from_fd(
    root_fd: int,
    root: Path,
) -> dict[str, object]:
    directories: list[str] = []
    files: list[dict[str, object]] = []

    def visit(directory_fd: int, relative_parts: tuple[str, ...]) -> None:
        scan_fd = os.dup(directory_fd)
        try:
            with os.scandir(scan_fd) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
                for entry in entries:
                    relative = PurePosixPath(
                        *relative_parts,
                        entry.name,
                    ).as_posix()
                    entry_path = root / relative
                    try:
                        mode = entry.stat(follow_symlinks=False).st_mode
                    except OSError as exc:
                        raise ValueError(
                            f"cannot inspect output tree entry: {entry_path}"
                        ) from exc
                    if stat.S_ISLNK(mode):
                        raise ValueError(
                            f"output tree contains a symbolic link: {entry_path}"
                        )
                    if stat.S_ISDIR(mode):
                        directories.append(relative)
                        with _open_child_directory(
                            directory_fd,
                            entry.name,
                            entry_path,
                        ) as child_fd:
                            visit(
                                child_fd,
                                (*relative_parts, entry.name),
                            )
                        continue
                    if not stat.S_ISREG(mode):
                        raise ValueError(
                            f"output tree contains a non-regular entry: {entry_path}"
                        )
                    data = _read_regular_file_at(
                        directory_fd,
                        entry.name,
                        entry_path,
                    )
                    files.append(
                        {
                            "byte_count": len(data),
                            "path": relative,
                            "sha256": hashlib.sha256(data).hexdigest(),
                        }
                    )
        finally:
            os.close(scan_fd)

    visit(root_fd, ())
    return {
        "file_count": len(files),
        "tree_sha256": sha256_json(
            {"directories": directories, "files": files}
        ),
    }


def _directory_tree_identity_at(
    parent_fd: int,
    name: str,
    root: Path,
) -> dict[str, object]:
    with _open_child_directory(parent_fd, name, root) as root_fd:
        return _directory_tree_identity_from_fd(root_fd, root)


def _quarantine_path(base: Path, generation: int) -> Path:
    return base / "quarantine" / f"generation-{generation:06d}-output"


def _quarantine_output(
    row: Mapping[str, object],
    *,
    base: Path,
    generation: int,
    state_fd: int,
) -> dict[str, object] | None:
    output_dir = _output_dir(row)
    quarantine_path = _quarantine_path(base, generation)
    _validate_canonical_absolute_path(
        str(quarantine_path),
        "recovery quarantine path",
    )
    output_exists = _path_lexists(output_dir)
    quarantine_root = base / "quarantine"
    quarantine_name = quarantine_path.name
    with _open_child_directory(
        state_fd,
        "quarantine",
        quarantine_root,
        create=True,
    ) as quarantine_fd:
        quarantine_exists = _entry_lexists_at(
            quarantine_fd,
            quarantine_name,
            quarantine_path,
        )
        if output_exists and quarantine_exists:
            raise RuntimeError(
                "source output and its canonical quarantine both exist"
            )
        if quarantine_exists:
            identity = _directory_tree_identity_at(
                quarantine_fd,
                quarantine_name,
                quarantine_path,
            )
            return {
                **identity,
                "quarantine_path": str(quarantine_path),
                "source_output_dir": str(output_dir),
            }
        if not output_exists:
            return None
        identity = _directory_tree_identity(output_dir)
        try:
            os.rename(
                output_dir,
                quarantine_name,
                dst_dir_fd=quarantine_fd,
            )
            os.fsync(quarantine_fd)
        except OSError as exc:
            raise RuntimeError(
                f"cannot atomically quarantine output directory: {output_dir}"
            ) from exc
        moved_identity = _directory_tree_identity_at(
            quarantine_fd,
            quarantine_name,
            quarantine_path,
        )
        if moved_identity != identity:
            raise RuntimeError("quarantined output tree identity changed")
        return {
            **identity,
            "quarantine_path": str(quarantine_path),
            "source_output_dir": str(output_dir),
        }


def _validate_quarantine(
    value: object,
    *,
    row: Mapping[str, object],
    base: Path,
    generation: int,
    state_fd: int,
) -> None:
    expected_path = _quarantine_path(base, generation)
    _validate_canonical_absolute_path(
        str(expected_path),
        "recovery quarantine path",
    )
    quarantine_root = base / "quarantine"
    if not _entry_lexists_at(
        state_fd,
        "quarantine",
        quarantine_root,
    ):
        if value is None:
            return
        raise ValueError("recorded recovery quarantine directory is missing")
    with _open_child_directory(
        state_fd,
        "quarantine",
        quarantine_root,
    ) as quarantine_fd:
        quarantine_exists = _entry_lexists_at(
            quarantine_fd,
            expected_path.name,
            expected_path,
        )
        if value is None:
            if quarantine_exists:
                raise ValueError(
                    "recovery quarantine exists but was not recorded"
                )
            return
        actual_identity = _directory_tree_identity_at(
            quarantine_fd,
            expected_path.name,
            expected_path,
        )
    if not isinstance(value, dict):
        raise ValueError("recovery quarantine must be an object or null")
    _require_exact_keys(value, QUARANTINE_KEYS, "recovery quarantine")
    if value["quarantine_path"] != str(expected_path):
        raise ValueError("recovery quarantine path is not canonical")
    if value["source_output_dir"] != str(_output_dir(row)):
        raise ValueError("recovery quarantine source does not match the row")
    expected_identity = {
        "file_count": _require_int(
            value["file_count"],
            "quarantine file_count",
            minimum=0,
        ),
        "tree_sha256": _require_sha256(
            value["tree_sha256"],
            "quarantine tree_sha256",
        ),
    }
    if actual_identity != expected_identity:
        raise ValueError("recovery quarantine tree identity mismatch")


@contextlib.contextmanager
def _transition_lock(directory: Path) -> Iterator[int]:
    with _open_state_directory(directory, create=True) as state_fd:
        if state_fd is None:
            raise RuntimeError(
                "state directory creation returned no descriptor"
            )
        try:
            lock_fd = os.open(
                ".transition.lock",
                (
                    os.O_CREAT
                    | os.O_RDWR
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                ),
                0o600,
                dir_fd=state_fd,
            )
        except OSError as exc:
            raise ValueError(
                "transition lock is symbolic or not a regular file"
            ) from exc
        try:
            if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
                raise ValueError("transition lock is not a regular file")
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            yield state_fd
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)


def _record_document(
    payload_type: str,
    payload: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "payload_type": payload_type,
        "payload": payload,
        "payload_sha256": sha256_json(payload),
    }


def _allocate_staging_generation(
    state_fd: int,
    generation: int,
) -> tuple[str, int]:
    prefix = f".generation-staging-{generation:06d}-"
    for _ in range(128):
        staging_name = f"{prefix}{os.urandom(16).hex()}"
        try:
            os.mkdir(
                staging_name,
                0o700,
                dir_fd=state_fd,
            )
        except FileExistsError:
            continue
        try:
            staging_fd = os.open(
                staging_name,
                _directory_open_flags(),
                dir_fd=state_fd,
            )
        except BaseException:
            try:
                os.rmdir(staging_name, dir_fd=state_fd)
            except FileNotFoundError:
                pass
            raise
        return staging_name, staging_fd
    raise RuntimeError("cannot allocate unique staging generation directory")


def _discard_staging_generation(
    state_fd: int,
    staging_fd: int,
    staging_name: str,
) -> None:
    scan_fd = os.dup(staging_fd)
    try:
        with os.scandir(scan_fd) as iterator:
            entries = sorted(iterator, key=lambda entry: entry.name)
            for entry in entries:
                try:
                    mode = entry.stat(follow_symlinks=False).st_mode
                except OSError as exc:
                    raise RuntimeError(
                        "cannot inspect staging generation during cleanup"
                    ) from exc
                if not stat.S_ISREG(mode):
                    raise RuntimeError(
                        "staging generation contains an unsafe entry"
                    )
                os.unlink(entry.name, dir_fd=staging_fd)
    finally:
        os.close(scan_fd)
    os.fsync(staging_fd)
    try:
        os.rmdir(staging_name, dir_fd=state_fd)
    except FileNotFoundError:
        return
    os.fsync(state_fd)


def _rename_staged_generation(
    state_fd: int,
    staging_name: str,
    final_name: str,
) -> None:
    final_path = Path(final_name)
    if _entry_lexists_at(state_fd, final_name, final_path):
        raise ValueError("state generation already exists or is unsafe")
    try:
        os.rename(
            staging_name,
            final_name,
            src_dir_fd=state_fd,
            dst_dir_fd=state_fd,
        )
    except OSError as exc:
        raise ValueError(
            "cannot atomically publish staged state generation"
        ) from exc


def _publish_staged_generation_record(
    state_fd: int,
    *,
    base: Path,
    generation: int,
    record_name: str,
    data: bytes,
) -> None:
    final_name = _generation_name(generation)
    staging_name, staging_fd = _allocate_staging_generation(
        state_fd,
        generation,
    )
    published = False
    try:
        _exclusive_publish_at(staging_fd, record_name, data)
        os.fsync(staging_fd)
        _rename_staged_generation(
            state_fd,
            staging_name,
            final_name,
        )
        published = True
        os.fsync(state_fd)
    finally:
        try:
            if not published:
                _discard_staging_generation(
                    state_fd,
                    staging_fd,
                    staging_name,
                )
        finally:
            os.close(staging_fd)
    final_path = base / final_name / record_name
    if not _entry_lexists_at(
        state_fd,
        final_name,
        final_path.parent,
    ):
        raise RuntimeError("published generation is not visible")


def _create_record(
    path: Path,
    payload_type: str,
    payload: dict[str, object],
    *,
    directory_fd: int | None = None,
    state_fd: int | None = None,
    generation: int | None = None,
    create_generation: bool = False,
) -> tuple[dict[str, object], str]:
    if directory_fd is not None and state_fd is not None:
        raise ValueError("record publication received conflicting descriptors")
    if (state_fd is None) != (generation is None):
        raise ValueError(
            "state record publication requires both state fd and generation"
        )
    if create_generation and state_fd is None:
        raise ValueError(
            "generation creation requires a state directory descriptor"
        )
    document = _record_document(payload_type, payload)
    data = canonical_json_bytes(document)
    if directory_fd is not None:
        _exclusive_publish_at(directory_fd, path.name, data)
    elif state_fd is not None and generation is not None:
        if path.parent.name != _generation_name(generation):
            raise ValueError("state record generation path is not canonical")
        if create_generation:
            _publish_staged_generation_record(
                state_fd,
                base=path.parent.parent,
                generation=generation,
                record_name=path.name,
                data=data,
            )
        else:
            with _open_generation_directory(
                state_fd,
                path.parent.parent,
                generation,
            ) as generation_fd:
                _exclusive_publish_at(generation_fd, path.name, data)
    else:
        _exclusive_publish(path, data)
    return document, hashlib.sha256(data).hexdigest()


def _read_record(
    path: Path,
    *,
    payload_type: str,
    payload_keys: set[str],
    directory_fd: int | None = None,
) -> tuple[dict[str, object], str]:
    document = (
        _strict_load(path)
        if directory_fd is None
        else _strict_load_at(
            directory_fd,
            path.name,
            path,
        )
    )
    _require_exact_keys(document, DOCUMENT_KEYS, payload_type)
    if document["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported {payload_type} schema_version")
    if document["payload_type"] != payload_type:
        raise ValueError(f"unexpected payload type at {path}")
    payload = document["payload"]
    if not isinstance(payload, dict):
        raise ValueError(f"{payload_type} payload must be an object")
    _require_exact_keys(payload, payload_keys, f"{payload_type} payload")
    claimed = _require_sha256(
        document["payload_sha256"],
        f"{payload_type} payload hash",
    )
    if sha256_json(payload) != claimed:
        raise ValueError(f"{payload_type} payload hash mismatch")
    return document, hashlib.sha256(canonical_json_bytes(document)).hexdigest()


def _claim_from_document(
    path: Path,
    generation: int,
    document: dict[str, object],
    digest: str,
) -> JobClaim:
    return JobClaim(
        path=path,
        generation=generation,
        sha256=digest,
        document=document,
    )


def _generation_numbers(state_fd: int) -> list[int]:
    try:
        scan_fd = os.open(
            ".",
            _directory_open_flags(),
            dir_fd=state_fd,
        )
    except OSError as exc:
        raise ValueError("cannot enumerate state generations") from exc
    numbers: list[int] = []
    try:
        with os.scandir(scan_fd) as iterator:
            for entry in iterator:
                match = _GENERATION_RE.fullmatch(entry.name)
                if match is None:
                    continue
                try:
                    mode = entry.stat(follow_symlinks=False).st_mode
                except OSError as exc:
                    raise ValueError(
                        "cannot inspect claim generation"
                    ) from exc
                if not stat.S_ISDIR(mode):
                    raise ValueError(
                        "claim generation is symbolic or not a real directory"
                    )
                numbers.append(int(match.group(1)))
    finally:
        os.close(scan_fd)
    numbers.sort()
    if numbers and numbers != list(range(1, numbers[-1] + 1)):
        raise ValueError("claim generations must be contiguous")
    return numbers


def _validate_claim_payload(
    claim: JobClaim,
    *,
    map_sha256: str,
    row_sha256: str,
    array_index: int,
) -> None:
    payload = claim.document["payload"]
    if not isinstance(payload, dict):
        raise ValueError("claim payload must be an object")
    if (
        payload["job_map_sha256"] != map_sha256
        or payload["row_sha256"] != row_sha256
        or payload["array_index"] != array_index
        or payload["generation"] != claim.generation
    ):
        raise ValueError("claim does not match its sealed job-map row")
    if claim.generation == 1:
        if (
            payload["recovered_from_claim_sha256"] is not None
            or payload["recovery_record_sha256"] is not None
        ):
            raise ValueError("first claim cannot contain a recovery link")
    else:
        _require_sha256(
            payload["recovered_from_claim_sha256"],
            "recovered_from_claim_sha256",
        )
        _require_sha256(
            payload["recovery_record_sha256"],
            "recovery_record_sha256",
        )


def _load_attempt(
    claim: JobClaim,
    *,
    generation_fd: int,
    map_sha256: str,
    row_sha256: str,
    array_index: int,
) -> tuple[dict[str, object], str] | None:
    if not _entry_lexists_at(
        generation_fd,
        claim.attempt_path.name,
        claim.attempt_path,
    ):
        return None
    if claim.generation == 1:
        raise ValueError("first claim cannot contain a recovery attempt")
    document, digest = _read_record(
        claim.attempt_path,
        payload_type=ATTEMPT_TYPE,
        payload_keys=ATTEMPT_PAYLOAD_KEYS,
        directory_fd=generation_fd,
    )
    attempt = document["payload"]
    if not isinstance(attempt, dict):
        raise ValueError("attempt payload must be an object")
    if (
        attempt["job_map_sha256"] != map_sha256
        or attempt["row_sha256"] != row_sha256
        or attempt["array_index"] != array_index
        or attempt["generation"] != claim.generation
        or attempt["claim_sha256"] != claim.sha256
    ):
        raise ValueError(
            "attempt does not match its recovered claim generation"
        )
    scheduler_job_id = attempt["scheduler_job_id"]
    if scheduler_job_id is not None:
        scheduler_match = (
            _SLURM_ARRAY_JOB_RE.fullmatch(scheduler_job_id)
            if isinstance(scheduler_job_id, str)
            else None
        )
        if (
            scheduler_match is None
            or int(scheduler_match.group("array_task_id")) != array_index
        ):
            raise ValueError(
                "attempt scheduler job does not match its array row"
            )
    return document, digest


def _load_recovery_record(
    previous: JobClaim,
    *,
    generation_fd: int,
    state_fd: int,
    row: Mapping[str, object],
    base: Path,
    map_sha256: str,
    row_sha256: str,
) -> tuple[dict[str, object], str]:
    attempt = _load_attempt(
        previous,
        generation_fd=generation_fd,
        map_sha256=map_sha256,
        row_sha256=row_sha256,
        array_index=int(row["array_index"]),
    )
    document, digest = _read_record(
        previous.recovery_path,
        payload_type=RECOVERY_TYPE,
        payload_keys=RECOVERY_PAYLOAD_KEYS,
        directory_fd=generation_fd,
    )
    recovery = document["payload"]
    if not isinstance(recovery, dict):
        raise ValueError("recovery payload must be an object")
    if recovery["restart_mode"] != "fresh":
        raise ValueError("recovery restart_mode must be fresh")
    _validate_quarantine(
        recovery["quarantine"],
        row=row,
        base=base,
        generation=previous.generation,
        state_fd=state_fd,
    )
    if (
        recovery["claim_sha256"] != previous.sha256
        or recovery["next_generation"] != previous.generation + 1
        or recovery["attempt_record_sha256"]
        != (attempt[1] if attempt is not None else None)
    ):
        raise ValueError("recovery record does not match its previous claim")
    if previous.generation > 1 and attempt is None:
        raise ValueError("recovered claim was superseded without an attempt")
    _require_sha256(
        recovery["recovery_audit_sha256"],
        "recovery_audit_sha256",
    )
    audit_type = recovery["recovery_audit_type"]
    if audit_type not in {
        TEST_ONLY_RECOVERY_AUDIT_TYPE,
        SLURM_RECOVERY_AUDIT_TYPE,
    }:
        raise ValueError("recovery audit type is not supported")
    if audit_type == SLURM_RECOVERY_AUDIT_TYPE:
        _, audit_sha256 = _read_record(
            previous.slurm_recovery_audit_path,
            payload_type=SLURM_RECOVERY_AUDIT_TYPE,
            payload_keys=_SLURM_AUDIT_PAYLOAD_KEYS,
            directory_fd=generation_fd,
        )
        if audit_sha256 != recovery["recovery_audit_sha256"]:
            raise ValueError(
                "typed SLURM recovery audit hash does not match recovery"
            )
    return document, digest


def _load_claim_chain(
    row: Mapping[str, object],
    *,
    state_fd: int,
    map_sha256: str,
    row_sha256: str,
    allow_pending_recovery: bool = False,
) -> list[JobClaim]:
    base = _state_dir(row)
    claims: list[JobClaim] = []
    with contextlib.ExitStack() as stack:
        generation_fds = {
            generation: stack.enter_context(
                _open_generation_directory(
                    state_fd,
                    base,
                    generation,
                )
            )
            for generation in _generation_numbers(state_fd)
        }
        for generation, generation_fd in generation_fds.items():
            path = base / _generation_name(generation) / "claim.json"
            document, digest = _read_record(
                path,
                payload_type=CLAIM_TYPE,
                payload_keys=CLAIM_PAYLOAD_KEYS,
                directory_fd=generation_fd,
            )
            claim = _claim_from_document(
                path,
                generation,
                document,
                digest,
            )
            _validate_claim_payload(
                claim,
                map_sha256=map_sha256,
                row_sha256=row_sha256,
                array_index=int(row["array_index"]),
            )
            if claims:
                previous = claims[-1]
                _, recovery_sha256 = _load_recovery_record(
                    previous,
                    generation_fd=generation_fds[previous.generation],
                    state_fd=state_fd,
                    row=row,
                    base=base,
                    map_sha256=map_sha256,
                    row_sha256=row_sha256,
                )
                current_payload = claim.document["payload"]
                if not isinstance(current_payload, dict):
                    raise ValueError("invalid claim recovery chain")
                if (
                    current_payload["recovered_from_claim_sha256"]
                    != previous.sha256
                    or current_payload["recovery_record_sha256"]
                    != recovery_sha256
                ):
                    raise ValueError("claim recovery chain hash mismatch")
            claims.append(claim)
        if claims:
            current = claims[-1]
            current_fd = generation_fds[current.generation]
            if _entry_lexists_at(
                current_fd,
                current.recovery_path.name,
                current.recovery_path,
            ):
                _load_recovery_record(
                    current,
                    generation_fd=current_fd,
                    state_fd=state_fd,
                    row=row,
                    base=base,
                    map_sha256=map_sha256,
                    row_sha256=row_sha256,
                )
                if not allow_pending_recovery:
                    raise ValueError(
                        "audited recovery is missing its next claim generation"
                    )
            _load_attempt(
                current,
                generation_fd=current_fd,
                map_sha256=map_sha256,
                row_sha256=row_sha256,
                array_index=int(row["array_index"]),
            )
        return claims


def _require_cost_execution_row(row: Mapping[str, object]) -> None:
    if (
        row.get("array_index") != 0
        or row.get("stage") != "stage-1-cost-benchmark"
        or row.get("role") != "cost-benchmark"
        or row.get("config_id") != "stage1_cost_v1"
        or row.get("subject") != 1
        or row.get("seed") != 20260720
        or row.get("partition") != "i64m1tga40u"
        or row.get("gres") != "gpu:a40:1"
    ):
        raise ValueError(
            "scheduler execution authority is restricted to the sealed "
            "Stage 1 cost row"
        )


def _cost_execution_identity(
    claim: JobClaim,
    document: Mapping[str, object],
    digest: str,
) -> dict[str, object]:
    payload = document["payload"]
    if not isinstance(payload, Mapping):
        raise ValueError("cost execution authority payload is invalid")
    return {
        "array_index": payload["array_index"],
        "attempt_payload_sha256": payload["attempt_payload_sha256"],
        "attempt_record_sha256": payload["attempt_record_sha256"],
        "claim_sha256": payload["claim_sha256"],
        "generation": payload["generation"],
        "job_map_sha256": payload["job_map_sha256"],
        "path": str(claim.cost_execution_path),
        "payload_sha256": document["payload_sha256"],
        "row_sha256": payload["row_sha256"],
        "scheduler_job_id": payload["scheduler_job_id"],
        "sha256": digest,
    }


def _load_cost_execution_at(
    claim: JobClaim,
    *,
    generation_fd: int,
    map_sha256: str,
    row_sha256: str,
    array_index: int,
) -> dict[str, object]:
    document, digest = _read_record(
        claim.cost_execution_path,
        payload_type=COST_EXECUTION_TYPE,
        payload_keys=COST_EXECUTION_PAYLOAD_KEYS,
        directory_fd=generation_fd,
    )
    execution = document["payload"]
    if not isinstance(execution, dict):
        raise ValueError("cost execution authority payload must be an object")
    scheduler_job_id = execution["scheduler_job_id"]
    scheduler_match = (
        _SLURM_ARRAY_JOB_RE.fullmatch(scheduler_job_id)
        if isinstance(scheduler_job_id, str)
        else None
    )
    if (
        execution["job_map_sha256"] != map_sha256
        or execution["row_sha256"] != row_sha256
        or execution["array_index"] != array_index
        or execution["generation"] != claim.generation
        or execution["claim_sha256"] != claim.sha256
        or scheduler_match is None
        or int(scheduler_match.group("array_task_id")) != array_index
    ):
        raise ValueError(
            "cost execution authority does not match its current claim row"
        )
    attempt = _load_attempt(
        claim,
        generation_fd=generation_fd,
        map_sha256=map_sha256,
        row_sha256=row_sha256,
        array_index=array_index,
    )
    if claim.generation == 1:
        expected_attempt_record_sha256 = None
        expected_attempt_payload_sha256 = None
    else:
        if attempt is None:
            raise ValueError(
                "recovered cost execution lacks its scheduler attempt"
            )
        attempt_payload = attempt[0]["payload"]
        if (
            not isinstance(attempt_payload, dict)
            or attempt_payload["scheduler_job_id"] != scheduler_job_id
        ):
            raise ValueError(
                "cost execution scheduler differs from its recovered attempt"
            )
        expected_attempt_record_sha256 = attempt[1]
        expected_attempt_payload_sha256 = attempt[0]["payload_sha256"]
    if (
        execution["attempt_record_sha256"]
        != expected_attempt_record_sha256
        or execution["attempt_payload_sha256"]
        != expected_attempt_payload_sha256
    ):
        raise ValueError(
            "cost execution authority attempt identity mismatch"
        )
    return _cost_execution_identity(claim, document, digest)


def load_cost_execution_authority(
    payload: Mapping[str, object],
    row: Mapping[str, object],
    *,
    expected_generation: int,
    expected_claim_sha256: str,
) -> dict[str, object]:
    """Reload the authority-published scheduler identity for a cost run."""

    _, checked_row, map_sha256, row_sha256 = _row_context(payload, row)
    _require_cost_execution_row(checked_row)
    if type(expected_generation) is not int or expected_generation <= 0:
        raise ValueError("expected cost execution generation is invalid")
    checked_claim_sha256 = _require_sha256(
        expected_claim_sha256,
        "expected cost execution claim SHA-256",
    )
    base = _state_dir(checked_row)
    with _transition_lock(base) as state_fd:
        claims = _load_claim_chain(
            checked_row,
            state_fd=state_fd,
            map_sha256=map_sha256,
            row_sha256=row_sha256,
        )
        if not claims:
            raise ValueError("cost execution authority has no current claim")
        current = claims[-1]
        if (
            current.generation != expected_generation
            or current.sha256 != checked_claim_sha256
        ):
            raise ValueError(
                "cost execution authority current claim identity changed"
            )
        with _open_generation_directory(
            state_fd,
            base,
            current.generation,
        ) as generation_fd:
            if not _entry_lexists_at(
                generation_fd,
                current.cost_execution_path.name,
                current.cost_execution_path,
            ):
                raise ValueError(
                    "cost execution authority record is missing"
                )
            return _load_cost_execution_at(
                current,
                generation_fd=generation_fd,
                map_sha256=map_sha256,
                row_sha256=row_sha256,
                array_index=int(checked_row["array_index"]),
            )


def publish_cost_execution_authority(
    payload: Mapping[str, object],
    row: Mapping[str, object],
    *,
    claim: JobClaim,
    scheduler_job_id: str,
) -> dict[str, object]:
    """Exclusively publish the scheduler identity before the cost child."""

    _, checked_row, map_sha256, row_sha256 = _row_context(payload, row)
    _require_cost_execution_row(checked_row)
    scheduler_match = (
        _SLURM_ARRAY_JOB_RE.fullmatch(scheduler_job_id)
        if isinstance(scheduler_job_id, str)
        else None
    )
    if (
        scheduler_match is None
        or int(scheduler_match.group("array_task_id"))
        != int(checked_row["array_index"])
    ):
        raise ValueError(
            "cost execution scheduler job does not match its array row"
        )
    base = _state_dir(checked_row)
    with _transition_lock(base) as state_fd:
        claims = _load_claim_chain(
            checked_row,
            state_fd=state_fd,
            map_sha256=map_sha256,
            row_sha256=row_sha256,
        )
        if not claims:
            raise ValueError("cost execution authority has no current claim")
        current = claims[-1]
        if (
            current.path != claim.path
            or current.generation != claim.generation
            or current.sha256 != claim.sha256
        ):
            raise ValueError(
                "cost execution authority claim changed before publication"
            )
        with _open_generation_directory(
            state_fd,
            base,
            current.generation,
        ) as generation_fd:
            if _entry_lexists_at(
                generation_fd,
                current.cost_execution_path.name,
                current.cost_execution_path,
            ):
                raise RuntimeError(
                    "cost execution authority already exists; rewrite refused"
                )
            attempt = _load_attempt(
                current,
                generation_fd=generation_fd,
                map_sha256=map_sha256,
                row_sha256=row_sha256,
                array_index=int(checked_row["array_index"]),
            )
            if current.generation == 1:
                attempt_record_sha256 = None
                attempt_payload_sha256 = None
            else:
                if attempt is None:
                    raise ValueError(
                        "recovered cost execution lacks an attempt record"
                    )
                attempt_payload = attempt[0]["payload"]
                if (
                    not isinstance(attempt_payload, dict)
                    or attempt_payload["scheduler_job_id"]
                    != scheduler_job_id
                ):
                    raise ValueError(
                        "cost execution scheduler differs from its attempt"
                    )
                attempt_record_sha256 = attempt[1]
                attempt_payload_sha256 = attempt[0]["payload_sha256"]
            execution_payload: dict[str, object] = {
                "job_map_sha256": map_sha256,
                "row_sha256": row_sha256,
                "array_index": checked_row["array_index"],
                "generation": current.generation,
                "claim_sha256": current.sha256,
                "scheduler_job_id": scheduler_job_id,
                "attempt_record_sha256": attempt_record_sha256,
                "attempt_payload_sha256": attempt_payload_sha256,
            }
            document, digest = _create_record(
                current.cost_execution_path,
                COST_EXECUTION_TYPE,
                execution_payload,
                directory_fd=generation_fd,
            )
            return _cost_execution_identity(current, document, digest)


def _validate_output_hashes(
    row: Mapping[str, object],
    output_hashes: Mapping[str, str],
) -> dict[str, str]:
    if not isinstance(output_hashes, Mapping):
        raise ValueError("output_hashes must be an object")
    schema = row["expected_completion_schema"]
    if not isinstance(schema, dict):
        raise ValueError("invalid expected completion schema")
    names = schema["required_output_hashes"]
    if set(output_hashes) != set(names):  # type: ignore[arg-type]
        raise ValueError(
            "completion output hashes do not match the sealed schema"
        )
    return {
        str(name): _require_sha256(output_hashes[name], str(name))
        for name in names  # type: ignore[union-attr]
    }


def _deep_freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _deep_freeze_json(child)
                for key, child in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze_json(child) for child in value)
    return value


def _deep_thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _deep_thaw_json(child) for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_deep_thaw_json(child) for child in value]
    return value


def _make_job_completion(
    *,
    payload: Mapping[str, object],
    row: Mapping[str, object],
    path: Path,
    digest: str,
    document: Mapping[str, object],
) -> JobCompletion:
    checked_payload, checked_row, _, _ = _row_context(payload, row)
    completion_payload = document.get("payload")
    if not isinstance(completion_payload, Mapping):
        raise ValueError("completion payload must be an object")
    outputs = completion_payload.get("output_hashes")
    if not isinstance(outputs, Mapping):
        raise ValueError("completion output_hashes must be an object")
    validated_outputs = _validate_output_hashes(checked_row, outputs)
    frozen_document = _deep_freeze_json(document)
    if not isinstance(frozen_document, Mapping):
        raise AssertionError("frozen completion document is not a mapping")
    frozen_outputs = _deep_freeze_json(validated_outputs)
    if not isinstance(frozen_outputs, Mapping):
        raise AssertionError("frozen completion outputs are not a mapping")
    return JobCompletion(
        path=path,
        sha256=digest,
        document=frozen_document,
        output_hashes=frozen_outputs,
        _job_map_snapshot=canonical_json_bytes(checked_payload),
        _array_index=int(checked_row["array_index"]),
    )


def _load_completion(
    payload: Mapping[str, object],
    row: Mapping[str, object],
    *,
    state_fd: int | None,
) -> JobCompletion:
    checked_payload, checked_row, map_sha256, row_sha256 = _row_context(
        payload,
        row,
    )
    path = Path(str(checked_row["completion_path"]))
    schema = checked_row["expected_completion_schema"]
    if not isinstance(schema, dict):
        raise ValueError("invalid expected completion schema")
    document, digest = _read_record(
        path,
        payload_type=str(schema["payload_type"]),
        payload_keys=COMPLETION_PAYLOAD_KEYS,
    )
    completion_payload = document["payload"]
    if not isinstance(completion_payload, dict):
        raise ValueError("completion payload must be an object")
    claims = (
        []
        if state_fd is None
        else _load_claim_chain(
            checked_row,
            state_fd=state_fd,
            map_sha256=map_sha256,
            row_sha256=row_sha256,
        )
    )
    if not claims:
        raise ValueError("completion has no immutable claim")
    current = claims[-1]
    if (
        completion_payload["job_map_sha256"] != map_sha256
        or completion_payload["row_sha256"] != row_sha256
        or completion_payload["array_index"] != checked_row["array_index"]
        or completion_payload["generation"] != current.generation
        or completion_payload["claim_sha256"] != current.sha256
    ):
        raise ValueError(
            "completion does not match its sealed row/current claim"
        )
    outputs = completion_payload["output_hashes"]
    if not isinstance(outputs, dict):
        raise ValueError("completion output_hashes must be an object")
    _validate_output_hashes(checked_row, outputs)
    return _make_job_completion(
        payload=checked_payload,
        row=checked_row,
        path=path,
        digest=digest,
        document=document,
    )


def _load_job_completion_current(
    payload: Mapping[str, object],
    row: Mapping[str, object],
) -> JobCompletion | None:
    _, checked_row, _, _ = _row_context(payload, row)
    path = Path(str(checked_row["completion_path"]))
    if not _path_lexists(path):
        return None
    with _open_state_directory(
        _state_dir(checked_row),
        create=False,
    ) as state_fd:
        return _load_completion(
            payload,
            checked_row,
            state_fd=state_fd,
        )


def load_job_completion(
    payload: Mapping[str, object],
    row: Mapping[str, object],
) -> JobCompletion | None:
    """Return a fully validated completion, or ``None`` only for absence.

    A present document is accepted only after the map, selected row, immutable
    claim/recovery chain, current generation, completion schema, and output
    hashes have all been validated.
    """

    return _load_job_completion_current(payload, row)


def _revalidate_job_completion(completion: JobCompletion) -> None:
    try:
        snapshot = json.loads(completion._job_map_snapshot)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "job completion map snapshot is invalid JSON"
        ) from exc
    if (
        not isinstance(snapshot, dict)
        or canonical_json_bytes(snapshot) != completion._job_map_snapshot
    ):
        raise ValueError("job completion map snapshot is not canonical")
    payload = validate_job_map(snapshot)
    rows = payload["rows"]
    if (
        not isinstance(rows, list)
        or completion._array_index < 0
        or completion._array_index >= len(rows)
    ):
        raise ValueError("job completion row snapshot is invalid")
    row = rows[completion._array_index]
    if not isinstance(row, Mapping):
        raise ValueError("job completion row snapshot is invalid")
    current = _load_job_completion_current(payload, row)
    if current is None:
        raise ValueError("job completion is now absent")
    expected_document = canonical_json_bytes(
        _deep_thaw_json(completion.document)
    )
    actual_document = canonical_json_bytes(_deep_thaw_json(current.document))
    if (
        current.path != completion.path
        or current.sha256 != completion.sha256
        or actual_document != expected_document
        or dict(current.output_hashes) != dict(completion.output_hashes)
    ):
        raise ValueError(
            "job completion snapshot differs from current validated state"
        )


def completion_output_hashes(
    payload: Mapping[str, object],
    row: Mapping[str, object],
) -> dict[str, str] | None:
    """Return sealed output hashes, or ``None`` only when completion is absent."""

    completion = load_job_completion(payload, row)
    if completion is None:
        return None
    _, checked_row, _, _ = _row_context(payload, row)
    completion_payload = completion.document["payload"]
    if not isinstance(completion_payload, Mapping):
        raise ValueError("completion payload must be an object")
    output_hashes = completion_payload["output_hashes"]
    if not isinstance(output_hashes, Mapping):
        raise ValueError("completion output_hashes must be an object")
    return dict(_validate_output_hashes(checked_row, output_hashes))


def completion_is_valid(
    payload: Mapping[str, object],
    row: Mapping[str, object],
) -> bool:
    """Return false only for absence; reject any present invalid completion."""

    return completion_output_hashes(payload, row) is not None


def should_submit_row(
    payload: Mapping[str, object],
    row: Mapping[str, object],
) -> bool:
    """Return whether a row lacks a valid completion and an active first claim."""

    _, checked_row, map_sha256, row_sha256 = _row_context(
        payload,
        row,
    )
    base = _state_dir(checked_row)
    with _open_state_directory(base, create=False) as state_fd:
        if _path_lexists(Path(str(checked_row["completion_path"]))):
            _load_completion(
                payload,
                checked_row,
                state_fd=state_fd,
            )
            return False
        claims = (
            []
            if state_fd is None
            else _load_claim_chain(
                checked_row,
                state_fd=state_fd,
                map_sha256=map_sha256,
                row_sha256=row_sha256,
            )
        )
        if not claims:
            should_submit = True
        else:
            current = claims[-1]
            if current.generation == 1:
                return False
            with _open_generation_directory(
                state_fd,
                base,
                current.generation,
            ) as generation_fd:
                should_submit = (
                    _load_attempt(
                        current,
                        generation_fd=generation_fd,
                        map_sha256=map_sha256,
                        row_sha256=row_sha256,
                        array_index=int(checked_row["array_index"]),
                    )
                    is None
                )
    if should_submit and _path_lexists(_output_dir(checked_row)):
        raise RuntimeError(
            "unaudited output exists; audited recovery is required before submission"
        )
    return should_submit


def claim_job_row(
    payload: Mapping[str, object],
    row: Mapping[str, object],
) -> JobClaim:
    """Create the first immutable claim, refusing implicit stale retries."""

    _, checked_row, map_sha256, row_sha256 = _row_context(
        payload,
        row,
    )
    base = _state_dir(checked_row)
    with _transition_lock(base) as state_fd:
        if _path_lexists(Path(str(checked_row["completion_path"]))):
            _load_completion(
                payload,
                checked_row,
                state_fd=state_fd,
            )
            raise RuntimeError("job row already has a valid completion")
        output_dir = _output_dir(checked_row)
        if _path_lexists(output_dir):
            raise RuntimeError(
                "unaudited output exists; audited recovery is required"
            )
        claims = _load_claim_chain(
            checked_row,
            state_fd=state_fd,
            map_sha256=map_sha256,
            row_sha256=row_sha256,
        )
        if claims:
            raise RuntimeError(
                "job row has an active/stale claim; audited recovery is required"
            )
        claim_payload: dict[str, object] = {
            "job_map_sha256": map_sha256,
            "row_sha256": row_sha256,
            "array_index": checked_row["array_index"],
            "generation": 1,
            "recovered_from_claim_sha256": None,
            "recovery_record_sha256": None,
        }
        path = base / "generation-000001" / "claim.json"
        document, digest = _create_record(
            path,
            CLAIM_TYPE,
            claim_payload,
            state_fd=state_fd,
            generation=1,
            create_generation=True,
        )
        return _claim_from_document(path, 1, document, digest)


def _single_row_argv_value(
    row: Mapping[str, object],
    flag: str,
) -> str:
    argv = row["argv"]
    if not isinstance(argv, list) or any(
        not isinstance(value, str) for value in argv
    ):
        raise ValueError("job row argv must be a list of strings")
    positions = [index for index, value in enumerate(argv) if value == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise ValueError(f"job row must contain {flag} exactly once")
    return argv[positions[0] + 1]


def _verified_submission_project_root(
    row: Mapping[str, object],
) -> Path:
    raw = Path(_single_row_argv_value(row, "--project-root"))
    if not raw.is_absolute() or ".." in raw.parts:
        raise ValueError("project root must be absolute and normalized")
    try:
        resolved = raw.resolve(strict=True)
    except OSError as exc:
        raise ValueError("project root cannot be verified") from exc
    if (
        resolved != raw
        or not resolved.is_dir()
        or not (resolved / ".git").exists()
    ):
        raise ValueError("project root is not a verified repository root")
    return resolved


def _compress_array_indices(indices: Sequence[int]) -> str:
    if not indices:
        raise ValueError("SLURM array cannot be empty")
    ordered = sorted(set(indices))
    if list(indices) != ordered:
        raise ValueError("SLURM array indices must be unique and sorted")
    ranges: list[str] = []
    start = previous = ordered[0]
    for index in ordered[1:]:
        if index == previous + 1:
            previous = index
            continue
        ranges.append(f"{start}-{previous}")
        start = previous = index
    ranges.append(f"{start}-{previous}")
    return ",".join(ranges)


def _parse_array_indices(
    specification: str,
    *,
    allowed: set[int],
    required: int,
) -> list[int]:
    if not specification:
        raise ValueError("SLURM SubmitLine has an empty array specification")
    indices: list[int] = []
    for component in specification.split(","):
        parts = component.split("-")
        if len(parts) == 1:
            start_text = end_text = parts[0]
        elif len(parts) == 2:
            start_text, end_text = parts
        else:
            raise ValueError("SLURM SubmitLine array is not canonical")
        for value in (start_text, end_text):
            if (
                not value.isascii()
                or not value.isdecimal()
                or str(int(value)) != value
            ):
                raise ValueError("SLURM SubmitLine array is not canonical")
        start, end = int(start_text), int(end_text)
        if end < start:
            raise ValueError("SLURM SubmitLine array range is reversed")
        indices.extend(range(start, end + 1))
    if _compress_array_indices(indices) != specification:
        raise ValueError("SLURM SubmitLine array is not canonical")
    if required not in indices or any(
        index not in allowed for index in indices
    ):
        raise ValueError(
            "SLURM SubmitLine array is outside the sealed job map"
        )
    return indices


def _expected_submission_command(
    payload: Mapping[str, object],
    row: Mapping[str, object],
    *,
    job_map_path: Path,
    indices: Sequence[int],
) -> list[str]:
    project_root = _verified_submission_project_root(row)
    try:
        sealed_map = Path(job_map_path).resolve(strict=True)
    except OSError as exc:
        raise ValueError("job-map path cannot be verified") from exc
    if sealed_map != Path(job_map_path) or not sealed_map.is_file():
        raise ValueError("job-map path must be absolute and canonical")
    bounds = payload["array_bounds"]
    if (
        not isinstance(bounds, list)
        or len(bounds) != 2
        or any(type(value) is not int for value in bounds)
    ):
        raise ValueError("job map has invalid immutable array bounds")
    slurm_script = (
        project_root / "experiments/samga_brain_rw/slurm/pilot_array.slurm"
    )
    try:
        verified_script = slurm_script.resolve(strict=True)
    except OSError as exc:
        raise ValueError("pilot SLURM script cannot be verified") from exc
    if verified_script != slurm_script or not verified_script.is_file():
        raise ValueError("pilot SLURM script is not canonical")
    stdout_path = project_root / str(row["stdout_path"])
    stderr_path = project_root / str(row["stderr_path"])
    if stdout_path.parent != stderr_path.parent:
        raise ValueError("sealed stdout and stderr directories differ")
    for value in (str(project_root), str(sealed_map)):
        if "," in value or "\n" in value:
            raise ValueError("sealed path cannot be encoded in SLURM --export")
    return [
        "sbatch",
        "--parsable",
        f"--chdir={project_root}",
        f"--partition={row['partition']}",
        f"--gres={row['gres']}",
        f"--cpus-per-task={row['cpus']}",
        f"--mem={row['memory']}",
        f"--time={row['time']}",
        f"--array={_compress_array_indices(indices)}",
        f"--output={stdout_path}",
        f"--error={stderr_path}",
        (
            "--export=ALL,"
            f"PROJECT_ROOT={project_root},"
            f"JOB_MAP={sealed_map},"
            f"JOB_MAP_SHA256={payload['payload_sha256']},"
            f"JOB_MAP_ARRAY_MIN={bounds[0]},"
            f"JOB_MAP_ARRAY_MAX={bounds[1]}"
        ),
        str(verified_script),
    ]


def _run_scheduler_query(
    runner: Any,
    command: list[str],
) -> str:
    result = runner(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    stdout = getattr(result, "stdout", None)
    if not isinstance(stdout, str):
        raise ValueError("scheduler command did not return text stdout")
    return stdout


def _parse_sacct_record(stdout: str) -> dict[str, str]:
    lines = stdout.splitlines()
    if len(lines) != 1 or not lines[0]:
        raise ValueError(
            "sacct must return exactly one unambiguous allocation record"
        )
    values = lines[0].split("|")
    if len(values) != len(SLURM_RECOVERY_SACCT_FIELDS):
        raise ValueError("sacct record does not match the fixed field schema")
    return dict(zip(SLURM_RECOVERY_SACCT_FIELDS, values, strict=True))


def _slurm_time_ns(value: str, label: str) -> int:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"sacct {label} is not an ISO timestamp") from exc
    if parsed.tzinfo is not None or parsed.microsecond != 0:
        raise ValueError(f"sacct {label} must use local whole seconds")
    return int(parsed.timestamp()) * 1_000_000_000


def _parse_tres(value: str, label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in value.split(","):
        key, separator, selected = item.partition("=")
        if not separator or not key or not selected or key in result:
            raise ValueError(f"sacct {label} is not canonical")
        result[key] = selected
    return result


def _validate_sacct_record(
    record: Mapping[str, str],
    *,
    payload: Mapping[str, object],
    row: Mapping[str, object],
    job_map_path: Path,
    failed_slurm_job: str,
    binding_mtime_ns: int,
) -> tuple[list[str], Path, str, int]:
    match = _SLURM_ARRAY_JOB_RE.fullmatch(failed_slurm_job)
    if match is None:
        raise ValueError("failed SLURM job must use <array-job>_<task-id>")
    array_task_id = int(match.group("array_task_id"))
    if array_task_id != int(row["array_index"]):
        raise ValueError("failed SLURM task does not match the sealed row")
    if _SLURM_JOB_ID_RE.fullmatch(record["JobIDRaw"]) is None:
        raise ValueError("sacct JobIDRaw must be a canonical positive job ID")
    username = pwd.getpwuid(os.getuid()).pw_name
    expected = {
        "JobID": failed_slurm_job,
        "UID": str(os.getuid()),
        "User": username,
        "Partition": str(row["partition"]),
    }
    for field, value in expected.items():
        if record[field] != value:
            raise ValueError(f"sacct {field} does not match the sealed row")
    failed_state = record["State"]
    cancelled_by_current_uid = f"CANCELLED by {os.getuid()}"
    if (
        failed_state not in _FAILED_SLURM_STATES
        and failed_state != cancelled_by_current_uid
    ):
        raise ValueError(
            f"sacct state is not a failed terminal state: {failed_state}"
        )
    start_ns = _slurm_time_ns(record["Start"], "Start")
    end_ns = _slurm_time_ns(record["End"], "End")
    submit_ns = _slurm_time_ns(record["Submit"], "Submit")
    eligible_ns = _slurm_time_ns(record["Eligible"], "Eligible")
    if not submit_ns <= eligible_ns <= start_ns <= end_ns:
        raise ValueError("sacct timestamps are not ordered")
    if not start_ns <= binding_mtime_ns <= end_ns + 999_999_999:
        raise ValueError(
            "scheduler binding record mtime is outside the SLURM run"
        )
    if record["ElapsedRaw"] != str((end_ns - start_ns) // 1_000_000_000):
        raise ValueError("sacct ElapsedRaw does not match Start and End")
    project_root = _verified_submission_project_root(row)
    if record["WorkDir"] != str(project_root):
        raise ValueError("sacct WorkDir does not match the project root")
    expected_minutes = (_time_seconds(row["time"]) + 59) // 60
    if record["TimelimitRaw"] != str(expected_minutes):
        raise ValueError("sacct TimelimitRaw does not match the sealed row")
    for field in ("AllocTRES", "ReqTRES"):
        tres = _parse_tres(record[field], field)
        required = {
            "cpu": str(row["cpus"]),
            "mem": str(row["memory"]),
            "gres/gpu:a40": "1",
        }
        for key, value in required.items():
            if tres.get(key) != value:
                raise ValueError(
                    f"sacct {field} does not match sealed resources"
                )
    submit_argv = shlex.split(record["SubmitLine"])
    array_tokens = [
        token for token in submit_argv if token.startswith("--array=")
    ]
    if len(array_tokens) != 1:
        raise ValueError("sacct SubmitLine has no unique array argument")
    rows = payload["rows"]
    if not isinstance(rows, list):
        raise ValueError("job map rows must be a list")
    allowed = {
        int(selected["array_index"])
        for selected in rows
        if isinstance(selected, Mapping)
    }
    array_indices = _parse_array_indices(
        array_tokens[0].removeprefix("--array="),
        allowed=allowed,
        required=int(row["array_index"]),
    )
    expected_argv = _expected_submission_command(
        payload,
        row,
        job_map_path=job_map_path,
        indices=array_indices,
    )
    if submit_argv != expected_argv:
        raise ValueError(
            "sacct SubmitLine does not match the sealed submission"
        )
    cluster = _require_nonempty_string(record["Cluster"], "sacct Cluster")
    return array_indices, project_root, cluster, array_task_id


def _concrete_log_identity(
    row: Mapping[str, object],
    *,
    project_root: Path,
    field: str,
    array_job_id: str,
    array_task_id: int,
) -> dict[str, object]:
    pattern = str(row[field])
    if pattern.count("%A") != 1 or pattern.count("%a") != 1:
        raise ValueError("sealed SLURM log pattern is not canonical")
    relative = pattern.replace("%A", array_job_id).replace(
        "%a", str(array_task_id)
    )
    path = project_root / relative
    data = _read_regular_file_path_nofollow(path)
    return {
        "path": str(path),
        "byte_count": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _claim_mtime_ns(
    claim: JobClaim,
    *,
    state_fd: int,
) -> int:
    with _open_generation_directory(
        state_fd,
        claim.path.parent.parent,
        claim.generation,
    ) as generation_fd:
        return _state_record_mtime_ns(
            generation_fd,
            claim.path.name,
            claim.path,
        )


def _state_record_mtime_ns(
    generation_fd: int,
    name: str,
    path: Path,
) -> int:
    try:
        identity = os.stat(
            name,
            dir_fd=generation_fd,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise ValueError(
            f"cannot inspect sealed state record: {path}"
        ) from exc
    if not stat.S_ISREG(identity.st_mode):
        raise ValueError(f"state record is not a sealed regular file: {path}")
    return identity.st_mtime_ns


def _collect_slurm_recovery_audit(
    payload: Mapping[str, object],
    row: Mapping[str, object],
    claim: JobClaim,
    *,
    state_fd: int,
    job_map_path: Path,
    failed_slurm_job: str,
    runner: Any,
) -> dict[str, object]:
    match = _SLURM_ARRAY_JOB_RE.fullmatch(failed_slurm_job)
    if match is None:
        raise ValueError("failed SLURM job must use <array-job>_<task-id>")
    claim_mtime_ns = _claim_mtime_ns(claim, state_fd=state_fd)
    attempt_identity: dict[str, object] | None = None
    binding_mtime_ns = claim_mtime_ns
    if claim.generation > 1:
        claim_payload = claim.document["payload"]
        if not isinstance(claim_payload, dict):
            raise ValueError("claim payload must be an object")
        with _open_generation_directory(
            state_fd,
            claim.path.parent.parent,
            claim.generation,
        ) as generation_fd:
            attempt = _load_attempt(
                claim,
                generation_fd=generation_fd,
                map_sha256=str(claim_payload["job_map_sha256"]),
                row_sha256=str(claim_payload["row_sha256"]),
                array_index=int(claim_payload["array_index"]),
            )
            if attempt is None:
                raise RuntimeError(
                    "recovered claim has no scheduler binding attempt"
                )
            attempt_payload = attempt[0]["payload"]
            if (
                not isinstance(attempt_payload, dict)
                or attempt_payload["scheduler_job_id"] != failed_slurm_job
            ):
                raise ValueError(
                    "failed SLURM job does not match the consumed attempt"
                )
            binding_mtime_ns = _state_record_mtime_ns(
                generation_fd,
                claim.attempt_path.name,
                claim.attempt_path,
            )
            attempt_identity = {
                "path": str(claim.attempt_path),
                "sha256": attempt[1],
                "payload_sha256": attempt[0]["payload_sha256"],
                "mtime_ns": binding_mtime_ns,
                "scheduler_job_id": failed_slurm_job,
            }
    sacct_command = [
        "sacct",
        "-X",
        "-D",
        "-j",
        failed_slurm_job,
        "--noheader",
        "--parsable2",
        f"--format={','.join(SLURM_RECOVERY_SACCT_FIELDS)}",
    ]
    first_stdout = _run_scheduler_query(runner, sacct_command)
    first_record = _parse_sacct_record(first_stdout)
    array_indices, project_root, cluster, array_task_id = (
        _validate_sacct_record(
            first_record,
            payload=payload,
            row=row,
            job_map_path=job_map_path,
            failed_slurm_job=failed_slurm_job,
            binding_mtime_ns=binding_mtime_ns,
        )
    )
    squeue_command = [
        "squeue",
        "-h",
        "-j",
        failed_slurm_job,
        "-o",
        "%i|%u|%T|%P|%N",
    ]
    squeue_stdout = _run_scheduler_query(runner, squeue_command)
    if squeue_stdout != "":
        raise RuntimeError("SLURM job still has a live squeue record")
    second_stdout = _run_scheduler_query(runner, sacct_command)
    if second_stdout != first_stdout:
        raise RuntimeError(
            "sacct terminal record changed during recovery audit"
        )
    second_record = _parse_sacct_record(second_stdout)
    if second_record != first_record:
        raise RuntimeError(
            "sacct terminal record changed during recovery audit"
        )
    map_path = Path(job_map_path)
    map_data = _read_regular_file_path_nofollow(map_path)
    map_document = _strict_json_bytes(map_data, map_path)
    if (
        map_document != payload
        or canonical_json_bytes(map_document) != map_data
    ):
        raise ValueError("job-map file differs from the loaded sealed map")
    claim_payload = claim.document["payload"]
    if not isinstance(claim_payload, dict):
        raise ValueError("claim payload must be an object")
    array_job_id = match.group("array_job_id")
    submission_argv = shlex.split(first_record["SubmitLine"])
    logs = {
        "stdout": _concrete_log_identity(
            row,
            project_root=project_root,
            field="stdout_path",
            array_job_id=array_job_id,
            array_task_id=array_task_id,
        ),
        "stderr": _concrete_log_identity(
            row,
            project_root=project_root,
            field="stderr_path",
            array_job_id=array_job_id,
            array_task_id=array_task_id,
        ),
    }
    final_squeue_stdout = _run_scheduler_query(runner, squeue_command)
    if final_squeue_stdout != "":
        raise RuntimeError(
            "SLURM job regained a live squeue record during recovery audit"
        )
    final_sacct_stdout = _run_scheduler_query(runner, sacct_command)
    if final_sacct_stdout != first_stdout:
        raise RuntimeError(
            "sacct terminal record changed before audit publication"
        )
    if _parse_sacct_record(final_sacct_stdout) != first_record:
        raise RuntimeError(
            "sacct terminal record changed before audit publication"
        )
    audit_payload: dict[str, object] = {
        "binding_mode": (
            "legacy_claim_scheduler_binding"
            if attempt_identity is None
            else "recovered_attempt_scheduler_binding"
        ),
        "job_map": {
            "path": str(map_path),
            "payload_sha256": payload["payload_sha256"],
            "file_sha256": hashlib.sha256(map_data).hexdigest(),
        },
        "row": {
            "array_index": row["array_index"],
            "row_sha256": sha256_json(row),
        },
        "claim": {
            "path": str(claim.path),
            "generation": claim.generation,
            "sha256": claim.sha256,
            "payload_sha256": claim.document["payload_sha256"],
            "mtime_ns": claim_mtime_ns,
        },
        "attempt": attempt_identity,
        "scheduler": {
            "cluster": cluster,
            "array_job_id": array_job_id,
            "array_task_id": array_task_id,
            "job_id": failed_slurm_job,
            "job_id_raw": first_record["JobIDRaw"],
            "uid": os.getuid(),
            "user": first_record["User"],
        },
        "sacct": {
            "argv": sacct_command,
            "fields": list(SLURM_RECOVERY_SACCT_FIELDS),
            "record": first_record,
            "stdout_sha256": hashlib.sha256(
                first_stdout.encode("utf-8")
            ).hexdigest(),
            "repeat_stdout_sha256": hashlib.sha256(
                final_sacct_stdout.encode("utf-8")
            ).hexdigest(),
            "sample_count": 3,
        },
        "squeue": {
            "argv": squeue_command,
            "rows": [],
            "stdout_sha256": hashlib.sha256(b"").hexdigest(),
            "repeat_stdout_sha256": hashlib.sha256(
                final_squeue_stdout.encode("utf-8")
            ).hexdigest(),
            "sample_count": 2,
        },
        "submission": {
            "argv": submission_argv,
            "argv_sha256": sha256_json(submission_argv),
            "array_indices": array_indices,
            "project_root": str(project_root),
            "slurm_script": submission_argv[-1],
        },
        "logs": logs,
    }
    return audit_payload


def _publish_or_reuse_slurm_audit(
    claim: JobClaim,
    payload: dict[str, object],
    *,
    generation_fd: int,
) -> str:
    expected_document = _record_document(
        SLURM_RECOVERY_AUDIT_TYPE,
        payload,
    )
    expected_bytes = canonical_json_bytes(expected_document)
    expected_sha256 = hashlib.sha256(expected_bytes).hexdigest()
    if _entry_lexists_at(
        generation_fd,
        claim.slurm_recovery_audit_path.name,
        claim.slurm_recovery_audit_path,
    ):
        actual = _read_regular_file_at(
            generation_fd,
            claim.slurm_recovery_audit_path.name,
            claim.slurm_recovery_audit_path,
        )
        if actual != expected_bytes:
            raise ValueError(
                "existing SLURM recovery audit differs from verified evidence"
            )
        return expected_sha256
    _, digest = _create_record(
        claim.slurm_recovery_audit_path,
        SLURM_RECOVERY_AUDIT_TYPE,
        payload,
        directory_fd=generation_fd,
    )
    return digest


def _idempotent_slurm_recovery_successor(
    claims: Sequence[JobClaim],
    *,
    checked_row: Mapping[str, object],
    state_fd: int,
    map_sha256: str,
    row_sha256: str,
    failed_slurm_job: str,
) -> JobClaim | None:
    if len(claims) < 2:
        return None
    current = claims[-1]
    previous = claims[-2]
    with _open_generation_directory(
        state_fd,
        previous.path.parent.parent,
        previous.generation,
    ) as previous_fd:
        if not _entry_lexists_at(
            previous_fd,
            previous.slurm_recovery_audit_path.name,
            previous.slurm_recovery_audit_path,
        ):
            return None
        audit_document, _ = _read_record(
            previous.slurm_recovery_audit_path,
            payload_type=SLURM_RECOVERY_AUDIT_TYPE,
            payload_keys=_SLURM_AUDIT_PAYLOAD_KEYS,
            directory_fd=previous_fd,
        )
        audit_payload = audit_document["payload"]
        if not isinstance(audit_payload, dict):
            raise ValueError("SLURM recovery audit payload must be an object")
        scheduler = audit_payload["scheduler"]
        if not isinstance(scheduler, dict):
            raise ValueError("SLURM recovery scheduler payload is invalid")
        if scheduler.get("job_id") != failed_slurm_job:
            return None
    if _path_lexists(_output_dir(checked_row)):
        raise RuntimeError(
            "idempotent recovery cannot reuse a claim with output"
        )
    with _open_generation_directory(
        state_fd,
        current.path.parent.parent,
        current.generation,
    ) as current_fd:
        attempt = _load_attempt(
            current,
            generation_fd=current_fd,
            map_sha256=map_sha256,
            row_sha256=row_sha256,
            array_index=int(checked_row["array_index"]),
        )
    if attempt is not None:
        raise RuntimeError(
            "failed SLURM job cannot be reused after the retry attempt"
        )
    return current


def recover_job_row_from_slurm(
    payload: Mapping[str, object],
    row: Mapping[str, object],
    *,
    job_map_path: Path,
    failed_slurm_job: str,
    runner: Any = subprocess.run,
) -> JobClaim:
    """Recover one failed row only after verifying live SLURM evidence."""

    _, checked_row, map_sha256, row_sha256 = _row_context(payload, row)
    map_path = Path(job_map_path)
    if not map_path.is_absolute():
        raise ValueError("job-map path must be absolute")
    base = _state_dir(checked_row)
    with _transition_lock(base) as state_fd:
        if _path_lexists(Path(str(checked_row["completion_path"]))):
            _load_completion(payload, checked_row, state_fd=state_fd)
            raise RuntimeError("cannot recover a completed job row")
        claims = _load_claim_chain(
            checked_row,
            state_fd=state_fd,
            map_sha256=map_sha256,
            row_sha256=row_sha256,
            allow_pending_recovery=True,
        )
        if not claims:
            raise RuntimeError("no stale claim exists to recover")
        idempotent = _idempotent_slurm_recovery_successor(
            claims,
            checked_row=checked_row,
            state_fd=state_fd,
            map_sha256=map_sha256,
            row_sha256=row_sha256,
            failed_slurm_job=failed_slurm_job,
        )
        if idempotent is not None:
            return idempotent
        current = claims[-1]
        with _open_generation_directory(
            state_fd,
            base,
            current.generation,
        ) as current_fd:
            current_attempt = _load_attempt(
                current,
                generation_fd=current_fd,
                map_sha256=map_sha256,
                row_sha256=row_sha256,
                array_index=int(checked_row["array_index"]),
            )
        if current.generation > 1 and current_attempt is None:
            raise RuntimeError(
                "recovered claim has not consumed its retry attempt"
            )
        audit_payload = _collect_slurm_recovery_audit(
            payload,
            checked_row,
            current,
            state_fd=state_fd,
            job_map_path=map_path,
            failed_slurm_job=failed_slurm_job,
            runner=runner,
        )
        with _open_generation_directory(
            state_fd,
            base,
            current.generation,
        ) as generation_fd:
            audit_sha256 = _publish_or_reuse_slurm_audit(
                current,
                audit_payload,
                generation_fd=generation_fd,
            )
        return _transition_to_recovered_claim_locked(
            payload,
            checked_row=checked_row,
            map_sha256=map_sha256,
            row_sha256=row_sha256,
            state_fd=state_fd,
            recovery_audit_sha256=audit_sha256,
            recovery_audit_type=SLURM_RECOVERY_AUDIT_TYPE,
        )


def _transition_to_recovered_claim_locked(
    payload: Mapping[str, object],
    *,
    checked_row: Mapping[str, object],
    map_sha256: str,
    row_sha256: str,
    state_fd: int,
    recovery_audit_sha256: str,
    recovery_audit_type: str,
) -> JobClaim:
    audit_sha256 = _require_sha256(
        recovery_audit_sha256,
        "recovery_audit_sha256",
    )
    if recovery_audit_type not in {
        TEST_ONLY_RECOVERY_AUDIT_TYPE,
        SLURM_RECOVERY_AUDIT_TYPE,
    }:
        raise ValueError("recovery audit type is not supported")
    base = _state_dir(checked_row)
    if _path_lexists(Path(str(checked_row["completion_path"]))):
        _load_completion(
            payload,
            checked_row,
            state_fd=state_fd,
        )
        raise RuntimeError("cannot recover a completed job row")
    claims = _load_claim_chain(
        checked_row,
        state_fd=state_fd,
        map_sha256=map_sha256,
        row_sha256=row_sha256,
        allow_pending_recovery=True,
    )
    if not claims:
        raise RuntimeError("no stale claim exists to recover")
    previous = claims[-1]
    next_generation = previous.generation + 1
    with _open_generation_directory(
        state_fd,
        base,
        previous.generation,
    ) as previous_fd:
        attempt = _load_attempt(
            previous,
            generation_fd=previous_fd,
            map_sha256=map_sha256,
            row_sha256=row_sha256,
            array_index=int(checked_row["array_index"]),
        )
        if previous.generation > 1 and attempt is None:
            raise RuntimeError(
                "recovered claim must consume its attempt before recovery"
            )
        if _entry_lexists_at(
            previous_fd,
            previous.recovery_path.name,
            previous.recovery_path,
        ):
            recovery_document, recovery_sha256 = _load_recovery_record(
                previous,
                generation_fd=previous_fd,
                state_fd=state_fd,
                row=checked_row,
                base=base,
                map_sha256=map_sha256,
                row_sha256=row_sha256,
            )
            recovery_payload = recovery_document["payload"]
            if not isinstance(recovery_payload, dict):
                raise ValueError("recovery payload must be an object")
            if (
                recovery_payload["recovery_audit_sha256"] != audit_sha256
                or recovery_payload["recovery_audit_type"]
                != recovery_audit_type
            ):
                raise ValueError(
                    "requested audit does not match the pending recovery"
                )
        else:
            quarantine = _quarantine_output(
                checked_row,
                base=base,
                generation=previous.generation,
                state_fd=state_fd,
            )
            recovery_payload = {
                "claim_sha256": previous.sha256,
                "next_generation": next_generation,
                "recovery_audit_sha256": audit_sha256,
                "recovery_audit_type": recovery_audit_type,
                "attempt_record_sha256": (
                    attempt[1] if attempt is not None else None
                ),
                "quarantine": quarantine,
                "restart_mode": "fresh",
            }
            _, recovery_sha256 = _create_record(
                previous.recovery_path,
                RECOVERY_TYPE,
                recovery_payload,
                directory_fd=previous_fd,
            )
    claim_payload: dict[str, object] = {
        "job_map_sha256": map_sha256,
        "row_sha256": row_sha256,
        "array_index": checked_row["array_index"],
        "generation": next_generation,
        "recovered_from_claim_sha256": previous.sha256,
        "recovery_record_sha256": recovery_sha256,
    }
    path = base / f"generation-{next_generation:06d}" / "claim.json"
    document, digest = _create_record(
        path,
        CLAIM_TYPE,
        claim_payload,
        state_fd=state_fd,
        generation=next_generation,
        create_generation=True,
    )
    return _claim_from_document(
        path,
        next_generation,
        document,
        digest,
    )


def _recover_job_row_unverified_for_testing(
    payload: Mapping[str, object],
    row: Mapping[str, object],
    *,
    recovery_audit_sha256: str,
) -> JobClaim:
    """Exercise recovery transitions without proving that a claim is stale.

    This helper is intentionally internal and is not exposed by the CLI.
    """

    _, checked_row, map_sha256, row_sha256 = _row_context(payload, row)
    base = _state_dir(checked_row)
    with _transition_lock(base) as state_fd:
        return _transition_to_recovered_claim_locked(
            payload,
            checked_row=checked_row,
            map_sha256=map_sha256,
            row_sha256=row_sha256,
            state_fd=state_fd,
            recovery_audit_sha256=recovery_audit_sha256,
            recovery_audit_type=TEST_ONLY_RECOVERY_AUDIT_TYPE,
        )


def consume_recovery_attempt(
    payload: Mapping[str, object],
    row: Mapping[str, object],
    *,
    scheduler_job_id: str | None = None,
) -> Path:
    """Atomically consume the current recovered generation's sole attempt."""

    _, checked_row, map_sha256, row_sha256 = _row_context(payload, row)
    if scheduler_job_id is not None:
        scheduler_match = _SLURM_ARRAY_JOB_RE.fullmatch(scheduler_job_id)
        if scheduler_match is None or int(
            scheduler_match.group("array_task_id")
        ) != int(checked_row["array_index"]):
            raise ValueError(
                "attempt scheduler job does not match the selected row"
            )
    base = _state_dir(checked_row)
    with _transition_lock(base) as state_fd:
        claims = _load_claim_chain(
            checked_row,
            state_fd=state_fd,
            map_sha256=map_sha256,
            row_sha256=row_sha256,
        )
        if _path_lexists(_output_dir(checked_row)):
            raise RuntimeError(
                "unaudited output exists; new audited recovery is required"
            )
        if not claims or claims[-1].generation == 1:
            raise RuntimeError("no recovered claim is available to attempt")
        current = claims[-1]
        previous = claims[-2]
        with _open_generation_directory(
            state_fd,
            base,
            previous.generation,
        ) as previous_fd:
            recovery_document, _ = _load_recovery_record(
                previous,
                generation_fd=previous_fd,
                state_fd=state_fd,
                row=checked_row,
                base=base,
                map_sha256=map_sha256,
                row_sha256=row_sha256,
            )
            recovery_payload = recovery_document["payload"]
            if not isinstance(recovery_payload, dict):
                raise ValueError("recovery payload must be an object")
            if (
                recovery_payload["recovery_audit_type"]
                == SLURM_RECOVERY_AUDIT_TYPE
            ):
                if scheduler_job_id is None:
                    raise RuntimeError(
                        "typed SLURM recovery requires a new scheduler job"
                    )
                audit_document, _ = _read_record(
                    previous.slurm_recovery_audit_path,
                    payload_type=SLURM_RECOVERY_AUDIT_TYPE,
                    payload_keys=_SLURM_AUDIT_PAYLOAD_KEYS,
                    directory_fd=previous_fd,
                )
                audit_payload = audit_document["payload"]
                if not isinstance(audit_payload, dict):
                    raise ValueError(
                        "SLURM recovery audit payload must be an object"
                    )
                scheduler = audit_payload["scheduler"]
                if not isinstance(scheduler, dict):
                    raise ValueError(
                        "SLURM recovery scheduler payload is invalid"
                    )
                if scheduler.get("job_id") == scheduler_job_id:
                    raise RuntimeError(
                        "failed/requeued SLURM job cannot consume its recovery"
                    )
        with _open_generation_directory(
            state_fd,
            base,
            current.generation,
        ) as current_fd:
            if (
                _load_attempt(
                    current,
                    generation_fd=current_fd,
                    map_sha256=map_sha256,
                    row_sha256=row_sha256,
                    array_index=int(checked_row["array_index"]),
                )
                is not None
            ):
                raise RuntimeError(
                    "recovered claim attempt was already consumed; "
                    "new audited recovery is required"
                )
            attempt_payload: dict[str, object] = {
                "job_map_sha256": map_sha256,
                "row_sha256": row_sha256,
                "array_index": checked_row["array_index"],
                "generation": current.generation,
                "claim_sha256": current.sha256,
                "scheduler_job_id": scheduler_job_id,
            }
            _create_record(
                current.attempt_path,
                ATTEMPT_TYPE,
                attempt_payload,
                directory_fd=current_fd,
            )
        return current.attempt_path


def complete_job_row(
    payload: Mapping[str, object],
    row: Mapping[str, object],
    output_hashes: Mapping[str, str],
    *,
    expected_claim_path: Path | str | None = None,
    expected_claim_sha256: str | None = None,
) -> JobCompletion:
    """Publish one completion, returning the original for an identical retry."""

    checked_payload, checked_row, map_sha256, row_sha256 = _row_context(
        payload,
        row,
    )
    if (expected_claim_path is None) != (expected_claim_sha256 is None):
        raise ValueError(
            "expected claim path and hash must be provided together"
        )
    checked_expected_claim_path = (
        Path(expected_claim_path) if expected_claim_path is not None else None
    )
    checked_expected_claim_sha256 = (
        _require_sha256(
            expected_claim_sha256,
            "expected_claim_sha256",
        )
        if expected_claim_sha256 is not None
        else None
    )
    outputs = _validate_output_hashes(checked_row, output_hashes)
    base = _state_dir(checked_row)
    completion_path = Path(str(checked_row["completion_path"]))
    with _transition_lock(base) as state_fd:
        claims = _load_claim_chain(
            checked_row,
            state_fd=state_fd,
            map_sha256=map_sha256,
            row_sha256=row_sha256,
        )
        if not claims:
            raise RuntimeError("job row must be claimed before completion")
        current = claims[-1]
        if checked_expected_claim_path is not None and (
            current.path != checked_expected_claim_path
            or current.sha256 != checked_expected_claim_sha256
        ):
            raise ValueError(
                "expected claim identity changed before completion"
            )
        if current.generation > 1:
            with _open_generation_directory(
                state_fd,
                base,
                current.generation,
            ) as current_fd:
                if (
                    _load_attempt(
                        current,
                        generation_fd=current_fd,
                        map_sha256=map_sha256,
                        row_sha256=row_sha256,
                        array_index=int(checked_row["array_index"]),
                    )
                    is None
                ):
                    raise RuntimeError(
                        "recovered claim cannot complete before its attempt"
                    )
        if _path_lexists(completion_path):
            existing = _load_completion(
                payload,
                checked_row,
                state_fd=state_fd,
            )
            existing_payload = existing.document["payload"]
            if (
                not isinstance(existing_payload, Mapping)
                or existing_payload["output_hashes"] != outputs
            ):
                raise RuntimeError(
                    "completion already exists with different output hashes"
                )
            return existing
        schema = checked_row["expected_completion_schema"]
        if not isinstance(schema, dict):
            raise ValueError("invalid expected completion schema")
        completion_payload: dict[str, object] = {
            "job_map_sha256": map_sha256,
            "row_sha256": row_sha256,
            "array_index": checked_row["array_index"],
            "generation": current.generation,
            "claim_sha256": current.sha256,
            "output_hashes": outputs,
        }
        document, digest = _create_record(
            completion_path,
            str(schema["payload_type"]),
            completion_payload,
        )
        return _make_job_completion(
            payload=checked_payload,
            row=checked_row,
            path=completion_path,
            digest=digest,
            document=document,
        )


def _slurm_job_id_for_attempt(array_index: int) -> str | None:
    array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
    array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if array_job_id is None and array_task_id is None:
        return None
    if array_job_id is None or array_task_id is None:
        raise ValueError("incomplete SLURM array scheduler identity")
    candidate = f"{array_job_id}_{array_task_id}"
    match = _SLURM_ARRAY_JOB_RE.fullmatch(candidate)
    if match is None or int(match.group("array_task_id")) != array_index:
        raise ValueError(
            "SLURM scheduler identity does not match the selected row"
        )
    return candidate


def _run_selected_row(args: argparse.Namespace) -> int:
    job_map = load_job_map(
        args.job_map,
        expected_sha256=args.job_map_sha256,
    )
    row = select_job_row(
        job_map,
        expected_sha256=args.job_map_sha256,
        array_index=args.array_index,
        array_min=args.array_min,
        array_max=args.array_max,
    )
    is_confirmation = "confirm" in str(row["stage"]).lower()
    if is_confirmation:
        if args.confirmation_seal is None or args.cell_claim is None:
            raise ValueError(
                "confirmation rows require --confirmation-seal and --cell-claim"
            )
        for path in (args.confirmation_seal, args.cell_claim):
            if not Path(path).is_file():
                raise ValueError(
                    f"confirmation interface file is missing: {path}"
                )
    if completion_is_valid(job_map, row):
        print(f"row {row['array_index']} already complete; skipping")
        return 0

    is_cost_benchmark = row.get("role") == "cost-benchmark"
    scheduler_job_id = (
        _slurm_job_id_for_attempt(args.array_index)
        if is_cost_benchmark
        else None
    )
    _, _, _, row_sha256 = _row_context(job_map, row)
    with _open_state_directory(
        _state_dir(row),
        create=False,
    ) as state_fd:
        claims = (
            []
            if state_fd is None
            else _load_claim_chain(
                row,
                state_fd=state_fd,
                map_sha256=str(job_map["payload_sha256"]),
                row_sha256=row_sha256,
            )
        )
    if claims:
        claim = claims[-1]
        if claim.generation == 1:
            raise RuntimeError(
                "active/stale first claim requires audited recovery before retry"
            )
    else:
        claim = claim_job_row(job_map, row)

    if claim.generation > 1:
        if scheduler_job_id is None:
            scheduler_job_id = _slurm_job_id_for_attempt(args.array_index)
        consume_recovery_attempt(
            job_map,
            row,
            scheduler_job_id=scheduler_job_id,
        )
    if _path_lexists(_output_dir(row)):
        raise RuntimeError(
            "unaudited output appeared before the sealed job attempt"
        )
    cost_execution: dict[str, object] | None = None
    if is_cost_benchmark:
        if scheduler_job_id is None:
            raise ValueError(
                "Stage 1 cost execution requires a SLURM scheduler identity"
            )
        cost_execution = publish_cost_execution_authority(
            job_map,
            row,
            claim=claim,
            scheduler_job_id=scheduler_job_id,
        )
    environment = os.environ.copy()
    environment.update(
        {
            "SAMGA_JOB_MAP": str(args.job_map),
            "SAMGA_JOB_MAP_SHA256": str(job_map["payload_sha256"]),
            "SAMGA_JOB_ROW_SHA256": row_sha256,
            "SAMGA_JOB_CLAIM": str(claim.path),
            "SAMGA_JOB_ARRAY_INDEX": str(args.array_index),
            "SAMGA_JOB_ARRAY_MIN": str(args.array_min),
            "SAMGA_JOB_ARRAY_MAX": str(args.array_max),
        }
    )
    if cost_execution is not None:
        environment["SAMGA_JOB_EXECUTION"] = str(cost_execution["path"])
        environment["SAMGA_JOB_EXECUTION_SHA256"] = str(
            cost_execution["sha256"]
        )
    if args.confirmation_seal is not None:
        environment["CONFIRMATION_SEAL"] = str(args.confirmation_seal)
    if args.cell_claim is not None:
        environment["CELL_CLAIM"] = str(args.cell_claim)
    result = subprocess.run(
        row["argv"],  # type: ignore[arg-type]
        env=environment,
        check=False,
    )
    if result.returncode != 0:
        return int(result.returncode)
    if not completion_is_valid(job_map, row):
        raise RuntimeError(
            "job command exited successfully without publishing its sealed completion"
        )
    return 0


def _add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--job-map", required=True, type=Path)
    parser.add_argument("--job-map-sha256", required=True)
    parser.add_argument("--array-index", required=True, type=int)
    parser.add_argument("--array-min", required=True, type=int)
    parser.add_argument("--array-max", required=True, type=int)


def _require_job_environment(name: str) -> str:
    value = os.environ.get(name)
    if value is None or value == "":
        raise ValueError(f"missing required environment variable {name}")
    return value


def _job_environment_index(name: str) -> int:
    text = _require_job_environment(name)
    if not text.isascii() or not text.isdecimal() or str(int(text)) != text:
        raise ValueError(f"{name} must be a canonical non-negative integer")
    return int(text)


def complete_job_row_from_environment(
    output_hashes: Mapping[str, str],
) -> JobCompletion:
    """Complete only the row and claim reconstructed from the job environment."""

    job_map_path = Path(_require_job_environment("SAMGA_JOB_MAP"))
    expected_map_sha256 = _require_sha256(
        _require_job_environment("SAMGA_JOB_MAP_SHA256"),
        "SAMGA_JOB_MAP_SHA256",
    )
    job_map = load_job_map(
        job_map_path,
        expected_sha256=expected_map_sha256,
    )
    array_index = _job_environment_index("SAMGA_JOB_ARRAY_INDEX")
    array_min = _job_environment_index("SAMGA_JOB_ARRAY_MIN")
    array_max = _job_environment_index("SAMGA_JOB_ARRAY_MAX")
    row = select_job_row(
        job_map,
        expected_sha256=expected_map_sha256,
        array_index=array_index,
        array_min=array_min,
        array_max=array_max,
    )
    expected_row_sha256 = _require_sha256(
        _require_job_environment("SAMGA_JOB_ROW_SHA256"),
        "SAMGA_JOB_ROW_SHA256",
    )
    actual_row_sha256 = sha256_json(row)
    if actual_row_sha256 != expected_row_sha256:
        raise ValueError(
            "selected row hash does not match SAMGA_JOB_ROW_SHA256"
        )

    _, checked_row, map_sha256, row_sha256 = _row_context(job_map, row)
    with _open_state_directory(
        _state_dir(checked_row),
        create=False,
    ) as state_fd:
        claims = (
            []
            if state_fd is None
            else _load_claim_chain(
                checked_row,
                state_fd=state_fd,
                map_sha256=map_sha256,
                row_sha256=row_sha256,
            )
        )
    if not claims:
        raise ValueError("SAMGA_JOB_CLAIM has no current claim")
    current_claim = claims[-1]
    submitted_claim_path = Path(_require_job_environment("SAMGA_JOB_CLAIM"))
    if submitted_claim_path != current_claim.path:
        raise ValueError(
            "SAMGA_JOB_CLAIM claim path does not match the selected row"
        )
    return complete_job_row(
        job_map,
        checked_row,
        output_hashes,
        expected_claim_path=current_claim.path,
        expected_claim_sha256=current_claim.sha256,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="build one immutable job map")
    build.add_argument("--rows", required=True, type=Path)
    build.add_argument("--output", required=True, type=Path)

    validate = subparsers.add_parser("validate", help="validate a job map")
    validate.add_argument("--job-map", required=True, type=Path)
    validate.add_argument("--job-map-sha256", required=True)

    select = subparsers.add_parser("select", help="print one validated row")
    _add_selection_arguments(select)

    complete_env = subparsers.add_parser(
        "complete-env",
        help="complete the exact row and claim bound in the job environment",
    )
    complete_env.add_argument("--output-hashes", required=True)

    recover_slurm = subparsers.add_parser(
        "recover-slurm",
        help="recover one failed row from verified SLURM evidence",
    )
    _add_selection_arguments(recover_slurm)
    recover_slurm.add_argument("--failed-slurm-job", required=True)

    run_row = subparsers.add_parser("run-row", help="run one exact array row")
    _add_selection_arguments(run_row)
    run_row.add_argument("--confirmation-seal", type=Path)
    run_row.add_argument("--cell-claim", type=Path)
    return parser


def _load_selected(
    args: argparse.Namespace,
) -> tuple[dict[str, object], dict[str, object]]:
    payload = load_job_map(
        args.job_map,
        expected_sha256=args.job_map_sha256,
    )
    row = select_job_row(
        payload,
        expected_sha256=args.job_map_sha256,
        array_index=args.array_index,
        array_min=args.array_min,
        array_max=args.array_max,
    )
    return payload, row


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        rows_document = _strict_load(args.rows)
        if set(rows_document) != {"rows"} or not isinstance(
            rows_document["rows"],
            list,
        ):
            raise ValueError(
                "rows input must be exactly an object with a rows list"
            )
        result = write_job_map(rows_document["rows"], args.output)
        print(result["payload_sha256"])
        return 0
    if args.command == "validate":
        result = load_job_map(
            args.job_map,
            expected_sha256=args.job_map_sha256,
        )
        print(result["payload_sha256"])
        return 0
    if args.command == "run-row":
        return _run_selected_row(args)
    if args.command == "complete-env":
        try:
            output_hashes = json.loads(args.output_hashes)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "--output-hashes must contain valid JSON"
            ) from exc
        if not isinstance(output_hashes, dict):
            raise ValueError("--output-hashes must decode to an object")
        completion = complete_job_row_from_environment(output_hashes)
        print(completion.path)
        return 0

    payload, row = _load_selected(args)
    if args.command == "recover-slurm":
        claim = recover_job_row_from_slurm(
            payload,
            row,
            job_map_path=args.job_map,
            failed_slurm_job=args.failed_slurm_job,
        )
        print(claim.path)
    elif args.command == "select":
        print(canonical_json_bytes(row).decode("utf-8"))
    else:  # pragma: no cover - argparse enforces the command set.
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
