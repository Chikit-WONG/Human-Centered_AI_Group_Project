#!/usr/bin/env python3
"""Build and enforce immutable SLURM job maps for SAMGA BrainRW runs."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from samga_brain_rw.hashing import canonical_json_bytes, sha256_json


SCHEMA_VERSION = 1
JOB_MAP_TYPE = "samga_brain_rw.job_map"
CLAIM_TYPE = "samga_brain_rw.job_claim"
RECOVERY_TYPE = "samga_brain_rw.job_claim_recovery"
LOG_ROOT = PurePosixPath("logs/samga_brain_rw")
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


@dataclass(frozen=True)
class JobCompletion:
    """An immutable completion for one sealed job-map row."""

    path: Path
    sha256: str
    document: dict[str, object]


def _require_exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{label} keys mismatch: missing={missing}, extra={extra}")


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


def _validate_log_path(value: object, label: str, suffix: str) -> str:
    text = _require_nonempty_string(value, label)
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must remain below logs/samga_brain_rw")
    try:
        path.relative_to(LOG_ROOT)
    except ValueError as exc:
        raise ValueError(f"{label} must remain below logs/samga_brain_rw") from exc
    if path.suffix != suffix:
        raise ValueError(f"{label} must end in {suffix}")
    return text


def _flag_value(argv: Sequence[str], flag: str) -> str:
    positions = [index for index, value in enumerate(argv) if value == flag]
    if len(positions) != 1:
        raise ValueError(f"sealed argv must contain {flag} exactly once")
    position = positions[0]
    if position + 1 >= len(argv) or argv[position + 1].startswith("--"):
        raise ValueError(f"sealed argv must bind a value to {flag}")
    return argv[position + 1]


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

    _validate_log_path(row["stdout_path"], "stdout_path", ".out")
    _validate_log_path(row["stderr_path"], "stderr_path", ".err")
    completion_path = _require_nonempty_string(
        row["completion_path"],
        "completion_path",
    )
    if _is_forbidden_path_token(completion_path):
        raise ValueError("completion_path contains a forbidden test/formal path")
    if Path(completion_path).suffix != ".json":
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
    )


def _logical_run_key(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        row["stage"],
        row["role"],
        row["config_id"],
        row["subject"],
        row["seed"],
    )


def _json_copy(value: object) -> Any:
    try:
        return json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("job map must contain only canonical JSON values") from exc


def build_job_map(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Validate, sort, index, and hash one homogeneous stage/resource map."""

    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        raise ValueError("job map requires at least one row")
    validated = [_validate_row(_json_copy(row), expect_index=False) for row in rows]
    stages = {str(row["stage"]) for row in validated}
    if len(stages) != 1:
        raise ValueError("job map must contain one homogeneous stage")
    resources = {_resource_key(row) for row in validated}
    if len(resources) != 1:
        raise ValueError("job map must contain one homogeneous resource class")
    logical_keys = [_logical_run_key(row) for row in validated]
    if len(set(logical_keys)) != len(logical_keys):
        raise ValueError("duplicate logical job-map row")
    for field in ("stdout_path", "stderr_path", "completion_path"):
        values = [str(row[field]) for row in validated]
        if len(set(values)) != len(values):
            raise ValueError(f"duplicate {field} in job map")

    ordered = sorted(validated, key=job_row_sort_key)
    indexed = [
        {"array_index": index, **row}
        for index, row in enumerate(ordered)
    ]
    body: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "payload_type": JOB_MAP_TYPE,
        "stage": next(iter(stages)),
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
        _validate_row(_json_copy(row), expect_index=True)
        for row in rows
    ]
    if any(row["stage"] != stage for row in validated):
        raise ValueError("job-map row stage mismatch")
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
    for field in ("stdout_path", "stderr_path", "completion_path"):
        values = [str(row[field]) for row in validated]
        if len(set(values)) != len(values):
            raise ValueError(f"duplicate {field} in job map")

    claimed = _require_sha256(payload["payload_sha256"], "job-map hash")
    body = {key: value for key, value in payload.items() if key != "payload_sha256"}
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
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
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


def _strict_load(path: Path) -> dict[str, object]:
    data = _read_regular_file(path)
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
    if (
        expected_sha256 is not None
        and payload["payload_sha256"]
        != _require_sha256(expected_sha256, "expected job-map hash")
    ):
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
    if type(index) is not int or index >= validated[  # type: ignore[operator]
        "row_count"
    ]:
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


def _state_dir(row: Mapping[str, object]) -> Path:
    completion = Path(str(row["completion_path"]))
    return (
        completion.parent
        / ".job-claims"
        / f"{completion.stem}-array-{int(row['array_index']):06d}"
    )


@contextlib.contextmanager
def _transition_lock(directory: Path) -> Iterator[None]:
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / ".transition.lock"
    fd = os.open(
        lock_path,
        os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


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


def _create_record(
    path: Path,
    payload_type: str,
    payload: dict[str, object],
) -> tuple[dict[str, object], str]:
    document = _record_document(payload_type, payload)
    data = canonical_json_bytes(document)
    _exclusive_publish(path, data)
    return document, hashlib.sha256(data).hexdigest()


def _read_record(
    path: Path,
    *,
    payload_type: str,
    payload_keys: set[str],
) -> tuple[dict[str, object], str]:
    document = _strict_load(path)
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


def _generation_numbers(base: Path) -> list[int]:
    if not base.exists():
        return []
    numbers = sorted(
        int(match.group(1))
        for child in base.iterdir()
        if child.is_dir()
        for match in [_GENERATION_RE.fullmatch(child.name)]
        if match is not None
    )
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


def _load_claim_chain(
    row: Mapping[str, object],
    *,
    map_sha256: str,
    row_sha256: str,
) -> list[JobClaim]:
    base = _state_dir(row)
    claims: list[JobClaim] = []
    for generation in _generation_numbers(base):
        path = base / f"generation-{generation:06d}" / "claim.json"
        document, digest = _read_record(
            path,
            payload_type=CLAIM_TYPE,
            payload_keys=CLAIM_PAYLOAD_KEYS,
        )
        claim = _claim_from_document(path, generation, document, digest)
        _validate_claim_payload(
            claim,
            map_sha256=map_sha256,
            row_sha256=row_sha256,
            array_index=int(row["array_index"]),
        )
        if claims:
            previous = claims[-1]
            recovery_document, recovery_sha256 = _read_record(
                previous.recovery_path,
                payload_type=RECOVERY_TYPE,
                payload_keys=RECOVERY_PAYLOAD_KEYS,
            )
            recovery = recovery_document["payload"]
            current_payload = claim.document["payload"]
            if not isinstance(recovery, dict) or not isinstance(current_payload, dict):
                raise ValueError("invalid claim recovery chain")
            if (
                recovery["claim_sha256"] != previous.sha256
                or recovery["next_generation"] != generation
                or current_payload["recovered_from_claim_sha256"] != previous.sha256
                or current_payload["recovery_record_sha256"] != recovery_sha256
            ):
                raise ValueError("claim recovery chain hash mismatch")
            _require_sha256(
                recovery["recovery_audit_sha256"],
                "recovery_audit_sha256",
            )
        claims.append(claim)
    if claims and claims[-1].recovery_path.exists():
        raise ValueError("audited recovery is missing its next claim generation")
    return claims


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
        raise ValueError("completion output hashes do not match the sealed schema")
    return {
        str(name): _require_sha256(output_hashes[name], str(name))
        for name in names  # type: ignore[union-attr]
    }


def _load_completion(
    payload: Mapping[str, object],
    row: Mapping[str, object],
) -> JobCompletion:
    _, checked_row, map_sha256, row_sha256 = _row_context(payload, row)
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
    claims = _load_claim_chain(
        checked_row,
        map_sha256=map_sha256,
        row_sha256=row_sha256,
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
        raise ValueError("completion does not match its sealed row/current claim")
    outputs = completion_payload["output_hashes"]
    if not isinstance(outputs, dict):
        raise ValueError("completion output_hashes must be an object")
    _validate_output_hashes(checked_row, outputs)
    return JobCompletion(path=path, sha256=digest, document=document)


def completion_is_valid(
    payload: Mapping[str, object],
    row: Mapping[str, object],
) -> bool:
    """Return false only for absence; reject any present invalid completion."""

    _, checked_row, _, _ = _row_context(payload, row)
    path = Path(str(checked_row["completion_path"]))
    if not path.exists():
        return False
    _load_completion(payload, checked_row)
    return True


def should_submit_row(
    payload: Mapping[str, object],
    row: Mapping[str, object],
) -> bool:
    """Return whether a row lacks a valid completion and an active first claim."""

    _, checked_row, map_sha256, row_sha256 = _row_context(payload, row)
    if completion_is_valid(payload, checked_row):
        return False
    claims = _load_claim_chain(
        checked_row,
        map_sha256=map_sha256,
        row_sha256=row_sha256,
    )
    return not claims or claims[-1].generation > 1


def claim_job_row(
    payload: Mapping[str, object],
    row: Mapping[str, object],
) -> JobClaim:
    """Create the first immutable claim, refusing implicit stale retries."""

    _, checked_row, map_sha256, row_sha256 = _row_context(payload, row)
    base = _state_dir(checked_row)
    with _transition_lock(base):
        if completion_is_valid(payload, checked_row):
            raise RuntimeError("job row already has a valid completion")
        claims = _load_claim_chain(
            checked_row,
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
        document, digest = _create_record(path, CLAIM_TYPE, claim_payload)
        return _claim_from_document(path, 1, document, digest)


def recover_job_row(
    payload: Mapping[str, object],
    row: Mapping[str, object],
    *,
    recovery_audit_sha256: str,
) -> JobClaim:
    """Recover a stale claim through a sealed audit record and next generation."""

    audit_sha256 = _require_sha256(
        recovery_audit_sha256,
        "recovery_audit_sha256",
    )
    _, checked_row, map_sha256, row_sha256 = _row_context(payload, row)
    base = _state_dir(checked_row)
    with _transition_lock(base):
        if completion_is_valid(payload, checked_row):
            raise RuntimeError("cannot recover a completed job row")
        claims = _load_claim_chain(
            checked_row,
            map_sha256=map_sha256,
            row_sha256=row_sha256,
        )
        if not claims:
            raise RuntimeError("no stale claim exists to recover")
        previous = claims[-1]
        next_generation = previous.generation + 1
        recovery_payload: dict[str, object] = {
            "claim_sha256": previous.sha256,
            "next_generation": next_generation,
            "recovery_audit_sha256": audit_sha256,
        }
        _, recovery_sha256 = _create_record(
            previous.recovery_path,
            RECOVERY_TYPE,
            recovery_payload,
        )
        claim_payload: dict[str, object] = {
            "job_map_sha256": map_sha256,
            "row_sha256": row_sha256,
            "array_index": checked_row["array_index"],
            "generation": next_generation,
            "recovered_from_claim_sha256": previous.sha256,
            "recovery_record_sha256": recovery_sha256,
        }
        path = (
            base
            / f"generation-{next_generation:06d}"
            / "claim.json"
        )
        document, digest = _create_record(path, CLAIM_TYPE, claim_payload)
        return _claim_from_document(
            path,
            next_generation,
            document,
            digest,
        )


def complete_job_row(
    payload: Mapping[str, object],
    row: Mapping[str, object],
    output_hashes: Mapping[str, str],
) -> JobCompletion:
    """Publish one completion, returning the original for an identical retry."""

    _, checked_row, map_sha256, row_sha256 = _row_context(payload, row)
    outputs = _validate_output_hashes(checked_row, output_hashes)
    base = _state_dir(checked_row)
    completion_path = Path(str(checked_row["completion_path"]))
    with _transition_lock(base):
        if completion_path.exists():
            existing = _load_completion(payload, checked_row)
            existing_payload = existing.document["payload"]
            if (
                not isinstance(existing_payload, dict)
                or existing_payload["output_hashes"] != outputs
            ):
                raise RuntimeError(
                    "completion already exists with different output hashes"
                )
            return existing
        claims = _load_claim_chain(
            checked_row,
            map_sha256=map_sha256,
            row_sha256=row_sha256,
        )
        if not claims:
            raise RuntimeError("job row must be claimed before completion")
        current = claims[-1]
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
        return JobCompletion(
            path=completion_path,
            sha256=digest,
            document=document,
        )


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
                raise ValueError(f"confirmation interface file is missing: {path}")
    if completion_is_valid(job_map, row):
        print(f"row {row['array_index']} already complete; skipping")
        return 0

    _, _, _, row_sha256 = _row_context(job_map, row)
    claims = _load_claim_chain(
        row,
        map_sha256=str(job_map["payload_sha256"]),
        row_sha256=row_sha256,
    )
    if claims:
        claim = claims[-1]
        if claim.generation == 1:
            raise RuntimeError(
                "active/stale first claim requires audited recovery before retry"
            )
    else:
        claim = claim_job_row(job_map, row)

    environment = os.environ.copy()
    environment.update(
        {
            "SAMGA_JOB_MAP": str(args.job_map),
            "SAMGA_JOB_MAP_SHA256": str(job_map["payload_sha256"]),
            "SAMGA_JOB_ROW_SHA256": row_sha256,
            "SAMGA_JOB_CLAIM": str(claim.path),
        }
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

    claim = subparsers.add_parser("claim", help="claim one validated row")
    _add_selection_arguments(claim)

    recover = subparsers.add_parser("recover", help="audit and recover a stale row")
    _add_selection_arguments(recover)
    recover.add_argument("--recovery-audit-sha256", required=True)

    complete = subparsers.add_parser("complete", help="complete one claimed row")
    _add_selection_arguments(complete)
    complete.add_argument("--output-hashes", required=True)

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
            raise ValueError("rows input must be exactly an object with a rows list")
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

    payload, row = _load_selected(args)
    if args.command == "select":
        print(canonical_json_bytes(row).decode("utf-8"))
    elif args.command == "claim":
        print(claim_job_row(payload, row).path)
    elif args.command == "recover":
        print(
            recover_job_row(
                payload,
                row,
                recovery_audit_sha256=args.recovery_audit_sha256,
            ).path
        )
    elif args.command == "complete":
        output_hashes = json.loads(args.output_hashes)
        if not isinstance(output_hashes, dict):
            raise ValueError("--output-hashes must decode to an object")
        print(complete_job_row(payload, row, output_hashes).path)
    else:  # pragma: no cover - argparse enforces the command set.
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
