#!/usr/bin/env python3
"""Run one sealed development-only BrainRW/CLIP-LoRA training cell."""

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
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from samga_brain_rw import brainrw as br
from samga_brain_rw.config import make_run_key
from samga_brain_rw.hashing import (
    canonical_json_bytes,
    ordered_ids_sha256,
    sha256_json,
)
from samga_brain_rw.scores import (
    BRAINRW_TERMINAL_STAGE,
    ScoreArtifact,
    TRAINING_SMOKE_STAGE,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_RUN_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "payload_type",
        "complete",
        "training_complete",
        "scope",
        "validation_scope",
        "observed_scopes",
        "subject",
        "seed",
        "run_key",
        "config_sha256",
        "manifest_sha256",
        "protocol_sha256",
        "input_hashes",
        "input_bundle_sha256",
        "checkpoint_sha256",
        "model_manifest_sha256",
        "target_manifest_sha256",
        "data_order_sha256",
        "task_initialization_sha256",
        "candidate_initialization_sha256",
        "effective_batch_size",
        "planned_steps",
        "completed_steps",
        "semantic_environment",
        "semantic_environment_sha256",
        "runtime_contract",
        "runtime_contract_sha256",
        "runtime_dtype",
        "runtime_evidence",
        "runtime_evidence_sha256",
        "git_sha",
        "git_provenance",
        "validation_metrics",
        "training_smoke_score_directory",
        "resumed_from_sha256",
    }
)


@dataclass(frozen=True)
class BrainRWOutputs:
    """Hashes and paths proven after one BrainRW runner invocation."""

    run_manifest_path: Path
    run_manifest_sha256: str
    checkpoint_path: Path
    checkpoint_sha256: str
    in_loop_metadata_path: Path | None = None
    in_loop_metadata_sha256: str | None = None
    score_directory: Path | None = None
    score_payload_sha256: str | None = None
    score_envelope_sha256: str | None = None


@dataclass(frozen=True)
class _ValidatedBrainRWProof:
    """One immutable training proof reused by scoring and completion."""

    config: br.BrainRWConfigIdentity
    manifest: br.ManifestIdentity
    checkpoint: br.LoadedBrainRWCheckpoint
    run_manifest: Mapping[str, object]
    run_manifest_bytes: bytes
    run_manifest_sha256: str
    identity: Mapping[str, object]
    outputs: BrainRWOutputs


SubprocessRunner = Callable[..., Any]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("smoke", "full"))
    parser.add_argument("--subject", required=True, type=int, choices=range(1, 11))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--resume", required=True)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--clip-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-input-bundle-sha256", required=True)
    parser.add_argument(
        "--expected-semantic-environment-sha256",
        required=True,
    )
    parser.add_argument("--run-key", required=True)
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--device", required=True, choices=("cuda",))
    return parser


def parse_arguments(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.seed < 0:
        parser.error("--seed must be non-negative")
    if (
        not isinstance(arguments.config_id, str)
        or _SAFE_ID_RE.fullmatch(arguments.config_id) is None
    ):
        parser.error("--config-id must be a safe identifier")
    for name in (
        "expected_config_sha256",
        "expected_input_bundle_sha256",
        "expected_semantic_environment_sha256",
    ):
        if _SHA256_RE.fullmatch(getattr(arguments, name)) is None:
            parser.error(
                f"--{name.replace('_', '-')} must be lowercase SHA-256"
            )
    expected_run_key = make_run_key(
        "brainrw-clip-lora",
        arguments.config_id,
        arguments.subject,
        arguments.seed,
        arguments.expected_config_sha256,
        arguments.expected_input_bundle_sha256,
    )
    if arguments.run_key != expected_run_key:
        parser.error("--run-key does not bind the declared BrainRW cell")
    if arguments.output_dir.name != arguments.run_key:
        parser.error("--output-dir basename must equal --run-key")
    if not arguments.resume:
        parser.error("--resume must be explicit")
    if arguments.mode == "smoke":
        if arguments.resume != "none":
            parser.error("smoke mode requires --resume none")
        if (
            type(arguments.max_train_steps) is not int
            or arguments.max_train_steps != 1
        ):
            parser.error("smoke mode requires --max-train-steps 1")
    else:
        if arguments.resume != "none":
            parser.error("full mode requires --resume none")
        if arguments.max_train_steps is not None:
            parser.error("full mode forbids --max-train-steps")
    return arguments


def _project_file(arguments: argparse.Namespace, relative: str) -> Path:
    return br.reject_development_path(
        arguments.project_root / relative,
        f"project file {relative}",
    )


def _preflight_inputs(arguments: argparse.Namespace) -> None:
    project_root = br.reject_development_path(
        arguments.project_root,
        "project root",
    )
    config = br.reject_development_path(arguments.config, "BrainRW config")
    manifest = br.reject_development_path(
        arguments.manifest,
        "protocol manifest",
    )
    br.reject_development_path(arguments.clip_path, "CLIP path")
    output = br.reject_development_path(
        arguments.output_dir,
        "BrainRW output",
    )
    if arguments.resume != "none":
        br.reject_development_path(
            Path(arguments.resume),
            "resume checkpoint",
        )
    if not project_root.is_dir():
        raise ValueError("project root must be an existing directory")
    expected_config = _project_file(
        arguments,
        "experiments/samga_brain_rw/configs/"
        "brainrw_clip_lora_v1.json",
    )
    expected_manifest = _project_file(
        arguments,
        "artifacts/samga_brain_rw/protocol/manifests/"
        f"sub-{arguments.subject:02d}_protocol.json",
    )
    if config != expected_config:
        raise ValueError("BrainRW config path differs from the sealed config")
    if manifest != expected_manifest:
        raise ValueError(
            "protocol manifest path differs from the sealed subject manifest"
        )
    if output.name != arguments.run_key:
        raise ValueError("BrainRW output directory differs from run_key")
    development_roots = (
        project_root / "artifacts/samga_brain_rw",
        project_root / "results/samga_brain_rw",
    )
    if not any(
        _is_below(output, root)
        for root in development_roots
    ):
        raise ValueError(
            "BrainRW output must remain below a development output root"
        )
    if not output.parent.is_dir():
        raise ValueError("BrainRW output parent must already exist")
    for relative in (
        "experiments/samga_brain_rw/train_brainrw.py",
        "experiments/samga_brain_rw/scripts/emit_brainrw_scores.py",
        "experiments/samga_brain_rw/scripts/build_job_map.py",
        "experiments/samga_brain_rw/scripts/run_brainrw_cell.py",
    ):
        if not _project_file(arguments, relative).is_file():
            raise ValueError(f"required project entry point is missing: {relative}")


def _is_below(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return path != root


def _validate_runtime_and_input_identity(
    arguments: argparse.Namespace,
) -> None:
    runtime = br.probe_brainrw_production_runtime()
    if runtime.device.type != arguments.device:
        raise ValueError("BrainRW runtime device differs from sealed device")
    if (
        runtime.semantic_environment_sha256
        != arguments.expected_semantic_environment_sha256
    ):
        raise ValueError(
            "BrainRW runtime semantic environment differs from the job map"
        )
    config = br.verify_brainrw_config(
        arguments.config,
        arguments.clip_path,
    )
    manifest = br.load_development_manifest_identity(
        arguments.manifest,
        expected_subject=arguments.subject,
    )
    run_key, input_bundle, _ = br.brainrw_run_key(
        config,
        manifest,
        arguments.subject,
        arguments.seed,
        runtime.semantic_environment_sha256,
    )
    if config.payload["config_id"] != arguments.config_id:
        raise ValueError("BrainRW config ID differs from the job map")
    if config.sha256 != arguments.expected_config_sha256:
        raise ValueError("BrainRW config SHA-256 differs from the job map")
    if input_bundle != arguments.expected_input_bundle_sha256:
        raise ValueError("BrainRW input bundle differs from the job map")
    if run_key != arguments.run_key:
        raise ValueError("BrainRW runtime run key differs from the job map")


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
        str(
            _project_file(
                arguments,
                "experiments/samga_brain_rw/train_brainrw.py",
            )
        ),
        "--scope",
        "train",
        "--validation-scope",
        "val-dev",
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
        "--clip-path",
        str(arguments.clip_path),
        "--output-dir",
        str(arguments.output_dir),
    ]
    if arguments.mode == "smoke":
        command.extend(
            ["--max-train-steps", str(arguments.max_train_steps)]
        )
    return command


def _score_command(
    arguments: argparse.Namespace,
    checkpoint_path: Path,
) -> list[str]:
    return [
        sys.executable,
        str(
            _project_file(
                arguments,
                "experiments/samga_brain_rw/scripts/"
                "emit_brainrw_scores.py",
            )
        ),
        "--scope",
        "val-dev",
        "--subject",
        str(arguments.subject),
        "--seed",
        str(arguments.seed),
        "--checkpoint",
        str(checkpoint_path),
        "--manifest",
        str(arguments.manifest),
        "--output-dir",
        str(arguments.output_dir / "val_dev_scores"),
    ]


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
        canonical_json_bytes(dict(sorted(output_hashes.items()))).decode(
            "utf-8"
        ),
    ]


def _strict_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


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

        def identity(item: os.stat_result) -> tuple[int, int, int, int, int]:
            return (
                item.st_dev,
                item.st_ino,
                item.st_size,
                item.st_mtime_ns,
                item.st_ctime_ns,
            )

        if identity(before) != identity(after):
            raise ValueError(f"{context} changed while it was read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _canonical_json_line(
    path: Path,
    context: str,
) -> tuple[dict[str, object], bytes]:
    raw = _stable_regular_bytes(path, context)
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise ValueError(f"{context} must contain one canonical JSON line")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{context} must contain a JSON object")
    if canonical_json_bytes(value) + b"\n" != raw:
        raise ValueError(f"{context} is not canonical JSON")
    return value, raw


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_run_manifest(
    value: Mapping[str, object],
    *,
    arguments: argparse.Namespace,
    checkpoint: br.LoadedBrainRWCheckpoint,
    config: br.BrainRWConfigIdentity,
    manifest: br.ManifestIdentity,
) -> None:
    if set(value) != _RUN_MANIFEST_KEYS:
        raise ValueError("BrainRW run manifest keys differ from the schema")
    checkpoint_payload = checkpoint.payload
    expected = {
        "schema_version": 1,
        "payload_type": "samga_brain_rw.brainrw_run_manifest",
        "complete": True,
        "scope": "train",
        "validation_scope": "val-dev",
        "observed_scopes": ["train", "val-dev"],
        "subject": arguments.subject,
        "seed": arguments.seed,
        "run_key": arguments.run_key,
        "config_sha256": arguments.expected_config_sha256,
        "manifest_sha256": manifest.manifest_sha256,
        "protocol_sha256": manifest.protocol_sha256,
        "input_bundle_sha256": arguments.expected_input_bundle_sha256,
        "checkpoint_sha256": checkpoint.sha256,
        "semantic_environment_sha256": (
            arguments.expected_semantic_environment_sha256
        ),
    }
    for field, expected_value in expected.items():
        if value[field] != expected_value:
            raise ValueError(f"BrainRW run manifest {field} mismatch")
    mirrored = {
        "input_hashes",
        "model_manifest_sha256",
        "target_manifest_sha256",
        "data_order_sha256",
        "task_initialization_sha256",
        "candidate_initialization_sha256",
        "effective_batch_size",
        "planned_steps",
        "semantic_environment",
        "runtime_contract",
        "runtime_contract_sha256",
        "runtime_dtype",
        "runtime_evidence",
        "runtime_evidence_sha256",
        "git_sha",
        "git_provenance",
        "validation_metrics",
        "resumed_from_sha256",
    }
    for field in mirrored:
        if value[field] != checkpoint_payload[field]:
            raise ValueError(
                f"BrainRW run manifest/checkpoint {field} mismatch"
            )
    if value["completed_steps"] != checkpoint_payload["global_step"]:
        raise ValueError(
            "BrainRW run manifest/checkpoint completed steps mismatch"
        )
    if value["training_complete"] != checkpoint_payload["training_complete"]:
        raise ValueError(
            "BrainRW run manifest/checkpoint completion flag mismatch"
        )
    if config.sha256 != value["config_sha256"]:
        raise ValueError("BrainRW run manifest verified config mismatch")
    _validate_locked_schedule(
        arguments,
        checkpoint,
        config,
        manifest,
        value,
    )


def _validate_locked_schedule(
    arguments: argparse.Namespace,
    checkpoint: br.LoadedBrainRWCheckpoint,
    config: br.BrainRWConfigIdentity,
    manifest: br.ManifestIdentity,
    run_manifest: Mapping[str, object],
) -> None:
    """Bind outputs to the locked 25x12,540/512 = 625-step recipe."""

    training = config.payload.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("BrainRW config training recipe is missing")
    if training.get("epochs") != 25:
        raise ValueError("BrainRW locked recipe requires 25 epochs")
    if training.get("batch_size") != 512:
        raise ValueError("BrainRW locked recipe requires batch size 512")
    if manifest.train_row_count != 12_540:
        raise ValueError(
            "BrainRW locked recipe requires 12540 training rows"
        )
    expected_steps = 25 * ((12_540 + 512 - 1) // 512)
    if expected_steps != 625:
        raise AssertionError("BrainRW locked step derivation is invalid")
    payload = checkpoint.payload
    if (
        payload.get("planned_steps") != expected_steps
        or run_manifest.get("planned_steps") != expected_steps
        or payload.get("effective_batch_size") != 512
        or run_manifest.get("effective_batch_size") != 512
    ):
        raise ValueError(
            "BrainRW output differs from the locked 625-step recipe"
        )
    if (
        "resumed_from_sha256" not in payload
        or payload["resumed_from_sha256"] is not None
        or "resumed_from_sha256" not in run_manifest
        or run_manifest["resumed_from_sha256"] is not None
    ):
        raise ValueError(
            "BrainRW locked recipe requires null resume parent provenance"
        )
    if arguments.mode == "smoke":
        expected_completed = 1
        expected_training_complete = False
        if (
            arguments.resume != "none"
            or arguments.max_train_steps != 1
        ):
            raise ValueError(
                "BrainRW smoke proof requires one fresh training step"
            )
    else:
        expected_completed = expected_steps
        expected_training_complete = True
        if (
            arguments.resume != "none"
            or arguments.max_train_steps is not None
        ):
            raise ValueError(
                "BrainRW full proof requires one fresh complete run"
            )
    if (
        payload.get("global_step") != expected_completed
        or payload.get("steps") != expected_completed
        or run_manifest.get("completed_steps") != expected_completed
        or payload.get("training_complete")
        is not expected_training_complete
        or run_manifest.get("training_complete")
        is not expected_training_complete
    ):
        raise ValueError(
            "BrainRW completed steps differ from the locked recipe"
        )


def _validate_score_identity(
    artifact: ScoreArtifact,
    *,
    arguments: argparse.Namespace,
    checkpoint: br.LoadedBrainRWCheckpoint,
    manifest: br.ManifestIdentity,
    expected_stage: str,
) -> None:
    checkpoint_payload = checkpoint.payload
    ordered_ids = tuple(manifest.val_dev_ordered_ids)
    ordered_hash = ordered_ids_sha256(ordered_ids)
    if ordered_hash != manifest.val_dev_ordered_ids_sha256:
        raise ValueError("verified manifest val-dev ordered-ID hash mismatch")
    if tuple(artifact.query_ids) != ordered_ids:
        raise ValueError("BrainRW score query IDs differ from the manifest")
    if tuple(artifact.gallery_ids) != ordered_ids:
        raise ValueError("BrainRW score gallery IDs differ from the manifest")
    expected_source_records = [
        {
            "manifest_sha256": manifest.manifest_sha256,
            "records_sha256": manifest.records_sha256,
            "role": "val-dev",
            "role_payload_sha256": manifest.val_dev_role_sha256,
            "source_manifest_sha256": manifest.source_manifest_sha256,
            "source_payload_byte_count": manifest.source_payload_byte_count,
            "source_payload_path": str(manifest.source_payload_path),
            "source_payload_sha256": manifest.source_payload_sha256,
        }
    ]
    source_records_sha256 = sha256_json(expected_source_records)
    expected = {
        "checkpoint_sha256": checkpoint.sha256,
        "config_sha256": arguments.expected_config_sha256,
        "protocol_sha256": manifest.protocol_sha256,
        "git_sha": checkpoint_payload["git_sha"],
        "seed": arguments.seed,
        "split_role": "val-dev",
        "stage": expected_stage,
        "subject": arguments.subject,
        "query_ids_sha256": ordered_hash,
        "gallery_ids_sha256": ordered_hash,
        "source_records_sha256": source_records_sha256,
    }
    for field, expected_value in expected.items():
        if artifact.metadata[field] != expected_value:
            raise ValueError(f"BrainRW score {field} mismatch")
        if artifact.provenance[field] != expected_value:
            raise ValueError(
                f"BrainRW score provenance {field} mismatch"
            )
    if artifact.metadata["source_records"] != expected_source_records:
        raise ValueError(
            "BrainRW score source records differ from the manifest"
        )
    if artifact.metadata["ordered_ids"] != [
        *ordered_ids,
        *ordered_ids,
    ]:
        raise ValueError("BrainRW score ordered IDs differ from the manifest")
    if expected_stage == BRAINRW_TERMINAL_STAGE:
        runtime = {
            "training_semantic_environment": checkpoint_payload[
                "semantic_environment"
            ],
            "training_semantic_environment_sha256": checkpoint_payload[
                "semantic_environment_sha256"
            ],
            "evaluation_semantic_environment": checkpoint_payload[
                "semantic_environment"
            ],
            "evaluation_semantic_environment_sha256": checkpoint_payload[
                "semantic_environment_sha256"
            ],
            "evaluation_runtime_contract": checkpoint_payload[
                "runtime_contract"
            ],
            "evaluation_runtime_contract_sha256": checkpoint_payload[
                "runtime_contract_sha256"
            ],
            "evaluation_runtime_evidence": checkpoint_payload[
                "runtime_evidence"
            ],
            "evaluation_runtime_evidence_sha256": checkpoint_payload[
                "runtime_evidence_sha256"
            ],
        }
        for field, expected_value in runtime.items():
            if artifact.metadata[field] != expected_value:
                raise ValueError(
                    f"BrainRW score runtime binding {field} mismatch"
                )
            if artifact.provenance[field] != expected_value:
                raise ValueError(
                    "BrainRW score provenance runtime binding "
                    f"{field} mismatch"
                )
    else:
        progress = {
            "global_step": checkpoint_payload["global_step"],
            "planned_steps": checkpoint_payload["planned_steps"],
            "training_complete": checkpoint_payload["training_complete"],
        }
        for field, expected_value in progress.items():
            if artifact.metadata[field] != expected_value:
                raise ValueError(
                    f"BrainRW smoke progress {field} mismatch"
                )
            if artifact.provenance[field] != expected_value:
                raise ValueError(
                    f"BrainRW smoke provenance {field} mismatch"
                )


def _normalized_path_text(path: Path) -> str:
    return str(Path(os.path.abspath(os.path.normpath(os.fspath(path)))))


def _make_validated_proof(
    *,
    arguments: argparse.Namespace,
    config: br.BrainRWConfigIdentity,
    manifest: br.ManifestIdentity,
    checkpoint: br.LoadedBrainRWCheckpoint,
    run_manifest: Mapping[str, object],
    run_manifest_bytes: bytes,
    outputs: BrainRWOutputs,
) -> _ValidatedBrainRWProof:
    run_manifest_sha256 = _sha256_bytes(run_manifest_bytes)
    if (
        outputs.checkpoint_sha256 != checkpoint.sha256
        or outputs.run_manifest_sha256 != run_manifest_sha256
    ):
        raise ValueError(
            "BrainRW proof hashes differ from the validated artifacts"
        )
    payload = checkpoint.payload
    identity = MappingProxyType(
        {
            "mode": arguments.mode,
            "subject": arguments.subject,
            "seed": arguments.seed,
            "run_key": arguments.run_key,
            "config_id": arguments.config_id,
            "config_path": str(config.path),
            "config_sha256": config.sha256,
            "clip_path": str(config.clip_path),
            "manifest_path": str(manifest.path),
            "manifest_sha256": manifest.manifest_sha256,
            "protocol_sha256": manifest.protocol_sha256,
            "records_sha256": manifest.records_sha256,
            "source_manifest_sha256": (
                manifest.source_manifest_sha256
            ),
            "source_payload_sha256": (
                manifest.source_payload_sha256
            ),
            "train_role_sha256": manifest.train_role_sha256,
            "val_dev_role_sha256": manifest.val_dev_role_sha256,
            "train_row_count": manifest.train_row_count,
            "val_dev_row_count": manifest.val_dev_row_count,
            "input_bundle_sha256": (
                arguments.expected_input_bundle_sha256
            ),
            "semantic_environment_sha256": (
                arguments.expected_semantic_environment_sha256
            ),
            "checkpoint_sha256": checkpoint.sha256,
            "run_manifest_sha256": run_manifest_sha256,
            "planned_steps": payload["planned_steps"],
            "global_step": payload["global_step"],
            "training_complete": payload["training_complete"],
            "resumed_from_sha256": payload["resumed_from_sha256"],
            "output_dir": _normalized_path_text(arguments.output_dir),
        }
    )
    return _ValidatedBrainRWProof(
        config=config,
        manifest=manifest,
        checkpoint=checkpoint,
        run_manifest=MappingProxyType(dict(run_manifest)),
        run_manifest_bytes=bytes(run_manifest_bytes),
        run_manifest_sha256=run_manifest_sha256,
        identity=identity,
        outputs=outputs,
    )


def _validate_proof_arguments(
    arguments: argparse.Namespace,
    proof: _ValidatedBrainRWProof,
) -> None:
    expected = {
        "mode": arguments.mode,
        "subject": arguments.subject,
        "seed": arguments.seed,
        "run_key": arguments.run_key,
        "config_id": arguments.config_id,
        "config_path": _normalized_path_text(arguments.config),
        "config_sha256": arguments.expected_config_sha256,
        "clip_path": _normalized_path_text(arguments.clip_path),
        "manifest_path": _normalized_path_text(arguments.manifest),
        "input_bundle_sha256": (
            arguments.expected_input_bundle_sha256
        ),
        "semantic_environment_sha256": (
            arguments.expected_semantic_environment_sha256
        ),
        "output_dir": _normalized_path_text(arguments.output_dir),
    }
    for field, expected_value in expected.items():
        if proof.identity.get(field) != expected_value:
            raise ValueError(
                f"BrainRW proof {field} differs from runner arguments"
            )
    if (
        "resumed_from_sha256" not in proof.identity
        or proof.identity["resumed_from_sha256"] is not None
    ):
        raise ValueError(
            "BrainRW proof must bind null resume parent provenance"
        )
    if (
        proof.outputs.checkpoint_sha256
        != proof.identity.get("checkpoint_sha256")
        or proof.outputs.run_manifest_sha256
        != proof.identity.get("run_manifest_sha256")
        or proof.run_manifest_sha256
        != proof.identity.get("run_manifest_sha256")
    ):
        raise ValueError("BrainRW proof output identity is inconsistent")


def _validate_brainrw_training_once(
    arguments: argparse.Namespace,
) -> _ValidatedBrainRWProof:
    """Load and validate each training-side artifact exactly once."""

    output = br.reject_development_path(
        arguments.output_dir,
        "BrainRW output",
    )
    if not output.is_dir() or output.name != arguments.run_key:
        raise ValueError("BrainRW output directory/run_key mismatch")
    config = br.verify_brainrw_config(
        arguments.config,
        arguments.clip_path,
    )
    manifest = br.load_development_manifest_identity(
        arguments.manifest,
        expected_subject=arguments.subject,
    )
    checkpoint_path = output / "checkpoint.pt"
    checkpoint = br.load_brainrw_checkpoint(
        checkpoint_path,
        requested_scope="train",
    )
    br.validate_brainrw_checkpoint_identity(
        checkpoint.payload,
        config=config,
        manifest=manifest,
        subject=arguments.subject,
        seed=arguments.seed,
    )
    if checkpoint.payload["semantic_environment_sha256"] != (
        arguments.expected_semantic_environment_sha256
    ):
        raise ValueError(
            "BrainRW checkpoint semantic environment differs from job map"
        )
    if checkpoint.payload["input_bundle_sha256"] != (
        arguments.expected_input_bundle_sha256
    ):
        raise ValueError("BrainRW checkpoint input bundle differs from job map")
    if checkpoint.payload["run_key"] != arguments.run_key:
        raise ValueError("BrainRW checkpoint run key differs from job map")
    run_manifest_path = output / "run_manifest.json"
    run_manifest, run_manifest_bytes = _canonical_json_line(
        run_manifest_path,
        "BrainRW run manifest",
    )
    _validate_run_manifest(
        run_manifest,
        arguments=arguments,
        checkpoint=checkpoint,
        config=config,
        manifest=manifest,
    )
    common = BrainRWOutputs(
        run_manifest_path=run_manifest_path,
        run_manifest_sha256=_sha256_bytes(run_manifest_bytes),
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint.sha256,
    )
    if arguments.mode == "smoke":
        if (
            checkpoint.payload["training_complete"] is not False
            or checkpoint.payload["global_step"]
            != arguments.max_train_steps
            or checkpoint.payload["global_step"]
            >= checkpoint.payload["planned_steps"]
            or run_manifest["training_smoke_score_directory"]
            != "training_smoke/in_loop"
        ):
            raise ValueError("BrainRW smoke output is not a partial smoke")
        score = ScoreArtifact.load(
            output / "training_smoke" / "in_loop",
            allowed_scopes={"val-dev"},
        )
        _validate_score_identity(
            score,
            arguments=arguments,
            checkpoint=checkpoint,
            manifest=manifest,
            expected_stage=TRAINING_SMOKE_STAGE,
        )
        metadata_path = score.directory / "metadata.json"
        outputs = replace(
            common,
            in_loop_metadata_path=metadata_path,
            in_loop_metadata_sha256=score.verified.envelope_sha256,
        )
    else:
        if (
            checkpoint.payload["training_complete"] is not True
            or checkpoint.payload["global_step"]
            != checkpoint.payload["planned_steps"]
            or run_manifest["training_smoke_score_directory"] is not None
        ):
            raise ValueError("BrainRW full output is not terminal")
        outputs = common
    return _make_validated_proof(
        arguments=arguments,
        config=config,
        manifest=manifest,
        checkpoint=checkpoint,
        run_manifest=run_manifest,
        run_manifest_bytes=run_manifest_bytes,
        outputs=outputs,
    )


def validate_brainrw_training_outputs(
    arguments: argparse.Namespace,
) -> BrainRWOutputs:
    """Compatibility wrapper returning one proof's training outputs."""

    return _validate_brainrw_training_once(arguments).outputs


def _validate_brainrw_outputs_from_proof(
    arguments: argparse.Namespace,
    proof: _ValidatedBrainRWProof,
) -> BrainRWOutputs:
    """Validate scores against exactly one frozen training proof."""

    _validate_proof_arguments(arguments, proof)
    outputs = proof.outputs
    if arguments.mode == "smoke":
        return outputs
    score = ScoreArtifact.load(
        arguments.output_dir / "val_dev_scores",
        allowed_scopes={"val-dev"},
    )
    _validate_score_identity(
        score,
        arguments=arguments,
        checkpoint=proof.checkpoint,
        manifest=proof.manifest,
        expected_stage=BRAINRW_TERMINAL_STAGE,
    )
    return replace(
        outputs,
        score_directory=score.directory,
        score_payload_sha256=score.verified.payload_sha256,
        score_envelope_sha256=score.verified.envelope_sha256,
    )


def validate_brainrw_outputs(
    arguments: argparse.Namespace,
    *,
    proof: _ValidatedBrainRWProof | None = None,
) -> BrainRWOutputs:
    """Capture once when needed, then validate through the frozen proof."""

    selected = (
        _validate_brainrw_training_once(arguments)
        if proof is None
        else proof
    )
    return _validate_brainrw_outputs_from_proof(
        arguments,
        selected,
    )


def validate_brainrw_map_config(
    argv: Sequence[str],
) -> br.BrainRWConfigIdentity:
    """Verify a sealed map command's config-declared local CLIP path."""

    if (
        isinstance(argv, (str, bytes, bytearray))
        or len(argv) < 3
        or any(not isinstance(value, str) or not value for value in argv)
        or argv[0] != "python"
    ):
        raise ValueError("sealed BrainRW runner argv is invalid")
    command = list(argv)
    try:
        arguments = parse_arguments(command[2:])
    except SystemExit as exc:
        raise ValueError(
            "sealed BrainRW runner argument parsing failed"
        ) from exc
    expected_runner = (
        br.reject_development_path(arguments.project_root, "project root")
        / "experiments/samga_brain_rw/scripts/run_brainrw_cell.py"
    )
    if Path(command[1]) != expected_runner:
        raise ValueError(
            "sealed BrainRW runner path differs from project-root binding"
        )
    config = br.verify_brainrw_config(
        arguments.config,
        arguments.clip_path,
    )
    if (
        Path(config.path)
        != Path(_normalized_path_text(arguments.config))
        or Path(config.clip_path)
        != Path(_normalized_path_text(arguments.clip_path))
        or config.payload.get("config_id") != arguments.config_id
        or config.sha256 != arguments.expected_config_sha256
    ):
        raise ValueError(
            "sealed BrainRW config/CLIP identity drifted from the job map"
        )
    return config


def validate_brainrw_command_outputs(
    argv: Sequence[str],
    *,
    expected_mode: str | None = None,
) -> BrainRWOutputs:
    """Reparse one sealed BrainRW runner command and verify its outputs."""

    if (
        isinstance(argv, (str, bytes, bytearray))
        or len(argv) < 3
        or any(not isinstance(value, str) or not value for value in argv)
    ):
        raise ValueError("sealed BrainRW runner argv must be a string sequence")
    command = list(argv)
    if command[0] != "python":
        raise ValueError("sealed BrainRW runner argv prefix mismatch")
    try:
        arguments = parse_arguments(command[2:])
    except SystemExit as exc:
        raise ValueError("sealed BrainRW runner argument parsing failed") from exc
    if expected_mode not in {None, "smoke", "full"}:
        raise ValueError("expected_mode must be smoke, full, or None")
    if expected_mode is not None and arguments.mode != expected_mode:
        raise ValueError("sealed BrainRW runner mode differs from expected mode")
    expected_runner = (
        br.reject_development_path(arguments.project_root, "project root")
        / "experiments/samga_brain_rw/scripts/run_brainrw_cell.py"
    )
    if Path(command[1]) != expected_runner:
        raise ValueError(
            "sealed BrainRW runner path differs from project-root binding"
        )
    return validate_brainrw_outputs(arguments)


def _completion_hashes(outputs: BrainRWOutputs, mode: str) -> dict[str, str]:
    hashes = {
        "final_checkpoint_sha256": outputs.checkpoint_sha256,
        "run_manifest_sha256": outputs.run_manifest_sha256,
    }
    if mode == "smoke":
        if outputs.in_loop_metadata_sha256 is None:
            raise ValueError("BrainRW smoke output lacks in-loop metadata")
        hashes["in_loop_metadata_sha256"] = (
            outputs.in_loop_metadata_sha256
        )
    else:
        if (
            outputs.score_payload_sha256 is None
            or outputs.score_envelope_sha256 is None
        ):
            raise ValueError("BrainRW full output lacks terminal score hashes")
        hashes["score_payload_sha256"] = outputs.score_payload_sha256
        hashes["score_envelope_sha256"] = outputs.score_envelope_sha256
    return hashes


def run_cell(
    arguments: argparse.Namespace,
    *,
    subprocess_runner: SubprocessRunner = subprocess.run,
) -> int:
    """Execute one sealed BrainRW cell and optionally complete its job row."""

    _preflight_inputs(arguments)
    _validate_runtime_and_input_identity(arguments)
    environment = _environment(arguments)
    subprocess_runner(
        _train_command(arguments),
        check=True,
        env=environment,
    )
    proof = _validate_brainrw_training_once(arguments)
    if arguments.mode == "full":
        subprocess_runner(
            _score_command(
                arguments,
                proof.outputs.checkpoint_path,
            ),
            check=True,
            env=environment,
        )
    outputs = _validate_brainrw_outputs_from_proof(
        arguments,
        proof,
    )
    if os.environ.get("SAMGA_JOB_MAP"):
        subprocess_runner(
            _complete_command(
                arguments,
                _completion_hashes(outputs, arguments.mode),
            ),
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
