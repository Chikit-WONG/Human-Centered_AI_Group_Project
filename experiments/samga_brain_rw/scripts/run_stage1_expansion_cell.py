#!/usr/bin/env python3
"""Run and seal one Stage 1 train-only expansion component cell."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from samga_brain_rw import brainrw as br
from samga_brain_rw.config import make_run_key
from samga_brain_rw.hashing import canonical_json_bytes, sha256_json
from samga_brain_rw.trainer import SCHEDULE_SHA256 as INTERNVIT_SCHEDULE_SHA256


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_NO_VALIDATION = {"performed": False, "validation_scope": "none"}
_BRAINRW_SCHEDULE = {
    "batch_size": 512,
    "batches_per_epoch": 25,
    "epochs": 25,
    "planned_steps": 625,
    "train_row_count": 12_540,
}
BRAINRW_SCHEDULE_SHA256 = sha256_json(_BRAINRW_SCHEDULE)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--component",
        required=True,
        choices=("internvit", "brainrw"),
    )
    parser.add_argument(
        "--validation-scope",
        required=True,
        choices=("none",),
    )
    parser.add_argument("--subject", required=True, type=int, choices=range(1, 11))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--resume", required=True)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--clip-path", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-input-bundle-sha256", required=True)
    parser.add_argument(
        "--expected-semantic-environment-sha256",
    )
    parser.add_argument("--locked-survivor-sha256", required=True)
    parser.add_argument("--run-key", required=True)
    parser.add_argument("--device", required=True, choices=("cuda",))
    return parser


def parse_arguments(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.seed < 0:
        parser.error("--seed must be non-negative")
    if arguments.resume != "none":
        parser.error("Stage 1 expansion requires --resume none")
    if _SAFE_ID_RE.fullmatch(arguments.config_id) is None:
        parser.error("--config-id must be a safe identifier")
    for name in (
        "expected_config_sha256",
        "expected_input_bundle_sha256",
        "locked_survivor_sha256",
    ):
        if _SHA256_RE.fullmatch(str(getattr(arguments, name))) is None:
            parser.error(f"--{name.replace('_', '-')} must be lowercase SHA-256")
    if arguments.component == "internvit":
        if arguments.feature_cache is None:
            parser.error("InternViT requires --feature-cache")
        if arguments.clip_path is not None:
            parser.error("InternViT forbids --clip-path")
        if arguments.expected_semantic_environment_sha256 is not None:
            parser.error(
                "InternViT forbids --expected-semantic-environment-sha256"
            )
        prefix = "stage0"
    else:
        if arguments.feature_cache is not None:
            parser.error("BrainRW forbids --feature-cache")
        if arguments.clip_path is None:
            parser.error("BrainRW requires --clip-path")
        semantic = arguments.expected_semantic_environment_sha256
        if semantic is None or _SHA256_RE.fullmatch(semantic) is None:
            parser.error(
                "BrainRW requires a lowercase semantic environment SHA-256"
            )
        prefix = "brainrw-clip-lora"
    expected_run_key = make_run_key(
        prefix,
        arguments.config_id,
        arguments.subject,
        arguments.seed,
        arguments.expected_config_sha256,
        arguments.expected_input_bundle_sha256,
    )
    if arguments.run_key != expected_run_key:
        parser.error("--run-key does not bind the declared expansion cell")
    if arguments.output_dir.name != arguments.run_key:
        parser.error("--output-dir basename must equal --run-key")
    return arguments


def _project_file(arguments: argparse.Namespace, relative: str) -> Path:
    return br.reject_development_path(
        arguments.project_root / relative,
        f"project file {relative}",
    )


def _is_below(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return path != root


def _preflight_paths(arguments: argparse.Namespace) -> None:
    root = br.reject_development_path(arguments.project_root, "project root")
    if not root.is_dir():
        raise ValueError("project root must be an existing directory")
    expected_config = _project_file(
        arguments,
        (
            "experiments/samga_brain_rw/configs/internvit_baseline_v1.json"
            if arguments.component == "internvit"
            else "experiments/samga_brain_rw/configs/brainrw_clip_lora_v1.json"
        ),
    )
    expected_manifest = _project_file(
        arguments,
        "artifacts/samga_brain_rw/protocol/manifests/"
        f"sub-{arguments.subject:02d}_protocol.json",
    )
    if arguments.config != expected_config:
        raise ValueError("config path differs from the sealed component config")
    if arguments.manifest != expected_manifest:
        raise ValueError("manifest path differs from the sealed subject manifest")
    output = br.reject_development_path(
        arguments.output_dir,
        "Stage 1 expansion output",
    )
    expected_parent = (
        root
        / "artifacts/samga_brain_rw/stage-1-expansion-train"
        / arguments.component
    )
    if output.parent != expected_parent or not _is_below(output, expected_parent):
        raise ValueError("output directory differs from the sealed expansion root")
    if not output.parent.is_dir():
        raise ValueError("component output parent must already exist")
    if output.exists():
        raise FileExistsError("component output already exists")
    if arguments.component == "internvit":
        expected_cache = _project_file(
            arguments,
            "artifacts/samga_reproduction/features/"
            "internvit_v2_5_multi_variant/variants/"
            "train_idx0_patch_mean/features.npy",
        )
        if arguments.feature_cache != expected_cache:
            raise ValueError("feature cache differs from the sealed cache")
    for relative in (
        "experiments/samga_brain_rw/train.py",
        "experiments/samga_brain_rw/train_brainrw.py",
        "experiments/samga_brain_rw/scripts/build_job_map.py",
        "experiments/samga_brain_rw/scripts/run_stage1_expansion_cell.py",
    ):
        if not _project_file(arguments, relative).is_file():
            raise ValueError(f"required project entry point is missing: {relative}")


def _training_command(arguments: argparse.Namespace) -> list[str]:
    common = [
        "--scope",
        "train",
        "--validation-scope",
        "none",
        "--subject",
        str(arguments.subject),
        "--seed",
        str(arguments.seed),
        "--resume",
        "none",
        "--config",
        str(arguments.config),
        "--manifest",
        str(arguments.manifest),
    ]
    if arguments.component == "internvit":
        return [
            sys.executable,
            str(_project_file(arguments, "experiments/samga_brain_rw/train.py")),
            *common[:4],
            "--stage",
            "0",
            *common[4:],
            "--feature-cache",
            str(arguments.feature_cache),
            "--output-dir",
            str(arguments.output_dir),
            "--device",
            arguments.device,
        ]
    return [
        sys.executable,
        str(
            _project_file(
                arguments,
                "experiments/samga_brain_rw/train_brainrw.py",
            )
        ),
        *common,
        "--clip-path",
        str(arguments.clip_path),
        "--output-dir",
        str(arguments.output_dir),
    ]


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


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


def _stable_regular_bytes(path: Path, context: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
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
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
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
        raise ValueError(f"{context} must contain an object")
    if canonical_json_bytes(value) + b"\n" != raw:
        raise ValueError(f"{context} is not canonical JSON")
    return value, raw


def _file_sha256(path: Path, context: str) -> str:
    return hashlib.sha256(_stable_regular_bytes(path, context)).hexdigest()


def _validate_common_manifest(
    value: Mapping[str, object],
    arguments: argparse.Namespace,
) -> None:
    expected = {
        "subject": arguments.subject,
        "seed": arguments.seed,
        "run_key": arguments.run_key,
        "config_sha256": arguments.expected_config_sha256,
        "validation_scope": "none",
        "observed_scopes": ["train"],
        "validation_metrics": _NO_VALIDATION,
    }
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            raise ValueError(f"train-only run manifest {field} mismatch")


def _validate_internvit(
    arguments: argparse.Namespace,
) -> tuple[dict[str, object], Mapping[str, object], Path, str, str]:
    import train as samga_train

    run_manifest_path = arguments.output_dir / "run_manifest.json"
    run_manifest, run_manifest_bytes = _canonical_json_line(
        run_manifest_path,
        "SAMGA train-only run manifest",
    )
    _validate_common_manifest(run_manifest, arguments)
    if (
        run_manifest.get("payload_type") != "samga_brain_rw.development_run"
        or run_manifest.get("stage") != 0
        or run_manifest.get("completed") is not True
        or run_manifest.get("in_loop_score_directory") is not None
        or run_manifest.get("max_train_steps") is not None
        or run_manifest.get("resume_source_checkpoint_sha256") is not None
        or run_manifest.get("global_step") != 1440
    ):
        raise ValueError("SAMGA train-only terminal evidence is invalid")
    final_name = run_manifest.get("final_checkpoint")
    if (
        not isinstance(final_name, str)
        or final_name != "checkpoint_epoch060.pt"
        or Path(final_name).name != final_name
    ):
        raise ValueError("SAMGA final checkpoint name is invalid")
    checkpoint_path = arguments.output_dir / final_name
    loaded = samga_train.load_samga_checkpoint(
        checkpoint_path,
        requested_scope="train",
    )
    checkpoint_payload = loaded.payload
    input_hashes = checkpoint_payload.get("input_hashes")
    candidate_spec = checkpoint_payload.get("candidate_spec")
    embedded_manifest = checkpoint_payload.get("run_manifest")
    if (
        not isinstance(input_hashes, Mapping)
        or not isinstance(candidate_spec, Mapping)
        or not isinstance(embedded_manifest, Mapping)
    ):
        raise ValueError("SAMGA checkpoint provenance mappings are invalid")
    input_bundle_sha256 = sha256_json(dict(sorted(input_hashes.items())))
    if (
        loaded.sha256 != run_manifest.get("final_checkpoint_sha256")
        or checkpoint_payload.get("subject") != arguments.subject
        or checkpoint_payload.get("seed") != arguments.seed
        or checkpoint_payload.get("config_sha256")
        != arguments.expected_config_sha256
        or input_bundle_sha256
        != arguments.expected_input_bundle_sha256
        or candidate_spec.get("run_key") != arguments.run_key
        or candidate_spec.get("semantic_config_sha256")
        != arguments.expected_config_sha256
        or candidate_spec.get("input_bundle_sha256")
        != arguments.expected_input_bundle_sha256
        or embedded_manifest.get("run_key") != arguments.run_key
        or embedded_manifest.get("config_sha256")
        != arguments.expected_config_sha256
        or checkpoint_payload.get("validation_metrics") != _NO_VALIDATION
        or input_hashes.get("validation_policy")
        != sha256_json({"validation_scope": "none"})
        or checkpoint_payload.get("epoch") != 60
        or checkpoint_payload.get("global_step") != 1440
    ):
        raise ValueError("SAMGA train-only checkpoint identity mismatch")
    return (
        run_manifest,
        checkpoint_payload,
        checkpoint_path,
        loaded.sha256,
        hashlib.sha256(run_manifest_bytes).hexdigest(),
    )


def _validate_brainrw(
    arguments: argparse.Namespace,
) -> tuple[dict[str, object], Mapping[str, object], Path, str, str]:
    config = br.verify_brainrw_config(arguments.config, arguments.clip_path)
    manifest = br.load_development_manifest_identity(
        arguments.manifest,
        expected_subject=arguments.subject,
    )
    checkpoint_path = arguments.output_dir / "checkpoint.pt"
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
    run_manifest_path = arguments.output_dir / "run_manifest.json"
    run_manifest, run_manifest_bytes = _canonical_json_line(
        run_manifest_path,
        "BrainRW train-only run manifest",
    )
    _validate_common_manifest(run_manifest, arguments)
    payload = checkpoint.payload
    if (
        run_manifest.get("payload_type")
        != "samga_brain_rw.brainrw_run_manifest"
        or run_manifest.get("complete") is not True
        or run_manifest.get("training_complete") is not True
        or run_manifest.get("scope") != "train"
        or run_manifest.get("completed_steps") != 625
        or run_manifest.get("planned_steps") != 625
        or run_manifest.get("training_smoke_score_directory") is not None
        or run_manifest.get("resumed_from_sha256") is not None
        or run_manifest.get("checkpoint_sha256") != checkpoint.sha256
        or run_manifest.get("input_bundle_sha256")
        != arguments.expected_input_bundle_sha256
        or payload.get("validation_scope") != "none"
        or payload.get("observed_scopes") != ["train"]
        or payload.get("validation_metrics") != _NO_VALIDATION
        or payload.get("global_step") != 625
        or payload.get("training_complete") is not True
        or payload.get("semantic_environment_sha256")
        != arguments.expected_semantic_environment_sha256
    ):
        raise ValueError("BrainRW train-only terminal evidence is invalid")
    return (
        run_manifest,
        payload,
        checkpoint_path,
        checkpoint.sha256,
        hashlib.sha256(run_manifest_bytes).hexdigest(),
    )


def seal_component_record(
    payload: Mapping[str, object],
) -> dict[str, object]:
    normalized = json.loads(canonical_json_bytes(dict(payload)))
    if not isinstance(normalized, dict):
        raise TypeError("component record payload must be an object")
    return {
        "schema_version": 1,
        "payload": normalized,
        "payload_sha256": sha256_json(normalized),
    }


def _component_payload(
    arguments: argparse.Namespace,
    run_manifest: Mapping[str, object],
    checkpoint_path: Path,
    checkpoint_payload: Mapping[str, object],
    checkpoint_sha256: str,
    run_manifest_sha256: str,
) -> dict[str, object]:
    input_hashes = checkpoint_payload.get("input_hashes")
    if not isinstance(input_hashes, Mapping):
        raise ValueError("checkpoint input_hashes are invalid")
    model_hash_name = (
        "model_sha256"
        if arguments.component == "internvit"
        else "clip_weights"
    )
    frozen_model_sha256 = input_hashes.get(model_hash_name)
    if not isinstance(frozen_model_sha256, str):
        raise ValueError("frozen base model hash is missing")
    manifest_hash_name = (
        "manifest_sha256"
        if arguments.component == "internvit"
        else "manifest"
    )
    manifest_sha256 = input_hashes.get(manifest_hash_name)
    if not isinstance(manifest_sha256, str):
        raise ValueError("training manifest hash is missing")
    return {
        "artifact_type": "samga_brain_rw.stage1_expansion_component",
        "schema_version": 1,
        "complete": True,
        "component": arguments.component,
        "subject": arguments.subject,
        "seed": arguments.seed,
        "epochs": 60 if arguments.component == "internvit" else 25,
        "schedule_sha256": (
            INTERNVIT_SCHEDULE_SHA256
            if arguments.component == "internvit"
            else BRAINRW_SCHEDULE_SHA256
        ),
        "config_id": arguments.config_id,
        "config_sha256": arguments.expected_config_sha256,
        "input_bundle_sha256": arguments.expected_input_bundle_sha256,
        "run_key": arguments.run_key,
        "manifest_sha256": manifest_sha256,
        "protocol_sha256": run_manifest.get("protocol_sha256"),
        "validation_scope": "none",
        "observed_scopes": ["train"],
        "validation_metrics": dict(_NO_VALIDATION),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "run_manifest_path": str(arguments.output_dir / "run_manifest.json"),
        "run_manifest_sha256": run_manifest_sha256,
        "locked_survivor_sha256": arguments.locked_survivor_sha256,
        "git_sha": run_manifest.get("git_sha"),
        "frozen_base_model_sha256": frozen_model_sha256,
    }


def _publish_completion(
    output_hashes: Mapping[str, str],
    arguments: argparse.Namespace,
) -> None:
    required_environment = (
        "SAMGA_JOB_MAP",
        "SAMGA_JOB_MAP_SHA256",
        "SAMGA_JOB_ROW_SHA256",
        "SAMGA_JOB_CLAIM",
        "SAMGA_JOB_ARRAY_INDEX",
        "SAMGA_JOB_ARRAY_MIN",
        "SAMGA_JOB_ARRAY_MAX",
    )
    if not any(name in os.environ for name in required_environment):
        return
    if not all(os.environ.get(name) for name in required_environment):
        raise ValueError("incomplete sealed job-map environment")
    subprocess.run(
        [
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
        ],
        check=True,
        env=os.environ.copy(),
    )


def run(arguments: argparse.Namespace) -> dict[str, str]:
    _preflight_paths(arguments)
    subprocess.run(
        _training_command(arguments),
        check=True,
        env=_environment(arguments),
    )
    if arguments.component == "internvit":
        (
            run_manifest,
            checkpoint_payload,
            checkpoint_path,
            checkpoint_sha,
            manifest_sha,
        ) = _validate_internvit(arguments)
    else:
        (
            run_manifest,
            checkpoint_payload,
            checkpoint_path,
            checkpoint_sha,
            manifest_sha,
        ) = _validate_brainrw(arguments)
    payload = _component_payload(
        arguments,
        run_manifest,
        checkpoint_path,
        checkpoint_payload,
        checkpoint_sha,
        manifest_sha,
    )
    record_path = arguments.output_dir / "component_record.json"
    br.write_development_file_exclusive(
        record_path,
        canonical_json_bytes(seal_component_record(payload)) + b"\n",
        context="Stage 1 expansion component record",
    )
    output_hashes = {
        "component_record_sha256": _file_sha256(
            record_path,
            "component record",
        ),
        "final_checkpoint_sha256": checkpoint_sha,
        "run_manifest_sha256": manifest_sha,
    }
    _publish_completion(output_hashes, arguments)
    return output_hashes


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    try:
        arguments = parse_arguments(argv)
        run(arguments)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
