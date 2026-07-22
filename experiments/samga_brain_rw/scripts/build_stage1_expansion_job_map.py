#!/usr/bin/env python3
"""Build the sealed 100-cell Stage 1 train-only expansion job map."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import build_job_map as job_maps
from samga_brain_rw import brainrw as br
from samga_brain_rw.config import make_run_key
from samga_brain_rw.hashing import canonical_json_bytes
from samga_brain_rw.runtime_contract import validate_environment_binding


_STAGE = "stage-1-expansion-train"
_ALLOWED_PARTITIONS = frozenset(
    {"i64m1tga40u", "i64m1tga40ue", "emergency_gpua40"}
)
_SAMGA_CONFIG_RELATIVE = Path(
    "experiments/samga_brain_rw/configs/internvit_baseline_v1.json"
)
_BRAINRW_CONFIG_RELATIVE = Path(
    "experiments/samga_brain_rw/configs/brainrw_clip_lora_v1.json"
)
_FEATURE_CACHE_RELATIVE = Path(
    "artifacts/samga_reproduction/features/internvit_v2_5_multi_variant/"
    "variants/train_idx0_patch_mean/features.npy"
)
_MANIFEST_ROOT_RELATIVE = Path("artifacts/samga_brain_rw/protocol/manifests")
_RUNNER_RELATIVE = Path(
    "experiments/samga_brain_rw/scripts/run_stage1_expansion_cell.py"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMPLETION_SCHEMA = {
    "schema_version": 1,
    "payload_type": "samga_brain_rw.stage1_expansion_completion",
    "required_output_hashes": [
        "component_record_sha256",
        "final_checkpoint_sha256",
        "run_manifest_sha256",
    ],
}


@dataclass(frozen=True)
class ExpansionIdentity:
    """Resolved identity for one train-only component cell."""

    component: str
    config_id: str
    config_sha256: str
    input_bundle_sha256: str
    run_key: str


def _require_sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _absolute_normalized(path: Path, context: str) -> Path:
    raw = os.fspath(path)
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise ValueError(f"{context} must be a non-empty path")
    absolute = Path(os.path.abspath(os.path.normpath(raw)))
    if Path(raw).is_absolute() and Path(raw) != absolute:
        raise ValueError(f"{context} must be normalized")
    return br.reject_development_path(absolute, context)


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


def _canonical_object(
    path: Path,
    context: str,
    *,
    require_canonical: bool = True,
) -> dict[str, object]:
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context} cannot be read") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{context} must contain an object")
    canonical = canonical_json_bytes(value)
    if require_canonical and raw not in (
        canonical,
        canonical + b"\n",
    ):
        raise ValueError(f"{context} must use canonical JSON")
    return value


def _declared_clip_path(config_path: Path) -> Path:
    payload = _canonical_object(
        config_path,
        "BrainRW semantic config",
        require_canonical=False,
    )
    clip = payload.get("clip")
    if not isinstance(clip, Mapping):
        raise ValueError("BrainRW semantic config lacks the clip object")
    declared = clip.get("path")
    if not isinstance(declared, str) or not declared:
        raise ValueError("BrainRW semantic config clip.path is invalid")
    path = Path(declared)
    normalized = Path(os.path.abspath(os.path.normpath(declared)))
    if not path.is_absolute() or path != normalized:
        raise ValueError(
            "BrainRW semantic config clip.path must be absolute and normalized"
        )
    return br.reject_development_path(normalized, "declared CLIP path")

@lru_cache(maxsize=None)
def _manifest_identity(
    manifest_path: Path,
    subject: int,
) -> br.ManifestIdentity:
    return br.load_development_manifest_identity(
        manifest_path,
        expected_subject=subject,
    )


@lru_cache(maxsize=None)
def _samga_preflight(config_path: Path) -> object:
    import train as samga_train

    return samga_train.preflight_upstream_config(config_path)


@lru_cache(maxsize=None)
def _samga_static_identity(
    config_path: Path,
    feature_cache: Path,
    manifest_path: Path,
    subject: int,
) -> tuple[object, object, Mapping[str, str]]:
    import train as samga_train

    manifest = _manifest_identity(manifest_path, subject)
    config = samga_train._load_training_config(
        config_path,
        feature_cache,
        manifest=manifest,
        preflight=_samga_preflight(config_path),
    )
    input_hashes = samga_train._build_input_hashes(
        manifest,
        config,
        validation_scope="none",
    )
    return manifest, config, input_hashes


@lru_cache(maxsize=None)
def _brainrw_config_identity(
    config_path: Path,
    clip_path: Path,
) -> br.BrainRWConfigIdentity:
    return br.verify_brainrw_config(config_path, clip_path)



def _resolve_samga_cell(
    *,
    config_path: Path,
    feature_cache: Path,
    manifest_path: Path,
    subject: int,
    seed: int,
    environment_binding: Mapping[str, object],
) -> ExpansionIdentity:
    import train as samga_train

    manifest, config, input_hashes = _samga_static_identity(
        config_path,
        feature_cache,
        manifest_path,
        subject,
    )
    config_id = config.payload.get("config_id")
    if not isinstance(config_id, str) or not config_id:
        raise ValueError("SAMGA config_id is invalid")
    candidate = samga_train.build_resolved_candidate_payload(
        stage=0,
        config_id=config_id,
        subject=subject,
        seed=seed,
        baseline_config_sha256=config.semantic.sha256,
        stage2_config_sha256=None,
        layernorm_config_id="s2-layernorm-off",
        whitening_config_id="s2-whitening-off",
        preprojector_config_id="s2-preproj-shared",
        adapter_kind="identity",
        adapter_rank=None,
        adapter_lr_ratio=None,
        whitening_payload_sha256=None,
        environment_binding=environment_binding,
        validation_scope="none",
    )
    resolved = samga_train.resolve_run_config(
        config.protocol,
        candidate,
        input_hashes,
    )
    return ExpansionIdentity(
        component="internvit",
        config_id=config_id,
        config_sha256=resolved.semantic_config_sha256,
        input_bundle_sha256=resolved.input_bundle_sha256,
        run_key=resolved.run_key,
    )


def _resolve_brainrw_cell(
    *,
    config_path: Path,
    clip_path: Path,
    manifest_path: Path,
    subject: int,
    seed: int,
    semantic_environment_sha256: str,
) -> ExpansionIdentity:
    config = _brainrw_config_identity(config_path, clip_path)
    manifest = _manifest_identity(manifest_path, subject)
    run_key, input_bundle_sha256, _ = br.brainrw_run_key(
        config,
        manifest,
        subject,
        seed,
        semantic_environment_sha256,
        "none",
    )
    config_id = config.payload.get("config_id")
    if not isinstance(config_id, str) or not config_id:
        raise ValueError("BrainRW config_id is invalid")
    return ExpansionIdentity(
        component="brainrw",
        config_id=config_id,
        config_sha256=config.sha256,
        input_bundle_sha256=input_bundle_sha256,
        run_key=run_key,
    )


def _resource(partition: str) -> dict[str, object]:
    if partition not in _ALLOWED_PARTITIONS:
        raise ValueError("partition is outside the sealed A40 allowlist")
    return {
        "partition": partition,
        "gres": "gpu:a40:1",
        "cpus": 8,
        "memory": "64G",
        "time": "04:00:00",
        "stdout_path": "logs/samga_brain_rw/stage1_expansion_%A_%a.out",
        "stderr_path": "logs/samga_brain_rw/stage1_expansion_%A_%a.err",
    }


def build_stage1_expansion_rows(
    *,
    project_root: Path,
    partition: str,
    semantic_environment_sha256: str,
    environment_binding: Mapping[str, object],
    locked_survivor_sha256: str,
) -> list[dict[str, object]]:
    """Return the exact 100-row Stage 1 train-only preparation topology."""

    root = _absolute_normalized(project_root, "project root")
    if not root.is_dir():
        raise ValueError("project root must be an existing directory")
    semantic_sha = _require_sha256(
        semantic_environment_sha256,
        "semantic environment",
    )
    locked_sha = _require_sha256(
        locked_survivor_sha256,
        "locked survivor",
    )
    resource = _resource(partition)
    samga_config = root / _SAMGA_CONFIG_RELATIVE
    brainrw_config = root / _BRAINRW_CONFIG_RELATIVE
    feature_cache = root / _FEATURE_CACHE_RELATIVE
    clip_path = _declared_clip_path(brainrw_config)
    runner = root / _RUNNER_RELATIVE
    rows: list[dict[str, object]] = []
    for component in ("internvit", "brainrw"):
        for subject in range(1, 11):
            manifest_path = (
                root
                / _MANIFEST_ROOT_RELATIVE
                / f"sub-{subject:02d}_protocol.json"
            )
            for seed in range(42, 47):
                if component == "internvit":
                    identity = _resolve_samga_cell(
                        config_path=samga_config,
                        feature_cache=feature_cache,
                        manifest_path=manifest_path,
                        subject=subject,
                        seed=seed,
                        environment_binding=environment_binding,
                    )
                    config_path = samga_config
                    extra = ["--feature-cache", str(feature_cache)]
                    role = "internvit-component"
                else:
                    identity = _resolve_brainrw_cell(
                        config_path=brainrw_config,
                        clip_path=clip_path,
                        manifest_path=manifest_path,
                        subject=subject,
                        seed=seed,
                        semantic_environment_sha256=semantic_sha,
                    )
                    config_path = brainrw_config
                    extra = [
                        "--clip-path",
                        str(clip_path),
                        "--expected-semantic-environment-sha256",
                        semantic_sha,
                    ]
                    role = "brainrw-component"
                if identity.component != component:
                    raise ValueError("resolved component identity mismatch")
                output_dir = (
                    root
                    / "artifacts/samga_brain_rw"
                    / _STAGE
                    / component
                    / identity.run_key
                )
                argv = [
                    "python",
                    str(runner),
                    "--component",
                    component,
                    "--validation-scope",
                    "none",
                    "--subject",
                    str(subject),
                    "--seed",
                    str(seed),
                    "--resume",
                    "none",
                    "--config",
                    str(config_path),
                    "--manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(output_dir),
                    "--project-root",
                    str(root),
                    "--config-id",
                    identity.config_id,
                    "--expected-config-sha256",
                    identity.config_sha256,
                    "--expected-input-bundle-sha256",
                    identity.input_bundle_sha256,
                    "--locked-survivor-sha256",
                    locked_sha,
                    "--run-key",
                    identity.run_key,
                    "--device",
                    "cuda",
                    *extra,
                ]
                rows.append(
                    {
                        "stage": _STAGE,
                        "role": role,
                        "config_id": identity.config_id,
                        "config_sha256": identity.config_sha256,
                        "input_bundle_sha256": identity.input_bundle_sha256,
                        "run_key": identity.run_key,
                        "subject": subject,
                        "seed": seed,
                        "argv": argv,
                        **resource,
                        "completion_path": str(
                            output_dir / "completion.json"
                        ),
                        "expected_completion_schema": dict(
                            _COMPLETION_SCHEMA
                        ),
                    }
                )
    return rows


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument(
        "--partition",
        required=True,
        choices=sorted(_ALLOWED_PARTITIONS),
    )
    parser.add_argument(
        "--environment-run-manifest",
        required=True,
        type=Path,
    )
    parser.add_argument("--locked-survivor", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    environment_manifest = _canonical_object(
        arguments.environment_run_manifest,
        "environment run manifest",
    )
    binding = validate_environment_binding(
        environment_manifest.get("environment")
    )
    semantic_sha = str(binding["semantic_environment_sha256"])
    locked_survivor = _canonical_object(
        arguments.locked_survivor,
        "locked survivor",
    )
    if locked_survivor.get("artifact_type") != (
        "samga_brain_rw.stage1_locked_survivor"
    ):
        raise ValueError("locked survivor artifact type mismatch")
    locked_sha = _file_sha256(arguments.locked_survivor)
    rows = build_stage1_expansion_rows(
        project_root=arguments.project_root,
        partition=arguments.partition,
        semantic_environment_sha256=semantic_sha,
        environment_binding=binding,
        locked_survivor_sha256=locked_sha,
    )
    root = _absolute_normalized(arguments.project_root, "project root")
    stage_root = root / "artifacts/samga_brain_rw" / _STAGE
    for component in ("internvit", "brainrw"):
        (stage_root / component).mkdir(parents=True, exist_ok=True)
    output = _absolute_normalized(arguments.output, "Stage 1 expansion job map")
    expected_root = root / "artifacts/samga_brain_rw/job_maps"
    try:
        output.relative_to(expected_root)
    except ValueError as exc:
        raise ValueError(
            "Stage 1 expansion job map must remain below the job-map root"
        ) from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = job_maps.write_job_map(rows, output)
    print(payload["payload_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
