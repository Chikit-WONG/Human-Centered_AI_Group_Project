#!/usr/bin/env python3
"""Build the sealed Stage 1 BrainRW smoke or 3x2 pilot job map."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import build_job_map as job_maps
from samga_brain_rw import brainrw as br

_CONFIG_RELATIVE = Path(
    "experiments/samga_brain_rw/configs/brainrw_clip_lora_v1.json"
)
_MANIFEST_ROOT_RELATIVE = Path(
    "artifacts/samga_brain_rw/protocol/manifests"
)
_RUNNER_RELATIVE = Path(
    "experiments/samga_brain_rw/scripts/run_brainrw_cell.py"
)
_CONFIG_ID = "brainrw_clip_lora_v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PHASES = frozenset({"smoke", "pilot"})
_SMOKE_HASHES = [
    "final_checkpoint_sha256",
    "in_loop_metadata_sha256",
    "run_manifest_sha256",
]
_FULL_HASHES = [
    "final_checkpoint_sha256",
    "run_manifest_sha256",
    "score_envelope_sha256",
    "score_payload_sha256",
]


def _absolute_normalized(path: Path, context: str) -> Path:
    raw = os.fspath(path)
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise ValueError(f"{context} must be a non-empty path")
    absolute = Path(os.path.abspath(os.path.normpath(raw)))
    if Path(raw).is_absolute() and Path(raw) != absolute:
        raise ValueError(f"{context} must be normalized")
    return br.reject_development_path(absolute, context)


def _completion_schema(phase: str) -> dict[str, object]:
    if phase == "smoke":
        payload_type = "samga_brain_rw.brainrw_smoke_completion"
        hashes = _SMOKE_HASHES
    else:
        payload_type = "samga_brain_rw.brainrw_full_completion"
        hashes = _FULL_HASHES
    return {
        "schema_version": 1,
        "payload_type": payload_type,
        "required_output_hashes": list(hashes),
    }


def _resource(phase: str) -> dict[str, object]:
    return {
        "partition": "debug" if phase == "smoke" else "i64m1tga40u",
        "gres": "gpu:a40:1",
        "cpus": 8,
        "memory": "64G",
        "time": "00:30:00" if phase == "smoke" else "02:00:00",
        "stdout_path": (
            "logs/samga_brain_rw/stage1_brainrw_%A_%a.out"
        ),
        "stderr_path": (
            "logs/samga_brain_rw/stage1_brainrw_%A_%a.err"
        ),
    }


def _config_id(config: object) -> str:
    payload = getattr(config, "payload", None)
    if not isinstance(payload, Mapping):
        raise ValueError("verified BrainRW config payload is invalid")
    value = payload.get("config_id")
    if value != _CONFIG_ID:
        raise ValueError("verified BrainRW config ID differs from Stage 1")
    return value


def _strict_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate config JSON key: {key}")
        value[key] = item
    return value


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"non-finite config JSON value is forbidden: {value}")


def _declared_clip_path(config_path: Path) -> Path:
    path = br.reject_development_path(
        config_path,
        "BrainRW semantic config",
    )
    try:
        raw = path.read_bytes()
        payload = json.loads(
            raw,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("BrainRW semantic config cannot be read") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("BrainRW semantic config must contain an object")
    clip = payload.get("clip")
    if not isinstance(clip, Mapping):
        raise ValueError("BrainRW semantic config lacks the clip object")
    declared = clip.get("path")
    if not isinstance(declared, str) or not declared:
        raise ValueError("BrainRW semantic config clip.path is invalid")
    pure = Path(declared)
    normalized = Path(os.path.abspath(os.path.normpath(declared)))
    if not pure.is_absolute() or pure != normalized:
        raise ValueError(
            "BrainRW semantic config clip.path must be absolute and normalized"
        )
    return br.reject_development_path(
        normalized,
        "BrainRW semantic config CLIP path",
    )


def build_stage1_brainrw_rows(
    *,
    project_root: Path,
    phase: str,
    semantic_environment_sha256: str,
) -> list[dict[str, object]]:
    """Return the exact sealed raw rows for one Stage 1 BrainRW phase."""

    if phase not in _PHASES:
        raise ValueError("phase must be smoke or pilot")
    if (
        not isinstance(semantic_environment_sha256, str)
        or _SHA256_RE.fullmatch(semantic_environment_sha256) is None
    ):
        raise ValueError(
            "semantic environment must be a lowercase SHA-256 digest"
        )
    root = _absolute_normalized(project_root, "project root")
    if not root.is_dir():
        raise ValueError("project root must be an existing directory")
    config_path = root / _CONFIG_RELATIVE
    declared_clip_path = _declared_clip_path(config_path)
    config = br.verify_brainrw_config(
        config_path,
        declared_clip_path,
    )
    if Path(config.clip_path) != declared_clip_path:
        raise ValueError(
            "verified CLIP path drifted from the semantic config"
        )
    config_id = _config_id(config)
    config_sha256 = str(config.sha256)
    if _SHA256_RE.fullmatch(config_sha256) is None:
        raise ValueError("verified BrainRW config SHA-256 is invalid")
    stage = f"stage-1-brainrw-{phase}"
    cells = (
        ((8, 42),)
        if phase == "smoke"
        else tuple(
            (subject, seed)
            for subject in (1, 5, 8)
            for seed in (42, 43)
        )
    )
    mode = "smoke" if phase == "smoke" else "full"
    resource = _resource(phase)
    rows: list[dict[str, object]] = []
    for subject, seed in cells:
        manifest_path = (
            root
            / _MANIFEST_ROOT_RELATIVE
            / f"sub-{subject:02d}_protocol.json"
        )
        manifest = br.load_development_manifest_identity(
            manifest_path,
            expected_subject=subject,
        )
        run_key, input_bundle_sha256, _ = br.brainrw_run_key(
            config,
            manifest,
            subject,
            seed,
            semantic_environment_sha256,
        )
        output_dir = (
            root
            / "artifacts/samga_brain_rw"
            / stage
            / run_key
        )
        argv = [
            "python",
            str(root / _RUNNER_RELATIVE),
            "--mode",
            mode,
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
            "--clip-path",
            str(declared_clip_path),
            "--output-dir",
            str(output_dir),
            "--project-root",
            str(root),
            "--config-id",
            config_id,
            "--expected-config-sha256",
            config_sha256,
            "--expected-input-bundle-sha256",
            input_bundle_sha256,
            "--expected-semantic-environment-sha256",
            semantic_environment_sha256,
            "--run-key",
            run_key,
            "--device",
            "cuda",
        ]
        if phase == "smoke":
            argv.extend(["--max-train-steps", "1"])
        rows.append(
            {
                "stage": stage,
                "role": "clip-branch",
                "config_id": config_id,
                "config_sha256": config_sha256,
                "input_bundle_sha256": input_bundle_sha256,
                "run_key": run_key,
                "subject": subject,
                "seed": seed,
                "argv": argv,
                **resource,
                "completion_path": str(output_dir / "completion.json"),
                "expected_completion_schema": _completion_schema(phase),
            }
        )
    return rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=sorted(_PHASES))
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument(
        "--semantic-environment-sha256",
        required=True,
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    rows = build_stage1_brainrw_rows(
        project_root=arguments.project_root,
        phase=arguments.phase,
        semantic_environment_sha256=(
            arguments.semantic_environment_sha256
        ),
    )
    root = _absolute_normalized(arguments.project_root, "project root")
    stage_root = br.reject_development_path(
        root
        / "artifacts/samga_brain_rw"
        / f"stage-1-brainrw-{arguments.phase}",
        "Stage 1 BrainRW output root",
    )
    stage_root.mkdir(parents=True, exist_ok=True)
    output = br.reject_development_path(
        arguments.output,
        "Stage 1 BrainRW job map",
    )
    expected_map_root = root / "artifacts/samga_brain_rw/job_maps"
    try:
        output.relative_to(expected_map_root)
    except ValueError as exc:
        raise ValueError(
            "Stage 1 BrainRW job map must remain below the job-map root"
        ) from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = job_maps.write_job_map(rows, output)
    print(payload["payload_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
