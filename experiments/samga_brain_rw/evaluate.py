#!/usr/bin/env python3
"""Development-only SAMGA checkpoint evaluator."""

from __future__ import annotations

import argparse
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from samga_brain_rw.brainrw import (
    ManifestIdentity,
    load_development_manifest_identity,
    reject_development_path,
)
from samga_brain_rw.config import SemanticConfig, make_run_key
from samga_brain_rw.data import POSTERIOR_CHANNELS, ProtocolSubjectDataset
from samga_brain_rw.feature_transforms import TrainWhitening
from samga_brain_rw.hashing import sha256_json
from samga_brain_rw.scores import ScoreArtifact
from samga_brain_rw.trainer import (
    SAMGARuntimeModel,
    TrainingCellSpec,
    evaluate_development_model,
)
from samga_brain_rw.upstream_samga import load_locked_upstream_components
from train import (
    PINNED_UPSTREAM_SHA,
    RUN_PAYLOAD_TYPE,
    SCHEDULE_SHA256,
    load_samga_checkpoint,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_CANDIDATE_KEYS = frozenset(
    {
        "schema_version",
        "config_id",
        "stage",
        "subject",
        "seed",
        "baseline_config_sha256",
        "stage2_config_sha256",
        "semantic_config_sha256",
        "input_bundle_sha256",
        "run_key",
        "layernorm_config_id",
        "whitening_config_id",
        "preprojector_config_id",
        "adapter_kind",
        "adapter_rank",
        "adapter_lr_ratio",
        "whitening_payload",
        "full_task_initialization_sha256",
        "shared_parameter_intersection_name",
        "shared_parameter_intersection_sha256",
        "architecture_specific_initialization_sha256",
        "data_order_sha256",
        "trajectory_sha256",
        "candidate_spec_sha256",
    }
)
_RUN_MANIFEST_KEYS = frozenset(
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", required=True, choices=("val-dev",))
    parser.add_argument(
        "--subject",
        required=True,
        type=int,
        choices=range(1, 11),
    )
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--feature-cache", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    return parser


def parse_arguments(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.seed < 0:
        parser.error("--seed must be non-negative")
    return arguments


def checkpoint_identity(
    payload: Mapping[str, object],
    *,
    subject: int,
    seed: int,
) -> tuple[int, int]:
    recorded_subject = payload.get("subject")
    recorded_seed = payload.get("seed")
    if recorded_subject != subject:
        raise ValueError("checkpoint subject mismatch")
    if recorded_seed != seed:
        raise ValueError("checkpoint seed mismatch")
    return subject, seed


@dataclass(frozen=True)
class EvaluationPaths:
    config: Path
    manifest: Path
    feature_cache: Path
    checkpoint: Path
    output_dir: Path


@dataclass(frozen=True)
class EvaluationConfig:
    semantic: SemanticConfig
    payload: Mapping[str, object]
    upstream_root: Path
    upstream_commit: str
    cache_sha256: str
    model_sha256: str
    batch_size: int


@dataclass(frozen=True)
class CandidateIdentity:
    payload: Mapping[str, object]
    stage: str
    stage_number: int
    config_id: str
    whitening: TrainWhitening | None


@dataclass(frozen=True)
class EvaluationIdentity:
    checkpoint_payload: Mapping[str, object]
    checkpoint_sha256: str
    candidate: CandidateIdentity
    run_manifest: Mapping[str, object]
    input_hashes: Mapping[str, str]


def _require_mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise ValueError(f"{context} must be a string-keyed mapping")
    return dict(value)


def _require_exact_keys(
    value: Mapping[str, object],
    expected: frozenset[str],
    context: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{context} keys differ from the locked schema; "
            f"missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _require_sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _require_identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a safe non-empty identifier")
    return value


def _guard_paths(arguments: argparse.Namespace) -> EvaluationPaths:
    paths = EvaluationPaths(
        config=reject_development_path(
            arguments.config,
            "SAMGA evaluation config",
        ),
        manifest=reject_development_path(
            arguments.manifest,
            "SAMGA evaluation protocol manifest",
        ),
        feature_cache=reject_development_path(
            arguments.feature_cache,
            "SAMGA evaluation feature cache",
        ),
        checkpoint=reject_development_path(
            arguments.checkpoint,
            "SAMGA evaluation checkpoint",
        ),
        output_dir=reject_development_path(
            arguments.output_dir,
            "SAMGA evaluation output",
        ),
    )
    if paths.output_dir.exists():
        raise FileExistsError(
            f"SAMGA evaluation output already exists: {paths.output_dir}"
        )
    return paths


def _declared_path(value: object, context: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{context} must be a non-empty text path")
    path = Path(value)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return reject_development_path(path, context)


def _load_config(path: Path, feature_cache: Path) -> EvaluationConfig:
    semantic = SemanticConfig.from_path(path)
    payload = semantic.canonical_payload()
    if payload.get("config_type") != "internvit_baseline":
        raise ValueError(
            "SAMGA evaluation requires an internvit_baseline config"
        )
    upstream = _require_mapping(payload.get("upstream"), "config upstream")
    model = _require_mapping(payload.get("model"), "config model")
    cache = _require_mapping(payload.get("cache"), "config cache")
    task = _require_mapping(payload.get("task"), "config task")

    upstream_root = _declared_path(
        upstream.get("path"),
        "configured upstream SAMGA root",
    )
    upstream_commit = upstream.get("git_commit")
    if upstream_commit != PINNED_UPSTREAM_SHA:
        raise ValueError("configured upstream SAMGA revision mismatch")
    _declared_path(model.get("path"), "configured InternViT model")
    configured_cache = _declared_path(
        cache.get("path"),
        "configured feature cache",
    )
    if configured_cache != feature_cache:
        raise ValueError(
            "CLI feature cache path differs from the semantic config"
        )
    cache_sha256 = _require_sha256(
        cache.get("sha256"),
        "config cache.sha256",
    )
    channels = task.get("channels")
    if not isinstance(channels, list) or tuple(channels) != POSTERIOR_CHANNELS:
        raise ValueError(
            "config channels differ from the locked posterior-channel order"
        )
    if task.get("force_global") is not True:
        raise ValueError("config must lock force_global=true")
    batch_size = task.get("batch_size")
    if type(batch_size) is not int or batch_size != 512:
        raise ValueError("config batch_size must be locked to 512")
    return EvaluationConfig(
        semantic=semantic,
        payload=payload,
        upstream_root=upstream_root,
        upstream_commit=upstream_commit,
        cache_sha256=cache_sha256,
        model_sha256=sha256_json(model),
        batch_size=batch_size,
    )


def _input_hashes(
    value: object,
    *,
    manifest: ManifestIdentity,
    config: EvaluationConfig,
) -> dict[str, str]:
    raw = _require_mapping(value, "checkpoint input_hashes")
    if not raw:
        raise ValueError("checkpoint input_hashes must not be empty")
    normalized: dict[str, str] = {}
    for key, digest in raw.items():
        _require_identifier(key, "checkpoint input_hashes key")
        normalized[key] = _require_sha256(
            digest,
            f"checkpoint input_hashes.{key}",
        )
    required = {
        "cache_sha256": config.cache_sha256,
        "manifest_sha256": manifest.manifest_sha256,
        "model_sha256": config.model_sha256,
        "protocol_sha256": manifest.protocol_sha256,
    }
    for key, expected in required.items():
        if normalized.get(key) != expected:
            raise ValueError(f"checkpoint input_hashes.{key} mismatch")
    return dict(sorted(normalized.items()))


def _candidate_identity(
    value: object,
    *,
    subject: int,
    seed: int,
    config: EvaluationConfig,
    checkpoint_payload: Mapping[str, object],
    input_hashes: Mapping[str, str],
) -> CandidateIdentity:
    candidate = _require_mapping(value, "candidate_spec")
    _require_exact_keys(candidate, _CANDIDATE_KEYS, "candidate_spec")
    if candidate["schema_version"] != 1:
        raise ValueError("candidate_spec schema_version must be 1")
    body = {
        key: candidate[key]
        for key in _CANDIDATE_KEYS
        if key != "candidate_spec_sha256"
    }
    candidate_sha256 = _require_sha256(
        candidate["candidate_spec_sha256"],
        "candidate_spec_sha256",
    )
    if sha256_json(body) != candidate_sha256:
        raise ValueError("candidate_spec SHA-256 mismatch")

    stage = candidate["stage"]
    if stage not in ("stage0", "stage2"):
        raise ValueError("candidate_spec stage must be stage0 or stage2")
    stage_number = 0 if stage == "stage0" else 2
    config_id = _require_identifier(
        candidate["config_id"],
        "candidate_spec config_id",
    )
    if candidate["subject"] != subject:
        raise ValueError("candidate_spec subject mismatch")
    if candidate["seed"] != seed:
        raise ValueError("candidate_spec seed mismatch")
    if (
        candidate["baseline_config_sha256"]
        != config.semantic.sha256
    ):
        raise ValueError("candidate_spec baseline config mismatch")
    if stage == "stage0":
        if candidate["stage2_config_sha256"] is not None:
            raise ValueError("Stage 0 cannot bind a Stage 2 config")
        if config_id != config.payload.get("config_id"):
            raise ValueError("Stage 0 candidate config_id mismatch")
    else:
        _require_sha256(
            candidate["stage2_config_sha256"],
            "candidate_spec stage2_config_sha256",
        )

    checkpoint_config = _require_sha256(
        checkpoint_payload.get("config_sha256"),
        "checkpoint config_sha256",
    )
    if candidate["semantic_config_sha256"] != checkpoint_config:
        raise ValueError("candidate_spec semantic config mismatch")
    normalized_inputs = dict(sorted(input_hashes.items()))
    input_bundle_sha256 = sha256_json(normalized_inputs)
    if candidate["input_bundle_sha256"] != input_bundle_sha256:
        raise ValueError("candidate_spec input bundle mismatch")
    expected_run_key = make_run_key(
        stage,
        config_id,
        subject,
        seed,
        checkpoint_config,
        input_bundle_sha256,
    )
    if candidate["run_key"] != expected_run_key:
        raise ValueError("candidate_spec run_key mismatch")

    layernorm = candidate["layernorm_config_id"]
    whitening_id = candidate["whitening_config_id"]
    preprojector = candidate["preprojector_config_id"]
    adapter_kind = candidate["adapter_kind"]
    if layernorm not in ("s2-layernorm-off", "s2-layernorm-on"):
        raise ValueError("candidate_spec layernorm_config_id is unknown")
    if whitening_id not in ("s2-whitening-off", "s2-whitening-on"):
        raise ValueError("candidate_spec whitening_config_id is unknown")
    if preprojector not in ("s2-preproj-shared", "s2-preproj-separate"):
        raise ValueError("candidate_spec preprojector_config_id is unknown")
    if adapter_kind not in (
        "identity",
        "adapter",
        "global_dense",
        "matched_projector",
    ):
        raise ValueError("candidate_spec adapter_kind is unknown")
    active = (
        int(layernorm == "s2-layernorm-on")
        + int(whitening_id == "s2-whitening-on")
        + int(preprojector == "s2-preproj-separate")
        + int(adapter_kind != "identity")
    )
    if (stage == "stage0" and active != 0) or active > 1:
        raise ValueError(
            "candidate_spec violates the one-factor-only policy"
        )
    rank = candidate["adapter_rank"]
    ratio = candidate["adapter_lr_ratio"]
    if adapter_kind == "identity":
        if rank is not None or ratio is not None:
            raise ValueError(
                "identity candidate cannot set adapter rank or LR ratio"
            )
    elif (
        type(rank) is not int
        or rank not in (8, 16, 32)
        or type(ratio) is not float
        or ratio not in (0.05, 0.1)
    ):
        raise ValueError(
            "adapter candidate rank/LR ratio differs from the registry"
        )

    whitening_payload = candidate["whitening_payload"]
    if whitening_id == "s2-whitening-on":
        if not isinstance(whitening_payload, Mapping):
            raise ValueError(
                "whitening-on requires a sealed TrainWhitening payload"
            )
        whitening = TrainWhitening.from_payload(whitening_payload)
    else:
        if whitening_payload is not None:
            raise ValueError(
                "whitening-off forbids a whitening payload"
            )
        whitening = None

    for key in (
        "full_task_initialization_sha256",
        "shared_parameter_intersection_sha256",
        "architecture_specific_initialization_sha256",
        "data_order_sha256",
        "trajectory_sha256",
    ):
        _require_sha256(candidate[key], f"candidate_spec {key}")
    _require_identifier(
        candidate["shared_parameter_intersection_name"],
        "candidate_spec shared_parameter_intersection_name",
    )
    if candidate["data_order_sha256"] != checkpoint_payload.get(
        "data_order_sha256"
    ):
        raise ValueError("candidate_spec data-order mismatch")
    if candidate["trajectory_sha256"] != checkpoint_payload.get(
        "trajectory_sha256"
    ):
        raise ValueError("candidate_spec trajectory mismatch")
    return CandidateIdentity(
        payload=candidate,
        stage=stage,
        stage_number=stage_number,
        config_id=config_id,
        whitening=whitening,
    )


def _run_manifest_identity(
    value: object,
    *,
    candidate: CandidateIdentity,
    checkpoint_payload: Mapping[str, object],
    manifest: ManifestIdentity,
    config: EvaluationConfig,
    subject: int,
    seed: int,
) -> dict[str, object]:
    run = _require_mapping(value, "run_manifest")
    _require_exact_keys(run, _RUN_MANIFEST_KEYS, "run_manifest")
    if run["schema_version"] != 1 or run["payload_type"] != RUN_PAYLOAD_TYPE:
        raise ValueError("run_manifest schema/payload identity mismatch")
    body = {
        key: run[key]
        for key in _RUN_MANIFEST_KEYS
        if key != "run_manifest_sha256"
    }
    if sha256_json(body) != _require_sha256(
        run["run_manifest_sha256"],
        "run_manifest_sha256",
    ):
        raise ValueError("run_manifest SHA-256 mismatch")
    expected = {
        "stage": candidate.stage_number,
        "subject": subject,
        "seed": seed,
        "config_id": candidate.config_id,
        "config_sha256": checkpoint_payload["config_sha256"],
        "protocol_sha256": manifest.protocol_sha256,
        "cache_sha256": config.cache_sha256,
        "upstream_sha": PINNED_UPSTREAM_SHA,
        "data_order_sha256": checkpoint_payload["data_order_sha256"],
        "candidate_spec_sha256": candidate.payload[
            "candidate_spec_sha256"
        ],
        "run_key": candidate.payload["run_key"],
    }
    for key, expected_value in expected.items():
        if run[key] != expected_value:
            raise ValueError(f"run_manifest {key} mismatch")
    git_sha = run["git_sha"]
    if not isinstance(git_sha, str) or _GIT_SHA_RE.fullmatch(git_sha) is None:
        raise ValueError("run_manifest git_sha must be lowercase 40-hex")
    return run


def _evaluation_identity(
    loaded_checkpoint: object,
    *,
    manifest: ManifestIdentity,
    config: EvaluationConfig,
    subject: int,
    seed: int,
) -> EvaluationIdentity:
    payload = _require_mapping(
        getattr(loaded_checkpoint, "payload", None),
        "loaded checkpoint payload",
    )
    checkpoint_sha256 = _require_sha256(
        getattr(loaded_checkpoint, "sha256", None),
        "loaded checkpoint SHA-256",
    )
    checkpoint_identity(payload, subject=subject, seed=seed)
    if payload.get("schedule_sha256") != SCHEDULE_SHA256:
        raise ValueError("checkpoint schedule mismatch")
    input_hashes = _input_hashes(
        payload.get("input_hashes"),
        manifest=manifest,
        config=config,
    )
    candidate = _candidate_identity(
        payload.get("candidate_spec"),
        subject=subject,
        seed=seed,
        config=config,
        checkpoint_payload=payload,
        input_hashes=input_hashes,
    )
    run_manifest = _run_manifest_identity(
        payload.get("run_manifest"),
        candidate=candidate,
        checkpoint_payload=payload,
        manifest=manifest,
        config=config,
        subject=subject,
        seed=seed,
    )
    return EvaluationIdentity(
        checkpoint_payload=payload,
        checkpoint_sha256=checkpoint_sha256,
        candidate=candidate,
        run_manifest=run_manifest,
        input_hashes=input_hashes,
    )


def _verify_dataset(
    dataset: object,
    *,
    manifest: ManifestIdentity,
    cache_sha256: str,
    subject: int,
) -> None:
    if getattr(dataset, "scope", None) != "val-dev":
        raise PermissionError("evaluation dataset must have val-dev scope")
    if getattr(dataset, "subject_id", None) != subject:
        raise ValueError("evaluation dataset subject mismatch")
    query_ids = tuple(getattr(dataset, "query_ids", ()))
    gallery_ids = tuple(getattr(dataset, "gallery_ids", ()))
    if (
        query_ids != manifest.val_dev_ordered_ids
        or gallery_ids != manifest.val_dev_ordered_ids
    ):
        raise ValueError("evaluation dataset ordered IDs mismatch")
    metadata = _require_mapping(
        getattr(dataset, "feature_cache_metadata", None),
        "feature cache metadata",
    )
    actual_cache_sha256 = metadata.get(
        "feature_sha256",
        metadata.get("cache_sha256"),
    )
    if actual_cache_sha256 != cache_sha256:
        raise ValueError("verified feature cache SHA-256 mismatch")


def _unused_checkpoint_operation(*args: object, **kwargs: object) -> object:
    raise RuntimeError("evaluation must never invoke checkpoint operations")


def _build_model(
    identity: EvaluationIdentity,
    *,
    manifest: ManifestIdentity,
    config: EvaluationConfig,
    paths: EvaluationPaths,
    subject: int,
    seed: int,
    device: str,
) -> SAMGARuntimeModel:
    components = load_locked_upstream_components(
        config.upstream_root,
        config.upstream_commit,
    )
    candidate = identity.candidate
    spec = TrainingCellSpec(
        components=components,
        manifest_path=manifest.path,
        feature_cache=paths.feature_cache,
        stage=candidate.stage_number,
        subject=subject,
        seed=seed,
        config_sha256=identity.checkpoint_payload["config_sha256"],
        schedule_sha256=identity.checkpoint_payload["schedule_sha256"],
        trajectory_sha256=identity.checkpoint_payload[
            "trajectory_sha256"
        ],
        data_order_sha256=identity.checkpoint_payload[
            "data_order_sha256"
        ],
        input_hashes=identity.input_hashes,
        run_manifest=identity.run_manifest,
        candidate_spec=candidate.payload,
        checkpoint_builder=_unused_checkpoint_operation,
        checkpoint_restorer=_unused_checkpoint_operation,
        checkpoint_sink=_unused_checkpoint_operation,
        batch_size=config.batch_size,
        num_workers=0,
        device=device,
        layernorm_config_id=candidate.payload["layernorm_config_id"],
        whitening_config_id=candidate.payload["whitening_config_id"],
        preprojector_config_id=candidate.payload[
            "preprojector_config_id"
        ],
        adapter_kind=candidate.payload["adapter_kind"],
        adapter_rank=candidate.payload["adapter_rank"],
        adapter_lr_ratio=candidate.payload["adapter_lr_ratio"],
        whitening=candidate.whitening,
    )
    model = SAMGARuntimeModel(spec)
    model.load_state_dict(
        identity.checkpoint_payload["model_state_dict"],
        strict=True,
    )
    return model


def run_evaluation(arguments: argparse.Namespace) -> int:
    paths = _guard_paths(arguments)
    manifest = load_development_manifest_identity(
        paths.manifest,
        expected_subject=arguments.subject,
    )
    if manifest.path != paths.manifest:
        raise ValueError("verified manifest path differs from the CLI path")
    config = _load_config(paths.config, paths.feature_cache)
    loaded_checkpoint = load_samga_checkpoint(
        paths.checkpoint,
        requested_scope="train",
    )
    identity = _evaluation_identity(
        loaded_checkpoint,
        manifest=manifest,
        config=config,
        subject=arguments.subject,
        seed=arguments.seed,
    )

    dataset = ProtocolSubjectDataset(
        manifest_path=manifest.path,
        scope="val-dev",
        seed=arguments.seed,
        selected_channels=POSTERIOR_CHANNELS,
        feature_cache=paths.feature_cache,
        smooth_probability=0.0,
    )
    _verify_dataset(
        dataset,
        manifest=manifest,
        cache_sha256=config.cache_sha256,
        subject=arguments.subject,
    )
    model = _build_model(
        identity,
        manifest=manifest,
        config=config,
        paths=paths,
        subject=arguments.subject,
        seed=arguments.seed,
        device=arguments.device,
    )
    result = evaluate_development_model(
        model,
        dataset,
        batch_size=config.batch_size,
        device=arguments.device,
        seed=arguments.seed,
    )
    source_records = [
        {
            "manifest_sha256": manifest.manifest_sha256,
            "records_sha256": manifest.records_sha256,
            "role": "val-dev",
            "role_payload_sha256": manifest.val_dev_role_sha256,
            "source_manifest_sha256": manifest.source_manifest_sha256,
        }
    ]
    ScoreArtifact.save(
        paths.output_dir,
        result.similarity,
        tuple(dataset.query_ids),
        tuple(dataset.gallery_ids),
        {
            "checkpoint_sha256": identity.checkpoint_sha256,
            "config_sha256": identity.checkpoint_payload[
                "config_sha256"
            ],
            "git_sha": identity.run_manifest["git_sha"],
            "protocol_sha256": manifest.protocol_sha256,
            "seed": arguments.seed,
            "source_records": source_records,
            "split_role": "val-dev",
            "stage": identity.candidate.stage,
            "subject": arguments.subject,
        },
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run_evaluation(parse_arguments(argv))


if __name__ == "__main__":
    raise SystemExit(main())
