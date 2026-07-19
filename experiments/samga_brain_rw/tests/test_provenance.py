from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from samga_brain_rw.access import (
    TypedArtifact,
    VerifiedArtifact,
    verify_typed_artifacts,
)
from samga_brain_rw.capability_map import build_stage0_capability_map
from samga_brain_rw.hashing import canonical_json_bytes, ordered_ids_sha256, sha256_json
from samga_brain_rw.provenance import (
    CAPABILITY_PAYLOAD_TYPES,
    DEFAULT_ORACLES,
    ENVIRONMENT_VARIABLE_ALLOWLIST,
    PACKAGE_VERSION_ALLOWLIST,
    EnvironmentSnapshot,
    ProvenanceInputs,
    SourceFileOracle,
    build_provenance_manifest,
    expected_capability_paths,
    preflight_provenance_inputs,
)
from scripts.preflight import (
    capture_environment,
    load_and_verify_capability_map,
    main as preflight_main,
    parse_args,
    publish_canonical_exclusive,
)


SUBJECTS = tuple(range(1, 11))


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


def _init_git_repository(path: Path) -> str:
    marker = path / "fixture.txt"
    marker.write_text("sealed synthetic repository\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "add", "fixture.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.name=Codex Test",
            "-c",
            "user.email=codex@example.invalid",
            "commit",
            "-qm",
            "synthetic fixture",
        ],
        check=True,
    )
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _capability(
    path: Path,
    payload_type: str,
    *,
    role: str | None = None,
) -> VerifiedArtifact:
    if role is None:
        provenance = {"fixture": "synthetic"}
        metadata = {
            "ordered_ids": [path.name],
            "source_records": [],
        }
        envelope = path.parent / f".{path.name}.synthetic-envelope.json"
        _write_json(
            envelope,
            {
                "metadata": metadata,
                "metadata_sha256": sha256_json(metadata),
                "ordered_ids_sha256": ordered_ids_sha256([path.name]),
                "payload_sha256": _digest(path),
                "payload_type": payload_type,
                "provenance": provenance,
                "provenance_sha256": sha256_json(provenance),
                "schema_version": 1,
                "scope": "train",
                "source_records_sha256": sha256_json([]),
            },
        )
    else:
        envelope = path
    descriptor = TypedArtifact(
        payload_type=payload_type,
        payload_path=path,
        envelope_path=envelope,
        role=role,
    )
    return verify_typed_artifacts("train", [descriptor])[0]


def _refresh_generic_capability(
    capability: VerifiedArtifact,
) -> VerifiedArtifact:
    """Refresh one strict fixture sidecar after an intentional payload edit."""

    path = capability.artifact.payload_path
    envelope_path = capability.artifact.envelope_path
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    envelope["payload_sha256"] = _digest(path)
    envelope["metadata"]["byte_count"] = path.stat().st_size
    envelope["metadata_sha256"] = sha256_json(envelope["metadata"])
    _write_json(envelope_path, envelope)
    return verify_typed_artifacts("train", [capability.artifact])[0]


def _semantic_payloads(
    root: Path,
    upstream_revision: str,
) -> tuple[dict[str, object], dict[str, object]]:
    upstream = root / "upstream"
    model = root / DEFAULT_ORACLES.model_revision
    clip_model = root / "clip-model"
    internvit = {
        "schema_version": 1,
        "config_type": "internvit_baseline",
        "config_id": "internvit_baseline_v1",
        "upstream": {
            "path": str(upstream),
            "git_commit": upstream_revision,
        },
        "model": {
            "repo": DEFAULT_ORACLES.internvit_repository,
            "revision": DEFAULT_ORACLES.model_revision,
            "path": str(model),
            "config_sha256": "",
            "preprocessor_sha256": "",
            "weight_sha256": {},
        },
        "cache": {
            "path": DEFAULT_ORACLES.selected_cache_declared_path,
            "sha256": "",
            "generator_git_revision": DEFAULT_ORACLES.cache_generator_revision,
            "canonical_train_manifest_sha256": "",
            "shape": [6, 5, 3200],
            "dtype": "float16",
            "layer_route": "idx0",
            "pooling": "patch_mean",
            "normalization": "none",
        },
        "task": {
            "layer_ids": [20, 24, 28, 32, 36],
            "image_dim": 3200,
            "prior_center": 28,
            "router_eval_mode": "global",
            "force_global": True,
            "channels": [
                "P7",
                "P5",
                "P3",
                "P1",
                "Pz",
                "P2",
                "P4",
                "P6",
                "P8",
                "PO7",
                "PO3",
                "POz",
                "PO4",
                "PO8",
                "O1",
                "Oz",
                "O2",
            ],
            "trial_averaging": 4,
            "smooth_probability": 0.3,
            "batch_size": 512,
            "epochs": 60,
            "stage1_epochs": 20,
            "stage1_learning_rate": 0.0001,
            "stage2_learning_rate": 0.00005,
            "mmd_start": 0.9,
            "mmd_end": 0.5,
            "image_l2_normalization": True,
            "eeg_l2_normalization": False,
        },
    }
    brainrw = {
        "schema_version": 1,
        "config_type": "brainrw_clip_lora",
        "config_id": "brainrw_clip_lora_v1",
        "clip": {
            "model_id": DEFAULT_ORACLES.clip_model_id,
            "path": str(clip_model),
            "config_sha256": "",
            "weights_sha256": "",
        },
        "brain_mlp": {"dropout": 0.1},
        "lora": {
            "targets": [
                "q_proj",
                "k_proj",
                "v_proj",
                "out_proj",
                "fc1",
                "fc2",
                "visual_projection",
            ],
            "rank": 32,
            "alpha": 32,
            "dropout": 0.0,
        },
        "optimizer": {
            "name": "AdamW",
            "brain_learning_rate": 0.0005,
            "visual_learning_rate": 0.00005,
            "weight_decay": 0.05,
            "schedule": "cosine",
        },
        "training": {
            "epochs": 25,
            "epoch_policy": "fixed",
            "precision": "bf16",
            "batch_size": 512,
            "trial_averaging": 4,
            "channels": list(internvit["task"]["channels"]),
        },
    }
    return internvit, brainrw


def _protocol_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "split_salt": "AIAA3800-SAMGA-SPLIT-v1\n",
        "stimulus_salt": "AIAA3800-SAMGA-STIM-v1\n",
        "expected_non_test_concepts": 3,
        "split_sizes": {"train": 1, "val-dev": 1, "val-confirm": 1},
        "pilot_subjects": [1, 5, 8],
        "pilot_seeds": [42, 43],
        "confirmation_subjects": list(SUBJECTS),
        "confirmation_seeds": [42, 43, 44, 45, 46],
        "historical_top1": 0.8902,
        "historical_top5": 0.9887,
        "paper_top1": 0.913,
        "paper_top5": 0.988,
        "pilot_gate": {
            "stage1_min_top1_delta": 0.003,
            "other_min_top1_delta": 0.005,
            "minimum_positive_cells": 4,
            "minimum_top5_delta": -0.002,
            "minimum_subject_mean_top1_delta": -0.02,
        },
        "confirmation_gate": {
            "minimum_top1_delta": 0.005,
            "ci95_lower_must_exceed": 0.0,
            "minimum_top5_delta": -0.002,
            "minimum_positive_subjects": 8,
            "minimum_subject_mean_top1_delta": -0.02,
        },
        "bootstrap": {
            "samples": 10000,
            "seed": 20260719,
            "resampling": (
                "independent_subject_and_seed_indices_with_replacement_"
                "cartesian_mean"
            ),
            "quantile_method": "linear",
        },
        "retrieval": {
            "method": "standard_independent_cosine",
            "similarity": "cosine",
            "assignment": "independent",
            "hungarian": False,
        },
        "output_paths": {
            "artifacts": "artifacts/samga_brain_rw",
            "logs": "logs/samga_brain_rw",
            "results": "results/samga_brain_rw",
        },
    }


@pytest.fixture()
def synthetic_inputs(
    tmp_path: Path,
) -> tuple[ProvenanceInputs, dict[str, VerifiedArtifact]]:
    repository = tmp_path / "repository"
    config_dir = repository / "experiments" / "samga_brain_rw" / "configs"
    source_manifest_dir = tmp_path / "source-manifests"
    protocol_manifest_dir = tmp_path / "protocol-manifests"
    feature_directory = tmp_path / "features" / "merged" / "train"
    variant_directory = tmp_path / "features" / "variants" / "train_idx0_patch_mean"
    data_root = tmp_path / "things_eeg_data"
    model_path = tmp_path / DEFAULT_ORACLES.model_revision
    clip_model_path = tmp_path / "clip-model"
    upstream_root = tmp_path / "upstream"
    for directory in (
        config_dir,
        source_manifest_dir,
        protocol_manifest_dir,
        feature_directory,
        variant_directory,
        data_root,
        model_path,
        clip_model_path,
        upstream_root,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    experiment_revision = _init_git_repository(repository)
    upstream_revision = _init_git_repository(upstream_root)

    protocol_path = config_dir / "protocol_v1.json"
    internvit_config_path = config_dir / "internvit_baseline_v1.json"
    brainrw_config_path = config_dir / "brainrw_clip_lora_v1.json"
    _write_json(protocol_path, _protocol_payload())

    model_files = {
        "internvit.config": model_path / "config.json",
        "internvit.preprocessor": model_path / "preprocessor_config.json",
        "internvit.modeling": model_path / "modeling_intern_vit.py",
        "internvit.weight.1": model_path / "model-00001-of-00003.safetensors",
        "internvit.weight.2": model_path / "model-00002-of-00003.safetensors",
        "internvit.weight.3": model_path / "model-00003-of-00003.safetensors",
        "clip.config": clip_model_path / "config.json",
        "clip.weights": clip_model_path / "model.safetensors",
    }
    for index, path in enumerate(model_files.values(), start=1):
        if path.suffix == ".json":
            _write_json(path, {"transformers_version": "4.37.2", "index": index})
        else:
            path.write_bytes(f"synthetic-model-file-{index}".encode())

    records = [
        {
            "concept_id": f"concept-{row // 2}",
            "image_id": f"image-{row}",
            "image_path": f"training_images/image-{row}.jpg",
            "row_index": row,
            "validation_query": False,
        }
        for row in range(6)
    ]
    records_sha256 = sha256_json(records)
    protocol_config_sha256 = sha256_json(_protocol_payload())

    source_oracles: list[SourceFileOracle] = []
    capabilities: dict[str, VerifiedArtifact] = {}
    split_assignment = {
        "payload_type": "samga_brain_rw.split_assignment",
        "protocol_config_sha256": protocol_config_sha256,
        "record_count": 6,
        "records_sha256": records_sha256,
        "schema_version": 1,
    }
    split_assignment_path = protocol_manifest_dir / "split_assignment.json"
    _write_json(split_assignment_path, split_assignment)
    manifest_summary_path = protocol_manifest_dir / "manifest_summary.json"
    summary_subjects: list[dict[str, object]] = []
    capabilities["protocol.split_assignment"] = _capability(
        split_assignment_path,
        CAPABILITY_PAYLOAD_TYPES["protocol.split_assignment"],
    )
    for subject in SUBJECTS:
        subject_label = f"sub-{subject:02d}"
        source_pt = (
            data_root
            / "Preprocessed_data_250Hz_whiten"
            / subject_label
            / "train.pt"
        )
        source_pt.parent.mkdir(parents=True, exist_ok=True)
        source_pt.write_bytes(f"synthetic-train-{subject}".encode())
        source_manifest = source_manifest_dir / f"{subject_label}_train.json"
        source_payload = {
            "ch_names": ["P7"],
            "eeg_dtype": "float32",
            "eeg_shape": [6, 1],
            "records": records,
            "records_sha256": records_sha256,
            "schema_version": 1,
            "source_pt": str(source_pt),
            "split": "train",
            "subject_id": subject,
            "validation_concepts": [],
            "validation_salt": "legacy-not-used",
        }
        _write_json(source_manifest, source_payload)
        protocol_manifest = (
            protocol_manifest_dir / f"{subject_label}_protocol.json"
        )
        train_role = {
            "concept_count": 3,
            "concept_ids": ["concept-0", "concept-1", "concept-2"],
            "gallery_ids": [],
            "ordered_ids": ["concept-0", "concept-1", "concept-2"],
            "payload_type": "samga_brain_rw.role_payload",
            "query_ids": [],
            "row_count": 6,
            "row_indices": list(range(6)),
            "schema_version": 1,
            "scope": "train",
        }
        role_provenance = {
            "protocol_config_sha256": protocol_config_sha256,
            "source_manifest_sha256": _digest(source_manifest),
        }
        train_descriptor = {
            "ordered_ids_sha256": ordered_ids_sha256(
                train_role["ordered_ids"]
            ),
            "payload_type": "samga_brain_rw.role_payload",
            "provenance_sha256": sha256_json(role_provenance),
            "role_payload_sha256": sha256_json(train_role),
            "schema_version": 1,
            "scope": "train",
            "source_records_sha256": records_sha256,
        }
        subject_split_assignment = {
            "protocol_config_sha256": protocol_config_sha256,
            "records_sha256": records_sha256,
        }
        _write_json(
            protocol_manifest,
            {
                "payload_type": "samga_brain_rw.subject_protocol_manifest",
                "protocol_config_sha256": protocol_config_sha256,
                "records_sha256": records_sha256,
                "role_artifacts": {
                    "train": train_descriptor,
                    "val-confirm": {},
                    "val-dev": {},
                },
                "role_payloads": {
                    "train": train_role,
                    "val-confirm": {},
                    "val-dev": {},
                },
                "schema_version": 1,
                "source_manifest_path": str(source_manifest),
                "source_manifest_sha256": _digest(source_manifest),
                "split_assignment": subject_split_assignment,
                "split_assignment_payload_sha256": sha256_json(
                    subject_split_assignment
                ),
                "subject_id": subject,
            },
        )
        summary_subjects.append(
            {
                "protocol_manifest": protocol_manifest.name,
                "protocol_manifest_sha256": _digest(protocol_manifest),
                "source_manifest_path": str(source_manifest),
                "source_manifest_sha256": _digest(source_manifest),
                "subject_id": subject,
            }
        )
        capabilities[f"source_manifest.{subject_label}"] = _capability(
            source_manifest,
            CAPABILITY_PAYLOAD_TYPES[f"source_manifest.{subject_label}"],
        )
        capabilities[f"protocol_manifest.{subject_label}"] = _capability(
            protocol_manifest,
            CAPABILITY_PAYLOAD_TYPES[f"protocol_manifest.{subject_label}"],
            role="train",
        )
        capabilities[f"source_train_pt.{subject_label}"] = _capability(
            source_pt,
            CAPABILITY_PAYLOAD_TYPES[f"source_train_pt.{subject_label}"],
        )
        source_oracles.append(
            SourceFileOracle(
                subject_id=subject,
                manifest_sha256=_digest(source_manifest),
                byte_count=source_pt.stat().st_size,
                sha256=_digest(source_pt),
            )
        )

    manifest_summary = {
        "payload_type": "samga_brain_rw.manifest_summary",
        "protocol_config_sha256": protocol_config_sha256,
        "record_count_per_subject": 6,
        "records_sha256": records_sha256,
        "schema_version": 1,
        "split_assignment_file_sha256": _digest(split_assignment_path),
        "split_assignment_payload_sha256": sha256_json(split_assignment),
        "subject_count": 10,
        "subjects": summary_subjects,
    }
    _write_json(manifest_summary_path, manifest_summary)
    capabilities["protocol.manifest_summary"] = _capability(
        manifest_summary_path,
        CAPABILITY_PAYLOAD_TYPES["protocol.manifest_summary"],
    )

    canonical_cache = variant_directory / "features.npy"
    merged_cache = feature_directory / "patch_mean.npy"
    clip_train_cache = tmp_path / "clip-cache" / "clip_layers_train.npy"
    clip_train_cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(canonical_cache, np.zeros((6, 5, 3200), dtype=np.float16))
    np.save(merged_cache, np.zeros((6, 10, 3200), dtype=np.float16))
    np.save(clip_train_cache, np.zeros((6, 5, 768), dtype=np.float16))
    cache_metadata = {
        "cache.internvit_selected_metadata": (
            variant_directory / "metadata.json",
            {
                "split": "train",
                "shape": [6, 5, 3200],
                "dtype": "float16",
                "feature_sha256": _digest(canonical_cache),
                "layer_route": "idx0",
                "source_axes": [0, 2, 4, 6, 8],
                "pooling": "patch_mean",
                "normalization": "none",
            },
        ),
        "cache.internvit_merged_metadata": (
            feature_directory / "metadata.json",
            {
                "split": "train",
                "shape": [6, 10, 3200],
                "dtype": "float16",
                "feature_sha256": _digest(merged_cache),
            },
        ),
        "cache.clip_train_metadata": (
            clip_train_cache.with_suffix(".npy.meta.json"),
            {
                "split": "train",
                "shape": [6, 5, 768],
                "dtype": "float16",
                "cache_sha256": _digest(clip_train_cache),
                "layers": [4, 6, 8, 10, 12],
            },
        ),
    }
    for key, (path, payload) in cache_metadata.items():
        _write_json(path, payload)
        capabilities[key] = _capability(
            path,
            CAPABILITY_PAYLOAD_TYPES[key],
        )
    capabilities["cache.internvit_selected"] = _capability(
        canonical_cache,
        CAPABILITY_PAYLOAD_TYPES["cache.internvit_selected"],
    )
    capabilities["cache.internvit_merged"] = _capability(
        merged_cache,
        CAPABILITY_PAYLOAD_TYPES["cache.internvit_merged"],
    )
    capabilities["cache.clip_train"] = _capability(
        clip_train_cache,
        CAPABILITY_PAYLOAD_TYPES["cache.clip_train"],
    )

    internvit, brainrw = _semantic_payloads(tmp_path, upstream_revision)
    internvit_model_hashes = {
        key: _digest(path) for key, path in model_files.items()
    }
    internvit["model"]["config_sha256"] = internvit_model_hashes[
        "internvit.config"
    ]
    internvit["model"]["preprocessor_sha256"] = internvit_model_hashes[
        "internvit.preprocessor"
    ]
    internvit["model"]["weight_sha256"] = {
        f"model-0000{index}-of-00003.safetensors": internvit_model_hashes[
            f"internvit.weight.{index}"
        ]
        for index in (1, 2, 3)
    }
    internvit["cache"]["sha256"] = _digest(canonical_cache)
    internvit["cache"]["canonical_train_manifest_sha256"] = source_oracles[
        0
    ].manifest_sha256
    brainrw["clip"]["config_sha256"] = internvit_model_hashes["clip.config"]
    brainrw["clip"]["weights_sha256"] = internvit_model_hashes["clip.weights"]
    _write_json(internvit_config_path, internvit)
    _write_json(brainrw_config_path, brainrw)

    capabilities["protocol"] = _capability(
        protocol_path,
        CAPABILITY_PAYLOAD_TYPES["protocol"],
    )
    capabilities["internvit_config"] = _capability(
        internvit_config_path,
        CAPABILITY_PAYLOAD_TYPES["internvit_config"],
    )
    capabilities["brainrw_config"] = _capability(
        brainrw_config_path,
        CAPABILITY_PAYLOAD_TYPES["brainrw_config"],
    )
    for key, path in model_files.items():
        capabilities[key] = _capability(
            path,
            CAPABILITY_PAYLOAD_TYPES[key],
        )

    oracles = replace(
        DEFAULT_ORACLES,
        upstream_revision=upstream_revision,
        protocol_config_sha256=sha256_json(_protocol_payload()),
        internvit_semantic_config_sha256=sha256_json(internvit),
        brainrw_semantic_config_sha256=sha256_json(brainrw),
        canonical_records_sha256=records_sha256,
        canonical_train_manifest_sha256=source_oracles[0].manifest_sha256,
        split_assignment_file_sha256=_digest(split_assignment_path),
        split_assignment_payload_sha256=sha256_json(split_assignment),
        manifest_summary_file_sha256=_digest(manifest_summary_path),
        record_count=6,
        concept_count=3,
        stimuli_per_concept=2,
        source_files=tuple(source_oracles),
        internvit_config_sha256=internvit_model_hashes["internvit.config"],
        internvit_preprocessor_sha256=internvit_model_hashes[
            "internvit.preprocessor"
        ],
        internvit_modeling_sha256=internvit_model_hashes[
            "internvit.modeling"
        ],
        internvit_weight_sha256=tuple(
            internvit_model_hashes[f"internvit.weight.{index}"]
            for index in (1, 2, 3)
        ),
        selected_cache_sha256=_digest(canonical_cache),
        selected_cache_size=canonical_cache.stat().st_size,
        selected_cache_shape=(6, 5, 3200),
        merged_cache_sha256=_digest(merged_cache),
        merged_cache_shape=(6, 10, 3200),
        clip_config_sha256=internvit_model_hashes["clip.config"],
        clip_weights_sha256=internvit_model_hashes["clip.weights"],
        clip_cache_sha256=_digest(clip_train_cache),
        clip_cache_size=clip_train_cache.stat().st_size,
        clip_cache_shape=(6, 5, 768),
        data_root=str(data_root),
        clip_model_path=str(clip_model_path),
    )
    environment = EnvironmentSnapshot(
        python_version="3.10.18",
        python_executable="/synthetic/env/bin/python",
        sys_prefix="/synthetic/env",
        platform="Synthetic-Linux",
        machine="x86_64",
        hostname="synthetic-node",
        package_versions={
            name: f"synthetic-{index}"
            for index, name in enumerate(PACKAGE_VERSION_ALLOWLIST)
        },
        selected_environment={
            name: None for name in ENVIRONMENT_VARIABLE_ALLOWLIST
        },
    )
    inputs = ProvenanceInputs(
        repository_root=repository,
        protocol_path=protocol_path,
        internvit_config_path=internvit_config_path,
        brainrw_config_path=brainrw_config_path,
        source_manifest_dir=source_manifest_dir,
        protocol_manifest_dir=protocol_manifest_dir,
        feature_directory=feature_directory,
        variant_directory=variant_directory,
        canonical_cache=canonical_cache,
        clip_train_cache=clip_train_cache,
        data_root=data_root,
        model_path=model_path,
        clip_model_path=clip_model_path,
        upstream_root=upstream_root,
        experiment_revision=experiment_revision,
        upstream_revision=upstream_revision,
        cache_generator_revision=DEFAULT_ORACLES.cache_generator_revision,
        verified_artifacts=capabilities,
        environment=environment,
        oracles=oracles,
    )

    map_path = build_stage0_capability_map(
        replace(inputs, verified_artifacts={}),
        tmp_path / "fixture-capabilities",
    )
    capabilities = load_and_verify_capability_map(
        map_path,
        expected_capability_paths(inputs),
    )
    inputs = replace(inputs, verified_artifacts=capabilities)
    return inputs, capabilities


def test_default_oracles_pin_every_approved_identity() -> None:
    assert DEFAULT_ORACLES.upstream_revision == (
        "1a63745b7ff6f98dad34b0f0b8246a9b5260d9c1"
    )
    assert DEFAULT_ORACLES.cache_generator_revision == (
        "a97b97a110c0fea7d4adafd5abce477c6cce525c"
    )
    assert DEFAULT_ORACLES.model_revision == (
        "9d1a4344077479c93d42584b6941c64d795d508d"
    )
    assert DEFAULT_ORACLES.protocol_config_sha256 == (
        "0a9bb1dc750145ec94c35aaaddf5a834d303be3e6f69c9740237d9b967fd48bd"
    )
    assert DEFAULT_ORACLES.internvit_semantic_config_sha256 == (
        "db3a696a31ceba0699c4039ca73130f75edd7d2ad69ce3e55c9ab7a5ecfc27de"
    )
    assert DEFAULT_ORACLES.brainrw_semantic_config_sha256 == (
        "7224907692b8516a2b07c5b6dcc242288a9a0738259c71abae8674d9cb99e53d"
    )
    assert DEFAULT_ORACLES.canonical_train_manifest_sha256 == (
        "42fd7316314eb02d69ee2234d4d8430afcfcc2a5f6834e9c7be64f38eccdbc85"
    )
    assert DEFAULT_ORACLES.split_assignment_file_sha256 == (
        "1d5ad2344797b359a3aeb04f1c298a7785b1492f2eee44e1d1a178e929ad70dc"
    )
    assert DEFAULT_ORACLES.split_assignment_payload_sha256 == (
        "4463e408af8644eed4c73a4d82832d402ba0b4f70b338f2e797216fd3698d912"
    )
    assert DEFAULT_ORACLES.manifest_summary_file_sha256 == (
        "bbd84cd87dda3ac5f02270a03923857e1e79c13a6acaa4c0a4d4556a1c413dce"
    )
    assert len(CAPABILITY_PAYLOAD_TYPES) == 49
    assert DEFAULT_ORACLES.canonical_records_sha256 == (
        "f59500f36e273f66fce5c2019670b076d75d538feccf296c7d7ed75f19ae3fac"
    )
    assert DEFAULT_ORACLES.internvit_config_sha256 == (
        "4fc4a1187b20575c0da8d27df2ad17f5ad6e8ac1c8b2af707bc8b263bd40c0a2"
    )
    assert DEFAULT_ORACLES.internvit_preprocessor_sha256 == (
        "0658115064c561026539aeeead9ed3b1a8e0cc90967df8c142849199f955d2b4"
    )
    assert DEFAULT_ORACLES.internvit_modeling_sha256 == (
        "56220ba82cb511d51d5f2fa71eebd728b330fbccad9dfb128088a8fcc8f7d260"
    )
    assert DEFAULT_ORACLES.internvit_weight_sha256 == (
        "9818659d13d932da8bc0c3b8ee15f5b5d68d8c94d66eb525be566066630111da",
        "4f0c10e72d6f6513f421baa6ec843d5508657435059c1d18b6b5fd7789f9d5b7",
        "d21c4fe0bc4af1425cfae1a59a8f5fbb00fde9d8e2888325a60913ac61b0494d",
    )
    assert DEFAULT_ORACLES.selected_cache_sha256 == (
        "539c7b62ae41c8112e22b3ddc3a6566d997465a10c36d16c8f2378855ba94c71"
    )
    assert DEFAULT_ORACLES.merged_cache_sha256 == (
        "e5b92033c3b0fd19d71ca825844568883a7fea85c5e97fd6b21db445ff93e1dc"
    )
    assert DEFAULT_ORACLES.clip_cache_sha256 == (
        "a31c1871082e1f052da3d055702455b464ea2345890eee33e447e09328c45ebb"
    )
    assert DEFAULT_ORACLES.selected_cache_shape == (16540, 5, 3200)
    assert DEFAULT_ORACLES.merged_cache_shape == (16540, 10, 3200)
    assert DEFAULT_ORACLES.clip_cache_shape == (16540, 5, 768)
    assert DEFAULT_ORACLES.data_root == (
        "/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/EEG_Project/"
        "EEG_Recon-RL/datasets/things_eeg_data"
    )
    assert DEFAULT_ORACLES.clip_model_path == (
        "/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/EEG_Project/"
        "models/CLIP-ViT-B-32-laion2B-s34B-b79K"
    )
    assert len(DEFAULT_ORACLES.source_files) == 10
    assert tuple(item.subject_id for item in DEFAULT_ORACLES.source_files) == SUBJECTS


def test_all_ten_train_pt_hashes_and_sizes_are_pinned() -> None:
    assert [
        (item.byte_count, item.sha256)
        for item in DEFAULT_ORACLES.source_files
    ] == [
        (2106803096, "815931d6e9eadcda5b1edd054248f81c1e9a42dd4abafe5eb075a80269090f20"),
        (2106802667, "b3e797dc48e8585e36b7a71a6fb40f837061e62fc24c992cab84431fdb110d21"),
        (2106802667, "94acbc554a5b179c19e280b2f8025704c91294be610f425d85c58c8f33f83776"),
        (2106802667, "d2a24081c559e77c88adbe152018f48476f5e5ae4995fe7496aaab72660a92f8"),
        (2106802667, "ded2b3f518347e9aa0a5dd9efe3744a35a7f19edcd743564c09d162d38b8890d"),
        (2106802667, "11768a5fa84f76c4b8067ab950df40bf906e71c75ff656cb8a1dbfbdef2c07b3"),
        (2106802667, "b858327f80d1b0fa4fffc1c9da26d72786ec9ee4833ce462feae4b136ce68a9a"),
        (2106803096, "c5997c48df5e1ef976066f63f6d13dc8ed9babaabe8d75a8f2d8592770c018b3"),
        (2106802667, "aec05e4dfeb822a30cba6b5955927a8bd1d9a4dee8d007a65b69bd11d36b19c1"),
        (2106802667, "61cb24fc9269d7760af9d12e2f32439c65a7aea5ae59f8c4efc6109a437fdeee"),
    ]


def test_build_manifest_uses_only_verified_explicit_inputs_and_exact_semantics(
    synthetic_inputs: tuple[ProvenanceInputs, dict[str, VerifiedArtifact]],
) -> None:
    inputs, capabilities = synthetic_inputs
    manifest = build_provenance_manifest(inputs)

    assert set(manifest) == {
        "schema_version",
        "payload_type",
        "scope",
        "passed",
        "protocol",
        "repositories",
        "data",
        "models",
        "caches",
        "extraction_semantics",
        "environment",
        "checks",
        "capability_inventory",
    }
    assert manifest["schema_version"] == 1
    assert manifest["payload_type"] == "samga_brain_rw.provenance_manifest"
    assert manifest["scope"] == "train"
    assert manifest["passed"] is True
    assert "payload_sha256" not in manifest
    assert all(manifest["checks"].values())
    assert manifest["repositories"] == {
        "cache_generator_revision": DEFAULT_ORACLES.cache_generator_revision,
        "experiment_revision": inputs.experiment_revision,
        "upstream_revision": inputs.upstream_revision,
    }
    assert manifest["data"]["concept_count"] == 3
    assert manifest["data"]["record_count"] == 6
    assert len(manifest["data"]["subjects"]) == 10
    assert manifest["extraction_semantics"] == {
        "downstream_image_l2_normalization": True,
        "extractor_normalization": "none",
        "layer_route": "idx0",
        "logical_layers": [20, 24, 28, 32, 36],
        "pooling": "mean(hidden[:,1:,:], axis=1)",
        "source_axes": [0, 2, 4, 6, 8],
    }
    assert manifest["caches"]["internvit_selected"]["shape"] == [6, 5, 3200]
    assert manifest["caches"]["clip_train"]["shape"] == [6, 5, 768]
    rendered = json.dumps(manifest).lower()
    assert "test_images" not in rendered
    assert "_test.json" not in rendered
    assert "val-confirm" not in json.dumps(manifest).lower()

    inventory = manifest["capability_inventory"]
    assert set(inventory) == {
        "artifact_count",
        "artifacts",
        "inventory_sha256",
    }
    assert inventory["artifact_count"] == 49
    artifacts = inventory["artifacts"]
    assert [entry["key"] for entry in artifacts] == list(
        CAPABILITY_PAYLOAD_TYPES
    )
    assert inventory["inventory_sha256"] == sha256_json(artifacts)
    assert all(
        set(entry)
        == {
            "envelope_path",
            "envelope_sha256",
            "key",
            "payload_path",
            "payload_sha256",
            "payload_type",
            "role",
        }
        for entry in artifacts
    )
    assert all(Path(entry["payload_path"]).is_absolute() for entry in artifacts)
    assert all(Path(entry["envelope_path"]).is_absolute() for entry in artifacts)
    for key, entry in zip(
        CAPABILITY_PAYLOAD_TYPES,
        artifacts,
        strict=True,
    ):
        capability = capabilities[key]
        assert entry == {
            "envelope_path": str(
                capability.artifact.envelope_path.absolute()
            ),
            "envelope_sha256": capability.envelope_sha256,
            "key": key,
            "payload_path": str(capability.artifact.payload_path.absolute()),
            "payload_sha256": capability.payload_sha256,
            "payload_type": capability.artifact.payload_type,
            "role": capability.artifact.role,
        }

    reordered = dict(reversed(list(manifest.items())))
    assert sha256_json(reordered) == sha256_json(manifest)


@pytest.mark.parametrize(
    "mutation",
    [
        "old_experiment_revision",
        "wrong_generator",
        "wrong_protocol_digest",
        "swapped_capability_key",
        "swapped_payload_path",
        "wrong_byte_count",
        "wrong_ordered_ids",
        "nonempty_source_records",
    ],
)
def test_generic_envelope_binding_rejects_self_consistent_mutation_before_load(
    synthetic_inputs: tuple[ProvenanceInputs, dict[str, VerifiedArtifact]],
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    inputs, capabilities = synthetic_inputs
    key = "internvit.modeling"
    capability = capabilities[key]
    envelope_path = capability.artifact.envelope_path
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    metadata = envelope["metadata"]
    provenance = envelope["provenance"]

    if mutation == "old_experiment_revision":
        provenance["experiment_revision"] = "0" * 40
    elif mutation == "wrong_generator":
        provenance["generator"] = "samga_brain_rw.capability_map.v0"
    elif mutation == "wrong_protocol_digest":
        provenance["protocol_config_sha256"] = "0" * 64
    elif mutation == "swapped_capability_key":
        metadata["capability_key"] = "internvit.preprocessor"
    elif mutation == "swapped_payload_path":
        metadata["absolute_payload_path"] = str(
            (capability.artifact.payload_path.parent / "swapped.py").absolute()
        )
    elif mutation == "wrong_byte_count":
        metadata["byte_count"] = capability.size + 1
    elif mutation == "wrong_ordered_ids":
        metadata["ordered_ids"] = ["internvit.preprocessor"]
    else:
        metadata["source_records"] = [{"source": "swapped"}]

    envelope["metadata_sha256"] = sha256_json(metadata)
    envelope["provenance_sha256"] = sha256_json(provenance)
    envelope["ordered_ids_sha256"] = ordered_ids_sha256(
        metadata["ordered_ids"]
    )
    envelope["source_records_sha256"] = sha256_json(
        metadata["source_records"]
    )
    _write_json(envelope_path, envelope)
    capabilities[key] = verify_typed_artifacts(
        "train",
        [capability.artifact],
    )[0]

    semantic_payload_loaded = False

    def fail_semantic_load(*_: object, **__: object) -> dict[str, object]:
        nonlocal semantic_payload_loaded
        semantic_payload_loaded = True
        raise AssertionError("semantic payload load must not run")

    monkeypatch.setattr(
        "samga_brain_rw.provenance._read_json",
        fail_semantic_load,
    )
    with pytest.raises(ValueError, match="generic capability envelope binding"):
        build_provenance_manifest(
            replace(inputs, verified_artifacts=dict(capabilities))
        )
    assert semantic_payload_loaded is False


def test_expected_paths_are_exact_and_do_not_scan_directories(
    synthetic_inputs: tuple[ProvenanceInputs, dict[str, VerifiedArtifact]],
) -> None:
    inputs, capabilities = synthetic_inputs
    expected = expected_capability_paths(inputs)
    assert tuple(expected) == tuple(CAPABILITY_PAYLOAD_TYPES)
    assert set(expected) == set(capabilities)
    assert all("_test" not in str(path) for path in expected.values())
    assert all("val-confirm" not in str(path) for path in expected.values())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "missing verified artifact capability"),
        ("path", "path mismatch"),
        ("type", "payload type mismatch"),
        ("scope", "scope must be train"),
        ("digest", "SHA-256 mismatch|changed"),
    ],
)
def test_capability_mismatch_fails_closed_before_semantic_load(
    synthetic_inputs: tuple[ProvenanceInputs, dict[str, VerifiedArtifact]],
    mutation: str,
    message: str,
) -> None:
    inputs, capabilities = synthetic_inputs
    mutated = dict(capabilities)
    key = "internvit.modeling"
    original = mutated[key]
    if mutation == "missing":
        del mutated[key]
    elif mutation == "path":
        mutated[key] = replace(
            original,
            artifact=replace(
                original.artifact,
                payload_path=inputs.model_path / "wrong.py",
            ),
        )
    elif mutation == "type":
        mutated[key] = replace(
            original,
            artifact=replace(original.artifact, payload_type="checkpoint"),
        )
    elif mutation == "scope":
        mutated[key] = replace(original, scope="val-dev")
    else:
        original.artifact.payload_path.write_bytes(b"mutated-after-verification")

    with pytest.raises((ValueError, KeyError), match=message):
        build_provenance_manifest(replace(inputs, verified_artifacts=mutated))


def test_protocol_role_manifest_is_bound_to_pinned_registry(
    synthetic_inputs: tuple[ProvenanceInputs, dict[str, VerifiedArtifact]],
) -> None:
    inputs, capabilities = synthetic_inputs
    key = "protocol_manifest.sub-01"
    path = capabilities[key].artifact.payload_path
    envelope = json.loads(path.read_text(encoding="utf-8"))
    train_payload = envelope["role_payloads"]["train"]
    train_payload["ordered_ids"] = list(reversed(train_payload["ordered_ids"]))
    train_descriptor = envelope["role_artifacts"]["train"]
    train_descriptor["ordered_ids_sha256"] = ordered_ids_sha256(
        train_payload["ordered_ids"]
    )
    train_descriptor["role_payload_sha256"] = sha256_json(train_payload)
    _write_json(path, envelope)
    capabilities[key] = _capability(
        path,
        CAPABILITY_PAYLOAD_TYPES[key],
        role="train",
    )

    with pytest.raises(ValueError, match="protocol manifest SHA-256 mismatch"):
        build_provenance_manifest(
            replace(inputs, verified_artifacts=dict(capabilities))
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("data_root", "alternate-data", "pinned data root"),
        ("clip_model_path", "alternate-clip", "pinned CLIP model path"),
    ],
)
def test_pinned_runtime_roots_are_enforced_before_artifact_verification(
    synthetic_inputs: tuple[ProvenanceInputs, dict[str, VerifiedArtifact]],
    field: str,
    replacement: str,
    message: str,
) -> None:
    inputs, _ = synthetic_inputs
    changed = replace(
        inputs,
        **{field: inputs.repository_root.parent / replacement},
    )
    with pytest.raises(ValueError, match=message):
        preflight_provenance_inputs(changed)


def test_source_record_counts_come_from_manifest_records_not_image_directories(
    synthetic_inputs: tuple[ProvenanceInputs, dict[str, VerifiedArtifact]],
) -> None:
    inputs, _ = synthetic_inputs
    empty_extra = inputs.data_root / "training_images" / "00010_alligator"
    empty_extra.mkdir(parents=True)
    manifest = build_provenance_manifest(inputs)
    assert manifest["data"]["concept_count"] == 3
    assert manifest["data"]["record_count"] == 6


def test_cache_header_or_metadata_mutation_is_rejected(
    synthetic_inputs: tuple[ProvenanceInputs, dict[str, VerifiedArtifact]],
) -> None:
    inputs, capabilities = synthetic_inputs
    metadata_capability = capabilities[
        "cache.internvit_selected_metadata"
    ]
    metadata_path = metadata_capability.artifact.payload_path
    payload = json.loads(metadata_path.read_text())
    payload["normalization"] = "l2"
    _write_json(metadata_path, payload)
    replacement = _refresh_generic_capability(metadata_capability)
    capabilities["cache.internvit_selected_metadata"] = replacement

    with pytest.raises(ValueError, match="normalization"):
        build_provenance_manifest(
            replace(inputs, verified_artifacts=dict(capabilities))
        )


def test_environment_snapshot_is_strictly_allowlisted() -> None:
    with pytest.raises(ValueError, match="package version keys"):
        EnvironmentSnapshot(
            python_version="3.10",
            python_executable="/env/python",
            sys_prefix="/env",
            platform="Linux",
            machine="x86_64",
            hostname="node",
            package_versions={
                **{name: None for name in PACKAGE_VERSION_ALLOWLIST},
                "secret-package": "1",
            },
            selected_environment={
                name: None for name in ENVIRONMENT_VARIABLE_ALLOWLIST
            },
        )
    with pytest.raises(ValueError, match="environment keys"):
        EnvironmentSnapshot(
            python_version="3.10",
            python_executable="/env/python",
            sys_prefix="/env",
            platform="Linux",
            machine="x86_64",
            hostname="node",
            package_versions={
                name: None for name in PACKAGE_VERSION_ALLOWLIST
            },
            selected_environment={
                **{
                    name: None
                    for name in ENVIRONMENT_VARIABLE_ALLOWLIST
                },
                "AWS_SECRET_ACCESS_KEY": "forbidden",
            },
        )


def test_forbidden_input_path_is_rejected_before_verification(
    synthetic_inputs: tuple[ProvenanceInputs, dict[str, VerifiedArtifact]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, _ = synthetic_inputs
    called = False

    def fail_if_called(*_: object, **__: object) -> tuple[()]:
        nonlocal called
        called = True
        return ()

    monkeypatch.setattr(
        "samga_brain_rw.provenance.verify_typed_artifacts",
        fail_if_called,
    )
    with pytest.raises(ValueError, match="forbidden"):
        build_provenance_manifest(
            replace(
                inputs,
                canonical_cache=inputs.repository_root
                / "test_images"
                / "features.npy",
            )
        )
    assert called is False

def _capability_map_payload(
    capabilities: dict[str, VerifiedArtifact],
) -> dict[str, object]:
    return {
        "artifacts": [
            {
                "envelope_path": str(
                    capabilities[key].artifact.envelope_path.absolute()
                ),
                "key": key,
                "payload_path": str(
                    capabilities[key].artifact.payload_path.absolute()
                ),
                "payload_type": capabilities[key].artifact.payload_type,
                "role": capabilities[key].artifact.role,
            }
            for key in CAPABILITY_PAYLOAD_TYPES
        ],
        "payload_type": "samga_brain_rw.capability_map",
        "schema_version": 1,
        "scope": "train",
    }


def _preflight_argv(
    inputs: ProvenanceInputs,
    capability_map: Path,
    output: Path,
) -> list[str]:
    values = (
        ("--repository-root", inputs.repository_root),
        ("--protocol", inputs.protocol_path),
        ("--internvit-config", inputs.internvit_config_path),
        ("--brainrw-config", inputs.brainrw_config_path),
        ("--source-manifest-dir", inputs.source_manifest_dir),
        ("--manifest-dir", inputs.protocol_manifest_dir),
        ("--feature-directory", inputs.feature_directory),
        ("--variant-directory", inputs.variant_directory),
        ("--canonical-cache", inputs.canonical_cache),
        ("--clip-train-cache", inputs.clip_train_cache),
        ("--data-root", inputs.data_root),
        ("--model-path", inputs.model_path),
        ("--clip-model-path", inputs.clip_model_path),
        ("--upstream-root", inputs.upstream_root),
        ("--experiment-revision", inputs.experiment_revision),
        ("--upstream-revision", inputs.upstream_revision),
        ("--cache-generator-revision", inputs.cache_generator_revision),
        ("--capability-map", capability_map),
        ("--output", output),
    )
    return [str(value) for pair in values for value in pair]


def test_preflight_binds_raw_capability_map_and_exact_inventory_projection(
    synthetic_inputs: tuple[ProvenanceInputs, dict[str, VerifiedArtifact]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, capabilities = synthetic_inputs
    capability_map = tmp_path / "cli-capability-map.json"
    output = tmp_path / "preflight-output.json"
    map_payload = _capability_map_payload(capabilities)
    _write_json(capability_map, map_payload)
    raw_map = capability_map.read_bytes()
    monkeypatch.setattr("scripts.preflight.DEFAULT_ORACLES", inputs.oracles)

    assert preflight_main(
        _preflight_argv(inputs, capability_map, output)
    ) == 0

    manifest = json.loads(output.read_text(encoding="utf-8"))
    inventory = manifest["capability_inventory"]
    assert manifest["capability_map"] == {
        "artifact_count": 49,
        "inventory_sha256": inventory["inventory_sha256"],
        "path": str(capability_map.absolute()),
        "raw_sha256": hashlib.sha256(raw_map).hexdigest(),
        "schema_version": 1,
    }
    projection = [
        {
            "envelope_path": item["envelope_path"],
            "key": item["key"],
            "payload_path": item["payload_path"],
            "payload_type": item["payload_type"],
            "role": item["role"],
        }
        for item in inventory["artifacts"]
    ]
    assert map_payload["artifacts"] == projection
    assert inventory["inventory_sha256"] == sha256_json(
        inventory["artifacts"]
    )


def test_preflight_rejects_raw_map_byte_mutation_without_publication(
    synthetic_inputs: tuple[ProvenanceInputs, dict[str, VerifiedArtifact]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, capabilities = synthetic_inputs
    capability_map = tmp_path / "mutated-capability-map.json"
    output = tmp_path / "must-not-exist.json"
    _write_json(capability_map, _capability_map_payload(capabilities))
    capability_map.write_bytes(capability_map.read_bytes() + b" ")
    monkeypatch.setattr("scripts.preflight.DEFAULT_ORACLES", inputs.oracles)

    with pytest.raises(ValueError, match="canonical JSON"):
        preflight_main(_preflight_argv(inputs, capability_map, output))

    assert not output.exists()


def test_preflight_rejects_old_sidecar_revision_without_publication(
    synthetic_inputs: tuple[ProvenanceInputs, dict[str, VerifiedArtifact]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, capabilities = synthetic_inputs
    key = "internvit.modeling"
    capability = capabilities[key]
    envelope_path = capability.artifact.envelope_path
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    envelope["provenance"]["experiment_revision"] = "0" * 40
    envelope["provenance_sha256"] = sha256_json(envelope["provenance"])
    _write_json(envelope_path, envelope)

    capability_map = tmp_path / "old-revision-capability-map.json"
    output = tmp_path / "must-not-exist.json"
    _write_json(capability_map, _capability_map_payload(capabilities))
    monkeypatch.setattr("scripts.preflight.DEFAULT_ORACLES", inputs.oracles)

    with pytest.raises(
        ValueError,
        match="generic capability envelope binding",
    ):
        preflight_main(_preflight_argv(inputs, capability_map, output))

    assert not output.exists()


def test_capability_map_strictly_verifies_all_49_explicit_descriptors(
    synthetic_inputs: tuple[ProvenanceInputs, dict[str, VerifiedArtifact]],
    tmp_path: Path,
) -> None:
    inputs, capabilities = synthetic_inputs
    path = tmp_path / "capability-map.json"
    _write_json(path, _capability_map_payload(capabilities))

    verified = load_and_verify_capability_map(
        path,
        expected_capability_paths(inputs),
    )

    assert len(verified) == 49
    assert tuple(verified) == tuple(CAPABILITY_PAYLOAD_TYPES)
    assert all(item.scope == "train" for item in verified.values())


def test_capability_map_rejects_noncanonical_json_before_artifact_io(
    synthetic_inputs: tuple[ProvenanceInputs, dict[str, VerifiedArtifact]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, capabilities = synthetic_inputs
    path = tmp_path / "capability-map.json"
    path.write_text(
        json.dumps(_capability_map_payload(capabilities)) + "\n",
        encoding="utf-8",
    )
    called = False

    def fail_if_called(*_: object, **__: object) -> tuple[()]:
        nonlocal called
        called = True
        return ()

    monkeypatch.setattr("scripts.preflight.verify_typed_artifacts", fail_if_called)
    with pytest.raises(ValueError, match="canonical JSON"):
        load_and_verify_capability_map(path, expected_capability_paths(inputs))
    assert called is False


def test_capability_map_rejects_forbidden_path_before_artifact_io(
    synthetic_inputs: tuple[ProvenanceInputs, dict[str, VerifiedArtifact]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, capabilities = synthetic_inputs
    payload = _capability_map_payload(capabilities)
    payload["artifacts"][0]["payload_path"] = str(
        (tmp_path / "test_images" / "protocol.json").absolute()
    )
    path = tmp_path / "capability-map.json"
    _write_json(path, payload)
    called = False

    def fail_if_called(*_: object, **__: object) -> tuple[()]:
        nonlocal called
        called = True
        return ()

    monkeypatch.setattr("scripts.preflight.verify_typed_artifacts", fail_if_called)
    with pytest.raises(ValueError, match="forbidden"):
        load_and_verify_capability_map(path, expected_capability_paths(inputs))
    assert called is False


def test_canonical_publication_is_exclusive_and_never_overwrites(
    tmp_path: Path,
) -> None:
    output = tmp_path / "preflight.json"
    payload = {"passed": True, "schema_version": 1}
    publish_canonical_exclusive(output, payload)
    assert output.read_bytes() == canonical_json_bytes(payload)

    with pytest.raises(FileExistsError):
        publish_canonical_exclusive(output, {"passed": False})
    assert output.read_bytes() == canonical_json_bytes(payload)
    assert not list(tmp_path.glob(".*.tmp"))


def test_environment_capture_uses_only_the_two_allowlists() -> None:
    environment = capture_environment(
        environ={
            "HF_HUB_OFFLINE": "1",
            "AWS_SECRET_ACCESS_KEY": "must-not-be-captured",
        },
        version_lookup=lambda package: f"fixture-{package}",
    )
    payload = environment.to_payload()
    assert set(payload["package_versions"]) == set(PACKAGE_VERSION_ALLOWLIST)
    assert set(payload["selected_environment"]) == set(
        ENVIRONMENT_VARIABLE_ALLOWLIST
    )
    assert payload["selected_environment"]["HF_HUB_OFFLINE"] == "1"
    assert "AWS_SECRET_ACCESS_KEY" not in json.dumps(payload)


def test_preflight_cli_requires_every_explicit_path_and_revision() -> None:
    values = {
        "--repository-root": "/repo",
        "--protocol": "/repo/protocol.json",
        "--internvit-config": "/repo/internvit.json",
        "--brainrw-config": "/repo/brainrw.json",
        "--source-manifest-dir": "/source",
        "--manifest-dir": "/manifests",
        "--feature-directory": "/features",
        "--variant-directory": "/variant",
        "--canonical-cache": "/variant/features.npy",
        "--clip-train-cache": "/cache/clip.npy",
        "--data-root": "/data",
        "--model-path": "/model",
        "--clip-model-path": "/clip",
        "--upstream-root": "/upstream",
        "--experiment-revision": "1" * 40,
        "--upstream-revision": "2" * 40,
        "--cache-generator-revision": "3" * 40,
        "--capability-map": "/caps.json",
        "--output": "/output.json",
    }
    argv = [value for pair in values.items() for value in pair]
    args = parse_args(argv)
    assert args.capability_map == Path("/caps.json")
    assert args.clip_train_cache == Path("/cache/clip.npy")
    assert args.data_root == Path("/data")
    assert args.internvit_config == Path("/repo/internvit.json")
    assert args.brainrw_config == Path("/repo/brainrw.json")


def test_git_head_mismatch_fails_before_any_capability_verification(
    synthetic_inputs: tuple[ProvenanceInputs, dict[str, VerifiedArtifact]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, _ = synthetic_inputs
    called = False

    def fail_if_called(*_: object, **__: object) -> tuple[()]:
        nonlocal called
        called = True
        return ()

    monkeypatch.setattr(
        "samga_brain_rw.provenance.verify_typed_artifacts",
        fail_if_called,
    )
    with pytest.raises(ValueError, match="experiment repository HEAD mismatch"):
        build_provenance_manifest(
            replace(inputs, experiment_revision="0" * 40)
        )
    assert called is False


def test_partial_modern_cache_metadata_is_rejected(
    synthetic_inputs: tuple[ProvenanceInputs, dict[str, VerifiedArtifact]],
) -> None:
    inputs, capabilities = synthetic_inputs
    key = "cache.internvit_selected_metadata"
    metadata_path = capabilities[key].artifact.payload_path
    metadata = json.loads(metadata_path.read_text())
    del metadata["normalization"]
    _write_json(metadata_path, metadata)
    capabilities[key] = _refresh_generic_capability(capabilities[key])
    with pytest.raises(ValueError, match="missing modern metadata fields"):
        build_provenance_manifest(
            replace(inputs, verified_artifacts=dict(capabilities))
        )


def test_clip_capabilities_bind_transformers_loader_files(
    synthetic_inputs: tuple[ProvenanceInputs, dict[str, VerifiedArtifact]],
) -> None:
    inputs, _ = synthetic_inputs
    paths = expected_capability_paths(inputs)
    assert paths["clip.config"].name == "config.json"
    assert paths["clip.weights"].name == "model.safetensors"
    assert "open_clip" not in str(paths["clip.config"])
    assert "open_clip" not in str(paths["clip.weights"])
    assert DEFAULT_ORACLES.clip_config_sha256 == (
        "1284cbff35169abb23a1c5525a8b0f543c7bd191d4b9aed63880c1571bc4191c"
    )
    assert DEFAULT_ORACLES.clip_weights_sha256 == (
        "74813fbcdc750f235c9784c367ca1394d2a5c25eb0aac92761752ac239db7cff"
    )
