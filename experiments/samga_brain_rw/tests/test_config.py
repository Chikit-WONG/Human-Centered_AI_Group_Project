from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from samga_brain_rw.config import (
    ProtocolConfig,
    SemanticConfig,
    make_run_key,
    resolve_run_config,
)


CHANNELS = [
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
]
PROTOCOL_SEMANTICS = {
    "schema_version": 1,
    "split_salt": "AIAA3800-SAMGA-SPLIT-v1\n",
    "stimulus_salt": "AIAA3800-SAMGA-STIM-v1\n",
    "expected_non_test_concepts": 1654,
    "split_sizes": {"train": 1254, "val-dev": 200, "val-confirm": 200},
    "pilot_subjects": [1, 5, 8],
    "pilot_seeds": [42, 43],
    "confirmation_subjects": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
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
            "independent_subject_and_seed_indices_with_replacement_cartesian_mean"
        ),
        "quantile_method": "linear",
    },
    "retrieval": {
        "method": "standard_independent_cosine",
        "similarity": "cosine",
        "assignment": "independent",
        "hungarian": False,
    },
}
OUTPUT_PATHS = {
    "artifacts": "artifacts/samga_brain_rw",
    "logs": "logs/samga_brain_rw",
    "results": "results/samga_brain_rw",
}


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _load_raw(path: Path) -> dict[str, object]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_protocol_locks_exact_semantics_thresholds_and_paths(configs_dir: Path) -> None:
    protocol = ProtocolConfig.from_path(configs_dir / "protocol_v1.json")

    assert protocol.canonical_payload() == PROTOCOL_SEMANTICS
    assert protocol.pilot_subjects == (1, 5, 8)
    assert protocol.pilot_seeds == (42, 43)
    assert protocol.confirmation_subjects == tuple(range(1, 11))
    assert protocol.confirmation_seeds == tuple(range(42, 47))
    assert protocol.output_paths.canonical_payload() == OUTPUT_PATHS
    assert protocol.retrieval.method == "standard_independent_cosine"
    assert protocol.retrieval.assignment == "independent"
    assert protocol.retrieval.hungarian is False
    assert len(protocol.sha256) == 64

    with pytest.raises(FrozenInstanceError):
        protocol.historical_top1 = 0.0  # type: ignore[misc]


def test_protocol_hash_is_canonical_and_excludes_output_paths(
    configs_dir: Path, tmp_path: Path
) -> None:
    source = _load_raw(configs_dir / "protocol_v1.json")
    reordered = dict(reversed(list(source.items())))
    reordered["pilot_gate"] = dict(
        reversed(list(source["pilot_gate"].items()))  # type: ignore[union-attr]
    )
    same = ProtocolConfig.from_path(_write_json(tmp_path / "reordered.json", reordered))
    original = ProtocolConfig.from_path(configs_dir / "protocol_v1.json")
    assert same.sha256 == original.sha256

    changed_semantics = json.loads(json.dumps(source))
    changed_semantics["historical_top1"] = 0.8903
    changed = ProtocolConfig.from_path(
        _write_json(tmp_path / "semantic-change.json", changed_semantics)
    )
    assert changed.sha256 != original.sha256

    changed_paths = json.loads(json.dumps(source))
    changed_paths["output_paths"]["logs"] = "elsewhere/logs"
    relocated = ProtocolConfig.from_path(
        _write_json(tmp_path / "path-change.json", changed_paths)
    )
    assert relocated.sha256 == original.sha256


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update({"surprise": True}), "unknown keys"),
        (
            lambda payload: payload["pilot_gate"].update({"surprise": True}),
            "unknown keys",
        ),
        (
            lambda payload: payload["output_paths"].update({"checkpoints": "x"}),
            "unknown keys",
        ),
    ],
)
def test_protocol_rejects_unknown_keys(
    configs_dir: Path,
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    payload = _load_raw(configs_dir / "protocol_v1.json")
    mutation(payload)  # type: ignore[operator]
    with pytest.raises(ValueError, match=message):
        ProtocolConfig.from_path(_write_json(tmp_path / "unknown.json", payload))


def test_protocol_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version": 1, "schema_version": 1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        ProtocolConfig.from_path(path)


def test_internvit_baseline_locks_provenance_and_runtime(configs_dir: Path) -> None:
    payload = SemanticConfig.from_path(
        configs_dir / "internvit_baseline_v1.json"
    ).canonical_payload()

    assert payload["schema_version"] == 1
    assert payload["config_type"] == "internvit_baseline"
    assert payload["config_id"] == "internvit_baseline_v1"
    assert payload["upstream"] == {
        "path": (
            "/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/EEG_Project/"
            "reference_code/codes_for_papers/SAMGA"
        ),
        "git_commit": "1a63745b7ff6f98dad34b0f0b8246a9b5260d9c1",
    }
    assert payload["model"] == {
        "repo": "OpenGVLab/InternViT-6B-448px-V2_5",
        "revision": "9d1a4344077479c93d42584b6941c64d795d508d",
        "path": (
            "/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/EEG_Project/"
            "models/InternViT-6B-448px-V2_5/"
            "9d1a4344077479c93d42584b6941c64d795d508d"
        ),
        "config_sha256": (
            "4fc4a1187b20575c0da8d27df2ad17f5ad6e8ac1c8b2af707bc8b263bd40c0a2"
        ),
        "preprocessor_sha256": (
            "0658115064c561026539aeeead9ed3b1a8e0cc90967df8c142849199f955d2b4"
        ),
        "weight_sha256": {
            "model-00001-of-00003.safetensors": (
                "9818659d13d932da8bc0c3b8ee15f5b5d68d8c94d66eb525be566066630111da"
            ),
            "model-00002-of-00003.safetensors": (
                "4f0c10e72d6f6513f421baa6ec843d5508657435059c1d18b6b5fd7789f9d5b7"
            ),
            "model-00003-of-00003.safetensors": (
                "d21c4fe0bc4af1425cfae1a59a8f5fbb00fde9d8e2888325a60913ac61b0494d"
            ),
        },
    }
    assert payload["cache"] == {
        "path": (
            "artifacts/samga_reproduction/features/internvit_v2_5_multi_variant/"
            "variants/train_idx0_patch_mean/features.npy"
        ),
        "sha256": "539c7b62ae41c8112e22b3ddc3a6566d997465a10c36d16c8f2378855ba94c71",
        "generator_git_revision": "a97b97a110c0fea7d4adafd5abce477c6cce525c",
        "canonical_train_manifest_sha256": (
            "42fd7316314eb02d69ee2234d4d8430afcfcc2a5f6834e9c7be64f38eccdbc85"
        ),
        "shape": [16540, 5, 3200],
        "dtype": "float16",
        "layer_route": "idx0",
        "pooling": "patch_mean",
        "normalization": "none",
    }
    assert payload["task"] == {
        "layer_ids": [20, 24, 28, 32, 36],
        "image_dim": 3200,
        "prior_center": 28,
        "router_eval_mode": "global",
        "force_global": True,
        "channels": CHANNELS,
        "trial_averaging": 4,
        "smooth_probability": 0.3,
        "batch_size": 512,
        "epochs": 60,
        "stage1_epochs": 20,
        "stage1_learning_rate": 1e-4,
        "stage2_learning_rate": 5e-5,
        "mmd_start": 0.9,
        "mmd_end": 0.5,
        "image_l2_normalization": True,
        "eeg_l2_normalization": False,
    }


def test_brainrw_clip_lora_locks_recipe_and_hashes(configs_dir: Path) -> None:
    payload = SemanticConfig.from_path(
        configs_dir / "brainrw_clip_lora_v1.json"
    ).canonical_payload()

    assert payload["clip"] == {
        "model_id": "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
        "path": (
            "/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/EEG_Project/"
            "models/CLIP-ViT-B-32-laion2B-s34B-b79K"
        ),
        "config_sha256": (
            "1284cbff35169abb23a1c5525a8b0f543c7bd191d4b9aed63880c1571bc4191c"
        ),
        "weights_sha256": (
            "74813fbcdc750f235c9784c367ca1394d2a5c25eb0aac92761752ac239db7cff"
        ),
    }
    assert payload["brain_mlp"] == {"dropout": 0.1}
    assert payload["lora"] == {
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
    }
    assert payload["optimizer"] == {
        "name": "AdamW",
        "brain_learning_rate": 5e-4,
        "visual_learning_rate": 5e-5,
        "weight_decay": 0.05,
        "schedule": "cosine",
    }
    assert payload["training"] == {
        "epochs": 25,
        "epoch_policy": "fixed",
        "precision": "bf16",
        "batch_size": 512,
        "trial_averaging": 4,
        "gradient_checkpointing": True,
        "channels": CHANNELS,
    }


def test_stage1_contains_exact_47_candidate_grid_and_formulas(
    configs_dir: Path,
) -> None:
    payload = SemanticConfig.from_path(
        configs_dir / "stage1_fusion_v1.json"
    ).canonical_payload()
    candidates = payload["candidates"]
    assert isinstance(candidates, list)
    assert len(candidates) == 47
    assert len({entry["config_id"] for entry in candidates}) == 47

    zscore = [entry for entry in candidates if entry["family"] == "zscore_convex"]
    temperature = [
        entry for entry in candidates if entry["family"] == "temperature_convex"
    ]
    rrf = [entry for entry in candidates if entry["family"] == "rrf"]
    assert [entry["alpha"] for entry in zscore] == [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ]
    assert {
        (
            entry["internvit_temperature"],
            entry["clip_temperature"],
            entry["alpha"],
        )
        for entry in temperature
    } == {
        (internvit_temperature, clip_temperature, alpha)
        for internvit_temperature in (0.5, 1.0, 2.0)
        for clip_temperature in (0.5, 1.0, 2.0)
        for alpha in (0.25, 0.5, 0.75)
    }
    assert {
        (entry["k"], entry["internvit_weight"]) for entry in rrf
    } == {
        (k, weight)
        for k in (10, 30, 60)
        for weight in (0.25, 0.5, 0.75)
    }
    assert {entry["formula"] for entry in zscore} == {
        "alpha * zscore_ddof0(S_I) + (1 - alpha) * zscore_ddof0(S_C)"
    }
    assert {entry["formula"] for entry in temperature} == {
        "alpha * S_I / T_I + (1 - alpha) * S_C / T_C"
    }
    assert {entry["formula"] for entry in rrf} == {
        "w / (k + rank_I) + (1 - w) / (k + rank_C)"
    }
    assert all(entry["rank_origin"] == 1 for entry in rrf)
    assert all(
        entry["score_tie_break"] == "gallery_id_utf8_bytewise" for entry in rrf
    )


def test_stage2_contains_only_preregistered_one_factor_candidates(
    configs_dir: Path,
) -> None:
    payload = SemanticConfig.from_path(
        configs_dir / "stage2_candidates_v1.json"
    ).canonical_payload()

    assert payload["combination_policy"] == "one_factor_only_no_post_hoc_combinations"
    assert [(entry["config_id"], entry["enabled"]) for entry in payload["layernorm"]] == [
        ("s2-layernorm-off", False),
        ("s2-layernorm-on", True),
    ]
    assert [(entry["config_id"], entry["enabled"]) for entry in payload["whitening"]] == [
        ("s2-whitening-off", False),
        ("s2-whitening-on", True),
    ]
    assert [
        (entry["config_id"], entry["mode"]) for entry in payload["preprojectors"]
    ] == [
        ("s2-preproj-shared", "shared"),
        ("s2-preproj-separate", "separate_per_layer"),
    ]
    assert [
        (entry["config_id"], entry["method"], entry["window"])
        for entry in payload["checkpoint_averaging"]
    ] == [
        ("s2-raw-epoch60-control", "raw", 1),
        ("s2-avg-last5", "arithmetic", 5),
        ("s2-avg-last10", "arithmetic", 10),
        ("s2-swa-last5", "swa", 5),
        ("s2-swa-last10", "swa", 10),
    ]

    adapter = payload["feature_adapter"]
    assert adapter["label"] == "cached InternViT feature adapter"
    assert adapter["formula"] == (
        "h'_l = h_l + gamma_l * B_l GELU(A_l LayerNorm(h_l))"
    )
    assert adapter["ranks"] == [8, 16, 32]
    assert adapter["learning_rate_ratios"] == [0.05, 0.1]
    assert {
        (entry["config_id"], entry["rank"], entry["learning_rate_ratio"])
        for entry in adapter["candidates"]
    } == {
        (f"s2-adapter-r{rank}-lr{ratio:.2f}", rank, ratio)
        for rank in (8, 16, 32)
        for ratio in (0.05, 0.1)
    }
    assert [
        (entry["config_id"], entry["kind"]) for entry in adapter["controls"]
    ] == [
        ("s2-adapter-identity-control", "identity"),
        ("s2-adapter-global-dense-control", "global_dense"),
        ("s2-adapter-matched-projector-control", "matched_projector"),
    ]


@pytest.mark.parametrize(
    "filename",
    [
        "internvit_baseline_v1.json",
        "brainrw_clip_lora_v1.json",
        "stage1_fusion_v1.json",
        "stage2_candidates_v1.json",
    ],
)
def test_semantic_configs_reject_unknown_top_level_keys(
    configs_dir: Path, tmp_path: Path, filename: str
) -> None:
    payload = _load_raw(configs_dir / filename)
    payload["surprise"] = True
    with pytest.raises(ValueError, match="unknown keys"):
        SemanticConfig.from_path(_write_json(tmp_path / filename, payload))


@pytest.mark.parametrize(
    ("filename", "container"),
    [
        ("internvit_baseline_v1.json", ("task",)),
        ("brainrw_clip_lora_v1.json", ("optimizer",)),
        ("stage1_fusion_v1.json", ("candidates", 0)),
        ("stage2_candidates_v1.json", ("feature_adapter",)),
    ],
)
def test_semantic_configs_reject_unknown_nested_keys(
    configs_dir: Path,
    tmp_path: Path,
    filename: str,
    container: tuple[object, ...],
) -> None:
    payload = _load_raw(configs_dir / filename)
    target: object = payload
    for key in container:
        target = target[key]  # type: ignore[index]
    target["surprise"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="unknown keys"):
        SemanticConfig.from_path(_write_json(tmp_path / filename, payload))


def _candidate(**overrides: object) -> dict[str, object]:
    candidate: dict[str, object] = {
        "schema_version": 1,
        "stage": "stage1",
        "config_id": "zconvex_a050",
        "subject": 1,
        "seed": 42,
        "semantics": {
            "family": "zscore_convex",
            "alpha": 0.5,
            "formula": (
                "alpha * zscore_ddof0(S_I) + (1 - alpha) * zscore_ddof0(S_C)"
            ),
        },
        "runtime": {
            "precision": "float64",
            "retrieval": "standard_independent_cosine",
        },
        "outputs": {
            "artifact_dir": "artifacts/samga_brain_rw/run-a",
            "log_path": "logs/samga_brain_rw/run-a.log",
            "result_path": "results/samga_brain_rw/run-a.json",
        },
    }
    candidate.update(overrides)
    return candidate


def test_resolved_run_hashes_all_semantics_and_inputs_but_not_outputs(
    configs_dir: Path,
) -> None:
    protocol = ProtocolConfig.from_path(configs_dir / "protocol_v1.json")
    inputs = {
        "model_sha256": "1" * 64,
        "cache_sha256": "2" * 64,
        "checkpoint_sha256": "3" * 64,
        "manifest_sha256": "4" * 64,
    }
    resolved = resolve_run_config(protocol, _candidate(), inputs)

    assert resolved.stage == "stage1"
    assert resolved.config_id == "zconvex_a050"
    assert resolved.subject == 1
    assert resolved.seed == 42
    assert len(resolved.semantic_config_sha256) == 64
    assert len(resolved.input_bundle_sha256) == 64
    assert resolved.semantic_config_sha256 in resolved.run_key
    assert resolved.input_bundle_sha256 in resolved.run_key
    assert protocol.sha256 not in resolved.run_key

    semantic_change = _candidate(
        semantics={
            "family": "zscore_convex",
            "alpha": 0.6,
            "formula": (
                "alpha * zscore_ddof0(S_I) + (1 - alpha) * zscore_ddof0(S_C)"
            ),
        }
    )
    runtime_change = _candidate(
        runtime={
            "precision": "float32",
            "retrieval": "standard_independent_cosine",
        }
    )
    output_change = _candidate(
        outputs={
            "artifact_dir": "somewhere/else",
            "log_path": "somewhere/else.log",
            "result_path": "somewhere/else.json",
        }
    )
    changed_input = dict(inputs, cache_sha256="9" * 64)

    for changed_candidate in (
        semantic_change,
        runtime_change,
        _candidate(subject=5),
        _candidate(seed=43),
    ):
        changed = resolve_run_config(protocol, changed_candidate, inputs)
        assert changed.semantic_config_sha256 != resolved.semantic_config_sha256
        assert changed.run_key != resolved.run_key

    changed = resolve_run_config(protocol, _candidate(), changed_input)
    assert changed.semantic_config_sha256 != resolved.semantic_config_sha256
    assert changed.input_bundle_sha256 != resolved.input_bundle_sha256
    assert changed.run_key != resolved.run_key

    relocated = resolve_run_config(protocol, output_change, inputs)
    assert relocated.semantic_config_sha256 == resolved.semantic_config_sha256
    assert relocated.input_bundle_sha256 == resolved.input_bundle_sha256
    assert relocated.run_key == resolved.run_key


def test_resolve_run_config_rejects_unknown_or_invalid_fields(
    configs_dir: Path,
) -> None:
    protocol = ProtocolConfig.from_path(configs_dir / "protocol_v1.json")
    inputs = {"manifest_sha256": "a" * 64}

    with pytest.raises(ValueError, match="unknown keys"):
        resolve_run_config(protocol, _candidate(surprise=True), inputs)
    with pytest.raises(ValueError, match="unknown keys"):
        resolve_run_config(
            protocol,
            _candidate(
                outputs={
                    "artifact_dir": "a",
                    "log_path": "b",
                    "result_path": "c",
                    "surprise": "d",
                }
            ),
            inputs,
        )
    with pytest.raises(ValueError, match="SHA-256"):
        resolve_run_config(protocol, _candidate(), {"manifest_sha256": "not-a-hash"})


def test_make_run_key_binds_full_semantic_and_input_hashes() -> None:
    config_hash = "a" * 64
    input_hash = "b" * 64
    assert make_run_key(
        "stage1", "zconvex_a050", 1, 42, config_hash, input_hash
    ) == (
        "stage1__zconvex_a050__sub-01__seed-42__"
        f"config-{config_hash}__inputs-{input_hash}"
    )
    with pytest.raises(ValueError):
        make_run_key("stage/1", "candidate", 1, 42, config_hash, input_hash)
    with pytest.raises(ValueError):
        make_run_key("stage1", "candidate", 1, 42, "short", input_hash)
