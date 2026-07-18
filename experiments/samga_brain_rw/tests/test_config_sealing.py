from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from samga_brain_rw.config import (
    ProtocolConfig,
    SemanticConfig,
    resolve_run_config,
)


STAGE1_SELECTION = {
    "scope": "val-dev",
    "retrieval": "standard_independent_cosine",
    "zscore_variance": "population_ddof0",
    "constant_row": "all_zero",
    "temperature_softmax": False,
    "branch_score_tie_break": "gallery_id_utf8_bytewise",
    "final_score_tie_break": "gallery_id_utf8_bytewise",
    "metric_tie_break": ["lower_compute", "config_id"],
}
STAGE1_FORMULAS = {
    "zscore_convex": (
        "alpha * zscore_ddof0(S_I) + (1 - alpha) * zscore_ddof0(S_C)"
    ),
    "temperature_convex": "alpha * S_I / T_I + (1 - alpha) * S_C / T_C",
    "rrf": "w / (k + rank_I) + (1 - w) / (k + rank_C)",
}
REQUIRED_INPUT_HASHES = {
    "model_sha256": "1" * 64,
    "cache_sha256": "2" * 64,
    "checkpoint_sha256": "3" * 64,
    "manifest_sha256": "4" * 64,
}
LAYER_ORDER = [20, 24, 28, 32, 36]
MATCHED_BY_RANK = {
    8: {
        "widths": [14, 14, 14, 14, 13],
        "adapter_parameters": 256005,
        "global_dense_parameters": 256001,
        "projector_parameters": 256133,
        "absolute_error": 128,
        "relative_error": 0.0004999902345657311,
    },
    16: {
        "widths": [28, 28, 28, 27, 27],
        "adapter_parameters": 512005,
        "global_dense_parameters": 512001,
        "projector_parameters": 512261,
        "absolute_error": 256,
        "relative_error": 0.0004999951172351833,
    },
    32: {
        "widths": [56, 55, 55, 55, 55],
        "adapter_parameters": 1024005,
        "global_dense_parameters": 1024001,
        "projector_parameters": 1024517,
        "absolute_error": 512,
        "relative_error": 0.0004999975586056709,
    },
}


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _candidate() -> dict[str, object]:
    return {
        "schema_version": 1,
        "stage": "stage2",
        "config_id": "s2-adapter-r8-lr0.05",
        "subject": 1,
        "seed": 42,
        "semantics": {"rank": 8, "learning_rate_ratio": 0.05},
        "runtime": {
            "precision": "float32",
            "retrieval": "standard_independent_cosine",
        },
    }


def _mutate_stage1(payload: dict[str, object], case: str) -> None:
    selection = payload["selection"]
    formulas = payload["formulas"]
    candidates = payload["candidates"]
    assert isinstance(selection, dict)
    assert isinstance(formulas, dict)
    assert isinstance(candidates, list)

    if case == "config_id":
        payload["config_id"] = "unsealed"
    elif case == "formal_scope":
        selection["scope"] = "formal-test"
    elif case == "retrieval":
        selection["retrieval"] = "hungarian"
    elif case == "metric_tie_order":
        selection["metric_tie_break"] = ["config_id", "lower_compute"]
    elif case == "top_formula":
        formulas["rrf"] = "unregistered formula"
    elif case == "empty":
        candidates.clear()
    elif case == "missing":
        candidates.pop()
    elif case == "extra":
        extra = copy.deepcopy(candidates[0])
        extra["config_id"] = "s1-z-extra"
        extra["alpha"] = 0.05
        candidates.append(extra)
    elif case == "reordered":
        candidates[0], candidates[1] = candidates[1], candidates[0]
    elif case == "candidate_id":
        candidates[0]["config_id"] = "s1-z-wrong"
    elif case == "candidate_formula":
        candidates[0]["formula"] = "wrong"
    elif case == "candidate_value":
        candidates[0]["alpha"] = 0.05
    elif case == "temperature_zero":
        candidates[11]["internvit_temperature"] = 0.0
    elif case == "rrf_rank_origin":
        candidates[-1]["rank_origin"] = 0
    else:  # pragma: no cover - test table invariant
        raise AssertionError(case)


def test_stage1_positive_contract_is_exact(configs_dir: Path) -> None:
    config = SemanticConfig.from_path(configs_dir / "stage1_fusion_v1.json")
    payload = config.canonical_payload()

    assert payload["config_id"] == "stage1_fusion_v1"
    assert payload["selection"] == STAGE1_SELECTION
    assert payload["formulas"] == STAGE1_FORMULAS
    assert len(payload["candidates"]) == 47
    assert config.sha256 == (
        "27cd33027c2fa322121f0c42732d2ecbe62f2544d6497652ae40f90b7dd8dc78"
    )


@pytest.mark.parametrize(
    "case",
    [
        "config_id",
        "formal_scope",
        "retrieval",
        "metric_tie_order",
        "top_formula",
        "empty",
        "missing",
        "extra",
        "reordered",
        "candidate_id",
        "candidate_formula",
        "candidate_value",
        "temperature_zero",
        "rrf_rank_origin",
    ],
)
def test_stage1_rejects_every_normative_grid_mutation(
    configs_dir: Path, tmp_path: Path, case: str
) -> None:
    payload = _load(configs_dir / "stage1_fusion_v1.json")
    _mutate_stage1(payload, case)

    with pytest.raises(ValueError, match="exact preregistered Stage 1 v1"):
        SemanticConfig.from_path(_write(tmp_path / f"stage1-{case}.json", payload))


def _mutate_stage2(payload: dict[str, object], case: str) -> None:
    adapter = payload["feature_adapter"]
    assert isinstance(adapter, dict)
    candidates = adapter["candidates"]
    controls = adapter["controls"]
    assert isinstance(candidates, list)
    assert isinstance(controls, list)

    if case == "config_id":
        payload["config_id"] = "unsealed"
    elif case == "combination_policy":
        payload["combination_policy"] = "allow_combinations"
    elif case == "empty_layernorm":
        payload["layernorm"] = []
    elif case == "cross_list_duplicate":
        payload["whitening"][0]["config_id"] = payload["layernorm"][0]["config_id"]
    elif case == "wrong_checkpoint_window":
        payload["checkpoint_averaging"][1]["window"] = 6
    elif case == "reordered_checkpoints":
        payload["checkpoint_averaging"][1:3] = reversed(
            payload["checkpoint_averaging"][1:3]
        )
    elif case == "ranks":
        adapter["ranks"] = [8, 16, 999]
    elif case == "candidate_rank":
        candidates[0]["rank"] = 999
    elif case == "empty_candidates":
        candidates.clear()
    elif case == "extra_candidate":
        extra = copy.deepcopy(candidates[0])
        extra["config_id"] = "s2-adapter-r999-lr0.05"
        extra["rank"] = 999
        candidates.append(extra)
    elif case == "reordered_candidates":
        candidates.reverse()
    elif case == "empty_controls":
        controls.clear()
    elif case == "control_kind":
        controls[1]["kind"] = "unregistered"
    else:  # pragma: no cover - test table invariant
        raise AssertionError(case)


@pytest.mark.parametrize(
    "case",
    [
        "config_id",
        "combination_policy",
        "empty_layernorm",
        "cross_list_duplicate",
        "wrong_checkpoint_window",
        "reordered_checkpoints",
        "ranks",
        "candidate_rank",
        "empty_candidates",
        "extra_candidate",
        "reordered_candidates",
        "empty_controls",
        "control_kind",
    ],
)
def test_stage2_rejects_every_normative_registry_mutation(
    configs_dir: Path, tmp_path: Path, case: str
) -> None:
    payload = _load(configs_dir / "stage2_candidates_v1.json")
    _mutate_stage2(payload, case)

    with pytest.raises(ValueError, match="exact preregistered Stage 2 v1"):
        SemanticConfig.from_path(_write(tmp_path / f"stage2-{case}.json", payload))


@pytest.mark.parametrize(
    "list_path",
    [
        ("layernorm",),
        ("whitening",),
        ("preprojectors",),
        ("checkpoint_averaging",),
        ("feature_adapter", "candidates"),
        ("feature_adapter", "controls"),
    ],
)
def test_stage2_rejects_duplicate_ids_in_every_registry_list(
    configs_dir: Path,
    tmp_path: Path,
    list_path: tuple[str, ...],
) -> None:
    payload = _load(configs_dir / "stage2_candidates_v1.json")
    target: object = payload
    for key in list_path:
        assert isinstance(target, dict)
        target = target[key]
    assert isinstance(target, list)
    target.append(copy.deepcopy(target[0]))

    with pytest.raises(ValueError, match="duplicate Stage 2 config_id"):
        SemanticConfig.from_path(_write(tmp_path / "stage2-duplicate.json", payload))


def test_stage2_seals_architectures_and_per_candidate_controls(
    configs_dir: Path,
) -> None:
    config = SemanticConfig.from_path(configs_dir / "stage2_candidates_v1.json")
    payload = config.canonical_payload()
    adapter = payload["feature_adapter"]

    assert adapter["dimensions"] == {
        "layer_order": LAYER_ORDER,
        "input_dimension": 3200,
        "output_dimension": 512,
        "layers": 5,
    }
    assert adapter["normalization"] == {
        "kind": "LayerNorm",
        "axis": "last",
        "affine": False,
        "compute_dtype": "float32",
        "eps": 1e-6,
        "applies_to": ["adapter", "global_dense", "matched_projector"],
    }
    assert adapter["identity_architecture"] == {
        "kind": "adapter_off_identity",
        "formula": "h'_l = h_l",
        "added_parameters": 0,
    }
    assert adapter["adapter_architecture"] == {
        "kind": "per_layer_residual_low_rank",
        "formula": "h'_l = h_l + gamma_l * B_l GELU(A_l LayerNorm(h_l))",
        "activation": "GELU",
        "down_projection": "A_l: Linear(D,r,bias=False)",
        "up_projection": "B_l: Linear(r,D,bias=False)",
        "zero_initialized": "B_l",
        "gamma": "per_layer_trainable_scalar",
        "gamma_initial": 1.0,
    }
    assert adapter["global_dense_architecture"] == {
        "kind": "global_dense_flattened_residual_bottleneck",
        "input_layout": "[B,L*D]",
        "formula": (
            "h' = h + gamma * B_global GELU(A_global LayerNorm(h))"
        ),
        "activation": "GELU",
        "down_projection": "A_global: Linear(L*D,r,bias=False)",
        "up_projection": "B_global: Linear(r,L*D,bias=False)",
        "zero_initialized": "B_global",
        "gamma": "single_trainable_scalar",
        "gamma_initial": 1.0,
    }
    assert adapter["matched_projector_architecture"] == {
        "kind": "per_layer_residual_output_space",
        "formula": "q'_l = q_l + gamma_l * Q_l GELU(R_l LayerNorm(h_l))",
        "activation": "GELU",
        "down_projection": "R_l: Linear(D,m_l,bias=False)",
        "up_projection": "Q_l: Linear(m_l,O,bias=False)",
        "zero_initialized": "Q_l",
        "gamma": "per_layer_trainable_scalar",
        "gamma_initial": 1.0,
    }
    assert adapter["controls"] == [
        {
            "config_id": "s2-adapter-identity-control",
            "kind": "identity",
            "architecture": "identity_architecture",
            "parameter_match_tolerance": 0.0,
        },
        {
            "config_id": "s2-adapter-global-dense-control",
            "kind": "global_dense",
            "architecture": "global_dense_architecture",
            "parameter_match_tolerance": 0.01,
        },
        {
            "config_id": "s2-adapter-matched-projector-control",
            "kind": "matched_projector",
            "architecture": "matched_projector_architecture",
            "parameter_match_tolerance": 0.01,
        },
    ]

    expected_order = [
        (rank, ratio) for rank in (8, 16, 32) for ratio in (0.05, 0.1)
    ]
    assert [
        (entry["rank"], entry["learning_rate_ratio"])
        for entry in adapter["candidates"]
    ] == expected_order
    for candidate in adapter["candidates"]:
        rank = candidate["rank"]
        ratio = candidate["learning_rate_ratio"]
        expected = MATCHED_BY_RANK[rank]
        assert candidate["adapter_parameters"] == expected["adapter_parameters"]
        assert candidate["control_bindings"] == {
            "identity": {
                "config_id": "s2-adapter-identity-control",
            },
            "global_dense": {
                "config_id": "s2-adapter-global-dense-control",
                "rank": rank,
                "learning_rate_ratio": ratio,
                "parameters": expected["global_dense_parameters"],
            },
            "matched_projector": {
                "config_id": "s2-adapter-matched-projector-control",
                "rank": rank,
                "learning_rate_ratio": ratio,
                "widths": expected["widths"],
                "adapter_parameters": expected["adapter_parameters"],
                "control_parameters": expected["projector_parameters"],
                "absolute_parameter_error": expected["absolute_error"],
                "relative_parameter_error": expected["relative_error"],
            },
        }
        assert (
            candidate["control_bindings"]["matched_projector"][
                "relative_parameter_error"
            ]
            < 0.01
        )

    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    assert config.sha256 == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert (
        config.sha256
        == "f7a34ae1ed66c0ac574669ad393645052c8c90f8c8bfc78a747094429415f263"
    )
    changed = copy.deepcopy(payload)
    changed["feature_adapter"]["normalization"]["eps"] = 2e-6
    changed_canonical = json.dumps(
        changed, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    assert hashlib.sha256(changed_canonical.encode("utf-8")).hexdigest() != config.sha256


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("feature_adapter", "dimensions", "layer_order"), [20, 24, 28, 32, 37]),
        (("feature_adapter", "normalization", "eps"), 2e-6),
        (("feature_adapter", "normalization", "affine"), True),
        (
            ("feature_adapter", "global_dense_architecture", "input_layout"),
            "[B,L,D]",
        ),
        (
            ("feature_adapter", "matched_projector_architecture", "zero_initialized"),
            "R_l",
        ),
        (("feature_adapter", "candidates", 0, "adapter_parameters"), 1),
        (
            (
                "feature_adapter",
                "candidates",
                0,
                "control_bindings",
                "global_dense",
                "parameters",
            ),
            1,
        ),
        (
            (
                "feature_adapter",
                "candidates",
                0,
                "control_bindings",
                "matched_projector",
                "widths",
            ),
            [1, 1, 1, 1, 1],
        ),
        (
            (
                "feature_adapter",
                "candidates",
                0,
                "control_bindings",
                "matched_projector",
                "relative_parameter_error",
            ),
            0.02,
        ),
    ],
)
def test_stage2_rejects_locked_architecture_and_binding_mutations(
    configs_dir: Path,
    tmp_path: Path,
    path: tuple[object, ...],
    replacement: object,
) -> None:
    payload = _load(configs_dir / "stage2_candidates_v1.json")
    target: object = payload
    for key in path[:-1]:
        if isinstance(key, int):
            assert isinstance(target, list) and len(target) > key, (
                f"sealed schema missing list path {path}"
            )
            target = target[key]
        else:
            assert isinstance(target, dict) and key in target, (
                f"sealed schema missing object path {path}"
            )
            target = target[key]
    final = path[-1]
    if isinstance(final, int):
        assert isinstance(target, list) and len(target) > final
        target[final] = replacement
    else:
        assert isinstance(target, dict) and final in target
        target[final] = replacement

    with pytest.raises(ValueError, match="exact preregistered Stage 2 v1"):
        SemanticConfig.from_path(
            _write(tmp_path / "stage2-architecture-mutation.json", payload)
        )


@pytest.mark.parametrize("missing_key", sorted(REQUIRED_INPUT_HASHES))
def test_resolved_run_requires_every_provenance_hash(
    configs_dir: Path,
    missing_key: str,
) -> None:
    protocol = ProtocolConfig.from_path(configs_dir / "protocol_v1.json")
    inputs = dict(REQUIRED_INPUT_HASHES)
    inputs.pop(missing_key)

    with pytest.raises(ValueError, match=missing_key):
        resolve_run_config(protocol, _candidate(), inputs)


def test_resolved_run_allows_additional_safe_named_hashes(configs_dir: Path) -> None:
    protocol = ProtocolConfig.from_path(configs_dir / "protocol_v1.json")
    inputs = dict(REQUIRED_INPUT_HASHES)
    inputs["environment_sha256"] = "5" * 64

    resolved = resolve_run_config(protocol, _candidate(), inputs)

    assert dict(resolved.input_hashes) == dict(sorted(inputs.items()))
