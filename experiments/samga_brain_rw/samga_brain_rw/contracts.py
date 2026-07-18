"""Code-owned oracles for the sealed Stage 1 and Stage 2 v1 registries."""

from __future__ import annotations


_ZSCORE_FORMULA = (
    "alpha * zscore_ddof0(S_I) + (1 - alpha) * zscore_ddof0(S_C)"
)
_TEMPERATURE_FORMULA = "alpha * S_I / T_I + (1 - alpha) * S_C / T_C"
_RRF_FORMULA = "w / (k + rank_I) + (1 - w) / (k + rank_C)"
_ADAPTER_FORMULA = "h'_l = h_l + gamma_l * B_l GELU(A_l LayerNorm(h_l))"


def stage1_v1_payload() -> dict[str, object]:
    """Return the exact ordered 47-candidate Stage 1 v1 document."""
    candidates: list[dict[str, object]] = []
    for index in range(11):
        candidates.append(
            {
                "config_id": f"s1-z-a{index * 10:03d}",
                "family": "zscore_convex",
                "formula": _ZSCORE_FORMULA,
                "alpha": index / 10,
            }
        )

    temperatures = ((0.5, "050"), (1.0, "100"), (2.0, "200"))
    alphas = ((0.25, "025"), (0.5, "050"), (0.75, "075"))
    for internvit_temperature, internvit_token in temperatures:
        for clip_temperature, clip_token in temperatures:
            for alpha, alpha_token in alphas:
                candidates.append(
                    {
                        "config_id": (
                            f"s1-temp-ti{internvit_token}-tc{clip_token}-"
                            f"a{alpha_token}"
                        ),
                        "family": "temperature_convex",
                        "formula": _TEMPERATURE_FORMULA,
                        "internvit_temperature": internvit_temperature,
                        "clip_temperature": clip_temperature,
                        "alpha": alpha,
                    }
                )

    for k in (10, 30, 60):
        for weight, weight_token in alphas:
            candidates.append(
                {
                    "config_id": f"s1-rrf-k{k:03d}-w{weight_token}",
                    "family": "rrf",
                    "formula": _RRF_FORMULA,
                    "k": k,
                    "internvit_weight": weight,
                    "rank_origin": 1,
                    "score_tie_break": "gallery_id_utf8_bytewise",
                    "final_tie_break": "gallery_id_utf8_bytewise",
                }
            )

    return {
        "schema_version": 1,
        "config_type": "stage1_fusion",
        "config_id": "stage1_fusion_v1",
        "selection": {
            "scope": "val-dev",
            "retrieval": "standard_independent_cosine",
            "zscore_variance": "population_ddof0",
            "constant_row": "all_zero",
            "temperature_softmax": False,
            "branch_score_tie_break": "gallery_id_utf8_bytewise",
            "final_score_tie_break": "gallery_id_utf8_bytewise",
            "metric_tie_break": ["lower_compute", "config_id"],
        },
        "formulas": {
            "zscore_convex": _ZSCORE_FORMULA,
            "temperature_convex": _TEMPERATURE_FORMULA,
            "rrf": _RRF_FORMULA,
        },
        "candidates": candidates,
    }


_MATCHED_BY_RANK = {
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


def _adapter_candidate(rank: int, ratio: float) -> dict[str, object]:
    matched = _MATCHED_BY_RANK[rank]
    return {
        "config_id": f"s2-adapter-r{rank}-lr{ratio:.2f}",
        "rank": rank,
        "learning_rate_ratio": ratio,
        "adapter_parameters": matched["adapter_parameters"],
        "control_bindings": {
            "identity": {
                "config_id": "s2-adapter-identity-control",
            },
            "global_dense": {
                "config_id": "s2-adapter-global-dense-control",
                "rank": rank,
                "learning_rate_ratio": ratio,
                "parameters": matched["global_dense_parameters"],
            },
            "matched_projector": {
                "config_id": "s2-adapter-matched-projector-control",
                "rank": rank,
                "learning_rate_ratio": ratio,
                "widths": list(matched["widths"]),
                "adapter_parameters": matched["adapter_parameters"],
                "control_parameters": matched["projector_parameters"],
                "absolute_parameter_error": matched["absolute_error"],
                "relative_parameter_error": matched["relative_error"],
            },
        },
    }


def stage2_v1_payload() -> dict[str, object]:
    """Return the exact one-factor Stage 2 registry and matched controls."""
    return {
        "schema_version": 1,
        "config_type": "stage2_candidates",
        "config_id": "stage2_candidates_v1",
        "combination_policy": "one_factor_only_no_post_hoc_combinations",
        "layernorm": [
            {
                "config_id": "s2-layernorm-off",
                "enabled": False,
                "role": "baseline",
                "fit_scope": "none",
            },
            {
                "config_id": "s2-layernorm-on",
                "enabled": True,
                "role": "candidate",
                "fit_scope": "runtime_no_fit",
            },
        ],
        "whitening": [
            {
                "config_id": "s2-whitening-off",
                "enabled": False,
                "role": "baseline",
                "fit_scope": "none",
            },
            {
                "config_id": "s2-whitening-on",
                "enabled": True,
                "role": "candidate",
                "fit_scope": "train_only",
            },
        ],
        "preprojectors": [
            {
                "config_id": "s2-preproj-shared",
                "mode": "shared",
                "role": "baseline",
            },
            {
                "config_id": "s2-preproj-separate",
                "mode": "separate_per_layer",
                "role": "candidate",
            },
        ],
        "checkpoint_averaging": [
            {
                "config_id": "s2-raw-epoch60-control",
                "method": "raw",
                "window": 1,
                "epochs": [60],
                "role": "control",
            },
            {
                "config_id": "s2-avg-last5",
                "method": "arithmetic",
                "window": 5,
                "epochs": [56, 57, 58, 59, 60],
                "role": "candidate",
            },
            {
                "config_id": "s2-avg-last10",
                "method": "arithmetic",
                "window": 10,
                "epochs": [51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
                "role": "candidate",
            },
            {
                "config_id": "s2-swa-last5",
                "method": "swa",
                "window": 5,
                "epochs": [56, 57, 58, 59, 60],
                "role": "candidate",
            },
            {
                "config_id": "s2-swa-last10",
                "method": "swa",
                "window": 10,
                "epochs": [51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
                "role": "candidate",
            },
        ],
        "feature_adapter": {
            "label": "cached InternViT feature adapter",
            "formula": _ADAPTER_FORMULA,
            "zero_initialized": "B_l",
            "gamma_initial": 1.0,
            "dimensions": {
                "layer_order": [20, 24, 28, 32, 36],
                "input_dimension": 3200,
                "output_dimension": 512,
                "layers": 5,
            },
            "normalization": {
                "kind": "LayerNorm",
                "axis": "last",
                "affine": False,
                "compute_dtype": "float32",
                "eps": 1e-6,
                "applies_to": ["adapter", "global_dense", "matched_projector"],
            },
            "identity_architecture": {
                "kind": "adapter_off_identity",
                "formula": "h'_l = h_l",
                "added_parameters": 0,
            },
            "adapter_architecture": {
                "kind": "per_layer_residual_low_rank",
                "formula": _ADAPTER_FORMULA,
                "activation": "GELU",
                "down_projection": "A_l: Linear(D,r,bias=False)",
                "up_projection": "B_l: Linear(r,D,bias=False)",
                "zero_initialized": "B_l",
                "gamma": "per_layer_trainable_scalar",
                "gamma_initial": 1.0,
            },
            "global_dense_architecture": {
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
            },
            "matched_projector_architecture": {
                "kind": "per_layer_residual_output_space",
                "formula": (
                    "q'_l = q_l + gamma_l * Q_l GELU(R_l LayerNorm(h_l))"
                ),
                "activation": "GELU",
                "down_projection": "R_l: Linear(D,m_l,bias=False)",
                "up_projection": "Q_l: Linear(m_l,O,bias=False)",
                "zero_initialized": "Q_l",
                "gamma": "per_layer_trainable_scalar",
                "gamma_initial": 1.0,
            },
            "ranks": [8, 16, 32],
            "learning_rate_ratios": [0.05, 0.1],
            "candidates": [
                _adapter_candidate(rank, ratio)
                for rank in (8, 16, 32)
                for ratio in (0.05, 0.1)
            ],
            "controls": [
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
            ],
            "tie_break": [
                "higher_val_dev_top5",
                "lower_parameter_count",
                "config_id",
            ],
        },
    }
