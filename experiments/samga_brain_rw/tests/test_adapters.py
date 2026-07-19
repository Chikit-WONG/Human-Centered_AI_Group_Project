from __future__ import annotations

import io
from collections.abc import Callable

import pytest
import torch

from samga_brain_rw import adapters


def _api():
    return adapters


def _parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _assert_float32_layernorm(module: torch.nn.Module) -> None:
    assert isinstance(module, torch.nn.LayerNorm)
    assert module.elementwise_affine is False
    assert module.eps == 1e-6
    assert tuple(module.normalized_shape)
    assert not tuple(module.parameters())


def test_residual_adapter_is_exact_identity_with_locked_structure() -> None:
    adapters = _api()
    torch.manual_seed(7)
    model = adapters.ResidualFeatureAdapter(
        hidden_size=3200,
        rank=8,
        layers=5,
    )
    features = torch.randn(2, 5, 3200, dtype=torch.float16)
    seen_dtypes: list[torch.dtype] = []
    handles = [
        norm.register_forward_pre_hook(
            lambda _module, args: seen_dtypes.append(args[0].dtype)
        )
        for norm in model.norms
    ]

    output = model(features)

    for handle in handles:
        handle.remove()
    assert torch.equal(output, features)
    assert output.dtype == features.dtype
    assert seen_dtypes == [torch.float32] * 5
    assert len(model.A) == len(model.B) == len(model.norms) == 5
    assert all(layer.bias is None for layer in (*model.A, *model.B))
    assert all(torch.count_nonzero(layer.weight) == 0 for layer in model.B)
    assert torch.equal(model.gamma, torch.ones(5))
    assert model.gamma.requires_grad
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    for norm in model.norms:
        _assert_float32_layernorm(norm)
    assert _parameter_count(model) == 256_005
    assert model.parameter_count == 256_005


def test_residual_adapter_first_step_gradient_reaches_b_then_a() -> None:
    adapters = _api()
    torch.manual_seed(11)
    model = adapters.ResidualFeatureAdapter(8, 3, 2)
    features = torch.randn(4, 2, 8)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    model(features).square().sum().backward()

    assert all(layer.weight.grad is not None for layer in model.B)
    assert all(torch.count_nonzero(layer.weight.grad) > 0 for layer in model.B)
    assert all(layer.weight.grad is not None for layer in model.A)
    assert all(torch.count_nonzero(layer.weight.grad) == 0 for layer in model.A)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    assert not torch.equal(model(features), features)

    model(features).square().sum().backward()

    assert all(layer.weight.grad is not None for layer in model.A)
    assert all(torch.count_nonzero(layer.weight.grad) > 0 for layer in model.A)


@pytest.mark.parametrize(
    ("rank", "expected"),
    [(8, 256_005), (16, 512_005), (32, 1_024_005)],
)
def test_residual_adapter_parameter_counts(rank: int, expected: int) -> None:
    adapters = _api()
    model = adapters.ResidualFeatureAdapter(3200, rank, 5)
    assert _parameter_count(model) == expected
    assert model.parameter_count == expected


def test_dense_control_matches_budget_and_is_distinct_identity() -> None:
    adapters = _api()
    target = 256_005
    model = adapters.DenseBottleneckControl(3200, target, 5)
    features = torch.randn(2, 5, 3200, dtype=torch.float16)
    seen_dtypes: list[torch.dtype] = []
    handle = model.norm.register_forward_pre_hook(
        lambda _module, args: seen_dtypes.append(args[0].dtype)
    )

    output = model(features)

    handle.remove()
    assert torch.equal(output, features)
    assert output.dtype == features.dtype
    assert seen_dtypes == [torch.float32]
    assert model.rank == 8
    assert model.A_global.in_features == 16_000
    assert model.A_global.out_features == 8
    assert model.B_global.in_features == 8
    assert model.B_global.out_features == 16_000
    assert model.A_global.bias is None
    assert model.B_global.bias is None
    assert torch.count_nonzero(model.B_global.weight) == 0
    assert model.gamma.shape == torch.Size([])
    assert model.gamma.item() == 1.0
    _assert_float32_layernorm(model.norm)
    assert _parameter_count(model) == 256_001
    assert model.parameter_match.target_parameters == target
    assert model.parameter_match.control_parameters == 256_001
    assert model.parameter_match.absolute_error == 4
    assert model.parameter_match.relative_error < 0.01


def test_matched_projector_matches_budget_and_is_output_identity() -> None:
    adapters = _api()
    target = 256_005
    model = adapters.MatchedPerLayerProjectorControl(
        input_dim=3200,
        output_dim=512,
        layers=5,
        target_parameters=target,
    )
    hidden = torch.randn(2, 5, 3200, dtype=torch.float16)
    projected = torch.randn(2, 5, 512, dtype=torch.float16)
    seen_dtypes: list[torch.dtype] = []
    handles = [
        norm.register_forward_pre_hook(
            lambda _module, args: seen_dtypes.append(args[0].dtype)
        )
        for norm in model.norms
    ]

    output = model(hidden, projected)

    for handle in handles:
        handle.remove()
    assert torch.equal(output, projected)
    assert output.dtype == projected.dtype
    assert seen_dtypes == [torch.float32] * 5
    assert model.widths == (14, 14, 14, 14, 13)
    assert all(layer.bias is None for layer in (*model.R, *model.Q))
    assert all(torch.count_nonzero(layer.weight) == 0 for layer in model.Q)
    assert torch.equal(model.gamma, torch.ones(5))
    for norm in model.norms:
        _assert_float32_layernorm(norm)
    assert _parameter_count(model) == 256_133
    assert model.parameter_match.target_parameters == target
    assert model.parameter_match.control_parameters == 256_133
    assert model.parameter_match.absolute_error == 128
    assert model.parameter_match.relative_error == pytest.approx(128 / target)
    assert model.parameter_match.relative_error < 0.01


@pytest.mark.parametrize(
    ("rank", "widths", "adapter_count", "projector_count", "error"),
    [
        (8, (14, 14, 14, 14, 13), 256_005, 256_133, 128),
        (16, (28, 28, 28, 27, 27), 512_005, 512_261, 256),
        (32, (56, 55, 55, 55, 55), 1_024_005, 1_024_517, 512),
    ],
)
def test_locked_per_layer_widths_and_errors(
    rank: int,
    widths: tuple[int, ...],
    adapter_count: int,
    projector_count: int,
    error: int,
) -> None:
    adapters = _api()
    assert adapters.match_per_layer_widths(rank) == widths
    assert 3712 * sum(widths) + 5 == projector_count
    assert 32_000 * rank + 5 == adapter_count
    assert abs(projector_count - adapter_count) == error
    assert error / adapter_count < 0.01


@pytest.mark.parametrize("rank", [8, 16, 32])
def test_dense_width_match_is_exact_grid_rank(rank: int) -> None:
    adapters = _api()
    target = 32_000 * rank + 5
    assert adapters.match_dense_width(3200, 5, target) == rank
    actual = 2 * 3200 * 5 * rank + 1
    assert abs(actual - target) / target < 0.01


def test_dense_width_rounds_half_ties_down() -> None:
    adapters = _api()
    assert adapters.match_dense_width(1, 1, 6, tolerance=0.2) == 2


@pytest.mark.parametrize(
    ("function_name", "args", "message"),
    [
        ("match_dense_width", (0, 5, 256_005), "hidden_size"),
        ("match_dense_width", (3200, 0, 256_005), "layers"),
        ("match_dense_width", (3200, 5, 0), "target_parameters"),
        ("match_dense_width", (3200, 5, 256_005, -0.1), "tolerance"),
        ("match_per_layer_widths", (0,), "adapter_rank"),
        ("match_per_layer_widths", (8, (20, 20)), "unique"),
    ],
)
def test_match_helpers_reject_invalid_contracts(
    function_name: str,
    args: tuple[object, ...],
    message: str,
) -> None:
    adapters = _api()
    with pytest.raises((TypeError, ValueError), match=message):
        getattr(adapters, function_name)(*args)


@pytest.mark.parametrize(
    "factory",
    [
        lambda api: api.ResidualFeatureAdapter(8, 2, 2),
        lambda api: api.DenseBottleneckControl(8, 65, 2),
    ],
)
def test_feature_models_reject_shape_dtype_and_nonfinite(
    factory: Callable[[object], torch.nn.Module],
) -> None:
    adapters = _api()
    model = factory(adapters)
    with pytest.raises(ValueError, match="shape"):
        model(torch.randn(2, 1, 8))
    with pytest.raises(TypeError, match="floating"):
        model(torch.ones(2, 2, 8, dtype=torch.int64))
    invalid = torch.randn(2, 2, 8)
    invalid[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model(invalid)


def test_matched_projector_rejects_shape_dtype_and_nonfinite() -> None:
    adapters = _api()
    model = adapters.MatchedPerLayerProjectorControl(8, 8, 2, 66)
    hidden = torch.randn(2, 2, 8)
    projected = torch.randn(2, 2, 8)
    with pytest.raises(ValueError, match="hidden.*shape"):
        model(hidden[:, :1], projected)
    with pytest.raises(ValueError, match="projected.*shape"):
        model(hidden, projected[:, :1])
    with pytest.raises(TypeError, match="same dtype"):
        model(hidden, projected.double())
    invalid = projected.clone()
    invalid[0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        model(hidden, invalid)


@pytest.mark.parametrize("kind", ["residual", "dense", "projector"])
def test_adapters_save_reload_exactly(kind: str) -> None:
    adapters = _api()
    torch.manual_seed(19)
    if kind == "residual":
        model = adapters.ResidualFeatureAdapter(8, 2, 2)
        reloaded = adapters.ResidualFeatureAdapter(8, 2, 2)
        with torch.no_grad():
            for layer in model.B:
                layer.weight.normal_()
        inputs = (torch.randn(3, 2, 8),)
    elif kind == "dense":
        model = adapters.DenseBottleneckControl(8, 65, 2)
        reloaded = adapters.DenseBottleneckControl(8, 65, 2)
        with torch.no_grad():
            model.B_global.weight.normal_()
        inputs = (torch.randn(3, 2, 8),)
    else:
        model = adapters.MatchedPerLayerProjectorControl(8, 8, 2, 66)
        reloaded = adapters.MatchedPerLayerProjectorControl(8, 8, 2, 66)
        with torch.no_grad():
            for layer in model.Q:
                layer.weight.normal_()
        inputs = (torch.randn(3, 2, 8), torch.randn(3, 2, 8))
    before = model(*inputs)
    serialized = io.BytesIO()
    torch.save(model.state_dict(), serialized)
    serialized.seek(0)

    reloaded.load_state_dict(torch.load(serialized, weights_only=True))

    assert torch.equal(reloaded(*inputs), before)


def test_exact_adapter_grid_and_matched_controls() -> None:
    adapters = _api()
    grid = adapters.build_stage2_adapter_grid()
    assert [
        (entry["config_id"], entry["rank"], entry["learning_rate_ratio"])
        for entry in grid
    ] == [
        (f"s2-adapter-r{rank}-lr{ratio:.2f}", rank, ratio)
        for rank in (8, 16, 32)
        for ratio in (0.05, 0.10)
    ]
    assert len(grid) == 6
    for entry in grid:
        rank = entry["rank"]
        target = 32_000 * rank + 5
        controls = entry["control_bindings"]
        assert entry["adapter_parameters"] == target
        assert set(controls) == {"identity", "global_dense", "matched_projector"}
        assert controls["identity"] == {
            "config_id": "s2-adapter-identity-control"
        }
        assert controls["global_dense"] == {
            "config_id": "s2-adapter-global-dense-control",
            "rank": rank,
            "learning_rate_ratio": entry["learning_rate_ratio"],
            "parameters": 32_000 * rank + 1,
        }
        projector = controls["matched_projector"]
        assert projector["widths"] == list(
            adapters.match_per_layer_widths(rank)
        )
        assert projector["adapter_parameters"] == target
        assert projector["control_parameters"] == 3712 * sum(
            projector["widths"]
        ) + 5
        assert projector["absolute_parameter_error"] == abs(
            projector["control_parameters"] - target
        )
        assert projector["relative_parameter_error"] < 0.01


def test_one_factor_only_rejects_combinations() -> None:
    adapters = _api()
    assert adapters.require_one_factor_only({}) is None
    assert adapters.require_one_factor_only({"feature_adapter": True}) == (
        "feature_adapter"
    )
    with pytest.raises(ValueError, match="one Stage 2 factor"):
        adapters.require_one_factor_only(
            {"feature_adapter": True, "whitening": True}
        )
    with pytest.raises(ValueError, match="unknown Stage 2 factor"):
        adapters.require_one_factor_only({"surprise": True})
    with pytest.raises(TypeError, match="boolean"):
        adapters.require_one_factor_only({"feature_adapter": 1})


def _summary(
    api,
    config_id: str,
    top1: float,
    top5: float,
    cost: float,
    parameters: int,
    digest: str,
):
    return api.Stage2Summary(
        config_id=config_id,
        macro_top1=top1,
        macro_top5=top5,
        inference_cost=cost,
        added_parameters=parameters,
        resolved_sha256=digest,
    )


def test_resolved_hash_aliases_are_not_independent_candidates() -> None:
    adapters = _api()
    shared = "a" * 64
    distinct = "b" * 64
    aliases = adapters.resolve_artifact_aliases(
        {
            "s2-whitening-off": shared,
            "s2-preproj-shared": shared,
            "s2-raw-epoch60-control": shared,
            "s2-whitening-on": distinct,
        }
    )
    assert aliases == {
        "s2-preproj-shared": "s2-preproj-shared",
        "s2-raw-epoch60-control": "s2-preproj-shared",
        "s2-whitening-off": "s2-preproj-shared",
        "s2-whitening-on": "s2-whitening-on",
    }
    summaries = [
        _summary(adapters, "s2-whitening-off", 0.5, 0.7, 2.0, 0, shared),
        _summary(adapters, "s2-preproj-shared", 0.5, 0.7, 2.0, 0, shared),
        _summary(adapters, "s2-whitening-on", 0.6, 0.8, 3.0, 0, distinct),
    ]
    collapsed = adapters.collapse_stage2_aliases(summaries)
    assert [entry.config_id for entry in collapsed] == [
        "s2-preproj-shared",
        "s2-whitening-on",
    ]


def test_aliases_with_inconsistent_metrics_are_rejected() -> None:
    adapters = _api()
    digest = "c" * 64
    summaries = [
        _summary(adapters, "alias-a", 0.5, 0.7, 2.0, 0, digest),
        _summary(adapters, "alias-b", 0.6, 0.7, 2.0, 0, digest),
    ]
    with pytest.raises(ValueError, match="aliased summaries must be identical"):
        adapters.collapse_stage2_aliases(summaries)


def test_stage2_gate_requires_all_three_adapter_controls() -> None:
    adapters = _api()
    assert adapters.adapter_gate_eligible(
        {"identity": True, "global_dense": True, "matched_projector": True}
    )
    assert not adapters.adapter_gate_eligible(
        {"identity": True, "global_dense": False, "matched_projector": True}
    )
    with pytest.raises(ValueError, match="exactly"):
        adapters.adapter_gate_eligible({"identity": True})


def test_stage2_selectors_apply_locked_tie_breaks_and_gate() -> None:
    adapters = _api()
    controls = [
        _summary(adapters, "control-a", 0.60, 0.80, 1.0, 0, "1" * 64),
        _summary(adapters, "control-b", 0.60, 0.81, 4.0, 0, "2" * 64),
        _summary(adapters, "control-c", 0.60, 0.81, 2.0, 0, "3" * 64),
        _summary(adapters, "control-d", 0.60, 0.81, 2.0, 0, "4" * 64),
    ]
    assert adapters.select_strongest_control(controls).config_id == "control-c"

    candidates = [
        _summary(
            adapters,
            "s2-adapter-r8-lr0.05",
            0.70,
            0.86,
            3.0,
            256_005,
            "5" * 64,
        ),
        _summary(
            adapters,
            "s2-adapter-r8-lr0.10",
            0.71,
            0.90,
            3.0,
            256_005,
            "6" * 64,
        ),
        _summary(
            adapters,
            "s2-adapter-r16-lr0.05",
            0.70,
            0.86,
            3.0,
            512_005,
            "7" * 64,
        ),
        _summary(
            adapters,
            "s2-adapter-r16-lr0.10",
            0.70,
            0.86,
            3.0,
            512_005,
            "8" * 64,
        ),
    ]
    gates = {
        "s2-adapter-r8-lr0.05": {
            "identity": True,
            "global_dense": True,
            "matched_projector": True,
        },
        "s2-adapter-r8-lr0.10": {
            "identity": True,
            "global_dense": False,
            "matched_projector": True,
        },
        "s2-adapter-r16-lr0.05": {
            "identity": True,
            "global_dense": True,
            "matched_projector": True,
        },
        "s2-adapter-r16-lr0.10": {
            "identity": True,
            "global_dense": True,
            "matched_projector": True,
        },
    }
    selected = adapters.select_stage2_candidate(candidates, gates)
    assert selected.config_id == "s2-adapter-r8-lr0.05"
