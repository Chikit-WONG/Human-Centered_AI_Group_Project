"""Residual cached-feature adapters and preregistered matched controls.

All normalization is non-affine LayerNorm evaluated in float32 with
``eps=1e-6``.  Zero-initialized up projections make every module an exact
identity at construction while preserving the first-step gradient into that
up projection.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real

import torch
import torch.nn.functional as F
from torch import Tensor, nn


LAYER_IDS = (20, 24, 28, 32, 36)
ADAPTER_RANKS = (8, 16, 32)
LEARNING_RATE_RATIOS = (0.05, 0.10)
ADAPTER_CONTROL_KINDS = ("identity", "global_dense", "matched_projector")
STAGE2_FACTORS = frozenset(
    {
        "layernorm",
        "whitening",
        "preprojectors",
        "checkpoint_averaging",
        "feature_adapter",
    }
)

_LOCKED_INPUT_DIM = 3200
_LOCKED_OUTPUT_DIM = 512
_LAYER_NORM_EPS = 1e-6
_PARAMETER_TOLERANCE = 0.01
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _positive_integer(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _tolerance(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("tolerance must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or result >= 1.0:
        raise ValueError("tolerance must be finite and in [0, 1)")
    return result


def _round_half_down_ratio(numerator: int, denominator: int) -> int:
    """Round a non-negative rational to nearest, resolving halves downward."""

    if numerator < 0 or denominator <= 0:
        raise ValueError("rounding ratio must be non-negative with denominator > 0")
    quotient, remainder = divmod(numerator, denominator)
    return quotient + int(2 * remainder > denominator)


@dataclass(frozen=True)
class ParameterMatch:
    """Recorded comparison between a control and its adapter budget."""

    target_parameters: int
    control_parameters: int
    absolute_error: int
    relative_error: float


def _parameter_match(
    target_parameters: int,
    control_parameters: int,
    tolerance: float,
) -> ParameterMatch:
    absolute_error = abs(control_parameters - target_parameters)
    relative_error = absolute_error / target_parameters
    if relative_error > tolerance:
        raise ValueError(
            "control parameter-count relative error exceeds tolerance: "
            f"{relative_error:.12g} > {tolerance:.12g}"
        )
    return ParameterMatch(
        target_parameters=target_parameters,
        control_parameters=control_parameters,
        absolute_error=absolute_error,
        relative_error=relative_error,
    )


def match_dense_width(
    hidden_size: int,
    layers: int,
    target_parameters: int,
    tolerance: float = _PARAMETER_TOLERANCE,
) -> int:
    """Return the half-ties-down global bottleneck width within tolerance."""

    dimension = _positive_integer(hidden_size, "hidden_size")
    layer_count = _positive_integer(layers, "layers")
    target = _positive_integer(target_parameters, "target_parameters")
    allowed_error = _tolerance(tolerance)
    denominator = 2 * dimension * layer_count
    width = _round_half_down_ratio(target - 1, denominator)
    if width <= 0:
        raise ValueError("matched dense width must be positive")
    actual = denominator * width + 1
    _parameter_match(target, actual, allowed_error)
    return width


def match_per_layer_widths(
    adapter_rank: int,
    layer_ids: Sequence[int] = LAYER_IDS,
    tolerance: float = _PARAMETER_TOLERANCE,
) -> tuple[int, ...]:
    """Match the locked 3200-to-512 projector budget by layer order."""

    rank = _positive_integer(adapter_rank, "adapter_rank")
    if isinstance(layer_ids, (str, bytes, bytearray)):
        raise TypeError("layer_ids must be a sequence of integers")
    identifiers = tuple(layer_ids)
    if not identifiers or any(
        type(identifier) is not int or identifier <= 0
        for identifier in identifiers
    ):
        raise ValueError("layer_ids must contain positive integers")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("layer_ids must be unique")
    allowed_error = _tolerance(tolerance)
    layer_count = len(identifiers)
    target = layer_count * (2 * _LOCKED_INPUT_DIM * rank + 1)
    coefficient = _LOCKED_INPUT_DIM + _LOCKED_OUTPUT_DIM
    total_width = _round_half_down_ratio(
        target - layer_count,
        coefficient,
    )
    base, remainder = divmod(total_width, layer_count)
    if base <= 0:
        raise ValueError("matched projector widths must be positive")
    widths = tuple(
        base + int(index < remainder) for index in range(layer_count)
    )
    actual = coefficient * sum(widths) + layer_count
    _parameter_match(target, actual, allowed_error)
    return widths


def _matched_widths_for_target(
    input_dim: int,
    output_dim: int,
    layers: int,
    target_parameters: int,
    tolerance: float,
) -> tuple[tuple[int, ...], ParameterMatch]:
    coefficient = input_dim + output_dim
    if target_parameters <= layers:
        raise ValueError("target_parameters is too small for per-layer gamma")
    total_width = _round_half_down_ratio(
        target_parameters - layers,
        coefficient,
    )
    base, remainder = divmod(total_width, layers)
    if base <= 0:
        raise ValueError("matched projector widths must be positive")
    widths = tuple(base + int(index < remainder) for index in range(layers))
    actual = coefficient * sum(widths) + layers
    return widths, _parameter_match(target_parameters, actual, tolerance)


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _validate_feature_tensor(
    value: object,
    name: str,
    layers: int,
    dimension: int,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating dtype")
    if value.ndim != 3 or value.shape[0] <= 0 or tuple(value.shape[1:]) != (
        layers,
        dimension,
    ):
        raise ValueError(
            f"{name} shape must be [B, {layers}, {dimension}] with B > 0"
        )
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")
    return value


def _finite_output(value: Tensor, name: str) -> Tensor:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values")
    return value


def _norm(dimension: int) -> nn.LayerNorm:
    return nn.LayerNorm(
        dimension,
        eps=_LAYER_NORM_EPS,
        elementwise_affine=False,
    )


def _linear(input_dim: int, output_dim: int) -> nn.Linear:
    return nn.Linear(
        input_dim,
        output_dim,
        bias=False,
        dtype=torch.float32,
    )


class ResidualFeatureAdapter(nn.Module):
    """Per-layer low-rank residual adapter over cached InternViT features."""

    def __init__(self, hidden_size: int, rank: int, layers: int) -> None:
        super().__init__()
        self.hidden_size = _positive_integer(hidden_size, "hidden_size")
        self.rank = _positive_integer(rank, "rank")
        self.layers = _positive_integer(layers, "layers")
        self.norms = nn.ModuleList(
            [_norm(self.hidden_size) for _ in range(self.layers)]
        )
        self.A = nn.ModuleList(
            [_linear(self.hidden_size, self.rank) for _ in range(self.layers)]
        )
        self.B = nn.ModuleList(
            [_linear(self.rank, self.hidden_size) for _ in range(self.layers)]
        )
        self.gamma = nn.Parameter(
            torch.ones(self.layers, dtype=torch.float32)
        )
        for layer in self.B:
            nn.init.zeros_(layer.weight)

    @property
    def parameter_count(self) -> int:
        return _parameter_count(self)

    def forward(self, features: Tensor) -> Tensor:
        inputs = _validate_feature_tensor(
            features,
            "features",
            self.layers,
            self.hidden_size,
        )
        outputs: list[Tensor] = []
        for index in range(self.layers):
            layer_input = inputs[:, index, :]
            normalized = self.norms[index](layer_input.float())
            update = self.B[index](F.gelu(self.A[index](normalized)))
            outputs.append(
                layer_input
                + (self.gamma[index] * update).to(dtype=layer_input.dtype)
            )
        return _finite_output(torch.stack(outputs, dim=1), "adapter output")


class DenseBottleneckControl(nn.Module):
    """Global flattened residual bottleneck permitting cross-layer mixing."""

    def __init__(
        self,
        hidden_size: int,
        target_parameters: int,
        layers: int,
    ) -> None:
        super().__init__()
        self.hidden_size = _positive_integer(hidden_size, "hidden_size")
        self.layers = _positive_integer(layers, "layers")
        self.target_parameters = _positive_integer(
            target_parameters,
            "target_parameters",
        )
        self.rank = match_dense_width(
            self.hidden_size,
            self.layers,
            self.target_parameters,
        )
        flattened = self.hidden_size * self.layers
        self.norm = _norm(flattened)
        self.A_global = _linear(flattened, self.rank)
        self.B_global = _linear(self.rank, flattened)
        nn.init.zeros_(self.B_global.weight)
        self.gamma = nn.Parameter(torch.ones((), dtype=torch.float32))
        actual = 2 * flattened * self.rank + 1
        self.parameter_match = _parameter_match(
            self.target_parameters,
            actual,
            _PARAMETER_TOLERANCE,
        )

    @property
    def parameter_count(self) -> int:
        return _parameter_count(self)

    def forward(self, features: Tensor) -> Tensor:
        inputs = _validate_feature_tensor(
            features,
            "features",
            self.layers,
            self.hidden_size,
        )
        flattened = inputs.reshape(inputs.shape[0], -1)
        normalized = self.norm(flattened.float())
        update = self.B_global(F.gelu(self.A_global(normalized)))
        output = flattened + (self.gamma * update).to(dtype=flattened.dtype)
        return _finite_output(
            output.reshape_as(inputs),
            "dense-control output",
        )


class MatchedPerLayerProjectorControl(nn.Module):
    """Parameter-matched per-layer residual branch in output space."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        layers: int,
        target_parameters: int,
    ) -> None:
        super().__init__()
        self.input_dim = _positive_integer(input_dim, "input_dim")
        self.output_dim = _positive_integer(output_dim, "output_dim")
        self.layers = _positive_integer(layers, "layers")
        self.target_parameters = _positive_integer(
            target_parameters,
            "target_parameters",
        )
        self.widths, self.parameter_match = _matched_widths_for_target(
            self.input_dim,
            self.output_dim,
            self.layers,
            self.target_parameters,
            _PARAMETER_TOLERANCE,
        )
        self.norms = nn.ModuleList(
            [_norm(self.input_dim) for _ in range(self.layers)]
        )
        self.R = nn.ModuleList(
            [_linear(self.input_dim, width) for width in self.widths]
        )
        self.Q = nn.ModuleList(
            [_linear(width, self.output_dim) for width in self.widths]
        )
        self.gamma = nn.Parameter(
            torch.ones(self.layers, dtype=torch.float32)
        )
        for layer in self.Q:
            nn.init.zeros_(layer.weight)

    @property
    def parameter_count(self) -> int:
        return _parameter_count(self)

    def forward(self, hidden: Tensor, projected: Tensor) -> Tensor:
        hidden_features = _validate_feature_tensor(
            hidden,
            "hidden",
            self.layers,
            self.input_dim,
        )
        projected_features = _validate_feature_tensor(
            projected,
            "projected",
            self.layers,
            self.output_dim,
        )
        if hidden_features.dtype != projected_features.dtype:
            raise TypeError("hidden and projected must have the same dtype")
        if hidden_features.device != projected_features.device:
            raise ValueError("hidden and projected must be on the same device")
        if hidden_features.shape[0] != projected_features.shape[0]:
            raise ValueError("hidden and projected batch shapes must match")
        outputs: list[Tensor] = []
        for index in range(self.layers):
            normalized = self.norms[index](
                hidden_features[:, index, :].float()
            )
            update = self.Q[index](F.gelu(self.R[index](normalized)))
            baseline = projected_features[:, index, :]
            outputs.append(
                baseline
                + (self.gamma[index] * update).to(dtype=baseline.dtype)
            )
        return _finite_output(
            torch.stack(outputs, dim=1),
            "projector-control output",
        )


def _adapter_parameters(rank: int) -> int:
    return len(LAYER_IDS) * (2 * _LOCKED_INPUT_DIM * rank + 1)


def build_stage2_adapter_grid() -> tuple[dict[str, object], ...]:
    """Return exactly the preregistered 3-rank by 2-LR adapter grid."""

    candidates: list[dict[str, object]] = []
    for rank in ADAPTER_RANKS:
        target = _adapter_parameters(rank)
        dense_rank = match_dense_width(
            _LOCKED_INPUT_DIM,
            len(LAYER_IDS),
            target,
        )
        dense_parameters = (
            2 * _LOCKED_INPUT_DIM * len(LAYER_IDS) * dense_rank + 1
        )
        widths = match_per_layer_widths(rank)
        projector_parameters = (
            (_LOCKED_INPUT_DIM + _LOCKED_OUTPUT_DIM) * sum(widths)
            + len(LAYER_IDS)
        )
        absolute_error = abs(projector_parameters - target)
        relative_error = absolute_error / target
        for ratio in LEARNING_RATE_RATIOS:
            candidates.append(
                {
                    "config_id": f"s2-adapter-r{rank}-lr{ratio:.2f}",
                    "rank": rank,
                    "learning_rate_ratio": ratio,
                    "adapter_parameters": target,
                    "control_bindings": {
                        "identity": {
                            "config_id": "s2-adapter-identity-control",
                        },
                        "global_dense": {
                            "config_id": (
                                "s2-adapter-global-dense-control"
                            ),
                            "rank": dense_rank,
                            "learning_rate_ratio": ratio,
                            "parameters": dense_parameters,
                        },
                        "matched_projector": {
                            "config_id": (
                                "s2-adapter-matched-projector-control"
                            ),
                            "rank": rank,
                            "learning_rate_ratio": ratio,
                            "widths": list(widths),
                            "adapter_parameters": target,
                            "control_parameters": projector_parameters,
                            "absolute_parameter_error": absolute_error,
                            "relative_parameter_error": relative_error,
                        },
                    },
                }
            )
    return tuple(candidates)


def require_one_factor_only(
    enabled_factors: Mapping[str, bool],
) -> str | None:
    """Reject post-hoc combinations and return the sole active factor."""

    if not isinstance(enabled_factors, Mapping):
        raise TypeError("enabled_factors must be a mapping")
    unknown = set(enabled_factors) - STAGE2_FACTORS
    if unknown:
        raise ValueError(f"unknown Stage 2 factor: {sorted(unknown)}")
    for name, enabled in enabled_factors.items():
        if type(enabled) is not bool:
            raise TypeError(f"Stage 2 factor {name} must be boolean")
    active = sorted(name for name, enabled in enabled_factors.items() if enabled)
    if len(active) > 1:
        raise ValueError("exactly zero or one Stage 2 factor may be enabled")
    return active[0] if active else None


def _config_id(value: object) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError("config_id must be non-empty normalized text")
    return value


def _sha256(value: object) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError("resolved_sha256 must be a lowercase SHA-256 digest")
    return value


def _metric(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return result


def _cost(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("inference_cost must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("inference_cost must be finite and non-negative")
    return result


@dataclass(frozen=True)
class Stage2Summary:
    """One six-cell macro summary used by locked Stage 2 selectors."""

    config_id: str
    macro_top1: float
    macro_top5: float
    inference_cost: float
    added_parameters: int
    resolved_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "config_id", _config_id(self.config_id))
        object.__setattr__(
            self,
            "macro_top1",
            _metric(self.macro_top1, "macro_top1"),
        )
        object.__setattr__(
            self,
            "macro_top5",
            _metric(self.macro_top5, "macro_top5"),
        )
        if self.macro_top5 < self.macro_top1:
            raise ValueError("macro_top5 must be at least macro_top1")
        object.__setattr__(self, "inference_cost", _cost(self.inference_cost))
        object.__setattr__(
            self,
            "added_parameters",
            _nonnegative_integer(self.added_parameters, "added_parameters"),
        )
        object.__setattr__(
            self,
            "resolved_sha256",
            _sha256(self.resolved_sha256),
        )


def resolve_artifact_aliases(
    resolved_hashes: Mapping[str, str],
) -> dict[str, str]:
    """Map identical resolved artifacts to one lexicographic canonical ID."""

    if not isinstance(resolved_hashes, Mapping):
        raise TypeError("resolved_hashes must be a mapping")
    groups: dict[str, list[str]] = {}
    for raw_config_id, raw_digest in resolved_hashes.items():
        config_id = _config_id(raw_config_id)
        digest = _sha256(raw_digest)
        groups.setdefault(digest, []).append(config_id)
    aliases: dict[str, str] = {}
    for config_ids in groups.values():
        canonical = min(config_ids)
        for config_id in config_ids:
            aliases[config_id] = canonical
    return {config_id: aliases[config_id] for config_id in sorted(aliases)}


def collapse_stage2_aliases(
    summaries: Sequence[Stage2Summary],
) -> tuple[Stage2Summary, ...]:
    """Collapse identical hashes and reject inconsistent duplicate evidence."""

    if isinstance(summaries, (str, bytes, bytearray)):
        raise TypeError("summaries must be a sequence of Stage2Summary values")
    entries = tuple(summaries)
    if not entries or any(not isinstance(item, Stage2Summary) for item in entries):
        raise ValueError("summaries must contain Stage2Summary values")
    if len({item.config_id for item in entries}) != len(entries):
        raise ValueError("duplicate Stage 2 summary config_id")
    groups: dict[str, list[Stage2Summary]] = {}
    for entry in entries:
        groups.setdefault(entry.resolved_sha256, []).append(entry)
    collapsed: list[Stage2Summary] = []
    for group in groups.values():
        ordered = sorted(group, key=lambda item: item.config_id)
        canonical = ordered[0]
        signature = (
            canonical.macro_top1,
            canonical.macro_top5,
            canonical.inference_cost,
            canonical.added_parameters,
        )
        for alias in ordered[1:]:
            if (
                alias.macro_top1,
                alias.macro_top5,
                alias.inference_cost,
                alias.added_parameters,
            ) != signature:
                raise ValueError("aliased summaries must be identical")
        collapsed.append(canonical)
    return tuple(sorted(collapsed, key=lambda item: item.config_id))


def adapter_gate_eligible(passed_controls: Mapping[str, bool]) -> bool:
    """Return true only when all three preregistered controls pass."""

    if not isinstance(passed_controls, Mapping):
        raise TypeError("passed_controls must be a mapping")
    if set(passed_controls) != set(ADAPTER_CONTROL_KINDS):
        raise ValueError(
            "passed_controls must contain exactly identity, global_dense, "
            "and matched_projector"
        )
    if any(type(value) is not bool for value in passed_controls.values()):
        raise TypeError("control gate decisions must be boolean")
    return all(passed_controls.values())


def select_strongest_control(
    summaries: Sequence[Stage2Summary],
) -> Stage2Summary:
    """Select macro Top-1, Top-5, lower cost, then config ID."""

    controls = collapse_stage2_aliases(summaries)
    return min(
        controls,
        key=lambda item: (
            -item.macro_top1,
            -item.macro_top5,
            item.inference_cost,
            item.config_id,
        ),
    )


def select_stage2_candidate(
    summaries: Sequence[Stage2Summary],
    control_gate_passes: Mapping[str, Mapping[str, bool]],
) -> Stage2Summary:
    """Select among fully gated candidates using the locked tie-break."""

    entries = tuple(summaries)
    if not entries or any(not isinstance(item, Stage2Summary) for item in entries):
        raise ValueError("summaries must contain Stage2Summary values")
    identifiers = {item.config_id for item in entries}
    allowed = {entry["config_id"] for entry in build_stage2_adapter_grid()}
    unknown = identifiers - allowed
    if unknown:
        raise ValueError(f"unknown Stage 2 adapter candidate: {sorted(unknown)}")
    if not isinstance(control_gate_passes, Mapping) or set(
        control_gate_passes
    ) != identifiers:
        raise ValueError("control_gate_passes must match candidate config IDs")
    eligible = [
        item
        for item in entries
        if adapter_gate_eligible(control_gate_passes[item.config_id])
    ]
    if not eligible:
        raise ValueError("no Stage 2 adapter candidate passed all controls")
    candidates = collapse_stage2_aliases(eligible)
    return min(
        candidates,
        key=lambda item: (
            -item.macro_top1,
            -item.macro_top5,
            item.added_parameters,
            item.config_id,
        ),
    )
