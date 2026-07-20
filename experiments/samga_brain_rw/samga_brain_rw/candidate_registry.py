"""Canonical Stage 2 candidate identities derived from the sealed registry."""

from __future__ import annotations

from collections.abc import Mapping


FactorIdentity = tuple[
    str,
    str,
    str,
    str,
    int | None,
    float | None,
]
DEFAULT_FACTOR_IDENTITY: FactorIdentity = (
    "s2-layernorm-off",
    "s2-whitening-off",
    "s2-preproj-shared",
    "identity",
    None,
    None,
)


def _mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise ValueError(f"{context} must be a string-keyed mapping")
    return dict(value)


def stage2_registry_identities(
    payload: Mapping[str, object],
) -> dict[str, frozenset[FactorIdentity]]:
    """Return every exact runnable factor tuple for each registry config ID."""

    allowed: dict[str, set[FactorIdentity]] = {}

    def add(config_id: object, identity: FactorIdentity) -> None:
        if not isinstance(config_id, str) or not config_id:
            raise ValueError("Stage 2 registry config_id is invalid")
        allowed.setdefault(config_id, set()).add(identity)

    for key, active_value, active_index in (
        ("layernorm", "s2-layernorm-on", 0),
        ("whitening", "s2-whitening-on", 1),
        ("preprojectors", "s2-preproj-separate", 2),
    ):
        entries = payload.get(key)
        if not isinstance(entries, list):
            raise ValueError(f"Stage 2 registry {key} must be a list")
        for raw in entries:
            entry = _mapping(raw, f"Stage 2 registry {key} entry")
            config_id = entry.get("config_id")
            identity = list(DEFAULT_FACTOR_IDENTITY)
            if config_id == active_value:
                identity[active_index] = active_value
            add(config_id, tuple(identity))  # type: ignore[arg-type]

    checkpoint_entries = payload.get("checkpoint_averaging")
    if not isinstance(checkpoint_entries, list):
        raise ValueError("Stage 2 checkpoint_averaging must be a list")
    for raw in checkpoint_entries:
        entry = _mapping(raw, "Stage 2 checkpoint entry")
        add(entry.get("config_id"), DEFAULT_FACTOR_IDENTITY)

    adapter = _mapping(
        payload.get("feature_adapter"),
        "Stage 2 feature_adapter",
    )
    candidates = adapter.get("candidates")
    controls = adapter.get("controls")
    if not isinstance(candidates, list) or not isinstance(controls, list):
        raise ValueError("Stage 2 adapter registry lists are invalid")
    control_kinds: dict[str, str] = {}
    for raw in controls:
        control = _mapping(raw, "Stage 2 adapter control")
        config_id = control.get("config_id")
        kind = control.get("kind")
        if (
            not isinstance(config_id, str)
            or kind
            not in {"identity", "global_dense", "matched_projector"}
        ):
            raise ValueError("Stage 2 adapter control is invalid")
        control_kinds[config_id] = str(kind)
        if kind == "identity":
            add(config_id, DEFAULT_FACTOR_IDENTITY)

    for raw in candidates:
        candidate = _mapping(raw, "Stage 2 adapter candidate")
        rank = candidate.get("rank")
        ratio = candidate.get("learning_rate_ratio")
        if type(rank) is not int or type(ratio) is not float:
            raise ValueError(
                "Stage 2 adapter candidate rank/LR types are invalid"
            )
        add(
            candidate.get("config_id"),
            (
                *DEFAULT_FACTOR_IDENTITY[:3],
                "adapter",
                rank,
                ratio,
            ),
        )
        bindings = _mapping(
            candidate.get("control_bindings"),
            "Stage 2 adapter control bindings",
        )
        for raw_binding in bindings.values():
            binding = _mapping(
                raw_binding,
                "Stage 2 adapter control binding",
            )
            control_id = binding.get("config_id")
            if not isinstance(control_id, str):
                raise ValueError(
                    "Stage 2 adapter binding config_id is invalid"
                )
            kind = control_kinds.get(control_id)
            if kind is None:
                raise ValueError(
                    "Stage 2 adapter binding references an unknown control"
                )
            if kind == "identity":
                add(control_id, DEFAULT_FACTOR_IDENTITY)
                continue
            binding_rank = binding.get("rank")
            binding_ratio = binding.get("learning_rate_ratio")
            if (
                type(binding_rank) is not int
                or type(binding_ratio) is not float
            ):
                raise ValueError(
                    "Stage 2 adapter control rank/LR types are invalid"
                )
            add(
                control_id,
                (
                    *DEFAULT_FACTOR_IDENTITY[:3],
                    kind,
                    binding_rank,
                    binding_ratio,
                ),
            )
    return {
        config_id: frozenset(identities)
        for config_id, identities in allowed.items()
    }


def select_stage2_factor_identity(
    allowed: frozenset[FactorIdentity],
    provided: FactorIdentity,
) -> FactorIdentity:
    """Select one registry tuple while rejecting every conflicting override."""

    if not allowed:
        raise ValueError("candidate_id is absent from the Stage 2 registry")
    if len(allowed) == 1:
        expected = next(iter(allowed))
        for actual, baseline, target in zip(
            provided,
            DEFAULT_FACTOR_IDENTITY,
            expected,
            strict=True,
        ):
            if actual != baseline and actual != target:
                raise ValueError(
                    "Stage 2 CLI factor conflicts with the registry identity"
                )
        return expected

    kinds = {identity[3] for identity in allowed}
    if len(kinds) != 1 or "identity" in kinds:
        raise ValueError("Stage 2 registry identity is ambiguous")
    kind = next(iter(kinds))
    if provided[:3] != DEFAULT_FACTOR_IDENTITY[:3]:
        raise ValueError(
            "Stage 2 CLI preprocessing conflicts with adapter control"
        )
    if provided[3] not in {"identity", kind}:
        raise ValueError(
            "Stage 2 CLI adapter kind conflicts with the registry identity"
        )
    proposed: FactorIdentity = (
        *DEFAULT_FACTOR_IDENTITY[:3],
        kind,
        provided[4],
        provided[5],
    )
    if proposed not in allowed:
        raise ValueError(
            "Stage 2 adapter control rank/LR pair is absent from the registry"
        )
    return proposed
