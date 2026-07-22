"""Shared canonical-ID perturbations for matching-fairness experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import itertools

import numpy as np

from .artifacts import ScoreArtifact


_STANDARD_ALGORITHM = "AIAA3800-STANDARD-SCENARIOS-v1"
_SELECTION_FIELDS = (
    "drop_query",
    "drop_gallery",
    "drop_pair",
    "duplicate_gallery",
)
_FORMAL_DUPLICATE_QUERY_COUNTS = frozenset({0, 10, 20})


@dataclass(frozen=True)
class ScenarioSpec:
    """One immutable point in the formal standard perturbation grid."""

    drop_query: int
    drop_gallery: int
    drop_pair: int
    duplicate_gallery: int

    def __post_init__(self) -> None:
        for field in _SELECTION_FIELDS:
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
        if self.drop_pair != 0:
            raise ValueError("the formal standard grid requires drop_pair=0")

    @property
    def slug(self) -> str:
        """Return a compact deterministic identifier for this scenario."""

        return (
            f"dropq{self.drop_query}_dropg{self.drop_gallery}_"
            f"dropp{self.drop_pair}_dupg{self.duplicate_gallery}"
        )


def standard_scenarios() -> tuple[ScenarioSpec, ...]:
    """Return the exact 27-point formal standard scenario grid."""

    return tuple(
        ScenarioSpec(*values)
        for values in itertools.product(
            (0, 5, 10),
            (0, 5, 10),
            (0,),
            (0, 10, 20),
        )
    )


def build_standard_manifest(
    canonical_image_ids: Sequence[str],
    seed: int = 42,
) -> dict[str, object]:
    """Build one model-independent set of canonical perturbation orders."""

    canonical_image_ids = _validated_ids(
        canonical_image_ids,
        "canonical image IDs",
    )
    seed = _validated_seed(seed)
    if len(canonical_image_ids) < 20:
        raise ValueError("at least 20 canonical image IDs are required")

    manifest: dict[str, object] = {
        "schema_version": 1,
        "algorithm_version": _STANDARD_ALGORITHM,
        "seed": seed,
        "canonical_image_ids": list(canonical_image_ids),
    }
    canonical_array = np.asarray(canonical_image_ids)
    for stream, field in enumerate(_SELECTION_FIELDS):
        rng = np.random.default_rng(np.random.SeedSequence([seed, stream]))
        manifest[field] = canonical_array[
            rng.permutation(len(canonical_image_ids))
        ].tolist()
    return manifest


def apply_standard_scenario(
    artifact: ScoreArtifact,
    manifest: Mapping[str, object],
    scenario: ScenarioSpec,
) -> ScoreArtifact:
    """Apply a shared perturbation by canonical identity, not local position."""

    canonical_ids, orders = _validated_standard_manifest(manifest)
    if not isinstance(scenario, ScenarioSpec):
        raise ValueError("scenario must be a ScenarioSpec")
    master_set = set(canonical_ids)
    if (
        len(artifact.target_canonical_ids) != len(canonical_ids)
        or set(artifact.target_canonical_ids) != master_set
        or len(set(artifact.target_canonical_ids)) != len(canonical_ids)
        or len(artifact.gallery_canonical_ids) != len(canonical_ids)
        or set(artifact.gallery_canonical_ids) != master_set
        or len(set(artifact.gallery_canonical_ids)) != len(canonical_ids)
    ):
        raise ValueError(
            "artifact query targets and gallery must match the canonical master"
        )
    artifact.validate()

    requested = {
        "drop_query": scenario.drop_query,
        "drop_gallery": scenario.drop_gallery,
        "drop_pair": scenario.drop_pair,
        "duplicate_gallery": scenario.duplicate_gallery,
    }
    for field, count in requested.items():
        if count > len(orders[field]):
            raise ValueError(f"scenario requests too many {field} IDs")

    selected_drop_query = tuple(orders["drop_query"][: scenario.drop_query])
    selected_drop_gallery = tuple(
        orders["drop_gallery"][: scenario.drop_gallery]
    )
    selected_drop_pair = tuple(orders["drop_pair"][: scenario.drop_pair])
    dropped_query_ids = set(selected_drop_query).union(selected_drop_pair)
    dropped_gallery_ids = set(selected_drop_gallery).union(selected_drop_pair)

    duplicate_candidates = (
        canonical_id
        for canonical_id in orders["duplicate_gallery"]
        if canonical_id not in dropped_gallery_ids
    )
    selected_duplicate_gallery = tuple(
        itertools.islice(duplicate_candidates, scenario.duplicate_gallery)
    )
    if len(selected_duplicate_gallery) != scenario.duplicate_gallery:
        raise ValueError("not enough post-drop gallery IDs to duplicate")

    selected = {
        "drop_query": selected_drop_query,
        "drop_gallery": selected_drop_gallery,
        "drop_pair": selected_drop_pair,
        "duplicate_gallery": selected_duplicate_gallery,
    }
    kept_rows = tuple(
        index
        for index, target in enumerate(artifact.target_canonical_ids)
        if target not in dropped_query_ids
    )
    kept_columns = tuple(
        index
        for index, canonical_id in enumerate(artifact.gallery_canonical_ids)
        if canonical_id not in dropped_gallery_ids
    )
    column_by_canonical_id = {
        canonical_id: index
        for index, canonical_id in enumerate(artifact.gallery_canonical_ids)
    }
    duplicate_columns = tuple(
        column_by_canonical_id[canonical_id]
        for canonical_id in selected_duplicate_gallery
    )

    similarity = artifact.similarity[np.ix_(kept_rows, kept_columns)]
    if duplicate_columns:
        duplicate_scores = artifact.similarity[np.ix_(kept_rows, duplicate_columns)]
        similarity = np.concatenate((similarity, duplicate_scores), axis=1)

    gallery_entry_ids = tuple(
        artifact.gallery_entry_ids[index] for index in kept_columns
    )
    duplicate_entry_ids = tuple(
        f"{artifact.gallery_entry_ids[source_column]}"
        f"__duplicate_entry_{duplicate_index:04d}"
        for duplicate_index, source_column in enumerate(duplicate_columns)
    )
    if set(gallery_entry_ids).intersection(duplicate_entry_ids):
        raise ValueError("generated duplicate gallery entry ID collides with base ID")
    gallery_canonical_ids = tuple(
        artifact.gallery_canonical_ids[index] for index in kept_columns
    ) + selected_duplicate_gallery
    target_canonical_ids = tuple(
        artifact.target_canonical_ids[index] for index in kept_rows
    )

    metadata = dict(artifact.metadata)
    metadata.update(
        {
            "scenario": scenario.slug,
            "selected_canonical_ids": selected,
        }
    )
    missing_targets = set(target_canonical_ids).difference(gallery_canonical_ids)
    if missing_targets:
        metadata["allow_unanswerable_targets"] = True
    else:
        metadata.pop("allow_unanswerable_targets", None)

    result = ScoreArtifact(
        similarity=similarity,
        query_ids=tuple(artifact.query_ids[index] for index in kept_rows),
        gallery_entry_ids=gallery_entry_ids + duplicate_entry_ids,
        gallery_canonical_ids=gallery_canonical_ids,
        target_canonical_ids=target_canonical_ids,
        metadata=metadata,
    )
    result.validate()
    return result


def build_duplicate_query_artifact(
    a: ScoreArtifact,
    b: ScoreArtifact,
    repeated_ids: Sequence[str],
    count: int,
) -> ScoreArtifact:
    """Append selected real EEG-B rows to the complete EEG-A query artifact."""

    if count not in _FORMAL_DUPLICATE_QUERY_COUNTS:
        raise ValueError("duplicate query count must be 0, 10, or 20")
    a.validate()
    b.validate()
    if a.similarity.shape != (200, 200) or b.similarity.shape != (200, 200):
        raise ValueError("A/B base artifacts must both be 200 x 200")
    if (
        a.gallery_entry_ids != b.gallery_entry_ids
        or a.gallery_canonical_ids != b.gallery_canonical_ids
    ):
        raise ValueError("A/B gallery IDs must match exactly")
    if (
        len(set(a.target_canonical_ids)) != len(a.target_canonical_ids)
        or len(set(b.target_canonical_ids)) != len(b.target_canonical_ids)
        or set(a.target_canonical_ids) != set(b.target_canonical_ids)
    ):
        raise ValueError("A/B source targets must be the same unique canonical IDs")

    repeated_ids = tuple(repeated_ids)
    selected = repeated_ids[:count]
    row_by_a_target = {
        target: row for row, target in enumerate(a.target_canonical_ids)
    }
    row_by_b_target = {
        target: row for row, target in enumerate(b.target_canonical_ids)
    }
    missing = tuple(target for target in selected if target not in row_by_b_target)
    if missing:
        raise ValueError(f"missing repeated target in EEG-B artifact: {missing}")
    if any(not isinstance(target, str) or not target for target in selected):
        raise ValueError("repeated target IDs must be non-empty strings")
    if len(set(selected)) != len(selected):
        raise ValueError("selected repeated target IDs must be unique")
    if len(selected) != count:
        raise ValueError("repeated IDs do not contain the requested count")

    b_rows = tuple(row_by_b_target[target] for target in selected)
    for target, b_row in zip(selected, b_rows):
        a_row = row_by_a_target[target]
        if (
            a.similarity[a_row].dtype == b.similarity[b_row].dtype
            and a.similarity[a_row].shape == b.similarity[b_row].shape
            and a.similarity[a_row].tobytes(order="C")
            == b.similarity[b_row].tobytes(order="C")
        ):
            raise ValueError(
                f"byte-identical A/B rows are not a real repeat: {target}"
            )

    if b_rows:
        similarity = np.concatenate((a.similarity, b.similarity[list(b_rows)]), axis=0)
    else:
        similarity = a.similarity.copy()
    query_ids = a.query_ids + tuple(f"{target}__eeg_b" for target in selected)
    targets = a.target_canonical_ids + selected
    metadata = dict(a.metadata)
    metadata["query_mode"] = f"dupq{count}"
    if set(targets).difference(a.gallery_canonical_ids):
        metadata["allow_unanswerable_targets"] = True
    else:
        metadata.pop("allow_unanswerable_targets", None)

    result = ScoreArtifact(
        similarity=similarity,
        query_ids=query_ids,
        gallery_entry_ids=a.gallery_entry_ids,
        gallery_canonical_ids=a.gallery_canonical_ids,
        target_canonical_ids=targets,
        metadata=metadata,
    )
    result.validate()
    return result


def _validated_standard_manifest(
    manifest: Mapping[str, object],
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    if not isinstance(manifest, Mapping):
        raise ValueError("standard manifest must be a mapping")
    canonical_ids = _validated_ids(
        manifest.get("canonical_image_ids", ()),
        "canonical image IDs",
    )
    if manifest.get("algorithm_version") != _STANDARD_ALGORITHM:
        raise ValueError("standard manifest algorithm version is unsupported")
    _validated_seed(manifest.get("seed"))
    canonical_set = set(canonical_ids)
    orders: dict[str, tuple[str, ...]] = {}
    for field in _SELECTION_FIELDS:
        order = _validated_ids(manifest.get(field, ()), f"{field} order")
        if len(order) != len(canonical_ids) or set(order) != canonical_set:
            raise ValueError(f"{field} order must permute the canonical master")
        orders[field] = order
    return canonical_ids, orders


def _validated_ids(values: object, label: str) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    result = tuple(values)
    if not result or any(not isinstance(value, str) or not value for value in result):
        raise ValueError(f"{label} must contain non-empty strings")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} must be unique")
    return result


def _validated_seed(seed: object) -> int:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    return int(seed)
