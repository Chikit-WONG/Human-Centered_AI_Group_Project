import dataclasses

import numpy as np
import pytest

from matching_fairness.artifacts import ScoreArtifact
from matching_fairness.scenarios import (
    ScenarioSpec,
    apply_standard_scenario,
    build_duplicate_query_artifact,
    build_standard_manifest,
    standard_scenarios,
)
from matching_fairness.trial_splits import select_duplicate_image_ids


def _canonical_ids() -> tuple[str, ...]:
    return tuple(f"image-{index:03d}" for index in range(200))


def _artifact(
    *,
    scale: float = 1.0,
    gallery_order: tuple[str, ...] | None = None,
    metadata: dict[str, object] | None = None,
) -> ScoreArtifact:
    canonical_ids = _canonical_ids()
    gallery_order = canonical_ids if gallery_order is None else gallery_order
    source_column = {canonical_id: index for index, canonical_id in enumerate(canonical_ids)}
    similarity = np.tile(
        np.array([source_column[value] for value in gallery_order], dtype=np.float64),
        (200, 1),
    )
    similarity += np.arange(200, dtype=np.float64)[:, None] / 1_000.0
    return ScoreArtifact(
        similarity=scale * similarity,
        query_ids=tuple(f"q-{index:03d}" for index in range(200)),
        gallery_entry_ids=tuple(f"entry-{value}" for value in gallery_order),
        gallery_canonical_ids=gallery_order,
        target_canonical_ids=canonical_ids,
        metadata={"model_slug": "fixture"} if metadata is None else metadata,
    )


def test_standard_grid_has_exactly_27_unique_scenarios() -> None:
    scenarios = standard_scenarios()

    assert len(scenarios) == len(set(scenarios)) == 27
    assert {s.drop_query for s in scenarios} == {0, 5, 10}
    assert {s.drop_gallery for s in scenarios} == {0, 5, 10}
    assert {s.drop_pair for s in scenarios} == {0}
    assert {s.duplicate_gallery for s in scenarios} == {0, 10, 20}


def test_scenario_spec_is_immutable_and_hashable() -> None:
    scenario = standard_scenarios()[0]

    assert hash(scenario) == hash(scenario)
    with pytest.raises(dataclasses.FrozenInstanceError):
        scenario.drop_query = 10  # type: ignore[misc]


def test_standard_manifest_uses_independent_seed_sequence_streams() -> None:
    canonical_ids = _canonical_ids()

    manifest = build_standard_manifest(canonical_ids, seed=42)

    for stream, field in enumerate(
        ("drop_query", "drop_gallery", "drop_pair", "duplicate_gallery")
    ):
        rng = np.random.default_rng(np.random.SeedSequence([42, stream]))
        expected = tuple(np.asarray(canonical_ids)[rng.permutation(200)].tolist())
        assert tuple(manifest[field]) == expected


def test_all_models_apply_the_same_canonical_manifest() -> None:
    canonical_ids = _canonical_ids()
    manifest = build_standard_manifest(canonical_ids, seed=42)
    scenarios = standard_scenarios()
    common = dict(
        query_ids=tuple(f"q-{index:03d}" for index in range(200)),
        gallery_entry_ids=canonical_ids,
        gallery_canonical_ids=canonical_ids,
        target_canonical_ids=canonical_ids,
    )
    first = apply_standard_scenario(
        ScoreArtifact(similarity=np.eye(200), metadata={"model_slug": "a"}, **common),
        manifest,
        scenarios[7],
    )
    second = apply_standard_scenario(
        ScoreArtifact(similarity=2 * np.eye(200), metadata={"model_slug": "b"}, **common),
        manifest,
        scenarios[7],
    )

    assert first.metadata["selected_canonical_ids"] == second.metadata[
        "selected_canonical_ids"
    ]


def test_standard_scenario_is_applied_by_canonical_id_in_local_orders() -> None:
    canonical_ids = _canonical_ids()
    manifest = build_standard_manifest(canonical_ids)
    scenario = ScenarioSpec(5, 5, 0, 10)
    artifact = _artifact(gallery_order=tuple(reversed(canonical_ids)))

    result = apply_standard_scenario(artifact, manifest, scenario)

    selected = result.metadata["selected_canonical_ids"]
    assert not set(selected["drop_query"]).intersection(result.target_canonical_ids)
    assert not set(selected["drop_gallery"]).intersection(
        result.gallery_canonical_ids
    )
    assert result.similarity.shape == (195, 205)
    assert len(set(result.gallery_entry_ids)) == 205
    assert result.gallery_canonical_ids[-10:] == selected["duplicate_gallery"]
    for offset, canonical_id in enumerate(selected["duplicate_gallery"]):
        original_column = artifact.gallery_canonical_ids.index(canonical_id)
        np.testing.assert_array_equal(
            result.similarity[:, -10 + offset],
            artifact.similarity[
                [
                    index
                    for index, target in enumerate(artifact.target_canonical_ids)
                    if target not in set(selected["drop_query"])
                ],
                original_column,
            ],
        )
        assert result.gallery_entry_ids[-10 + offset].endswith(
            f"__duplicate_entry_{offset:04d}"
        )


def test_duplicate_candidates_skip_dropped_gallery_ids_without_reintroduction() -> None:
    manifest = build_standard_manifest(_canonical_ids())
    dropped = set(manifest["drop_gallery"][:10])
    initial_duplicate_candidates = set(manifest["duplicate_gallery"][:20])
    assert dropped.intersection(initial_duplicate_candidates)

    result = apply_standard_scenario(
        _artifact(),
        manifest,
        ScenarioSpec(0, 10, 0, 20),
    )

    selected = result.metadata["selected_canonical_ids"]
    assert len(selected["duplicate_gallery"]) == 20
    assert not dropped.intersection(selected["duplicate_gallery"])
    assert not dropped.intersection(result.gallery_canonical_ids)
    assert result.similarity.shape == (200, 210)
    assert result.metadata["allow_unanswerable_targets"] is True


def test_allow_unanswerable_flag_is_present_only_for_actually_missing_targets() -> None:
    manifest = build_standard_manifest(_canonical_ids())

    complete = apply_standard_scenario(
        _artifact(metadata={"model_slug": "fixture", "allow_unanswerable_targets": True}),
        manifest,
        ScenarioSpec(5, 0, 0, 0),
    )
    incomplete = apply_standard_scenario(
        _artifact(),
        manifest,
        ScenarioSpec(0, 5, 0, 0),
    )

    assert "allow_unanswerable_targets" not in complete.metadata
    assert incomplete.metadata["allow_unanswerable_targets"] is True


def test_standard_application_rejects_a_different_canonical_master_set() -> None:
    manifest = build_standard_manifest(_canonical_ids())
    artifact = _artifact()
    mismatched = ScoreArtifact(
        similarity=artifact.similarity,
        query_ids=artifact.query_ids,
        gallery_entry_ids=artifact.gallery_entry_ids,
        gallery_canonical_ids=artifact.gallery_canonical_ids,
        target_canonical_ids=artifact.target_canonical_ids[:-1] + ("other",),
        metadata=artifact.metadata,
    )

    with pytest.raises(ValueError, match="canonical master"):
        apply_standard_scenario(mismatched, manifest, standard_scenarios()[0])


def test_duplicate_query_shapes_are_exact_and_dupq10_is_nested_in_dupq20() -> None:
    a = _artifact()
    b = _artifact(scale=2.0)
    repeated_ids = select_duplicate_image_ids(_canonical_ids())

    dupq0 = build_duplicate_query_artifact(a, b, repeated_ids, 0)
    dupq10 = build_duplicate_query_artifact(a, b, repeated_ids, 10)
    dupq20 = build_duplicate_query_artifact(a, b, repeated_ids, 20)

    assert dupq0.similarity.shape == (200, 200)
    assert dupq10.similarity.shape == (210, 200)
    assert dupq20.similarity.shape == (220, 200)
    assert dupq20.gallery_entry_ids == a.gallery_entry_ids
    assert dupq20.gallery_canonical_ids == a.gallery_canonical_ids
    assert dupq10.target_canonical_ids[200:] == repeated_ids[:10]
    assert dupq20.target_canonical_ids[200:210] == repeated_ids[:10]
    assert dupq20.query_ids[-20:] == tuple(
        f"{target}__eeg_b" for target in repeated_ids
    )
    np.testing.assert_array_equal(dupq20.similarity[200:], b.similarity[
        [b.target_canonical_ids.index(target) for target in repeated_ids]
    ])


@pytest.mark.parametrize("count", (-1, 5, 21))
def test_duplicate_query_count_is_limited_to_formal_modes(count: int) -> None:
    with pytest.raises(ValueError, match="0, 10, or 20"):
        build_duplicate_query_artifact(
            _artifact(),
            _artifact(scale=2.0),
            select_duplicate_image_ids(_canonical_ids()),
            count,
        )


def test_duplicate_query_rejects_gallery_id_mismatch() -> None:
    a = _artifact()
    b = _artifact(scale=2.0)
    mismatched_b = ScoreArtifact(
        similarity=b.similarity,
        query_ids=b.query_ids,
        gallery_entry_ids=b.gallery_entry_ids[:-1] + ("other-entry",),
        gallery_canonical_ids=b.gallery_canonical_ids,
        target_canonical_ids=b.target_canonical_ids,
        metadata=b.metadata,
    )

    with pytest.raises(ValueError, match="gallery IDs must match"):
        build_duplicate_query_artifact(
            a,
            mismatched_b,
            select_duplicate_image_ids(_canonical_ids()),
            10,
        )


def test_duplicate_query_rejects_non_200_by_200_base_artifacts() -> None:
    ids = tuple(f"image-{index:03d}" for index in range(199))
    artifact = ScoreArtifact(
        similarity=np.eye(199),
        query_ids=tuple(f"q-{index:03d}" for index in range(199)),
        gallery_entry_ids=ids,
        gallery_canonical_ids=ids,
        target_canonical_ids=ids,
        metadata={"model_slug": "fixture"},
    )

    with pytest.raises(ValueError, match="200 x 200"):
        build_duplicate_query_artifact(artifact, artifact, (), 0)


def test_duplicate_query_rejects_missing_repeated_target() -> None:
    with pytest.raises(ValueError, match="missing repeated target"):
        build_duplicate_query_artifact(
            _artifact(),
            _artifact(scale=2.0),
            ("not-a-target",) * 20,
            10,
        )


def test_duplicate_query_rejects_copied_a_rows_as_real_repeats() -> None:
    a = _artifact()

    with pytest.raises(ValueError, match="byte-identical"):
        build_duplicate_query_artifact(
            a,
            a,
            select_duplicate_image_ids(_canonical_ids()),
            10,
        )
