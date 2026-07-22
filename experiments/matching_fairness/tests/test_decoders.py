from __future__ import annotations

import inspect
import math
import re

import numpy as np

from matching_fairness.decoders import (
    decode_greedy,
    decode_hungarian,
    decode_independent,
    decode_sinkhorn,
    decode_stable,
)


def test_independent_uses_stable_rowwise_argmax() -> None:
    similarity = np.asarray([[0.5, 0.5, 0.4], [0.1, 0.8, 0.8]])

    assignment = decode_independent(similarity)

    assert assignment.gallery_indices.tolist() == [0, 1]
    assert assignment.unmatched_mask.tolist() == [False, False]
    assert assignment.strict_one_to_one is False


def test_greedy_is_deterministic_and_leaves_excess_queries_unmatched() -> None:
    similarity = np.asarray(
        [
            [0.9, 0.8],
            [0.9, 0.7],
            [0.6, 0.5],
        ]
    )

    first = decode_greedy(similarity)
    second = decode_greedy(similarity.copy())

    assert first.gallery_indices.tolist() == [0, 1, -1]
    assert np.array_equal(first.gallery_indices, second.gallery_indices)
    assert first.unmatched_mask.tolist() == [False, False, True]
    assert first.strict_one_to_one is True


def test_hungarian_seeded_rectangular_assignment_reports_unmatched() -> None:
    similarity = np.asarray(
        [
            [0.9, 0.1],
            [0.8, 0.7],
            [0.2, 0.6],
        ]
    )

    first = decode_hungarian(similarity, seed=42)
    second = decode_hungarian(similarity.copy(), seed=42)

    assert np.array_equal(first.gallery_indices, second.gallery_indices)
    assert first.unmatched_mask.sum() == 1
    assert len(set(first.gallery_indices[first.gallery_indices >= 0])) == 2
    assert first.metadata["seed"] == 42
    assert first.metadata["matched_count"] == 2
    assert first.metadata["unmatched_count"] == 1
    assert math.isfinite(first.metadata["assigned_sum_similarity"])


def test_hungarian_binds_deterministic_row_and_column_permutations() -> None:
    tied = np.ones((4, 3), dtype=np.float64)

    seed_42 = decode_hungarian(tied, seed=42)
    repeated = decode_hungarian(tied.copy(), seed=42)
    seed_7 = decode_hungarian(tied, seed=7)
    other_shape = decode_hungarian(np.ones((5, 3)), seed=42)

    for field in ("row_permutation_sha256", "column_permutation_sha256"):
        value = seed_42.metadata[field]
        assert isinstance(value, str)
        assert re.fullmatch(r"[0-9a-f]{64}", value)
        assert repeated.metadata[field] == value
    assert seed_7.metadata["row_permutation_sha256"] != seed_42.metadata[
        "row_permutation_sha256"
    ]
    assert other_shape.metadata["row_permutation_sha256"] != seed_42.metadata[
        "row_permutation_sha256"
    ]


def test_query_proposing_stable_matching_has_no_blocking_pair() -> None:
    similarity = np.asarray(
        [
            [0.9, 0.8, 0.1],
            [0.9, 0.7, 0.6],
            [0.3, 0.8, 0.7],
            [0.2, 0.1, 0.9],
        ]
    )

    assignment = decode_stable(similarity)

    matched = assignment.gallery_indices
    assert assignment.strict_one_to_one is True
    assert assignment.unmatched_mask.sum() == 1
    assert len(set(matched[matched >= 0])) == 3
    gallery_partner = {
        int(gallery): query
        for query, gallery in enumerate(matched)
        if gallery >= 0
    }
    for query in range(similarity.shape[0]):
        assigned_gallery = int(matched[query])
        for gallery in range(similarity.shape[1]):
            query_prefers = assigned_gallery < 0 or (
                similarity[query, gallery] > similarity[query, assigned_gallery]
            )
            partner = gallery_partner.get(gallery)
            gallery_prefers = partner is None or (
                similarity[query, gallery] > similarity[partner, gallery]
                or (
                    similarity[query, gallery] == similarity[partner, gallery]
                    and query < partner
                )
            )
            assert not (query_prefers and gallery_prefers)


def test_sinkhorn_is_finite_deterministic_and_reports_convergence() -> None:
    similarity = np.asarray(
        [
            [1.0, 0.2, -0.3],
            [0.1, 0.9, 0.0],
        ]
    )
    kwargs = {"temperature": 0.05, "max_iterations": 500, "tolerance": 1e-8}

    first = decode_sinkhorn(similarity, **kwargs)
    second = decode_sinkhorn(similarity.copy(), **kwargs)

    assert np.array_equal(first.gallery_indices, second.gallery_indices)
    assert first.metadata == second.metadata
    assert first.strict_one_to_one is False
    assert first.metadata["converged"] is True
    assert 1 <= first.metadata["iterations"] <= 500
    assert math.isfinite(first.metadata["marginal_error"])
    assert first.metadata["marginal_error"] <= 1e-8
    assert math.isfinite(first.metadata["plan_min"])
    assert math.isfinite(first.metadata["plan_max"])
    assert math.isfinite(first.metadata["plan_sum"])
    assert isinstance(first.metadata["plan_sha256"], str)
    assert len(first.metadata["plan_sha256"]) == 64


def test_decoder_interfaces_do_not_accept_targets_or_ground_truth() -> None:
    for decoder in (
        decode_independent,
        decode_greedy,
        decode_hungarian,
        decode_stable,
        decode_sinkhorn,
    ):
        names = set(inspect.signature(decoder).parameters)
        assert "target" not in " ".join(names).lower()
        assert "label" not in " ".join(names).lower()


def test_tie_breaking_does_not_depend_on_ground_truth_order() -> None:
    similarity = np.ones((3, 3), dtype=np.float64)

    independent = decode_independent(similarity)
    greedy = decode_greedy(similarity)
    stable = decode_stable(similarity)

    assert independent.gallery_indices.tolist() == [0, 0, 0]
    assert greedy.gallery_indices.tolist() == [0, 1, 2]
    assert stable.gallery_indices.tolist() == [0, 1, 2]


def test_decoders_reject_nonfinite_or_empty_matrices() -> None:
    for similarity in (
        np.empty((0, 2)),
        np.asarray([[np.nan]]),
        np.asarray([1.0, 2.0]),
    ):
        try:
            decode_independent(similarity)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid matrix was accepted")
