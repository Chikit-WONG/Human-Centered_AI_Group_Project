#!/usr/bin/env python3
"""Unit tests for the optional global one-to-one retrieval decoder."""

from __future__ import annotations

import itertools
import unittest

import numpy as np
import torch

from scripts.evaluate_retrieval import (
    solve_assignment_with_order_permutations,
    solve_one_to_one_assignment,
)


class HungarianAssignmentTests(unittest.TestCase):
    def test_identity_matrix(self) -> None:
        similarity = torch.eye(4, dtype=torch.float32)
        assigned = solve_one_to_one_assignment(similarity)
        self.assertTrue(torch.equal(assigned, torch.arange(4)))

    def test_collision_is_resolved_by_global_optimum(self) -> None:
        similarity = torch.tensor(
            [
                [10.0, 0.0, 0.0],
                [11.0, 10.0, 0.0],
                [0.0, 0.0, 8.0],
            ]
        )
        self.assertEqual(similarity.argmax(dim=1).tolist(), [0, 0, 2])
        assigned = solve_one_to_one_assignment(similarity)
        self.assertEqual(assigned.tolist(), [0, 1, 2])

    def test_objective_matches_brute_force(self) -> None:
        similarity = torch.tensor(
            [
                [0.2, 0.8, 0.1, 0.3],
                [0.9, 0.1, 0.4, 0.2],
                [0.3, 0.2, 0.5, 0.7],
                [0.1, 0.4, 0.8, 0.6],
            ],
            dtype=torch.float64,
        )
        assigned = solve_one_to_one_assignment(similarity).numpy()
        rows = np.arange(4)
        actual = float(similarity.numpy()[rows, assigned].sum())
        expected = max(
            float(similarity.numpy()[rows, permutation].sum())
            for permutation in itertools.permutations(range(4))
        )
        self.assertAlmostEqual(actual, expected, places=12)

    def test_exact_tie_still_returns_a_bijection(self) -> None:
        assigned = solve_one_to_one_assignment(torch.zeros((3, 3)))
        self.assertEqual(sorted(assigned.tolist()), [0, 1, 2])

    def test_unique_optimum_is_invariant_after_order_mapping(self) -> None:
        similarity = torch.tensor(
            [
                [0.9, 0.1, 0.2, 0.0],
                [0.3, 0.8, 0.1, 0.2],
                [0.1, 0.0, 0.7, 0.4],
                [0.2, 0.3, 0.1, 0.9],
            ]
        )
        expected = solve_one_to_one_assignment(similarity)
        for seed in range(10):
            rng = np.random.default_rng(seed)
            actual = solve_assignment_with_order_permutations(
                similarity,
                rng.permutation(4),
                rng.permutation(4),
            )
            self.assertTrue(torch.equal(actual, expected))

    def test_non_diagonal_id_mapping_is_scored_by_target_index(self) -> None:
        query_ids = ["image-a", "image-b", "image-c"]
        gallery_ids = ["image-b", "image-c", "image-a"]
        gallery_index = {
            image_id: index for index, image_id in enumerate(gallery_ids)
        }
        targets = torch.tensor(
            [gallery_index[image_id] for image_id in query_ids]
        )
        similarity = torch.tensor(
            [[0.0, 0.0, 3.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]]
        )
        assigned = solve_one_to_one_assignment(similarity)
        self.assertTrue(assigned.eq(targets).all())

    def test_invalid_inputs_are_rejected(self) -> None:
        for invalid in (
            torch.zeros(3),
            torch.zeros((2, 3)),
            torch.empty((0, 0)),
            torch.tensor([[0.0, float("nan")], [1.0, 0.0]]),
        ):
            with self.subTest(shape=tuple(invalid.shape)):
                with self.assertRaises(ValueError):
                    solve_one_to_one_assignment(invalid)

        with self.assertRaises(ValueError):
            solve_assignment_with_order_permutations(
                torch.eye(3),
                np.asarray([0, 0, 2]),
                np.asarray([0, 1, 2]),
            )


if __name__ == "__main__":
    unittest.main()
