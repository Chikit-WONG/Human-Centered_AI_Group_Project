#!/usr/bin/env python3
"""Unit tests for cross-seed retrieval aggregation helpers."""

from __future__ import annotations

import argparse
import math
import unittest

from scripts.aggregate_multiseed_metrics import (
    build_seed_rows,
    build_subject_rows,
    parse_seeds,
    sample_stdev,
)


def make_row(seed: int, subject_id: int, top1_count: int, top5_count: int) -> dict:
    return {
        "seed": seed,
        "subject_id": subject_id,
        "subject": f"sub-{subject_id:02d}",
        "sample_count": 200,
        "top1_count": top1_count,
        "top5_count": top5_count,
        "top1_percent": top1_count / 2,
        "top5_percent": top5_count / 2,
        "repeat_verified": True,
    }


class MultiSeedAggregationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            make_row(42, 1, 160, 190),
            make_row(42, 2, 180, 198),
            make_row(43, 1, 164, 192),
            make_row(43, 2, 176, 196),
        ]

    def test_seed_parser_preserves_declared_order(self) -> None:
        self.assertEqual(parse_seeds("42, 7,99"), [42, 7, 99])
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_seeds("42,42")

    def test_sample_sd_uses_ddof_one(self) -> None:
        self.assertAlmostEqual(sample_stdev([1.0, 3.0]), math.sqrt(2.0))
        self.assertEqual(sample_stdev([5.0]), 0.0)

    def test_seed_level_mean_is_computed_after_subject_aggregation(self) -> None:
        seeds = build_seed_rows(self.rows, [42, 43])
        self.assertEqual([row["top1_percent"] for row in seeds], [85.0, 85.0])
        self.assertEqual([row["top5_percent"] for row in seeds], [97.0, 97.0])
        self.assertTrue(all(row["repeat_verified_for_all"] for row in seeds))

    def test_subject_rows_report_seed_variability(self) -> None:
        subjects = build_subject_rows(self.rows, [1, 2])
        self.assertEqual(subjects[0]["mean_top1_percent"], 81.0)
        self.assertAlmostEqual(
            subjects[0]["sample_sd_top1_points"], math.sqrt(2.0)
        )
        self.assertEqual(subjects[1]["mean_top1_percent"], 89.0)


if __name__ == "__main__":
    unittest.main()
