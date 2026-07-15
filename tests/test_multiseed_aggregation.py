#!/usr/bin/env python3
"""Unit tests for cross-seed retrieval aggregation helpers."""

from __future__ import annotations

import argparse
import math
import unittest

from scripts.aggregate_multiseed_metrics import (
    build_cross_seed_metric_fields,
    build_seed_rows,
    build_subject_rows,
    default_output_name,
    parse_seeds,
    render_subject_table,
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
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_seeds("42,-1")

    def test_default_output_name_is_compact_only_for_contiguous_order(self) -> None:
        self.assertEqual(default_output_name([42, 43, 44, 45, 46]), "seeds42-46")
        self.assertEqual(default_output_name([42, 44, 46]), "seeds42_44_46")
        self.assertEqual(default_output_name([44, 43, 42]), "seeds44_43_42")
        self.assertEqual(default_output_name([42]), "seeds42")

    def test_cross_seed_fields_are_generic_with_five_seed_aliases_only(self) -> None:
        arguments = {
            "mean_top1": 87.0,
            "sd_top1": 1.0,
            "mean_top5": 98.0,
            "sd_top5": 0.2,
            "between_subject_sd_top1": 3.0,
            "between_subject_sd_top5": 1.5,
        }
        two_seed = build_cross_seed_metric_fields(seed_count=2, **arguments)
        self.assertEqual(two_seed["cross_seed_mean_top1_percent"], 87.0)
        self.assertNotIn("five_seed_mean_top1_percent", two_seed)

        five_seed = build_cross_seed_metric_fields(seed_count=5, **arguments)
        self.assertEqual(
            five_seed["five_seed_mean_top1_percent"],
            five_seed["cross_seed_mean_top1_percent"],
        )
        self.assertEqual(
            five_seed[
                "between_subject_sample_sd_of_five_seed_means_top5_points"
            ],
            five_seed[
                "between_subject_sample_sd_of_cross_seed_means_top5_points"
            ],
        )

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

    def test_subject_table_uses_requested_seed_count(self) -> None:
        subjects = build_subject_rows(self.rows, [1, 2])
        english = "\n".join(
            render_subject_table(subjects, chinese=False, seed_count=2)
        )
        chinese = "\n".join(
            render_subject_table(subjects, chinese=True, seed_count=2)
        )
        self.assertIn("2-seed mean", english)
        self.assertIn("2-seed 均值", chinese)
        self.assertNotIn("five-seed", english)


if __name__ == "__main__":
    unittest.main()
