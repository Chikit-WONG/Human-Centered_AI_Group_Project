#!/usr/bin/env python3
"""Unit tests for semantic validation of per-query retrieval predictions."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.aggregate_subject_metrics import validate_predictions


FIELDS = [
    "query_index",
    "subject_id",
    "gt_image_id",
    "gt_rank",
    "gt_cosine_similarity",
    "top1_image_id",
    "top1_cosine_similarity",
    "top5_image_ids",
    "top5_cosine_similarities",
    "correct_top1",
    "correct_top5",
]


def make_rows() -> list[dict[str, object]]:
    image_ids = [f"image-{index}" for index in range(5)]
    similarities = [0.9, 0.8, 0.7, 0.6, 0.5]
    rows: list[dict[str, object]] = []
    for query_index, gt_image_id in enumerate(image_ids):
        ranked_ids = [gt_image_id, *[item for item in image_ids if item != gt_image_id]]
        rows.append(
            {
                "query_index": query_index,
                "subject_id": 1,
                "gt_image_id": gt_image_id,
                "gt_rank": 1,
                "gt_cosine_similarity": similarities[0],
                "top1_image_id": gt_image_id,
                "top1_cosine_similarity": similarities[0],
                "top5_image_ids": json.dumps(ranked_ids),
                "top5_cosine_similarities": json.dumps(similarities),
                "correct_top1": 1,
                "correct_top5": 1,
            }
        )
    return rows


class SubjectMetricValidationTests(unittest.TestCase):
    def validate(self, rows: list[dict[str, object]]) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            validate_predictions(
                path,
                subject_id=1,
                sample_count=5,
                gallery_size=5,
                top1_count=5,
                top5_count=5,
            )

    def test_semantically_consistent_predictions_pass(self) -> None:
        self.validate(make_rows())

    def test_self_consistent_but_wrong_correctness_flag_is_rejected(self) -> None:
        rows = make_rows()
        rows[0]["correct_top1"] = 0
        with self.assertRaisesRegex(ValueError, "Top-1 fields disagree"):
            self.validate(rows)

    def test_duplicate_query_index_is_rejected(self) -> None:
        rows = make_rows()
        rows[1]["query_index"] = 0
        with self.assertRaisesRegex(ValueError, "query indices"):
            self.validate(rows)

    def test_top1_must_equal_first_top5_item(self) -> None:
        rows = make_rows()
        rows[0]["top1_image_id"] = "image-1"
        with self.assertRaisesRegex(ValueError, "first Top-5 item"):
            self.validate(rows)

    def test_non_finite_similarity_is_rejected(self) -> None:
        rows = make_rows()
        rows[0]["top5_cosine_similarities"] = json.dumps(
            [0.9, float("nan"), 0.7, 0.6, 0.5]
        )
        with self.assertRaisesRegex(ValueError, "non-finite Top-5 similarity"):
            self.validate(rows)


if __name__ == "__main__":
    unittest.main()
