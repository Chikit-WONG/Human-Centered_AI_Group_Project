#!/usr/bin/env python3
"""Tests for multiseed SLURM-array argument/range validation."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "submit_multiseed_array.slurm"


class SubmitMultiSeedArrayTests(unittest.TestCase):
    def test_default_array_maps_exactly_to_seed46_subject10_at_index39(self) -> None:
        environment = os.environ.copy()
        environment.update(
            {
                "SLURM_ARRAY_TASK_ID": "39",
                "SLURM_ARRAY_TASK_MIN": "0",
                "SLURM_ARRAY_TASK_MAX": "39",
                "SLURM_ARRAY_TASK_STEP": "1",
                "SLURM_ARRAY_TASK_COUNT": "40",
            }
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            fake_srun = Path(temporary_directory) / "srun"
            fake_srun.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            fake_srun.chmod(0o755)
            environment["PATH"] = f"{temporary_directory}:{environment['PATH']}"
            completed = subprocess.run(
                ["bash", str(SCRIPT)],
                cwd=PROJECT_ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("subject_id=10", completed.stdout)
        self.assertIn("seed=46", completed.stdout)

    def test_custom_seed_count_requires_matching_array_override(self) -> None:
        environment = os.environ.copy()
        environment.update(
            {
                "SLURM_ARRAY_TASK_ID": "0",
                "SLURM_ARRAY_TASK_MIN": "0",
                "SLURM_ARRAY_TASK_MAX": "39",
                "SLURM_ARRAY_TASK_STEP": "1",
                "SLURM_ARRAY_TASK_COUNT": "40",
            }
        )
        completed = subprocess.run(
            ["bash", str(SCRIPT), "formal", "51", "52"],
            cwd=PROJECT_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("2 seed(s) require exactly 20 array tasks", completed.stderr)
        self.assertIn("sbatch --array=0-19%2", completed.stderr)

    def test_invalid_custom_seed_is_rejected_before_task_mapping(self) -> None:
        environment = os.environ.copy()
        environment["SLURM_ARRAY_TASK_ID"] = "0"
        completed = subprocess.run(
            ["bash", str(SCRIPT), "formal", "51", "bad"],
            cwd=PROJECT_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("non-negative integer: bad", completed.stderr)


if __name__ == "__main__":
    unittest.main()
