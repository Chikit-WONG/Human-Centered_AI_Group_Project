from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_stage1_finalizer_imports_with_declared_pythonpath_only(
    experiment_root: Path,
) -> None:
    project_root = experiment_root.parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.fspath(experiment_root)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from samga_brain_rw.stage1_finalizer import finalize_stage1",
        ],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
