from __future__ import annotations

from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


def test_slurm_scripts_resolve_from_submit_directory() -> None:
    scripts = sorted((EXPERIMENT_ROOT / "slurm").glob("*.slurm"))
    assert len(scripts) == 5
    for path in scripts:
        source = path.read_text(encoding="utf-8")
        assert "SLURM_SUBMIT_DIR" in source
        assert 'dirname "$0"' not in source
        assert "gpu:a40:1" in source


def test_conda_activation_tolerates_optional_hook_variables() -> None:
    source = (EXPERIMENT_ROOT / "scripts/common.sh").read_text(encoding="utf-8")
    before, after = source.split("conda activate test")
    assert "set +u" in before
    assert "set -u" in after
