from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def experiment_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def configs_dir(experiment_root: Path) -> Path:
    return experiment_root / "configs"
