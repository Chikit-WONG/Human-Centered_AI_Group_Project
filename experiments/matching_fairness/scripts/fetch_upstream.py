from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from matching_fairness.provenance import (  # noqa: E402
    OFFICIAL_SOURCE_BRANCH,
    OFFICIAL_SOURCE_URL,
    SourceLock,
    inspect_checkout,
)


UPSTREAM_ROOT = Path(
    "/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/EEG_Project/"
    "reference_code/codes_for_papers/EEG_Image_decode"
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = (
    REPOSITORY_ROOT.parent
    / "test/brain-rw/results/matching_fairness_v3/manifests/upstream_lock.json"
)


def resolve_detached_checkout(path: Path, url: str, branch: str) -> SourceLock:
    if not (path / ".git").exists():
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--branch", branch, url, str(path)],
            check=True,
        )
    subprocess.run(["git", "-C", str(path), "fetch", "origin", branch], check=True)
    commit = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "FETCH_HEAD"], text=True
    ).strip()
    subprocess.run(
        ["git", "-C", str(path), "checkout", "--detach", commit], check=True
    )
    actual = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != commit:
        raise RuntimeError(f"checkout mismatch: expected {commit}, found {actual}")
    return inspect_checkout(
        path,
        expected_url=url,
        expected_branch=branch,
    )


def write_source_lock(lock: SourceLock, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(lock.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and lock the official ATM source")
    parser.add_argument("--path", type=Path, default=UPSTREAM_ROOT)
    parser.add_argument("--url", default=OFFICIAL_SOURCE_URL)
    parser.add_argument("--branch", default=OFFICIAL_SOURCE_BRANCH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    arguments = parser.parse_args()
    lock = resolve_detached_checkout(arguments.path, arguments.url, arguments.branch)
    write_source_lock(lock, arguments.manifest)
    print(json.dumps(lock.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
