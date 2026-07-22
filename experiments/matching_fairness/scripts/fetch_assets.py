from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from matching_fairness.provenance import sha256_file  # noqa: E402


HF_DATASET_REPO = "LidongYang/EEG_Image_decode"
ASSET_ROOT = Path(
    "/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/EEG_Project/"
    "models/EEG_Image_decode_assets"
)
ASSET_RELATIVE_PATHS = (
    "Preprocessed_data_250Hz/sub-08/preprocessed_eeg_training.npy",
    "Preprocessed_data_250Hz/sub-08/preprocessed_eeg_test.npy",
    "ViT-H-14_features_train.pt",
    "ViT-H-14_features_test.pt",
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = (
    REPOSITORY_ROOT.parent
    / "test/brain-rw/results/matching_fairness_v3/manifests/assets_lock.json"
)


def inventory_assets(asset_root: Path) -> dict[str, dict[str, Any]]:
    asset_root = Path(asset_root)
    root_resolved = asset_root.resolve(strict=True)
    files: dict[str, dict[str, Any]] = {}
    for relative in ASSET_RELATIVE_PATHS:
        path = asset_root / relative
        try:
            resolved = path.resolve(strict=True)
        except FileNotFoundError as error:
            raise FileNotFoundError(f"required official asset is missing: {path}") from error
        try:
            resolved.relative_to(root_resolved)
        except ValueError as error:
            raise ValueError(
                f"asset symbolic link resolves outside asset root: {path} -> {resolved}"
            ) from error
        if not resolved.is_file():
            raise ValueError(f"required official asset is not a regular file: {path}")
        files[relative] = {
            "bytes": resolved.stat().st_size,
            "sha256": sha256_file(resolved),
        }
    return files


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def fetch_assets(
    *, asset_root: Path = ASSET_ROOT, manifest_path: Path = DEFAULT_MANIFEST
) -> dict[str, Any]:
    command = ["hf", "download", HF_DATASET_REPO, "--repo-type", "dataset"]
    for relative in ASSET_RELATIVE_PATHS:
        command.extend(["--include", relative])
    command.extend(["--local-dir", str(asset_root)])
    subprocess.run(command, check=True)
    result: dict[str, Any] = {
        "repo_id": HF_DATASET_REPO,
        "repo_type": "dataset",
        "asset_root": str(asset_root),
        "files": inventory_assets(asset_root),
    }
    _write_json(result, manifest_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch the four minimal official assets")
    parser.add_argument("--asset-root", type=Path, default=ASSET_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    arguments = parser.parse_args()
    result = fetch_assets(
        asset_root=arguments.asset_root,
        manifest_path=arguments.manifest,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
