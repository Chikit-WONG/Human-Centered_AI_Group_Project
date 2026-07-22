from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from matching_fairness.config import Protocol  # noqa: E402
from matching_fairness.provenance import inspect_checkout, sha256_file  # noqa: E402
from scripts.fetch_assets import (  # noqa: E402
    ASSET_RELATIVE_PATHS,
    ASSET_ROOT,
    inventory_assets,
)
from scripts.fetch_upstream import UPSTREAM_ROOT  # noqa: E402


EXPECTED_PACKAGE_VERSIONS = {
    "torch": "2.5.0",
    "torchvision": "0.20.0",
    "torchaudio": "2.5.0",
    "numpy": "1.26.4",
    "pandas": "2.3.3",
    "scipy": "1.15.3",
    "scikit-learn": "1.6.1",
    "mne": "1.9.0",
    "einops": "0.8.1",
    "braindecode": "0.8.1",
    "wandb": "0.19.10",
    "open-clip-torch": "2.26.1",
    "pytorch-cuda": "12.4",
    "clip": "a9b1bf5920416aaeaec965c25dd9e8f98c864f16",
}
IMPORT_SPECS = {
    "torch": ("torch", "torch"),
    "torchvision": ("torchvision", "torchvision"),
    "torchaudio": ("torchaudio", "torchaudio"),
    "numpy": ("numpy", "numpy"),
    "pandas": ("pandas", "pandas"),
    "scipy": ("scipy", "scipy"),
    "scikit-learn": ("sklearn", "scikit-learn"),
    "mne": ("mne", "mne"),
    "einops": ("einops", "einops"),
    "braindecode": ("braindecode", "braindecode"),
    "wandb": ("wandb", "wandb"),
    "open-clip-torch": ("open_clip", "open-clip-torch"),
}
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(
    "/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/EEG_Project/"
    "EEG_Recon-RL/datasets/things_eeg_data"
)
DEFAULT_PROTOCOL = EXPERIMENT_ROOT / "configs/protocol_sub08_seed42.json"
DEFAULT_BRAINRW_TEST = DATA_ROOT / "Preprocessed_data_250Hz_whiten/sub-08/test.pt"
DEFAULT_TEST_IMAGES = DATA_ROOT / "test_images"
DEFAULT_MANIFEST = (
    REPOSITORY_ROOT.parent
    / "test/brain-rw/results/matching_fairness_v3/manifests/preflight.json"
)


@dataclass(frozen=True)
class RuntimeInfo:
    environment_name: str
    python_version: tuple[int, int, int]
    package_versions: Mapping[str, str]


@dataclass(frozen=True)
class PreflightExpectations:
    eeg_tail: tuple[int, int] = (63, 250)
    test_shape: tuple[int, int, int, int] = (200, 80, 63, 250)
    train_feature_rows: int = 16_540
    test_feature_rows: int = 200
    session_counts: Mapping[int, int] = field(
        default_factory=lambda: {0: 20, 1: 20, 2: 20, 3: 20}
    )


def _installed_version(module: Any, distribution: str) -> str:
    value = getattr(module, "__version__", None)
    if value is not None:
        return str(value)
    return importlib.metadata.version(distribution)


def _installed_clip_commit() -> str:
    importlib.import_module("clip")
    distribution = importlib.metadata.distribution("clip")
    direct_url = distribution.read_text("direct_url.json")
    if direct_url is None:
        raise RuntimeError("OpenAI CLIP installation has no VCS direct_url metadata")
    payload = json.loads(direct_url)
    commit = payload.get("vcs_info", {}).get("commit_id")
    if not commit:
        raise RuntimeError("OpenAI CLIP installation has no recorded VCS commit")
    return str(commit)


def collect_runtime_info() -> RuntimeInfo:
    versions: dict[str, str] = {}
    imported: dict[str, Any] = {}
    for logical_name, (module_name, distribution) in IMPORT_SPECS.items():
        try:
            module = importlib.import_module(module_name)
        except Exception as error:
            raise RuntimeError(f"critical package failed to import: {module_name}") from error
        imported[logical_name] = module
        versions[logical_name] = _installed_version(module, distribution)
    cuda_version = getattr(imported["torch"].version, "cuda", None)
    versions["pytorch-cuda"] = str(cuda_version)
    versions["clip"] = _installed_clip_commit()
    environment_name = os.environ.get("CONDA_DEFAULT_ENV") or Path(sys.prefix).name
    return RuntimeInfo(
        environment_name=environment_name,
        python_version=(
            sys.version_info.major,
            sys.version_info.minor,
            sys.version_info.micro,
        ),
        package_versions=versions,
    )


def _normalized_version(value: str) -> str:
    return value.split("+", 1)[0]


def validate_runtime(runtime: RuntimeInfo) -> None:
    if runtime.environment_name != "atm_native":
        raise ValueError(
            f"environment name must be atm_native, found {runtime.environment_name}"
        )
    if runtime.python_version[:2] != (3, 12):
        raise ValueError(
            "Python must be 3.12.x, found "
            + ".".join(str(value) for value in runtime.python_version)
        )
    for package, expected in EXPECTED_PACKAGE_VERSIONS.items():
        actual = runtime.package_versions.get(package)
        if actual is None:
            raise ValueError(f"critical package did not import: {package}")
        if _normalized_version(str(actual)) != expected:
            raise ValueError(
                f"critical package version mismatch for {package}: "
                f"expected {expected}, found {actual}"
            )


def _load_official_eeg(path: Path, numpy: Any) -> Any:
    loaded = numpy.load(path, allow_pickle=True)
    try:
        if isinstance(loaded, Mapping):
            payload = loaded
        elif getattr(loaded, "shape", None) == ():
            payload = loaded.item()
        else:
            raise ValueError(f"official EEG file must contain a mapping: {path}")
        if not isinstance(payload, Mapping):
            raise ValueError(f"official EEG file must contain a mapping: {path}")
        if "preprocessed_eeg_data" not in payload:
            raise ValueError(f"official EEG file lacks preprocessed_eeg_data: {path}")
        return numpy.asarray(payload["preprocessed_eeg_data"])
    finally:
        close = getattr(loaded, "close", None)
        if close is not None:
            close()


def _require_feature_payload(path: Path, rows: int, torch: Any) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError(f"official feature file must contain a mapping: {path}")
    required = {"img_features", "text_features"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"official feature file lacks required keys {missing}: {path}")
    shapes: dict[str, Any] = {}
    for key in sorted(required):
        feature = payload[key]
        if not hasattr(feature, "shape") or len(feature.shape) == 0:
            raise ValueError(f"official feature {key} has no row dimension: {path}")
        if int(feature.shape[0]) != rows:
            raise ValueError(
                f"official feature {key} row mismatch: expected {rows}, "
                f"found {feature.shape[0]}"
            )
        shapes[key] = [int(value) for value in feature.shape]
    return shapes


def _validate_brainrw(
    path: Path,
    official_test_images: Path,
    expectations: PreflightExpectations,
    numpy: Any,
    torch: Any,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("BrainRW test.pt must contain a mapping")
    required = {"eeg", "label", "img", "text", "session", "ch_names", "times"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"BrainRW test.pt lacks required keys: {missing}")
    eeg = numpy.asarray(payload["eeg"])
    if tuple(eeg.shape) != expectations.test_shape:
        raise ValueError(
            f"BrainRW test EEG shape mismatch: expected {expectations.test_shape}, "
            f"found {tuple(eeg.shape)}"
        )
    leading = expectations.test_shape[:2]
    sessions = numpy.asarray(payload["session"])
    images = numpy.asarray(payload["img"])
    if tuple(sessions.shape) != leading:
        raise ValueError(f"BrainRW session shape mismatch: expected {leading}")
    if tuple(images.shape) != leading:
        raise ValueError(f"BrainRW image shape mismatch: expected {leading}")

    expected_counts = {
        int(key): int(value) for key, value in expectations.session_counts.items()
    }
    for image_index, session_row in enumerate(sessions):
        numeric = [float(value) for value in session_row.tolist()]
        if any(not value.is_integer() for value in numeric):
            raise ValueError(f"BrainRW image {image_index} contains non-integral sessions")
        counts = dict(Counter(int(value) for value in numeric))
        if counts != expected_counts:
            raise ValueError(
                f"BrainRW image {image_index} session counts mismatch: "
                f"expected {expected_counts}, found {counts}"
            )

    brain_ids: list[str] = []
    for image_index, image_row in enumerate(images):
        row_ids = [Path(str(value)).stem for value in image_row.tolist()]
        if len(set(row_ids)) != 1:
            raise ValueError(f"BrainRW image row {image_index} has mixed identities")
        brain_ids.append(row_ids[0])
    if len(set(brain_ids)) != len(brain_ids):
        raise ValueError("BrainRW test.pt contains duplicate canonical image identities")

    suffixes = {".jpg", ".jpeg", ".png"}
    official_ids = sorted(
        path.stem
        for path in official_test_images.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )
    if len(set(official_ids)) != len(official_ids):
        raise ValueError("official test images contain duplicate stem identities")
    if official_ids != sorted(brain_ids):
        raise ValueError("official and BrainRW test image identities do not match")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "eeg_shape": list(eeg.shape),
        "image_count": len(brain_ids),
        "session_counts": {str(key): value for key, value in expected_counts.items()},
    }


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_preflight(
    *,
    protocol_path: Path,
    checkout_path: Path,
    asset_root: Path,
    brainrw_test_path: Path,
    official_test_images: Path,
    manifest_path: Path,
    runtime: RuntimeInfo | None = None,
    expectations: PreflightExpectations | None = None,
) -> dict[str, Any]:
    runtime = collect_runtime_info() if runtime is None else runtime
    expectations = PreflightExpectations() if expectations is None else expectations
    validate_runtime(runtime)
    protocol = Protocol.load(protocol_path)
    protocol.assert_formal_scope()
    source_lock = inspect_checkout(checkout_path)
    asset_inventory = inventory_assets(asset_root)

    numpy = importlib.import_module("numpy")
    torch = importlib.import_module("torch")
    train_path = asset_root / ASSET_RELATIVE_PATHS[0]
    test_path = asset_root / ASSET_RELATIVE_PATHS[1]
    train_eeg = _load_official_eeg(train_path, numpy)
    test_eeg = _load_official_eeg(test_path, numpy)
    if tuple(train_eeg.shape[-2:]) != expectations.eeg_tail:
        raise ValueError(
            f"official train EEG tail mismatch: expected {expectations.eeg_tail}, "
            f"found {tuple(train_eeg.shape[-2:])}"
        )
    if tuple(test_eeg.shape[-2:]) != expectations.eeg_tail:
        raise ValueError(
            f"official test EEG tail mismatch: expected {expectations.eeg_tail}, "
            f"found {tuple(test_eeg.shape[-2:])}"
        )
    if tuple(test_eeg.shape) != expectations.test_shape:
        raise ValueError(
            f"official test EEG shape mismatch: expected {expectations.test_shape}, "
            f"found {tuple(test_eeg.shape)}"
        )
    train_feature_path = asset_root / ASSET_RELATIVE_PATHS[2]
    test_feature_path = asset_root / ASSET_RELATIVE_PATHS[3]
    train_feature_shapes = _require_feature_payload(
        train_feature_path, expectations.train_feature_rows, torch
    )
    test_feature_shapes = _require_feature_payload(
        test_feature_path, expectations.test_feature_rows, torch
    )
    brainrw = _validate_brainrw(
        brainrw_test_path,
        official_test_images,
        expectations,
        numpy,
        torch,
    )
    result: dict[str, Any] = {
        "status": "passed",
        "protocol": {"subject": protocol.subject, "seed": protocol.seed},
        "runtime": {
            "environment_name": runtime.environment_name,
            "python_version": list(runtime.python_version),
            "package_versions": dict(runtime.package_versions),
        },
        "source": source_lock.to_dict(),
        "assets": {
            "root": str(asset_root),
            "files": asset_inventory,
        },
        "official_data": {
            "train_shape": list(train_eeg.shape),
            "test_shape": list(test_eeg.shape),
            "train_feature_shapes": train_feature_shapes,
            "test_feature_shapes": test_feature_shapes,
        },
        "brainrw": brainrw,
    }
    _write_json(result, manifest_path)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deterministic matching-fairness gates")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--checkout", type=Path, default=UPSTREAM_ROOT)
    parser.add_argument("--asset-root", type=Path, default=ASSET_ROOT)
    parser.add_argument("--brainrw-test", type=Path, default=DEFAULT_BRAINRW_TEST)
    parser.add_argument("--official-test-images", type=Path, default=DEFAULT_TEST_IMAGES)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--environment-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "protocol": str(arguments.protocol),
                    "checkout": str(arguments.checkout),
                    "asset_root": str(arguments.asset_root),
                    "brainrw_test": str(arguments.brainrw_test),
                    "official_test_images": str(arguments.official_test_images),
                    "manifest": str(arguments.manifest),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    runtime = collect_runtime_info()
    validate_runtime(runtime)
    if arguments.environment_only:
        print(
            json.dumps(
                {
                    "status": "passed",
                    "environment_name": runtime.environment_name,
                    "python_version": list(runtime.python_version),
                    "package_versions": dict(runtime.package_versions),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    result = run_preflight(
        protocol_path=arguments.protocol,
        checkout_path=arguments.checkout,
        asset_root=arguments.asset_root,
        brainrw_test_path=arguments.brainrw_test,
        official_test_images=arguments.official_test_images,
        manifest_path=arguments.manifest,
        runtime=runtime,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
