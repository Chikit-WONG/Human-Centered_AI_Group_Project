import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

from matching_fairness.provenance import inspect_checkout, sha256_file
from scripts.fetch_assets import (
    ASSET_RELATIVE_PATHS,
    HF_DATASET_REPO,
    fetch_assets,
    inventory_assets,
)
from scripts.preflight import PreflightExpectations, RuntimeInfo, run_preflight
from scripts.fetch_upstream import resolve_detached_checkout, write_source_lock


CONFIG = Path("experiments/matching_fairness/configs/protocol_sub08_seed42.json")
OFFICIAL_URL = "https://github.com/dongyangli-del/EEG_Image_decode.git"
REQUIRED_SOURCE_FILES = (
    "Retrieval/train_unified.py",
    "Retrieval/retrieval_engine.py",
    "Retrieval/eeg_encoders.py",
    "eegdatasets.py",
    "encoder_utils.py",
    "models/atms.py",
)


def _git(checkout: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-C", str(checkout), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _make_checkout(tmp_path: Path, *, detached: bool = True) -> Path:
    checkout = tmp_path / "checkout"
    subprocess.run(
        ["git", "init", str(checkout)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    _git(checkout, "config", "user.email", "test@example.com")
    _git(checkout, "config", "user.name", "Test")
    _git(checkout, "remote", "add", "origin", OFFICIAL_URL)
    for relative in REQUIRED_SOURCE_FILES:
        path = checkout / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n", encoding="utf-8")
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-m", "fixture")
    if detached:
        _git(checkout, "checkout", "--detach", "HEAD")
    return checkout


def _write_official_npy(path: Path, eeg: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(
        path,
        {
            "preprocessed_eeg_data": eeg,
            "times": np.arange(eeg.shape[-1]),
            "ch_names": np.asarray([f"c{i}" for i in range(eeg.shape[-2])]),
        },
        allow_pickle=True,
    )


def _write_tiny_preflight_fixtures(tmp_path: Path) -> dict[str, Path]:
    assets = tmp_path / "assets"
    train_eeg = np.zeros((6, 3, 5), dtype=np.float32)
    test_eeg = np.zeros((2, 4, 3, 5), dtype=np.float32)
    _write_official_npy(
        assets / "Preprocessed_data_250Hz/sub-08/preprocessed_eeg_training.npy",
        train_eeg,
    )
    _write_official_npy(
        assets / "Preprocessed_data_250Hz/sub-08/preprocessed_eeg_test.npy",
        test_eeg,
    )
    torch.save(
        {
            "img_features": torch.zeros(6, 2),
            "text_features": torch.ones(6, 2),
        },
        assets / "ViT-H-14_features_train.pt",
    )
    torch.save(
        {
            "img_features": torch.zeros(2, 2),
            "text_features": torch.ones(2, 2),
        },
        assets / "ViT-H-14_features_test.pt",
    )

    brainrw = tmp_path / "sub-08/test.pt"
    brainrw.parent.mkdir(parents=True)
    image_names = ("alpha.jpg", "beta.jpg")
    torch.save(
        {
            "eeg": test_eeg,
            "label": np.repeat(np.arange(2)[:, None], 4, axis=1),
            "img": np.asarray([[name] * 4 for name in image_names]),
            "text": np.asarray([[stem] * 4 for stem in ("alpha", "beta")]),
            "session": np.asarray([[0, 1, 2, 3]] * 2),
            "ch_names": ["c0", "c1", "c2"],
            "times": np.arange(5),
        },
        brainrw,
    )
    images = tmp_path / "test_images"
    images.mkdir()
    for name in image_names:
        (images / name).write_bytes(b"fixture")
    return {"asset_root": assets, "brainrw_test": brainrw, "image_root": images}


def _valid_runtime() -> RuntimeInfo:
    return RuntimeInfo(
        environment_name="atm_native",
        python_version=(3, 12, 7),
        package_versions={
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
        },
    )


def test_sha256_file_is_content_sensitive(tmp_path: Path) -> None:
    path = tmp_path / "x"
    path.write_bytes(b"abc")
    assert sha256_file(path) == (
        "ba7816bf8f01cfea414140de5dae2223"
        "b00361a396177a9cb410ff61f20015ad"
    )


def test_source_lock_rejects_non_detached_checkout(tmp_path: Path) -> None:
    checkout = _make_checkout(tmp_path, detached=False)
    with pytest.raises(ValueError, match="detached"):
        inspect_checkout(checkout)


def test_source_lock_records_clean_detached_checkout(tmp_path: Path) -> None:
    checkout = _make_checkout(tmp_path)
    lock = inspect_checkout(checkout)
    assert lock.url == OFFICIAL_URL
    assert lock.branch == "develop"
    assert len(lock.commit) == 40
    assert len(lock.checkout_sha256) == 64


def test_resolve_detached_checkout_from_local_bare_remote(tmp_path: Path) -> None:
    remote = tmp_path / "origin.git"
    subprocess.run(
        ["git", "init", "--bare", str(remote)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    seed = tmp_path / "seed"
    subprocess.run(
        ["git", "init", str(seed)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    _git(seed, "config", "user.email", "test@example.com")
    _git(seed, "config", "user.name", "Test")
    for relative in REQUIRED_SOURCE_FILES:
        path = seed / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\n", encoding="utf-8")
    _git(seed, "add", ".")
    _git(seed, "commit", "-m", "fixture")
    _git(seed, "branch", "-M", "develop")
    _git(seed, "remote", "add", "origin", str(remote))
    _git(seed, "push", "-u", "origin", "develop")
    expected_commit = subprocess.check_output(
        ["git", "-C", str(seed), "rev-parse", "HEAD"], text=True
    ).strip()

    checkout = tmp_path / "resolved"
    lock = resolve_detached_checkout(checkout, str(remote), "develop")

    assert lock.url == str(remote)
    assert lock.branch == "develop"
    assert lock.commit == expected_commit
    assert len(lock.checkout_sha256) == 64
    detached = subprocess.run(
        ["git", "-C", str(checkout), "symbolic-ref", "-q", "HEAD"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert detached.returncode == 1
    assert all((checkout / relative).is_file() for relative in REQUIRED_SOURCE_FILES)

    manifest = tmp_path / "manifests/upstream_lock.json"
    write_source_lock(lock, manifest)
    assert json.loads(manifest.read_text(encoding="utf-8")) == lock.to_dict()


def test_source_lock_rejects_dirty_or_wrong_remote(tmp_path: Path) -> None:
    checkout = _make_checkout(tmp_path)
    (checkout / REQUIRED_SOURCE_FILES[0]).write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="clean"):
        inspect_checkout(checkout)
    _git(checkout, "checkout", "--", REQUIRED_SOURCE_FILES[0])
    _git(checkout, "remote", "set-url", "origin", "https://example.com/wrong.git")
    with pytest.raises(ValueError, match="remote URL"):
        inspect_checkout(checkout)


def test_fetch_assets_uses_exact_hf_argv_and_records_real_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset_root = tmp_path / "assets"
    expected_contents: dict[str, bytes] = {}
    for index, relative in enumerate(ASSET_RELATIVE_PATHS):
        path = asset_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        contents = f"asset-{index}".encode()
        path.write_bytes(contents)
        expected_contents[relative] = contents
    calls: list[list[str]] = []

    def fake_run(arguments: list[str], *, check: bool) -> None:
        assert check is True
        calls.append(arguments)

    monkeypatch.setattr("scripts.fetch_assets.subprocess.run", fake_run)
    manifest = tmp_path / "assets_lock.json"
    result = fetch_assets(asset_root=asset_root, manifest_path=manifest)

    assert HF_DATASET_REPO == "LidongYang/EEG_Image_decode"
    assert ASSET_RELATIVE_PATHS == (
        "Preprocessed_data_250Hz/sub-08/preprocessed_eeg_training.npy",
        "Preprocessed_data_250Hz/sub-08/preprocessed_eeg_test.npy",
        "ViT-H-14_features_train.pt",
        "ViT-H-14_features_test.pt",
    )
    assert calls == [
        [
            "hf",
            "download",
            "LidongYang/EEG_Image_decode",
            "--repo-type",
            "dataset",
            "--include",
            "Preprocessed_data_250Hz/sub-08/preprocessed_eeg_training.npy",
            "--include",
            "Preprocessed_data_250Hz/sub-08/preprocessed_eeg_test.npy",
            "--include",
            "ViT-H-14_features_train.pt",
            "--include",
            "ViT-H-14_features_test.pt",
            "--local-dir",
            str(asset_root),
        ]
    ]
    assert json.loads(manifest.read_text(encoding="utf-8")) == result
    for relative, contents in expected_contents.items():
        assert result["files"][relative] == {
            "bytes": len(contents),
            "sha256": sha256_file(asset_root / relative),
        }


def test_asset_inventory_rejects_symlink_outside_root(tmp_path: Path) -> None:
    asset_root = tmp_path / "assets"
    for relative in ASSET_RELATIVE_PATHS:
        path = asset_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    outside = tmp_path / "outside"
    outside.write_bytes(b"outside")
    escaped = asset_root / ASSET_RELATIVE_PATHS[0]
    escaped.unlink()
    escaped.symlink_to(outside)
    with pytest.raises(ValueError, match="outside asset root"):
        inventory_assets(asset_root)


def test_preflight_accepts_complete_tiny_fixtures(tmp_path: Path) -> None:
    checkout = _make_checkout(tmp_path)
    paths = _write_tiny_preflight_fixtures(tmp_path)
    manifest = tmp_path / "preflight.json"
    expectations = PreflightExpectations(
        eeg_tail=(3, 5),
        test_shape=(2, 4, 3, 5),
        train_feature_rows=6,
        test_feature_rows=2,
        session_counts={0: 1, 1: 1, 2: 1, 3: 1},
    )
    runtime = _valid_runtime()
    result = run_preflight(
        protocol_path=CONFIG,
        checkout_path=checkout,
        asset_root=paths["asset_root"],
        brainrw_test_path=paths["brainrw_test"],
        official_test_images=paths["image_root"],
        manifest_path=manifest,
        runtime=runtime,
        expectations=expectations,
    )
    assert result["status"] == "passed"
    assert result["official_data"]["test_shape"] == [2, 4, 3, 5]
    assert result["brainrw"]["session_counts"] == {
        "0": 1,
        "1": 1,
        "2": 1,
        "3": 1,
    }
    assert json.loads(manifest.read_text(encoding="utf-8")) == result


def test_preflight_rejects_escaping_asset_before_deserialization(
    tmp_path: Path,
) -> None:
    checkout = _make_checkout(tmp_path)
    paths = _write_tiny_preflight_fixtures(tmp_path)
    outside = tmp_path / "outside.npy"
    outside.write_bytes(b"not a NumPy pickle")
    escaped = paths["asset_root"] / ASSET_RELATIVE_PATHS[0]
    escaped.unlink()
    escaped.symlink_to(outside)
    manifest = tmp_path / "preflight.json"

    with pytest.raises(ValueError, match="outside asset root"):
        run_preflight(
            protocol_path=CONFIG,
            checkout_path=checkout,
            asset_root=paths["asset_root"],
            brainrw_test_path=paths["brainrw_test"],
            official_test_images=paths["image_root"],
            manifest_path=manifest,
            runtime=_valid_runtime(),
            expectations=PreflightExpectations(
                eeg_tail=(3, 5),
                test_shape=(2, 4, 3, 5),
                train_feature_rows=6,
                test_feature_rows=2,
                session_counts={0: 1, 1: 1, 2: 1, 3: 1},
            ),
        )
    assert not manifest.exists()


def test_preflight_rejects_image_identity_mismatch(tmp_path: Path) -> None:
    checkout = _make_checkout(tmp_path)
    paths = _write_tiny_preflight_fixtures(tmp_path)
    (paths["image_root"] / "beta.jpg").rename(paths["image_root"] / "wrong.jpg")
    with pytest.raises(ValueError, match="image identities"):
        run_preflight(
            protocol_path=CONFIG,
            checkout_path=checkout,
            asset_root=paths["asset_root"],
            brainrw_test_path=paths["brainrw_test"],
            official_test_images=paths["image_root"],
            manifest_path=tmp_path / "preflight.json",
            runtime=_valid_runtime(),
            expectations=PreflightExpectations(
                eeg_tail=(3, 5),
                test_shape=(2, 4, 3, 5),
                train_feature_rows=6,
                test_feature_rows=2,
                session_counts={0: 1, 1: 1, 2: 1, 3: 1},
            ),
        )
