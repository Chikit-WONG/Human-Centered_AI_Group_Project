from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from samga_brain_rw.hashing import canonical_json_bytes
from samga_brain_rw.scores import ScoreArtifact


ROLE_DIRECTORIES = {
    "in_loop": "in_loop",
    "saved_checkpoint": "saved_checkpoint",
    "repeat_emission": "repeat_emission",
    "reload_evaluation": "reload_evaluation",
}


def _metadata(*, config_sha256: str = "b" * 64) -> dict[str, object]:
    return {
        "checkpoint_sha256": "a" * 64,
        "config_sha256": config_sha256,
        "git_sha": "c" * 40,
        "protocol_sha256": "d" * 64,
        "seed": 42,
        "source_records": [
            {"record_id": "query-b"},
            {"record_id": "query-a"},
        ],
        "split_role": "val-dev",
        "stage": "stage-0",
        "subject": 1,
    }


def _scores_and_ids() -> tuple[np.ndarray, list[str], list[str]]:
    queries = ["query-b", "query-a"]
    galleries = [
        "query-a",
        "zeta",
        "query-b",
        "alpha",
        "other-1",
        "other-2",
    ]
    scores = np.array(
        [
            [0.1, 0.9, 0.8, 0.9, 0.0, -1.0],
            [1.0, 0.2, 0.1, 0.3, 0.4, 0.5],
        ],
        dtype=np.float32,
    )
    return scores, queries, galleries


def _save_bundle(
    run_directory: Path,
    role: str,
    *,
    scores: np.ndarray | None = None,
    query_ids: list[str] | None = None,
    gallery_ids: list[str] | None = None,
    metadata: dict[str, object] | None = None,
) -> Path:
    base_scores, base_queries, base_galleries = _scores_and_ids()
    directory = run_directory / ROLE_DIRECTORIES[role]
    ScoreArtifact.save(
        directory,
        base_scores if scores is None else scores,
        base_queries if query_ids is None else query_ids,
        base_galleries if gallery_ids is None else gallery_ids,
        _metadata() if metadata is None else metadata,
    )
    return directory


def _save_complete_run(run_directory: Path) -> dict[str, np.ndarray]:
    base, _, _ = _scores_and_ids()
    matrices = {
        "in_loop": base.copy(),
        "saved_checkpoint": base.copy(),
        "repeat_emission": base.copy(),
        "reload_evaluation": base.copy(),
    }
    matrices["saved_checkpoint"][0, 0] += np.float32(2e-7)
    matrices["repeat_emission"][1, 1] -= np.float32(3e-7)
    matrices["reload_evaluation"][0, 4] += np.float32(4e-7)
    for role, matrix in matrices.items():
        _save_bundle(run_directory, role, scores=matrix)
    return matrices


def _run_cli(
    experiment_root: Path,
    run_directory: Path,
    output: Path,
    *,
    scope: str = "val-dev",
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(experiment_root)
    return subprocess.run(
        [
            sys.executable,
            str(experiment_root / "scripts" / "check_baseline_parity.py"),
            "--run-dir",
            str(run_directory),
            "--scope",
            scope,
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )


def _load_cli(experiment_root: Path) -> ModuleType:
    path = experiment_root / "scripts" / "check_baseline_parity.py"
    spec = importlib.util.spec_from_file_location("baseline_parity_test_cli", path)
    if spec is None or spec.loader is None:
        raise AssertionError("could not load baseline parity CLI")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def test_cli_verifies_all_four_bundles_and_writes_bound_canonical_report(
    experiment_root: Path,
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "fresh-baseline-run"
    matrices = _save_complete_run(run_directory)
    output = tmp_path / "baseline_parity.json"

    completed = _run_cli(experiment_root, run_directory, output)

    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_bytes())
    assert report["schema_version"] == 1
    assert report["report_type"] == "samga_brain_rw.baseline_parity"
    assert report["scope"] == "val-dev"
    assert report["passed"] is True
    assert report["run_directory"] == str(run_directory.resolve())
    assert report["tolerance"] == 1e-6
    assert set(report["artifacts"]) == set(ROLE_DIRECTORIES)
    assert len(report["comparisons"]) == 6
    assert report["summary"]["artifact_count"] == 4
    assert report["summary"]["comparison_count"] == 6
    expected_maximum = max(
        float(
            np.max(
                np.abs(left.astype(np.longdouble) - right.astype(np.longdouble))
            )
        )
        for left_index, left in enumerate(matrices.values())
        for right in tuple(matrices.values())[left_index + 1 :]
    )
    assert report["summary"]["maximum_absolute_score_difference"] == pytest.approx(
        expected_maximum,
        rel=0.0,
        abs=1e-15,
    )
    for role, directory_name in ROLE_DIRECTORIES.items():
        artifact = report["artifacts"][role]
        assert artifact["directory"] == directory_name
        assert set(artifact["files"]) == {
            "metadata.json",
            "predictions.csv",
            "similarity.npy",
        }
        for name, descriptor in artifact["files"].items():
            path = run_directory / directory_name / name
            assert descriptor["sha256"] == _sha256(path)
            assert descriptor["size"] == path.stat().st_size
    assert output.read_bytes() == canonical_json_bytes(report) + b"\n"


def test_cli_is_exclusive_and_preserves_preexisting_output(
    experiment_root: Path,
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "fresh-baseline-run"
    _save_complete_run(run_directory)
    output = tmp_path / "baseline_parity.json"
    output.write_bytes(b"owned")

    completed = _run_cli(experiment_root, run_directory, output)

    assert completed.returncode != 0
    assert output.read_bytes() == b"owned"


@pytest.mark.parametrize("scope", ["train", "val-confirm", "formal-test", "test"])
def test_cli_rejects_every_scope_except_val_dev_before_artifact_loading(
    experiment_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scope: str,
) -> None:
    cli = _load_cli(experiment_root)
    monkeypatch.setattr(
        cli.ScoreArtifact,
        "load",
        lambda *args, **kwargs: pytest.fail("must reject scope before score load"),
    )
    with pytest.raises(SystemExit):
        cli.main(
            [
                "--run-dir",
                str(tmp_path / "unused"),
                "--scope",
                scope,
                "--output",
                str(tmp_path / "unused.json"),
            ]
        )


@pytest.mark.parametrize(
    "relative_path",
    [
        Path("val-confirm") / "run",
        Path("formal-test") / "run",
        Path("test_images") / "run",
        Path("sub-01_test.json") / "run",
    ],
)
def test_cli_rejects_sealed_run_directory_components(
    experiment_root: Path,
    tmp_path: Path,
    relative_path: Path,
) -> None:
    safe = tmp_path / "safe-run"
    _save_complete_run(safe)
    sealed = tmp_path / relative_path
    sealed.parent.mkdir(parents=True, exist_ok=True)
    safe.rename(sealed)
    output = tmp_path / "report.json"

    completed = _run_cli(experiment_root, sealed, output)

    assert completed.returncode != 0
    assert not output.exists()


def test_cli_rejects_sealed_or_symlinked_output_without_escape(
    experiment_root: Path,
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "fresh-baseline-run"
    _save_complete_run(run_directory)
    sealed_output = tmp_path / "formal-test" / "report.json"
    sealed = _run_cli(experiment_root, run_directory, sealed_output)
    assert sealed.returncode != 0
    assert not sealed_output.parent.exists()

    outside = tmp_path / "outside"
    outside.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(outside, target_is_directory=True)
    escaped_output = linked_parent / "report.json"
    linked = _run_cli(experiment_root, run_directory, escaped_output)
    assert linked.returncode != 0
    assert not (outside / "report.json").exists()


def test_cli_rejects_symlinked_bundle_directory(
    experiment_root: Path,
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "fresh-baseline-run"
    _save_complete_run(run_directory)
    original = run_directory / ROLE_DIRECTORIES["reload_evaluation"]
    parked = tmp_path / "parked-reload"
    original.rename(parked)
    original.symlink_to(parked, target_is_directory=True)
    output = tmp_path / "report.json"

    completed = _run_cli(experiment_root, run_directory, output)

    assert completed.returncode != 0
    assert not output.exists()


def test_cli_detects_run_directory_replacement_during_score_load(
    experiment_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_directory = tmp_path / "fresh-baseline-run"
    replacement = tmp_path / "replacement-run"
    _save_complete_run(run_directory)
    _save_complete_run(replacement)
    parked = tmp_path / "parked-run"
    output = tmp_path / "report.json"
    cli = _load_cli(experiment_root)
    real_score_artifact = ScoreArtifact
    swapped = False

    class SwappingScoreArtifact:
        @classmethod
        def load(cls, directory: Path, allowed_scopes: set[str]) -> ScoreArtifact:
            nonlocal swapped
            if not swapped:
                run_directory.rename(parked)
                replacement.rename(run_directory)
                swapped = True
            return real_score_artifact.load(directory, allowed_scopes)

    monkeypatch.setattr(cli, "ScoreArtifact", SwappingScoreArtifact)
    with pytest.raises(SystemExit):
        cli.main(
            [
                "--run-dir",
                str(run_directory),
                "--scope",
                "val-dev",
                "--output",
                str(output),
            ]
        )
    assert not output.exists()


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("ordered_ids", "ordered"),
        ("predictions", "prediction|metric"),
        ("scores", "tolerance|score"),
        ("provenance", "provenance"),
    ],
)
def test_cli_rejects_every_parity_contract_mismatch(
    experiment_root: Path,
    tmp_path: Path,
    failure: str,
    message: str,
) -> None:
    run_directory = tmp_path / f"fresh-baseline-{failure}"
    base, queries, galleries = _scores_and_ids()
    for role in ROLE_DIRECTORIES:
        if role != "saved_checkpoint":
            _save_bundle(run_directory, role)
            continue
        if failure == "ordered_ids":
            _save_bundle(
                run_directory,
                role,
                scores=base[::-1].copy(),
                query_ids=list(reversed(queries)),
            )
        elif failure == "predictions":
            changed = base.copy()
            changed[0, galleries.index("query-b")] = 1.1
            _save_bundle(run_directory, role, scores=changed)
        elif failure == "scores":
            changed = base.copy()
            changed[1, 1] += np.float32(2e-6)
            _save_bundle(run_directory, role, scores=changed)
        else:
            _save_bundle(
                run_directory,
                role,
                metadata=_metadata(config_sha256="e" * 64),
            )
    output = tmp_path / f"{failure}.json"

    completed = _run_cli(experiment_root, run_directory, output)

    assert completed.returncode != 0
    assert not output.exists()
    assert re.search(message, completed.stderr, flags=re.IGNORECASE)
