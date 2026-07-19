from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import samga_brain_rw.scores as score_module
from samga_brain_rw.hashing import ordered_ids_sha256
from samga_brain_rw.scores import (
    ScoreArtifact,
    independent_retrieval_metrics,
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _metadata() -> dict[str, object]:
    return {
        "checkpoint_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "git_sha": "c" * 40,
        "protocol_sha256": "d" * 64,
        "seed": 17,
        "source_records": [
            {"record_id": "query-b"},
            {"record_id": "query-a"},
        ],
        "split_role": "val-dev",
        "stage": "stage-1",
        "subject": 1,
    }


def _scores_and_ids() -> tuple[np.ndarray, list[str], list[str]]:
    query_ids = ["query-b", "query-a"]
    gallery_ids = [
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
    return scores, query_ids, gallery_ids


def _save_bundle(root: Path) -> tuple[Path, np.ndarray, list[str], list[str]]:
    scores, query_ids, gallery_ids = _scores_and_ids()
    directory = root / "score-bundle"
    ScoreArtifact.save(
        directory,
        scores,
        query_ids,
        gallery_ids,
        _metadata(),
    )
    return directory, scores, query_ids, gallery_ids


def _read_envelope(directory: Path) -> dict[str, object]:
    return json.loads((directory / "metadata.json").read_text("utf-8"))


def _write_envelope(directory: Path, envelope: dict[str, object]) -> None:
    (directory / "metadata.json").write_bytes(
        _canonical_json_bytes(envelope) + b"\n"
    )


def _run_emit_scores(
    experiment_root: Path,
    similarity: Path,
    envelope: Path,
    predictions: Path,
    output: Path,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(experiment_root)
    return subprocess.run(
        [
            sys.executable,
            str(experiment_root / "scripts" / "emit_scores.py"),
            "--input-similarity",
            str(similarity),
            "--input-envelope",
            str(envelope),
            "--input-predictions",
            str(predictions),
            "--output-directory",
            str(output),
        ],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )


def test_metrics_target_by_id_and_stable_utf8_ties() -> None:
    scores, query_ids, gallery_ids = _scores_and_ids()

    metrics = independent_retrieval_metrics(scores, query_ids, gallery_ids)

    assert metrics.query_count == 2
    assert metrics.gallery_count == 6
    assert metrics.top1_count == 1
    assert metrics.top5_count == 2
    assert metrics.top1_rate == 0.5
    assert metrics.top5_rate == 1.0
    first, second = metrics.predictions
    assert first.query_id == "query-b"
    assert first.target_gallery_id == "query-b"
    assert first.predicted_gallery_id == "alpha"
    assert first.target_rank == 3
    assert first.top1 is False
    assert first.top5 is True
    assert second.query_id == "query-a"
    assert second.predicted_gallery_id == "query-a"
    assert second.target_rank == 1


def test_top5_uses_exact_one_based_target_rank() -> None:
    gallery_ids = ["a", "b", "c", "d", "e", "target"]
    scores = np.array([[6, 5, 4, 3, 2, 1]], dtype=np.float32)

    metrics = independent_retrieval_metrics(
        scores,
        ["target"],
        gallery_ids,
    )

    assert metrics.top1_count == 0
    assert metrics.top5_count == 0
    assert metrics.predictions[0].target_rank == 6


@pytest.mark.parametrize(
    ("query_ids", "gallery_ids", "message"),
    [
        (["a", "a"], ["a", "b"], "duplicate query"),
        (["a"], ["a", "a"], "duplicate gallery"),
        (["missing"], ["a", "b"], "missing"),
    ],
)
def test_metrics_reject_duplicate_or_missing_ids(
    query_ids: list[str],
    gallery_ids: list[str],
    message: str,
) -> None:
    scores = np.zeros((len(query_ids), len(gallery_ids)), dtype=np.float32)

    with pytest.raises(ValueError, match=message):
        independent_retrieval_metrics(scores, query_ids, gallery_ids)


@pytest.mark.parametrize(
    "scores",
    [
        np.zeros((2, 2, 1), dtype=np.float32),
        np.zeros((2, 2), dtype=np.int64),
        np.array([[0.0, np.nan]], dtype=np.float32),
        np.array([[0.0, np.inf]], dtype=np.float32),
    ],
)
def test_metrics_reject_invalid_score_arrays(scores: np.ndarray) -> None:
    query_ids = [f"id-{index}" for index in range(scores.shape[0])]
    gallery_ids = ["id-0", "other"]
    with pytest.raises(ValueError):
        independent_retrieval_metrics(scores, query_ids, gallery_ids)


def test_save_creates_complete_strict_bundle_and_load_round_trips(
    tmp_path: Path,
) -> None:
    directory, scores, query_ids, gallery_ids = _save_bundle(tmp_path)

    assert set(os.listdir(directory)) == {
        "similarity.npy",
        "metadata.json",
        "predictions.csv",
    }
    envelope = _read_envelope(directory)
    assert set(envelope) == {
        "schema_version",
        "payload_type",
        "scope",
        "source_records_sha256",
        "ordered_ids_sha256",
        "payload_sha256",
        "provenance",
        "provenance_sha256",
        "metadata",
        "metadata_sha256",
    }
    assert envelope["schema_version"] == 1
    assert envelope["payload_type"] == "samga_brain_rw.score_matrix"
    assert envelope["scope"] == "val-dev"
    bound = envelope["metadata"]
    assert bound["complete"] is True
    assert bound["query_ids"] == query_ids
    assert bound["gallery_ids"] == gallery_ids
    assert bound["query_ids_sha256"] == ordered_ids_sha256(query_ids)
    assert bound["gallery_ids_sha256"] == ordered_ids_sha256(gallery_ids)
    assert envelope["ordered_ids_sha256"] == ordered_ids_sha256(
        query_ids + gallery_ids
    )
    assert envelope["payload_sha256"] == hashlib.sha256(
        (directory / "similarity.npy").read_bytes()
    ).hexdigest()
    assert bound["predictions_sha256"] == hashlib.sha256(
        (directory / "predictions.csv").read_bytes()
    ).hexdigest()
    assert envelope["metadata_sha256"] == _sha256_json(bound)
    assert envelope["provenance_sha256"] == _sha256_json(envelope["provenance"])

    loaded = ScoreArtifact.load(directory, allowed_scopes={"val-dev"})

    np.testing.assert_array_equal(loaded.similarity, scores)
    assert loaded.similarity.flags.writeable is False
    assert loaded.scope == "val-dev"
    assert loaded.query_ids_sha256 == ordered_ids_sha256(query_ids)
    assert loaded.gallery_ids_sha256 == ordered_ids_sha256(gallery_ids)
    assert loaded.query_ids == tuple(query_ids)
    assert loaded.gallery_ids == tuple(gallery_ids)
    assert loaded.metrics.top1_count == 1
    assert loaded.metrics.top5_count == 2
    assert loaded.verified.artifact.payload_type == (
        "samga_brain_rw.score_matrix"
    )


def test_predictions_csv_binds_exact_ids_predictions_and_ranks(
    tmp_path: Path,
) -> None:
    directory, _, _, _ = _save_bundle(tmp_path)

    with (directory / "predictions.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert list(rows[0]) == [
        "query_index",
        "query_id",
        "target_gallery_id",
        "predicted_gallery_id",
        "target_rank",
        "top1",
        "top5",
    ]
    assert rows[0] == {
        "query_index": "0",
        "query_id": "query-b",
        "target_gallery_id": "query-b",
        "predicted_gallery_id": "alpha",
        "target_rank": "3",
        "top1": "0",
        "top5": "1",
    }


def test_save_is_exclusive_and_never_reuses_partial_or_complete_directory(
    tmp_path: Path,
) -> None:
    scores, query_ids, gallery_ids = _scores_and_ids()
    directory = tmp_path / "scores"
    ScoreArtifact.save(directory, scores, query_ids, gallery_ids, _metadata())
    before = {
        name: (directory / name).read_bytes()
        for name in os.listdir(directory)
    }

    with pytest.raises(FileExistsError):
        ScoreArtifact.save(
            directory,
            scores,
            query_ids,
            gallery_ids,
            _metadata(),
        )
    assert {
        name: (directory / name).read_bytes()
        for name in os.listdir(directory)
    } == before

    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "similarity.npy").write_bytes(b"partial")
    with pytest.raises(FileExistsError):
        ScoreArtifact.save(
            partial,
            scores,
            query_ids,
            gallery_ids,
            _metadata(),
        )


def test_failed_save_removes_the_incomplete_directory_it_created(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scores, query_ids, gallery_ids = _scores_and_ids()
    directory = tmp_path / "interrupted"
    real_write = score_module._write_exclusive_file

    def interrupted_write(
        directory_fd: int,
        name: str,
        payload: bytes,
    ) -> None:
        if name == "predictions.csv":
            raise OSError("simulated interrupted publication")
        real_write(directory_fd, name, payload)

    monkeypatch.setattr(score_module, "_write_exclusive_file", interrupted_write)
    with pytest.raises(OSError, match="interrupted"):
        ScoreArtifact.save(
            directory, scores, query_ids, gallery_ids, _metadata()
        )
    assert not directory.exists()


def test_save_rejects_non_val_dev_or_nonfinite_before_creating_directory(
    tmp_path: Path,
) -> None:
    scores, query_ids, gallery_ids = _scores_and_ids()
    formal_metadata = _metadata()
    formal_metadata["split_role"] = "formal-test"

    with pytest.raises(ValueError, match="val-dev"):
        ScoreArtifact.save(
            tmp_path / "formal",
            scores,
            query_ids,
            gallery_ids,
            formal_metadata,
        )
    assert not (tmp_path / "formal").exists()

    nonfinite = scores.copy()
    nonfinite[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        ScoreArtifact.save(
            tmp_path / "nonfinite",
            nonfinite,
            query_ids,
            gallery_ids,
            _metadata(),
        )
    assert not (tmp_path / "nonfinite").exists()


@pytest.mark.parametrize(
    "allowed_scopes",
    [
        set(),
        {"train"},
        {"formal-test"},
        {"val-confirm"},
        {"val-dev", "formal-test"},
    ],
)
def test_load_rejects_disallowed_scope_before_numpy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    allowed_scopes: set[str],
) -> None:
    directory, _, _, _ = _save_bundle(tmp_path)
    load_count = 0

    def tracked_load(*args, **kwargs):
        nonlocal load_count
        load_count += 1
        pytest.fail("np.load must not run for a disallowed scope")

    monkeypatch.setattr(score_module.np, "load", tracked_load)
    with pytest.raises((ValueError, PermissionError), match="scope|val-dev"):
        ScoreArtifact.load(directory, allowed_scopes=allowed_scopes)
    assert load_count == 0


def test_formal_envelope_scope_rejects_before_numpy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory, _, _, _ = _save_bundle(tmp_path)
    envelope = _read_envelope(directory)
    envelope["scope"] = "formal-test"
    _write_envelope(directory, envelope)
    load_count = 0

    def tracked_load(*args, **kwargs):
        nonlocal load_count
        load_count += 1
        pytest.fail("np.load must not run for a formal envelope")

    monkeypatch.setattr(score_module.np, "load", tracked_load)
    with pytest.raises(PermissionError, match="cannot consume"):
        ScoreArtifact.load(directory, allowed_scopes={"val-dev"})
    assert load_count == 0


def test_similarity_payload_hash_mismatch_rejects_before_numpy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory, _, _, _ = _save_bundle(tmp_path)
    with (directory / "similarity.npy").open("ab") as handle:
        handle.write(b"tampered")
    load_count = 0

    def tracked_load(*args, **kwargs):
        nonlocal load_count
        load_count += 1
        pytest.fail("np.load must not run after payload hash rejection")

    monkeypatch.setattr(score_module.np, "load", tracked_load)
    with pytest.raises(ValueError, match="payload SHA-256 mismatch"):
        ScoreArtifact.load(directory, allowed_scopes={"val-dev"})
    assert load_count == 0


def test_predictions_hash_mismatch_rejects_before_numpy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory, _, _, _ = _save_bundle(tmp_path)
    with (directory / "predictions.csv").open("ab") as handle:
        handle.write(b"tampered")
    load_count = 0

    def tracked_load(*args, **kwargs):
        nonlocal load_count
        load_count += 1
        pytest.fail("np.load must not run after prediction hash rejection")

    monkeypatch.setattr(score_module.np, "load", tracked_load)
    with pytest.raises(ValueError, match="predictions SHA-256 mismatch"):
        ScoreArtifact.load(directory, allowed_scopes={"val-dev"})
    assert load_count == 0


@pytest.mark.parametrize(
    "field",
    ["query_ids", "gallery_ids", "source_records"],
)
def test_bound_id_and_source_hashes_are_recomputed_before_numpy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    directory, _, _, _ = _save_bundle(tmp_path)
    envelope = _read_envelope(directory)
    bound = envelope["metadata"]
    if field == "query_ids":
        bound[field] = list(reversed(bound[field]))
    elif field == "gallery_ids":
        bound[field] = list(reversed(bound[field]))
    else:
        bound[field] = list(reversed(bound[field]))
        envelope["source_records_sha256"] = _sha256_json(bound[field])
    envelope["metadata_sha256"] = _sha256_json(bound)
    envelope["ordered_ids_sha256"] = ordered_ids_sha256(bound["ordered_ids"])
    _write_envelope(directory, envelope)
    load_count = 0

    def tracked_load(*args, **kwargs):
        nonlocal load_count
        load_count += 1
        pytest.fail("np.load must not run after bound hash rejection")

    monkeypatch.setattr(score_module.np, "load", tracked_load)
    with pytest.raises(ValueError, match="SHA-256|ordered IDs"):
        ScoreArtifact.load(directory, allowed_scopes={"val-dev"})
    assert load_count == 0


def test_csv_semantics_are_recomputed_even_if_attacker_rehashes_it(
    tmp_path: Path,
) -> None:
    directory, _, _, _ = _save_bundle(tmp_path)
    prediction_path = directory / "predictions.csv"
    rows = list(
        csv.reader(io.StringIO(prediction_path.read_text("utf-8"), newline=""))
    )
    rows[1][3] = "forged-prediction"
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerows(rows)
    forged = output.getvalue().encode("utf-8")
    prediction_path.write_bytes(forged)

    envelope = _read_envelope(directory)
    envelope["metadata"]["predictions_sha256"] = hashlib.sha256(
        forged
    ).hexdigest()
    envelope["metadata_sha256"] = _sha256_json(envelope["metadata"])
    _write_envelope(directory, envelope)

    with pytest.raises(ValueError, match="predictions CSV"):
        ScoreArtifact.load(directory, allowed_scopes={"val-dev"})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("similarity_shape", [999, 999]),
        ("similarity_dtype", "float64"),
    ],
)
def test_declared_shape_and_dtype_must_match_loaded_array(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    directory, _, _, _ = _save_bundle(tmp_path)
    envelope = _read_envelope(directory)
    envelope["metadata"][field] = value
    envelope["metadata_sha256"] = _sha256_json(envelope["metadata"])
    _write_envelope(directory, envelope)
    with pytest.raises(ValueError, match="shape|dtype"):
        ScoreArtifact.load(directory, allowed_scopes={"val-dev"})


def test_missing_completion_marker_or_extra_file_is_rejected_before_numpy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory, _, _, _ = _save_bundle(tmp_path)
    envelope = _read_envelope(directory)
    envelope["metadata"]["complete"] = False
    envelope["metadata_sha256"] = _sha256_json(envelope["metadata"])
    _write_envelope(directory, envelope)
    monkeypatch.setattr(
        score_module.np,
        "load",
        lambda *args, **kwargs: pytest.fail("np.load must not run"),
    )
    with pytest.raises(ValueError, match="complete"):
        ScoreArtifact.load(directory, allowed_scopes={"val-dev"})

    other, _, _, _ = _save_bundle(tmp_path / "other")
    (other / "extra.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="file set"):
        ScoreArtifact.load(other, allowed_scopes={"val-dev"})


def test_emit_scores_cli_reemits_one_complete_typed_val_dev_bundle(
    tmp_path: Path,
    experiment_root: Path,
) -> None:
    source, scores, query_ids, gallery_ids = _save_bundle(tmp_path / "source")
    output = tmp_path / "reemitted"

    result = _run_emit_scores(
        experiment_root,
        source / "similarity.npy",
        source / "metadata.json",
        source / "predictions.csv",
        output,
    )

    assert result.returncode == 0, result.stderr
    loaded = ScoreArtifact.load(output, allowed_scopes={"val-dev"})
    np.testing.assert_array_equal(loaded.similarity, scores)
    assert loaded.query_ids == tuple(query_ids)
    assert loaded.gallery_ids == tuple(gallery_ids)


def test_emit_scores_cli_rejects_descriptor_paths_from_different_bundles(
    tmp_path: Path,
    experiment_root: Path,
) -> None:
    first, _, _, _ = _save_bundle(tmp_path / "first")
    second, _, _, _ = _save_bundle(tmp_path / "second")
    output = tmp_path / "output"

    result = _run_emit_scores(
        experiment_root,
        first / "similarity.npy",
        second / "metadata.json",
        first / "predictions.csv",
        output,
    )

    assert result.returncode != 0
    assert "same complete typed bundle" in result.stderr
    assert not output.exists()


def test_emit_scores_cli_rejects_non_val_dev_scope(
    tmp_path: Path,
    experiment_root: Path,
) -> None:
    source, _, _, _ = _save_bundle(tmp_path / "source")
    envelope = _read_envelope(source)
    envelope["scope"] = "formal-test"
    _write_envelope(source, envelope)
    output = tmp_path / "output"

    result = _run_emit_scores(
        experiment_root,
        source / "similarity.npy",
        source / "metadata.json",
        source / "predictions.csv",
        output,
    )

    assert result.returncode != 0
    assert "cannot consume" in result.stderr
    assert not output.exists()


def test_emit_scores_cli_preserves_preexisting_output_on_conflict(
    tmp_path: Path,
    experiment_root: Path,
) -> None:
    source, _, _, _ = _save_bundle(tmp_path / "source")
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "sentinel"
    sentinel.write_bytes(b"owned")

    result = _run_emit_scores(
        experiment_root,
        source / "similarity.npy",
        source / "metadata.json",
        source / "predictions.csv",
        output,
    )

    assert result.returncode != 0
    assert sentinel.read_bytes() == b"owned"
    assert set(output.iterdir()) == {sentinel}


def _rebind_source_records(
    directory: Path,
    source_records: list[object],
) -> None:
    envelope = _read_envelope(directory)
    bound = envelope["metadata"]
    provenance = envelope["provenance"]
    source_hash = _sha256_json(source_records)
    bound["source_records"] = source_records
    bound["source_records_sha256"] = source_hash
    provenance["source_records_sha256"] = source_hash
    envelope["source_records_sha256"] = source_hash
    envelope["metadata_sha256"] = _sha256_json(bound)
    envelope["provenance_sha256"] = _sha256_json(provenance)
    _write_envelope(directory, envelope)


def _rebind_subject(directory: Path, subject: int) -> None:
    envelope = _read_envelope(directory)
    envelope["metadata"]["subject"] = subject
    envelope["provenance"]["subject"] = subject
    envelope["metadata_sha256"] = _sha256_json(envelope["metadata"])
    envelope["provenance_sha256"] = _sha256_json(envelope["provenance"])
    _write_envelope(directory, envelope)


def test_late_extra_file_after_typed_verification_rejects_before_numpy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory, _, _, _ = _save_bundle(tmp_path)
    real_verify = score_module.verify_typed_artifacts
    load_count = 0

    def verify_then_add_extra(*args, **kwargs):
        capabilities = real_verify(*args, **kwargs)
        (directory / "late-extra.txt").write_bytes(b"late")
        return capabilities

    def tracked_load(*args, **kwargs):
        nonlocal load_count
        load_count += 1
        pytest.fail("np.load must not run after a late extra file appears")

    monkeypatch.setattr(
        score_module, "verify_typed_artifacts", verify_then_add_extra
    )
    monkeypatch.setattr(score_module.np, "load", tracked_load)

    with pytest.raises(ValueError, match="file set"):
        ScoreArtifact.load(directory, allowed_scopes={"val-dev"})
    assert load_count == 0


def test_extra_file_added_during_numpy_load_is_rejected_at_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory, _, _, _ = _save_bundle(tmp_path)
    real_load = score_module.np.load

    def load_then_add_extra(*args, **kwargs):
        value = real_load(*args, **kwargs)
        (directory / "late-extra.txt").write_bytes(b"late")
        return value

    monkeypatch.setattr(score_module.np, "load", load_then_add_extra)

    with pytest.raises(ValueError, match="file set"):
        ScoreArtifact.load(directory, allowed_scopes={"val-dev"})


def test_envelope_replacement_after_verifier_is_rejected_before_numpy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory, _, _, _ = _save_bundle(tmp_path)
    replacement = tmp_path / "replacement-metadata.json"
    real_verify = score_module.verify_typed_artifacts
    load_count = 0

    def verify_then_replace(*args, **kwargs):
        capabilities = real_verify(*args, **kwargs)
        replacement.write_bytes((directory / "metadata.json").read_bytes())
        (directory / "metadata.json").unlink()
        os.link(replacement, directory / "metadata.json")
        return capabilities

    def tracked_load(*args, **kwargs):
        nonlocal load_count
        load_count += 1
        pytest.fail("np.load must not run after envelope replacement")

    monkeypatch.setattr(
        score_module, "verify_typed_artifacts", verify_then_replace
    )
    monkeypatch.setattr(score_module.np, "load", tracked_load)

    with pytest.raises(ValueError, match="envelope.*identity"):
        ScoreArtifact.load(directory, allowed_scopes={"val-dev"})
    assert load_count == 0


_FORBIDDEN_SOURCE_RECORDS = [
    [{"manifest_path": "/sealed/sub-01_test.json"}],
    [{"scope": "val-confirm"}],
    [{"split": "formal-test"}],
    [{"test": {"metrics": {"top1": 0.9}}}],
    [
        {
            "formal": {
                "scores": [0.9],
                "predictions": ["query-a"],
                "rank": 1,
                "top5": 1,
            }
        }
    ],
    [{"image_path": "/sealed/test_images/image-001.jpg"}],
    [
        {
            "record_digest": (
                "02d7e33b3fe8e5a571f8db232ca5fa86abb0c16981876ec84"
                "feae7ba64636f1a"
            )
        }
    ],
]


@pytest.mark.parametrize("source_records", _FORBIDDEN_SOURCE_RECORDS)
def test_save_strictly_rejects_test_or_formal_source_records(
    tmp_path: Path,
    source_records: list[object],
) -> None:
    scores, query_ids, gallery_ids = _scores_and_ids()
    metadata = _metadata()
    metadata["source_records"] = source_records
    directory = tmp_path / "forbidden-source"

    with pytest.raises((PermissionError, ValueError), match="test|formal|val-dev"):
        ScoreArtifact.save(
            directory,
            scores,
            query_ids,
            gallery_ids,
            metadata,
        )
    assert not directory.exists()


@pytest.mark.parametrize("source_records", _FORBIDDEN_SOURCE_RECORDS)
def test_load_strictly_rejects_test_or_formal_source_records(
    tmp_path: Path,
    source_records: list[object],
) -> None:
    directory, _, _, _ = _save_bundle(tmp_path)
    _rebind_source_records(directory, source_records)

    with pytest.raises(
        (PermissionError, ValueError),
        match="denied|test|formal|val-dev",
    ):
        ScoreArtifact.load(directory, allowed_scopes={"val-dev"})


def test_subject_must_be_between_one_and_ten_on_save_and_load(
    tmp_path: Path,
) -> None:
    scores, query_ids, gallery_ids = _scores_and_ids()
    metadata = _metadata()
    metadata["subject"] = 11
    rejected = tmp_path / "subject-11"

    with pytest.raises(ValueError, match="subject.*1.*10"):
        ScoreArtifact.save(
            rejected,
            scores,
            query_ids,
            gallery_ids,
            metadata,
        )
    assert not rejected.exists()

    directory, _, _, _ = _save_bundle(tmp_path / "load")
    _rebind_subject(directory, 11)
    with pytest.raises(ValueError, match="subject.*1.*10"):
        ScoreArtifact.load(directory, allowed_scopes={"val-dev"})


def test_loaded_similarity_cannot_regain_write_permission(
    tmp_path: Path,
) -> None:
    directory, _, _, _ = _save_bundle(tmp_path)

    loaded = ScoreArtifact.load(directory, allowed_scopes={"val-dev"})

    assert loaded.similarity.flags.writeable is False
    with pytest.raises(ValueError):
        loaded.similarity.setflags(write=True)


@pytest.mark.parametrize(
    "relative_output",
    [
        Path("test_images") / "scores",
        Path("val-confirm") / "scores",
        Path("formal-test") / "nested" / "scores",
        Path("nested") / "sub-01_test.json" / "scores",
        Path(
            "02d7e33b3fe8e5a571f8db232ca5fa86abb0c16981876ec84"
            "feae7ba64636f1a"
        )
        / "scores",
    ],
)
def test_save_rejects_forbidden_output_components_before_publication(
    tmp_path: Path,
    relative_output: Path,
) -> None:
    scores, query_ids, gallery_ids = _scores_and_ids()
    directory = tmp_path / relative_output

    with pytest.raises((PermissionError, ValueError), match="output|forbidden"):
        ScoreArtifact.save(
            directory,
            scores,
            query_ids,
            gallery_ids,
            _metadata(),
        )
    assert not directory.exists()


def test_component_symlink_swap_cannot_escape_score_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    safe = tmp_path / "safe"
    parent = safe / "parent"
    parent.mkdir(parents=True)
    parked = tmp_path / "parked"
    outside = tmp_path / "outside"
    (outside / "parent").mkdir(parents=True)
    destination = parent / "bundle"
    scores, query_ids, gallery_ids = _scores_and_ids()
    real_preflight = score_module._preflight_path

    def preflight_then_swap(path: Path, context: str) -> None:
        real_preflight(path, context)
        if context == "score output parent":
            safe.rename(parked)
            safe.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(score_module, "_preflight_path", preflight_then_swap)

    with pytest.raises(ValueError, match="component|symlink|secure"):
        ScoreArtifact.save(
            destination,
            scores,
            query_ids,
            gallery_ids,
            _metadata(),
        )
    assert not (outside / "parent" / "bundle").exists()
    assert not (parked / "parent" / "bundle").exists()


@pytest.mark.parametrize("nested", [False, True])
def test_emit_scores_rejects_output_at_or_inside_source_without_mutation(
    tmp_path: Path,
    experiment_root: Path,
    nested: bool,
) -> None:
    source, _, _, _ = _save_bundle(tmp_path / "source")
    before = {
        name: (source / name).read_bytes()
        for name in os.listdir(source)
    }
    output = source / "nested" if nested else source

    result = _run_emit_scores(
        experiment_root,
        source / "similarity.npy",
        source / "metadata.json",
        source / "predictions.csv",
        output,
    )

    assert result.returncode != 0
    assert "outside the source bundle" in result.stderr
    assert set(os.listdir(source)) == set(before)
    assert {
        name: (source / name).read_bytes()
        for name in os.listdir(source)
    } == before
