from pathlib import Path
import hashlib
import json

import numpy as np
import pytest

import matching_fairness.artifacts as artifact_module
from matching_fairness.artifacts import (
    ScoreArtifact,
    independent_ranks,
    read_score_artifact,
    write_score_artifact,
)


def _artifact(
    *,
    similarity: np.ndarray | None = None,
    query_ids: tuple[str, ...] = ("q0", "q1"),
    gallery_entry_ids: tuple[str, ...] = ("entry0", "entry1"),
    gallery_canonical_ids: tuple[str, ...] = ("image0", "image1"),
    target_canonical_ids: tuple[str, ...] = ("image0", "image1"),
    metadata: dict[str, object] | None = None,
) -> ScoreArtifact:
    return ScoreArtifact(
        similarity=(
            np.eye(2, dtype=np.float32)
            if similarity is None
            else similarity
        ),
        query_ids=query_ids,
        gallery_entry_ids=gallery_entry_ids,
        gallery_canonical_ids=gallery_canonical_ids,
        target_canonical_ids=target_canonical_ids,
        metadata={"model_slug": "fixture"} if metadata is None else metadata,
    )


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n"


def _ordered_ids_sha256(values: list[str]) -> str:
    encoded = json.dumps(
        values,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_round_trip_preserves_ids_and_matrix(tmp_path: Path) -> None:
    artifact = ScoreArtifact(
        similarity=np.eye(3),
        query_ids=("q0", "q1", "q2"),
        gallery_entry_ids=("e0", "e1", "e2"),
        gallery_canonical_ids=("i0", "i1", "i2"),
        target_canonical_ids=("i0", "i1", "i2"),
        metadata={"model_slug": "fixture"},
    )
    write_score_artifact(tmp_path / "score", artifact)
    loaded = read_score_artifact(tmp_path / "score")
    np.testing.assert_array_equal(loaded.similarity, artifact.similarity)
    assert loaded.query_ids == artifact.query_ids
    assert loaded.target_canonical_ids == artifact.target_canonical_ids


def test_targets_are_resolved_by_canonical_id_not_diagonal() -> None:
    artifact = ScoreArtifact(
        similarity=np.array([[0.1, 0.9], [0.8, 0.2]]),
        query_ids=("q-a", "q-b"),
        gallery_entry_ids=("entry-b", "entry-a"),
        gallery_canonical_ids=("image-b", "image-a"),
        target_canonical_ids=("image-a", "image-b"),
        metadata={"model_slug": "fixture"},
    )
    assert independent_ranks(artifact).tolist() == [1, 1]


def test_write_creates_exact_canonical_bundle_with_audit_hashes(
    tmp_path: Path,
) -> None:
    artifact = _artifact(
        metadata={
            "checkpoint_role": "val_selected_formal",
            "checkpoint_sha256": "a" * 64,
            "data_hashes": {"test": "b" * 64},
            "model_slug": "fixture",
            "query_mode": "standard",
            "score_semantics": "cosine_similarity_higher_is_better",
            "seed": 42,
            "source_commit": "c" * 40,
            "subject": "sub-08",
        }
    )
    directory = tmp_path / "score"

    write_score_artifact(directory, artifact)

    assert {path.name for path in directory.iterdir()} == {
        "metadata.json",
        "similarity.npy",
    }
    metadata_text = (directory / "metadata.json").read_text(encoding="utf-8")
    payload = json.loads(metadata_text)
    assert metadata_text == _canonical_json(payload)
    assert payload["similarity_sha256"] == hashlib.sha256(
        (directory / "similarity.npy").read_bytes()
    ).hexdigest()
    for field in (
        "query_ids",
        "gallery_entry_ids",
        "gallery_canonical_ids",
        "target_canonical_ids",
    ):
        assert payload[f"{field}_sha256"] == _ordered_ids_sha256(payload[field])
    for field, value in artifact.metadata.items():
        assert payload[field] == value


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_validation_rejects_non_finite_similarity(bad_value: float) -> None:
    similarity = np.eye(2)
    similarity[0, 0] = bad_value

    with pytest.raises(ValueError, match="NaN or Inf"):
        _artifact(similarity=similarity).validate()


def test_validation_rejects_duplicate_gallery_entry_ids() -> None:
    with pytest.raises(ValueError, match="gallery entry IDs must be unique"):
        _artifact(gallery_entry_ids=("entry0", "entry0")).validate()


def test_validation_rejects_duplicate_query_ids() -> None:
    with pytest.raises(ValueError, match="query IDs must be unique"):
        _artifact(query_ids=("q0", "q0")).validate()


def test_validation_rejects_row_id_mismatch() -> None:
    with pytest.raises(ValueError, match="query metadata does not match rows"):
        _artifact(query_ids=("q0",)).validate()


def test_validation_rejects_missing_target_by_default() -> None:
    with pytest.raises(ValueError, match="target canonical IDs missing from gallery"):
        _artifact(target_canonical_ids=("image0", "absent")).validate()


def test_explicitly_allowed_unanswerable_target_gets_gallery_size_plus_one_rank(
    tmp_path: Path,
) -> None:
    artifact = _artifact(
        target_canonical_ids=("image0", "absent"),
        metadata={
            "allow_unanswerable_targets": True,
            "model_slug": "fixture",
        },
    )
    directory = tmp_path / "score"

    write_score_artifact(directory, artifact)
    loaded = read_score_artifact(directory)

    assert loaded.target_canonical_ids == ("image0", "absent")
    assert loaded.metadata["allow_unanswerable_targets"] is True
    assert independent_ranks(loaded).tolist() == [1, 3]


def test_read_rejects_tampered_matrix_hash(tmp_path: Path) -> None:
    directory = tmp_path / "score"
    write_score_artifact(directory, _artifact())
    with (directory / "similarity.npy").open("ab") as stream:
        stream.write(b"tampered")

    with pytest.raises(ValueError, match="similarity SHA-256 mismatch"):
        read_score_artifact(directory)


@pytest.mark.parametrize(
    "field",
    (
        "query_ids",
        "gallery_entry_ids",
        "gallery_canonical_ids",
        "target_canonical_ids",
    ),
)
def test_read_verifies_every_stored_ordered_id_hash(
    tmp_path: Path,
    field: str,
) -> None:
    directory = tmp_path / "score"
    write_score_artifact(directory, _artifact())
    path = directory / "metadata.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = list(reversed(payload[field]))
    path.write_text(_canonical_json(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=f"{field} SHA-256 mismatch"):
        read_score_artifact(directory)


@pytest.mark.parametrize("present_file", ("similarity.npy", "metadata.json"))
def test_read_rejects_partial_artifact(
    tmp_path: Path,
    present_file: str,
) -> None:
    directory = tmp_path / "partial"
    directory.mkdir()
    (directory / present_file).write_bytes(b"partial")

    with pytest.raises(ValueError, match="exactly similarity.npy and metadata.json"):
        read_score_artifact(directory)


def test_write_is_exclusive_and_preserves_existing_artifact(tmp_path: Path) -> None:
    directory = tmp_path / "score"
    write_score_artifact(directory, _artifact())
    before = {path.name: path.read_bytes() for path in directory.iterdir()}

    with pytest.raises(FileExistsError):
        write_score_artifact(directory, _artifact(similarity=2 * np.eye(2)))

    assert {path.name: path.read_bytes() for path in directory.iterdir()} == before


def test_failed_write_does_not_publish_partial_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def interrupted_save(*args: object, **kwargs: object) -> None:
        raise OSError("interrupted")

    monkeypatch.setattr(artifact_module.np, "save", interrupted_save)
    directory = tmp_path / "score"

    with pytest.raises(OSError, match="interrupted"):
        write_score_artifact(directory, _artifact())

    assert not directory.exists()
    assert list(tmp_path.iterdir()) == []
