from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import samga_brain_rw.cache_parity as parity
from samga_brain_rw.access import TypedArtifact
from samga_brain_rw.hashing import ordered_ids_sha256


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


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json_bytes(value))


def _cache_artifact(
    root: Path,
    array: np.ndarray,
    *,
    metadata_extra: dict[str, object] | None = None,
) -> TypedArtifact:
    root.mkdir(parents=True, exist_ok=True)
    payload_path = root / "features.npy"
    with payload_path.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    provenance = {
        "producer": "synthetic-stage0-test",
        "source_components": ["tests", "fixtures"],
    }
    metadata: dict[str, object] = {
        "ordered_ids": [f"record-{index:03d}" for index in range(array.shape[0])],
        "source_records": [
            {"record_id": f"record-{index:03d}"}
            for index in range(array.shape[0])
        ],
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "split": "train",
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    envelope = {
        "schema_version": 1,
        "payload_type": "samga_brain_rw.train_cache",
        "scope": "train",
        "source_records_sha256": _sha256_json(metadata["source_records"]),
        "ordered_ids_sha256": ordered_ids_sha256(metadata["ordered_ids"]),
        "payload_sha256": hashlib.sha256(payload_path.read_bytes()).hexdigest(),
        "provenance": provenance,
        "provenance_sha256": _sha256_json(provenance),
        "metadata": metadata,
        "metadata_sha256": _sha256_json(metadata),
    }
    envelope_path = root / "features.envelope.json"
    _write_json(envelope_path, envelope)
    return TypedArtifact(
        payload_type="samga_brain_rw.train_cache",
        payload_path=payload_path,
        envelope_path=envelope_path,
    )


def _role_payload(
    role: str,
    *,
    rows: list[int],
    concept_ids: list[str],
    ordered_ids: list[str],
) -> dict[str, object]:
    validation = role != "train"
    return {
        "concept_count": len(concept_ids),
        "concept_ids": concept_ids,
        "gallery_ids": ordered_ids if validation else [],
        "ordered_ids": ordered_ids,
        "payload_type": "samga_brain_rw.role_payload",
        "query_ids": ordered_ids if validation else [],
        "row_count": len(rows),
        "row_indices": rows,
        "schema_version": 1,
        "scope": role,
    }


def _protocol_manifest(
    subject: int,
    *,
    train_rows: list[int],
    train_concept_ids: list[str],
    val_dev_rows: list[int],
    val_dev_concept_ids: list[str],
    val_dev_ids: list[str],
) -> dict[str, object]:
    records_sha256 = "1" * 64
    protocol_config_sha256 = "2" * 64
    source_manifest_sha256 = hashlib.sha256(
        f"source-{subject}".encode("utf-8")
    ).hexdigest()
    provenance = {
        "protocol_config_sha256": protocol_config_sha256,
        "source_manifest_sha256": source_manifest_sha256,
    }
    role_payloads = {
        "train": _role_payload(
            "train",
            rows=train_rows,
            concept_ids=train_concept_ids,
            ordered_ids=train_concept_ids,
        ),
        "val-dev": _role_payload(
            "val-dev",
            rows=val_dev_rows,
            concept_ids=val_dev_concept_ids,
            ordered_ids=val_dev_ids,
        ),
        "val-confirm": _role_payload(
            "val-confirm",
            rows=[3],
            concept_ids=["confirm-concept"],
            ordered_ids=["confirm-stimulus"],
        ),
    }
    role_artifacts = {
        role: {
            "ordered_ids_sha256": ordered_ids_sha256(payload["ordered_ids"]),
            "payload_type": "samga_brain_rw.role_payload",
            "provenance_sha256": _sha256_json(provenance),
            "role_payload_sha256": _sha256_json(payload),
            "schema_version": 1,
            "scope": role,
            "source_records_sha256": records_sha256,
        }
        for role, payload in role_payloads.items()
    }
    split_assignment = {
        "payload_type": "samga_brain_rw.split_assignment",
        "protocol_config_sha256": protocol_config_sha256,
        "records_sha256": records_sha256,
        "schema_version": 1,
    }
    return {
        "payload_type": "samga_brain_rw.subject_protocol_manifest",
        "protocol_config_sha256": protocol_config_sha256,
        "records_sha256": records_sha256,
        "role_artifacts": role_artifacts,
        "role_payloads": role_payloads,
        "schema_version": 1,
        "source_manifest_path": f"sub-{subject:02d}_train.json",
        "source_manifest_sha256": source_manifest_sha256,
        "split_assignment": split_assignment,
        "split_assignment_payload_sha256": _sha256_json(split_assignment),
        "subject_id": subject,
    }


def _manifest_directory(
    root: Path,
    *,
    train_rows: list[int] | None = None,
    val_dev_rows: list[int] | None = None,
    subject_overrides: dict[int, dict[str, object]] | None = None,
) -> Path:
    directory = root / "manifests"
    directory.mkdir(parents=True, exist_ok=True)
    train_rows = [0, 2, 4, 6] if train_rows is None else train_rows
    val_dev_rows = [1, 5] if val_dev_rows is None else val_dev_rows
    for subject in range(1, 11):
        options: dict[str, object] = {
            "train_rows": list(train_rows),
            "train_concept_ids": ["train-concept-a", "train-concept-b"],
            "val_dev_rows": list(val_dev_rows),
            "val_dev_concept_ids": ["dev-concept-a", "dev-concept-b"],
            "val_dev_ids": ["dev-stimulus-a", "dev-stimulus-b"],
        }
        if subject_overrides and subject in subject_overrides:
            options.update(subject_overrides[subject])
        manifest = _protocol_manifest(subject, **options)  # type: ignore[arg-type]
        _write_json(directory / f"sub-{subject:02d}_protocol.json", manifest)
    return directory


def _small_fixture(
    tmp_path: Path,
    *,
    array: np.ndarray | None = None,
    train_rows: list[int] | None = None,
    val_dev_rows: list[int] | None = None,
    subject_overrides: dict[int, dict[str, object]] | None = None,
    metadata_extra: dict[str, object] | None = None,
) -> tuple[Path, TypedArtifact, np.ndarray]:
    if array is None:
        array = np.arange(8 * 2 * 3, dtype=np.float16).reshape(8, 2, 3)
    artifact = _cache_artifact(
        tmp_path / "cache",
        array,
        metadata_extra=metadata_extra,
    )
    manifests = _manifest_directory(
        tmp_path,
        train_rows=train_rows,
        val_dev_rows=val_dev_rows,
        subject_overrides=subject_overrides,
    )
    return manifests, artifact, array


@pytest.mark.parametrize(
    "scopes",
    [
        (),
        ("train",),
        ("val-dev", "train"),
        ("train", "train"),
        ("train", "val-confirm"),
        ("train", "formal-test"),
        "train",
    ],
)
def test_forbidden_scope_sequences_fail_before_any_verifier_or_numpy_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scopes,
) -> None:
    calls = {"verify": 0, "numpy": 0}

    def forbidden_verify(*args, **kwargs):
        calls["verify"] += 1
        pytest.fail("scope rejection must precede typed-artifact verification")

    def forbidden_numpy(*args, **kwargs):
        calls["numpy"] += 1
        pytest.fail("scope rejection must precede np.load")

    monkeypatch.setattr(parity, "verify_typed_artifacts", forbidden_verify)
    monkeypatch.setattr(parity.np, "load", forbidden_numpy)
    artifact = TypedArtifact(
        payload_type="feature-cache",
        payload_path=tmp_path / "missing.npy",
        envelope_path=tmp_path / "missing.envelope.json",
    )

    with pytest.raises(ValueError, match="scopes"):
        parity.build_stage0_cache_parity(
            tmp_path / "missing-manifests",
            artifact,
            scopes=scopes,
            strict=False,
        )
    assert calls == {"verify": 0, "numpy": 0}


@pytest.mark.parametrize(
    "forbidden_component",
    [
        "test_images",
        "02d7e33b3fe8e5a571f8db232ca5fa86abb0c16981876ec84feae7ba64636f1a",
    ],
)
def test_lexical_manifest_denial_happens_before_numpy_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    forbidden_component: str,
) -> None:
    artifact = TypedArtifact(
        payload_type="feature-cache",
        payload_path=tmp_path / "missing.npy",
        envelope_path=tmp_path / "missing.envelope.json",
    )
    monkeypatch.setattr(
        parity.np,
        "load",
        lambda *args, **kwargs: pytest.fail("np.load must not run"),
    )

    with pytest.raises(ValueError, match="denied"):
        parity.build_stage0_cache_parity(
            tmp_path / forbidden_component,
            artifact,
            scopes=("train", "val-dev"),
            strict=False,
        )


def test_symlinked_manifest_component_fails_before_numpy_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    artifact = TypedArtifact(
        payload_type="feature-cache",
        payload_path=tmp_path / "missing.npy",
        envelope_path=tmp_path / "missing.envelope.json",
    )
    monkeypatch.setattr(
        parity.np,
        "load",
        lambda *args, **kwargs: pytest.fail("np.load must not run"),
    )

    with pytest.raises(ValueError, match="symlink"):
        parity.build_stage0_cache_parity(
            linked,
            artifact,
            scopes=("train", "val-dev"),
            strict=False,
        )


def test_small_scoped_views_are_exhaustively_direct_indexed(
    tmp_path: Path,
) -> None:
    manifests, artifact, array = _small_fixture(tmp_path)

    report = parity.build_stage0_cache_parity(
        manifests,
        artifact,
        scopes=("train", "val-dev"),
        strict=False,
        chunk_rows=2,
    )

    assert report["passed"] is True
    assert report["scope"] == "val-dev"
    assert report["scopes"] == ["train", "val-dev"]
    assert report["verification_kind"] == "exhaustive_scoped_view_direct_index"
    assert report["independent_cache_compared"] is False
    assert report["bit_identical_comparison_performed"] is False
    assert report["canonical_cache"]["shape"] == [8, 2, 3]
    assert report["canonical_cache"]["dtype"] == "float16"
    assert report["canonical_cache"]["finite"] is True
    train = report["scope_views"]["train"]
    val_dev = report["scope_views"]["val-dev"]
    assert train["row_count"] == 4
    assert train["feature_shape"] == [4, 2, 3]
    assert train["ordered_row_indices_sha256"] == parity.hash_ordered_rows(
        [0, 2, 4, 6]
    )
    assert train["direct_index_feature_bytes_sha256"] == parity.hash_feature_bytes(
        np.ascontiguousarray(array[[0, 2, 4, 6]])
    )
    assert val_dev["row_count"] == 2
    assert val_dev["feature_shape"] == [2, 2, 3]
    assert val_dev["direct_index_feature_bytes_sha256"] == parity.hash_feature_bytes(
        np.ascontiguousarray(array[[1, 5]])
    )
    assert len(report["subjects"]) == 10
    assert all(
        subject["selected_roles"] == ["train", "val-dev"]
        for subject in report["subjects"]
    )
    assert "val-confirm" not in json.dumps(report)


def test_cross_subject_row_or_id_mapping_must_match_exactly(
    tmp_path: Path,
) -> None:
    manifests, artifact, _ = _small_fixture(
        tmp_path,
        subject_overrides={
            10: {
                "train_rows": [0, 2, 4, 7],
                "train_concept_ids": [
                    "train-concept-a",
                    "train-concept-changed",
                ],
            }
        },
    )

    with pytest.raises(ValueError, match="subject mappings differ"):
        parity.build_stage0_cache_parity(
            manifests,
            artifact,
            scopes=("train", "val-dev"),
            strict=False,
        )


@pytest.mark.parametrize(
    ("train_rows", "message"),
    [
        ([0, 2, 2, 6], "duplicate"),
        ([0, 2, 4, 8], "out of range"),
        ([0, -1, 4, 6], "out of range|non-negative"),
    ],
)
def test_duplicate_or_out_of_range_rows_are_rejected(
    tmp_path: Path,
    train_rows: list[int],
    message: str,
) -> None:
    manifests, artifact, _ = _small_fixture(
        tmp_path,
        train_rows=train_rows,
    )

    with pytest.raises(ValueError, match=message):
        parity.build_stage0_cache_parity(
            manifests,
            artifact,
            scopes=("train", "val-dev"),
            strict=False,
        )


def test_train_and_val_dev_rows_must_be_disjoint(tmp_path: Path) -> None:
    manifests, artifact, _ = _small_fixture(
        tmp_path,
        val_dev_rows=[1, 2],
    )

    with pytest.raises(ValueError, match="disjoint"):
        parity.build_stage0_cache_parity(
            manifests,
            artifact,
            scopes=("train", "val-dev"),
            strict=False,
        )


def test_strict_mode_pins_real_counts_shape_dtype_and_contiguity() -> None:
    parity.require_strict_role_count("train", 12_540)
    parity.require_strict_role_count("val-dev", 200)
    with pytest.raises(ValueError, match="12540"):
        parity.require_strict_role_count("train", 12_539)
    with pytest.raises(ValueError, match="200"):
        parity.require_strict_role_count("val-dev", 199)

    header = SimpleNamespace(
        shape=(16_540, 5, 3_200),
        dtype=np.dtype(np.float16),
        flags=SimpleNamespace(c_contiguous=True),
    )
    parity.validate_cache_layout(header, strict=True)
    for bad_header in (
        SimpleNamespace(
            shape=(16_539, 5, 3_200),
            dtype=np.dtype(np.float16),
            flags=SimpleNamespace(c_contiguous=True),
        ),
        SimpleNamespace(
            shape=(16_540, 5, 3_200),
            dtype=np.dtype(np.float32),
            flags=SimpleNamespace(c_contiguous=True),
        ),
        SimpleNamespace(
            shape=(16_540, 5, 3_200),
            dtype=np.dtype(np.float16),
            flags=SimpleNamespace(c_contiguous=False),
        ),
    ):
        with pytest.raises(ValueError):
            parity.validate_cache_layout(bad_header, strict=True)


def test_layout_checks_apply_in_synthetic_mode(tmp_path: Path) -> None:
    wrong_dtype = np.zeros((8, 2, 3), dtype=np.float32)
    manifests, artifact, _ = _small_fixture(tmp_path, array=wrong_dtype)
    with pytest.raises(ValueError, match="float16"):
        parity.build_stage0_cache_parity(
            manifests,
            artifact,
            scopes=("train", "val-dev"),
            strict=False,
        )


def test_payload_hash_mismatch_rejects_before_numpy_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifests, artifact, _ = _small_fixture(tmp_path)
    with artifact.payload_path.open("ab") as handle:
        handle.write(b"tampered")
    load_count = 0

    def tracked_load(*args, **kwargs):
        nonlocal load_count
        load_count += 1
        pytest.fail("np.load must not run after access verification fails")

    monkeypatch.setattr(parity.np, "load", tracked_load)
    with pytest.raises(ValueError, match="payload SHA-256 mismatch"):
        parity.build_stage0_cache_parity(
            manifests,
            artifact,
            scopes=("train", "val-dev"),
            strict=False,
        )
    assert load_count == 0


def test_bound_test_metadata_rejects_before_numpy_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifests, artifact, _ = _small_fixture(
        tmp_path,
        metadata_extra={"split": "test"},
    )
    load_count = 0

    def tracked_load(*args, **kwargs):
        nonlocal load_count
        load_count += 1
        pytest.fail("np.load must not run after metadata denial")

    monkeypatch.setattr(parity.np, "load", tracked_load)
    with pytest.raises(PermissionError, match="test split"):
        parity.build_stage0_cache_parity(
            manifests,
            artifact,
            scopes=("train", "val-dev"),
            strict=False,
        )
    assert load_count == 0


def test_nonfinite_cache_value_fails_closed(tmp_path: Path) -> None:
    array = np.arange(8 * 2 * 3, dtype=np.float16).reshape(8, 2, 3)
    array[4, 1, 2] = np.nan
    manifests, artifact, _ = _small_fixture(tmp_path, array=array)

    with pytest.raises(ValueError, match="non-finite"):
        parity.build_stage0_cache_parity(
            manifests,
            artifact,
            scopes=("train", "val-dev"),
            strict=False,
        )


def test_feature_byte_hash_distinguishes_nan_payload_bits() -> None:
    left_bits = np.array([0x7E01], dtype=np.uint16)
    right_bits = np.array([0x7E02], dtype=np.uint16)
    left = left_bits.view(np.float16)
    same = left_bits.copy().view(np.float16)
    right = right_bits.view(np.float16)

    assert np.isnan(left[0]) and np.isnan(right[0])
    assert parity.hash_feature_bytes(left) == parity.hash_feature_bytes(same)
    assert parity.hash_feature_bytes(left) != parity.hash_feature_bytes(right)


def test_report_writer_is_canonical_exclusive_and_contains_no_val_confirm(
    tmp_path: Path,
) -> None:
    output = tmp_path / "reports" / "stage0.json"
    report = {
        "schema_version": 1,
        "payload_type": "samga_brain_rw.stage0_cache_scoped_view_report",
        "scope": "val-dev",
        "scopes": ["train", "val-dev"],
        "passed": True,
        "independent_cache_compared": False,
    }

    parity.write_stage0_cache_parity_report(output, report)

    assert output.read_bytes() == _canonical_json_bytes(report) + b"\n"
    assert b"val-confirm" not in output.read_bytes()
    with pytest.raises(FileExistsError):
        parity.write_stage0_cache_parity_report(output, report)


def test_output_test_images_component_is_rejected_before_directory_creation(
    tmp_path: Path,
) -> None:
    output = tmp_path / "test_images" / "report.json"

    with pytest.raises(ValueError, match="test_images"):
        parity.write_stage0_cache_parity_report(output, {"passed": True})
    assert not output.parent.exists()


def _load_cache_parity_cli():
    script = (
        Path(__file__).parents[1]
        / "scripts"
        / "check_stage0_cache_parity.py"
    )
    spec = importlib.util.spec_from_file_location(
        "check_stage0_cache_parity_test",
        script,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_requires_explicit_typed_cache_envelope_and_writes_exclusively(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _load_cache_parity_cli()
    captured: dict[str, object] = {}
    report = {
        "schema_version": 1,
        "payload_type": "samga_brain_rw.stage0_cache_scoped_view_report",
        "scope": "val-dev",
        "scopes": ["train", "val-dev"],
        "passed": True,
        "independent_cache_compared": False,
    }

    def fake_build(manifest_dir, canonical_cache, scopes, **kwargs):
        captured["manifest_dir"] = manifest_dir
        captured["canonical_cache"] = canonical_cache
        captured["scopes"] = tuple(scopes)
        return report

    monkeypatch.setattr(cli, "build_stage0_cache_parity", fake_build)
    output = tmp_path / "report.json"
    result = cli.main(
        [
            "--manifest-dir",
            os.fspath(tmp_path / "manifests"),
            "--canonical-cache",
            os.fspath(tmp_path / "features.npy"),
            "--canonical-cache-envelope",
            os.fspath(tmp_path / "features.envelope.json"),
            "--scopes",
            "train",
            "val-dev",
            "--output",
            os.fspath(output),
        ]
    )

    assert result == 0
    assert captured["manifest_dir"] == tmp_path / "manifests"
    typed = captured["canonical_cache"]
    assert isinstance(typed, TypedArtifact)
    assert typed.payload_type == "samga_brain_rw.train_cache"
    assert typed.envelope_path == tmp_path / "features.envelope.json"
    assert captured["scopes"] == ("train", "val-dev")
    assert output.read_bytes() == _canonical_json_bytes(report) + b"\n"
    with pytest.raises(FileExistsError):
        cli.main(
            [
                "--manifest-dir", os.fspath(tmp_path / "manifests"),
                "--canonical-cache", os.fspath(tmp_path / "features.npy"),
                "--canonical-cache-envelope",
                os.fspath(tmp_path / "features.envelope.json"),
                "--scopes", "train", "val-dev",
                "--output", os.fspath(output),
            ]
        )


def test_cli_rejects_forbidden_output_before_building_or_opening_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _load_cache_parity_cli()
    build_count = 0

    def forbidden_build(*args, **kwargs):
        nonlocal build_count
        build_count += 1
        pytest.fail("output path classification must precede input access")

    monkeypatch.setattr(cli, "build_stage0_cache_parity", forbidden_build)
    output = tmp_path / "test_images" / "report.json"

    with pytest.raises(ValueError, match="test_images"):
        cli.main(
            [
                "--manifest-dir",
                os.fspath(tmp_path / "missing-manifests"),
                "--canonical-cache",
                os.fspath(tmp_path / "missing.npy"),
                "--canonical-cache-envelope",
                os.fspath(tmp_path / "missing.envelope.json"),
                "--scopes",
                "train",
                "val-dev",
                "--output",
                os.fspath(output),
            ]
        )
    assert build_count == 0
    assert not output.parent.exists()
