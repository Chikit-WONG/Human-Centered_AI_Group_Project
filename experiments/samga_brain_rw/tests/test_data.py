from __future__ import annotations

import hashlib
import io
import json
import os
import pickle
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from samga_brain_rw import data as data_module
from samga_brain_rw.data import POSTERIOR_CHANNELS, ProtocolSubjectDataset
from samga_brain_rw.hashing import (
    canonical_json_bytes,
    sha256_json,
)
from samga_brain_rw.splits import build_subject_protocol_manifest, partition_concepts


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def _records() -> list[dict[str, object]]:
    return [
        {
            "concept_id": f"{concept:05d}_concept",
            "image_id": f"{concept:05d}_stimulus_{stimulus:02d}",
            "image_path": (
                f"training_images/{concept:05d}_concept/"
                f"{concept:05d}_stimulus_{stimulus:02d}.jpg"
            ),
            "row_index": row,
            "validation_query": False,
        }
        for row, (concept, stimulus) in enumerate(
            (concept, stimulus)
            for concept in range(1, 1_655)
            for stimulus in range(1, 11)
        )
    ]


@pytest.fixture(scope="module")
def sealed_subject(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    root = tmp_path_factory.mktemp("protocol-subject")
    source_pt = root / "sub-01" / "train.pt"
    source_pt.parent.mkdir(parents=True)
    source_pt.write_bytes(b"synthetic torch payload is monkeypatched")
    channels = ("Fz", *reversed(POSTERIOR_CHANNELS))
    records = _records()
    records_sha256 = hashlib.sha256(
        canonical_json_bytes(records)
    ).hexdigest()
    source = root / "sub-01_train.json"
    source_document = {
        "ch_names": list(channels),
        "eeg_dtype": "float32",
        "eeg_shape": [16_540, 4, len(channels), 250],
        "records": records,
        "records_sha256": records_sha256,
        "schema_version": 1,
        "source_pt": str(source_pt),
        "split": "train",
        "subject_id": 1,
        "validation_concepts": [],
        "validation_salt": "legacy-ignored",
    }
    _write_json(source, source_document)
    assignment = partition_concepts(records)
    protocol_document = build_subject_protocol_manifest(source, assignment)
    protocol = root / "sub-01_protocol.json"
    _write_json(protocol, protocol_document)

    cache = root / "features.npy"
    expected_cache_row = (
        np.arange(5 * 3_200, dtype=np.int64) % 1_024
    ).reshape(5, 3_200).astype(np.float16)
    cache_map = np.lib.format.open_memmap(
        cache,
        mode="w+",
        dtype=np.float16,
        shape=(16_540, 5, 3_200),
    )
    first_train_row = protocol_document["role_payloads"]["train"]["row_indices"][0]
    cache_map[first_train_row] = expected_cache_row
    cache_map.flush()
    del cache_map
    cache_metadata = {
        "cache_sha256": _file_sha256(cache),
        "complete": True,
        "dtype": "float16",
        "layer_ids": [20, 24, 28, 32, 36],
        "partial_rows": False,
        "records_sha256": records_sha256,
        "row_end": 16_540,
        "row_start": 0,
        "schema_version": 1,
        "shape": [16_540, 5, 3_200],
        "split": "train",
    }
    _write_json(cache.with_suffix(".npy.meta.json"), cache_metadata)
    base_eeg = np.arange(
        4 * len(channels) * 250,
        dtype=np.float32,
    ).reshape(4, len(channels), 250)
    return SimpleNamespace(
        base_eeg=base_eeg,
        cache=cache,
        expected_cache_row=expected_cache_row,
        channels=channels,
        eeg=np.broadcast_to(base_eeg, (16_540, *base_eeg.shape)),
        protocol=protocol,
        protocol_document=protocol_document,
        records=records,
        root=root,
        source=source,
        source_document=source_document,
        source_pt=source_pt,
    )


def _install_torch_load(
    monkeypatch: pytest.MonkeyPatch,
    fixture: SimpleNamespace,
    *,
    eeg: np.ndarray | None = None,
) -> None:
    value = fixture.eeg if eeg is None else eeg

    def fake_load(
        handle: object,
        *,
        map_location: str,
        weights_only: bool,
    ) -> dict[str, object]:
        assert callable(getattr(handle, "read", None))
        assert callable(getattr(handle, "fileno", None))
        assert map_location == "cpu"
        assert weights_only is True
        return {"ch_names": list(fixture.channels), "eeg": value}

    monkeypatch.setattr(torch, "load", fake_load)


def test_source_payload_digest_is_verified_before_weights_only_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = tmp_path / "sub-01" / "train.pt"
    payload.parent.mkdir()
    payload.write_bytes(b"not deserialized when the digest is wrong")
    touched = False

    def forbidden_load(*_args: object, **_kwargs: object) -> object:
        nonlocal touched
        touched = True
        raise AssertionError("torch.load must follow digest verification")

    monkeypatch.setattr(torch, "load", forbidden_load)
    with pytest.raises(ValueError, match="SHA-256"):
        data_module._load_torch_payload(
            payload,
            expected_sha256="0" * 64,
        )
    assert touched is False


def test_restricted_protocol5_unpickler_allows_only_pinned_numpy_globals(
    tmp_path: Path,
) -> None:
    unpickler = data_module._RestrictedNumpyUnpickler(
        io.BytesIO()
    )
    frombuffer = unpickler.find_class(
        "numpy.core.numeric",
        "_frombuffer",
    )
    assert callable(frombuffer)
    assert np.array_equal(
        frombuffer(
            b"\x01\x02",
            np.dtype("uint8"),
            (2,),
            "C",
        ),
        np.asarray([1, 2], dtype=np.uint8),
    )
    assert unpickler.find_class(
        "numpy",
        "dtype",
    ) is np.dtype
    with pytest.raises(
        pickle.UnpicklingError,
        match="global",
    ):
        unpickler.find_class(
            "numpy._core.numeric",
            "_frombuffer",
        )

    sentinel = tmp_path / "arbitrary-code-ran"

    class Exploit:
        def __reduce__(self) -> tuple[object, tuple[str]]:
            return (
                os.system,
                (f"touch {sentinel}",),
            )

    with pytest.raises(
        pickle.UnpicklingError,
        match="global",
    ):
        data_module._restricted_numpy_unpickle(
            io.BytesIO(
                pickle.dumps(Exploit(), protocol=5)
            )
        )
    assert not sentinel.exists()


def test_pinned_protocol5_archive_never_falls_back_to_unsafe_torch_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = tmp_path / "sub-01" / "train.pt"
    payload.parent.mkdir()
    sentinel = tmp_path / "archive-code-ran"

    class Exploit:
        def __reduce__(self) -> tuple[object, tuple[str]]:
            return (
                os.system,
                (f"touch {sentinel}",),
            )

    with zipfile.ZipFile(
        payload,
        mode="w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
    ) as archive:
        archive.writestr(
            "train/data.pkl",
            pickle.dumps(Exploit(), protocol=5),
        )
        archive.writestr("train/version", b"3\n")
    touched = False

    def forbidden_torch_load(
        *_args: object,
        **_kwargs: object,
    ) -> object:
        nonlocal touched
        touched = True
        raise AssertionError("pinned NumPy archive must use restricted pickle")

    monkeypatch.setattr(torch, "load", forbidden_torch_load)
    with pytest.raises(ValueError, match="safely"):
        data_module._load_torch_payload(
            payload,
            expected_sha256=_file_sha256(payload),
        )
    assert touched is False
    assert not sentinel.exists()


def test_source_manifest_identity_binds_actual_train_payload_bytes(
    sealed_subject: SimpleNamespace,
) -> None:
    identity = data_module.inspect_source_payload_identity(
        sealed_subject.source,
        expected_manifest_sha256=_file_sha256(
            sealed_subject.source
        ),
        subject=1,
    )
    assert identity.path == sealed_subject.source_pt
    assert identity.byte_count == sealed_subject.source_pt.stat().st_size
    assert identity.sha256 == _file_sha256(
        sealed_subject.source_pt
    )
    with pytest.raises(ValueError, match="manifest SHA-256"):
        data_module.inspect_source_payload_identity(
            sealed_subject.source,
            expected_manifest_sha256="0" * 64,
            subject=1,
        )


@pytest.mark.parametrize(
    "component",
    ("test", "formal", "formal-test", "formal_test", "val-confirm", "val_confirm"),
)
def test_source_payload_rejects_every_sealed_path_component(
    tmp_path: Path,
    component: str,
) -> None:
    with pytest.raises(PermissionError, match="sealed"):
        data_module._preflight_development_path(
            tmp_path / component / "train.pt",
            "source train EEG",
        )


def test_protocol_roles_expose_exact_rows_ids_and_shared_cache_mapping(
    sealed_subject: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_torch_load(monkeypatch, sealed_subject)
    train = ProtocolSubjectDataset(
        sealed_subject.protocol,
        "train",
        42,
        POSTERIOR_CHANNELS,
        sealed_subject.cache,
        0.0,
    )
    val_dev = ProtocolSubjectDataset(
        sealed_subject.protocol,
        "val-dev",
        42,
        POSTERIOR_CHANNELS,
        sealed_subject.cache,
        0.3,
    )
    train_role = sealed_subject.protocol_document["role_payloads"]["train"]
    val_role = sealed_subject.protocol_document["role_payloads"]["val-dev"]
    assert (len(train), len(train.concept_ids), train.stimuli_per_concept) == (
        12_540,
        1_254,
        10,
    )
    assert (len(val_dev), len(val_dev.concept_ids), val_dev.stimuli_per_concept) == (
        200,
        200,
        1,
    )
    assert train.row_indices == tuple(train_role["row_indices"])
    assert train.ordered_ids == tuple(train_role["ordered_ids"])
    assert val_dev.row_indices == tuple(val_role["row_indices"])
    assert val_dev.ordered_ids == tuple(val_role["ordered_ids"])
    assert val_dev.query_ids == val_dev.gallery_ids == val_dev.ordered_ids
    assert set(train.concept_ids).isdisjoint(val_dev.concept_ids)
    first = train[0]
    row = train.row_indices[0]
    source_record = sealed_subject.records[row]
    assert first["row_index"] == row
    assert first["concept_id"] == source_record["concept_id"]
    assert first["image_id"] == source_record["image_id"]
    assert first["subject_id"] == 1
    assert torch.equal(
        first["layer_features"],
        torch.from_numpy(sealed_subject.expected_cache_row.astype(np.float32)),
    )


@pytest.mark.parametrize("field", ["concept_ids", "row_indices"])
def test_train_and_val_dev_cannot_overlap(
    sealed_subject: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    _install_torch_load(monkeypatch, sealed_subject)
    protocol = json.loads(sealed_subject.protocol.read_bytes())
    train = protocol["role_payloads"]["train"]
    val_dev = protocol["role_payloads"]["val-dev"]
    val_dev[field][0] = train[field][0]
    protocol["role_artifacts"]["val-dev"]["role_payload_sha256"] = sha256_json(
        val_dev
    )
    tampered_path = sealed_subject.root / field / "sub-01_protocol.json"
    _write_json(tampered_path, protocol)
    with pytest.raises(ValueError, match="overlap"):
        ProtocolSubjectDataset(
            tampered_path,
            "train",
            42,
            POSTERIOR_CHANNELS,
            None,
            0.0,
        )


def _moving_average(signal: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    time_len = signal.shape[-1]
    left = np.maximum(0, np.arange(time_len) - kernel_size // 2)
    right = np.minimum(time_len, np.arange(time_len) + kernel_size // 2 + 1)
    cumulative = np.pad(
        np.cumsum(signal, axis=-1, dtype=np.float32),
        ((0, 0), (1, 0)),
    )
    return (cumulative[:, right] - cumulative[:, left]) / (
        right - left
    )[None, :]


def _expected_smoothed(
    eeg: np.ndarray,
    *,
    seed: int,
    row: int,
    probability: float,
) -> np.ndarray:
    material = (
        b"SAMGA-PROTOCOL-SMOOTH-v1\0"
        + str(seed).encode("ascii")
        + b"\0"
        + str(row).encode("ascii")
    )
    rng_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    mask = np.random.default_rng(rng_seed).random(eeg.shape[0]) < probability
    result = eeg.copy()
    result[mask] = _moving_average(result[mask])
    return result


def test_four_trial_mean_channel_order_and_smoothing_are_deterministic(
    sealed_subject: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_torch_load(monkeypatch, sealed_subject)
    plain = ProtocolSubjectDataset(
        sealed_subject.protocol, "train", 77, POSTERIOR_CHANNELS, None, 0.0
    )
    smoothed = ProtocolSubjectDataset(
        sealed_subject.protocol, "train", 77, POSTERIOR_CHANNELS, None, 0.5
    )
    repeated = ProtocolSubjectDataset(
        sealed_subject.protocol, "train", 77, POSTERIOR_CHANNELS, None, 0.5
    )
    val_dev = ProtocolSubjectDataset(
        sealed_subject.protocol, "val-dev", 77, POSTERIOR_CHANNELS, None, 1.0
    )
    channel_indices = [
        sealed_subject.channels.index(channel) for channel in POSTERIOR_CHANNELS
    ]
    expected_plain = sealed_subject.base_eeg.mean(
        axis=0,
        dtype=np.float32,
    )[channel_indices]
    expected_smoothed = _expected_smoothed(
        expected_plain,
        seed=77,
        row=plain.row_indices[0],
        probability=0.5,
    )
    assert np.array_equal(plain[0]["eeg"].numpy(), expected_plain)
    assert np.array_equal(smoothed[0]["eeg"].numpy(), expected_smoothed)
    assert torch.equal(smoothed[0]["eeg"], smoothed[0]["eeg"])
    assert torch.equal(smoothed[0]["eeg"], repeated[0]["eeg"])
    assert np.array_equal(val_dev[0]["eeg"].numpy(), expected_plain)


@pytest.mark.parametrize(
    "scope",
    ["val-confirm", "formal-refit", "formal-input", "formal-test", "test"],
)
def test_sealed_scopes_reject_before_eeg_or_cache_load(
    sealed_subject: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    scope: str,
) -> None:
    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: pytest.fail("must reject before torch.load"),
    )
    with pytest.raises((PermissionError, ValueError), match="scope|train|val-dev"):
        ProtocolSubjectDataset(
            sealed_subject.protocol,
            scope,
            42,
            POSTERIOR_CHANNELS,
            sealed_subject.cache,
            0.3,
        )


def test_subject_source_and_cache_hash_bindings_are_enforced(
    sealed_subject: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_torch_load(monkeypatch, sealed_subject)
    mismatch = sealed_subject.root / "sub-02_protocol.json"
    mismatch.write_bytes(sealed_subject.protocol.read_bytes())
    with pytest.raises(ValueError, match="subject"):
        ProtocolSubjectDataset(
            mismatch, "train", 42, POSTERIOR_CHANNELS, None, 0.0
        )
    tampered = json.loads(sealed_subject.source.read_bytes())
    tampered["validation_salt"] = "tampered"
    _write_json(sealed_subject.source, tampered)
    try:
        with pytest.raises(ValueError, match="source manifest.*SHA-256"):
            ProtocolSubjectDataset(
                sealed_subject.protocol, "train", 42, POSTERIOR_CHANNELS, None, 0.0
            )
    finally:
        _write_json(sealed_subject.source, sealed_subject.source_document)
    metadata_path = sealed_subject.cache.with_suffix(".npy.meta.json")
    metadata = json.loads(metadata_path.read_bytes())
    metadata["records_sha256"] = "0" * 64
    _write_json(metadata_path, metadata)
    try:
        with pytest.raises(ValueError, match="cache.*record"):
            ProtocolSubjectDataset(
                sealed_subject.protocol,
                "train",
                42,
                POSTERIOR_CHANNELS,
                sealed_subject.cache,
                0.0,
            )
    finally:
        metadata["records_sha256"] = sealed_subject.source_document[
            "records_sha256"
        ]
        _write_json(metadata_path, metadata)


def test_requires_exact_posterior_channels_and_four_trials(
    sealed_subject: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_torch_load(monkeypatch, sealed_subject)
    with pytest.raises(ValueError, match="17|posterior|channels"):
        ProtocolSubjectDataset(
            sealed_subject.protocol,
            "train",
            42,
            POSTERIOR_CHANNELS[:-1],
            None,
            0.0,
        )
    three_trials = np.broadcast_to(
        sealed_subject.base_eeg[:3],
        (16_540, 3, len(sealed_subject.channels), 250),
    )
    _install_torch_load(monkeypatch, sealed_subject, eeg=three_trials)
    with pytest.raises(ValueError, match="four trials"):
        ProtocolSubjectDataset(
            sealed_subject.protocol,
            "train",
            42,
            POSTERIOR_CHANNELS,
            None,
            0.0,
        )
