from __future__ import annotations

import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from samga_brain_rw.hashing import (
    canonical_json_bytes,
    concept_digest,
    ordered_ids_sha256,
    stimulus_digest,
)
from samga_brain_rw.splits import (
    build_subject_protocol_manifest,
    partition_concepts,
)
import scripts.build_protocol_manifests as manifest_cli
from scripts.build_protocol_manifests import main


DESCRIPTOR_KEYS = {
    "schema_version",
    "payload_type",
    "scope",
    "source_records_sha256",
    "ordered_ids_sha256",
    "role_payload_sha256",
    "provenance_sha256",
}

SOURCE_RAW_SHA256 = (
    "42fd7316314eb02d69ee2234d4d8430afcfcc2a5f6834e9c7be64f38eccdbc85",
    "1d6275829da9f423c090d48350dfe106ac27759225265b9a3c796ddb4f77d0a0",
    "123ba9dfdd983173fe6b5f6a739c515ca2b12ab101898549756a6d9a8462086e",
    "eb133c98c761de61bb87154dd140df2f82047512fbd8e170a50e7cfaf005e7e5",
    "f278c3a6efafeffc278b871ae111792fbb0cf41ee05cd11e6e24d3497afd7b6b",
    "a88bdf485d0d05548c45ffda0b9fdbd9aad69207bcc88b258ea860da0d7244e8",
    "12c6629989cf6b0fdf0aff963c0f690f21a2e46978b40aed54a39e3230d8d52b",
    "703f9e305822da747c4fa5ee61c277578e5e7d3da42947bf2b17742909e3425d",
    "6d30eca14797961805d3d113de2cbabbc448f1f3f83abb48f51c3565e440377b",
    "abde70e302375e9ca3d94c5d2ce593e4be699fe817e17bdfc255112d8523483e",
)
RECORDS_SHA256 = "f59500f36e273f66fce5c2019670b076d75d538feccf296c7d7ed75f19ae3fac"
ORDERED_ID_SHA256 = {
    "train_concept_ids": "ae5aeda4101f8740ebcb63464ca9cf5e126c81b2f124f5caa8f7b57b7a9fad24",
    "val_dev_concept_ids": "c8c00ff2b15d98cdcb74d533037d52435bc12e09797151e46b52b86aedba1d15",
    "val_dev_query_ids": "512c222859a31b753ee31c5d6a1ddd1c81bb06e2dd5784d325f4480967162314",
    "val_confirm_concept_ids": "27cfd5b3d0b46f3e8303953ede106e0716f410aee2b5756dd6fb5ad0324908bb",
    "val_confirm_query_ids": "7a77db6d8d214e4a8192472dc7a760b58763d49c8c9f88fcb55c97bb124ec9fd",
}


@pytest.fixture(scope="session")
def real_train_manifests() -> tuple[tuple[Path, bytes, dict[str, object]], ...]:
    repository_root = Path(__file__).resolve().parents[3]
    source_dir = repository_root / "artifacts" / "samga_lora" / "manifests"
    loaded = []
    for subject in range(1, 11):
        path = source_dir / f"sub-{subject:02d}_train.json"
        raw = path.read_bytes()
        payload = json.loads(raw)
        assert isinstance(payload, dict)
        loaded.append((path, raw, payload))
    return tuple(loaded)


def _records(concept_count: int = 1_654) -> list[dict[str, object]]:
    return [
        {
            "concept_id": f"{concept:05d}_concept",
            "image_id": f"{concept:05d}_stimulus_{stimulus:02d}",
            "image_path": f"images/{concept:05d}/{stimulus:02d}.jpg",
            "row_index": row_index,
            "validation_query": False,
        }
        for row_index, (concept, stimulus) in enumerate(
            (concept, stimulus)
            for concept in range(1, concept_count + 1)
            for stimulus in range(10)
        )
    ]


def _write_manifest(
    path: Path,
    records: list[dict[str, object]],
    *,
    subject_id: str = "sub-01",
    split: str = "train",
) -> None:
    records_sha256 = hashlib.sha256(canonical_json_bytes(records)).hexdigest()
    payload = {
        "ch_names": ["Cz"],
        "eeg_dtype": "float32",
        "eeg_shape": [len(records), 1, 1],
        "records": records,
        "records_sha256": records_sha256,
        "schema_version": 1,
        "source_pt": f"{subject_id}.pt",
        "split": split,
        "subject_id": subject_id,
        "validation_concepts": [],
        "validation_salt": "legacy-samga-lora-val-v1",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def test_hashing_golden_vectors_and_ordered_list_bytes() -> None:
    assert concept_digest("00001_aardvark") == (
        "2d23e6dd6dabce52f966c3bf7e858199d60d4506627f4fea4433507b7fab2015"
    )
    assert concept_digest("00001_aardvark") == hashlib.sha256(
        "AIAA3800-SAMGA-SPLIT-v1\n00001_aardvark".encode("utf-8")
    ).hexdigest()
    assert stimulus_digest(
        "val-dev", "概念", "刺激"
    ) == hashlib.sha256(
        "AIAA3800-SAMGA-STIM-v1\nval-dev\n概念\n刺激".encode("utf-8")
    ).hexdigest()
    assert ordered_ids_sha256(["α", "β"]) == hashlib.sha256(
        "α\nβ".encode("utf-8")
    ).hexdigest()


def test_real_train_only_golden_vectors_are_shared_by_all_subjects(
    real_train_manifests: tuple[tuple[Path, bytes, dict[str, object]], ...],
) -> None:
    for (_, raw, payload), expected_raw_sha256 in zip(
        real_train_manifests, SOURCE_RAW_SHA256, strict=True
    ):
        assert hashlib.sha256(raw).hexdigest() == expected_raw_sha256
        assert payload["split"] == "train"
        assert payload["records_sha256"] == RECORDS_SHA256
        assert len(payload["records"]) == 16_540
        assert hashlib.sha256(
            canonical_json_bytes(payload["records"])
        ).hexdigest() == RECORDS_SHA256

    first_records = real_train_manifests[0][2]["records"]
    assignment = partition_concepts(first_records)
    registry = assignment.to_payload()
    concepts = registry["concepts"]
    assert concepts[199]["concept_id"] == "01611_wheel"
    assert concepts[199]["split_rank"] == 200
    assert concepts[200]["concept_id"] == "00950_octopus"
    assert concepts[200]["split_rank"] == 201
    assert concepts[399]["concept_id"] == "00657_handlebar"
    assert concepts[399]["split_rank"] == 400
    assert concepts[400]["concept_id"] == "00242_card"
    assert concepts[400]["split_rank"] == 401
    for key, expected in ORDERED_ID_SHA256.items():
        assert registry["ordered_id_sha256"][key] == expected

    shared = []
    stimulus_hashes = []
    for path, _, _ in real_train_manifests:
        subject = build_subject_protocol_manifest(path, assignment)
        shared.append(canonical_json_bytes(subject["split_assignment"]))
        stimulus_hashes.append(
            (
                subject["split_assignment"]["ordered_id_sha256"][
                    "val_dev_query_ids"
                ],
                subject["split_assignment"]["ordered_id_sha256"][
                    "val_confirm_query_ids"
                ],
            )
        )
    assert len(set(shared)) == 1
    assert len(set(stimulus_hashes)) == 1


def test_legacy_roles_and_selections_do_not_drive_new_assignment(
    tmp_path: Path,
) -> None:
    records = _records()
    legacy_mutation = json.loads(json.dumps(records))
    for record in legacy_mutation[::10]:
        record["validation_query"] = True
    original = partition_concepts(records).to_payload()
    mutated = partition_concepts(legacy_mutation).to_payload()
    assert mutated["concepts"] == original["concepts"]
    assert mutated["ordered_ids"] == original["ordered_ids"]
    assert mutated["ordered_id_sha256"] == original["ordered_id_sha256"]

    source = tmp_path / "sub-01_train.json"
    _write_manifest(source, records)
    first = build_subject_protocol_manifest(source, partition_concepts(records))
    payload = json.loads(source.read_bytes())
    payload["validation_concepts"] = ["legacy-must-be-ignored"]
    source.write_bytes(canonical_json_bytes(payload) + b"\n")
    second = build_subject_protocol_manifest(source, partition_concepts(records))
    assert first["split_assignment"] == second["split_assignment"]
    assert first["role_payloads"] == second["role_payloads"]


def test_partition_has_exact_roles_boundaries_and_unique_queries() -> None:
    records = _records()
    assignment = partition_concepts(records)
    payload = assignment.to_payload()
    concepts = payload["concepts"]
    expected_ids = sorted(
        {record["concept_id"] for record in records},
        key=lambda concept_id: (concept_digest(concept_id), concept_id),
    )

    assert [entry["concept_id"] for entry in concepts] == expected_ids
    assert [(concepts[index]["split_rank"], concepts[index]["split_role"]) for index in (199, 200, 399, 400)] == [
        (200, "val-dev"),
        (201, "val-confirm"),
        (400, "val-confirm"),
        (401, "train"),
    ]
    assert all(len(entry["stimulus_ids"]) == 10 for entry in concepts)
    validation = [entry for entry in concepts if entry["split_role"] != "train"]
    assert len(validation) == 400
    assert len({entry["selected_stimulus_id"] for entry in validation}) == 400
    assert len(payload["ordered_ids"]["train_concept_ids"]) == 1_254
    assert len(payload["ordered_ids"]["val_dev_query_ids"]) == 200
    assert len(payload["ordered_ids"]["val_confirm_query_ids"]) == 200


@pytest.mark.parametrize("concept_count", [1_653, 1_655])
def test_partition_rejects_wrong_concept_count(concept_count: int) -> None:
    with pytest.raises(ValueError, match="exactly 1654 concepts"):
        partition_concepts(_records(concept_count))


def test_partition_rejects_duplicate_pair_non_ten_stimuli_and_row_gap() -> None:
    duplicate = _records()
    duplicate[-1]["concept_id"] = duplicate[-2]["concept_id"]
    duplicate[-1]["image_id"] = duplicate[-2]["image_id"]
    with pytest.raises(ValueError, match="duplicate"):
        partition_concepts(duplicate)

    non_ten = _records()
    non_ten[-1]["concept_id"] = non_ten[-11]["concept_id"]
    non_ten[-1]["image_id"] = "replacement_unique_stimulus"
    with pytest.raises(ValueError, match="exactly 10 stimuli"):
        partition_concepts(non_ten)

    row_gap = _records()
    row_gap[-1]["row_index"] = len(row_gap)
    with pytest.raises(ValueError, match="contiguous"):
        partition_concepts(row_gap)


def test_subject_sidecar_has_recomputable_typed_role_envelopes(
    tmp_path: Path,
) -> None:
    records = _records()
    source = tmp_path / "sub-01_train.json"
    _write_manifest(source, records)
    manifest = build_subject_protocol_manifest(source, partition_concepts(records))

    assert manifest["schema_version"] == 1
    assert manifest["payload_type"] == "samga_brain_rw.subject_protocol_manifest"
    for role in ("train", "val-dev", "val-confirm"):
        descriptor = manifest["role_artifacts"][role]
        assert set(descriptor) == DESCRIPTOR_KEYS
        role_payload = manifest["role_payloads"][role]
        assert descriptor["payload_type"] == (
            role_payload["payload_type"]
        ) == "samga_brain_rw.role_payload"
        assert descriptor["source_records_sha256"] == manifest["records_sha256"]
        assert descriptor["ordered_ids_sha256"] == ordered_ids_sha256(
            role_payload["ordered_ids"]
)
        assert descriptor["role_payload_sha256"] == hashlib.sha256(
            canonical_json_bytes(role_payload)
        ).hexdigest()
        provenance = {
            "protocol_config_sha256": manifest["protocol_config_sha256"],
            "source_manifest_sha256": manifest["source_manifest_sha256"],
        }
        assert descriptor["provenance_sha256"] == hashlib.sha256(
            canonical_json_bytes(provenance)
        ).hexdigest()
    assert manifest["role_payloads"]["train"]["concept_count"] == 1_254
    assert manifest["role_payloads"]["train"]["row_count"] == 12_540
    assert manifest["role_payloads"]["val-dev"]["concept_count"] == 200
    assert manifest["role_payloads"]["val-dev"]["row_count"] == 200
    assert manifest["role_payloads"]["val-confirm"]["concept_count"] == 200
    assert manifest["role_payloads"]["val-confirm"]["row_count"] == 200


def test_subject_manifest_rejects_test_split_and_record_order_mismatch(
    tmp_path: Path,
) -> None:
    records = _records()
    assignment = partition_concepts(records)
    test_source = tmp_path / "sub-01_train.json"
    _write_manifest(test_source, records, split="test")
    with pytest.raises(ValueError, match="split must be train"):
        build_subject_protocol_manifest(test_source, assignment)

    records[0], records[1] = records[1], records[0]
    records[0]["row_index"] = 0
    records[1]["row_index"] = 1
    _write_manifest(test_source, records)
    with pytest.raises(ValueError, match="record order|records_sha256"):
        build_subject_protocol_manifest(test_source, assignment)


def test_subject_manifest_rejects_identity_hash_and_json_schema_failures(
    tmp_path: Path,
) -> None:
    records = _records()
    assignment = partition_concepts(records)
    source = tmp_path / "sub-01_train.json"

    _write_manifest(source, records, subject_id="sub-02")
    with pytest.raises(ValueError, match="subject_id.*filename"):
        build_subject_protocol_manifest(source, assignment)

    _write_manifest(source, records)
    payload = json.loads(source.read_bytes())
    payload["records_sha256"] = "0" * 64
    source.write_bytes(canonical_json_bytes(payload) + b"\n")
    with pytest.raises(ValueError, match="records_sha256"):
        build_subject_protocol_manifest(source, assignment)

    source.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        build_subject_protocol_manifest(source, assignment)

    source.write_text(
        '{"schema_version":1,"schema_version":1}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        build_subject_protocol_manifest(source, assignment)


def test_cli_uses_explicit_names_and_directory_atomic_idempotence(
    tmp_path: Path,
    configs_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = _records()
    source_dir = tmp_path / "source"
    for subject in range(1, 11):
        _write_manifest(
            source_dir / f"sub-{subject:02d}_train.json",
            records,
            subject_id=f"sub-{subject:02d}",
        )

    def forbid_scan(*args: object, **kwargs: object) -> object:
        raise AssertionError("Path directory scans are forbidden")

    monkeypatch.setattr(Path, "glob", forbid_scan)
    monkeypatch.setattr(Path, "rglob", forbid_scan)
    monkeypatch.setattr(Path, "iterdir", forbid_scan)
    output_dir = tmp_path / "output"
    arguments = [
        "--protocol",
        str(configs_dir / "protocol_v1.json"),
        "--source-manifest-dir",
        str(source_dir),
        "--output-dir",
        str(output_dir),
    ]
    output_dir.mkdir()
    empty_inode = output_dir.stat().st_ino
    with pytest.raises(FileExistsError, match="partial|complete"):
        main(arguments)
    assert output_dir.stat().st_ino == empty_inode
    assert list(os.listdir(output_dir)) == []
    output_dir.rmdir()
    output_dir.mkdir()
    sentinel = output_dir / "preexisting"
    sentinel.write_bytes(b"do-not-replace")
    original_inode = output_dir.stat().st_ino
    with pytest.raises(FileExistsError, match="partial|complete"):
        main(arguments)
    assert output_dir.stat().st_ino == original_inode
    assert sentinel.read_bytes() == b"do-not-replace"
    sentinel.unlink()
    output_dir.rmdir()
    assert main(arguments) == 0
    expected_names = {
        "split_assignment.json",
        "manifest_summary.json",
        *(f"sub-{subject:02d}_protocol.json" for subject in range(1, 11)),
    }
    assert set(os.listdir(output_dir)) == expected_names
    first = {
        name: (output_dir / name).read_bytes()
        for name in sorted(expected_names)
    }
    assignment_payload = json.loads(first["split_assignment.json"])
    summary = json.loads(first["manifest_summary.json"])
    assignment_payload_sha256 = hashlib.sha256(
        canonical_json_bytes(assignment_payload)
    ).hexdigest()
    assignment_file_sha256 = hashlib.sha256(
        first["split_assignment.json"]
    ).hexdigest()
    assert summary["split_assignment_payload_sha256"] == (
        assignment_payload_sha256
    )
    assert summary["split_assignment_file_sha256"] == assignment_file_sha256
    assert "split_assignment_sha256" not in summary
    for subject in range(1, 11):
        subject_payload = json.loads(
            first[f"sub-{subject:02d}_protocol.json"]
        )
        assert subject_payload[
            "split_assignment_payload_sha256"
        ] == assignment_payload_sha256
        assert "split_assignment_sha256" not in subject_payload
    assert main(arguments) == 0
    assert {
        name: (output_dir / name).read_bytes()
        for name in sorted(expected_names)
    } == first

    (output_dir / "manifest_summary.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="conflict|byte-identical"):
        main(arguments)
    (output_dir / "manifest_summary.json").write_bytes(
        first["manifest_summary.json"]
    )
    (output_dir / "sub-10_protocol.json").unlink()
    with pytest.raises(FileExistsError, match="partial|complete"):
        main(arguments)


def test_atomic_publish_race_never_replaces_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "race-output"
    outputs = {
        "only.json": b'{"expected":true}\n',
        "manifest_summary.json": b'{"complete":true}\n',
    }
    captured: dict[str, int] = {}

    def inject_racing_destination(
        destination: Path,
        rendered: dict[str, bytes],
    ) -> bool:
        assert destination == output_dir
        assert rendered == outputs
        if not destination.exists():
            destination.mkdir()
            (destination / "racer").write_bytes(b"preserve-me")
            captured["inode"] = destination.stat().st_ino
        return False

    monkeypatch.setattr(
        manifest_cli,
        "_reuse_existing_identical",
        inject_racing_destination,
    )
    with pytest.raises(FileExistsError, match="output conflict|partial"):
        manifest_cli._publish_atomic_directory(output_dir, outputs)
    assert output_dir.stat().st_ino == captured["inode"]
    assert (output_dir / "racer").read_bytes() == b"preserve-me"
    assert not (output_dir / "only.json").exists()
    assert not any(
        name.startswith(f".{output_dir.name}.staging-")
        for name in os.listdir(output_dir.parent)
    )


def test_publish_writes_summary_last_and_cleans_owned_partial_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "publish-failure"
    outputs = {
        "split_assignment.json": b"{}\n",
        **{
            f"sub-{subject:02d}_protocol.json": b"{}\n"
            for subject in range(1, 11)
        },
        "manifest_summary.json": b'{"complete":true}\n',
    }
    attempted: list[str] = []
    real_open = Path.open

    def fail_before_summary(
        path: Path,
        mode: str = "r",
        *args: object,
        **kwargs: object,
    ) -> object:
        attempted.append(path.name)
        if path.name == "sub-10_protocol.json":
            raise OSError("injected pre-summary failure")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_before_summary)
    with pytest.raises(OSError, match="injected pre-summary failure"):
        manifest_cli._publish_atomic_directory(output_dir, outputs)
    assert "manifest_summary.json" not in attempted
    assert not output_dir.exists()


def test_two_cooperating_publishers_have_one_destination_mkdir_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "concurrent-output"
    outputs = {
        "split_assignment.json": b"{}\n",
        **{
            f"sub-{subject:02d}_protocol.json": b"{}\n"
            for subject in range(1, 11)
        },
        "manifest_summary.json": b'{"complete":true}\n',
    }
    barrier = threading.Barrier(2)
    local = threading.local()
    real_reuse = manifest_cli._reuse_existing_identical
    real_mkdir = os.mkdir
    count_lock = threading.Lock()
    successful_destination_mkdirs = 0

    def synchronized_first_reuse(
        destination: Path,
        rendered: dict[str, bytes],
    ) -> bool:
        calls = getattr(local, "reuse_calls", 0)
        local.reuse_calls = calls + 1
        result = real_reuse(destination, rendered)
        if calls == 0:
            barrier.wait(timeout=10)
        return result

    def counted_mkdir(
        path: object,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> None:
        nonlocal successful_destination_mkdirs
        real_mkdir(path, mode, dir_fd=dir_fd)
        if Path(os.fsdecode(os.fspath(path))) == output_dir:
            with count_lock:
                successful_destination_mkdirs += 1

    monkeypatch.setattr(
        manifest_cli,
        "_reuse_existing_identical",
        synchronized_first_reuse,
    )
    monkeypatch.setattr(manifest_cli.os, "mkdir", counted_mkdir)

    def publish() -> str:
        try:
            manifest_cli._publish_atomic_directory(output_dir, outputs)
        except FileExistsError:
            return "conflict"
        return "ok"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: publish(), range(2)))
    assert "ok" in results
    assert set(results) <= {"ok", "conflict"}
    assert successful_destination_mkdirs == 1
    assert set(os.listdir(output_dir)) == set(outputs)
    assert (output_dir / "manifest_summary.json").read_bytes() == outputs[
        "manifest_summary.json"
    ]


def test_tracked_compact_registry_copies_match_runtime_generation() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    runtime = (
        repository_root / "artifacts" / "samga_brain_rw"
        / "protocol" / "manifests"
    )
    tracked = (
        repository_root / "experiments" / "samga_brain_rw"
        / "registries" / "protocol_v1"
    )
    for name in ("split_assignment.json", "manifest_summary.json"):
        assert (tracked / name).read_bytes() == (runtime / name).read_bytes()


def test_cross_subject_record_mismatch_leaves_no_output_or_staging(
    tmp_path: Path,
    configs_dir: Path,
) -> None:
    records = _records()
    source_dir = tmp_path / "source"
    for subject in range(1, 11):
        subject_records = records
        if subject == 10:
            subject_records = list(records)
            subject_records[0], subject_records[1] = (
                subject_records[1],
                subject_records[0],
            )
            subject_records[0]["row_index"] = 0
            subject_records[1]["row_index"] = 1
        _write_manifest(
            source_dir / f"sub-{subject:02d}_train.json",
            subject_records,
            subject_id=f"sub-{subject:02d}",
        )
    output_dir = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="record order|records_sha256"):
        main(
            [
                "--protocol",
                str(configs_dir / "protocol_v1.json"),
                "--source-manifest-dir",
                str(source_dir),
                "--output-dir",
                str(output_dir),
            ]
        )
    assert not output_dir.exists()
    assert not any(
        name.startswith(f".{output_dir.name}.staging-")
        for name in os.listdir(output_dir.parent)
    )


def test_invalid_tenth_manifest_leaves_zero_final_output(
    tmp_path: Path,
    configs_dir: Path,
) -> None:
    records = _records()
    source_dir = tmp_path / "source"
    for subject in range(1, 11):
        _write_manifest(
            source_dir / f"sub-{subject:02d}_train.json",
            records,
            subject_id=f"sub-{subject:02d}",
            split="test" if subject == 10 else "train",
        )
    output_dir = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="split must be train"):
        main(
            [
                "--protocol",
                str(configs_dir / "protocol_v1.json"),
                "--source-manifest-dir",
                str(source_dir),
                "--output-dir",
                str(output_dir),
            ]
        )
    assert not output_dir.exists()
    assert not any(
        name.startswith(f".{output_dir.name}.staging-")
        for name in os.listdir(output_dir.parent)
    )
