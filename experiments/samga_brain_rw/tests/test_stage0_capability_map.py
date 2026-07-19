from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest

from samga_brain_rw import capability_map as capability_map_module
from samga_brain_rw.capability_map import (
    CAPABILITY_MAP_FILENAME,
    build_stage0_capability_map,
)
from samga_brain_rw.hashing import (
    canonical_json_bytes,
    ordered_ids_sha256,
    sha256_json,
)
from samga_brain_rw.provenance import (
    CAPABILITY_PAYLOAD_TYPES,
    ProvenanceInputs,
    build_provenance_manifest,
    expected_capability_paths,
)
from scripts.build_stage0_capability_map import parse_args
from scripts.preflight import load_and_verify_capability_map


_GENERIC_ENVELOPE_KEYS = {
    "metadata",
    "metadata_sha256",
    "ordered_ids_sha256",
    "payload_sha256",
    "payload_type",
    "provenance",
    "provenance_sha256",
    "schema_version",
    "scope",
    "source_records_sha256",
}
_MAP_ENTRY_KEYS = {
    "envelope_path",
    "key",
    "payload_path",
    "payload_type",
    "role",
}


def _load_provenance_fixture_builder():
    source = Path(__file__).with_name("test_provenance.py")
    spec = importlib.util.spec_from_file_location(
        "_stage0_capability_fixture_source",
        source,
    )
    if spec is None or spec.loader is None:  # pragma: no cover - invariant
        raise RuntimeError("cannot load the synthetic provenance fixture")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.synthetic_inputs.__wrapped__


_build_synthetic_inputs = _load_provenance_fixture_builder()


@pytest.fixture()
def synthetic_inputs(tmp_path: Path) -> ProvenanceInputs:
    inputs, _ = _build_synthetic_inputs(tmp_path)
    return inputs


def test_builder_writes_exact_canonical_map_and_39_verifiable_sidecars(
    synthetic_inputs: ProvenanceInputs,
    tmp_path: Path,
) -> None:
    output = tmp_path / "stage0-capabilities"

    map_path = build_stage0_capability_map(synthetic_inputs, output)

    assert map_path == output / CAPABILITY_MAP_FILENAME
    files = sorted(path.name for path in output.iterdir())
    assert len(files) == 40
    assert files.count(CAPABILITY_MAP_FILENAME) == 1
    assert sum(name.endswith(".envelope.json") for name in files) == 39

    raw_map = map_path.read_bytes()
    payload = json.loads(raw_map)
    assert raw_map == canonical_json_bytes(payload)
    assert not raw_map.endswith(b"\n")
    assert set(payload) == {
        "artifacts",
        "payload_type",
        "schema_version",
        "scope",
    }
    assert payload["payload_type"] == "samga_brain_rw.capability_map"
    assert payload["schema_version"] == 1
    assert payload["scope"] == "train"
    assert len(payload["artifacts"]) == 49
    assert [entry["key"] for entry in payload["artifacts"]] == list(
        CAPABILITY_PAYLOAD_TYPES
    )

    generic_count = 0
    protocol_count = 0
    for entry in payload["artifacts"]:
        assert set(entry) == _MAP_ENTRY_KEYS
        payload_path = Path(entry["payload_path"])
        envelope_path = Path(entry["envelope_path"])
        assert payload_path.is_absolute()
        assert envelope_path.is_absolute()
        if entry["key"].startswith("protocol_manifest."):
            protocol_count += 1
            assert entry["role"] == "train"
            assert envelope_path == payload_path
            continue

        generic_count += 1
        assert entry["role"] is None
        assert envelope_path.parent == output.absolute()
        envelope = json.loads(envelope_path.read_bytes())
        assert set(envelope) == _GENERIC_ENVELOPE_KEYS
        assert envelope["schema_version"] == 1
        assert envelope["scope"] == "train"
        assert envelope["payload_type"] == entry["payload_type"]
        assert envelope["payload_sha256"]
        metadata = envelope["metadata"]
        provenance = envelope["provenance"]
        assert set(metadata) == {
            "absolute_payload_path",
            "byte_count",
            "capability_key",
            "ordered_ids",
            "source_records",
        }
        assert metadata["absolute_payload_path"] == str(payload_path)
        assert metadata["byte_count"] == payload_path.stat().st_size
        assert metadata["capability_key"] == entry["key"]
        assert metadata["ordered_ids"] == [entry["key"]]
        assert metadata["source_records"] == []
        assert envelope["metadata_sha256"] == sha256_json(metadata)
        assert envelope["provenance_sha256"] == sha256_json(provenance)
        assert provenance == {
            "experiment_revision": synthetic_inputs.experiment_revision,
            "generator": "samga_brain_rw.capability_map.v1",
            "protocol_config_sha256": synthetic_inputs.oracles.protocol_config_sha256,
        }
        assert envelope["ordered_ids_sha256"] == ordered_ids_sha256(
            metadata["ordered_ids"]
        )
        assert envelope["source_records_sha256"] == sha256_json([])

    assert generic_count == 39
    assert protocol_count == 10
    verified = load_and_verify_capability_map(
        map_path,
        expected_capability_paths(synthetic_inputs),
    )
    assert len(verified) == 49
    assert tuple(verified) == tuple(CAPABILITY_PAYLOAD_TYPES)

    manifest = build_provenance_manifest(
        replace(synthetic_inputs, verified_artifacts=verified)
    )
    inventory = manifest["capability_inventory"]
    assert inventory["artifact_count"] == 49
    assert inventory["inventory_sha256"] == sha256_json(
        inventory["artifacts"]
    )


def test_builder_never_globs_and_creates_output_only_after_all_validation(
    synthetic_inputs: ProvenanceInputs,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "stage0-capabilities"
    hashed = 0
    semantic_validation_complete = False
    writes: list[str] = []
    original_hash = capability_map_module._hash_regular_file
    original_validate = capability_map_module._validate_snapshots_semantically
    original_create = capability_map_module._create_output_directory
    original_write = capability_map_module._write_exclusive

    def forbid_glob(*_: object, **__: object) -> object:
        raise AssertionError("directory scanning is forbidden")

    def count_hash(path: Path):
        nonlocal hashed
        result = original_hash(path)
        hashed += 1
        return result

    def record_validation(inputs, snapshots):
        nonlocal semantic_validation_complete
        result = original_validate(inputs, snapshots)
        semantic_validation_complete = True
        return result

    def check_create(path: Path):
        assert hashed == 49
        assert semantic_validation_complete is True
        return original_create(path)

    def record_write(directory_fd: int, name: str, data: bytes) -> None:
        writes.append(name)
        original_write(directory_fd, name, data)

    monkeypatch.setattr(Path, "glob", forbid_glob)
    monkeypatch.setattr(Path, "rglob", forbid_glob)
    monkeypatch.setattr(capability_map_module, "_hash_regular_file", count_hash)
    monkeypatch.setattr(
        capability_map_module,
        "_validate_snapshots_semantically",
        record_validation,
    )
    monkeypatch.setattr(
        capability_map_module,
        "_create_output_directory",
        check_create,
    )
    monkeypatch.setattr(
        capability_map_module,
        "_write_exclusive",
        record_write,
    )

    build_stage0_capability_map(synthetic_inputs, output)

    assert len(writes) == 40
    assert writes[-1] == CAPABILITY_MAP_FILENAME
    assert all(name.endswith(".envelope.json") for name in writes[:-1])


@pytest.mark.parametrize("forbidden_component", ["formal-test", "test_images"])
def test_forbidden_input_fails_before_output_creation(
    synthetic_inputs: ProvenanceInputs,
    tmp_path: Path,
    forbidden_component: str,
) -> None:
    output = tmp_path / "stage0-capabilities"
    forbidden = tmp_path / forbidden_component / "protocol.json"
    changed = replace(synthetic_inputs, protocol_path=forbidden)

    with pytest.raises(ValueError, match="forbidden|denied"):
        build_stage0_capability_map(changed, output)

    assert not output.exists()


def test_symlink_input_fails_before_output_creation(
    synthetic_inputs: ProvenanceInputs,
    tmp_path: Path,
) -> None:
    output = tmp_path / "stage0-capabilities"
    linked = tmp_path / "linked-protocol.json"
    linked.symlink_to(synthetic_inputs.protocol_path)

    with pytest.raises(ValueError, match="symlink"):
        build_stage0_capability_map(
            replace(synthetic_inputs, protocol_path=linked),
            output,
        )

    assert not output.exists()


def test_mutation_after_hash_fails_before_output_creation(
    synthetic_inputs: ProvenanceInputs,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "stage0-capabilities"
    original_hash = capability_map_module._hash_regular_file
    mutated = False

    def mutate_after_hash(path: Path):
        nonlocal mutated
        snapshot = original_hash(path)
        if path == synthetic_inputs.protocol_path and not mutated:
            path.write_bytes(path.read_bytes() + b" ")
            mutated = True
        return snapshot

    monkeypatch.setattr(
        capability_map_module,
        "_hash_regular_file",
        mutate_after_hash,
    )
    with pytest.raises(ValueError, match="changed|identity|SHA-256"):
        build_stage0_capability_map(synthetic_inputs, output)

    assert mutated is True
    assert not output.exists()


def test_existing_output_directory_is_preserved(
    synthetic_inputs: ProvenanceInputs,
    tmp_path: Path,
) -> None:
    output = tmp_path / "stage0-capabilities"
    output.mkdir()
    sentinel = output / "belongs-to-user.txt"
    sentinel.write_text("preserve me", encoding="utf-8")

    with pytest.raises(FileExistsError):
        build_stage0_capability_map(synthetic_inputs, output)

    assert sentinel.read_text(encoding="utf-8") == "preserve me"
    assert sorted(path.name for path in output.iterdir()) == [sentinel.name]


def test_mid_write_failure_removes_only_the_new_output_directory(
    synthetic_inputs: ProvenanceInputs,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "stage0-capabilities"
    original_write = capability_map_module._write_exclusive
    calls = 0

    def fail_third_write(directory_fd: int, name: str, data: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("injected write failure")
        original_write(directory_fd, name, data)

    monkeypatch.setattr(
        capability_map_module,
        "_write_exclusive",
        fail_third_write,
    )

    with pytest.raises(OSError, match="injected write failure"):
        build_stage0_capability_map(synthetic_inputs, output)

    assert calls == 3
    assert not output.exists()


def test_cleanup_never_removes_a_replacement_output_directory(
    synthetic_inputs: ProvenanceInputs,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "stage0-capabilities"
    displaced = tmp_path / "displaced-owned-output"
    original_write = capability_map_module._write_exclusive
    calls = 0

    def replace_path_then_fail(
        directory_fd: int,
        name: str,
        data: bytes,
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            output.rename(displaced)
            output.mkdir()
            raise OSError("injected replacement race")
        original_write(directory_fd, name, data)

    monkeypatch.setattr(
        capability_map_module,
        "_write_exclusive",
        replace_path_then_fail,
    )

    with pytest.raises(OSError, match="injected replacement race"):
        build_stage0_capability_map(synthetic_inputs, output)

    assert output.is_dir()
    assert list(output.iterdir()) == []
    assert displaced.is_dir()
    assert list(displaced.iterdir()) == []


def test_output_creation_race_never_removes_another_process_directory(
    synthetic_inputs: ProvenanceInputs,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "stage0-capabilities"
    original_validate = capability_map_module._validate_snapshots_semantically

    def race_after_validation(inputs, snapshots) -> None:
        original_validate(inputs, snapshots)
        output.mkdir()

    monkeypatch.setattr(
        capability_map_module,
        "_validate_snapshots_semantically",
        race_after_validation,
    )

    with pytest.raises(FileExistsError):
        build_stage0_capability_map(synthetic_inputs, output)

    assert output.is_dir()
    assert list(output.iterdir()) == []


def test_cli_requires_preflight_paths_revisions_and_new_output_directory() -> None:
    values = {
        "--repository-root": "/repo",
        "--protocol": "/repo/protocol.json",
        "--internvit-config": "/repo/internvit.json",
        "--brainrw-config": "/repo/brainrw.json",
        "--source-manifest-dir": "/source",
        "--manifest-dir": "/manifests",
        "--feature-directory": "/features",
        "--variant-directory": "/variant",
        "--canonical-cache": "/variant/features.npy",
        "--clip-train-cache": "/cache/clip.npy",
        "--data-root": "/data",
        "--model-path": "/model",
        "--clip-model-path": "/clip",
        "--upstream-root": "/upstream",
        "--experiment-revision": "1" * 40,
        "--upstream-revision": "2" * 40,
        "--cache-generator-revision": "3" * 40,
        "--output-directory": "/output/capabilities",
    }
    args = parse_args([value for pair in values.items() for value in pair])

    assert args.output_directory == Path("/output/capabilities")
    assert args.clip_model_path == Path("/clip")
    assert not hasattr(args, "capability_map")
    assert not hasattr(args, "output")
