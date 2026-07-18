from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import samga_brain_rw.artifacts as artifacts_module

from samga_brain_rw.artifacts import (
    ArtifactIntegrityError,
    ArtifactStateError,
    ConfirmationCellLedger,
    ConfirmationSeal,
    FinalRunAudit,
    FinalRunSeal,
    FormalCellLedger,
    FormalInputLedger,
    FormalInputLedgerSnapshot,
    FormalPreparationAudit,
    FormalPreparationSeal,
    RefitArtifactLedger,
    RefitCell,
    expected_formal_cell_keys,
)


def _h(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _refit_cell(
    key: str = "candidate-sub01-seed42",
    *,
    checkpoint: str = "checkpoint",
) -> RefitCell:
    return RefitCell(
        cell_key=key,
        job_id=f"job-{key}",
        subject_set=(1,),
        seed=42,
        role="candidate",
        component_schedule_sha256=_h("schedule"),
        config_sha256=_h("config"),
        manifest_set_sha256=_h("manifests"),
        checkpoint_sha256=_h(checkpoint),
        frozen_base_model_sha256=_h("base"),
        adapter_sha256=_h("adapter"),
        train_cache_sha256=_h("train-cache"),
        dependency_sha256=(_h("dependency"),),
    )


def _preparation_seal(path: Path) -> FormalPreparationSeal:
    return FormalPreparationSeal.create(
        final_selection_sha256=_h("selection"),
        confirmation_registry_sha256=_h("confirmation-registry"),
        refit_plan_sha256=_h("refit-plan"),
        refit_artifact_ledger_sha256=_h("refit-ledger"),
        formal_input_request_sha256=_h("formal-input-request"),
        expected_formal_cell_keys_sha256=_h("formal-cell-keys"),
        git_sha="1" * 40,
        upstream_sha="2" * 40,
        output_path=path,
    )


def _final_run_seal(path: Path, *, formal_input_ledger: str) -> FinalRunSeal:
    return FinalRunSeal.create(
        final_selection_sha256=_h("selection"),
        candidate_config_sha256=_h("candidate"),
        control_config_sha256=_h("control"),
        confirmation_registry_sha256=_h("confirmation-registry"),
        refit_plan_sha256=_h("refit-plan"),
        refit_artifact_ledger_sha256=_h("refit-ledger"),
        formal_input_ledger_sha256=formal_input_ledger,
        formal_job_map_sha256=_h("formal-job-map"),
        git_sha="1" * 40,
        upstream_sha="2" * 40,
        output_path=path,
    )


def test_confirmation_seal_is_canonical_exclusive_and_never_replaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_a = tmp_path / "seal-a.json"
    output_b = tmp_path / "seal-b.json"

    def forbidden_replace(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("immutable publication must not call os.replace")

    monkeypatch.setattr(os, "replace", forbidden_replace)
    first = ConfirmationSeal.create(
        survivor_config_sha256=[_h("b"), _h("a")],
        registry_sha256=_h("registry"),
        job_map_sha256=_h("map"),
        output_path=output_a,
    )
    reordered = ConfirmationSeal.create(
        survivor_config_sha256=[_h("a"), _h("b")],
        registry_sha256=_h("registry"),
        job_map_sha256=_h("map"),
        output_path=output_b,
    )

    assert first.payload_sha256 == reordered.payload_sha256
    original = output_a.read_bytes()
    with pytest.raises(FileExistsError):
        ConfirmationSeal.create(
            survivor_config_sha256=[_h("different")],
            registry_sha256=_h("registry"),
            job_map_sha256=_h("map"),
            output_path=output_a,
        )
    assert output_a.read_bytes() == original
    assert not list(tmp_path.glob(".seal-a.json.tmp-*"))


def test_concurrent_seal_creation_has_exactly_one_winner(tmp_path: Path) -> None:
    output = tmp_path / "seal.json"

    def create(index: int) -> str:
        try:
            ConfirmationSeal.create(
                survivor_config_sha256=[_h(f"candidate-{index}")],
                registry_sha256=_h("registry"),
                job_map_sha256=_h("map"),
                output_path=output,
            )
            return "won"
        except FileExistsError:
            return "lost"

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(create, range(8)))

    assert outcomes.count("won") == 1
    assert outcomes.count("lost") == 7
    ConfirmationSeal.verify(output)


def test_confirmation_seal_rejects_duplicates_bad_hashes_and_invalid_existing_file(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        ConfirmationSeal.create(
            survivor_config_sha256=[_h("a"), _h("a")],
            registry_sha256=_h("registry"),
            job_map_sha256=_h("map"),
            output_path=tmp_path / "duplicate.json",
        )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        ConfirmationSeal.create(
            survivor_config_sha256=[_h("a").upper()],
            registry_sha256=_h("registry"),
            job_map_sha256=_h("map"),
            output_path=tmp_path / "uppercase.json",
        )

    invalid = tmp_path / "occupied.json"
    invalid.write_bytes(b"not-json")
    with pytest.raises(FileExistsError):
        ConfirmationSeal.create(
            survivor_config_sha256=[_h("a")],
            registry_sha256=_h("registry"),
            job_map_sha256=_h("map"),
            output_path=invalid,
        )
    assert invalid.read_bytes() == b"not-json"


def test_refit_ledger_binds_cells_canonically_and_detects_mutation(
    tmp_path: Path,
) -> None:
    cell_a = _refit_cell("candidate-sub01-seed42")
    cell_b = _refit_cell("control-sub01-seed42", checkpoint="control-checkpoint")
    first = RefitArtifactLedger.create(
        [cell_b, cell_a],
        tmp_path / "ledger-a.json",
    )
    reordered = RefitArtifactLedger.create(
        [cell_a, cell_b],
        tmp_path / "ledger-b.json",
    )
    assert first.payload_sha256 == reordered.payload_sha256
    assert first.payload["cells"][0]["cell_key"] == cell_a.cell_key

    with pytest.raises(ValueError, match="duplicate cell_key"):
        RefitArtifactLedger.create(
            [cell_a, cell_a],
            tmp_path / "duplicates.json",
        )

    document = json.loads(first.path.read_text(encoding="utf-8"))
    document["payload"]["cells"][0]["checkpoint_sha256"] = _h("mutated")
    first.path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError):
        first.verify_unchanged()


def test_confirmation_claim_completion_is_separate_immutable_and_consumed(
    tmp_path: Path,
) -> None:
    ledger = ConfirmationCellLedger(
        tmp_path / "confirmation-cells",
        job_map_sha256=_h("confirmation-job-map"),
    )
    claim = ledger.claim(
        seal_sha256=_h("confirmation-seal"),
        stage=2,
        role="candidate",
        subject=8,
        seed=42,
    )
    claim_bytes = claim.path.read_bytes()
    with pytest.raises(FileExistsError):
        ledger.claim(
            seal_sha256=_h("confirmation-seal"),
            stage=2,
            role="candidate",
            subject=8,
            seed=42,
        )

    assert not claim.is_complete()
    completion = claim.complete(
        {
            "metrics_sha256": _h("metrics"),
            "predictions_sha256": _h("predictions"),
        }
    )
    assert claim.path.read_bytes() == claim_bytes
    assert completion.path.name == "completion.json"
    assert completion.payload["claim_sha256"] == claim.sha256
    assert claim.is_complete()
    with pytest.raises(ArtifactStateError, match="consumed"):
        claim.assert_unconsumed()
    with pytest.raises(FileExistsError):
        claim.complete({"metrics_sha256": _h("new-metrics")})


def test_stale_claim_recovery_preserves_claim_and_requires_audit(
    tmp_path: Path,
) -> None:
    ledger = ConfirmationCellLedger(
        tmp_path / "confirmation-cells",
        job_map_sha256=_h("confirmation-job-map"),
    )
    original = ledger.claim(
        seal_sha256=_h("confirmation-seal"),
        stage=1,
        role="candidate",
        subject=1,
        seed=43,
    )
    original_bytes = original.path.read_bytes()

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        ledger.recover(original, "not-an-audit")
    recovered = ledger.recover(original, _h("stale-claim-audit"))

    assert original.path.read_bytes() == original_bytes
    assert original.recovery_path.exists()
    assert recovered.generation == 2
    assert recovered.payload["recovered_from_claim_sha256"] == original.sha256
    with pytest.raises(ArtifactStateError, match="recovered"):
        original.assert_unconsumed()
    with pytest.raises(FileExistsError):
        ledger.recover(original, _h("second-audit"))


def test_preparation_seal_and_separate_audit_bind_exact_payload(
    tmp_path: Path,
) -> None:
    seal = _preparation_seal(tmp_path / "preparation.json")
    audit = FormalPreparationAudit.create(
        preparation_seal_sha256=seal.sha256,
        expected_payload_sha256=seal.payload_sha256,
        output_path=tmp_path / "preparation-audit.json",
    )
    FormalPreparationAudit.verify(
        audit.path,
        expected_preparation_seal_sha256=seal.sha256,
        expected_payload_sha256=seal.payload_sha256,
    )
    with pytest.raises(ArtifactIntegrityError):
        FormalPreparationAudit.verify(
            audit.path,
            expected_preparation_seal_sha256=_h("another-seal"),
            expected_payload_sha256=seal.payload_sha256,
        )
    with pytest.raises(ArtifactIntegrityError):
        FormalPreparationAudit.verify(
            audit.path,
            expected_preparation_seal_sha256=seal.sha256,
            expected_payload_sha256=_h("another-payload"),
        )


def test_formal_input_ledger_requires_completed_nonempty_unique_recipes(
    tmp_path: Path,
) -> None:
    ledger = FormalInputLedger(tmp_path / "formal-input-claims")
    with pytest.raises(ArtifactStateError, match="nonempty"):
        ledger.finalize(tmp_path / "empty-ledger.json")

    seal_hash = _h("preparation-seal")
    audit_hash = _h("preparation-audit")
    claim = ledger.claim(
        preparation_seal_sha256=seal_hash,
        preparation_audit_sha256=audit_hash,
        recipe_id="candidate-internvit-cache",
    )
    with pytest.raises(FileExistsError):
        ledger.claim(
            preparation_seal_sha256=seal_hash,
            preparation_audit_sha256=audit_hash,
            recipe_id="candidate-internvit-cache",
        )
    with pytest.raises(ArtifactStateError, match="incomplete"):
        ledger.finalize(tmp_path / "incomplete-ledger.json")

    claim.complete(
        manifest_sha256=_h("manifest"),
        ordered_ids_sha256=_h("ordered-ids"),
        preprocessing_sha256=_h("preprocessing"),
        base_model_sha256=_h("base-model"),
        payload_sha256=_h("formal-cache"),
        adapter_sha256=_h("adapter"),
    )
    snapshot = ledger.finalize(tmp_path / "formal-input-ledger.json")
    assert snapshot.payload["entry_count"] == 1
    entry = snapshot.payload["entries"][0]
    assert entry["preparation_seal_sha256"] == seal_hash
    assert entry["preparation_audit_sha256"] == audit_hash
    assert entry["recipe_id"] == "candidate-internvit-cache"


def test_formal_input_completion_rejects_missing_or_unknown_dependencies(
    tmp_path: Path,
) -> None:
    claim = FormalInputLedger(tmp_path / "formal-input-claims").claim(
        preparation_seal_sha256=_h("seal"),
        preparation_audit_sha256=_h("audit"),
        recipe_id="direct-input",
    )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        claim.complete(
            manifest_sha256=_h("manifest"),
            ordered_ids_sha256=_h("ids"),
            preprocessing_sha256="",
            base_model_sha256=_h("base"),
            payload_sha256=_h("payload"),
        )


def test_final_run_seal_verification_and_separate_audit_are_exact(
    tmp_path: Path,
) -> None:
    seal = _final_run_seal(
        tmp_path / "final-run.json",
        formal_input_ledger=_h("nonempty-formal-input-ledger"),
    )
    verified = FinalRunSeal.verify(seal.path, seal.payload_sha256)
    assert verified.sha256 == seal.sha256
    with pytest.raises(ArtifactIntegrityError):
        FinalRunSeal.verify(seal.path, _h("wrong-payload"))

    audit = FinalRunAudit.create(
        final_run_seal_sha256=seal.sha256,
        expected_payload_sha256=seal.payload_sha256,
        output_path=tmp_path / "final-run-audit.json",
    )
    FinalRunAudit.verify(
        audit.path,
        expected_final_run_seal_sha256=seal.sha256,
        expected_payload_sha256=seal.payload_sha256,
    )
    with pytest.raises(ArtifactIntegrityError):
        FinalRunAudit.verify(
            audit.path,
            expected_final_run_seal_sha256=_h("wrong-seal"),
            expected_payload_sha256=seal.payload_sha256,
        )


def test_final_run_seal_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    malformed = tmp_path / "duplicate-keys.json"
    malformed.write_text(
        '{"schema_version":1,"schema_version":1,'
        '"artifact_type":"final_run_seal","payload":{},'
        f'"payload_sha256":"{_h("payload")}"}}',
        encoding="utf-8",
    )
    with pytest.raises(ArtifactIntegrityError, match="duplicate"):
        FinalRunSeal.verify(malformed, _h("payload"))


def test_formal_grid_and_claims_are_exactly_candidate_control_10_by_5(
    tmp_path: Path,
) -> None:
    keys = expected_formal_cell_keys()
    assert len(keys) == 100
    assert len(set(keys)) == 100
    assert {key.role for key in keys} == {"candidate", "control"}
    assert {key.subject for key in keys} == set(range(1, 11))
    assert {key.seed for key in keys} == set(range(42, 47))

    ledger = FormalCellLedger(
        tmp_path / "formal-cells",
        formal_job_map_sha256=_h("formal-job-map"),
    )
    claim = ledger.claim(
        final_run_seal_sha256=_h("final-run-seal"),
        final_run_audit_sha256=_h("final-run-audit"),
        role="candidate",
        subject=10,
        seed=46,
    )
    claim.complete(
        {
            "metrics_sha256": _h("metrics"),
            "predictions_sha256": _h("predictions"),
        }
    )
    with pytest.raises(ValueError, match="role"):
        ledger.claim(
            final_run_seal_sha256=_h("seal"),
            final_run_audit_sha256=_h("audit"),
            role="ablation",
            subject=1,
            seed=42,
        )
    with pytest.raises(ValueError, match="seed"):
        ledger.claim(
            final_run_seal_sha256=_h("seal"),
            final_run_audit_sha256=_h("audit"),
            role="control",
            subject=1,
            seed=47,
        )


def test_artifact_reader_rejects_symlinked_parent_components(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real"
    seal = ConfirmationSeal.create(
        [_h("candidate")],
        _h("registry"),
        _h("job-map"),
        real / "seal.json",
    )
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)

    with pytest.raises(ArtifactIntegrityError, match="symlink"):
        ConfirmationSeal.verify(alias / seal.path.name)


def test_artifact_reader_rejects_change_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seal = ConfirmationSeal.create(
        [_h("candidate")],
        _h("registry"),
        _h("job-map"),
        tmp_path / "seal.json",
    )
    original_read = artifacts_module.os.read
    changed = False

    def read_then_touch(fd: int, count: int) -> bytes:
        nonlocal changed
        chunk = original_read(fd, count)
        if chunk and not changed:
            changed = True
            current = seal.path.stat()
            os.utime(
                seal.path,
                ns=(current.st_atime_ns, current.st_mtime_ns + 1_000_000),
            )
        return chunk

    monkeypatch.setattr(artifacts_module.os, "read", read_then_touch)
    with pytest.raises(ArtifactIntegrityError, match="changed during read"):
        ConfirmationSeal.verify(seal.path)


def test_artifact_reader_rejects_short_read_against_stable_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "bytes.bin"
    path.write_bytes(b"x" * 128)
    original_read = artifacts_module.os.read
    calls = 0

    def truncated_read(fd: int, count: int) -> bytes:
        nonlocal calls
        calls += 1
        if calls == 1:
            return original_read(fd, 16)
        return b""

    monkeypatch.setattr(artifacts_module.os, "read", truncated_read)
    with pytest.raises(ArtifactIntegrityError, match="byte count"):
        artifacts_module._read_regular_file(path)


def test_verifier_rejects_noncanonical_raw_json(tmp_path: Path) -> None:
    seal = ConfirmationSeal.create(
        [_h("candidate")],
        _h("registry"),
        _h("job-map"),
        tmp_path / "seal.json",
    )
    document = json.loads(seal.path.read_text(encoding="utf-8"))
    seal.path.write_text(json.dumps(document, indent=2), encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match="canonical"):
        ConfirmationSeal.verify(seal.path)


def test_verifier_requires_exact_integer_schema(tmp_path: Path) -> None:
    seal = ConfirmationSeal.create(
        [_h("candidate")],
        _h("registry"),
        _h("job-map"),
        tmp_path / "seal.json",
    )
    document = json.loads(seal.path.read_text(encoding="utf-8"))
    document["schema_version"] = True
    seal.path.write_bytes(artifacts_module.canonical_json_bytes(document))

    with pytest.raises(ArtifactIntegrityError, match="schema_version"):
        ConfirmationSeal.verify(seal.path)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_verifier_rejects_non_finite_json_constants(
    tmp_path: Path,
    constant: str,
) -> None:
    seal = ConfirmationSeal.create(
        [_h("candidate")],
        _h("registry"),
        _h("job-map"),
        tmp_path / "seal.json",
    )
    raw = seal.path.read_text(encoding="utf-8").replace(
        '"schema_version":1',
        f'"schema_version":{constant}',
    )
    seal.path.write_text(raw, encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match="non-finite"):
        ConfirmationSeal.verify(seal.path)


def test_frozen_records_expose_recursively_immutable_payloads(
    tmp_path: Path,
) -> None:
    seal = ConfirmationSeal.create(
        [_h("candidate")],
        _h("registry"),
        _h("job-map"),
        tmp_path / "seal.json",
    )

    with pytest.raises(TypeError):
        seal.payload["registry_sha256"] = _h("mutated")  # type: ignore[index]
    survivors = seal.payload["survivor_config_sha256"]
    assert isinstance(survivors, tuple)
    with pytest.raises(AttributeError):
        survivors.append(_h("mutated"))  # type: ignore[union-attr]


def test_publication_rejects_symlinked_parent(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)

    with pytest.raises(ArtifactIntegrityError, match="symlink"):
        ConfirmationSeal.create(
            [_h("candidate")],
            _h("registry"),
            _h("job-map"),
            alias / "seal.json",
        )
    assert not (real / "seal.json").exists()


def test_publication_unlinks_temp_before_parent_fsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_fsync = artifacts_module.os.fsync
    directory_fsyncs = 0

    def checked_fsync(fd: int) -> None:
        nonlocal directory_fsyncs
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            directory_fsyncs += 1
            assert not list(tmp_path.glob(".seal.json.tmp-*"))
        original_fsync(fd)

    monkeypatch.setattr(artifacts_module.os, "fsync", checked_fsync)
    ConfirmationSeal.create(
        [_h("candidate")],
        _h("registry"),
        _h("job-map"),
        tmp_path / "seal.json",
    )
    assert directory_fsyncs == 1


def test_failed_publication_unlinks_fsyncs_and_closes_parent_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "seal.json"
    ConfirmationSeal.create(
        [_h("winner")],
        _h("registry"),
        _h("job-map"),
        output,
    )
    original_fsync = artifacts_module.os.fsync
    directory_fsyncs = 0

    def checked_fsync(fd: int) -> None:
        nonlocal directory_fsyncs
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            directory_fsyncs += 1
            assert not list(tmp_path.glob(".seal.json.tmp-*"))
        original_fsync(fd)

    monkeypatch.setattr(artifacts_module.os, "fsync", checked_fsync)
    before_fds = set(os.listdir("/proc/self/fd"))
    with pytest.raises(FileExistsError):
        ConfirmationSeal.create(
            [_h("loser")],
            _h("registry"),
            _h("job-map"),
            output,
        )
    after_fds = set(os.listdir("/proc/self/fd"))

    assert directory_fsyncs == 1
    assert after_fds == before_fds


def test_confirmation_claim_loads_and_completes_across_processes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "confirmation"
    job_map_sha256 = _h("confirmation-job-map")
    claim = ConfirmationCellLedger(
        root,
        job_map_sha256=job_map_sha256,
    ).claim(
        seal_sha256=_h("confirmation-seal"),
        stage=2,
        role="candidate",
        subject=8,
        seed=42,
    )
    code = """
import sys
from pathlib import Path
from samga_brain_rw.artifacts import ConfirmationCellLedger
ledger = ConfirmationCellLedger(Path(sys.argv[1]), job_map_sha256=sys.argv[2])
claim = ledger.load_claim(stage=2, role="candidate", subject=8, seed=42)
claim.complete({"metrics_sha256": sys.argv[3]})
"""
    subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(root),
            job_map_sha256,
            _h("metrics"),
        ],
        check=True,
        env={**os.environ, "PYTHONPATH": "experiments/samga_brain_rw"},
    )
    assert claim.is_complete()


def test_formal_cell_ledger_loads_current_recovered_generation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "formal-cells"
    job_map_sha256 = _h("formal-job-map")
    ledger = FormalCellLedger(root, formal_job_map_sha256=job_map_sha256)
    first = ledger.claim(
        final_run_seal_sha256=_h("final-seal"),
        final_run_audit_sha256=_h("final-audit"),
        role="control",
        subject=3,
        seed=44,
    )
    recovered = ledger.recover(first, _h("recovery-audit"))

    loaded = FormalCellLedger(
        root,
        formal_job_map_sha256=job_map_sha256,
    ).load_claim(role="control", subject=3, seed=44)

    assert loaded.generation == 2
    assert loaded.sha256 == recovered.sha256
    loaded.assert_unconsumed()


def test_formal_input_ledger_loads_current_recovered_generation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "formal-input"
    ledger = FormalInputLedger(root)
    first = ledger.claim(
        preparation_seal_sha256=_h("preparation-seal"),
        preparation_audit_sha256=_h("preparation-audit"),
        recipe_id="candidate-cache",
    )
    recovered = ledger.recover(first, _h("recovery-audit"))

    loaded = FormalInputLedger(root).load_claim("candidate-cache")

    assert loaded.generation == 2
    assert loaded.sha256 == recovered.sha256
    loaded.assert_unconsumed()


def _complete_formal_input_recipe(
    ledger: FormalInputLedger,
    recipe_id: str,
) -> None:
    claim = ledger.claim(
        preparation_seal_sha256=_h("preparation-seal"),
        preparation_audit_sha256=_h("preparation-audit"),
        recipe_id=recipe_id,
    )
    claim.complete(
        manifest_sha256=_h(f"{recipe_id}-manifest"),
        ordered_ids_sha256=_h(f"{recipe_id}-ordered-ids"),
        preprocessing_sha256=_h(f"{recipe_id}-preprocessing"),
        base_model_sha256=_h(f"{recipe_id}-base"),
        payload_sha256=_h(f"{recipe_id}-payload"),
    )


def test_formal_input_snapshot_verifies_self_contained_claim_and_completion(
    tmp_path: Path,
) -> None:
    ledger = FormalInputLedger(tmp_path / "formal-input")
    _complete_formal_input_recipe(ledger, "candidate-cache")
    snapshot = ledger.finalize(tmp_path / "ledger.json")

    verified = FormalInputLedgerSnapshot.verify(
        snapshot.path,
        expected_payload_sha256=snapshot.payload_sha256,
    )

    assert verified.sha256 == snapshot.sha256
    entry = verified.payload["entries"][0]
    assert entry["claim_generation"] == 1
    assert entry["claim_path"].endswith("generation-000001/claim.json")
    assert entry["claim_payload"]["recipe_id"] == "candidate-cache"
    assert entry["completion_payload"]["claim_sha256"] == entry["claim_sha256"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda entry: entry.update({"unknown": True}),
        lambda entry: entry.update({"claim_sha256": _h("wrong-claim")}),
        lambda entry: entry["claim_payload"].update(
            {"preparation_seal_sha256": _h("wrong-seal")}
        ),
        lambda entry: entry.update({"completion_sha256": _h("wrong-completion")}),
        lambda entry: entry.update(
            {"claim_path": "recipe-wrong/generation-000001/claim.json"}
        ),
    ],
)
def test_formal_input_snapshot_rejects_broken_embedded_bindings(
    tmp_path: Path,
    mutation,
) -> None:
    ledger = FormalInputLedger(tmp_path / "formal-input")
    _complete_formal_input_recipe(ledger, "candidate-cache")
    snapshot = ledger.finalize(tmp_path / "ledger.json")
    document = json.loads(snapshot.path.read_text(encoding="utf-8"))
    mutation(document["payload"]["entries"][0])
    document["payload_sha256"] = artifacts_module.sha256_json(document["payload"])
    snapshot.path.write_bytes(artifacts_module.canonical_json_bytes(document))

    with pytest.raises(ArtifactIntegrityError):
        FormalInputLedgerSnapshot.verify(snapshot.path)


def test_formal_input_snapshot_rejects_unsorted_entries(
    tmp_path: Path,
) -> None:
    ledger = FormalInputLedger(tmp_path / "formal-input")
    _complete_formal_input_recipe(ledger, "a-recipe")
    _complete_formal_input_recipe(ledger, "b-recipe")
    snapshot = ledger.finalize(tmp_path / "ledger.json")
    document = json.loads(snapshot.path.read_text(encoding="utf-8"))
    document["payload"]["entries"].reverse()
    document["payload_sha256"] = artifacts_module.sha256_json(document["payload"])
    snapshot.path.write_bytes(artifacts_module.canonical_json_bytes(document))

    with pytest.raises(ArtifactIntegrityError, match="sorted"):
        FormalInputLedgerSnapshot.verify(snapshot.path)


def test_claim_loader_rejects_unknown_fields_and_broken_recovery_chain(
    tmp_path: Path,
) -> None:
    confirmation_root = tmp_path / "confirmation"
    confirmation = ConfirmationCellLedger(
        confirmation_root,
        job_map_sha256=_h("confirmation-map"),
    )
    confirmation_claim = confirmation.claim(
        seal_sha256=_h("confirmation-seal"),
        stage=1,
        role="candidate",
        subject=1,
        seed=42,
    )
    document = json.loads(
        confirmation_claim.path.read_text(encoding="utf-8")
    )
    document["payload"]["unknown"] = True
    document["payload_sha256"] = artifacts_module.sha256_json(
        document["payload"]
    )
    confirmation_claim.path.write_bytes(
        artifacts_module.canonical_json_bytes(document)
    )
    with pytest.raises(ArtifactIntegrityError, match="keys"):
        ConfirmationCellLedger(
            confirmation_root,
            job_map_sha256=_h("confirmation-map"),
        ).load_claim(stage=1, role="candidate", subject=1, seed=42)

    formal_root = tmp_path / "formal"
    formal = FormalCellLedger(
        formal_root,
        formal_job_map_sha256=_h("formal-map"),
    )
    first = formal.claim(
        final_run_seal_sha256=_h("final-seal"),
        final_run_audit_sha256=_h("final-audit"),
        role="candidate",
        subject=2,
        seed=43,
    )
    recovered = formal.recover(first, _h("recovery-audit"))
    recovered_document = json.loads(
        recovered.path.read_text(encoding="utf-8")
    )
    recovered_document["payload"]["recovered_from_claim_sha256"] = _h(
        "wrong-previous-claim"
    )
    recovered_document["payload_sha256"] = artifacts_module.sha256_json(
        recovered_document["payload"]
    )
    recovered.path.write_bytes(
        artifacts_module.canonical_json_bytes(recovered_document)
    )
    with pytest.raises(ArtifactIntegrityError, match="recovery chain"):
        FormalCellLedger(
            formal_root,
            formal_job_map_sha256=_h("formal-map"),
        ).load_claim(role="candidate", subject=2, seed=43)



def test_claim_loader_requires_exact_integer_recovery_generation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "formal"
    job_map_sha256 = _h("formal-map")
    ledger = FormalCellLedger(
        root,
        formal_job_map_sha256=job_map_sha256,
    )
    first = ledger.claim(
        final_run_seal_sha256=_h("final-seal"),
        final_run_audit_sha256=_h("final-audit"),
        role="candidate",
        subject=2,
        seed=43,
    )
    recovered = ledger.recover(first, _h("recovery-audit"))

    recovery_document = json.loads(
        first.recovery_path.read_text(encoding="utf-8")
    )
    recovery_document["payload"]["next_generation"] = 2.0
    recovery_document["payload_sha256"] = artifacts_module.sha256_json(
        recovery_document["payload"]
    )
    recovery_bytes = artifacts_module.canonical_json_bytes(recovery_document)
    first.recovery_path.write_bytes(recovery_bytes)

    recovered_document = json.loads(
        recovered.path.read_text(encoding="utf-8")
    )
    recovered_document["payload"]["recovery_record_sha256"] = (
        hashlib.sha256(recovery_bytes).hexdigest()
    )
    recovered_document["payload_sha256"] = artifacts_module.sha256_json(
        recovered_document["payload"]
    )
    recovered.path.write_bytes(
        artifacts_module.canonical_json_bytes(recovered_document)
    )

    with pytest.raises(ArtifactIntegrityError, match="generation"):
        FormalCellLedger(
            root,
            formal_job_map_sha256=job_map_sha256,
        ).load_claim(role="candidate", subject=2, seed=43)
