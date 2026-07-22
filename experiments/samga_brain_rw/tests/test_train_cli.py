from __future__ import annotations

import copy
import hashlib
import json
import random
import stat
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import train as samga_train
from evaluate import (
    checkpoint_identity,
    parse_arguments as parse_evaluate_arguments,
)
from samga_brain_rw.brainrw import ManifestIdentity
from samga_brain_rw.runtime_contract import (
    PINNED_SEMANTIC_ENVIRONMENT,
    PRODUCTION_RUNTIME_CONTRACT,
    build_environment_binding,
)
from samga_brain_rw.trainer import (
    TrainingIdentities,
    _validate_checkpoint_schema,
)


def _h(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _fake_checkpoint_bundle(
    directory: Path,
    name: str,
) -> tuple[Path, str]:
    checkpoint = directory / name
    checkpoint.write_bytes(f"checkpoint:{name}".encode("utf-8"))
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    sidecar = samga_train.samga_checkpoint_sidecar(checkpoint)
    sidecar.write_bytes(
        samga_train.canonical_json_bytes(
            {
                "complete": True,
                "payload_sha256": digest,
                "payload_type": "samga_brain_rw.epoch_checkpoint",
                "schema_version": 1,
                "scope": "train",
            }
        )
        + b"\n"
    )
    return checkpoint, digest


def test_checkpoint_pruning_verifies_pair_and_fsyncs_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, digest = _fake_checkpoint_bundle(
        tmp_path,
        "checkpoint_epoch001.pt",
    )
    sidecar = samga_train.samga_checkpoint_sidecar(checkpoint)
    fsynced_directories: list[bool] = []
    real_fsync = samga_train.os.fsync

    def record_fsync(descriptor: int) -> None:
        fsynced_directories.append(
            stat.S_ISDIR(samga_train.os.fstat(descriptor).st_mode)
        )
        real_fsync(descriptor)

    monkeypatch.setattr(samga_train.os, "fsync", record_fsync)

    samga_train._prune_checkpoint_bundle(
        checkpoint,
        expected_sha256=digest,
    )

    assert not checkpoint.exists()
    assert not sidecar.exists()
    assert fsynced_directories == [True]


def test_checkpoint_pruning_streams_checkpoint_but_reads_small_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, digest = _fake_checkpoint_bundle(
        tmp_path,
        "checkpoint_epoch001.pt",
    )
    sidecar = samga_train.samga_checkpoint_sidecar(checkpoint)
    byte_reads: list[str] = []
    real_read = samga_train._read_prunable_regular

    def reject_checkpoint_byte_read(
        parent: object,
        leaf: str,
        *,
        context: str,
    ) -> tuple[bytes, tuple[int, ...]]:
        if leaf == checkpoint.name:
            raise AssertionError(
                "checkpoint payload must use streaming SHA-256"
            )
        byte_reads.append(leaf)
        return real_read(parent, leaf, context=context)

    monkeypatch.setattr(
        samga_train,
        "_read_prunable_regular",
        reject_checkpoint_byte_read,
    )

    samga_train._prune_checkpoint_bundle(
        checkpoint,
        expected_sha256=digest,
    )

    assert byte_reads == [sidecar.name]
    assert not checkpoint.exists()
    assert not sidecar.exists()


def test_checkpoint_pruning_rejects_symlink_without_touching_target(
    tmp_path: Path,
) -> None:
    target, digest = _fake_checkpoint_bundle(
        tmp_path,
        "target.pt",
    )
    checkpoint = tmp_path / "checkpoint_epoch001.pt"
    checkpoint.symlink_to(target)
    sidecar = samga_train.samga_checkpoint_sidecar(checkpoint)
    sidecar.write_bytes(
        samga_train.canonical_json_bytes(
            {
                "complete": True,
                "payload_sha256": digest,
                "payload_type": "samga_brain_rw.epoch_checkpoint",
                "schema_version": 1,
                "scope": "train",
            }
        )
        + b"\n"
    )

    with pytest.raises((PermissionError, ValueError), match="safe|regular|symlink"):
        samga_train._prune_checkpoint_bundle(
            checkpoint,
            expected_sha256=digest,
        )

    assert target.is_file()
    assert checkpoint.is_symlink()
    assert sidecar.is_file()


def test_checkpoint_pruning_uses_randomized_tombstones_before_unlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, digest = _fake_checkpoint_bundle(
        tmp_path,
        "checkpoint_epoch001.pt",
    )
    renamed: list[tuple[str, str]] = []
    real_rename = samga_train.os.rename

    def record_rename(
        source: str,
        destination: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
    ) -> None:
        renamed.append((source, destination))
        real_rename(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(samga_train.os, "rename", record_rename)

    samga_train._prune_checkpoint_bundle(
        checkpoint,
        expected_sha256=digest,
    )

    assert {source for source, _ in renamed} == {
        checkpoint.name,
        samga_train.samga_checkpoint_sidecar(checkpoint).name,
    }
    assert len({destination for _, destination in renamed}) == 2
    assert all(
        destination.startswith(".checkpoint_epoch")
        and ".prune-" in destination
        for _, destination in renamed
    )
    assert not any(tmp_path.glob("*.prune-*"))


def test_checkpoint_pruning_restores_tombstones_on_post_rename_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, digest = _fake_checkpoint_bundle(
        tmp_path,
        "checkpoint_epoch001.pt",
    )
    sidecar = samga_train.samga_checkpoint_sidecar(checkpoint)
    real_rename = samga_train.os.rename
    swapped = False

    def swap_before_checkpoint_rename(
        source: str,
        destination: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
    ) -> None:
        nonlocal swapped
        if source == checkpoint.name and not swapped:
            swapped = True
            samga_train.os.unlink(source, dir_fd=src_dir_fd)
            descriptor = samga_train.os.open(
                source,
                samga_train.os.O_WRONLY
                | samga_train.os.O_CREAT
                | samga_train.os.O_EXCL,
                0o600,
                dir_fd=src_dir_fd,
            )
            try:
                samga_train.os.write(descriptor, b"foreign-checkpoint")
                samga_train.os.fsync(descriptor)
            finally:
                samga_train.os.close(descriptor)
        real_rename(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(
        samga_train.os,
        "rename",
        swap_before_checkpoint_rename,
    )

    with pytest.raises(ValueError, match="identity.*renam|changed.*prun"):
        samga_train._prune_checkpoint_bundle(
            checkpoint,
            expected_sha256=digest,
        )

    assert swapped
    assert checkpoint.read_bytes() == b"foreign-checkpoint"
    assert sidecar.is_file()
    assert not any(tmp_path.glob("*.prune-*"))


def test_transient_checkpoint_retention_keeps_only_latest_bundle(
    tmp_path: Path,
) -> None:
    retention = samga_train._CheckpointRetention()
    first, first_digest = _fake_checkpoint_bundle(
        tmp_path,
        "checkpoint_epoch001.pt",
    )
    second, second_digest = _fake_checkpoint_bundle(
        tmp_path,
        "checkpoint_epoch002_step00000001.pt",
    )

    retention.record_published(
        first,
        first_digest,
        retain_for_averaging=False,
    )
    retention.record_published(
        second,
        second_digest,
        retain_for_averaging=False,
    )
    retention.validate_final(
        completed=False,
        final_checkpoint=second,
    )

    assert not first.exists()
    assert not samga_train.samga_checkpoint_sidecar(first).exists()
    assert second.is_file()
    assert samga_train.samga_checkpoint_sidecar(second).is_file()
    assert retention.hashes == {second.name: second_digest}


def test_full_checkpoint_retention_keeps_exact_epochs_51_through_60(
    tmp_path: Path,
) -> None:
    retention = samga_train._CheckpointRetention()
    transient, transient_digest = _fake_checkpoint_bundle(
        tmp_path,
        "checkpoint_epoch050.pt",
    )
    retention.record_published(
        transient,
        transient_digest,
        retain_for_averaging=False,
    )
    retained: list[tuple[Path, str]] = []
    for epoch in range(51, 61):
        checkpoint, digest = _fake_checkpoint_bundle(
            tmp_path,
            f"checkpoint_epoch{epoch:03d}.pt",
        )
        retention.record_published(
            checkpoint,
            digest,
            retain_for_averaging=True,
        )
        retained.append((checkpoint, digest))

    retention.validate_final(
        completed=True,
        final_checkpoint=retained[-1][0],
    )

    assert not transient.exists()
    assert not samga_train.samga_checkpoint_sidecar(transient).exists()
    assert retention.hashes == {
        checkpoint.name: digest
        for checkpoint, digest in retained
    }
    assert all(
        checkpoint.is_file()
        and samga_train.samga_checkpoint_sidecar(checkpoint).is_file()
        for checkpoint, _ in retained
    )


def test_late_partial_retention_keeps_durable_prefix_and_latest_transient(
    tmp_path: Path,
) -> None:
    retention = samga_train._CheckpointRetention()
    retained: list[tuple[Path, str]] = []
    for epoch in (51, 52):
        checkpoint, digest = _fake_checkpoint_bundle(
            tmp_path,
            f"checkpoint_epoch{epoch:03d}.pt",
        )
        retention.record_published(
            checkpoint,
            digest,
            retain_for_averaging=True,
        )
        retained.append((checkpoint, digest))
    transient, transient_digest = _fake_checkpoint_bundle(
        tmp_path,
        "checkpoint_epoch053_step00000525.pt",
    )

    retention.record_published(
        transient,
        transient_digest,
        retain_for_averaging=False,
    )
    retention.validate_final(
        completed=False,
        final_checkpoint=transient,
    )

    assert retention.hashes == {
        **{
            checkpoint.name: digest
            for checkpoint, digest in retained
        },
        transient.name: transient_digest,
    }
    assert all(checkpoint.is_file() for checkpoint, _ in retained)
    assert transient.is_file()


def test_late_partial_retention_accepts_exact_durable_epoch_boundary(
    tmp_path: Path,
) -> None:
    retention = samga_train._CheckpointRetention()
    retained: list[tuple[Path, str]] = []
    for epoch in (51, 52):
        checkpoint, digest = _fake_checkpoint_bundle(
            tmp_path,
            f"checkpoint_epoch{epoch:03d}.pt",
        )
        retention.record_published(
            checkpoint,
            digest,
            retain_for_averaging=True,
        )
        retained.append((checkpoint, digest))

    retention.validate_final(
        completed=False,
        final_checkpoint=retained[-1][0],
    )

    assert retention.hashes == {
        checkpoint.name: digest
        for checkpoint, digest in retained
    }


def test_durable_checkpoint_retention_requires_contiguous_prefix(
    tmp_path: Path,
) -> None:
    retention = samga_train._CheckpointRetention()
    checkpoint, digest = _fake_checkpoint_bundle(
        tmp_path,
        "checkpoint_epoch052.pt",
    )

    with pytest.raises(ValueError, match="contiguous|epoch 51"):
        retention.record_published(
            checkpoint,
            digest,
            retain_for_averaging=True,
        )


def _realistic_adamw_state_dict() -> dict[str, object]:
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model(torch.ones(1, 3)).sum().backward()
    optimizer.step()
    state = optimizer.state_dict()
    assert state["state"]
    assert all(type(key) is int and key >= 0 for key in state["state"])
    return state


def _production_environment_binding() -> dict[str, object]:
    return build_environment_binding(
        PINNED_SEMANTIC_ENVIRONMENT,
        PRODUCTION_RUNTIME_CONTRACT,
    )


@pytest.mark.parametrize("validation_scope", ["val-dev", "none"])
def test_train_parser_requires_locked_development_arguments(
    tmp_path: Path, validation_scope: str
) -> None:
    args = samga_train.parse_arguments(
        [
            "--scope",
            "train",
            "--validation-scope",
            validation_scope,
            "--stage",
            "0",
            "--subject",
            "8",
            "--seed",
            "42",
            "--resume",
            "none",
            "--config",
            str(tmp_path / "config.json"),
            "--manifest",
            str(tmp_path / "sub-08_protocol.json"),
            "--feature-cache",
            str(tmp_path / "features.npy"),
            "--output-dir",
            str(tmp_path / "run"),
            "--max-train-steps",
            "1",
        ]
    )
    assert args.scope == "train"
    assert args.validation_scope == validation_scope
    assert args.subject == 8
    assert args.seed == 42
    assert args.resume == "none"
    assert args.max_train_steps == 1
    assert args.device == "cuda"


@pytest.mark.parametrize("device", ["auto", "cpu", "cuda:1"])
def test_train_parser_rejects_nonproduction_device(
    tmp_path: Path,
    device: str,
) -> None:
    argv = [
        "--scope",
        "train",
        "--validation-scope",
        "val-dev",
        "--stage",
        "0",
        "--subject",
        "8",
        "--seed",
        "42",
        "--resume",
        "none",
        "--config",
        str(tmp_path / "config.json"),
        "--manifest",
        str(tmp_path / "sub-08_protocol.json"),
        "--feature-cache",
        str(tmp_path / "features.npy"),
        "--output-dir",
        str(tmp_path / "run"),
        "--device",
        device,
    ]
    with pytest.raises(SystemExit):
        samga_train.parse_arguments(argv)


def test_production_runtime_preflight_precedes_every_path_or_data_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def reject_runtime(device: str) -> object:
        events.append(f"runtime:{device}")
        raise RuntimeError("production runtime probe failed")

    def forbidden_paths(arguments: object) -> object:
        del arguments
        events.append("paths")
        raise AssertionError("paths must not be touched before runtime")

    monkeypatch.setattr(
        samga_train,
        "require_production_runtime",
        reject_runtime,
    )
    monkeypatch.setattr(
        samga_train,
        "_guard_training_paths",
        forbidden_paths,
    )

    with pytest.raises(RuntimeError, match="runtime probe"):
        samga_train.run_training(SimpleNamespace(device="cuda"))

    assert events == ["runtime:cuda"]


def test_clean_repository_and_upstream_preflight_precede_manifest_and_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    environment = _production_environment_binding()
    paths = samga_train.TrainingPaths(
        config=tmp_path / "config.json",
        manifest=tmp_path / "manifest.json",
        feature_cache=tmp_path / "features.npy",
        output_dir=tmp_path / "run",
        stage2_config=None,
        whitening_artifact=None,
        resume_checkpoint=None,
    )
    payload = {
        "config_type": "internvit_baseline",
        "upstream": {
            "path": str(tmp_path / "SAMGA"),
            "git_commit": samga_train.PINNED_UPSTREAM_SHA,
        },
    }

    class Semantic:
        sha256 = _h("early-config")

        def canonical_payload(self) -> dict[str, object]:
            return copy.deepcopy(payload)

    class SemanticFactory:
        @classmethod
        def from_path(cls, path: Path) -> Semantic:
            events.append("config")
            assert path == paths.config
            return Semantic()

    monkeypatch.setattr(
        samga_train,
        "require_production_runtime",
        lambda device: events.append("runtime")
        or SimpleNamespace(
            device=torch.device("cuda:0"),
            environment_binding=environment,
            contract=PRODUCTION_RUNTIME_CONTRACT,
            evidence={},
        ),
    )
    monkeypatch.setattr(
        samga_train,
        "clean_repository_git_sha",
        lambda: events.append("git") or "1" * 40,
    )
    monkeypatch.setattr(
        samga_train,
        "_guard_training_paths",
        lambda arguments: events.append("paths") or paths,
    )
    monkeypatch.setattr(samga_train, "SemanticConfig", SemanticFactory)

    def reject_upstream(path: Path, commit: str) -> object:
        events.append("upstream")
        assert path == (tmp_path / "SAMGA")
        assert commit == samga_train.PINNED_UPSTREAM_SHA
        raise ValueError("upstream import contract rejected")

    monkeypatch.setattr(
        samga_train,
        "load_locked_upstream_components",
        reject_upstream,
    )
    monkeypatch.setattr(
        samga_train,
        "load_development_manifest_identity",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("manifest is forbidden before upstream preflight")
        ),
    )
    monkeypatch.setattr(
        samga_train,
        "_development_dataset",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("dataset is forbidden before upstream preflight")
        ),
    )

    with pytest.raises(ValueError, match="upstream import contract"):
        samga_train.run_training(SimpleNamespace(device="cuda", subject=8))

    assert events == ["runtime", "git", "paths", "config", "upstream"]


def test_resume_environment_is_validated_before_manifest_or_dataset_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _production_environment_binding()
    mismatched = copy.deepcopy(environment)
    mismatched["semantic_environment"]["numpy"] = "2.0.0"
    mismatched["semantic_environment_sha256"] = samga_train.sha256_json(
        mismatched["semantic_environment"]
    )
    paths = samga_train.TrainingPaths(
        config=tmp_path / "config.json",
        manifest=tmp_path / "sub-08_protocol.json",
        feature_cache=tmp_path / "features.npy",
        output_dir=tmp_path / "run",
        stage2_config=None,
        whitening_artifact=None,
        resume_checkpoint=tmp_path / "checkpoint_epoch001.pt",
    )
    events: list[str] = []

    monkeypatch.setattr(
        samga_train,
        "require_production_runtime",
        lambda device: SimpleNamespace(
            device=torch.device("cuda:0"),
            environment_binding=environment,
            contract=PRODUCTION_RUNTIME_CONTRACT,
            evidence={"accelerator_name": "NVIDIA A40"},
        ),
    )
    monkeypatch.setattr(
        samga_train,
        "clean_repository_git_sha",
        lambda: events.append("git") or "1" * 40,
    )
    monkeypatch.setattr(
        samga_train,
        "preflight_upstream_config",
        lambda path: events.append("upstream") or SimpleNamespace(),
    )
    monkeypatch.setattr(
        samga_train,
        "_guard_training_paths",
        lambda arguments: events.append("paths") or paths,
    )
    monkeypatch.setattr(
        samga_train,
        "load_samga_checkpoint",
        lambda path, requested_scope: (
            events.append("resume")
            or samga_train.LoadedSAMGACheckpoint(
                payload={"epoch": 1, "environment": mismatched},
                sha256=_h("actual-resume-file"),
            )
        ),
    )

    def forbidden_manifest(*args: object, **kwargs: object) -> object:
        del args, kwargs
        events.append("manifest")
        raise AssertionError(
            "manifest/data access must follow resume environment validation"
        )

    monkeypatch.setattr(
        samga_train,
        "load_development_manifest_identity",
        forbidden_manifest,
    )
    monkeypatch.setattr(
        samga_train,
        "_development_dataset",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("ProtocolSubjectDataset must not be constructed")
        ),
    )

    with pytest.raises(ValueError, match="resume checkpoint environment"):
        samga_train.run_training(SimpleNamespace(device="cuda"))

    assert events == ["git", "paths", "upstream", "resume"]


@pytest.mark.parametrize(
    ("resume_epoch", "expect_rejected"),
    ((50, False), (51, True), (60, True)),
)
def test_resume_epoch_retention_boundary_is_enforced_before_manifest_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resume_epoch: int,
    expect_rejected: bool,
) -> None:
    environment = _production_environment_binding()
    paths = samga_train.TrainingPaths(
        config=tmp_path / "config.json",
        manifest=tmp_path / "sub-08_protocol.json",
        feature_cache=tmp_path / "features.npy",
        output_dir=tmp_path / "run",
        stage2_config=None,
        whitening_artifact=None,
        resume_checkpoint=tmp_path / f"checkpoint_epoch{resume_epoch:03d}.pt",
    )
    events: list[str] = []
    monkeypatch.setattr(
        samga_train,
        "require_production_runtime",
        lambda _device: SimpleNamespace(
            device=torch.device("cuda:0"),
            environment_binding=environment,
            contract=PRODUCTION_RUNTIME_CONTRACT,
            evidence={"accelerator_name": "NVIDIA A40"},
        ),
    )
    monkeypatch.setattr(
        samga_train,
        "clean_repository_git_sha",
        lambda: events.append("git") or "1" * 40,
    )
    monkeypatch.setattr(
        samga_train,
        "_guard_training_paths",
        lambda _arguments: events.append("paths") or paths,
    )
    monkeypatch.setattr(
        samga_train,
        "preflight_upstream_config",
        lambda _path: events.append("upstream") or SimpleNamespace(),
    )
    monkeypatch.setattr(
        samga_train,
        "load_samga_checkpoint",
        lambda _path, requested_scope: (
            events.append("resume")
            or samga_train.LoadedSAMGACheckpoint(
                payload={
                    "epoch": resume_epoch,
                    "environment": environment,
                },
                sha256=_h(f"resume-{resume_epoch}"),
            )
        ),
    )

    def stop_at_manifest(*_args: object, **_kwargs: object) -> object:
        events.append("manifest")
        raise RuntimeError("manifest sentinel")

    monkeypatch.setattr(
        samga_train,
        "load_development_manifest_identity",
        stop_at_manifest,
    )

    if expect_rejected:
        with pytest.raises(
            ValueError,
            match=r"resume.*epoch.*(?:51|60)|fresh.*recovery",
        ):
            samga_train.run_training(SimpleNamespace(device="cuda", subject=8))
        assert events == ["git", "paths", "upstream", "resume"]
    else:
        with pytest.raises(RuntimeError, match="manifest sentinel"):
            samga_train.run_training(SimpleNamespace(device="cuda", subject=8))
        assert events == ["git", "paths", "upstream", "resume", "manifest"]


def test_train_main_dispatches_the_production_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[object] = []
    monkeypatch.setattr(
        samga_train,
        "run_training",
        lambda arguments: observed.append(arguments) or 0,
    )
    result = samga_train.main(
        [
            "--scope",
            "train",
            "--validation-scope",
            "val-dev",
            "--stage",
            "0",
            "--subject",
            "8",
            "--seed",
            "42",
            "--resume",
            "none",
            "--config",
            str(tmp_path / "config.json"),
            "--manifest",
            str(tmp_path / "sub-08_protocol.json"),
            "--feature-cache",
            str(tmp_path / "features.npy"),
            "--output-dir",
            str(tmp_path / "run"),
            "--max-train-steps",
            "1",
        ]
    )
    assert result == 0
    assert len(observed) == 1
    assert observed[0].subject == 8
    assert observed[0].validation_scope == "val-dev"


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--scope", "test"),
        ("--validation-scope", "val-confirm"),
        ("--stage", "3"),
        ("--subject", "11"),
        ("--resume", ""),
    ],
)
def test_train_parser_rejects_unsealed_or_implicit_inputs(
    tmp_path: Path,
    flag: str,
    value: str,
) -> None:
    argv = [
        "--scope",
        "train",
        "--validation-scope",
        "val-dev",
        "--stage",
        "0",
        "--subject",
        "8",
        "--seed",
        "42",
        "--resume",
        "none",
        "--config",
        str(tmp_path / "config.json"),
        "--manifest",
        str(tmp_path / "sub-08_protocol.json"),
        "--feature-cache",
        str(tmp_path / "features.npy"),
        "--output-dir",
        str(tmp_path / "run"),
    ]
    index = argv.index(flag)
    argv[index + 1] = value
    with pytest.raises(SystemExit):
        samga_train.parse_arguments(argv)


@pytest.mark.parametrize(
    "path",
    [
        "formal/run",
        "formal-test/run",
        "test/run",
        "sub-08_test.json",
        "val-confirm/run",
    ],
)
def test_development_path_guard_rejects_sealed_names(
    tmp_path: Path,
    path: str,
) -> None:
    with pytest.raises(ValueError, match="sealed|test|formal|confirm"):
        samga_train.require_development_path(tmp_path / path, "probe")


def test_checkpoint_metadata_guard_accepts_realistic_optimizer_state() -> None:
    samga_train._reject_sealed_checkpoint_metadata(
        {"optimizer_state_dict": _realistic_adamw_state_dict()}
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"runtime_state": {0: {}}},
        {"optimizer_state_dict": {"state": {-1: {}}}},
        {"optimizer_state_dict": {"state": {True: {}}}},
    ],
)
def test_checkpoint_metadata_guard_rejects_noncanonical_integer_keys(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="keys must be strings"):
        samga_train._reject_sealed_checkpoint_metadata(payload)


def test_checkpoint_metadata_guard_scans_values_below_optimizer_parameter_ids() -> (
    None
):
    optimizer_state = _realistic_adamw_state_dict()
    parameter_id = next(iter(optimizer_state["state"]))
    optimizer_state["state"][parameter_id]["source"] = "formal-test"

    with pytest.raises(PermissionError, match="sealed"):
        samga_train._reject_sealed_checkpoint_metadata(
            {"optimizer_state_dict": optimizer_state}
        )


def test_stage_schedule_is_exact() -> None:
    assert samga_train.mmd_weight_for_epoch(1) == pytest.approx(0.9)
    assert samga_train.mmd_weight_for_epoch(20) == pytest.approx(0.5)
    assert samga_train.mmd_weight_for_epoch(21) == 0.0
    assert samga_train.learning_rate_for_epoch(1) == pytest.approx(1e-4)
    assert samga_train.learning_rate_for_epoch(20) == pytest.approx(1e-4)
    assert samga_train.learning_rate_for_epoch(21) == pytest.approx(5e-5)
    assert samga_train.learning_rate_for_epoch(60) == pytest.approx(5e-5)


def test_checkpoint_roundtrip_restores_full_resume_state(tmp_path: Path) -> None:
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    generator = torch.Generator().manual_seed(123)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    expected_python = random.getstate()
    expected_numpy = np.random.get_state()
    expected_torch = torch.get_rng_state().clone()
    expected_loader = generator.get_state().clone()
    payload = samga_train.build_epoch_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=51,
        global_step=99,
        subject=8,
        seed=42,
        config_sha256=_h("config"),
        schedule_sha256=_h("schedule"),
        trajectory_sha256=_h("trajectory"),
        data_order_sha256=_h("order"),
        generator=generator,
        validation_metrics={"top1": 0.5, "top5": 0.9},
        input_hashes={"manifest_sha256": _h("manifest")},
        environment=_production_environment_binding(),
        effective_batch=512,
    )
    assert payload["optimizer_stage"] == "stage2"
    assert payload["global_step"] == 99
    assert payload["data_order_sha256"] == _h("order")

    random.random()
    np.random.rand()
    torch.rand(2)
    generator.seed()
    restored_step = samga_train.restore_training_checkpoint(
        payload,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        generator=generator,
        expected_subject=8,
        expected_seed=42,
        expected_config_sha256=_h("config"),
        expected_schedule_sha256=_h("schedule"),
        expected_trajectory_sha256=_h("trajectory"),
        expected_data_order_sha256=_h("order"),
    )
    assert restored_step == (51, 99)
    assert random.getstate() == expected_python
    actual_numpy = np.random.get_state()
    assert actual_numpy[0] == expected_numpy[0]
    assert np.array_equal(actual_numpy[1], expected_numpy[1])
    assert actual_numpy[2:] == expected_numpy[2:]
    assert torch.equal(torch.get_rng_state(), expected_torch)
    assert torch.equal(generator.get_state(), expected_loader)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("subject", 7),
        ("seed", 43),
        ("config_sha256", _h("wrong-config")),
        ("schedule_sha256", _h("wrong-schedule")),
        ("trajectory_sha256", _h("wrong-trajectory")),
        ("data_order_sha256", _h("wrong-order")),
    ],
)
def test_resume_rejects_identity_mismatch(
    field: str,
    value: object,
) -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    generator = torch.Generator().manual_seed(1)
    payload = samga_train.build_epoch_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=1,
        global_step=1,
        subject=8,
        seed=42,
        config_sha256=_h("config"),
        schedule_sha256=_h("schedule"),
        trajectory_sha256=_h("trajectory"),
        data_order_sha256=_h("order"),
        generator=generator,
        validation_metrics={"top1": 0.0, "top5": 0.0},
        input_hashes={"manifest_sha256": _h("manifest")},
        environment=_production_environment_binding(),
        effective_batch=512,
    )
    payload[field] = value
    with pytest.raises(ValueError, match="mismatch"):
        samga_train.restore_training_checkpoint(
            payload,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            generator=generator,
            expected_subject=8,
            expected_seed=42,
            expected_config_sha256=_h("config"),
            expected_schedule_sha256=_h("schedule"),
            expected_trajectory_sha256=_h("trajectory"),
            expected_data_order_sha256=_h("order"),
        )


def test_checkpoint_identity_rejects_evaluator_subject_seed_mismatch() -> None:
    payload = {"subject": 8, "seed": 42}
    assert checkpoint_identity(payload, subject=8, seed=42) == (8, 42)
    with pytest.raises(ValueError, match="subject"):
        checkpoint_identity(payload, subject=7, seed=42)
    with pytest.raises(ValueError, match="seed"):
        checkpoint_identity(payload, subject=8, seed=43)


def test_evaluator_parser_requires_locked_val_dev_arguments(
    tmp_path: Path,
) -> None:
    args = parse_evaluate_arguments(
        [
            "--scope",
            "val-dev",
            "--subject",
            "8",
            "--seed",
            "42",
            "--config",
            str(tmp_path / "config.json"),
            "--manifest",
            str(tmp_path / "sub-08_protocol.json"),
            "--feature-cache",
            str(tmp_path / "features.npy"),
            "--checkpoint",
            str(tmp_path / "checkpoint_epoch060.pt"),
            "--output-dir",
            str(tmp_path / "scores"),
        ]
    )
    assert args.scope == "val-dev"
    assert args.subject == 8
    assert args.seed == 42
    assert args.checkpoint.name == "checkpoint_epoch060.pt"


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--scope", "val-confirm"),
        ("--scope", "formal-test"),
        ("--subject", "11"),
        ("--seed", "-1"),
    ],
)
def test_evaluator_parser_rejects_unsealed_or_invalid_arguments(
    tmp_path: Path,
    flag: str,
    value: str,
) -> None:
    argv = [
        "--scope",
        "val-dev",
        "--subject",
        "8",
        "--seed",
        "42",
        "--config",
        str(tmp_path / "config.json"),
        "--manifest",
        str(tmp_path / "sub-08_protocol.json"),
        "--feature-cache",
        str(tmp_path / "features.npy"),
        "--checkpoint",
        str(tmp_path / "checkpoint_epoch060.pt"),
        "--output-dir",
        str(tmp_path / "scores"),
    ]
    position = argv.index(flag)
    argv[position + 1] = value
    with pytest.raises(SystemExit):
        parse_evaluate_arguments(argv)


def _manifest_identity(tmp_path: Path) -> ManifestIdentity:
    return ManifestIdentity(
        path=tmp_path / "sub-08_protocol.json",
        subject=8,
        manifest_sha256=_h("manifest"),
        protocol_sha256=_h("protocol"),
        records_sha256=_h("records"),
        source_manifest_sha256=_h("source-manifest"),
        source_payload_path=tmp_path / "sub-08" / "train.pt",
        source_payload_sha256=_h("source-payload"),
        source_payload_byte_count=123,
        train_role_sha256=_h("train-role"),
        val_dev_role_sha256=_h("val-dev-role"),
        train_ordered_ids=("concept-a", "concept-b"),
        val_dev_ordered_ids=("image-a", "image-b"),
        train_ordered_ids_sha256=samga_train.ordered_ids_sha256(
            ["concept-a", "concept-b"]
        ),
        val_dev_ordered_ids_sha256=samga_train.ordered_ids_sha256(
            ["image-a", "image-b"]
        ),
    )


def _checkpoint_payload(
    tmp_path: Path,
    *,
    validation_scope: str = "val-dev",
) -> dict[str, object]:
    if validation_scope not in ("val-dev", "none"):
        raise ValueError("invalid fixture validation scope")
    manifest = _manifest_identity(tmp_path)
    ordered_ids = list(manifest.train_ordered_ids)
    if validation_scope == "val-dev":
        ordered_ids.extend(manifest.val_dev_ordered_ids)
    input_hashes = {
        "cache_sha256": _h("cache"),
        "checkpoint_sha256": _h("no-initial-checkpoint"),
        "manifest_sha256": manifest.manifest_sha256,
        "model_sha256": _h("model"),
        "ordered_ids_sha256": samga_train.ordered_ids_sha256(ordered_ids),
        "protocol_sha256": manifest.protocol_sha256,
        "records_sha256": manifest.records_sha256,
        "source_manifest_sha256": manifest.source_manifest_sha256,
        "source_payload_byte_count_sha256": samga_train.sha256_json(
            manifest.source_payload_byte_count
        ),
        "source_payload_path_sha256": samga_train.sha256_json(
            str(manifest.source_payload_path)
        ),
        "source_payload_sha256": manifest.source_payload_sha256,
        "train_ordered_ids_sha256": manifest.train_ordered_ids_sha256,
        "train_role_sha256": manifest.train_role_sha256,
        "val_dev_ordered_ids_sha256": (manifest.val_dev_ordered_ids_sha256),
        "val_dev_role_sha256": manifest.val_dev_role_sha256,
    }
    input_bundle_sha256 = samga_train.sha256_json(dict(sorted(input_hashes.items())))
    run_key = (
        "stage0__internvit_baseline_v1__sub-08__seed-42__"
        f"config-{_h('config')}__inputs-{input_bundle_sha256}"
    )
    candidate_body = {
        "schema_version": 1,
        "config_id": "internvit_baseline_v1",
        "stage": "stage0",
        "subject": 8,
        "seed": 42,
        "baseline_config_sha256": _h("baseline-config"),
        "stage2_config_sha256": None,
        "semantic_config_sha256": _h("config"),
        "input_bundle_sha256": input_bundle_sha256,
        "run_key": run_key,
        "layernorm_config_id": "s2-layernorm-off",
        "whitening_config_id": "s2-whitening-off",
        "preprojector_config_id": "s2-preproj-shared",
        "adapter_kind": "identity",
        "adapter_rank": None,
        "adapter_lr_ratio": None,
        "whitening_payload": None,
        "full_task_initialization_sha256": _h("full-init"),
        "shared_parameter_intersection_name": "fixture-shared",
        "shared_parameter_intersection_sha256": _h("shared-init"),
        "architecture_specific_initialization_sha256": _h("specific-init"),
        "data_order_sha256": _h("order"),
        "trajectory_sha256": _h("trajectory"),
    }
    candidate_spec = {
        **candidate_body,
        "candidate_spec_sha256": samga_train.sha256_json(candidate_body),
    }
    run_manifest = samga_train.build_run_manifest(
        stage=0,
        subject=8,
        seed=42,
        config_id="internvit_baseline_v1",
        config_sha256=_h("config"),
        protocol_sha256=manifest.protocol_sha256,
        cache_sha256=input_hashes["cache_sha256"],
        git_sha="1" * 40,
        upstream_sha=samga_train.PINNED_UPSTREAM_SHA,
        data_order_sha256=_h("order"),
        candidate_spec_sha256=candidate_spec["candidate_spec_sha256"],
        run_key=run_key,
    )
    generator = torch.Generator().manual_seed(123)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda _: 1.0,
    )
    if validation_scope == "none":
        validation_metrics = {
            "performed": False,
            "validation_scope": "none",
        }
    else:
        validation_metrics = {
            "query_count": 2,
            "gallery_count": 2,
            "top1_count": 1,
            "top5_count": 2,
            "top1_rate": 0.5,
            "top5_rate": 1.0,
            "router_mode": "global",
            "similarity": "cosine",
        }
    payload = samga_train.build_epoch_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=1,
        global_step=1,
        subject=8,
        seed=42,
        config_sha256=_h("config"),
        schedule_sha256=samga_train.SCHEDULE_SHA256,
        trajectory_sha256=_h("trajectory"),
        data_order_sha256=_h("order"),
        generator=generator,
        validation_metrics=validation_metrics,
        input_hashes=input_hashes,
        environment=_production_environment_binding(),
        effective_batch=512,
        sampler_state={
            "schema_version": 1,
            "dataset_size": 1024,
            "seed": 42,
            "epoch": 1,
            "position": 512,
            "order": list(range(1024)),
        },
        run_manifest=run_manifest,
        candidate_spec=candidate_spec,
    )
    payload["runtime_state"] = {
        "schema_version": 1,
        "epoch_complete": False,
        "next_epoch": 1,
        "resume_source_checkpoint_sha256": None,
        "optimizer_base_lr": 1e-4,
        "iterator_generator_state": generator.get_state(),
        "snapshot_epochs": [],
        "required_retained_epochs": list(range(51, 61)),
    }
    payload["retention"] = {
        "policy": "retain_exact_epochs_51_through_60",
        "required_epochs": list(range(51, 61)),
        "retain_for_averaging": False,
    }
    return payload


def test_samga_checkpoint_bundle_is_exclusive_typed_and_reloadable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "checkpoint_epoch001_step00000001.pt"
    payload = _checkpoint_payload(tmp_path)
    written_leaves: list[str] = []
    real_write = samga_train._write_relative_exclusive

    def recording_write(
        parent: object,
        name: str,
        raw: bytes,
        *,
        context: str,
    ) -> tuple[int, int, int]:
        written_leaves.append(name)
        return real_write(parent, name, raw, context=context)

    monkeypatch.setattr(
        samga_train,
        "_write_relative_exclusive",
        recording_write,
    )
    digest = samga_train.save_samga_checkpoint(
        path,
        payload,
        _manifest_identity(tmp_path),
    )
    assert len(digest) == 64
    assert path.is_file()
    assert samga_train.samga_checkpoint_sidecar(path).is_file()
    assert path.name not in written_leaves
    assert samga_train.samga_checkpoint_sidecar(path).name not in written_leaves
    assert all(name.startswith(".") for name in written_leaves)
    assert not tuple(tmp_path.glob(".*.tmp-*"))

    loaded = samga_train.load_samga_checkpoint(
        path,
        requested_scope="train",
    )
    assert loaded.sha256 == digest
    assert loaded.payload["subject"] == 8
    assert loaded.payload["seed"] == 42
    assert loaded.payload["model_state_sha256"] == payload["model_state_sha256"]
    assert set(loaded.payload) == set(payload)
    assert "scope" not in loaded.payload
    _validate_checkpoint_schema(loaded.payload, "resume checkpoint")

    with pytest.raises(FileExistsError):
        samga_train.save_samga_checkpoint(
            path,
            payload,
            _manifest_identity(tmp_path),
        )


def test_train_only_checkpoint_round_trip_records_only_train_scope(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoint_epoch001_step00000001.pt"
    payload = _checkpoint_payload(
        tmp_path, validation_scope="none"
    )
    samga_train.save_samga_checkpoint(
        path, payload, _manifest_identity(tmp_path)
    )
    serialized = torch.load(path, map_location="cpu", weights_only=True)
    assert serialized["validation_scope"] == "none"
    assert serialized["observed_scopes"] == ["train"]
    envelope = json.loads(
        samga_train.samga_checkpoint_sidecar(path).read_text(encoding="utf-8")
    )
    assert envelope["metadata"]["observed_scopes"] == ["train"]
    assert envelope["metadata"]["val_dev_ordered_ids"] == []
    assert [
        record["role"] for record in envelope["metadata"]["source_records"]
    ] == ["train"]
    loaded = samga_train.load_samga_checkpoint(path, requested_scope="train")
    assert loaded.payload["validation_metrics"] == {
        "performed": False,
        "validation_scope": "none",
    }


def test_samga_checkpoint_loader_rejects_resigned_semantic_conflict(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoint_epoch001_step00000001.pt"
    samga_train.save_samga_checkpoint(
        path,
        _checkpoint_payload(tmp_path),
        _manifest_identity(tmp_path),
    )
    sidecar = samga_train.samga_checkpoint_sidecar(path)
    envelope = json.loads(sidecar.read_text(encoding="utf-8"))
    envelope["provenance"]["protocol_sha256"] = _h("other-protocol")
    envelope["provenance_sha256"] = samga_train.sha256_json(envelope["provenance"])
    sidecar.write_text(
        json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="protocol_sha256 binding mismatch"):
        samga_train.load_samga_checkpoint(path, requested_scope="train")


def _training_identities() -> TrainingIdentities:
    return TrainingIdentities(
        data_order_sha256=_h("data-order"),
        trajectory_sha256=_h("trajectory"),
        full_task_initialization_sha256=_h("full-init"),
        shared_parameter_intersection_name="locked_shared_parameter_intersection",
        shared_parameter_intersection_sha256=_h("shared-init"),
        architecture_specific_initialization_sha256=_h("specific-init"),
    )


def test_candidate_spec_canonically_binds_runtime_identities() -> None:
    spec = samga_train.build_candidate_spec(
        stage=0,
        config_id="internvit_baseline_v1",
        subject=8,
        seed=42,
        baseline_config_sha256=_h("baseline-config"),
        stage2_config_sha256=None,
        semantic_config_sha256=_h("semantic-config"),
        input_bundle_sha256=_h("input-bundle"),
        run_key="stage0__internvit_baseline_v1__sub-08__seed-42",
        layernorm_config_id="s2-layernorm-off",
        whitening_config_id="s2-whitening-off",
        preprojector_config_id="s2-preproj-shared",
        adapter_kind="identity",
        adapter_rank=None,
        adapter_lr_ratio=None,
        whitening=None,
        identities=_training_identities(),
    )
    body = {key: value for key, value in spec.items() if key != "candidate_spec_sha256"}
    assert spec["stage"] == "stage0"
    assert spec["whitening_payload"] is None
    assert spec["data_order_sha256"] == _h("data-order")
    assert spec["candidate_spec_sha256"] == samga_train.sha256_json(body)


def test_run_manifest_is_canonical_and_binds_all_inputs() -> None:
    first = samga_train.build_run_manifest(
        stage=0,
        subject=8,
        seed=42,
        config_id="internvit_baseline_v1",
        config_sha256=_h("config"),
        protocol_sha256=_h("protocol"),
        cache_sha256=_h("cache"),
        git_sha="a" * 40,
        upstream_sha=samga_train.PINNED_UPSTREAM_SHA,
        data_order_sha256=_h("order"),
        candidate_spec_sha256=_h("candidate-spec"),
        run_key="stage0__internvit_baseline_v1__sub-08__seed-42",
    )
    second = samga_train.build_run_manifest(
        **{key: value for key, value in first.items() if key != "run_manifest_sha256"}
    )
    assert first == second
    body = {key: value for key, value in first.items() if key != "run_manifest_sha256"}
    expected = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert first["run_manifest_sha256"] == expected


def test_train_only_input_hashes_bind_only_train_ids(tmp_path: Path) -> None:
    manifest = _manifest_identity(tmp_path)
    config = SimpleNamespace(
        cache_sha256=_h("cache"),
        model_sha256=_h("model"),
    )
    development = samga_train._build_input_hashes(manifest, config)
    train_only = samga_train._build_input_hashes(
        manifest, config, validation_scope="none"
    )

    assert development["ordered_ids_sha256"] == samga_train.ordered_ids_sha256(
        [*manifest.train_ordered_ids, *manifest.val_dev_ordered_ids]
    )
    assert train_only["ordered_ids_sha256"] == manifest.train_ordered_ids_sha256
    assert development["ordered_ids_sha256"] != train_only["ordered_ids_sha256"]


def test_cached_dataset_factory_reuses_each_verified_role(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest_identity(tmp_path)
    paths = samga_train.TrainingPaths(
        config=tmp_path / "config.json",
        manifest=manifest.path,
        feature_cache=tmp_path / "features.npy",
        output_dir=tmp_path / "run",
        stage2_config=None,
        whitening_artifact=None,
        resume_checkpoint=None,
    )
    train_dataset = object()
    validation_dataset = object()
    builds: list[tuple[str, int]] = []
    verifications: list[tuple[object, str, str]] = []

    def build_dataset(
        received_paths: samga_train.TrainingPaths,
        received_manifest: ManifestIdentity,
        *,
        scope: str,
        seed: int,
    ) -> object:
        assert received_paths is paths
        assert received_manifest is manifest
        builds.append((scope, seed))
        assert scope == "val-dev"
        return validation_dataset

    def verify_dataset(
        dataset: object,
        *,
        manifest: ManifestIdentity,
        scope: str,
        cache_sha256: str,
    ) -> None:
        assert manifest is not None
        verifications.append((dataset, scope, cache_sha256))

    monkeypatch.setattr(
        samga_train,
        "_development_dataset",
        build_dataset,
    )
    monkeypatch.setattr(
        samga_train,
        "_verify_development_dataset",
        verify_dataset,
    )
    factory, datasets = samga_train._cached_dataset_factory(
        paths,
        manifest,
        seed=42,
        cache_sha256=_h("cache"),
        train_dataset=train_dataset,
    )
    common = {
        "manifest_path": manifest.path,
        "seed": 42,
        "selected_channels": samga_train.POSTERIOR_CHANNELS,
        "feature_cache": paths.feature_cache,
    }

    assert (
        factory(
            **common,
            scope="train",
            smooth_probability=0.3,
        )
        is train_dataset
    )
    assert (
        factory(
            **common,
            scope="val-dev",
            smooth_probability=0.0,
        )
        is validation_dataset
    )
    assert (
        factory(
            **common,
            scope="val-dev",
            smooth_probability=0.0,
        )
        is validation_dataset
    )
    assert datasets == {
        "train": train_dataset,
        "val-dev": validation_dataset,
    }
    assert builds == [("val-dev", 42)]
    assert verifications == [(validation_dataset, "val-dev", _h("cache"))]

    with pytest.raises(ValueError, match="arguments"):
        factory(
            **common,
            scope="val-dev",
            smooth_probability=0.3,
        )


def test_resolved_candidate_hashes_the_complete_runtime_environment() -> None:
    common = {
        "stage": 0,
        "config_id": "internvit_baseline_v1",
        "subject": 8,
        "seed": 42,
        "baseline_config_sha256": _h("baseline-config"),
        "stage2_config_sha256": None,
        "layernorm_config_id": "s2-layernorm-off",
        "whitening_config_id": "s2-whitening-off",
        "preprojector_config_id": "s2-preproj-shared",
        "adapter_kind": "identity",
        "adapter_rank": None,
        "adapter_lr_ratio": None,
        "whitening_payload_sha256": None,
    }
    first_environment = _production_environment_binding()
    first = samga_train.build_resolved_candidate_payload(
        **common,
        environment_binding=first_environment,
    )
    changed_semantic = dict(PINNED_SEMANTIC_ENVIRONMENT)
    changed_semantic["numpy"] = "1.26.5"
    changed_environment = build_environment_binding(
        changed_semantic,
        PRODUCTION_RUNTIME_CONTRACT,
    )
    changed = samga_train.build_resolved_candidate_payload(
        **common,
        environment_binding=changed_environment,
    )
    train_only = samga_train.build_resolved_candidate_payload(
        **common,
        environment_binding=first_environment,
        validation_scope="none",
    )

    assert first["runtime"]["environment"] == first_environment
    assert changed["runtime"]["environment"] == changed_environment
    assert samga_train.sha256_json(first) != samga_train.sha256_json(changed)
    assert train_only["runtime"]["force_global_validation"] is False
    assert samga_train.sha256_json(first) != samga_train.sha256_json(train_only)
    assert first["runtime"]["environment"]["runtime_contract"]["device"] == "cuda:0"
    assert (
        first["runtime"]["environment"]["semantic_environment"]["HF_DATASETS_OFFLINE"]
        == "1"
    )


@pytest.mark.parametrize(
    ("field", "changed_value"),
    [
        ("stage1_learning_rate", 2e-4),
        ("stage2_learning_rate", 1e-5),
        ("optimizer", "different-optimizer"),
        ("betas", [0.8, 0.999]),
        ("eps", 1e-7),
        ("weight_decay", 0.01),
        ("amsgrad", True),
        ("maximize", True),
        ("foreach", False),
        ("capturable", True),
        ("differentiable", True),
        ("fused", False),
        ("scheduler", "different-scheduler"),
    ],
)
def test_every_optimizer_recipe_field_changes_candidate_and_run_identity(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    changed_value: object,
) -> None:
    assert field in samga_train.SCHEDULE
    common = {
        "stage": 0,
        "config_id": "internvit_baseline_v1",
        "subject": 8,
        "seed": 42,
        "baseline_config_sha256": _h("baseline-config"),
        "stage2_config_sha256": None,
        "layernorm_config_id": "s2-layernorm-off",
        "whitening_config_id": "s2-whitening-off",
        "preprojector_config_id": "s2-preproj-shared",
        "adapter_kind": "identity",
        "adapter_rank": None,
        "adapter_lr_ratio": None,
        "whitening_payload_sha256": None,
        "environment_binding": _production_environment_binding(),
    }
    inputs = {
        "cache_sha256": _h("cache"),
        "checkpoint_sha256": _h("checkpoint"),
        "manifest_sha256": _h("manifest"),
        "model_sha256": _h("model"),
    }
    protocol = samga_train.ProtocolConfig.from_path(samga_train._PROTOCOL_CONFIG_PATH)
    baseline_candidate = samga_train.build_resolved_candidate_payload(**common)
    baseline = samga_train.resolve_run_config(
        protocol,
        baseline_candidate,
        inputs,
    )

    changed_schedule = copy.deepcopy(samga_train.SCHEDULE)
    changed_schedule[field] = changed_value
    changed_schedule_sha256 = samga_train.sha256_json(changed_schedule)
    assert changed_schedule_sha256 != samga_train.SCHEDULE_SHA256
    monkeypatch.setattr(
        samga_train,
        "SCHEDULE_SHA256",
        changed_schedule_sha256,
    )
    changed_candidate = samga_train.build_resolved_candidate_payload(**common)
    changed = samga_train.resolve_run_config(
        protocol,
        changed_candidate,
        inputs,
    )

    assert changed_candidate["runtime"]["schedule_sha256"] == changed_schedule_sha256
    assert changed.semantic_config_sha256 != baseline.semantic_config_sha256
    assert changed.run_key != baseline.run_key


def test_runtime_manifest_metadata_contains_full_environment_contract_and_evidence() -> (
    None
):
    environment = _production_environment_binding()
    evidence = {
        "accelerator_name": "NVIDIA A40",
        "attention_backend": "naive",
    }
    metadata = samga_train._runtime_manifest_metadata(
        SimpleNamespace(
            environment_binding=environment,
            contract=PRODUCTION_RUNTIME_CONTRACT,
            evidence=evidence,
        )
    )

    assert metadata["environment"] == environment
    assert metadata["runtime_contract"] == PRODUCTION_RUNTIME_CONTRACT
    assert metadata["runtime_evidence"] == evidence
    assert (
        metadata["semantic_environment_sha256"]
        == environment["semantic_environment_sha256"]
    )
    assert metadata["runtime_contract_sha256"] == environment["runtime_contract_sha256"]


@pytest.mark.parametrize(
    ("completed", "expected_stage"),
    [(False, "training_smoke/in_loop"), (True, "stage2")],
)
def test_samga_in_loop_score_round_trip_distinguishes_partial_training(
    tmp_path: Path,
    completed: bool,
    expected_stage: str,
) -> None:
    checkpoint_sha256 = _h("actual-published-checkpoint")
    metadata = samga_train._build_in_loop_score_metadata(
        completed=completed,
        global_step=1 if not completed else 120,
        planned_steps=120,
        checkpoint_sha256=checkpoint_sha256,
        config_sha256=_h("config"),
        git_sha="1" * 40,
        protocol_sha256=_h("protocol"),
        seed=42,
        source_records=[{"record_id": "development-record"}],
        stage=2,
        subject=8,
    )
    output = tmp_path / ("partial" if not completed else "complete")
    samga_train.ScoreArtifact.save(
        output,
        np.eye(2, dtype=np.float32),
        ("a", "b"),
        ("a", "b"),
        metadata,
    )
    loaded = samga_train.ScoreArtifact.load(
        output,
        allowed_scopes={"val-dev"},
    )

    assert loaded.provenance["checkpoint_sha256"] == checkpoint_sha256
    assert loaded.provenance["stage"] == expected_stage
    if completed:
        assert "training_complete" not in loaded.provenance
        assert "global_step" not in loaded.provenance
        assert "planned_steps" not in loaded.provenance
    else:
        assert loaded.provenance["training_complete"] is False
        assert loaded.provenance["global_step"] == 1
        assert loaded.provenance["planned_steps"] == 120
