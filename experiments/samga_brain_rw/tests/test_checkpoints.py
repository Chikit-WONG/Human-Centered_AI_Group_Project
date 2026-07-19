from __future__ import annotations

import copy
import hashlib
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import samga_brain_rw.checkpoints as checkpoints_module

from samga_brain_rw.hashing import (
    canonical_json_bytes,
    ordered_ids_sha256,
    sha256_json,
)
from samga_brain_rw.checkpoints import (
    AVERAGING_CANDIDATES,
    average_state_dicts,
    build_averaged_checkpoint,
    hash_state_dict,
    swa_state_dicts,
)


def _h(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_checkpoint(
    path: Path,
    *,
    epoch: int,
    subject: int = 1,
    seed: int = 42,
    config_sha256: str | None = None,
    schedule_sha256: str | None = None,
    optimizer_stage: str = "stage2",
    trajectory_sha256: str | None = None,
    state: dict[str, torch.Tensor] | None = None,
) -> Path:
    payload = {
        "schema_version": 1,
        "payload_type": "samga_brain_rw.epoch_checkpoint",
        "epoch": epoch,
        "subject": subject,
        "seed": seed,
        "config_sha256": config_sha256 or _h("config"),
        "schedule_sha256": schedule_sha256 or _h("schedule"),
        "optimizer_stage": optimizer_stage,
        "trajectory_sha256": trajectory_sha256 or _h("trajectory"),
        "model_state_dict": state
        or {
            "weight": torch.tensor([float(epoch), float(epoch + 2)]),
            "counter": torch.tensor([7], dtype=torch.int64),
        },
        "optimizer_state_dict": {"must_not_be_averaged": epoch},
    }
    torch.save(payload, path)
    return _make_typed_checkpoint(path)


def _window(tmp_path: Path, start: int = 56) -> list[Path]:
    return [
        _write_checkpoint(tmp_path / f"checkpoint_epoch{epoch:03d}.pt", epoch=epoch)
        for epoch in range(start, 61)
    ]


def _make_typed_checkpoint(path: Path) -> Path:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload.update(
        {
            "global_step": int(payload["epoch"]) * 10,
            "data_order_sha256": _h("data-order"),
            "model_state_sha256": hash_state_dict(payload["model_state_dict"]),
            "scheduler_state_dict": {},
            "python_rng_state": [1, 2, 3],
            "numpy_rng_state": {},
            "torch_rng_state": torch.tensor([1, 2, 3], dtype=torch.uint8),
            "cuda_rng_states": [],
            "loader_generator_state": torch.tensor([4, 5], dtype=torch.uint8),
            "sampler_state_dict": {},
            "validation_metrics": {},
            "input_hashes": {},
            "effective_batch": {},
            "environment": {},
            "run_manifest": {},
            "candidate_spec": {},
            "runtime_state": {},
            "retention": {"retain_for_averaging": True},
            "scope": "train",
            "validation_scope": "val-dev",
            "observed_scopes": ["train", "val-dev"],
        }
    )
    torch.save(payload, path)
    payload_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    source_records = [
        {
            "manifest_sha256": _h("manifest"),
            "records_sha256": _h("records"),
            "role": role,
            "role_payload_sha256": _h(role),
            "source_manifest_sha256": _h("source-manifest"),
            "source_payload_sha256": _h("source-payload"),
        }
        for role in ("train", "val-dev")
    ]
    ordered_ids = ["train:0", "val-dev:0"]
    provenance = {
        "config_sha256": payload["config_sha256"],
        "manifest_sha256": _h("manifest"),
        "protocol_sha256": _h("protocol"),
        "seed": payload["seed"],
        "subject": payload["subject"],
    }
    metadata = {
        "complete": True,
        "observed_scopes": ["train", "val-dev"],
        "ordered_ids": ordered_ids,
        "retention": payload["retention"],
        "source_records": source_records,
    }
    envelope = {
        "schema_version": 1,
        "payload_type": "samga_brain_rw.epoch_checkpoint",
        "scope": "train",
        "source_records_sha256": sha256_json(source_records),
        "ordered_ids_sha256": ordered_ids_sha256(ordered_ids),
        "payload_sha256": payload_sha256,
        "provenance": provenance,
        "provenance_sha256": sha256_json(provenance),
        "metadata": metadata,
        "metadata_sha256": sha256_json(metadata),
    }
    path.with_suffix(path.suffix + ".meta.json").write_bytes(
        canonical_json_bytes(envelope) + b"\n"
    )
    return path


def _typed_window(
    tmp_path: Path,
    start: int = 56,
    *,
    seed: object = 42,
) -> list[Path]:
    return [
        _write_checkpoint(
            tmp_path / f"typed_checkpoint_epoch{epoch:03d}.pt",
            epoch=epoch,
            seed=seed,  # type: ignore[arg-type]
        )
        for epoch in range(start, 61)
    ]


def _install_intermediate_directory_swap(
    monkeypatch: pytest.MonkeyPatch,
    *,
    safe_directory: Path,
    sealed_directory: Path,
    target: Path,
) -> tuple[dict[str, bool], Path]:
    original_open = os.open
    backup = safe_directory.with_name(f"{safe_directory.name}-original")
    state = {"swapped": False}

    def swap() -> None:
        safe_directory.rename(backup)
        safe_directory.symlink_to(sealed_directory, target_is_directory=True)
        state["swapped"] = True

    def swapping_open(
        path: str | bytes | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        raw = os.fspath(path)
        if not state["swapped"] and dir_fd is None and Path(raw) == target:
            # The pre-fix implementation opens the full path after lstat.
            swap()
        if dir_fd is None:
            descriptor = original_open(path, flags, mode)
        else:
            descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        if (
            not state["swapped"]
            and dir_fd is not None
            and raw == safe_directory.name
            and flags & os.O_DIRECTORY
        ):
            # Secure traversal has opened the directory descriptor; swap its
            # pathname before the leaf is opened to exercise identity recheck.
            swap()
        return descriptor

    monkeypatch.setattr(os, "open", swapping_open)
    return state, backup


def test_exact_candidate_registry_is_locked() -> None:
    assert AVERAGING_CANDIDATES == {
        "s2-avg-last5": ("arithmetic", (56, 57, 58, 59, 60)),
        "s2-avg-last10": (
            "arithmetic",
            (51, 52, 53, 54, 55, 56, 57, 58, 59, 60),
        ),
        "s2-swa-last5": ("swa", (56, 57, 58, 59, 60)),
        "s2-swa-last10": (
            "swa",
            (51, 52, 53, 54, 55, 56, 57, 58, 59, 60),
        ),
    }


def test_averaging_rejects_checkpoint_without_typed_sidecar(
    tmp_path: Path,
) -> None:
    paths = _window(tmp_path)
    paths[0].with_suffix(paths[0].suffix + ".meta.json").unlink()
    with pytest.raises(ValueError, match="envelope|sidecar|typed"):
        average_state_dicts(paths)


def test_averaging_accepts_seed_zero_from_typed_checkpoint_bundle(
    tmp_path: Path,
) -> None:
    averaged = average_state_dicts(_typed_window(tmp_path, seed=0))
    assert torch.equal(averaged["weight"], torch.tensor([58.0, 60.0]))


def test_build_averaged_checkpoint_accepts_seed_zero(
    tmp_path: Path,
) -> None:
    result = build_averaged_checkpoint(
        _typed_window(tmp_path, seed=0),
        candidate_id="s2-avg-last5",
    )
    assert result["seed"] == 0


@pytest.mark.parametrize("seed", [-1, True, False])
def test_averaging_rejects_negative_or_boolean_seed(
    tmp_path: Path,
    seed: object,
) -> None:
    with pytest.raises(ValueError, match="seed"):
        average_state_dicts(_typed_window(tmp_path, seed=seed))


def test_averaging_rejects_checkpoint_payload_hash_tamper(
    tmp_path: Path,
) -> None:
    paths = _typed_window(tmp_path)
    payload = torch.load(paths[-1], map_location="cpu", weights_only=True)
    payload["model_state_dict"]["weight"][0] += 1
    torch.save(payload, paths[-1])

    with pytest.raises(ValueError, match="payload.*SHA-256|digest|hash"):
        average_state_dicts(paths)


def test_averaging_rejects_checkpoint_sidecar_tamper(
    tmp_path: Path,
) -> None:
    paths = _typed_window(tmp_path)
    sidecar = paths[-1].with_suffix(paths[-1].suffix + ".meta.json")
    document = json.loads(sidecar.read_text(encoding="utf-8"))
    document["provenance"]["seed"] = 0
    sidecar.write_bytes(canonical_json_bytes(document) + b"\n")

    with pytest.raises(ValueError, match="provenance.*SHA-256|envelope|hash"):
        average_state_dicts(paths)


def test_arithmetic_average_uses_floating_state_only(tmp_path: Path) -> None:
    paths = _window(tmp_path)
    averaged = average_state_dicts(paths)
    assert torch.equal(averaged["weight"], torch.tensor([58.0, 60.0]))
    assert averaged["weight"].dtype == torch.float32
    assert torch.equal(averaged["counter"], torch.tensor([7]))
    assert "must_not_be_averaged" not in averaged


def test_swa_is_updated_once_per_locked_epoch_and_matches_arithmetic(
    tmp_path: Path,
) -> None:
    paths = _window(tmp_path, 51)
    arithmetic = average_state_dicts(paths)
    swa = swa_state_dicts(paths)
    assert set(swa) == set(arithmetic)
    for key in arithmetic:
        assert torch.equal(swa[key], arithmetic[key])
    assert hash_state_dict(swa) == hash_state_dict(arithmetic)


def test_state_hash_accepts_scalar_tensor() -> None:
    first = hash_state_dict({"temperature": torch.tensor(0.07)})
    second = hash_state_dict({"temperature": torch.tensor(0.07)})
    assert first == second


def test_swa_preserves_real_float32_averaged_model_update_semantics(
    tmp_path: Path,
) -> None:
    values = (
        0.3455841839313507,
        0.8216181397438049,
        0.3304370641708374,
        -1.3031572103500366,
        0.9053558707237244,
    )
    paths = [
        _write_checkpoint(
            tmp_path / f"checkpoint_epoch{epoch:03d}.pt",
            epoch=epoch,
            state={
                "weight": torch.tensor([value], dtype=torch.float32),
                "counter": torch.tensor([7], dtype=torch.int64),
            },
        )
        for epoch, value in zip(range(56, 61), values, strict=True)
    ]
    carrier = torch.nn.Linear(1, 1, bias=False)
    expected = torch.optim.swa_utils.AveragedModel(carrier)
    for value in values:
        with torch.no_grad():
            carrier.weight.fill_(value)
        expected.update_parameters(carrier)

    swa = swa_state_dicts(paths)
    arithmetic = average_state_dicts(paths)
    assert torch.equal(swa["weight"], expected.module.weight.reshape(1))
    assert not torch.equal(swa["weight"], arithmetic["weight"])

    built = build_averaged_checkpoint(paths, candidate_id="s2-swa-last5")
    assert built["alias_of"] is None


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(config_sha256=_h("other")), "config"),
        (lambda payload: payload.update(subject=5), "subject"),
        (lambda payload: payload.update(seed=43), "seed"),
        (lambda payload: payload.update(schedule_sha256=_h("other")), "schedule"),
        (lambda payload: payload.update(optimizer_stage="stage1"), "optimizer stage"),
        (lambda payload: payload.update(trajectory_sha256=_h("other")), "trajectory"),
    ],
)
def test_rejects_mismatched_checkpoint_identity(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    paths = _window(tmp_path)
    payload = torch.load(paths[-1], map_location="cpu", weights_only=False)
    mutation(payload)  # type: ignore[operator]
    torch.save(payload, paths[-1])
    _make_typed_checkpoint(paths[-1])
    with pytest.raises(ValueError, match=message):
        average_state_dicts(paths)


@pytest.mark.parametrize(
    ("state", "message"),
    [
        ({"different": torch.ones(2)}, "key"),
        (
            {"weight": torch.ones(3), "counter": torch.tensor([7])},
            "shape",
        ),
        (
            {
                "weight": torch.ones(2, dtype=torch.float64),
                "counter": torch.tensor([7]),
            },
            "dtype",
        ),
        (
            {"weight": torch.ones(2), "counter": torch.tensor([8])},
            "non-floating",
        ),
    ],
)
def test_rejects_state_contract_mismatch(
    tmp_path: Path,
    state: dict[str, torch.Tensor],
    message: str,
) -> None:
    paths = _window(tmp_path)
    payload = torch.load(paths[-1], map_location="cpu", weights_only=False)
    payload["model_state_dict"] = state
    torch.save(payload, paths[-1])
    _make_typed_checkpoint(paths[-1])
    with pytest.raises(ValueError, match=message):
        average_state_dicts(paths)


@pytest.mark.parametrize(
    "epochs",
    [
        (57, 58, 59, 60),
        (55, 56, 57, 58, 59, 60),
        (56, 57, 58, 60, 59),
        (51, 52, 53, 54, 55, 56, 57, 58, 59),
    ],
)
def test_rejects_incomplete_extra_or_reordered_windows(
    tmp_path: Path,
    epochs: tuple[int, ...],
) -> None:
    paths = [
        _write_checkpoint(tmp_path / f"epoch-{index}.pt", epoch=epoch)
        for index, epoch in enumerate(epochs)
    ]
    with pytest.raises(ValueError, match="window"):
        average_state_dicts(paths)


def test_rejects_window_from_wrong_optimizer_stage(tmp_path: Path) -> None:
    paths = [
        _write_checkpoint(
            tmp_path / f"checkpoint_epoch{epoch:03d}.pt",
            epoch=epoch,
            optimizer_stage="stage1",
        )
        for epoch in range(56, 61)
    ]
    with pytest.raises(ValueError, match="stage2"):
        average_state_dicts(paths)


def test_build_result_binds_sources_control_and_alias(tmp_path: Path) -> None:
    paths = _window(tmp_path)
    result = build_averaged_checkpoint(
        paths,
        candidate_id="s2-swa-last5",
    )
    assert result["candidate_id"] == "s2-swa-last5"
    assert result["method"] == "swa"
    assert result["epochs"] == [56, 57, 58, 59, 60]
    assert result["strict_control_epoch"] == 60
    assert (
        result["strict_control_checkpoint_sha256"]
        == result["source_checkpoints"][-1]["sha256"]
    )
    assert result["alias_of"] == "s2-avg-last5"
    assert result["model_state_sha256"] == hash_state_dict(result["model_state_dict"])
    assert "optimizer_state_dict" not in result


def test_averaged_payload_hash_binds_every_semantic_field_and_tensor(
    tmp_path: Path,
) -> None:
    result = build_averaged_checkpoint(
        _window(tmp_path),
        candidate_id="s2-swa-last5",
    )
    assert result["payload_sha256"] == (
        checkpoints_module.hash_averaged_checkpoint_payload(result)
    )
    verified = checkpoints_module.verify_averaged_checkpoint_payload(result)
    assert verified["payload_sha256"] == result["payload_sha256"]

    targets = (
        "schema_version",
        "payload_type",
        "candidate_id",
        "method",
        "epochs",
        "subject",
        "seed",
        "config_sha256",
        "schedule_sha256",
        "optimizer_stage",
        "trajectory_sha256",
        "source_checkpoints",
        "strict_control_epoch",
        "strict_control_checkpoint_sha256",
        "alias_of",
        "model_state_dict",
        "model_state_sha256",
        "payload_sha256",
    )
    for target in targets:
        tampered = copy.deepcopy(result)
        if target == "schema_version":
            tampered[target] = 2
        elif target == "payload_type":
            tampered[target] = "samga_brain_rw.other"
        elif target == "candidate_id":
            tampered[target] = "s2-avg-last10"
        elif target == "method":
            tampered[target] = "arithmetic"
        elif target == "epochs":
            tampered[target][0] = 55
        elif target == "subject":
            tampered[target] = 2
        elif target == "seed":
            tampered[target] = 43
        elif target in {
            "config_sha256",
            "schedule_sha256",
            "trajectory_sha256",
            "strict_control_checkpoint_sha256",
            "model_state_sha256",
            "payload_sha256",
        }:
            tampered[target] = "0" * 64
        elif target == "optimizer_stage":
            tampered[target] = "stage1"
        elif target == "source_checkpoints":
            tampered[target][0]["sha256"] = "0" * 64
        elif target == "strict_control_epoch":
            tampered[target] = 59
        elif target == "alias_of":
            tampered[target] = None
        elif target == "model_state_dict":
            tampered[target]["weight"][0] += 1
        else:  # pragma: no cover - target list is exhaustive
            raise AssertionError(target)
        with pytest.raises(ValueError, match="payload.*SHA-256|hash"):
            checkpoints_module.verify_averaged_checkpoint_payload(tampered)


def test_averaged_checkpoint_loader_rejects_tampered_payload(tmp_path: Path) -> None:
    result = build_averaged_checkpoint(
        _window(tmp_path),
        candidate_id="s2-avg-last5",
    )
    valid = tmp_path / "valid-averaged.pt"
    torch.save(result, valid)
    loaded = checkpoints_module.load_averaged_checkpoint(valid)
    assert loaded["payload_sha256"] == result["payload_sha256"]

    result["subject"] = 2
    tampered = tmp_path / "tampered-averaged.pt"
    torch.save(result, tampered)
    with pytest.raises(ValueError, match="payload.*SHA-256|hash"):
        checkpoints_module.load_averaged_checkpoint(tampered)


@pytest.mark.parametrize(
    "relative",
    (
        Path("test") / "averaged.pt",
        Path("formal") / "averaged.pt",
        Path("sub-01_test.json"),
        Path("sub-01_test.pt"),
    ),
)
def test_development_checkpoint_guard_rejects_all_test_artifact_names(
    tmp_path: Path,
    relative: Path,
) -> None:
    with pytest.raises(ValueError, match="sealed"):
        checkpoints_module.validate_development_checkpoint_path(
            tmp_path / relative,
            "checkpoint",
        )


def test_averaging_rejects_typed_checkpoint_under_sealed_directory(
    tmp_path: Path,
) -> None:
    sealed_directory = tmp_path / "formal-test"
    sealed_directory.mkdir()
    paths = _typed_window(sealed_directory)

    with pytest.raises(ValueError, match="sealed"):
        average_state_dicts(paths)


def test_checkpoint_read_fails_closed_on_intermediate_symlink_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    safe_directory = tmp_path / "development"
    sealed_directory = tmp_path / "formal-test"
    safe_directory.mkdir()
    sealed_directory.mkdir()
    target = safe_directory / "checkpoint.pt"
    target.write_bytes(b"development")
    (sealed_directory / target.name).write_bytes(b"sealed")
    state, _ = _install_intermediate_directory_swap(
        monkeypatch,
        safe_directory=safe_directory,
        sealed_directory=sealed_directory,
        target=target,
    )

    with pytest.raises((OSError, ValueError)):
        checkpoints_module._read_checkpoint_bytes(target)
    assert state["swapped"]


def test_checkpoint_write_fails_closed_on_intermediate_symlink_swap(
    experiment_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = build_averaged_checkpoint(
        _window(tmp_path),
        candidate_id="s2-avg-last5",
    )
    script = runpy.run_path(
        str(experiment_root / "scripts" / "build_averaged_checkpoint.py")
    )
    write_exclusive = script["_write_exclusive"]
    safe_directory = tmp_path / "development-output"
    sealed_directory = tmp_path / "formal-test-output"
    safe_directory.mkdir()
    sealed_directory.mkdir()
    output = safe_directory / "averaged.pt"
    state, backup = _install_intermediate_directory_swap(
        monkeypatch,
        safe_directory=safe_directory,
        sealed_directory=sealed_directory,
        target=output,
    )

    with pytest.raises((OSError, ValueError)):
        write_exclusive(output, payload)
    assert state["swapped"]
    assert not (sealed_directory / output.name).exists()
    assert not (backup / output.name).exists()


def test_cli_writes_exclusively_and_rejects_wrong_candidate(
    experiment_root: Path,
    tmp_path: Path,
) -> None:
    paths = _window(tmp_path)
    output = tmp_path / "averaged.pt"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(experiment_root)
    command = [
        sys.executable,
        str(experiment_root / "scripts" / "build_averaged_checkpoint.py"),
        "--candidate-id",
        "s2-avg-last5",
        "--output",
        str(output),
    ]
    for path in paths:
        command.extend(("--checkpoint", str(path)))
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    saved = torch.load(output, map_location="cpu", weights_only=False)
    assert saved["candidate_id"] == "s2-avg-last5"

    original = output.read_bytes()
    repeated = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert repeated.returncode != 0
    assert output.read_bytes() == original

    wrong = command.copy()
    wrong[wrong.index("s2-avg-last5")] = "s2-avg-last10"
    wrong[wrong.index(str(output))] = str(tmp_path / "wrong.pt")
    rejected = subprocess.run(
        wrong,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert rejected.returncode != 0
    assert not (tmp_path / "wrong.pt").exists()

    formal = command.copy()
    formal[formal.index(str(output))] = str(tmp_path / "formal-test" / "averaged.pt")
    rejected_formal = subprocess.run(
        formal,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert rejected_formal.returncode != 0
    assert not (tmp_path / "formal-test").exists()
