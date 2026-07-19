from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

import evaluate as samga_evaluate
from samga_brain_rw.brainrw import ManifestIdentity
from samga_brain_rw.hashing import sha256_json


def _h(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _argv(tmp_path: Path) -> list[str]:
    return [
        "--scope",
        "val-dev",
        "--subject",
        "8",
        "--seed",
        "42",
        "--config",
        str(tmp_path / "internvit_baseline_v1.json"),
        "--manifest",
        str(tmp_path / "sub-08_protocol.json"),
        "--feature-cache",
        str(tmp_path / "features.npy"),
        "--checkpoint",
        str(tmp_path / "checkpoint_epoch060.pt"),
        "--output-dir",
        str(tmp_path / "scores"),
    ]


def test_parser_requires_explicit_development_only_paths(
    tmp_path: Path,
) -> None:
    arguments = samga_evaluate.parse_arguments(_argv(tmp_path))

    assert arguments.scope == "val-dev"
    assert arguments.subject == 8
    assert arguments.seed == 42
    assert arguments.config == tmp_path / "internvit_baseline_v1.json"
    assert arguments.manifest == tmp_path / "sub-08_protocol.json"
    assert arguments.feature_cache == tmp_path / "features.npy"
    assert arguments.checkpoint == tmp_path / "checkpoint_epoch060.pt"
    assert arguments.output_dir == tmp_path / "scores"


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--scope", "val-confirm"),
        ("--scope", "formal-test"),
        ("--scope", "test"),
        ("--subject", "0"),
        ("--subject", "11"),
        ("--seed", "-1"),
    ],
)
def test_parser_rejects_non_development_or_invalid_identity(
    tmp_path: Path,
    flag: str,
    value: str,
) -> None:
    argv = _argv(tmp_path)
    position = argv.index(flag)
    argv[position + 1] = value

    with pytest.raises(SystemExit):
        samga_evaluate.parse_arguments(argv)


@pytest.mark.parametrize(
    "missing",
    [
        "--config",
        "--manifest",
        "--feature-cache",
        "--checkpoint",
        "--output-dir",
    ],
)
def test_parser_rejects_every_missing_explicit_path(
    tmp_path: Path,
    missing: str,
) -> None:
    argv = _argv(tmp_path)
    position = argv.index(missing)
    del argv[position : position + 2]

    with pytest.raises(SystemExit):
        samga_evaluate.parse_arguments(argv)


def _manifest(tmp_path: Path) -> ManifestIdentity:
    return ManifestIdentity(
        path=(tmp_path / "sub-08_protocol.json").absolute(),
        subject=8,
        manifest_sha256=_h("manifest"),
        protocol_sha256=_h("protocol"),
        records_sha256=_h("records"),
        source_manifest_sha256=_h("source-manifest"),
        train_role_sha256=_h("train-role"),
        val_dev_role_sha256=_h("val-dev-role"),
        train_ordered_ids=("train-a", "train-b"),
        val_dev_ordered_ids=("image-a", "image-b"),
        train_ordered_ids_sha256=_h("train-ids"),
        val_dev_ordered_ids_sha256=_h("val-dev-ids"),
    )


def _config_payload(tmp_path: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "config_type": "internvit_baseline",
        "config_id": "internvit_baseline_v1",
        "upstream": {
            "path": str((tmp_path / "SAMGA").absolute()),
            "git_commit": samga_evaluate.PINNED_UPSTREAM_SHA,
        },
        "model": {
            "repo": "OpenGVLab/InternViT-6B-448px-V2_5",
            "revision": "9d1a4344077479c93d42584b6941c64d795d508d",
            "path": str((tmp_path / "InternViT").absolute()),
            "config_sha256": _h("model-config"),
            "preprocessor_sha256": _h("preprocessor"),
            "weight_sha256": {"model.safetensors": _h("model-weight")},
        },
        "cache": {
            "path": str((tmp_path / "features.npy").absolute()),
            "sha256": _h("cache"),
        },
        "task": {
            "batch_size": 512,
            "channels": list(samga_evaluate.POSTERIOR_CHANNELS),
            "force_global": True,
        },
    }


@dataclass(frozen=True)
class _Semantic:
    payload: dict[str, object]
    sha256: str = _h("baseline-config")

    def canonical_payload(self) -> dict[str, object]:
        return self.payload


def _candidate_spec(
    *,
    input_hashes: dict[str, str],
    config_sha256: str,
    whitening: bool = False,
) -> dict[str, object]:
    input_bundle_sha256 = sha256_json(dict(sorted(input_hashes.items())))
    stage = "stage2" if whitening else "stage0"
    config_id = "s2-whitening-on" if whitening else "internvit_baseline_v1"
    run_key = samga_evaluate.make_run_key(
        stage,
        config_id,
        8,
        42,
        config_sha256,
        input_bundle_sha256,
    )
    body: dict[str, object] = {
        "schema_version": 1,
        "config_id": config_id,
        "stage": stage,
        "subject": 8,
        "seed": 42,
        "baseline_config_sha256": _h("baseline-config"),
        "stage2_config_sha256": _h("stage2-config") if whitening else None,
        "semantic_config_sha256": config_sha256,
        "input_bundle_sha256": input_bundle_sha256,
        "run_key": run_key,
        "layernorm_config_id": "s2-layernorm-off",
        "whitening_config_id": (
            "s2-whitening-on" if whitening else "s2-whitening-off"
        ),
        "preprojector_config_id": "s2-preproj-shared",
        "adapter_kind": "identity",
        "adapter_rank": None,
        "adapter_lr_ratio": None,
        "whitening_payload": {"sealed": "train-only"} if whitening else None,
        "full_task_initialization_sha256": _h("full-init"),
        "shared_parameter_intersection_name": "samga_shared_task_v1",
        "shared_parameter_intersection_sha256": _h("shared-init"),
        "architecture_specific_initialization_sha256": _h("architecture-init"),
        "data_order_sha256": _h("data-order"),
        "trajectory_sha256": _h("trajectory"),
    }
    return {**body, "candidate_spec_sha256": sha256_json(body)}


def _checkpoint(
    tmp_path: Path,
    *,
    whitening: bool = False,
) -> SimpleNamespace:
    manifest = _manifest(tmp_path)
    config = _config_payload(tmp_path)
    input_hashes = {
        "cache_sha256": _h("cache"),
        "manifest_sha256": manifest.manifest_sha256,
        "model_sha256": sha256_json(config["model"]),
        "protocol_sha256": manifest.protocol_sha256,
    }
    config_sha256 = _h("resolved-config")
    candidate = _candidate_spec(
        input_hashes=input_hashes,
        config_sha256=config_sha256,
        whitening=whitening,
    )
    run_body = {
        "schema_version": 1,
        "payload_type": samga_evaluate.RUN_PAYLOAD_TYPE,
        "stage": 2 if whitening else 0,
        "subject": 8,
        "seed": 42,
        "config_id": candidate["config_id"],
        "config_sha256": config_sha256,
        "protocol_sha256": manifest.protocol_sha256,
        "cache_sha256": _h("cache"),
        "git_sha": "a" * 40,
        "upstream_sha": samga_evaluate.PINNED_UPSTREAM_SHA,
        "data_order_sha256": _h("data-order"),
        "candidate_spec_sha256": candidate["candidate_spec_sha256"],
        "run_key": candidate["run_key"],
    }
    run_manifest = {
        **run_body,
        "run_manifest_sha256": sha256_json(run_body),
    }
    payload = {
        "subject": 8,
        "seed": 42,
        "config_sha256": config_sha256,
        "schedule_sha256": samga_evaluate.SCHEDULE_SHA256,
        "trajectory_sha256": _h("trajectory"),
        "data_order_sha256": _h("data-order"),
        "input_hashes": input_hashes,
        "run_manifest": run_manifest,
        "candidate_spec": candidate,
        "model_state_dict": {"weight": torch.ones(1)},
    }
    return SimpleNamespace(
        payload=MappingProxyType(payload),
        sha256=_h("checkpoint"),
    )


class _FakeDataset(Dataset[dict[str, object]]):
    scope = "val-dev"
    subject_id = 8
    query_ids = ("image-a", "image-b")
    gallery_ids = ("image-a", "image-b")
    feature_cache_metadata = MappingProxyType(
        {"feature_sha256": _h("cache")}
    )

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict[str, object]:
        raise AssertionError("the patched evaluator must not iterate")


def _install_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    checkpoint: SimpleNamespace | None = None,
) -> tuple[list[str], dict[str, object]]:
    events: list[str] = []
    captured: dict[str, object] = {}
    loaded = checkpoint or _checkpoint(tmp_path)
    semantic = _Semantic(_config_payload(tmp_path))

    def guard(path: Path, context: str) -> Path:
        events.append(f"guard:{context}")
        return Path(path).absolute()

    def load_manifest(path: Path, *, expected_subject: int) -> ManifestIdentity:
        events.append("manifest")
        assert expected_subject == 8
        return _manifest(tmp_path)

    class SemanticFactory:
        @classmethod
        def from_path(cls, path: Path) -> _Semantic:
            events.append("config")
            return semantic

    def load_checkpoint(path: Path, *, requested_scope: str) -> SimpleNamespace:
        events.append("checkpoint")
        assert requested_scope == "train"
        return loaded

    def dataset_factory(**kwargs: object) -> _FakeDataset:
        events.append("dataset")
        assert "manifest" in events
        assert "config" in events
        assert "checkpoint" in events
        assert kwargs == {
            "manifest_path": (tmp_path / "sub-08_protocol.json").absolute(),
            "scope": "val-dev",
            "seed": 42,
            "selected_channels": samga_evaluate.POSTERIOR_CHANNELS,
            "feature_cache": (tmp_path / "features.npy").absolute(),
            "smooth_probability": 0.0,
        }
        return _FakeDataset()

    def load_components(path: Path, commit: str) -> object:
        events.append("upstream")
        assert path == (tmp_path / "SAMGA").absolute()
        assert commit == samga_evaluate.PINNED_UPSTREAM_SHA
        return object()

    def spec_factory(**kwargs: object) -> object:
        events.append("spec")
        captured["spec"] = kwargs
        return SimpleNamespace()

    class FakeModel:
        def load_state_dict(
            self,
            state: object,
            *,
            strict: bool,
        ) -> object:
            events.append("load-state")
            assert strict is True
            assert isinstance(state, dict)
            return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    def model_factory(spec: object) -> FakeModel:
        events.append("model")
        return FakeModel()

    def evaluate_model(
        model: object,
        dataset: object,
        *,
        batch_size: int,
        device: str,
        seed: int,
    ) -> SimpleNamespace:
        events.append("evaluate")
        assert isinstance(model, FakeModel)
        assert isinstance(dataset, _FakeDataset)
        assert (batch_size, device, seed) == (512, "auto", 42)
        return SimpleNamespace(
            similarity=np.asarray(
                [[0.9, 0.1], [0.2, 0.8]],
                dtype=np.float32,
            )
        )

    class FakeScores:
        @staticmethod
        def save(
            directory: Path,
            similarity: np.ndarray,
            query_ids: tuple[str, ...],
            gallery_ids: tuple[str, ...],
            metadata: dict[str, object],
        ) -> None:
            events.append("save")
            captured["score"] = {
                "directory": directory,
                "similarity": similarity,
                "query_ids": query_ids,
                "gallery_ids": gallery_ids,
                "metadata": metadata,
            }

    monkeypatch.setattr(samga_evaluate, "reject_development_path", guard)
    monkeypatch.setattr(
        samga_evaluate,
        "load_development_manifest_identity",
        load_manifest,
    )
    monkeypatch.setattr(samga_evaluate, "SemanticConfig", SemanticFactory)
    monkeypatch.setattr(
        samga_evaluate,
        "load_samga_checkpoint",
        load_checkpoint,
    )
    monkeypatch.setattr(
        samga_evaluate,
        "ProtocolSubjectDataset",
        dataset_factory,
    )
    monkeypatch.setattr(
        samga_evaluate,
        "load_locked_upstream_components",
        load_components,
    )
    monkeypatch.setattr(
        samga_evaluate,
        "TrainingCellSpec",
        spec_factory,
    )
    monkeypatch.setattr(
        samga_evaluate,
        "SAMGARuntimeModel",
        model_factory,
    )
    monkeypatch.setattr(
        samga_evaluate,
        "evaluate_development_model",
        evaluate_model,
    )
    monkeypatch.setattr(samga_evaluate, "ScoreArtifact", FakeScores)
    return events, captured


def test_main_verifies_all_identities_before_dataset_and_wires_public_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events, captured = _install_runtime(monkeypatch, tmp_path)

    assert samga_evaluate.main(_argv(tmp_path)) == 0

    assert events.index("manifest") < events.index("dataset")
    assert events.index("config") < events.index("dataset")
    assert events.index("checkpoint") < events.index("dataset")
    assert events.index("dataset") < events.index("evaluate") < events.index("save")
    spec = captured["spec"]
    assert isinstance(spec, dict)
    assert spec["subject"] == 8
    assert spec["seed"] == 42
    assert spec["stage"] == 0
    assert spec["layernorm_config_id"] == "s2-layernorm-off"
    assert spec["whitening_config_id"] == "s2-whitening-off"
    assert spec["preprojector_config_id"] == "s2-preproj-shared"
    assert spec["adapter_kind"] == "identity"

    score = captured["score"]
    assert isinstance(score, dict)
    assert score["directory"] == (tmp_path / "scores").absolute()
    assert score["query_ids"] == ("image-a", "image-b")
    assert score["gallery_ids"] == ("image-a", "image-b")
    assert set(score["metadata"]) == {
        "checkpoint_sha256",
        "config_sha256",
        "git_sha",
        "protocol_sha256",
        "seed",
        "source_records",
        "split_role",
        "stage",
        "subject",
    }
    assert score["metadata"]["checkpoint_sha256"] == _h("checkpoint")
    assert score["metadata"]["config_sha256"] == _h("resolved-config")
    assert score["metadata"]["git_sha"] == "a" * 40
    assert score["metadata"]["protocol_sha256"] == _h("protocol")
    assert score["metadata"]["split_role"] == "val-dev"
    assert score["metadata"]["stage"] == "stage0"


def test_checkpoint_identity_mismatch_fails_before_dataset_or_cache_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loaded = _checkpoint(tmp_path)
    payload = dict(loaded.payload)
    payload["seed"] = 43
    mismatched = SimpleNamespace(
        payload=MappingProxyType(payload),
        sha256=loaded.sha256,
    )
    events, _ = _install_runtime(
        monkeypatch,
        tmp_path,
        checkpoint=mismatched,
    )

    with pytest.raises(ValueError, match="seed"):
        samga_evaluate.main(_argv(tmp_path))

    assert "dataset" not in events
    assert "upstream" not in events
    assert "evaluate" not in events
    assert "save" not in events


def test_unknown_candidate_identity_field_fails_closed_before_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loaded = _checkpoint(tmp_path)
    payload = dict(loaded.payload)
    candidate = dict(payload["candidate_spec"])
    candidate["surprise"] = "not-preregistered"
    payload["candidate_spec"] = candidate
    events, _ = _install_runtime(
        monkeypatch,
        tmp_path,
        checkpoint=SimpleNamespace(
            payload=MappingProxyType(payload),
            sha256=loaded.sha256,
        ),
    )

    with pytest.raises(ValueError, match="candidate_spec.*keys|unknown"):
        samga_evaluate.main(_argv(tmp_path))

    assert "dataset" not in events


def test_existing_output_directory_is_rejected_before_any_input_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "scores").mkdir()
    events, _ = _install_runtime(monkeypatch, tmp_path)

    with pytest.raises(FileExistsError, match="output"):
        samga_evaluate.main(_argv(tmp_path))

    assert "manifest" not in events
    assert "config" not in events
    assert "checkpoint" not in events
    assert "dataset" not in events


def test_whitening_checkpoint_rebuilds_only_from_sealed_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loaded = _checkpoint(tmp_path, whitening=True)
    _, captured = _install_runtime(
        monkeypatch,
        tmp_path,
        checkpoint=loaded,
    )
    whitening = object()
    seen: list[object] = []

    class WhiteningFactory:
        @classmethod
        def from_payload(cls, payload: object) -> object:
            seen.append(payload)
            return whitening

    monkeypatch.setattr(samga_evaluate, "TrainWhitening", WhiteningFactory)

    assert samga_evaluate.main(_argv(tmp_path)) == 0

    assert seen == [{"sealed": "train-only"}]
    spec = captured["spec"]
    assert isinstance(spec, dict)
    assert spec["stage"] == 2
    assert spec["whitening_config_id"] == "s2-whitening-on"
    assert spec["whitening"] is whitening


def test_sealed_cli_path_is_rejected_before_any_identity_or_data_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    argv = _argv(tmp_path)
    argv[argv.index("--manifest") + 1] = str(
        tmp_path / "val-confirm" / "sub-08_protocol.json"
    )
    monkeypatch.setattr(
        samga_evaluate,
        "load_development_manifest_identity",
        lambda *args, **kwargs: pytest.fail(
            "sealed path must fail before manifest loading"
        ),
    )
    monkeypatch.setattr(
        samga_evaluate,
        "load_samga_checkpoint",
        lambda *args, **kwargs: pytest.fail(
            "sealed path must fail before checkpoint loading"
        ),
    )

    with pytest.raises(PermissionError, match="sealed"):
        samga_evaluate.main(argv)


def _reseal_candidate(
    payload: dict[str, object],
    **updates: object,
) -> None:
    candidate = dict(payload["candidate_spec"])
    candidate.update(updates)
    body = {
        key: value
        for key, value in candidate.items()
        if key != "candidate_spec_sha256"
    }
    candidate["candidate_spec_sha256"] = sha256_json(body)
    payload["candidate_spec"] = candidate


def _reseal_run(
    payload: dict[str, object],
    **updates: object,
) -> None:
    run = dict(payload["run_manifest"])
    run.update(updates)
    body = {
        key: value
        for key, value in run.items()
        if key != "run_manifest_sha256"
    }
    run["run_manifest_sha256"] = sha256_json(body)
    payload["run_manifest"] = run


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("config", "semantic config"),
        ("cache", "cache_sha256"),
        ("manifest", "manifest_sha256"),
        ("protocol", "protocol_sha256"),
        ("run_candidate", "candidate_spec_sha256"),
        ("run_key", "run_key"),
        ("candidate_subject", "candidate_spec subject"),
    ],
)
def test_every_checkpoint_provenance_layer_fails_before_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    loaded = _checkpoint(tmp_path)
    payload = dict(loaded.payload)
    if mutation == "config":
        payload["config_sha256"] = _h("wrong-config")
    elif mutation in {"cache", "manifest", "protocol"}:
        inputs = dict(payload["input_hashes"])
        inputs[f"{mutation}_sha256"] = _h(f"wrong-{mutation}")
        payload["input_hashes"] = inputs
    elif mutation == "run_candidate":
        _reseal_run(payload, candidate_spec_sha256=_h("wrong-candidate"))
    elif mutation == "run_key":
        _reseal_run(payload, run_key="stage0__wrong")
    elif mutation == "candidate_subject":
        _reseal_candidate(payload, subject=7)
        candidate = payload["candidate_spec"]
        assert isinstance(candidate, dict)
        _reseal_run(
            payload,
            candidate_spec_sha256=candidate["candidate_spec_sha256"],
        )
    else:
        raise AssertionError("unknown test mutation")
    events, _ = _install_runtime(
        monkeypatch,
        tmp_path,
        checkpoint=SimpleNamespace(
            payload=MappingProxyType(payload),
            sha256=loaded.sha256,
        ),
    )

    with pytest.raises(ValueError, match=message):
        samga_evaluate.main(_argv(tmp_path))

    assert "dataset" not in events
    assert "evaluate" not in events
    assert "save" not in events
