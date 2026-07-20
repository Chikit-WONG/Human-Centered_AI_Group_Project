from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

import train as samga_train
import evaluate as samga_evaluate
from samga_brain_rw.brainrw import ManifestIdentity
from samga_brain_rw.config import ProtocolConfig, SemanticConfig, resolve_run_config
from samga_brain_rw.hashing import sha256_json
from samga_brain_rw.runtime_contract import (
    PINNED_SEMANTIC_ENVIRONMENT,
    PRODUCTION_RUNTIME_CONTRACT,
    build_environment_binding,
)


def _h(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _production_environment_binding() -> dict[str, object]:
    return build_environment_binding(
        PINNED_SEMANTIC_ENVIRONMENT,
        PRODUCTION_RUNTIME_CONTRACT,
    )


_PROTOCOL = ProtocolConfig.from_path(samga_train._PROTOCOL_CONFIG_PATH)
_STAGE2_CONFIG_PATH = (
    Path(samga_evaluate.__file__).resolve().parent
    / "configs"
    / "stage2_candidates_v1.json"
)
_STAGE2 = SemanticConfig.from_path(_STAGE2_CONFIG_PATH)


def _argv(
    tmp_path: Path,
    *,
    checkpoint: SimpleNamespace | None = None,
    output_dir: Path | None = None,
    checkpoint_kind: str = "raw",
) -> list[str]:
    loaded = checkpoint or _checkpoint(tmp_path)
    candidate = loaded.payload["candidate_spec"]
    assert isinstance(candidate, (dict, MappingProxyType))
    run_key = candidate["run_key"]
    assert isinstance(run_key, str)
    output = output_dir or tmp_path / run_key / "saved_checkpoint"
    argv = [
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
        str(output),
    ]
    if checkpoint_kind != "raw":
        output_index = argv.index("--output-dir")
        argv[output_index:output_index] = ["--checkpoint-kind", checkpoint_kind]
    return argv


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
    assert arguments.checkpoint_kind == "raw"
    assert arguments.output_dir.name == "saved_checkpoint"
    assert arguments.device == "cuda"


@pytest.mark.parametrize("device", ["auto", "cpu", "cuda:1"])
def test_parser_rejects_nonproduction_device(
    tmp_path: Path,
    device: str,
) -> None:
    argv = [*_argv(tmp_path), "--device", device]
    with pytest.raises(SystemExit):
        samga_evaluate.parse_arguments(argv)


def test_runtime_preflight_fails_before_evaluation_path_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def reject_runtime(device: str) -> object:
        events.append(f"runtime:{device}")
        raise RuntimeError("evaluation runtime probe failed")

    def forbidden_guard(arguments: object) -> object:
        del arguments
        events.append("paths")
        raise AssertionError("paths must not precede runtime preflight")

    monkeypatch.setattr(
        samga_evaluate,
        "require_production_runtime",
        reject_runtime,
    )
    monkeypatch.setattr(
        samga_evaluate,
        "_guard_paths",
        forbidden_guard,
    )

    with pytest.raises(RuntimeError, match="runtime probe"):
        samga_evaluate.run_evaluation(SimpleNamespace(device="cuda"))

    assert events == ["runtime:cuda"]


def test_clean_repository_and_upstream_preflight_precede_manifest_and_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    environment = _production_environment_binding()
    payload = _config_payload(tmp_path)

    monkeypatch.setattr(
        samga_evaluate,
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
        samga_evaluate,
        "clean_repository_git_sha",
        lambda: events.append("git") or "a" * 40,
        raising=False,
    )
    monkeypatch.setattr(
        samga_evaluate,
        "_guard_paths",
        lambda arguments: events.append("paths")
        or samga_evaluate.EvaluationPaths(
            config=tmp_path / "internvit_baseline_v1.json",
            manifest=tmp_path / "manifest.json",
            feature_cache=tmp_path / "features.npy",
            checkpoint=tmp_path / "checkpoint.pt",
            output_dir=tmp_path / "output",
        ),
    )

    class SemanticFactory:
        @classmethod
        def from_path(cls, path: Path) -> _Semantic:
            events.append("config")
            return _Semantic(payload)

    monkeypatch.setattr(samga_evaluate, "SemanticConfig", SemanticFactory)
    monkeypatch.setattr(
        samga_evaluate,
        "load_locked_upstream_components",
        lambda path, commit: events.append("upstream")
        or (_ for _ in ()).throw(ValueError("upstream import contract rejected")),
    )
    monkeypatch.setattr(
        samga_evaluate,
        "load_development_manifest_identity",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("manifest is forbidden before upstream preflight")
        ),
    )
    monkeypatch.setattr(
        samga_evaluate,
        "ProtocolSubjectDataset",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("dataset is forbidden before upstream preflight")
        ),
    )

    with pytest.raises(ValueError, match="upstream import contract"):
        samga_evaluate.run_evaluation(SimpleNamespace(device="cuda", subject=8))

    assert events == ["runtime", "git", "paths", "config", "upstream"]


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
        protocol_sha256=_PROTOCOL.sha256,
        records_sha256=_h("records"),
        source_manifest_sha256=_h("source-manifest"),
        source_payload_path=(tmp_path / "sub-08" / "train.pt").absolute(),
        source_payload_sha256=_h("source-payload"),
        source_payload_byte_count=123456,
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


def _input_hashes(
    tmp_path: Path,
    manifest: ManifestIdentity,
) -> dict[str, str]:
    config = _config_payload(tmp_path)
    return samga_train._build_input_hashes(
        manifest,
        SimpleNamespace(
            cache_sha256=_h("cache"),
            model_sha256=sha256_json(config["model"]),
        ),
    )


def _evaluation_config(
    tmp_path: Path,
) -> samga_evaluate.EvaluationConfig:
    payload = _config_payload(tmp_path)
    return samga_evaluate.EvaluationConfig(
        semantic=_Semantic(payload),
        payload=payload,
        protocol=_PROTOCOL,
        stage2_semantic=_STAGE2,
        stage2_payload=_STAGE2.canonical_payload(),
        upstream_root=(tmp_path / "SAMGA").absolute(),
        upstream_commit=samga_evaluate.PINNED_UPSTREAM_SHA,
        cache_sha256=_h("cache"),
        model_sha256=sha256_json(payload["model"]),
        batch_size=512,
    )


def _candidate_spec(
    *,
    input_hashes: dict[str, str],
    whitening: bool = False,
) -> dict[str, object]:
    stage = "stage2" if whitening else "stage0"
    config_id = "s2-whitening-on" if whitening else "internvit_baseline_v1"
    whitening_payload = (
        {
            "sealed": "train-only",
            "payload_sha256": _h("whitening-payload"),
        }
        if whitening
        else None
    )
    candidate_payload = samga_train.build_resolved_candidate_payload(
        stage=2 if whitening else 0,
        config_id=config_id,
        subject=8,
        seed=42,
        environment_binding=_production_environment_binding(),
        baseline_config_sha256=_h("baseline-config"),
        stage2_config_sha256=_STAGE2.sha256 if whitening else None,
        layernorm_config_id="s2-layernorm-off",
        whitening_config_id=("s2-whitening-on" if whitening else "s2-whitening-off"),
        preprojector_config_id="s2-preproj-shared",
        adapter_kind="identity",
        adapter_rank=None,
        adapter_lr_ratio=None,
        whitening_payload_sha256=(_h("whitening-payload") if whitening else None),
    )
    resolved = resolve_run_config(_PROTOCOL, candidate_payload, input_hashes)
    body: dict[str, object] = {
        "schema_version": 1,
        "config_id": config_id,
        "stage": stage,
        "subject": 8,
        "seed": 42,
        "baseline_config_sha256": _h("baseline-config"),
        "stage2_config_sha256": _STAGE2.sha256 if whitening else None,
        "semantic_config_sha256": resolved.semantic_config_sha256,
        "input_bundle_sha256": resolved.input_bundle_sha256,
        "run_key": resolved.run_key,
        "layernorm_config_id": "s2-layernorm-off",
        "whitening_config_id": ("s2-whitening-on" if whitening else "s2-whitening-off"),
        "preprojector_config_id": "s2-preproj-shared",
        "adapter_kind": "identity",
        "adapter_rank": None,
        "adapter_lr_ratio": None,
        "whitening_payload": whitening_payload,
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
    input_hashes = _input_hashes(tmp_path, manifest)
    candidate = _candidate_spec(
        input_hashes=input_hashes,
        whitening=whitening,
    )
    config_sha256 = candidate["semantic_config_sha256"]
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
        "environment": _production_environment_binding(),
        "config_sha256": config_sha256,
        "schedule_sha256": samga_evaluate.SCHEDULE_SHA256,
        "trajectory_sha256": _h("trajectory"),
        "data_order_sha256": _h("data-order"),
        "epoch": 60,
        "runtime_state": {
            "epoch_complete": True,
            "next_epoch": 61,
            "resume_source_checkpoint_sha256": None,
        },
        "input_hashes": input_hashes,
        "run_manifest": run_manifest,
        "candidate_spec": candidate,
        "model_state_dict": {"weight": torch.ones(1)},
    }
    return SimpleNamespace(
        payload=MappingProxyType(
            {
                **payload,
                "candidate_spec": MappingProxyType(candidate),
            }
        ),
        sha256=_h("checkpoint"),
    )


def test_evaluator_accepts_exact_training_input_hash_schema(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    loaded = _checkpoint(tmp_path)
    expected = samga_train._build_input_hashes(
        manifest,
        SimpleNamespace(
            cache_sha256=_h("cache"),
            model_sha256=sha256_json(_config_payload(tmp_path)["model"]),
        ),
    )

    identity = samga_evaluate._evaluation_identity(
        loaded,
        manifest=manifest,
        config=_evaluation_config(tmp_path),
        subject=8,
        seed=42,
    )

    assert identity.input_hashes == dict(sorted(expected.items()))
    assert identity.input_hashes["ordered_ids_sha256"] == expected["ordered_ids_sha256"]
    assert identity.input_hashes["records_sha256"] == manifest.records_sha256
    assert (
        identity.input_hashes["train_ordered_ids_sha256"]
        == manifest.train_ordered_ids_sha256
    )
    assert (
        identity.input_hashes["val_dev_ordered_ids_sha256"]
        == manifest.val_dev_ordered_ids_sha256
    )


@pytest.mark.parametrize(
    "field",
    [
        "ordered_ids_sha256",
        "records_sha256",
        "train_ordered_ids_sha256",
        "val_dev_ordered_ids_sha256",
    ],
)
def test_each_training_manifest_input_hash_mismatch_fails_closed(
    tmp_path: Path,
    field: str,
) -> None:
    manifest = _manifest(tmp_path)
    loaded = _checkpoint(tmp_path)
    payload = dict(loaded.payload)
    inputs = dict(payload["input_hashes"])
    inputs[field] = _h(f"wrong-{field}")
    payload["input_hashes"] = inputs

    with pytest.raises(ValueError, match=field):
        samga_evaluate._evaluation_identity(
            _mutated_checkpoint(loaded, payload),
            manifest=manifest,
            config=_evaluation_config(tmp_path),
            subject=8,
            seed=42,
        )


@pytest.mark.parametrize("mutation", ["missing", "unknown"])
def test_training_input_hash_schema_rejects_missing_and_unknown_keys(
    tmp_path: Path,
    mutation: str,
) -> None:
    manifest = _manifest(tmp_path)
    loaded = _checkpoint(tmp_path)
    payload = dict(loaded.payload)
    inputs = dict(payload["input_hashes"])
    if mutation == "missing":
        del inputs["ordered_ids_sha256"]
    else:
        inputs["unknown_sha256"] = _h("unknown")
    payload["input_hashes"] = inputs

    with pytest.raises(ValueError, match="input_hashes.*schema"):
        samga_evaluate._evaluation_identity(
            _mutated_checkpoint(loaded, payload),
            manifest=manifest,
            config=_evaluation_config(tmp_path),
            subject=8,
            seed=42,
        )


class _FakeDataset(Dataset[dict[str, object]]):
    scope = "val-dev"
    subject_id = 8
    query_ids = ("image-a", "image-b")
    gallery_ids = ("image-a", "image-b")
    row_indices = (12_540, 12_541)
    feature_cache_metadata = MappingProxyType({"feature_sha256": _h("cache")})

    def __init__(self, tmp_path: Path) -> None:
        self.manifest_path = (tmp_path / "sub-08_protocol.json").absolute()

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict[str, object]:
        raise AssertionError("the patched evaluator must not iterate")


def _install_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    checkpoint: SimpleNamespace | None = None,
    manifest_identity: ManifestIdentity | None = None,
) -> tuple[list[str], dict[str, object]]:
    events: list[str] = []
    captured: dict[str, object] = {}
    loaded = checkpoint or _checkpoint(tmp_path)
    verified_manifest = manifest_identity or _manifest(tmp_path)
    semantic = _Semantic(_config_payload(tmp_path))

    def production_runtime(device: str) -> SimpleNamespace:
        events.append("runtime")
        assert device == "cuda"
        return SimpleNamespace(
            device=torch.device("cuda:0"),
            environment_binding=_production_environment_binding(),
            contract=PRODUCTION_RUNTIME_CONTRACT,
            evidence={"accelerator_name": "NVIDIA A40"},
        )

    def guard(path: Path, context: str) -> Path:
        events.append(f"guard:{context}")
        return Path(path).absolute()

    def load_manifest(path: Path, *, expected_subject: int) -> ManifestIdentity:
        events.append("manifest")
        assert expected_subject == 8
        return verified_manifest

    class SemanticFactory:
        @classmethod
        def from_path(cls, path: Path) -> _Semantic:
            if Path(path) == _STAGE2_CONFIG_PATH:
                events.append("stage2-config")
                return _STAGE2  # type: ignore[return-value]
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
            "expected_source_payload_sha256": _h("source-payload"),
        }
        return _FakeDataset(tmp_path)

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
        device: object,
        seed: int,
    ) -> SimpleNamespace:
        events.append("evaluate")
        assert isinstance(model, FakeModel)
        assert isinstance(dataset, _FakeDataset)
        assert (batch_size, device, seed) == (
            512,
            torch.device("cuda:0"),
            42,
        )
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

    monkeypatch.setattr(
        samga_evaluate,
        "require_production_runtime",
        production_runtime,
    )
    monkeypatch.setattr(
        samga_evaluate,
        "clean_repository_git_sha",
        lambda: events.append("git") or "a" * 40,
    )
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

    assert events[0] == "runtime"
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
    assert spec["environment"] == _production_environment_binding()
    assert spec["device"] == torch.device("cuda:0")
    assert spec["resume_source_checkpoint_sha256"] is None

    checkpoint = _checkpoint(tmp_path)
    candidate = checkpoint.payload["candidate_spec"]
    assert isinstance(candidate, MappingProxyType)
    expected_output = tmp_path / str(candidate["run_key"]) / "saved_checkpoint"
    score = captured["score"]
    assert isinstance(score, dict)
    assert score["directory"] == expected_output.absolute()
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
    assert score["metadata"]["config_sha256"] == candidate["semantic_config_sha256"]
    assert score["metadata"]["git_sha"] == "a" * 40
    assert score["metadata"]["protocol_sha256"] == _PROTOCOL.sha256
    assert score["metadata"]["split_role"] == "val-dev"
    assert score["metadata"]["stage"] == "stage0"
    assert score["metadata"]["source_records"][0]["run_key"] == candidate["run_key"]


def test_current_evaluator_git_sha_must_match_checkpoint_before_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events, _ = _install_runtime(monkeypatch, tmp_path)
    monkeypatch.setattr(
        samga_evaluate,
        "clean_repository_git_sha",
        lambda: events.append("git") or "b" * 40,
        raising=False,
    )

    with pytest.raises(ValueError, match="Git SHA|git_sha"):
        samga_evaluate.main(_argv(tmp_path))

    assert "dataset" not in events
    assert "evaluate" not in events
    assert "save" not in events


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
    assert "upstream" in events
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
    argv = _argv(tmp_path)
    output = Path(argv[argv.index("--output-dir") + 1])
    output.mkdir(parents=True)
    events, _ = _install_runtime(monkeypatch, tmp_path)

    with pytest.raises(FileExistsError, match="output"):
        samga_evaluate.main(argv)

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

    class FakeWhitening:
        payload_sha256 = _h("whitening-payload")
        input_provenance_sha256 = _h("manifest")
        cache_provenance_sha256 = _h("cache")
        canonical_train_rows = tuple(range(12_540))

        def to_payload(self) -> dict[str, object]:
            return {
                "sealed": "train-only",
                "payload_sha256": self.payload_sha256,
            }

    whitening = FakeWhitening()
    seen: list[object] = []

    class WhiteningFactory:
        @classmethod
        def from_payload(cls, payload: object) -> object:
            seen.append(payload)
            return whitening

    monkeypatch.setattr(samga_evaluate, "TrainWhitening", WhiteningFactory)

    assert samga_evaluate.main(_argv(tmp_path, checkpoint=loaded)) == 0

    assert seen == [
        {
            "sealed": "train-only",
            "payload_sha256": _h("whitening-payload"),
        }
    ]
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
        "require_production_runtime",
        lambda device: SimpleNamespace(
            environment_binding=_production_environment_binding()
        ),
    )
    monkeypatch.setattr(
        samga_evaluate,
        "clean_repository_git_sha",
        lambda: "a" * 40,
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
        key: value for key, value in candidate.items() if key != "candidate_spec_sha256"
    }
    candidate["candidate_spec_sha256"] = sha256_json(body)
    payload["candidate_spec"] = candidate


def _reseal_run(
    payload: dict[str, object],
    **updates: object,
) -> None:
    run = dict(payload["run_manifest"])
    run.update(updates)
    body = {key: value for key, value in run.items() if key != "run_manifest_sha256"}
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


def _rebind_resolved_candidate(
    payload: dict[str, object],
    input_hashes: dict[str, str],
) -> None:
    candidate = dict(payload["candidate_spec"])
    whitening_payload = candidate["whitening_payload"]
    whitening_payload_sha256 = None
    if isinstance(whitening_payload, dict):
        value = whitening_payload.get("payload_sha256")
        assert isinstance(value, str)
        whitening_payload_sha256 = value
    stage_number = int(str(candidate["stage"]).removeprefix("stage"))
    semantic_payload = samga_train.build_resolved_candidate_payload(
        stage=stage_number,
        config_id=str(candidate["config_id"]),
        subject=int(candidate["subject"]),
        seed=int(candidate["seed"]),
        environment_binding=payload["environment"],
        baseline_config_sha256=str(candidate["baseline_config_sha256"]),
        stage2_config_sha256=candidate["stage2_config_sha256"],
        layernorm_config_id=str(candidate["layernorm_config_id"]),
        whitening_config_id=str(candidate["whitening_config_id"]),
        preprojector_config_id=str(candidate["preprojector_config_id"]),
        adapter_kind=str(candidate["adapter_kind"]),
        adapter_rank=candidate["adapter_rank"],
        adapter_lr_ratio=candidate["adapter_lr_ratio"],
        whitening_payload_sha256=whitening_payload_sha256,
    )
    resolved = resolve_run_config(_PROTOCOL, semantic_payload, input_hashes)
    candidate.update(
        semantic_config_sha256=resolved.semantic_config_sha256,
        input_bundle_sha256=resolved.input_bundle_sha256,
        run_key=resolved.run_key,
    )
    payload["config_sha256"] = resolved.semantic_config_sha256
    payload["input_hashes"] = input_hashes
    payload["candidate_spec"] = candidate
    _reseal_candidate(payload)
    candidate = payload["candidate_spec"]
    assert isinstance(candidate, dict)
    _reseal_run(
        payload,
        config_id=candidate["config_id"],
        config_sha256=resolved.semantic_config_sha256,
        candidate_spec_sha256=candidate["candidate_spec_sha256"],
        run_key=resolved.run_key,
    )


def _mutated_checkpoint(
    loaded: SimpleNamespace,
    payload: dict[str, object],
) -> SimpleNamespace:
    return SimpleNamespace(
        payload=MappingProxyType(payload),
        sha256=loaded.sha256,
    )


def test_checkpoint_environment_mismatch_fails_before_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loaded = _checkpoint(tmp_path)
    payload = dict(loaded.payload)
    semantic_environment = dict(PINNED_SEMANTIC_ENVIRONMENT)
    semantic_environment["numpy"] = "1.26.5"
    payload["environment"] = build_environment_binding(
        semantic_environment,
        PRODUCTION_RUNTIME_CONTRACT,
    )
    input_hashes = dict(payload["input_hashes"])
    _rebind_resolved_candidate(payload, input_hashes)
    events, _ = _install_runtime(
        monkeypatch,
        tmp_path,
        checkpoint=_mutated_checkpoint(loaded, payload),
    )

    with pytest.raises(
        ValueError,
        match="environment.*production runtime",
    ):
        samga_evaluate.main(_argv(tmp_path))

    assert "dataset" not in events
    assert "upstream" in events


def _install_fake_whitening(
    monkeypatch: pytest.MonkeyPatch,
    loaded: SimpleNamespace,
    *,
    input_provenance_sha256: str = _h("manifest"),
    cache_provenance_sha256: str = _h("cache"),
    changed_after_first_round_trip: bool = False,
) -> object:
    candidate = loaded.payload["candidate_spec"]
    assert isinstance(candidate, (dict, MappingProxyType))
    sealed = candidate["whitening_payload"]
    assert isinstance(sealed, dict)

    class FakeWhitening:
        payload_sha256 = _h("whitening-payload")
        canonical_train_rows = tuple(range(12_540))

        def __init__(self) -> None:
            self.input_provenance_sha256 = input_provenance_sha256
            self.cache_provenance_sha256 = cache_provenance_sha256
            self.calls = 0

        def to_payload(self) -> dict[str, object]:
            self.calls += 1
            if changed_after_first_round_trip and self.calls > 1:
                return {
                    **sealed,
                    "sealed": "changed-after-model-load",
                }
            return dict(sealed)

    whitening = FakeWhitening()

    class Factory:
        @classmethod
        def from_payload(cls, value: object) -> object:
            assert value == sealed
            return whitening

    monkeypatch.setattr(samga_evaluate, "TrainWhitening", Factory)
    return whitening


@pytest.mark.parametrize(
    "field",
    [
        "source_manifest_sha256",
        "source_payload_sha256",
        "source_payload_path_sha256",
        "source_payload_byte_count_sha256",
        "train_role_sha256",
        "val_dev_role_sha256",
    ],
)
def test_missing_each_source_identity_fails_before_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
) -> None:
    loaded = _checkpoint(tmp_path)
    payload = dict(loaded.payload)
    inputs = dict(payload["input_hashes"])
    del inputs[field]
    _rebind_resolved_candidate(payload, inputs)
    mutated = _mutated_checkpoint(loaded, payload)
    events, _ = _install_runtime(
        monkeypatch,
        tmp_path,
        checkpoint=mutated,
    )

    with pytest.raises(ValueError, match=field):
        samga_evaluate.main(_argv(tmp_path, checkpoint=mutated))

    assert "dataset" not in events


def test_initial_checkpoint_input_is_locked_to_no_initial_sentinel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loaded = _checkpoint(tmp_path)
    payload = dict(loaded.payload)
    inputs = dict(payload["input_hashes"])
    inputs["checkpoint_sha256"] = _h("self-invented-initial-checkpoint")
    _rebind_resolved_candidate(payload, inputs)
    mutated = _mutated_checkpoint(loaded, payload)
    events, _ = _install_runtime(
        monkeypatch,
        tmp_path,
        checkpoint=mutated,
    )

    with pytest.raises(ValueError, match="checkpoint_sha256"):
        samga_evaluate.main(_argv(tmp_path, checkpoint=mutated))

    assert "dataset" not in events


def test_stage2_sha_must_match_fixed_canonical_registry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loaded = _checkpoint(tmp_path, whitening=True)
    _install_fake_whitening(monkeypatch, loaded)
    payload = dict(loaded.payload)
    candidate = dict(payload["candidate_spec"])
    candidate["stage2_config_sha256"] = _h("forged-stage2-registry")
    payload["candidate_spec"] = candidate
    _rebind_resolved_candidate(payload, dict(payload["input_hashes"]))
    mutated = _mutated_checkpoint(loaded, payload)
    events, _ = _install_runtime(
        monkeypatch,
        tmp_path,
        checkpoint=mutated,
    )

    with pytest.raises(ValueError, match="Stage 2 config"):
        samga_evaluate.main(_argv(tmp_path, checkpoint=mutated))

    assert "dataset" not in events


def test_candidate_id_must_match_exact_registry_factor_and_rank_lr(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loaded = _checkpoint(tmp_path, whitening=True)
    payload = dict(loaded.payload)
    candidate = dict(payload["candidate_spec"])
    candidate.update(
        config_id="s2-adapter-r8-lr0.05",
        whitening_config_id="s2-whitening-off",
        whitening_payload=None,
        adapter_kind="adapter",
        adapter_rank=32,
        adapter_lr_ratio=0.1,
    )
    payload["candidate_spec"] = candidate
    _rebind_resolved_candidate(payload, dict(payload["input_hashes"]))
    mutated = _mutated_checkpoint(loaded, payload)
    events, _ = _install_runtime(
        monkeypatch,
        tmp_path,
        checkpoint=mutated,
    )

    with pytest.raises(ValueError, match="registry"):
        samga_evaluate.main(_argv(tmp_path, checkpoint=mutated))

    assert "dataset" not in events


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("epoch", 59, "epoch 60"),
        ("epoch", True, "epoch.*integer"),
        ("epoch_complete", False, "epoch_complete"),
        ("epoch_complete", 1, "epoch_complete.*boolean"),
    ],
)
def test_official_evaluation_requires_complete_epoch60(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    loaded = _checkpoint(tmp_path)
    payload = dict(loaded.payload)
    if field == "epoch":
        payload["epoch"] = value
    else:
        runtime = dict(payload["runtime_state"])
        runtime[field] = value
        payload["runtime_state"] = runtime
    mutated = _mutated_checkpoint(loaded, payload)
    events, _ = _install_runtime(
        monkeypatch,
        tmp_path,
        checkpoint=mutated,
    )

    with pytest.raises(ValueError, match=message):
        samga_evaluate.main(_argv(tmp_path, checkpoint=mutated))

    assert "dataset" not in events


def test_checkpoint_identity_rejects_boolean_as_integer() -> None:
    with pytest.raises(ValueError, match="subject.*integer"):
        samga_evaluate.checkpoint_identity(
            {"subject": True, "seed": 0},
            subject=1,
            seed=0,
        )


@pytest.mark.parametrize("container", ["candidate", "run"])
def test_nested_schema_integer_rejects_boolean(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    container: str,
) -> None:
    loaded = _checkpoint(tmp_path)
    payload = dict(loaded.payload)
    if container == "candidate":
        _reseal_candidate(payload, schema_version=True)
        candidate = payload["candidate_spec"]
        assert isinstance(candidate, dict)
        _reseal_run(
            payload,
            candidate_spec_sha256=candidate["candidate_spec_sha256"],
        )
    else:
        _reseal_run(payload, schema_version=True)
    mutated = _mutated_checkpoint(loaded, payload)
    events, _ = _install_runtime(
        monkeypatch,
        tmp_path,
        checkpoint=mutated,
    )

    with pytest.raises(ValueError, match="schema_version.*integer"):
        samga_evaluate.main(_argv(tmp_path, checkpoint=mutated))

    assert "dataset" not in events


def test_output_directory_parent_must_equal_candidate_run_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events, _ = _install_runtime(monkeypatch, tmp_path)
    wrong = tmp_path / "forged-run-key" / "saved_checkpoint"

    with pytest.raises(ValueError, match="run_key"):
        samga_evaluate.main(_argv(tmp_path, output_dir=wrong))

    assert "dataset" not in events


def test_fixed_protocol_sha_must_match_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    wrong_protocol = _h("forged-protocol")
    manifest = replace(_manifest(tmp_path), protocol_sha256=wrong_protocol)
    loaded = _checkpoint(tmp_path)
    payload = dict(loaded.payload)
    inputs = dict(payload["input_hashes"])
    inputs["protocol_sha256"] = wrong_protocol
    _rebind_resolved_candidate(payload, inputs)
    _reseal_run(payload, protocol_sha256=wrong_protocol)
    mutated = _mutated_checkpoint(loaded, payload)
    events, _ = _install_runtime(
        monkeypatch,
        tmp_path,
        checkpoint=mutated,
        manifest_identity=manifest,
    )

    with pytest.raises(ValueError, match="fixed protocol"):
        samga_evaluate.main(_argv(tmp_path, checkpoint=mutated))

    assert "dataset" not in events


@pytest.mark.parametrize(
    ("input_hash", "cache_hash", "message"),
    [
        (_h("wrong-manifest"), _h("cache"), "manifest provenance"),
        (_h("manifest"), _h("wrong-cache"), "cache provenance"),
    ],
)
def test_whitening_binds_manifest_and_cache_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    input_hash: str,
    cache_hash: str,
    message: str,
) -> None:
    loaded = _checkpoint(tmp_path, whitening=True)
    _install_fake_whitening(
        monkeypatch,
        loaded,
        input_provenance_sha256=input_hash,
        cache_provenance_sha256=cache_hash,
    )
    events, _ = _install_runtime(
        monkeypatch,
        tmp_path,
        checkpoint=loaded,
    )

    with pytest.raises(ValueError, match=message):
        samga_evaluate.main(_argv(tmp_path, checkpoint=loaded))

    assert "dataset" not in events


def test_whitening_payload_is_rechecked_after_model_state_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loaded = _checkpoint(tmp_path, whitening=True)
    _install_fake_whitening(
        monkeypatch,
        loaded,
        changed_after_first_round_trip=True,
    )
    events, _ = _install_runtime(
        monkeypatch,
        tmp_path,
        checkpoint=loaded,
    )

    with pytest.raises(ValueError, match="changed after model load"):
        samga_evaluate.main(_argv(tmp_path, checkpoint=loaded))

    assert "evaluate" not in events


def test_explicit_averaged_checkpoint_fails_with_missing_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loaded = _checkpoint(tmp_path)
    events, _ = _install_runtime(
        monkeypatch,
        tmp_path,
        checkpoint=loaded,
    )
    calls: list[Path] = []

    def load_averaged(path: Path) -> dict[str, object]:
        calls.append(path)
        return {
            "payload_type": "samga_brain_rw.averaged_checkpoint",
            "model_state_dict": {"weight": torch.ones(1)},
        }

    monkeypatch.setattr(
        samga_evaluate,
        "load_averaged_checkpoint",
        load_averaged,
        raising=False,
    )

    with pytest.raises(
        ValueError,
        match="averaged.*candidate_spec.*run_manifest.*input_hashes.*runtime_state",
    ):
        samga_evaluate.main(
            _argv(
                tmp_path,
                checkpoint=loaded,
                checkpoint_kind="averaged",
            )
        )

    assert calls == [tmp_path / "checkpoint_epoch060.pt"]
    assert "dataset" not in events
