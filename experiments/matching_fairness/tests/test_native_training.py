from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import subprocess
import textwrap

import pytest
import torch

from matching_fairness.native_training import (
    EpochRecord,
    NativeTrainConfig,
    select_validation_checkpoint,
    train_native,
)
from matching_fairness.provenance import inspect_checkout, sha256_file


FAKE_URL = "https://example.invalid/official-eeg-image-decode.git"


@dataclasses.dataclass(frozen=True)
class FakeNativeInputs:
    checkout: Path
    source_lock: Path
    training_eeg: Path
    training_features: Path


def _git(checkout: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-C", str(checkout), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _write_fake_upstream(checkout: Path) -> None:
    sources = {
        "Retrieval/train_unified.py": "raise RuntimeError('train_unified is sealed')\n",
        "source_origin.txt": "1\n",
        "models/atms.py": (
            "from pathlib import Path\n"
            "ORIGIN = int((Path(__file__).resolve().parents[1] / "
            "'source_origin.txt').read_text())\n"
        ),
        "models/loss.py": (
            "from pathlib import Path\n"
            "ORIGIN = int((Path(__file__).resolve().parents[1] / "
            "'source_origin.txt').read_text())\n"
        ),
        "eegdatasets.py": r'''
            from pathlib import Path
            import torch
            from torch.utils.data import Dataset

            class EEGDataset(Dataset):
                def __init__(
                    self,
                    data_path,
                    img_dir_training=None,
                    img_dir_test=None,
                    feature_type="ViT-H-14",
                    features_dir=None,
                    features_path=None,
                    latent_dir=".",
                    preloaded_features=None,
                    exclude_subject=None,
                    subjects=None,
                    train=True,
                    time_window=None,
                    classes=None,
                    pictures=None,
                    avg_trials=False,
                ):
                    assert train is True
                    assert img_dir_test is None
                    assert avg_trials is True
                    assert subjects == ["sub-08"]
                    assert feature_type == "ViT-H-14"
                    assert features_path is None
                    assert features_dir is None
                    assert preloaded_features is not None
                    eeg_path = (
                        Path(data_path)
                        / "sub-08"
                        / "preprocessed_eeg_training.npy"
                    )
                    assert eeg_path.is_file()
                    image_root = Path(img_dir_training)
                    class_dirs = sorted(path for path in image_root.iterdir() if path.is_dir())
                    assert len(class_dirs) == 1_654
                    assert all(len(tuple(path.glob("*.jpg"))) == 10 for path in class_dirs)
                    assert all(
                        image.stat().st_size == 0
                        for class_dir in class_dirs
                        for image in class_dir.glob("*.jpg")
                    )
                    self.data = torch.zeros(16_540, 1)
                    self.labels = torch.arange(1_654).repeat_interleave(10)
                    self.text = [path.name for path in class_dirs]
                    self.img = [
                        str(image)
                        for class_dir in class_dirs
                        for image in sorted(class_dir.glob("*.jpg"))
                    ]
                    self.text_features = preloaded_features["text_features"]
                    self.img_features = preloaded_features["img_features"]
                    if eeg_path.read_bytes() == b"truncate text features":
                        self.text_features = self.text_features[:1]

                def __len__(self):
                    return len(self.data)

                def __getitem__(self, index):
                    text_index = index // 10
                    return (
                        self.data[index],
                        self.labels[index],
                        self.text[text_index],
                        self.text_features[text_index],
                        self.img[index],
                        self.img_features[index],
                    )
        ''',
        "encoder_utils.py": r'''
            def stratified_condition_split(
                n_classes=1654,
                conditions_per_class=10,
                trials_per_condition=4,
                val_ratio=0.1,
                seed=42,
            ):
                assert n_classes == 1_654
                assert conditions_per_class == 10
                assert trials_per_condition == 1
                assert val_ratio == 0.1
                assert seed == 42
                val_indices = list(range(0, 16_540, 10))
                val_set = set(val_indices)
                train_indices = [i for i in range(16_540) if i not in val_set]
                return train_indices, val_indices
        ''',
        "Retrieval/eeg_encoders.py": r'''
            import torch
            from models.loss import ORIGIN

            class FakeEncoder(torch.nn.Module):
                def __init__(self, encoder_type):
                    super().__init__()
                    self.encoder_type = encoder_type
                    self.weight = torch.nn.Parameter(torch.zeros(()))
                    self.register_buffer("forward_arity", torch.zeros((), dtype=torch.long))
                    self.register_buffer("source_origin", torch.tensor(ORIGIN))

                def forward(self, *arguments):
                    expected = 1 if self.encoder_type == "NICE" else 2
                    if len(arguments) != expected:
                        raise AssertionError(
                            f"{self.encoder_type} expected {expected} arguments, "
                            f"received {len(arguments)}"
                        )
                    self.forward_arity.fill_(len(arguments))
                    return arguments[0] * self.weight

            def build_encoder(
                encoder_type,
                n_chans=63,
                n_times=250,
                joint_train=False,
                **kwargs,
            ):
                assert encoder_type in {"NICE", "ATMS"}
                assert n_chans == 63
                assert n_times == 250
                assert joint_train is False
                assert kwargs == {}
                return FakeEncoder(encoder_type)
        ''',
        "Retrieval/retrieval_engine.py": r'''
            import importlib
            from pathlib import Path
            import random
            import numpy as np
            import torch

            _train_calls = 0
            _val_calls = 0

            class EMA:
                def __init__(self, model, decay=0.999):
                    assert decay == 0.999
                    self.decay = decay
                    self.shadow = {
                        name: parameter.data.clone()
                        for name, parameter in model.named_parameters()
                        if parameter.requires_grad
                    }
                    self.backup = {}

                def update(self, model):
                    for name, parameter in model.named_parameters():
                        if parameter.requires_grad:
                            self.shadow[name].mul_(self.decay).add_(
                                parameter.data, alpha=1 - self.decay
                            )

                def apply_shadow(self, model):
                    self.backup = {}
                    for name, parameter in model.named_parameters():
                        if parameter.requires_grad:
                            self.backup[name] = parameter.data.clone()
                            parameter.data.copy_(self.shadow[name])

                def restore(self, model):
                    for name, parameter in model.named_parameters():
                        if parameter.requires_grad:
                            parameter.data.copy_(self.backup[name])
                    self.backup = {}

            def _forward(sub, model, use_subject_id):
                eeg = torch.ones(2, 1, device=model.weight.device)
                if use_subject_id:
                    assert sub == "sub-08"
                    subject_ids = torch.full(
                        (2,), 7, dtype=torch.long, device=model.weight.device
                    )
                    return model(eeg, subject_ids)
                return model(eeg)

            def _assert_official_package_paths():
                checkout = Path(__file__).resolve().parents[1]
                expected = {
                    "models": (checkout / "models",),
                    "Retrieval": (checkout / "Retrieval",),
                }
                for name, expected_paths in expected.items():
                    package = importlib.import_module(name)
                    actual_paths = tuple(
                        Path(path).resolve() for path in package.__path__
                    )
                    spec_paths = tuple(
                        Path(path).resolve()
                        for path in package.__spec__.submodule_search_locations
                    )
                    assert actual_paths == expected_paths
                    assert spec_paths == expected_paths

            def train_epoch(
                sub,
                eeg_model,
                dataloader,
                optimizer,
                device,
                text_features_all,
                img_features_all,
                config,
                use_subject_id=False,
                normalize_feats=False,
                ema=None,
                logit_scale_type="exp",
            ):
                global _train_calls
                _train_calls += 1
                atms = importlib.import_module("models.atms")
                assert atms.ORIGIN == eeg_model.source_origin.item()
                _assert_official_package_paths()
                assert isinstance(optimizer, torch.optim.AdamW)
                assert optimizer.param_groups[0]["lr"] == 3e-4
                assert dataloader.batch_size == 1_024
                assert dataloader.drop_last is True
                assert dataloader.dataset.dataset.text_features is text_features_all
                assert dataloader.dataset.dataset.img_features is img_features_all
                assert use_subject_id == (eeg_model.encoder_type == "ATMS")
                assert normalize_feats == (eeg_model.encoder_type == "ATMS")
                assert ema is not None
                assert logit_scale_type == "exp"
                _forward(sub, eeg_model, use_subject_id)
                with torch.no_grad():
                    eeg_model.weight.add_(1.0)
                ema.update(eeg_model)
                seeded_noise = random.random() + np.random.random() + torch.rand(()).item()
                train_loss = (0.1 if _train_calls == 1 else 0.9) + seeded_noise / 100
                return train_loss, 0.5, torch.zeros(2, 1)

            def compute_val_loss(
                sub,
                eeg_model,
                val_dataloader,
                device,
                use_subject_id=False,
                normalize_feats=False,
                logit_scale_type="exp",
            ):
                global _val_calls
                _val_calls += 1
                assert val_dataloader.batch_size == 1_024
                assert val_dataloader.drop_last is False
                assert use_subject_id == (eeg_model.encoder_type == "ATMS")
                assert normalize_feats == (eeg_model.encoder_type == "ATMS")
                assert logit_scale_type == "exp"
                _forward(sub, eeg_model, use_subject_id)
                return 0.4 if _val_calls == 1 else 0.2

            def train_loop(*args, **kwargs):
                raise AssertionError("sealed wrapper must not call train_loop")

            def evaluate(*args, **kwargs):
                raise AssertionError("sealed wrapper must not evaluate test data")
        ''',
    }
    for relative, source in sources.items():
        path = checkout / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source).lstrip(), encoding="utf-8")


@pytest.fixture()
def fake_native_inputs(tmp_path: Path) -> FakeNativeInputs:
    checkout = tmp_path / "official-checkout"
    subprocess.run(
        ["git", "init", str(checkout)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    _git(checkout, "config", "user.email", "test@example.com")
    _git(checkout, "config", "user.name", "Test")
    _git(checkout, "remote", "add", "origin", FAKE_URL)
    _write_fake_upstream(checkout)
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-m", "complete fake upstream")
    _git(checkout, "checkout", "--detach", "HEAD")

    lock = inspect_checkout(
        checkout,
        expected_url=FAKE_URL,
        expected_branch="develop",
    )
    source_lock = tmp_path / "upstream_lock.json"
    source_lock.write_text(
        json.dumps(lock.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    training_eeg = (
        tmp_path
        / "assets/Preprocessed_data_250Hz/sub-08/preprocessed_eeg_training.npy"
    )
    training_eeg.parent.mkdir(parents=True)
    training_eeg.write_bytes(b"sealed fake training EEG")
    training_features = tmp_path / "assets/ViT-H-14_features_train.pt"
    torch.save(
        {
            "img_features": torch.ones(16_540, 1),
            "text_features": torch.ones(16_540, 1),
        },
        training_features,
    )
    return FakeNativeInputs(
        checkout=checkout,
        source_lock=source_lock,
        training_eeg=training_eeg,
        training_features=training_features,
    )


def _config(
    inputs: FakeNativeInputs,
    output_dir: Path,
    *,
    model: str = "nice",
    epochs: int = 2,
) -> NativeTrainConfig:
    return NativeTrainConfig(
        source_checkout=inputs.checkout,
        source_lock=inputs.source_lock,
        training_eeg=inputs.training_eeg,
        training_features=inputs.training_features,
        output_dir=output_dir,
        model=model,
        subject="sub-08",
        epochs=epochs,
    )


def test_validation_loss_selects_lowest_then_earliest() -> None:
    records = [
        EpochRecord(epoch=1, val_loss=0.4, checkpoint=Path("1.pth")),
        EpochRecord(epoch=2, val_loss=0.2, checkpoint=Path("2.pth")),
        EpochRecord(epoch=3, val_loss=0.2, checkpoint=Path("3.pth")),
    ]

    assert select_validation_checkpoint(records).epoch == 2


def test_training_config_has_no_test_inputs_or_extra_path_mapping() -> None:
    fields = {field.name for field in dataclasses.fields(NativeTrainConfig)}

    assert fields == {
        "source_checkout",
        "source_lock",
        "training_eeg",
        "training_features",
        "output_dir",
        "model",
        "subject",
        "seed",
        "epochs",
        "batch_size",
        "learning_rate",
        "val_ratio",
        "early_stopping_patience",
        "ema_decay",
        "logit_scale_type",
        "avg_trials",
        "n_chans",
        "n_times",
    }


def test_sealed_fake_upstream_flow_selects_validation_and_saves_ema(
    fake_native_inputs: FakeNativeInputs,
    tmp_path: Path,
) -> None:
    result = train_native(_config(fake_native_inputs, tmp_path / "run"))

    assert [record.epoch for record in result.records] == [1, 2]
    assert result.selected.epoch == 2
    assert result.records[0].train_loss < result.records[1].train_loss
    assert result.history_sha256 == sha256_file(result.history)
    first = torch.load(result.records[0].checkpoint, weights_only=True)
    second = torch.load(result.records[1].checkpoint, weights_only=True)
    best = torch.load(result.best_checkpoint, weights_only=True)
    assert first["weight"].item() == pytest.approx(0.001)
    assert second["weight"].item() == pytest.approx(0.002999)
    assert best["weight"].item() == pytest.approx(second["weight"].item())
    assert sorted(path.name for path in result.checkpoint_dir.glob("epoch_*.pth")) == [
        "epoch_0001.pth",
        "epoch_0002.pth",
    ]
    manifest = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert manifest["selection"] == {
        "checkpoint": "epoch_0002.pth",
        "epoch": 2,
        "val_loss": 0.2,
    }
    assert len(manifest["checkpoints"]) == 2
    assert manifest["inputs"] == {
        "training_eeg": {
            "name": "preprocessed_eeg_training.npy",
            "sha256": sha256_file(fake_native_inputs.training_eeg),
        },
        "training_features": {
            "name": "ViT-H-14_features_train.pt",
            "sha256": sha256_file(fake_native_inputs.training_features),
        },
    }


def test_nice_uses_single_argument_forward_signature(
    fake_native_inputs: FakeNativeInputs,
    tmp_path: Path,
) -> None:
    result = train_native(_config(fake_native_inputs, tmp_path / "nice"))
    state = torch.load(result.best_checkpoint, weights_only=True)

    assert state["forward_arity"].item() == 1


def test_atm_s_uses_subject_id_forward_signature_and_normalized_features(
    fake_native_inputs: FakeNativeInputs,
    tmp_path: Path,
) -> None:
    result = train_native(
        _config(fake_native_inputs, tmp_path / "atm-s", model="atm_s")
    )
    state = torch.load(result.best_checkpoint, weights_only=True)

    assert state["forward_arity"].item() == 2


def test_patience_stops_after_exactly_ten_formal_non_improvements(
    fake_native_inputs: FakeNativeInputs,
    tmp_path: Path,
) -> None:
    result = train_native(_config(fake_native_inputs, tmp_path / "patience", epochs=500))

    assert len(result.records) == 12
    assert result.records[-1].epoch == 12
    assert result.selected.epoch == 2
    assert len(tuple(result.checkpoint_dir.glob("epoch_*.pth"))) == 12


def test_same_seed_restart_has_identical_history_hash(
    fake_native_inputs: FakeNativeInputs,
    tmp_path: Path,
) -> None:
    first = train_native(_config(fake_native_inputs, tmp_path / "restart-a"))
    second = train_native(_config(fake_native_inputs, tmp_path / "restart-b"))

    assert first.history.read_bytes() == second.history.read_bytes()
    assert first.history_sha256 == second.history_sha256


def test_source_lock_is_validated_before_official_modules_are_loaded(
    fake_native_inputs: FakeNativeInputs,
    tmp_path: Path,
) -> None:
    (fake_native_inputs.checkout / "Retrieval/retrieval_engine.py").write_text(
        "raise AssertionError('dirty module was imported')\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="clean worktree"):
        train_native(_config(fake_native_inputs, tmp_path / "dirty"))


def test_cli_maps_only_formal_protocol_and_training_inputs(tmp_path: Path) -> None:
    from matching_fairness.config import Protocol
    from scripts.train_native import native_config_from_protocol

    protocol = Protocol.load(
        Path("experiments/matching_fairness/configs/protocol_sub08_seed42.json")
    )
    config = native_config_from_protocol(
        protocol=protocol,
        source_checkout=tmp_path / "checkout",
        source_lock=tmp_path / "upstream_lock.json",
        training_eeg=tmp_path / "preprocessed_eeg_training.npy",
        training_features=tmp_path / "ViT-H-14_features_train.pt",
        output_dir=tmp_path / "output",
        model="nice",
    )

    assert config.subject == "sub-08"
    assert config.seed == 42
    assert config.epochs == 500
    assert config.batch_size == 1_024
    assert config.learning_rate == 3e-4
    assert config.val_ratio == 0.1
    assert config.early_stopping_patience == 10
    assert config.ema_decay == 0.999
    assert config.avg_trials is True


def test_post_materialization_text_features_must_keep_all_training_rows(
    fake_native_inputs: FakeNativeInputs,
    tmp_path: Path,
) -> None:
    fake_native_inputs.training_eeg.write_bytes(b"truncate text features")

    with pytest.raises(ValueError, match="text_features length must be 16,540"):
        train_native(_config(fake_native_inputs, tmp_path / "truncated"))


def test_existing_stale_output_is_rejected_without_modification(
    fake_native_inputs: FakeNativeInputs,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "stale-output"
    stale = output_dir / "checkpoints/nice/epoch_0020.pth"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale checkpoint bytes")
    before = {
        path.relative_to(output_dir): path.read_bytes()
        for path in output_dir.rglob("*")
        if path.is_file()
    }

    with pytest.raises(FileExistsError, match="output directory already exists"):
        train_native(_config(fake_native_inputs, output_dir))

    after = {
        path.relative_to(output_dir): path.read_bytes()
        for path in output_dir.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_existing_empty_output_directory_is_rejected_without_modification(
    fake_native_inputs: FakeNativeInputs,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "empty-output"
    output_dir.mkdir()

    with pytest.raises(FileExistsError, match="output directory already exists"):
        train_native(_config(fake_native_inputs, output_dir))

    assert tuple(output_dir.iterdir()) == ()


def _make_additional_fake_inputs(root: Path, *, source_origin: int) -> FakeNativeInputs:
    root.mkdir(parents=True)
    checkout = root / "official-checkout"
    subprocess.run(
        ["git", "init", str(checkout)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    _git(checkout, "config", "user.email", "test@example.com")
    _git(checkout, "config", "user.name", "Test")
    _git(checkout, "remote", "add", "origin", FAKE_URL)
    _write_fake_upstream(checkout)
    (checkout / "source_origin.txt").write_text(
        f"{source_origin}\n", encoding="utf-8"
    )
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-m", f"fake upstream {source_origin}")
    _git(checkout, "checkout", "--detach", "HEAD")

    lock = inspect_checkout(
        checkout,
        expected_url=FAKE_URL,
        expected_branch="develop",
    )
    source_lock = root / "upstream_lock.json"
    source_lock.write_text(
        json.dumps(lock.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    training_eeg = (
        root
        / "assets/Preprocessed_data_250Hz/sub-08/preprocessed_eeg_training.npy"
    )
    training_eeg.parent.mkdir(parents=True)
    training_eeg.write_bytes(b"sealed fake training EEG")
    training_features = root / "assets/ViT-H-14_features_train.pt"
    torch.save(
        {
            "img_features": torch.ones(16_540, 1),
            "text_features": torch.ones(16_540, 1),
        },
        training_features,
    )
    return FakeNativeInputs(
        checkout=checkout,
        source_lock=source_lock,
        training_eeg=training_eeg,
        training_features=training_features,
    )


def test_locked_import_context_is_hermetic_and_restores_foreign_modules(
    fake_native_inputs: FakeNativeInputs,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    from types import ModuleType

    def is_official_namespace(name: str) -> bool:
        return name in {
            "eegdatasets",
            "encoder_utils",
            "Retrieval",
            "models",
        } or name.startswith(("Retrieval.", "models."))

    for name in tuple(sys.modules):
        if is_official_namespace(name):
            monkeypatch.delitem(sys.modules, name, raising=False)
    foreign_models = ModuleType("models")
    foreign_models.__file__ = str(tmp_path / "foreign/models/__init__.py")
    foreign_models.__path__ = [str(fake_native_inputs.checkout / "models")]
    foreign_loss = ModuleType("models.loss")
    foreign_loss.__file__ = str(tmp_path / "foreign/models/loss.py")
    foreign_loss.ORIGIN = -1
    foreign_models.loss = foreign_loss
    monkeypatch.setitem(sys.modules, "models", foreign_models)
    monkeypatch.setitem(sys.modules, "models.loss", foreign_loss)
    expected_ambient = {
        name: module
        for name, module in sys.modules.items()
        if is_official_namespace(name)
    }

    first = train_native(_config(fake_native_inputs, tmp_path / "hermetic-first"))
    first_state = torch.load(first.best_checkpoint, weights_only=True)
    assert first_state["source_origin"].item() == 1
    assert {
        name: module
        for name, module in sys.modules.items()
        if is_official_namespace(name)
    } == expected_ambient
    assert sys.modules["models"] is foreign_models
    assert sys.modules["models.loss"] is foreign_loss

    second_inputs = _make_additional_fake_inputs(
        tmp_path / "second-upstream", source_origin=2
    )
    second = train_native(_config(second_inputs, tmp_path / "hermetic-second"))
    second_state = torch.load(second.best_checkpoint, weights_only=True)
    assert second_state["source_origin"].item() == 2
    assert {
        name: module
        for name, module in sys.modules.items()
        if is_official_namespace(name)
    } == expected_ambient
    assert sys.modules["models"] is foreign_models
    assert sys.modules["models.loss"] is foreign_loss


def test_unloaded_foreign_regular_models_package_never_executes(
    fake_native_inputs: FakeNativeInputs,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    def is_official_namespace(name: str) -> bool:
        return name in {
            "eegdatasets",
            "encoder_utils",
            "Retrieval",
            "models",
        } or name.startswith(("Retrieval.", "models."))

    for name in tuple(sys.modules):
        if is_official_namespace(name):
            monkeypatch.delitem(sys.modules, name, raising=False)

    foreign_root = tmp_path / "ambient"
    foreign_models = foreign_root / "models"
    foreign_models.mkdir(parents=True)
    marker = foreign_root / "foreign_models_executed.txt"
    (foreign_models / "__init__.py").write_text(
        "from pathlib import Path\n"
        "marker = Path(__file__).resolve().parents[1] / "
        "\"foreign_models_executed.txt\"\n"
        "marker.write_text(\"init\", encoding=\"utf-8\")\n",
        encoding="utf-8",
    )
    (foreign_models / "loss.py").write_text(
        "from pathlib import Path\n"
        "marker = Path(__file__).resolve().parents[1] / "
        "\"foreign_models_executed.txt\"\n"
        "with marker.open(\"a\", encoding=\"utf-8\") as stream:\n"
        "    stream.write(\"loss\")\n"
        "ORIGIN = -1\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(foreign_root))
    expected_path = list(sys.path)
    expected_modules = {
        name: module
        for name, module in sys.modules.items()
        if is_official_namespace(name)
    }

    import_error = None
    try:
        result = train_native(
            _config(fake_native_inputs, tmp_path / "unloaded-foreign")
        )
    except ImportError as error:
        import_error = error

    marker_content = marker.read_text(encoding="utf-8") if marker.exists() else ""
    assert not marker.exists(), (
        "foreign regular models package executed before origin rejection: "
        f"{marker_content!r}; {import_error!r}"
    )
    if import_error is not None:
        raise import_error
    state = torch.load(result.best_checkpoint, weights_only=True)
    assert state["source_origin"].item() == 1
    assert sys.path == expected_path
    assert {
        name: module
        for name, module in sys.modules.items()
        if is_official_namespace(name)
    } == expected_modules
