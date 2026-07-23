from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import csv
from dataclasses import dataclass
import importlib
from importlib.machinery import ModuleSpec
from importlib.util import module_from_spec
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import threading
from types import ModuleType
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .provenance import SourceLock, inspect_checkout, sha256_file


ENCODERS = {
    "nice": {
        "encoder_type": "NICE",
        "use_subject_id": False,
        "normalize_feats": False,
    },
    "atm_s": {
        "encoder_type": "ATMS",
        "use_subject_id": True,
        "normalize_feats": True,
    },
}

_N_CLASSES = 1_654
_CONDITIONS_PER_CLASS = 10
_AVERAGED_TRIALS_PER_CONDITION = 1
_N_TRAINING_SAMPLES = _N_CLASSES * _CONDITIONS_PER_CLASS
_FEATURE_WIDTH = 1_024
_TRAINING_EEG_NAME = "preprocessed_eeg_training.npy"
_TRAINING_FEATURES_NAME = "ViT-H-14_features_train.pt"
_OFFICIAL_NAMESPACES = ("eegdatasets", "encoder_utils", "Retrieval", "models")
_OFFICIAL_MODULE_IMPORT_LOCK = threading.RLock()


@dataclass(frozen=True)
class NativeTrainConfig:
    source_checkout: Path
    source_lock: Path
    training_eeg: Path
    training_features: Path
    output_dir: Path
    model: str
    subject: str
    seed: int = 42
    epochs: int = 500
    batch_size: int = 1024
    learning_rate: float = 3e-4
    val_ratio: float = 0.1
    early_stopping_patience: int = 10
    ema_decay: float = 0.999
    logit_scale_type: str = "exp"
    avg_trials: bool = True
    n_chans: int = 63
    n_times: int = 250


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    val_loss: float
    checkpoint: Path
    train_loss: float = math.nan
    train_accuracy: float = math.nan
    checkpoint_sha256: str = ""


@dataclass(frozen=True)
class TrainingResult:
    records: tuple[EpochRecord, ...]
    selected: EpochRecord
    checkpoint_dir: Path
    best_checkpoint: Path
    history: Path
    manifest: Path
    history_sha256: str
    stopped_early: bool


@dataclass(frozen=True)
class _OfficialModules:
    EEGDataset: type
    build_encoder: Any
    train_epoch: Any
    compute_val_loss: Any
    EMA: type
    stratified_condition_split: Any


def select_validation_checkpoint(records: Sequence[EpochRecord]) -> EpochRecord:
    if not records:
        raise ValueError("no epoch records")
    return min(records, key=lambda record: (record.val_loss, record.epoch))


def train_native(config: NativeTrainConfig) -> TrainingResult:
    """Train one official NICE or ATM-S encoder without constructing test data."""
    _validate_config(config)
    source_lock = _validate_source_lock(config.source_checkout, config.source_lock)
    _prepare_output_root(config.output_dir)
    with _official_source_context(config.source_checkout):
        return _train_native_in_context(config, source_lock)


def _train_native_in_context(
    config: NativeTrainConfig, source_lock: SourceLock
) -> TrainingResult:
    modules = _load_official_modules(config.source_checkout, source_lock)
    input_hashes = {
        "training_eeg": sha256_file(config.training_eeg),
        "training_features": sha256_file(config.training_features),
    }
    features = _load_training_features(config.training_features)
    _seed_everything(config.seed)

    dataset = _build_training_dataset(modules, config, features)
    train_indices, val_indices = modules.stratified_condition_split(
        n_classes=_N_CLASSES,
        conditions_per_class=_CONDITIONS_PER_CLASS,
        trials_per_condition=_AVERAGED_TRIALS_PER_CONDITION,
        val_ratio=config.val_ratio,
        seed=config.seed,
    )
    _validate_split(train_indices, val_indices, len(dataset))
    train_loader, val_loader = _make_loaders(
        dataset,
        train_indices,
        val_indices,
        config,
    )

    encoder = ENCODERS[config.model]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = modules.build_encoder(
        encoder["encoder_type"],
        n_chans=config.n_chans,
        n_times=config.n_times,
        joint_train=False,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    ema = modules.EMA(model, decay=config.ema_decay)

    checkpoint_dir = config.output_dir / "checkpoints" / config.model
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    best_checkpoint = checkpoint_dir / "best_val.pth"
    records: list[EpochRecord] = []
    best_val_loss = math.inf
    non_improvements = 0
    stopped_early = False

    for epoch in range(1, config.epochs + 1):
        train_loss, train_accuracy, _ = modules.train_epoch(
            config.subject,
            model,
            train_loader,
            optimizer,
            device,
            features["text_features"],
            features["img_features"],
            config,
            use_subject_id=encoder["use_subject_id"],
            normalize_feats=encoder["normalize_feats"],
            ema=ema,
            logit_scale_type=config.logit_scale_type,
        )
        train_loss = _finite_float(train_loss, "train loss")
        train_accuracy = _finite_float(train_accuracy, "train accuracy")

        ema.apply_shadow(model)
        try:
            val_loss = _finite_float(
                modules.compute_val_loss(
                    config.subject,
                    model,
                    val_loader,
                    device,
                    use_subject_id=encoder["use_subject_id"],
                    normalize_feats=encoder["normalize_feats"],
                    logit_scale_type=config.logit_scale_type,
                ),
                "validation loss",
            )
            checkpoint = checkpoint_dir / f"epoch_{epoch:04d}.pth"
            checkpoint_bytes = _state_dict_bytes(model)
            _atomic_write_bytes(checkpoint, checkpoint_bytes)
            checkpoint_hash = sha256_file(checkpoint)
            records.append(
                EpochRecord(
                    epoch=epoch,
                    val_loss=val_loss,
                    checkpoint=checkpoint,
                    train_loss=train_loss,
                    train_accuracy=train_accuracy,
                    checkpoint_sha256=checkpoint_hash,
                )
            )
            if val_loss < best_val_loss:
                _atomic_write_bytes(best_checkpoint, checkpoint_bytes)
                best_val_loss = val_loss
                non_improvements = 0
            else:
                non_improvements += 1
        finally:
            ema.restore(model)

        if non_improvements >= config.early_stopping_patience:
            stopped_early = True
            break

    selected = select_validation_checkpoint(records)
    if sha256_file(best_checkpoint) != selected.checkpoint_sha256:
        raise RuntimeError("best validation checkpoint does not match selector")
    _ensure_inputs_unchanged(config, input_hashes)

    history = checkpoint_dir / "history.csv"
    _atomic_write_bytes(history, _history_bytes(records))
    history_hash = sha256_file(history)
    manifest = checkpoint_dir / "checkpoint_manifest.json"
    _atomic_write_json(
        manifest,
        _checkpoint_manifest(
            config=config,
            source_lock=source_lock,
            input_hashes=input_hashes,
            records=records,
            selected=selected,
            best_checkpoint=best_checkpoint,
            history_hash=history_hash,
            stopped_early=stopped_early,
        ),
    )
    return TrainingResult(
        records=tuple(records),
        selected=selected,
        checkpoint_dir=checkpoint_dir,
        best_checkpoint=best_checkpoint,
        history=history,
        manifest=manifest,
        history_sha256=history_hash,
        stopped_early=stopped_early,
    )


def _validate_config(config: NativeTrainConfig) -> None:
    if config.model not in ENCODERS:
        raise ValueError(f"model must be one of {tuple(ENCODERS)}")
    if config.subject != "sub-08" or config.seed != 42:
        raise ValueError("native formal training requires sub-08 / seed-42")
    expected = {
        "batch_size": 1024,
        "learning_rate": 3e-4,
        "val_ratio": 0.1,
        "early_stopping_patience": 10,
        "ema_decay": 0.999,
        "logit_scale_type": "exp",
        "avg_trials": True,
        "n_chans": 63,
        "n_times": 250,
    }
    for field, value in expected.items():
        if getattr(config, field) != value:
            raise ValueError(f"formal native training requires {field}={value!r}")
    if isinstance(config.epochs, bool) or not isinstance(config.epochs, int):
        raise ValueError("epochs must be a positive integer")
    if config.epochs <= 0:
        raise ValueError("epochs must be a positive integer")

    paths = {
        "source checkout": config.source_checkout,
        "source lock": config.source_lock,
        "training EEG": config.training_eeg,
        "training features": config.training_features,
    }
    for label, path in paths.items():
        if path.is_symlink():
            raise ValueError(f"{label} must not be a symbolic link: {path}")
    if not config.source_checkout.is_dir():
        raise ValueError(f"source checkout is not a directory: {config.source_checkout}")
    for label in ("source lock", "training EEG", "training features"):
        path = paths[label]
        if not path.is_file():
            raise ValueError(f"{label} is not a file: {path}")
    if config.training_eeg.name != _TRAINING_EEG_NAME:
        raise ValueError(f"training EEG must be named {_TRAINING_EEG_NAME}")
    if config.training_eeg.parent.name != config.subject:
        raise ValueError("training EEG subject directory does not match subject")
    if config.training_features.name != _TRAINING_FEATURES_NAME:
        raise ValueError(f"training features must be named {_TRAINING_FEATURES_NAME}")


def _prepare_output_root(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _validate_source_lock(checkout: Path, manifest: Path) -> SourceLock:
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid source lock: {manifest}") from error
    expected_keys = {"url", "branch", "commit", "checkout_sha256"}
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError("source lock must contain exactly the SourceLock fields")
    if any(not isinstance(payload[key], str) for key in expected_keys):
        raise ValueError("source lock fields must be strings")
    expected = SourceLock(**payload)
    actual = inspect_checkout(
        checkout,
        expected_url=expected.url,
        expected_branch=expected.branch,
    )
    if actual != expected:
        raise ValueError(
            "source checkout does not match source lock: "
            f"expected {expected.to_dict()}, found {actual.to_dict()}"
        )
    return actual


def _is_official_namespace(name: str) -> bool:
    return name in _OFFICIAL_NAMESPACES or name.startswith(("Retrieval.", "models."))


def _verify_official_module_origins(checkout: Path) -> None:
    checkout = checkout.resolve()
    for name, module in tuple(sys.modules.items()):
        if not _is_official_namespace(name):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is not None:
            _verify_official_path(name, module_file, checkout)
            continue
        module_path = getattr(module, "__path__", None)
        paths = tuple(module_path) if module_path is not None else ()
        if not paths:
            raise ImportError(f"official namespace {name} lacks locked checkout paths")
        for path in paths:
            _verify_official_path(name, path, checkout)


def _verify_official_path(name: str, path: Any, checkout: Path) -> None:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(checkout)
    except ValueError as error:
        raise ImportError(
            f"official namespace {name} escaped locked checkout: {resolved}"
        ) from error


def _pin_official_package(name: str, path: Path) -> None:
    location = str(path.resolve())
    specification = ModuleSpec(name, loader=None, is_package=True)
    specification.submodule_search_locations[:] = [location]
    package = module_from_spec(specification)
    sys.modules[name] = package


@contextmanager
def _official_source_context(checkout: Path) -> Iterator[None]:
    checkout = checkout.resolve()
    with _OFFICIAL_MODULE_IMPORT_LOCK:
        original_path = list(sys.path)
        original_dont_write_bytecode = sys.dont_write_bytecode
        ambient_modules = {
            name: module
            for name, module in tuple(sys.modules.items())
            if _is_official_namespace(name)
        }
        for name in ambient_modules:
            sys.modules.pop(name, None)
        _pin_official_package("models", checkout / "models")
        _pin_official_package("Retrieval", checkout / "Retrieval")
        sys.path[:0] = [str(checkout / "Retrieval"), str(checkout)]
        sys.dont_write_bytecode = True
        importlib.invalidate_caches()
        try:
            yield
            _verify_official_module_origins(checkout)
        finally:
            for name in tuple(sys.modules):
                if _is_official_namespace(name):
                    sys.modules.pop(name, None)
            sys.modules.update(ambient_modules)
            sys.path[:] = original_path
            sys.dont_write_bytecode = original_dont_write_bytecode
            importlib.invalidate_caches()


def _load_official_module(path: Path, name: str) -> ModuleType:
    module = importlib.import_module(name)
    module_file = getattr(module, "__file__", None)
    if module_file is None or Path(module_file).resolve() != path.resolve():
        raise ImportError(f"official module resolved outside locked file: {name}")
    return module


def _required_symbol(module: ModuleType, name: str) -> Any:
    value = getattr(module, name, None)
    if value is None or not callable(value):
        raise ImportError(f"official module {module.__file__} lacks callable {name}")
    return value


def _load_official_modules(
    checkout: Path, _source_lock: SourceLock
) -> _OfficialModules:
    datasets = _load_official_module(checkout / "eegdatasets.py", "eegdatasets")
    encoder_utils = _load_official_module(
        checkout / "encoder_utils.py", "encoder_utils"
    )
    encoders = _load_official_module(
        checkout / "Retrieval/eeg_encoders.py", "Retrieval.eeg_encoders"
    )
    engine = _load_official_module(
        checkout / "Retrieval/retrieval_engine.py", "Retrieval.retrieval_engine"
    )
    _verify_official_module_origins(checkout)
    return _OfficialModules(
        EEGDataset=_required_symbol(datasets, "EEGDataset"),
        build_encoder=_required_symbol(encoders, "build_encoder"),
        train_epoch=_required_symbol(engine, "train_epoch"),
        compute_val_loss=_required_symbol(engine, "compute_val_loss"),
        EMA=_required_symbol(engine, "EMA"),
        stratified_condition_split=_required_symbol(
            encoder_utils, "stratified_condition_split"
        ),
    )


def _load_training_features(path: Path) -> dict[str, torch.Tensor]:
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"could not safely load training features: {path}") from error
    if not isinstance(loaded, Mapping) or set(loaded) != {
        "img_features",
        "text_features",
    }:
        raise ValueError(
            "training features must contain exactly img_features and text_features"
        )
    img_features = loaded["img_features"]
    text_features = loaded["text_features"]
    if not isinstance(img_features, torch.Tensor) or not isinstance(
        text_features, torch.Tensor
    ):
        raise ValueError("training features must be tensors")
    if len(img_features) != _N_TRAINING_SAMPLES:
        raise ValueError("training img_features must have 16,540 rows")
    if len(text_features) != _N_CLASSES:
        raise ValueError("training text_features must have 1,654 rows")
    expected_shapes = {
        "img_features": (_N_TRAINING_SAMPLES, _FEATURE_WIDTH),
        "text_features": (_N_CLASSES, _FEATURE_WIDTH),
    }
    for key, feature in (
        ("img_features", img_features),
        ("text_features", text_features),
    ):
        shape = tuple(feature.shape)
        if shape != expected_shapes[key]:
            raise ValueError(
                f"training {key} shape must be {expected_shapes[key]}, found {shape}"
            )
    return {"img_features": img_features, "text_features": text_features}


@contextmanager
def _temporary_training_images() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="matching-fairness-train-images-") as root:
        image_root = Path(root)
        for class_index in range(_N_CLASSES):
            class_dir = image_root / f"{class_index:04d}_class_{class_index:04d}"
            class_dir.mkdir()
            for condition in range(_CONDITIONS_PER_CLASS):
                (class_dir / f"{condition:02d}.jpg").touch()
        yield image_root


def _build_training_dataset(
    modules: _OfficialModules,
    config: NativeTrainConfig,
    features: Mapping[str, torch.Tensor],
) -> Any:
    with _temporary_training_images() as image_root:
        dataset = modules.EEGDataset(
            str(config.training_eeg.parent.parent),
            img_dir_training=str(image_root),
            feature_type="ViT-H-14",
            features_dir=None,
            features_path=None,
            preloaded_features=dict(features),
            subjects=[config.subject],
            train=True,
            avg_trials=config.avg_trials,
        )
        _validate_training_dataset(dataset)
    return dataset


def _validate_training_dataset(dataset: Any) -> None:
    required_lengths = {
        "data": _N_TRAINING_SAMPLES,
        "labels": _N_TRAINING_SAMPLES,
        "img_features": _N_TRAINING_SAMPLES,
        "text_features": _N_CLASSES,
        "text": _N_CLASSES,
        "img": _N_TRAINING_SAMPLES,
    }
    for attribute, expected in required_lengths.items():
        value = getattr(dataset, attribute, None)
        try:
            actual = len(value)
        except TypeError as error:
            raise ValueError(f"official dataset lacks sized {attribute}") from error
        if actual != expected:
            raise ValueError(
                f"official dataset {attribute} length must be {expected:,}, found {actual:,}"
            )
    expected_feature_shapes = {
        "img_features": (_N_TRAINING_SAMPLES, _FEATURE_WIDTH),
        "text_features": (_N_CLASSES, _FEATURE_WIDTH),
    }
    for attribute, expected in expected_feature_shapes.items():
        value = getattr(dataset, attribute)
        shape = tuple(getattr(value, "shape", ()))
        if shape != expected:
            raise ValueError(
                f"official dataset {attribute} shape must be {expected}, found {shape}"
            )


def _validate_split(
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    dataset_length: int,
) -> None:
    train = tuple(train_indices)
    val = tuple(val_indices)
    if len(train) != 14_886 or len(val) != 1_654:
        raise ValueError("official split must contain 14,886 train and 1,654 val rows")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in train + val):
        raise ValueError("official split indices must be integers")
    train_set = set(train)
    val_set = set(val)
    if len(train_set) != len(train) or len(val_set) != len(val):
        raise ValueError("official split indices must be unique")
    if train_set & val_set:
        raise ValueError("official train and validation indices overlap")
    if train_set | val_set != set(range(dataset_length)):
        raise ValueError("official split does not partition the training dataset")
    for class_index in range(_N_CLASSES):
        start = class_index * _CONDITIONS_PER_CLASS
        class_indices = set(range(start, start + _CONDITIONS_PER_CLASS))
        if len(class_indices & val_set) != 1 or len(class_indices & train_set) != 9:
            raise ValueError("official split must hold out one condition per class")


def _make_loaders(
    dataset: Any,
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    config: NativeTrainConfig,
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        generator=generator,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    return train_loader, val_loader


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _finite_float(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _state_dict_bytes(model: torch.nn.Module) -> bytes:
    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    stream = io.BytesIO()
    torch.save(state, stream)
    return stream.getvalue()


def _history_bytes(records: Sequence[EpochRecord]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        (
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "checkpoint",
            "checkpoint_sha256",
        )
    )
    for record in records:
        writer.writerow(
            (
                record.epoch,
                format(record.train_loss, ".17g"),
                format(record.train_accuracy, ".17g"),
                format(record.val_loss, ".17g"),
                record.checkpoint.name,
                record.checkpoint_sha256,
            )
        )
    return stream.getvalue().encode("utf-8")


def _checkpoint_manifest(
    *,
    config: NativeTrainConfig,
    source_lock: SourceLock,
    input_hashes: Mapping[str, str],
    records: Sequence[EpochRecord],
    selected: EpochRecord,
    best_checkpoint: Path,
    history_hash: str,
    stopped_early: bool,
) -> dict[str, Any]:
    encoder = ENCODERS[config.model]
    return {
        "schema_version": 1,
        "model": config.model,
        "encoder_type": encoder["encoder_type"],
        "subject": config.subject,
        "seed": config.seed,
        "source": source_lock.to_dict(),
        "inputs": {
            "training_eeg": {
                "name": config.training_eeg.name,
                "sha256": input_hashes["training_eeg"],
            },
            "training_features": {
                "name": config.training_features.name,
                "sha256": input_hashes["training_features"],
            },
        },
        "hyperparameters": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "val_ratio": config.val_ratio,
            "early_stopping_patience": config.early_stopping_patience,
            "ema_decay": config.ema_decay,
            "logit_scale_type": config.logit_scale_type,
            "avg_trials": config.avg_trials,
            "n_chans": config.n_chans,
            "n_times": config.n_times,
        },
        "encoder_behavior": {
            "use_subject_id": encoder["use_subject_id"],
            "normalize_feats": encoder["normalize_feats"],
        },
        "checkpoints": [
            {
                "epoch": record.epoch,
                "val_loss": record.val_loss,
                "checkpoint": record.checkpoint.name,
                "sha256": record.checkpoint_sha256,
            }
            for record in records
        ],
        "selection": {
            "epoch": selected.epoch,
            "val_loss": selected.val_loss,
            "checkpoint": selected.checkpoint.name,
        },
        "best_checkpoint": {
            "name": best_checkpoint.name,
            "sha256": sha256_file(best_checkpoint),
        },
        "history": {"name": "history.csv", "sha256": history_hash},
        "stopped_early": stopped_early,
    }


def _ensure_inputs_unchanged(
    config: NativeTrainConfig,
    expected_hashes: Mapping[str, str],
) -> None:
    actual = {
        "training_eeg": sha256_file(config.training_eeg),
        "training_features": sha256_file(config.training_features),
    }
    if actual != dict(expected_hashes):
        raise ValueError("training input changed during native training")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    _atomic_write_bytes(path, encoded)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
