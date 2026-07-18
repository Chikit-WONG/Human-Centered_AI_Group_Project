#!/usr/bin/env python3
"""Prepare immutable EEG and visual-feature inputs for the released SAMGA code."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


LAYERS = (20, 24, 28, 32, 36)
TRAIN_SHAPE = (1654, 10, 4, 63, 250)
TEST_SHAPE = (200, 1, 80, 63, 250)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.partial-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def save_npy_once(path: Path, values: np.ndarray, expected_shape: tuple[int, ...]) -> None:
    if path.is_file():
        existing = np.load(path, mmap_mode="r")
        if tuple(existing.shape) != expected_shape or existing.dtype != values.dtype:
            raise ValueError(
                f"Existing {path} has {existing.shape}/{existing.dtype}, "
                f"expected {expected_shape}/{values.dtype}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.partial-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    written = np.load(temporary, mmap_mode="r")
    if tuple(written.shape) != expected_shape or written.dtype != values.dtype:
        raise RuntimeError(f"Failed to write expected array to {temporary}")
    os.replace(temporary, path)


def selected_subjects(raw: Iterable[int]) -> list[int]:
    values = sorted(set(int(value) for value in raw))
    if not values or values[0] < 1 or values[-1] > 10:
        raise ValueError("subjects must be non-empty values in 1..10")
    return values


def prepare_eeg(args: argparse.Namespace) -> None:
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()
    subjects = selected_subjects(args.subjects)
    reference_channels: list[str] | None = None
    records: list[dict[str, Any]] = []
    for subject in subjects:
        subject_name = f"sub-{subject:02d}"
        source_dir = source_root / subject_name
        output_dir = output_root / subject_name
        train_source = source_dir / "train.pt"
        test_source = source_dir / "test.pt"
        if not train_source.is_file() or not test_source.is_file():
            raise FileNotFoundError(f"Missing source EEG for {subject_name}")
        train_loaded = torch.load(
            train_source, map_location="cpu", weights_only=False, mmap=True
        )
        test_loaded = torch.load(
            test_source, map_location="cpu", weights_only=False, mmap=True
        )
        train = np.asarray(train_loaded["eeg"])
        test = np.asarray(test_loaded["eeg"])
        if tuple(train.shape) != (16540, 4, 63, 250):
            raise ValueError(f"Unexpected {train_source} EEG shape: {train.shape}")
        if tuple(test.shape) != (200, 80, 63, 250):
            raise ValueError(f"Unexpected {test_source} EEG shape: {test.shape}")
        if train.dtype != np.float16 or test.dtype != np.float16:
            raise ValueError(f"Expected source float16 EEG, got {train.dtype}/{test.dtype}")
        channels = [str(value) for value in train_loaded["ch_names"]]
        if channels != [str(value) for value in test_loaded["ch_names"]]:
            raise ValueError(f"Train/test channel mismatch for {subject_name}")
        if reference_channels is None:
            reference_channels = channels
        elif channels != reference_channels:
            raise ValueError(f"Cross-subject channel mismatch for {subject_name}")
        save_npy_once(output_dir / "train.npy", train.reshape(TRAIN_SHAPE), TRAIN_SHAPE)
        save_npy_once(output_dir / "test.npy", test[:, None, ...], TEST_SHAPE)
        records.append(
            {
                "subject": subject_name,
                "train_source": str(train_source),
                "test_source": str(test_source),
                "train_source_bytes": train_source.stat().st_size,
                "test_source_bytes": test_source.stat().st_size,
                "dtype": "float16",
                "train_shape": list(TRAIN_SHAPE),
                "test_shape": list(TEST_SHAPE),
            }
        )
    if reference_channels is None:
        raise RuntimeError("No EEG subjects were prepared")
    atomic_json(output_root / "info.json", {"ch_names": reference_channels})
    atomic_json(
        output_root / "provenance.json",
        {
            "schema_version": 1,
            "description": "SAMGA layout derived without re-normalization from MVNN-whitened .pt files",
            "subjects": records,
        },
    )


def normalize_chunk(values: np.ndarray, mode: str, eps: float) -> np.ndarray:
    work = values.astype(np.float32, copy=False)
    if mode == "none":
        return work
    if mode == "layernorm":
        mean = work.mean(axis=-1, keepdims=True)
        variance = np.mean((work - mean) ** 2, axis=-1, keepdims=True)
        return (work - mean) / np.sqrt(variance + eps)
    if mode == "rmsnorm":
        return work / np.sqrt(np.mean(work**2, axis=-1, keepdims=True) + eps)
    raise ValueError(f"Unsupported normalization: {mode}")


def write_layer_file(
    source: np.ndarray,
    output: Path,
    *,
    layer_index: int,
    split: str,
    normalization: str,
    eps: float,
    chunk_rows: int,
) -> None:
    object_shape = (1654, 10, 3200) if split == "train" else (200, 1, 3200)
    if output.is_file():
        existing = np.load(output, mmap_mode="r")
        if tuple(existing.shape) != object_shape or existing.dtype != np.float16:
            raise ValueError(f"Existing feature file is incompatible: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.partial-{os.getpid()}")
    flat_rows = object_shape[0] * object_shape[1]
    target = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float16, shape=object_shape
    )
    target_flat = target.reshape(flat_rows, 3200)
    for start in range(0, flat_rows, chunk_rows):
        end = min(start + chunk_rows, flat_rows)
        chunk = np.asarray(source[start:end, layer_index, :])
        target_flat[start:end] = normalize_chunk(chunk, normalization, eps).astype(
            np.float16
        )
    target.flush()
    del target
    os.replace(temporary, output)


def prepare_features(args: argparse.Namespace) -> None:
    train_path = Path(args.train_cache).resolve()
    test_path = Path(args.test_cache).resolve()
    output_dir = Path(args.output_dir).resolve()
    train = np.load(train_path, mmap_mode="r")
    test = np.load(test_path, mmap_mode="r")
    expected_train = (16540, len(args.layers), 3200)
    expected_test = (200, len(args.layers), 3200)
    if tuple(train.shape) != expected_train or tuple(test.shape) != expected_test:
        raise ValueError(
            f"Stacked feature shape mismatch: {train.shape}/{test.shape}, "
            f"expected {expected_train}/{expected_test}"
        )
    if train.dtype != np.float16 or test.dtype != np.float16:
        raise ValueError(f"Expected float16 features, got {train.dtype}/{test.dtype}")
    for index, layer in enumerate(args.layers):
        write_layer_file(
            train,
            output_dir / f"image_train_layer{layer}.npy",
            layer_index=index,
            split="train",
            normalization=args.normalization,
            eps=args.eps,
            chunk_rows=args.chunk_rows,
        )
        write_layer_file(
            test,
            output_dir / f"image_test_layer{layer}.npy",
            layer_index=index,
            split="test",
            normalization=args.normalization,
            eps=args.eps,
            chunk_rows=args.chunk_rows,
        )
    atomic_json(
        output_dir / "provenance.json",
        {
            "schema_version": 1,
            "train_cache": str(train_path),
            "test_cache": str(test_path),
            "source_train_shape": list(train.shape),
            "source_test_shape": list(test.shape),
            "logical_layers": [int(value) for value in args.layers],
            "normalization": args.normalization,
            "normalization_eps": args.eps,
            "dtype": "float16",
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    eeg = subparsers.add_parser("eeg")
    eeg.add_argument("--source-root", required=True)
    eeg.add_argument("--output-root", required=True)
    eeg.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 11)))
    eeg.set_defaults(function=prepare_eeg)
    features = subparsers.add_parser("features")
    features.add_argument("--train-cache", required=True)
    features.add_argument("--test-cache", required=True)
    features.add_argument("--output-dir", required=True)
    features.add_argument("--layers", type=int, nargs="+", default=list(LAYERS))
    features.add_argument(
        "--normalization", choices=("none", "layernorm", "rmsnorm"), default="none"
    )
    features.add_argument("--eps", type=float, default=1e-6)
    features.add_argument("--chunk-rows", type=int, default=2048)
    features.set_defaults(function=prepare_features)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
