from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from matching_fairness.trial_splits import build_trial_manifest
from main.data import load_things_brain_dataset
from scripts.evaluate_retrieval import (
    build_brainrw_score_artifact,
    load_trial_indices_by_image,
    parse_args,
)


def _write_brain_file(
    root: Path,
    *,
    split: str = "test",
    trial_count: int = 8,
    four_dimensional: bool = True,
) -> torch.Tensor:
    subject_dir = root / "sub-08"
    subject_dir.mkdir(parents=True)
    if four_dimensional:
        eeg = torch.arange(
            2 * trial_count * 3 * 4,
            dtype=torch.float32,
        ).reshape(2, trial_count, 3, 4)
        images = np.array(
            [
                ["image-0.jpg"] * trial_count,
                ["image-1.jpg"] * trial_count,
            ],
            dtype=object,
        )
    else:
        eeg = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
        images = np.array(["image-0.jpg", "image-1.jpg"], dtype=object)
    torch.save(
        {
            "eeg": eeg,
            "label": torch.arange(2),
            "img": images,
        },
        subject_dir / f"{split}.pt",
    )
    return eeg


def test_explicit_trial_indices_are_averaged_per_image(tmp_path: Path) -> None:
    eeg = _write_brain_file(tmp_path)
    selection = {"image-0": [0, 2, 4, 6], "image-1": [1, 3, 5, 7]}

    dataset = load_things_brain_dataset(
        data_directory=str(tmp_path),
        split="test",
        subject_ids=8,
        avg_trials=True,
        trial_indices_by_image=selection,
    )

    assert len(dataset) == 2
    np.testing.assert_allclose(
        dataset[0]["eeg"],
        eeg[0, [0, 2, 4, 6]].mean(dim=0).numpy(),
    )
    np.testing.assert_allclose(
        dataset[1]["eeg"],
        eeg[1, [1, 3, 5, 7]].mean(dim=0).numpy(),
    )


@pytest.mark.parametrize(
    "selection, message",
    (
        ({"image-0": [0, 1]}, "exact image-ID coverage"),
        (
            {"image-0": [0, 1], "image-1": [2, 3], "extra": [4, 5]},
            "exact image-ID coverage",
        ),
        (
            {"image-0": [0, 0], "image-1": [1, 2]},
            "unique",
        ),
        (
            {"image-0": [0, 8], "image-1": [1, 2]},
            "range",
        ),
        (
            {"image-0": [0, 1], "image-1": [2, 3], 7: [4, 5]},
            "image-ID",
        ),
    ),
)
def test_explicit_trial_selection_rejects_invalid_mapping(
    tmp_path: Path,
    selection: dict[str, list[int]],
    message: str,
) -> None:
    _write_brain_file(tmp_path)

    with pytest.raises(ValueError, match=message):
        load_things_brain_dataset(
            data_directory=str(tmp_path),
            split="test",
            subject_ids=8,
            avg_trials=True,
            trial_indices_by_image=selection,
        )


def test_explicit_trial_selection_is_test_only(tmp_path: Path) -> None:
    _write_brain_file(tmp_path, split="train")

    with pytest.raises(ValueError, match="split='test'"):
        load_things_brain_dataset(
            data_directory=str(tmp_path),
            split="train",
            subject_ids=8,
            avg_trials=True,
            trial_indices_by_image={"image-0": [0], "image-1": [1]},
        )


def test_explicit_trial_selection_requires_averaging(tmp_path: Path) -> None:
    _write_brain_file(tmp_path)

    with pytest.raises(ValueError, match="avg_trials=True"):
        load_things_brain_dataset(
            data_directory=str(tmp_path),
            split="test",
            subject_ids=8,
            avg_trials=False,
            trial_indices_by_image={"image-0": [0], "image-1": [1]},
        )


def test_explicit_trial_selection_requires_four_dimensions(tmp_path: Path) -> None:
    _write_brain_file(tmp_path, four_dimensional=False)

    with pytest.raises(ValueError, match="4-D"):
        load_things_brain_dataset(
            data_directory=str(tmp_path),
            split="test",
            subject_ids=8,
            avg_trials=True,
            trial_indices_by_image={"image-0": [0], "image-1": [0]},
        )


def test_formal_trial_selection_requires_exact_count(tmp_path: Path) -> None:
    _write_brain_file(tmp_path)

    with pytest.raises(ValueError, match="exactly 40"):
        load_things_brain_dataset(
            data_directory=str(tmp_path),
            split="test",
            subject_ids=8,
            avg_trials=True,
            trial_indices_by_image={
                "image-0": [0, 1, 2, 3],
                "image-1": [4, 5, 6, 7],
            },
            expected_trial_count=40,
        )


def test_existing_averaging_output_is_unchanged_without_selection(
    tmp_path: Path,
) -> None:
    eeg = _write_brain_file(tmp_path)

    dataset = load_things_brain_dataset(
        data_directory=str(tmp_path),
        split="test",
        subject_ids=8,
        avg_trials=True,
    )

    np.testing.assert_array_equal(dataset[0]["eeg"], eeg[0].mean(dim=0).numpy())
    np.testing.assert_array_equal(dataset[1]["eeg"], eeg[1].mean(dim=0).numpy())



def _trial_manifest(path: Path) -> Path:
    import json

    sessions = np.tile(np.repeat(np.arange(4), 20), (2, 1))
    manifest = build_trial_manifest(("image-0", "image-1"), sessions)
    path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return path


def _required_evaluator_arguments() -> list[str]:
    return [
        "--brain-model-path",
        "brain.pt",
        "--vision-adapter-path",
        "adapter",
        "--pretrained-model-name-or-path",
        "clip",
        "--brain-directory",
        "brain",
        "--image-directory",
        "images",
        "--selected-channels",
        "Cz",
        "--metrics-output",
        "metrics.json",
        "--predictions-output",
        "predictions.csv",
    ]


def test_trial_manifest_loads_exact_canonical_half_mapping(tmp_path: Path) -> None:
    manifest = _trial_manifest(tmp_path / "trials.json")

    selection = load_trial_indices_by_image(manifest, "a")

    expected = build_trial_manifest(
        ("image-0", "image-1"),
        np.tile(np.repeat(np.arange(4), 20), (2, 1)),
    )
    expected_selection = {
        image_id: tuple(
            index
            for session in expected["images"][image_id].values()
            for index in session["a"]
        )
        for image_id in expected["image_ids"]
    }
    assert selection == expected_selection


def test_standard_trial_half_rejects_manifest(tmp_path: Path) -> None:
    manifest = _trial_manifest(tmp_path / "trials.json")

    with pytest.raises(SystemExit):
        parse_args(
            [
                *_required_evaluator_arguments(),
                "--trial-half",
                "standard",
                "--trial-split-manifest",
                str(manifest),
            ]
        )


@pytest.mark.parametrize("half", ("a", "b"))
def test_repeated_trial_half_requires_manifest(half: str) -> None:
    with pytest.raises(SystemExit):
        parse_args([*_required_evaluator_arguments(), "--trial-half", half])


def test_standard_parser_defaults_preserve_existing_mode() -> None:
    arguments = parse_args(_required_evaluator_arguments())

    assert arguments.trial_half == "standard"
    assert arguments.trial_split_manifest is None



def test_brainrw_score_artifact_uses_canonical_id_targets(
    tmp_path: Path,
) -> None:
    similarity = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    brain_model = tmp_path / "brain-model"
    adapter = tmp_path / "adapter"
    vision_base = tmp_path / "vision-base"
    for directory, content in (
        (brain_model, b"brain"),
        (adapter, b"adapter"),
        (vision_base, b"vision"),
    ):
        directory.mkdir()
        (directory / "weights.bin").write_bytes(content)
    brain_test = tmp_path / "test.pt"
    brain_test.write_bytes(b"test eeg")
    trial_manifest = _trial_manifest(tmp_path / "trials.json")
    evaluator = tmp_path / "evaluate_retrieval.py"
    evaluator.write_text("# sealed evaluator\n", encoding="utf-8")
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps({"subject": "sub-08", "seed": 42}),
        encoding="utf-8",
    )

    artifact = build_brainrw_score_artifact(
        similarity=similarity,
        query_embeddings=np.eye(2, dtype=np.float32),
        query_ids=("image-0", "image-1"),
        gallery_ids=("image-1", "image-0"),
        trial_half="a",
        brain_model_path=brain_model,
        vision_adapter_path=adapter,
        pretrained_model_path=vision_base,
        brain_test_path=brain_test,
        trial_manifest_path=trial_manifest,
        protocol_path=protocol,
        subject="sub-08",
        seed=42,
        evaluator_path=evaluator,
        top1_count=2,
        top5_count=2,
    )

    assert artifact.target_canonical_ids == ("image-0", "image-1")
    assert set(artifact.metadata["model_content_sha256"]) == {
        "brain_model",
        "vision_adapter",
        "pretrained_vision_base",
    }
    assert len(artifact.metadata["trial_manifest_sha256"]) == 64
    assert len(artifact.metadata["brain_test_sha256"]) == 64
    assert len(artifact.metadata["protocol_sha256"]) == 64
    assert len(artifact.metadata["evaluator_sha256"]) == 64
    assert artifact.metadata["subject"] == "sub-08"
    assert artifact.metadata["seed"] == 42
    assert artifact.gallery_canonical_ids == ("image-1", "image-0")
    assert artifact.metadata["model_slug"] == "our_project"
    assert artifact.metadata["trial_half"] == "a"
    assert len(artifact.metadata["query_embeddings_sha256"]) == 64



def test_evaluator_rejects_trial_manifest_with_tampered_sha(tmp_path: Path) -> None:
    import json

    manifest_path = _trial_manifest(tmp_path / "trials.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["images"]["image-0"]["0"]["sha256"] = {
        str(index): "0" * 64 for index in range(20)
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256"):
        load_trial_indices_by_image(manifest_path, "a")
