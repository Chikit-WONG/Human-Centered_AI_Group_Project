"""Deterministic, session-balanced trial splits for duplicate EEG queries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib

import numpy as np


_TRIAL_ALGORITHM = "AIAA3800-DUPLICATE-EEG-v1"
_DUPLICATE_QUERY_ALGORITHM = "AIAA3800-DUPLICATE-QUERY-v1"
_SESSIONS_PER_IMAGE = 4
_TRIALS_PER_SESSION = 20
_TRIALS_PER_HALF_SESSION = 10
_DUPLICATE_IMAGE_COUNT = 20


def build_trial_manifest(
    image_ids: Sequence[str],
    sessions: np.ndarray,
    seed: int = 42,
) -> dict[str, object]:
    """Assign every image's trials to disjoint, session-balanced A/B halves."""

    image_ids = _validated_image_ids(image_ids)
    seed = _validated_seed(seed)
    sessions = np.asarray(sessions)
    if sessions.ndim != 2 or sessions.shape[0] != len(image_ids):
        raise ValueError("sessions must have one 1-D trial row per image ID")
    if np.issubdtype(sessions.dtype, np.integer):
        sessions = sessions.astype(np.int64, copy=False)
    elif (
        np.issubdtype(sessions.dtype, np.floating)
        and np.isfinite(sessions).all()
        and np.equal(sessions, np.floor(sessions)).all()
    ):
        sessions = sessions.astype(np.int64)
    else:
        raise ValueError("session IDs must be finite integers")

    images: dict[str, dict[str, object]] = {}
    for image_index, image_id in enumerate(image_ids):
        session_row = sessions[image_index]
        session_ids = sorted(int(value) for value in np.unique(session_row))
        if len(session_ids) != _SESSIONS_PER_IMAGE:
            raise ValueError("each image must have exactly 4 session IDs")

        image_manifest: dict[str, object] = {}
        for session_id in session_ids:
            trial_indices = [
                int(index)
                for index in np.flatnonzero(session_row == session_id)
            ]
            if len(trial_indices) != _TRIALS_PER_SESSION:
                raise ValueError(
                    "each image session must contain exactly 20 trials"
                )
            digests = {
                trial_index: _trial_digest(
                    image_id,
                    session_id,
                    trial_index,
                    seed,
                )
                for trial_index in trial_indices
            }
            ordered = sorted(
                trial_indices,
                key=lambda trial_index: (digests[trial_index], trial_index),
            )
            image_manifest[str(session_id)] = {
                "a": ordered[:_TRIALS_PER_HALF_SESSION],
                "b": ordered[_TRIALS_PER_HALF_SESSION:],
                "sha256": {
                    str(trial_index): digests[trial_index]
                    for trial_index in trial_indices
                },
            }
        images[image_id] = image_manifest

    return {
        "schema_version": 1,
        "algorithm_version": _TRIAL_ALGORITHM,
        "seed": seed,
        "image_ids": list(image_ids),
        "images": images,
    }


def average_trial_half(
    eeg: np.ndarray,
    image_ids: Sequence[str],
    manifest: Mapping[str, object],
    half: str,
) -> np.ndarray:
    """Average the 40 real trials assigned to one half for every EEG image."""

    image_ids = _validated_image_ids(image_ids)
    if half not in {"a", "b"}:
        raise ValueError("half must be 'a' or 'b'")
    if not isinstance(eeg, np.ndarray) or eeg.ndim < 2:
        raise ValueError("eeg must be a NumPy array with image and trial axes")
    if eeg.shape[0] != len(image_ids):
        raise ValueError("eeg must have one image row per image ID")
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must be a mapping")
    images = manifest.get("images")
    if not isinstance(images, Mapping):
        raise ValueError("manifest images must be a mapping")

    averages = []
    for image_index, image_id in enumerate(image_ids):
        image_manifest = images.get(image_id)
        if not isinstance(image_manifest, Mapping):
            raise ValueError(f"manifest is missing image ID: {image_id}")
        if len(image_manifest) != _SESSIONS_PER_IMAGE:
            raise ValueError("each manifest image must contain exactly 4 sessions")

        selected: list[int] = []
        all_selected: list[int] = []
        for split in image_manifest.values():
            if not isinstance(split, Mapping):
                raise ValueError("manifest session split must be a mapping")
            a = _validated_trial_indices(split.get("a"), eeg.shape[1])
            b = _validated_trial_indices(split.get("b"), eeg.shape[1])
            if len(a) != _TRIALS_PER_HALF_SESSION or len(b) != _TRIALS_PER_HALF_SESSION:
                raise ValueError("each manifest session half must contain 10 trials")
            if set(a).intersection(b):
                raise ValueError("manifest session halves must not overlap")
            selected.extend(a if half == "a" else b)
            all_selected.extend(a)
            all_selected.extend(b)
        if len(set(all_selected)) != _SESSIONS_PER_IMAGE * _TRIALS_PER_SESSION:
            raise ValueError("manifest sessions must account for 80 distinct trials")
        averages.append(np.mean(eeg[image_index, selected], axis=0))

    return np.stack(averages, axis=0)


def select_duplicate_image_ids(
    image_ids: Sequence[str],
    seed: int = 42,
) -> tuple[str, ...]:
    """Select the shared ordered set of 20 images with real EEG repeats."""

    image_ids = _validated_image_ids(image_ids)
    seed = _validated_seed(seed)
    if len(image_ids) < _DUPLICATE_IMAGE_COUNT:
        raise ValueError("at least 20 unique image IDs are required")
    return tuple(
        sorted(
            image_ids,
            key=lambda image_id: (
                _duplicate_query_digest(image_id, seed),
                image_id,
            ),
        )[:_DUPLICATE_IMAGE_COUNT]
    )


def _trial_digest(
    image_id: str,
    session_id: int,
    trial_index: int,
    seed: int,
) -> str:
    payload = (
        f"{_TRIAL_ALGORITHM}\n{seed}\n"
        f"{image_id}\n{session_id}\n{trial_index}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _duplicate_query_digest(image_id: str, seed: int) -> str:
    payload = f"{_DUPLICATE_QUERY_ALGORITHM}\n{seed}\n{image_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validated_image_ids(image_ids: Sequence[str]) -> tuple[str, ...]:
    values = tuple(image_ids)
    if not values or any(not isinstance(value, str) or not value for value in values):
        raise ValueError("image IDs must be non-empty strings")
    if len(set(values)) != len(values):
        raise ValueError("image IDs must be unique")
    return values


def _validated_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    return int(seed)


def _validated_trial_indices(value: object, trial_count: int) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("manifest trial indices must be sequences")
    indices = tuple(value)
    if any(
        isinstance(index, bool)
        or not isinstance(index, (int, np.integer))
        or not 0 <= int(index) < trial_count
        for index in indices
    ):
        raise ValueError("manifest trial index is invalid")
    result = tuple(int(index) for index in indices)
    if len(set(result)) != len(result):
        raise ValueError("manifest trial indices must be unique")
    return result
