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


def validate_trial_manifest(
    manifest: Mapping[str, object],
    image_ids: Sequence[str],
) -> dict[str, dict[str, tuple[int, ...]]]:
    """Validate the complete formal seed-42 trial manifest contract."""
    image_ids = _validated_image_ids(image_ids)
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must be a mapping")
    expected_top_keys = {
        "schema_version",
        "algorithm_version",
        "seed",
        "image_ids",
        "images",
    }
    if set(manifest) != expected_top_keys:
        raise ValueError(
            "trial manifest must contain exactly the formal schema keys"
        )
    if manifest.get("schema_version") != 1:
        raise ValueError("trial manifest schema_version must be 1")
    if manifest.get("algorithm_version") != _TRIAL_ALGORITHM:
        raise ValueError(
            f"trial manifest algorithm_version must be {_TRIAL_ALGORITHM}"
        )
    if manifest.get("seed") != 42:
        raise ValueError("formal trial manifest requires seed=42")
    declared_ids = manifest.get("image_ids")
    if not isinstance(declared_ids, list) or tuple(declared_ids) != image_ids:
        raise ValueError(
            "trial manifest ordered canonical image IDs do not match consumer"
        )
    images = manifest.get("images")
    if not isinstance(images, Mapping) or set(images) != set(image_ids):
        raise ValueError("trial manifest image keys do not match canonical image IDs")

    expected_session_keys = {str(value) for value in range(_SESSIONS_PER_IMAGE)}
    expected_trial_indices = set(range(_SESSIONS_PER_IMAGE * _TRIALS_PER_SESSION))
    normalized: dict[str, dict[str, tuple[int, ...]]] = {}
    for image_id in image_ids:
        image_manifest = images[image_id]
        if (
            not isinstance(image_manifest, Mapping)
            or set(image_manifest) != expected_session_keys
        ):
            raise ValueError("trial manifest session keys must be exactly 0,1,2,3")
        selected = {"a": [], "b": []}
        all_indices: list[int] = []
        for session_id in range(_SESSIONS_PER_IMAGE):
            split = image_manifest[str(session_id)]
            if not isinstance(split, Mapping) or set(split) != {"a", "b", "sha256"}:
                raise ValueError(
                    "manifest session must contain exactly a, b, and sha256"
                )
            a = _validated_trial_indices(split["a"], len(expected_trial_indices))
            b = _validated_trial_indices(split["b"], len(expected_trial_indices))
            if len(a) != _TRIALS_PER_HALF_SESSION or len(b) != _TRIALS_PER_HALF_SESSION:
                raise ValueError("each manifest session half must contain 10 trials")
            if set(a).intersection(b):
                raise ValueError("manifest session halves must not overlap")
            session_indices = a + b
            if len(set(session_indices)) != _TRIALS_PER_SESSION:
                raise ValueError("manifest session must contain 20 distinct trials")
            expected_hashes = {
                str(index): _trial_digest(image_id, session_id, index, 42)
                for index in session_indices
            }
            hashes = split["sha256"]
            if not isinstance(hashes, Mapping) or dict(hashes) != expected_hashes:
                raise ValueError("manifest per-trial SHA-256 ledger is invalid")
            ordered = tuple(
                sorted(
                    session_indices,
                    key=lambda index: (expected_hashes[str(index)], index),
                )
            )
            if a != ordered[:_TRIALS_PER_HALF_SESSION] or b != ordered[_TRIALS_PER_HALF_SESSION:]:
                raise ValueError("manifest halves do not follow the specified SHA-256 order")
            selected["a"].extend(a)
            selected["b"].extend(b)
            all_indices.extend(session_indices)
        if set(all_indices) != expected_trial_indices or len(all_indices) != len(
            expected_trial_indices
        ):
            raise ValueError("manifest sessions must account for exactly trials 0..79")
        normalized[image_id] = {
            "a": tuple(selected["a"]),
            "b": tuple(selected["b"]),
        }
    return normalized


def trial_indices_by_image(
    manifest: Mapping[str, object],
    image_ids: Sequence[str],
    half: str,
) -> dict[str, tuple[int, ...]]:
    if half not in {"a", "b"}:
        raise ValueError("half must be a or b")
    validated = validate_trial_manifest(manifest, image_ids)
    return {image_id: validated[image_id][half] for image_id in image_ids}


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
    selected_by_image = trial_indices_by_image(manifest, image_ids, half)
    if eeg.shape[1] != _SESSIONS_PER_IMAGE * _TRIALS_PER_SESSION:
        raise ValueError("formal EEG must contain exactly 80 trials per image")

    averages = []
    for image_index, image_id in enumerate(image_ids):
        averages.append(np.mean(eeg[image_index, selected_by_image[image_id]], axis=0))

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
