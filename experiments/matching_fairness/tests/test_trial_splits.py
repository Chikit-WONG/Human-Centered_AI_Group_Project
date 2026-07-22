import hashlib

import numpy as np
import pytest

from matching_fairness.trial_splits import (
    average_trial_half,
    build_trial_manifest,
    select_duplicate_image_ids,
)


def _sessions(image_count: int = 2) -> np.ndarray:
    return np.tile(np.repeat(np.arange(4), 20), (image_count, 1))


def test_every_session_is_split_ten_ten_without_overlap() -> None:
    image_ids = ("img-0", "img-1")
    sessions = _sessions()

    manifest = build_trial_manifest(image_ids, sessions, seed=42)

    for image_index, image_id in enumerate(image_ids):
        for session in range(4):
            split = manifest["images"][image_id][str(session)]
            a = set(split["a"])
            b = set(split["b"])
            assert len(a) == len(b) == 10
            assert not (a & b)
            assert a | b == set(np.flatnonzero(sessions[image_index] == session))


def test_trial_split_uses_the_specified_sha256_order() -> None:
    image_id = "img-0"
    sessions = _sessions(image_count=1)

    manifest = build_trial_manifest((image_id,), sessions, seed=42)

    trial_indices = np.flatnonzero(sessions[0] == 0).tolist()
    expected = sorted(
        trial_indices,
        key=lambda trial_index: (
            hashlib.sha256(
                (
                    "AIAA3800-DUPLICATE-EEG-v1\n42\n"
                    f"{image_id}\n0\n{trial_index}"
                ).encode("utf-8")
            ).hexdigest(),
            trial_index,
        ),
    )
    split = manifest["images"][image_id]["0"]
    assert split["a"] == expected[:10]
    assert split["b"] == expected[10:]
    assert split["sha256"] == {
        str(trial_index): hashlib.sha256(
            (
                "AIAA3800-DUPLICATE-EEG-v1\n42\n"
                f"{image_id}\n0\n{trial_index}"
            ).encode("utf-8")
        ).hexdigest()
        for trial_index in trial_indices
    }


def test_integral_float_session_labels_are_normalized() -> None:
    integer_sessions = _sessions(image_count=1)

    expected = build_trial_manifest(("a",), integer_sessions)
    actual = build_trial_manifest(("a",), integer_sessions.astype(np.float32))

    assert actual == expected


def test_half_averages_use_different_real_trials() -> None:
    eeg = np.arange(2 * 80 * 3 * 4).reshape(2, 80, 3, 4)
    sessions = _sessions()
    manifest = build_trial_manifest(("a", "b"), sessions, seed=42)

    a = average_trial_half(eeg, ("a", "b"), manifest, "a")
    b = average_trial_half(eeg, ("a", "b"), manifest, "b")

    assert a.shape == b.shape == (2, 3, 4)
    assert not np.array_equal(a, b)


@pytest.mark.parametrize(
    "sessions, message",
    (
        (np.tile(np.repeat(np.arange(2), 40), (2, 1)), "exactly 4 session IDs"),
        (
            np.tile(np.concatenate([np.repeat(0, 19), np.repeat(1, 21), np.repeat(2, 20), np.repeat(3, 20)]), (2, 1)),
            "exactly 20 trials",
        ),
    ),
)
def test_manifest_rejects_nonconforming_session_layout(
    sessions: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_trial_manifest(("a", "b"), sessions)


def test_manifest_validates_each_image_session_layout() -> None:
    sessions = _sessions()
    sessions[1, 0] = 1

    with pytest.raises(ValueError, match="exactly 20 trials"):
        build_trial_manifest(("a", "b"), sessions)


def test_average_half_follows_image_ids_instead_of_manifest_order() -> None:
    sessions = _sessions()
    manifest = build_trial_manifest(("a", "b"), sessions)
    eeg = np.stack(
        [np.full((80, 2), 20.0), np.full((80, 2), 10.0)],
        axis=0,
    )

    averaged = average_trial_half(eeg, ("b", "a"), manifest, "a")

    np.testing.assert_array_equal(averaged[:, 0], np.array([20.0, 10.0]))


def test_duplicate_selection_uses_exact_sha256_order_and_is_nested() -> None:
    image_ids = tuple(f"image-{index:03d}" for index in range(200))

    selected = select_duplicate_image_ids(image_ids, seed=42)

    expected = tuple(
        sorted(
            image_ids,
            key=lambda image_id: (
                hashlib.sha256(
                    (
                        "AIAA3800-DUPLICATE-QUERY-v1\n42\n"
                        f"{image_id}"
                    ).encode("utf-8")
                ).hexdigest(),
                image_id,
            ),
        )[:20]
    )
    assert selected == expected
    assert selected[:10] == select_duplicate_image_ids(image_ids)[:10]


def test_duplicate_selection_requires_twenty_unique_image_ids() -> None:
    with pytest.raises(ValueError, match="at least 20 unique"):
        select_duplicate_image_ids(tuple(f"image-{index}" for index in range(19)))
