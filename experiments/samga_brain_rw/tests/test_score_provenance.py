from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import train as samga_train
from samga_brain_rw.brainrw import ManifestIdentity
from samga_brain_rw.score_provenance import (
    development_score_source_records,
)


def _h(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _manifest(tmp_path: Path) -> ManifestIdentity:
    return ManifestIdentity(
        path=(tmp_path / "sub-08_protocol.json").absolute(),
        subject=8,
        manifest_sha256=_h("manifest"),
        protocol_sha256=_h("protocol"),
        records_sha256=_h("records"),
        source_manifest_sha256=_h("source-manifest"),
        source_payload_path=(tmp_path / "sub-08" / "train.pt").absolute(),
        source_payload_sha256=_h("source-payload"),
        source_payload_byte_count=123,
        train_role_sha256=_h("train-role"),
        val_dev_role_sha256=_h("val-dev-role"),
        train_ordered_ids=("train-a",),
        val_dev_ordered_ids=("image-a",),
        train_ordered_ids_sha256=_h("train-ids"),
        val_dev_ordered_ids_sha256=_h("val-dev-ids"),
    )


def _run_key() -> str:
    return (
        "stage2__s2-layernorm-on__sub-08__seed-42__"
        f"config-{'a' * 64}__inputs-{'b' * 64}"
    )


def test_train_and_evaluator_share_exact_score_source_record(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    expected = development_score_source_records(
        manifest,
        run_key=_run_key(),
    )
    assert samga_train._score_source_records(
        manifest,
        run_key=_run_key(),
    ) == expected
    assert expected == [
        {
            "manifest_sha256": _h("manifest"),
            "records_sha256": _h("records"),
            "role": "val-dev",
            "role_payload_sha256": _h("val-dev-role"),
            "run_key": _run_key(),
            "source_manifest_sha256": _h("source-manifest"),
            "source_payload_byte_count": 123,
            "source_payload_path": str(
                (tmp_path / "sub-08" / "train.pt").absolute()
            ),
            "source_payload_sha256": _h("source-payload"),
        }
    ]


@pytest.mark.parametrize(
    "run_key",
    [
        "stage2__candidate__sub-08__seed-42",
        (
            "stage2__candidate__sub-08__seed-42__"
            f"config-{'a' * 64}__inputs-{'G' * 64}"
        ),
        "formal-test",
    ],
)
def test_score_source_record_rejects_noncanonical_run_key(
    tmp_path: Path,
    run_key: str,
) -> None:
    with pytest.raises(ValueError, match="run_key"):
        development_score_source_records(
            _manifest(tmp_path),
            run_key=run_key,
        )
