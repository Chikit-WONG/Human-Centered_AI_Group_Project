#!/usr/bin/env python3
"""Build the sealed SAMGA train/validation protocol manifests."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from pathlib import Path
from typing import Sequence

from samga_brain_rw.config import ProtocolConfig
from samga_brain_rw.hashing import (
    SPLIT_SALT,
    STIMULUS_SALT,
    canonical_json_bytes,
)
from samga_brain_rw.splits import (
    EXPECTED_CONCEPTS,
    EXPECTED_RECORDS,
    VAL_CONFIRM_CONCEPTS,
    VAL_DEV_CONCEPTS,
    build_subject_protocol_manifest_from_loaded,
    load_source_manifest,
    partition_concepts,
)


def _json_file_bytes(value: object) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _validate_protocol(protocol: ProtocolConfig) -> None:
    split_sizes = protocol.split_sizes
    expected_train = (
        EXPECTED_CONCEPTS - VAL_DEV_CONCEPTS - VAL_CONFIRM_CONCEPTS
    )
    if protocol.split_salt != SPLIT_SALT:
        raise ValueError("protocol split_salt does not match split v1")
    if protocol.stimulus_salt != STIMULUS_SALT:
        raise ValueError("protocol stimulus_salt does not match stimulus v1")
    if protocol.expected_non_test_concepts != EXPECTED_CONCEPTS:
        raise ValueError(
            f"protocol must require exactly {EXPECTED_CONCEPTS} concepts"
        )
    if (
        split_sizes.train,
        split_sizes.val_dev,
        split_sizes.val_confirm,
    ) != (expected_train, VAL_DEV_CONCEPTS, VAL_CONFIRM_CONCEPTS):
        raise ValueError("protocol split sizes do not match the sealed v1 split")


def _render_outputs(
    protocol_path: Path,
    source_manifest_dir: Path,
) -> dict[str, bytes]:
    protocol = ProtocolConfig.from_path(protocol_path)
    _validate_protocol(protocol)
    source_paths = tuple(
        source_manifest_dir / f"sub-{subject:02d}_train.json"
        for subject in range(1, 11)
    )

    # Every source is opened exactly once. No output is considered until all
    # ten strict train-only inputs have validated.
    loaded = tuple(load_source_manifest(path) for path in source_paths)
    reference_hash = loaded[0].records_sha256
    for source in loaded[1:]:
        if source.records_sha256 != reference_hash:
            raise ValueError(
                "cross-subject records_sha256 or record order mismatch"
            )

    assignment = partition_concepts(
        loaded[0].records,
        protocol_config_sha256=protocol.sha256,
    )
    subject_manifests = tuple(
        build_subject_protocol_manifest_from_loaded(source, assignment)
        for source in loaded
    )
    assignment_payload = assignment.to_payload()
    assignment_file_bytes = _json_file_bytes(assignment_payload)

    outputs: dict[str, bytes] = {
        "split_assignment.json": assignment_file_bytes
    }
    subjects: list[dict[str, object]] = []
    for subject, source, manifest in zip(
        range(1, 11), loaded, subject_manifests, strict=True
    ):
        filename = f"sub-{subject:02d}_protocol.json"
        payload_bytes = _json_file_bytes(manifest)
        outputs[filename] = payload_bytes
        subjects.append(
            {
                "protocol_manifest": filename,
                "protocol_manifest_sha256": hashlib.sha256(
                    payload_bytes
                ).hexdigest(),
                "source_manifest_path": str(source.path),
                "source_manifest_sha256": source.raw_sha256,
                "subject_id": source.subject_id,
            }
        )

    ordered_hashes = assignment_payload["ordered_id_sha256"]
    summary = {
        "ordered_id_sha256": ordered_hashes,
        "payload_type": "samga_brain_rw.manifest_summary",
        "protocol_config_path": str(protocol_path),
        "protocol_config_sha256": protocol.sha256,
        "record_count_per_subject": EXPECTED_RECORDS,
        "records_sha256": reference_hash,
        "role_counts": {
            "train": EXPECTED_CONCEPTS
            - VAL_DEV_CONCEPTS
            - VAL_CONFIRM_CONCEPTS,
            "val-confirm": VAL_CONFIRM_CONCEPTS,
            "val-dev": VAL_DEV_CONCEPTS,
        },
        "role_row_counts_per_subject": {
            "train": (
                EXPECTED_CONCEPTS
                - VAL_DEV_CONCEPTS
                - VAL_CONFIRM_CONCEPTS
            )
            * 10,
            "val-confirm": VAL_CONFIRM_CONCEPTS,
            "val-dev": VAL_DEV_CONCEPTS,
        },
        "schema_version": 1,
        "split_assignment_file_sha256": hashlib.sha256(
            assignment_file_bytes
        ).hexdigest(),
        "split_assignment_payload_sha256": assignment.sha256,
        "subject_count": len(subject_manifests),
        "subjects": subjects,
    }
    outputs["manifest_summary.json"] = _json_file_bytes(summary)
    return outputs


def _reuse_existing_identical(
    output_dir: Path,
    outputs: dict[str, bytes],
) -> bool:
    if not output_dir.exists():
        return False
    if not output_dir.is_dir():
        raise FileExistsError(
            f"output conflict: {output_dir} is not a directory"
        )
    missing = [
        name for name in outputs if not (output_dir / name).is_file()
    ]
    if missing:
        raise FileExistsError(
            "existing output is partial, not complete: "
            f"missing {sorted(missing)}"
        )
    actual_names = set(os.listdir(output_dir))
    expected_names = set(outputs)
    if actual_names != expected_names:
        raise FileExistsError(
            "existing output conflicts with the complete expected file set"
        )
    for name, expected_bytes in outputs.items():
        if (output_dir / name).read_bytes() != expected_bytes:
            raise FileExistsError(
                "existing output conflict: reuse requires byte-identical files"
            )
    return True


def _publish_atomic_directory(
    output_dir: Path,
    outputs: dict[str, bytes],
) -> None:
    if _reuse_existing_identical(output_dir, outputs):
        return
    if "manifest_summary.json" not in outputs:
        raise ValueError("manifest_summary.json completion marker is required")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    owns_output = False
    try:
        try:
            os.mkdir(output_dir)
            owns_output = True
        except FileExistsError as exc:
            try:
                if _reuse_existing_identical(output_dir, outputs):
                    return
            except FileExistsError as reuse_error:
                raise FileExistsError(
                    "output conflict: a concurrent publisher created a "
                    "partial or non-identical destination"
                ) from reuse_error
            raise FileExistsError(
                "output conflict: destination appeared during publication"
            ) from exc

        payload_names = sorted(
            name for name in outputs if name != "manifest_summary.json"
        )
        for name in (*payload_names, "manifest_summary.json"):
            destination = output_dir / name
            with destination.open("xb") as handle:
                handle.write(outputs[name])
                handle.flush()
                os.fsync(handle.fileno())

        output_fd = os.open(
            output_dir,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(output_fd)
        finally:
            os.close(output_fd)
        parent_fd = os.open(
            output_dir.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        if owns_output and output_dir.exists():
            shutil.rmtree(output_dir)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    outputs = _render_outputs(
        arguments.protocol,
        arguments.source_manifest_dir,
    )
    _publish_atomic_directory(arguments.output_dir, outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
