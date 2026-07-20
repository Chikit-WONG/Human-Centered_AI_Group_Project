"""Shared canonical provenance records for development score artifacts."""

from __future__ import annotations

import re
from pathlib import Path

from .brainrw import ManifestIdentity


_RUN_KEY_RE = re.compile(
    r"^stage(?:0|2)__[A-Za-z0-9._-]+__sub-\d{2}__seed-\d+__"
    r"config-[0-9a-f]{64}__inputs-[0-9a-f]{64}$"
)


def development_score_source_records(
    manifest: ManifestIdentity,
    *,
    run_key: str,
) -> list[dict[str, object]]:
    """Build the exact source record shared by all parity emissions."""

    if not isinstance(manifest, ManifestIdentity):
        raise TypeError("manifest must be a verified ManifestIdentity")
    if not isinstance(run_key, str) or _RUN_KEY_RE.fullmatch(run_key) is None:
        raise ValueError("run_key is not a canonical development run key")
    source_payload_path = Path(manifest.source_payload_path)
    if not source_payload_path.is_absolute():
        raise ValueError("source payload path must be absolute")
    if (
        type(manifest.source_payload_byte_count) is not int
        or manifest.source_payload_byte_count <= 0
    ):
        raise ValueError("source payload byte count must be positive")
    return [
        {
            "manifest_sha256": manifest.manifest_sha256,
            "records_sha256": manifest.records_sha256,
            "role": "val-dev",
            "role_payload_sha256": manifest.val_dev_role_sha256,
            "run_key": run_key,
            "source_manifest_sha256": manifest.source_manifest_sha256,
            "source_payload_byte_count": manifest.source_payload_byte_count,
            "source_payload_path": str(source_payload_path),
            "source_payload_sha256": manifest.source_payload_sha256,
        }
    ]
