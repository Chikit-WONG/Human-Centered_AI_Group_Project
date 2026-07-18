"""Canonical UTF-8 hashing primitives for the sealed SAMGA protocol."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Literal


SPLIT_SALT = "AIAA3800-SAMGA-SPLIT-v1\n"
STIMULUS_SALT = "AIAA3800-SAMGA-STIM-v1\n"


def canonical_json_bytes(value: object) -> bytes:
    """Serialize JSON deterministically without a trailing newline."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def concept_digest(concept_id: str) -> str:
    if not isinstance(concept_id, str) or not concept_id:
        raise ValueError("concept_id must be a non-empty string")
    return hashlib.sha256(f"{SPLIT_SALT}{concept_id}".encode("utf-8")).hexdigest()


def stimulus_digest(
    split: Literal["val-dev", "val-confirm"],
    concept_id: str,
    stimulus_id: str,
) -> str:
    if split not in ("val-dev", "val-confirm"):
        raise ValueError("split must be val-dev or val-confirm")
    if not isinstance(concept_id, str) or not concept_id:
        raise ValueError("concept_id must be a non-empty string")
    if not isinstance(stimulus_id, str) or not stimulus_id:
        raise ValueError("stimulus_id must be a non-empty string")
    payload = f"{STIMULUS_SALT}{split}\n{concept_id}\n{stimulus_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def ordered_ids_sha256(ordered_ids: Sequence[str]) -> str:
    values = list(ordered_ids)
    if any(not isinstance(value, str) for value in values):
        raise ValueError("ordered IDs must all be strings")
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()
