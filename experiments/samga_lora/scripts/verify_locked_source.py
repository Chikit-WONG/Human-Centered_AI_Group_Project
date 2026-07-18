#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Refuse formal runs after locked source changes")
    parser.add_argument("--locked-config", required=True)
    parser.add_argument("--experiment-root", required=True)
    args = parser.parse_args()
    with open(args.locked_config, "r", encoding="utf-8") as handle:
        locked = json.load(handle)
    if not locked.get("gate_passed"):
        raise RuntimeError("Pilot gate did not pass")
    root = Path(args.experiment_root).resolve()
    expected = locked.get("source_sha256", {})
    if not expected:
        raise ValueError("Locked config has no source hashes")
    mismatches = []
    for relative, expected_hash in expected.items():
        path = root / relative
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            mismatches.append(f"{relative}: {actual_hash} != {expected_hash}")
    if mismatches:
        raise RuntimeError("Locked source changed:\n" + "\n".join(mismatches))


if __name__ == "__main__":
    main()
