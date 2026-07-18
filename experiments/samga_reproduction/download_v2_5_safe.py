#!/usr/bin/env python3
"""Safely download and verify the pinned InternViT-6B V2.5 checkpoint.

No artifact is published under its final filename until its exact byte count
and SHA-256 match the metadata pinned below.  A complete provenance record is
likewise published only after every model artifact has been verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import requests


MODEL_REPO = "OpenGVLab/InternViT-6B-448px-V2_5"
MODEL_REVISION = "9d1a4344077479c93d42584b6941c64d795d508d"
MIRROR_BASE = (
    "https://modelscope.cn/models/OpenGVLab/"
    "InternViT-6B-448px-V2_5/resolve/master"
)
OFFICIAL_BASE = f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}"
PROVENANCE_FILENAME = "model_provenance.json"
CONTENT_RANGE_PATTERN = re.compile(r"bytes ([0-9]+)-([0-9]+)/([0-9]+)")


@dataclass(frozen=True)
class FileSpec:
    filename: str
    size: int
    sha256: str


SMALL_FILES = (
    FileSpec(
        "config.json",
        801,
        "4fc4a1187b20575c0da8d27df2ad17f5ad6e8ac1c8b2af707bc8b263bd40c0a2",
    ),
    FileSpec(
        "configuration_intern_vit.py",
        5_479,
        "e620864fe9f2ef0104b39ea496cb844e1b363caaf8208e6f0bef1a72f31f00a3",
    ),
    FileSpec(
        "flash_attention.py",
        3_370,
        "d84f36949763545b58039d28669f9dc46fcace6c94b796e3f91a92553f5f5cad",
    ),
    FileSpec(
        "model.safetensors.index.json",
        43_846,
        "94d376c898c00585a38a588df9ff354fa965eafa9a1d56f69c1c8bad7ad08502",
    ),
    FileSpec(
        "modeling_intern_vit.py",
        14_047,
        "56220ba82cb511d51d5f2fa71eebd728b330fbccad9dfb128088a8fcc8f7d260",
    ),
    FileSpec(
        "preprocessor_config.json",
        287,
        "0658115064c561026539aeeead9ed3b1a8e0cc90967df8c142849199f955d2b4",
    ),
)


@dataclass(frozen=True)
class Shard:
    filename: str
    size: int
    sha256: str


SHARDS = (
    Shard(
        "model-00001-of-00003.safetensors",
        4_988_565_944,
        "9818659d13d932da8bc0c3b8ee15f5b5d68d8c94d66eb525be566066630111da",
    ),
    Shard(
        "model-00002-of-00003.safetensors",
        4_937_250_176,
        "4f0c10e72d6f6513f421baa6ec843d5508657435059c1d18b6b5fd7789f9d5b7",
    ),
    Shard(
        "model-00003-of-00003.safetensors",
        1_147_238_088,
        "d21c4fe0bc4af1425cfae1a59a8f5fbb00fde9d8e2888325a60913ac61b0494d",
    ),
)


def digest(path: Path) -> str:
    if path.is_symlink():
        raise RuntimeError(f"Refusing to hash symlink: {path}")
    value = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def file_is_verified(path: Path, size: int, sha256: str) -> bool:
    if path.is_symlink():
        raise RuntimeError(f"Refusing pre-existing symlink: {path}")
    if not path.exists():
        return False
    if not path.is_file():
        raise RuntimeError(f"Expected a regular file: {path}")
    return path.stat().st_size == size and digest(path) == sha256


def remove_regular_file(path: Path) -> bool:
    if path.is_symlink():
        raise RuntimeError(f"Refusing pre-existing symlink: {path}")
    if not path.exists():
        return False
    if not path.is_file():
        raise RuntimeError(f"Expected a regular file: {path}")
    path.unlink()
    return True


def open_new_file(path: Path, flags: int) -> int:
    return os.open(
        path,
        flags | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )


def write_all(descriptor: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError(f"Short write while creating descriptor {descriptor}")
        remaining = remaining[written:]


def pwrite_all(descriptor: int, payload: bytes, position: int) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.pwrite(descriptor, remaining, position)
        if written <= 0:
            raise OSError(f"Short pwrite at offset {position}")
        position += written
        remaining = remaining[written:]


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def reject_child_symlinks(output_dir: Path) -> None:
    for child in output_dir.iterdir():
        if child.is_symlink():
            raise RuntimeError(f"Refusing pre-existing child symlink: {child}")


def invalidate_provenance(output_dir: Path) -> None:
    removed = remove_regular_file(output_dir / PROVENANCE_FILENAME)
    for temporary in output_dir.glob(f"{PROVENANCE_FILENAME}.partial*"):
        removed = remove_regular_file(temporary) or removed
    if removed:
        fsync_directory(output_dir)


def validate_content_range(
    value: str, expected_start: int, expected_end: int, expected_total: int
) -> None:
    match = CONTENT_RANGE_PATTERN.fullmatch(value)
    if match is None:
        raise RuntimeError(f"Unexpected Content-Range: {value}")
    actual = tuple(int(part) for part in match.groups())
    expected = (expected_start, expected_end, expected_total)
    if actual != expected:
        raise RuntimeError(
            f"Unexpected Content-Range: {value}; expected "
            f"bytes {expected_start}-{expected_end}/{expected_total}"
        )


def fetch_small_file(
    session: requests.Session, output_dir: Path, spec: FileSpec
) -> dict[str, str | int]:
    path = output_dir / spec.filename
    temporary = output_dir / f"{spec.filename}.partial"
    remove_regular_file(temporary)
    if file_is_verified(path, spec.size, spec.sha256):
        return {
            "sha256": spec.sha256,
            "size": spec.size,
            "source": "existing-verified",
        }
    remove_regular_file(path)

    last_error: Exception | None = None
    for base in (OFFICIAL_BASE, MIRROR_BASE):
        source = f"{base}/{spec.filename}"
        try:
            descriptor = open_new_file(temporary, os.O_WRONLY)
            received = 0
            try:
                with session.get(
                    source,
                    headers={"Accept-Encoding": "identity"},
                    stream=True,
                    timeout=(30, 180),
                ) as response:
                    response.raise_for_status()
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if not block:
                            continue
                        if received + len(block) > spec.size:
                            raise IOError(
                                f"Oversized body for {spec.filename}: "
                                f">{spec.size} bytes"
                            )
                        write_all(descriptor, block)
                        received += len(block)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            if received != spec.size:
                raise IOError(
                    f"Short body for {spec.filename}: {received}/{spec.size}"
                )
            actual = digest(temporary)
            if actual != spec.sha256:
                raise ValueError(
                    f"SHA-256 mismatch for {spec.filename}: {actual}"
                )
            os.replace(temporary, path)
            fsync_directory(output_dir)
            return {
                "sha256": actual,
                "size": spec.size,
                "source": source,
            }
        except Exception as error:
            last_error = error
            remove_regular_file(temporary)
    raise RuntimeError(f"Unable to download {spec.filename}") from last_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safely download the pinned InternViT V2.5 checkpoint"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk-mib", type=int, default=128)
    parser.add_argument("--retries", type=int, default=12)
    return parser.parse_args()


def write_complete_provenance(output_dir: Path, provenance: dict[str, object]) -> None:
    temporary = output_dir / f"{PROVENANCE_FILENAME}.partial"
    descriptor = open_new_file(temporary, os.O_WRONLY)
    try:
        encoded = (
            json.dumps(provenance, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        write_all(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, output_dir / PROVENANCE_FILENAME)
    fsync_directory(output_dir)


def main() -> None:
    args = parse_args()
    if args.workers <= 0 or args.chunk_mib <= 0 or args.retries <= 0:
        raise ValueError("workers, chunk-mib and retries must be positive")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    invalidate_provenance(output_dir)
    reject_child_symlinks(output_dir)

    session = requests.Session()
    small_file_records: dict[str, dict[str, str | int]] = {}
    for spec in SMALL_FILES:
        small_file_records[spec.filename] = fetch_small_file(
            session, output_dir, spec
        )

    chunk_bytes = args.chunk_mib * 1024 * 1024
    descriptors: dict[str, int] = {}
    tasks: list[tuple[Shard, int, int]] = []
    for shard in SHARDS:
        path = output_dir / shard.filename
        temporary = output_dir / f"{shard.filename}.partial"
        remove_regular_file(temporary)
        if file_is_verified(path, shard.size, shard.sha256):
            print(f"verified-existing {shard.filename}", flush=True)
            continue
        remove_regular_file(path)
        descriptor = open_new_file(temporary, os.O_RDWR)
        descriptors[shard.filename] = descriptor
        os.ftruncate(descriptor, shard.size)
        for start in range(0, shard.size, chunk_bytes):
            tasks.append((shard, start, min(start + chunk_bytes, shard.size) - 1))

    completed_bytes = 0
    started = time.time()
    progress_lock = threading.Lock()

    def transfer(task: tuple[Shard, int, int]) -> tuple[str, int]:
        shard, start, end = task
        expected = end - start + 1
        last_error: Exception | None = None
        for attempt in range(1, args.retries + 1):
            position = start
            try:
                with requests.get(
                    f"{MIRROR_BASE}/{shard.filename}",
                    headers={
                        "Range": f"bytes={start}-{end}",
                        "Accept-Encoding": "identity",
                    },
                    stream=True,
                    allow_redirects=True,
                    timeout=(30, 180),
                ) as response:
                    if response.status_code != 206:
                        raise RuntimeError(f"Unexpected HTTP {response.status_code}")
                    validate_content_range(
                        response.headers.get("Content-Range", ""),
                        start,
                        end,
                        shard.size,
                    )
                    for block in response.iter_content(
                        chunk_size=4 * 1024 * 1024
                    ):
                        if not block:
                            continue
                        if len(block) > end + 1 - position:
                            raise IOError(
                                f"Oversized range body for {shard.filename}: "
                                f"{position - start + len(block)}/{expected}"
                            )
                        pwrite_all(
                            descriptors[shard.filename], block, position
                        )
                        position += len(block)
                if position - start != expected:
                    raise IOError(
                        f"Short range for {shard.filename}: "
                        f"{position - start}/{expected}"
                    )
                return shard.filename, expected
            except Exception as error:
                last_error = error
                if attempt < args.retries:
                    time.sleep(min(5 * attempt, 30))
        raise RuntimeError(
            f"Failed range {start}-{end} for {shard.filename}"
        ) from last_error

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(transfer, task) for task in tasks]
            for future in as_completed(futures):
                filename, count = future.result()
                with progress_lock:
                    completed_bytes += count
                    elapsed = max(time.time() - started, 1e-6)
                    print(
                        f"range-complete {filename} transferred={completed_bytes} "
                        f"rate_mib_s={completed_bytes / elapsed / 1024 / 1024:.2f}",
                        flush=True,
                    )
    finally:
        for descriptor in descriptors.values():
            os.fsync(descriptor)
            os.close(descriptor)

    hashes: dict[str, str] = {}
    for shard in SHARDS:
        path = output_dir / shard.filename
        if shard.filename in descriptors:
            temporary = output_dir / f"{shard.filename}.partial"
            actual = digest(temporary)
            if (
                temporary.stat().st_size != shard.size
                or actual != shard.sha256
            ):
                raise ValueError(
                    f"Verification failed for {temporary}: "
                    f"{temporary.stat().st_size}/{actual}"
                )
            os.replace(temporary, path)
            fsync_directory(output_dir)
        else:
            actual = digest(path)
            if path.stat().st_size != shard.size or actual != shard.sha256:
                raise ValueError(
                    f"Verification failed for {path}: "
                    f"{path.stat().st_size}/{actual}"
                )
        hashes[shard.filename] = actual
        print(f"verified {shard.filename} {actual}", flush=True)

    for spec in SMALL_FILES:
        if not file_is_verified(
            output_dir / spec.filename, spec.size, spec.sha256
        ):
            raise ValueError(f"Final verification failed for {spec.filename}")

    provenance: dict[str, object] = {
        "schema_version": 1,
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "model_weight_sha256": hashes,
        "small_files": small_file_records,
        "small_file_sha256": {
            spec.filename: spec.sha256 for spec in SMALL_FILES
        },
        "complete": True,
    }
    write_complete_provenance(output_dir, provenance)


if __name__ == "__main__":
    main()
