#!/usr/bin/env python3
"""Download verified InternViT LFS shards with concurrent HTTP range requests.

The ModelScope copy resolves to LFS objects whose URL hashes match the official
Hugging Face shard SHA-256 values. Small configuration/code files should still
come from the pinned official Hugging Face revision.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import requests


MIRROR_BASE = "https://modelscope.cn/models/thomas/InternViT-6B-448px-V1-5/resolve/master"
OFFICIAL_BASE = (
    "https://huggingface.co/OpenGVLab/InternViT-6B-448px-V1-5/resolve/"
    "03e138c81d3fd538c77439fd43a42c067d827427"
)
SMALL_FILES = (
    "config.json",
    "configuration_intern_vit.py",
    "flash_attention.py",
    "model.safetensors.index.json",
    "modeling_intern_vit.py",
    "preprocessor_config.json",
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
        "331fc0e79147081bb4260491b4db121aaf4252e0f29ed3509ae5df11bd8ae41e",
    ),
    Shard(
        "model-00002-of-00003.safetensors",
        4_937_250_176,
        "4785be9bec8771f0b25a2f33c52fe9e53623068eb0e7d72aa01e410c43a91cbc",
    ),
    Shard(
        "model-00003-of-00003.safetensors",
        1_147_238_088,
        "95fe64ed513580d1fbd4257823adfd5b7b1a283c70cdd2771453443dd1f0b6b6",
    ),
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download pinned InternViT shards")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk-mib", type=int, default=128)
    parser.add_argument("--retries", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers <= 0 or args.chunk_mib <= 0 or args.retries <= 0:
        raise ValueError("workers, chunk-mib, and retries must be positive")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_bytes = args.chunk_mib * 1024 * 1024
    for filename in SMALL_FILES:
        path = output_dir / filename
        if path.is_file() and path.stat().st_size > 0:
            continue
        temporary = path.with_suffix(path.suffix + ".partial")
        with requests.get(
            f"{OFFICIAL_BASE}/{filename}", stream=True, timeout=(30, 180)
        ) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    if block:
                        handle.write(block)
        os.replace(temporary, path)
        print(f"official-file {filename} {digest(path)}", flush=True)
    descriptors: dict[str, int] = {}
    tasks: list[tuple[Shard, int, int]] = []
    for shard in SHARDS:
        path = output_dir / shard.filename
        current = path.stat().st_size if path.exists() else 0
        if current == shard.size and digest(path) == shard.sha256:
            print(f"verified-existing {shard.filename}", flush=True)
            continue
        if current >= shard.size:
            raise ValueError(f"Invalid existing size for {path}: {current}/{shard.size}")
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT)
        descriptors[shard.filename] = descriptor
        os.ftruncate(descriptor, shard.size)
        for start in range(current, shard.size, chunk_bytes):
            tasks.append((shard, start, min(start + chunk_bytes, shard.size) - 1))

    lock = threading.Lock()
    completed_bytes = 0
    started = time.time()

    def transfer(task: tuple[Shard, int, int]) -> tuple[str, int]:
        shard, start, end = task
        expected = end - start + 1
        url = f"{MIRROR_BASE}/{shard.filename}"
        last_error: Exception | None = None
        for attempt in range(1, args.retries + 1):
            position = start
            try:
                with requests.get(
                    url,
                    headers={
                        "Range": f"bytes={start}-{end}",
                        "Accept-Encoding": "identity",
                    },
                    stream=True,
                    allow_redirects=True,
                    timeout=(30, 180),
                ) as response:
                    if response.status_code != 206:
                        raise RuntimeError(
                            f"Range request returned HTTP {response.status_code} for {shard.filename}"
                        )
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {start}-{end}/"):
                        raise RuntimeError(f"Unexpected Content-Range: {content_range}")
                    for block in response.iter_content(chunk_size=4 * 1024 * 1024):
                        if block:
                            os.pwrite(descriptors[shard.filename], block, position)
                            position += len(block)
                if position - start != expected:
                    raise IOError(
                        f"Short range for {shard.filename}: {position - start}/{expected}"
                    )
                return shard.filename, expected
            except Exception as error:  # network retries need the original exception context
                last_error = error
                time.sleep(min(5 * attempt, 30))
        raise RuntimeError(f"Failed range {start}-{end} for {shard.filename}") from last_error

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(transfer, task) for task in tasks]
            for future in as_completed(futures):
                filename, count = future.result()
                with lock:
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

    for shard in SHARDS:
        path = output_dir / shard.filename
        actual = digest(path)
        if path.stat().st_size != shard.size or actual != shard.sha256:
            raise ValueError(
                f"Final verification failed for {path}: {path.stat().st_size}/{actual}"
            )
        print(f"verified {shard.filename} {actual}", flush=True)


if __name__ == "__main__":
    main()
