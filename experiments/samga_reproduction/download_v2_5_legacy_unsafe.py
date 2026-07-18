#!/usr/bin/env python3
"""Download and verify the pinned InternViT-6B V2.5 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def fetch_small_file(session: requests.Session, output_dir: Path, filename: str) -> None:
    path = output_dir / filename
    if path.is_file() and path.stat().st_size:
        return
    temporary = path.with_name(f"{path.name}.partial")
    last_error: Exception | None = None
    for base in (OFFICIAL_BASE, MIRROR_BASE):
        try:
            with session.get(f"{base}/{filename}", stream=True, timeout=(30, 180)) as response:
                response.raise_for_status()
                with temporary.open("wb") as handle:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if block:
                            handle.write(block)
            os.replace(temporary, path)
            return
        except Exception as error:
            last_error = error
    raise RuntimeError(f"Unable to download {filename}") from last_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk-mib", type=int, default=128)
    parser.add_argument("--retries", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers <= 0 or args.chunk_mib <= 0 or args.retries <= 0:
        raise ValueError("workers, chunk-mib and retries must be positive")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    for filename in SMALL_FILES:
        fetch_small_file(session, output_dir, filename)

    chunk_bytes = args.chunk_mib * 1024 * 1024
    descriptors: dict[str, int] = {}
    tasks: list[tuple[Shard, int, int]] = []
    for shard in SHARDS:
        path = output_dir / shard.filename
        if (
            path.is_file()
            and path.stat().st_size == shard.size
            and digest(path) == shard.sha256
        ):
            print(f"verified-existing {shard.filename}", flush=True)
            continue
        if path.exists():
            print(f"restart-invalid-partial {shard.filename}", flush=True)
            path.unlink()
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT)
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
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {start}-{end}/"):
                        raise RuntimeError(f"Unexpected Content-Range: {content_range}")
                    for block in response.iter_content(chunk_size=4 * 1024 * 1024):
                        if block:
                            os.pwrite(descriptors[shard.filename], block, position)
                            position += len(block)
                if position - start != expected:
                    raise IOError(f"Short range: {position - start}/{expected}")
                return shard.filename, expected
            except Exception as error:
                last_error = error
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
        actual = digest(path)
        if path.stat().st_size != shard.size or actual != shard.sha256:
            raise ValueError(f"Verification failed for {path}: {actual}")
        hashes[shard.filename] = actual
        print(f"verified {shard.filename} {actual}", flush=True)
    provenance = {
        "schema_version": 1,
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "model_weight_sha256": hashes,
        "small_file_sha256": {
            filename: digest(output_dir / filename) for filename in SMALL_FILES
        },
        "complete": True,
    }
    temporary = output_dir / f"model_provenance.json.partial-{os.getpid()}"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, output_dir / "model_provenance.json")


if __name__ == "__main__":
    main()
