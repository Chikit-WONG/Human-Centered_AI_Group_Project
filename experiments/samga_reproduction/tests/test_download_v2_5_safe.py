from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import pytest


REPRODUCTION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRODUCTION_ROOT))

import download_v2_5_safe as downloader  # noqa: E402


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class FakeResponse:
    def __init__(
        self,
        payload: bytes,
        *,
        status_code: int = 200,
        content_range: str | None = None,
    ) -> None:
        self.payload = payload
        self.status_code = status_code
        self.headers = {}
        if content_range is not None:
            self.headers["Content-Range"] = content_range

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        del chunk_size
        yield self.payload


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.urls: list[str] = []

    def get(self, url: str, **kwargs: object) -> FakeResponse:
        del kwargs
        self.urls.append(url)
        if not self.responses:
            raise AssertionError(f"Unexpected request: {url}")
        return self.responses.pop(0)


def configure_tiny_run(
    monkeypatch: pytest.MonkeyPatch,
    output_dir: Path,
    payload: bytes = b"safe",
) -> downloader.Shard:
    shard = downloader.Shard("model-tiny.safetensors", len(payload), sha256(payload))
    monkeypatch.setattr(downloader, "SHARDS", (shard,))
    monkeypatch.setattr(downloader, "SMALL_FILES", ())
    monkeypatch.setattr(
        downloader,
        "parse_args",
        lambda: SimpleNamespace(
            output_dir=str(output_dir),
            workers=1,
            chunk_mib=1,
            retries=1,
        ),
    )
    monkeypatch.setattr(downloader.time, "sleep", lambda _: None)
    return shard


def test_pinned_small_file_metadata_matches_revision_bytes() -> None:
    expected = {
        "config.json": (
            801,
            "4fc4a1187b20575c0da8d27df2ad17f5ad6e8ac1c8b2af707bc8b263bd40c0a2",
        ),
        "configuration_intern_vit.py": (
            5_479,
            "e620864fe9f2ef0104b39ea496cb844e1b363caaf8208e6f0bef1a72f31f00a3",
        ),
        "flash_attention.py": (
            3_370,
            "d84f36949763545b58039d28669f9dc46fcace6c94b796e3f91a92553f5f5cad",
        ),
        "model.safetensors.index.json": (
            43_846,
            "94d376c898c00585a38a588df9ff354fa965eafa9a1d56f69c1c8bad7ad08502",
        ),
        "modeling_intern_vit.py": (
            14_047,
            "56220ba82cb511d51d5f2fa71eebd728b330fbccad9dfb128088a8fcc8f7d260",
        ),
        "preprocessor_config.json": (
            287,
            "0658115064c561026539aeeead9ed3b1a8e0cc90967df8c142849199f955d2b4",
        ),
    }

    actual = {
        spec.filename: (spec.size, spec.sha256) for spec in downloader.SMALL_FILES
    }

    assert downloader.MODEL_REVISION == "9d1a4344077479c93d42584b6941c64d795d508d"
    assert actual == expected


def test_small_file_falls_back_only_after_hash_check(tmp_path: Path) -> None:
    good = b"pinned-content"
    spec = downloader.FileSpec("config.json", len(good), sha256(good))
    session = FakeSession([FakeResponse(b"unpinned-bytes"), FakeResponse(good)])
    output = tmp_path / spec.filename
    output.write_bytes(b"stale")

    record = downloader.fetch_small_file(session, tmp_path, spec)

    assert output.read_bytes() == good
    assert not (tmp_path / f"{spec.filename}.partial").exists()
    assert record == {
        "sha256": spec.sha256,
        "size": spec.size,
        "source": f"{downloader.MIRROR_BASE}/{spec.filename}",
    }
    assert session.urls == [
        f"{downloader.OFFICIAL_BASE}/{spec.filename}",
        f"{downloader.MIRROR_BASE}/{spec.filename}",
    ]


def test_small_file_rejects_all_mismatched_sources(tmp_path: Path) -> None:
    good = b"pinned-content"
    spec = downloader.FileSpec("config.json", len(good), sha256(good))
    session = FakeSession([FakeResponse(b"unpinned-bytes"), FakeResponse(b"mirror-bad-hsh")])

    with pytest.raises(RuntimeError, match="Unable to download"):
        downloader.fetch_small_file(session, tmp_path, spec)

    assert not (tmp_path / spec.filename).exists()
    assert not (tmp_path / f"{spec.filename}.partial").exists()


def test_failed_range_invalidates_old_complete_and_never_exposes_formal_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "model"
    output_dir.mkdir()
    shard = configure_tiny_run(monkeypatch, output_dir)
    provenance = output_dir / "model_provenance.json"
    provenance.write_text('{"complete": true}\n', encoding="utf-8")
    response = FakeResponse(
        b"x",
        status_code=206,
        content_range=f"bytes 0-{shard.size - 1}/{shard.size}",
    )
    monkeypatch.setattr(downloader.requests, "get", lambda *args, **kwargs: response)

    with pytest.raises(RuntimeError, match="Failed range"):
        downloader.main()

    assert not provenance.exists()
    assert not (output_dir / shard.filename).exists()
    assert (output_dir / f"{shard.filename}.partial").exists()


@pytest.mark.parametrize(
    "content_range",
    [
        "bytes 0-3/5",
        "bytes 0-3/*",
        "bytes 0-3/4 trailing",
        "bytes 1-3/4",
        "bytes 0-2/4",
    ],
)
def test_range_requires_exact_start_end_and_total(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    content_range: str,
) -> None:
    output_dir = tmp_path / "model"
    shard = configure_tiny_run(monkeypatch, output_dir)
    response = FakeResponse(b"safe", status_code=206, content_range=content_range)
    monkeypatch.setattr(downloader.requests, "get", lambda *args, **kwargs: response)

    with pytest.raises(RuntimeError, match="Failed range"):
        downloader.main()

    assert not (output_dir / shard.filename).exists()


def test_oversized_range_body_is_rejected_before_pwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "model"
    shard = configure_tiny_run(monkeypatch, output_dir)
    response = FakeResponse(
        b"safe!",
        status_code=206,
        content_range=f"bytes 0-{shard.size - 1}/{shard.size}",
    )
    monkeypatch.setattr(downloader.requests, "get", lambda *args, **kwargs: response)
    pwrite_calls: list[tuple[int, bytes, int]] = []
    real_pwrite = downloader.os.pwrite

    def recording_pwrite(descriptor: int, block: bytes, position: int) -> int:
        pwrite_calls.append((descriptor, block, position))
        return real_pwrite(descriptor, block, position)

    monkeypatch.setattr(downloader.os, "pwrite", recording_pwrite)

    with pytest.raises(RuntimeError, match="Failed range"):
        downloader.main()

    assert pwrite_calls == []
    assert not (output_dir / shard.filename).exists()


def test_short_pwrite_is_retried_until_block_is_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "model"
    payload = b"safe"
    shard = configure_tiny_run(monkeypatch, output_dir, payload)
    response = FakeResponse(
        payload,
        status_code=206,
        content_range=f"bytes 0-{shard.size - 1}/{shard.size}",
    )
    monkeypatch.setattr(downloader.requests, "get", lambda *args, **kwargs: response)
    real_pwrite = downloader.os.pwrite
    write_sizes: list[int] = []

    def short_pwrite(descriptor: int, block: bytes, position: int) -> int:
        count = min(2, len(block))
        write_sizes.append(count)
        return real_pwrite(descriptor, block[:count], position)

    monkeypatch.setattr(downloader.os, "pwrite", short_pwrite)

    downloader.main()

    assert (output_dir / shard.filename).read_bytes() == payload
    assert write_sizes == [2, 2]
    assert not (output_dir / f"{shard.filename}.partial").exists()


@pytest.mark.parametrize(
    "child_name",
    [
        "model-tiny.safetensors",
        "model-tiny.safetensors.partial",
        "model_provenance.json",
        "model_provenance.json.partial",
    ],
)
def test_preexisting_child_symlinks_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    child_name: str,
) -> None:
    output_dir = tmp_path / "model"
    output_dir.mkdir()
    payload = b"safe"
    shard = configure_tiny_run(monkeypatch, output_dir, payload)
    outside = tmp_path / "outside"
    outside.write_bytes(payload)
    if child_name != shard.filename:
        (output_dir / shard.filename).write_bytes(payload)
    (output_dir / child_name).symlink_to(outside)
    monkeypatch.setattr(
        downloader.requests,
        "get",
        lambda *args, **kwargs: pytest.fail("network must not be reached"),
    )

    with pytest.raises(RuntimeError, match="symlink"):
        downloader.main()


def test_complete_provenance_records_verified_small_file_source_and_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "model"
    output_dir.mkdir()
    payload = b"safe"
    shard = configure_tiny_run(monkeypatch, output_dir, payload)
    (output_dir / shard.filename).write_bytes(payload)
    small_payload = b"pinned-config"
    small = downloader.FileSpec(
        "config.json", len(small_payload), sha256(small_payload)
    )
    monkeypatch.setattr(downloader, "SMALL_FILES", (small,))
    session = FakeSession([FakeResponse(small_payload)])
    monkeypatch.setattr(downloader.requests, "Session", lambda: session)

    downloader.main()

    provenance = json.loads(
        (output_dir / "model_provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["complete"] is True
    assert provenance["small_files"][small.filename] == {
        "sha256": small.sha256,
        "size": small.size,
        "source": f"{downloader.OFFICIAL_BASE}/{small.filename}",
    }
    assert provenance["small_file_sha256"] == {small.filename: small.sha256}
    assert provenance["model_weight_sha256"] == {shard.filename: shard.sha256}
    assert not (output_dir / "model_provenance.json.partial").exists()
