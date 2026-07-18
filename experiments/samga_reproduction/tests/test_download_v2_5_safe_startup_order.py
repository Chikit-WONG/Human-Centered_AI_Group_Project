from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPRODUCTION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRODUCTION_ROOT))

import download_v2_5_safe as downloader  # noqa: E402


def test_regular_complete_provenance_is_invalidated_before_shard_symlink_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "model"
    output_dir.mkdir()
    provenance = output_dir / "model_provenance.json"
    provenance.write_text('{"complete": true}\n', encoding="utf-8")
    payload = b"safe"
    shard = downloader.Shard(
        "model-tiny.safetensors",
        len(payload),
        hashlib.sha256(payload).hexdigest(),
    )
    outside = tmp_path / "outside"
    outside.write_bytes(payload)
    (output_dir / shard.filename).symlink_to(outside)
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

    with pytest.raises(RuntimeError, match="symlink"):
        downloader.main()

    assert not provenance.exists()
    assert outside.read_bytes() == payload


def test_provenance_symlink_is_rejected_without_touching_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "model"
    output_dir.mkdir()
    outside = tmp_path / "outside-provenance"
    outside.write_text('{"complete": true}\n', encoding="utf-8")
    provenance = output_dir / "model_provenance.json"
    provenance.symlink_to(outside)
    monkeypatch.setattr(downloader, "SHARDS", ())
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

    with pytest.raises(RuntimeError, match="symlink"):
        downloader.main()

    assert provenance.is_symlink()
    assert outside.read_text(encoding="utf-8") == '{"complete": true}\n'
