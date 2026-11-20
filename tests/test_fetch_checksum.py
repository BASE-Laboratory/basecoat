"""Tests for SHA-256 dataset fetching, cache reuse, corruption recovery, and atomic cleanup."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from basecoat import ChecksumError, fetch_file


def test_fetch_file_verifies_sha256_reuses_cache_and_cleans_up_on_mismatch(tmp_path: Path) -> None:
    remote_dir = tmp_path / "remote"
    cache_dir = tmp_path / "cache"
    remote_dir.mkdir()

    payload = b"basecoat deterministic binary fixture payload\n" * 128
    expected_sha = hashlib.sha256(payload).hexdigest()
    src_file = remote_dir / "fixture.bin"
    src_file.write_bytes(payload)

    # Cross-platform file:// URI (works on both Linux and Windows)
    uri = src_file.resolve().as_uri()

    # 1. Valid SHA-256 download
    cached = fetch_file(uri, sha256=expected_sha, cache_dir=cache_dir)
    assert cached == cache_dir / "fixture.bin"
    assert cached.read_bytes() == payload

    # 2. Cache hit works even if upstream file is removed
    src_file.unlink()
    cached_hit = fetch_file(uri, sha256=expected_sha, cache_dir=cache_dir)
    assert cached_hit == cached

    # 3. Corrupted cache file is automatically evicted and re-fetched
    src_file.write_bytes(payload)
    cached.write_bytes(b"corrupted local bytes")
    repaired = fetch_file(uri, sha256=expected_sha, cache_dir=cache_dir)
    assert repaired.read_bytes() == payload

    # 4. Wrong SHA-256 raises ChecksumError and leaves zero temporary .part files behind
    wrong_sha = "0" * 64
    with pytest.raises(ChecksumError, match="SHA-256 mismatch"):
        fetch_file(uri, sha256=wrong_sha, filename="bad.bin", cache_dir=cache_dir)

    leftovers = list(cache_dir.glob("*.part"))
    assert leftovers == []
    assert not (cache_dir / "bad.bin").exists()
