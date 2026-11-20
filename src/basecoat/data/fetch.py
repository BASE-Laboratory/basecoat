"""Checksum-verified dataset fetching with Windows-safe atomic replacement."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

from ..exceptions import ChecksumError


def _compute_sha256(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(chunk_bytes)
            if not buf:
                break
            digest.update(buf)
    return digest.hexdigest().lower()


def default_cache_dir() -> Path:
    """Resolve basecoat cache directory, respecting ``BASECOAT_CACHE_DIR``."""
    env_override = os.environ.get("BASECOAT_CACHE_DIR")
    if env_override:
        return Path(env_override).expanduser().resolve()
    return Path.home() / ".cache" / "basecoat"


def fetch_file(
    url: str,
    *,
    sha256: str,
    filename: str | None = None,
    cache_dir: str | Path | None = None,
    timeout_s: float = 60.0,
) -> Path:
    """Fetch *url* into a local cache directory and verify its exact SHA-256 digest.

    Uses Windows-safe descriptor closing prior to ``Path.replace`` so concurrent or
    NTFS filesystem runs never fail with file-sharing violations.
    """
    expected = sha256.strip().lower()
    if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
        raise ValueError(
            f"sha256 must be a 64-character hexadecimal SHA-256 digest, got {sha256!r}."
        )

    target_dir = (
        Path(cache_dir).expanduser().resolve() if cache_dir is not None else default_cache_dir()
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    leaf = (
        filename.strip()
        if filename is not None
        else url.split("?", 1)[0].rstrip("/").split("/")[-1]
    )
    if not leaf or "/" in leaf or "\\" in leaf or leaf in (".", ".."):
        raise ValueError(
            f"Invalid cache filename {leaf!r}; pass a plain filename='...' without path separators."
        )

    dest = target_dir / leaf
    if dest.is_file():
        actual_cached = _compute_sha256(dest)
        if actual_cached == expected:
            return dest
        dest.unlink(missing_ok=True)

    fd, tmp_name = tempfile.mkstemp(prefix=f".{leaf}.", suffix=".part", dir=str(target_dir))
    os.close(fd)
    tmp_path = Path(tmp_name)

    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp, open(tmp_path, "wb") as out_fh:
            shutil.copyfileobj(resp, out_fh, length=1024 * 1024)

        actual = _compute_sha256(tmp_path)
        if actual != expected:
            raise ChecksumError(
                f"SHA-256 mismatch for '{url}': expected {expected}, got {actual}. "
                f"Corrupted temporary file was removed."
            )
        tmp_path.replace(dest)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    return dest
