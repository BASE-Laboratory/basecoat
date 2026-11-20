"""Automatic provenance metadata helpers."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__version__ = "0.3.0"


def resolve_git_sha(cwd: Path | None = None) -> str | None:
    """Return the short 12-char git SHA for *cwd* (or None if unavailable)."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            timeout=1.0,
            check=False,
        )
        if proc.returncode == 0:
            sha = proc.stdout.strip()
            return sha if sha else None
    except (OSError, subprocess.SubprocessError):
        return None
    return None


def stamp_provenance(
    existing: Mapping[str, Any] | None = None,
    *,
    cwd: Path | None = None,
) -> dict[str, Any]:
    """Merge caller provenance with basecoat version, git SHA, and UTC timestamp."""
    out: dict[str, Any] = dict(existing) if existing is not None else {}
    out["basecoat_version"] = __version__
    sha = resolve_git_sha(cwd=cwd)
    if sha is not None or "git_sha" not in out:
        out["git_sha"] = sha
    out["written_at_utc"] = datetime.now(timezone.utc).isoformat()
    return out
