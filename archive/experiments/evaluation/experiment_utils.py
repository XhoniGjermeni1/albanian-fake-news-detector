"""Small integrity helpers shared by historical experiments."""

from __future__ import annotations

import hashlib
from pathlib import Path

def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 fingerprint of one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
