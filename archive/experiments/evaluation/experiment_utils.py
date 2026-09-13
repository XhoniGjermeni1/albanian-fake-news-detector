# Përmban llogaritjen e SHA-256 për të provuar që dataset-et dhe artefaktet
# e përdorura nga eksperimentet nuk janë ndryshuar gjatë riprodhimit.

from __future__ import annotations

import hashlib
from pathlib import Path

def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
