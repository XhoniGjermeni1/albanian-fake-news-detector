"""Small I/O and Markdown helpers shared by historical experiments."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 fingerprint of one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def markdown_value(value: object) -> str:
    """Format one value using the historical compact-table convention."""
    if pd.isna(value):
        return "n/a"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value).replace("|", "/").replace("\n", " ")


def dataframe_to_markdown(dataframe: pd.DataFrame, columns: list[str]) -> str:
    """Render a compact table without the optional tabulate dependency."""
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(markdown_value(row[column]) for column in columns) + " |"
        for _, row in dataframe[columns].iterrows()
    ]
    return "\n".join([header, separator, *rows])


def escaped_markdown_value(value: object) -> str:
    """Format one value using the classifier-report pipe convention."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value).replace("|", "\\|")


def escaped_dataframe_to_markdown(
    dataframe: pd.DataFrame,
    columns: list[str],
) -> str:
    """Render a table while escaping pipe characters in values."""
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| "
        + " | ".join(escaped_markdown_value(row[column]) for column in columns)
        + " |"
        for _, row in dataframe[columns].iterrows()
    ]
    return "\n".join([header, separator, *rows])


def format_percent(value: float) -> str:
    """Format one ratio as a percentage with two decimals."""
    return f"{100 * value:.2f}%"
