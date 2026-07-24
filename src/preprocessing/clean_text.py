"""Basic text preprocessing for the first baseline model."""

from __future__ import annotations

import re

import pandas as pd

SPACE_PATTERN = re.compile(r"\s+")


def normalize_spaces(text: str) -> str:
    """Replace multiple spaces/newlines with one space."""
    if pd.isna(text):
        return ""

    return SPACE_PATTERN.sub(" ", str(text)).strip()


def combine_title_content(title: str, content: str) -> str:
    """Join title and content into one model input text."""
    clean_title = normalize_spaces(title)
    clean_content = normalize_spaces(content)

    if clean_title and clean_content:
        return f"{clean_title}. {clean_content}"
    if clean_title:
        return clean_title
    return clean_content


def prepare_text_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create clean text columns used by the baseline model."""
    clean_dataframe = dataframe.copy()

    clean_dataframe["title_clean"] = clean_dataframe["title"].apply(normalize_spaces)
    clean_dataframe["content_clean"] = clean_dataframe["content"].apply(normalize_spaces)
    clean_dataframe["model_text"] = clean_dataframe.apply(
        lambda row: combine_title_content(row["title_clean"], row["content_clean"]),
        axis=1,
    )

    return clean_dataframe
