"""Simple validation checks for the processed dataset."""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = [
    "article_id",
    "pair_id",
    "label",
    "label_name",
    "title",
    "content",
    "raw_text",
    "file_path",
    "source_split",
]


def numeric_stats(values: pd.Series) -> dict:
    """Return basic statistics for a numeric pandas Series."""
    values = values.fillna(0)
    return {
        "min": int(values.min()),
        "max": int(values.max()),
        "mean": round(float(values.mean()), 2),
        "median": round(float(values.median()), 2),
    }


def text_length_stats(text: pd.Series) -> dict:
    """Return basic character-length statistics for a text column."""
    lengths = text.fillna("").astype(str).str.len()
    return numeric_stats(lengths)


def validate_dataset(
    dataframe: pd.DataFrame,
    short_text_threshold: int = 80,
    print_report: bool = True,
) -> dict:
    """Validate the loaded articles and return a summary dictionary."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    title = dataframe["title"].fillna("").astype(str)
    content = dataframe["content"].fillna("").astype(str)
    raw_text = dataframe["raw_text"].fillna("").astype(str)

    label_counts = dataframe["label"].value_counts().sort_index()
    label_name_counts = dataframe["label_name"].value_counts().sort_index()

    true_pair_ids = set(dataframe.loc[dataframe["label"] == 0, "pair_id"].dropna().astype(int))
    fake_pair_ids = set(dataframe.loc[dataframe["label"] == 1, "pair_id"].dropna().astype(int))

    non_empty_raw_text = raw_text.str.strip()
    duplicate_rows = non_empty_raw_text[non_empty_raw_text != ""].duplicated(keep=False).sum()
    duplicate_groups = non_empty_raw_text[non_empty_raw_text != ""].value_counts().gt(1).sum()

    summary = {
        "total_articles": int(len(dataframe)),
        "label_counts": {int(label): int(count) for label, count in label_counts.items()},
        "label_name_counts": {str(label): int(count) for label, count in label_name_counts.items()},
        "missing_titles": int(title.str.strip().eq("").sum()),
        "missing_contents": int(content.str.strip().eq("").sum()),
        "missing_pair_ids": int(dataframe["pair_id"].isna().sum()),
        "duplicate_raw_text_rows": int(duplicate_rows),
        "duplicate_raw_text_groups": int(duplicate_groups),
        "short_text_threshold": short_text_threshold,
        "short_articles": int(raw_text.str.len().lt(short_text_threshold).sum()),
        "pair_ids_only_true": sorted(true_pair_ids - fake_pair_ids),
        "pair_ids_only_fake": sorted(fake_pair_ids - true_pair_ids),
        "length_stats": {
            "title_chars": text_length_stats(title),
            "content_chars": text_length_stats(content),
            "raw_text_chars": text_length_stats(raw_text),
            "raw_text_words": numeric_stats(raw_text.str.split().str.len()),
        },
    }

    if print_report:
        print_validation_report(summary)

    return summary


def print_validation_report(summary: dict) -> None:
    """Print the validation summary in a readable way."""
    print("=== Dataset validation report ===")
    print(f"Total articles: {summary['total_articles']}")
    print(f"Label counts: {summary['label_counts']}")
    print(f"Label name counts: {summary['label_name_counts']}")
    print(f"Missing titles: {summary['missing_titles']}")
    print(f"Missing contents: {summary['missing_contents']}")
    print(f"Missing pair IDs: {summary['missing_pair_ids']}")
    print(
        f"Duplicate raw texts: {summary['duplicate_raw_text_rows']} rows "
        f"in {summary['duplicate_raw_text_groups']} groups"
    )
    print(f"Short articles: {summary['short_articles']} below {summary['short_text_threshold']} characters")
    print(f"Pair IDs only in true: {summary['pair_ids_only_true']}")
    print(f"Pair IDs only in fake: {summary['pair_ids_only_fake']}")
    print(f"Length stats: {summary['length_stats']}")
