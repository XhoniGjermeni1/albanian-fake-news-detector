"""Shared data preparation for leakage-safe model evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.features.linguistic_features import extract_linguistic_features
from src.preprocessing.clean_text import combine_title_content

LENGTH_LABELS = [
    "very_short_le_60",
    "short_61_120",
    "medium_121_250",
    "long_gt_250",
]

LENGTH_DISPLAY = {
    "very_short_le_60": "Shumë të shkurtër (<=60)",
    "short_61_120": "Të shkurtër (61-120)",
    "medium_121_250": "Mesatarë (121-250)",
    "long_gt_250": "Të gjatë (>250)",
}


def refresh_model_text(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Rebuild model text with the authoritative preprocessing function."""
    result = dataframe.copy().reset_index(drop=True)
    current_text = pd.Series(
        [
            combine_title_content(row.title, row.content)
            for row in result.itertuples(index=False)
        ],
        dtype="object",
    )
    stale_rows = 0
    if "model_text" in result.columns:
        stale_rows = int(
            result["model_text"].astype(str).reset_index(drop=True).ne(current_text).sum()
        )
    result["model_text"] = current_text
    return result, stale_rows


def exclude_train_duplicates_from_test(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Exclude exact train-text copies from an evaluation test set."""
    train_texts = set(train_data["model_text"])
    duplicate_mask = test_data["model_text"].isin(train_texts)
    excluded_ids = test_data.loc[duplicate_mask, "article_id"].astype(str).tolist()
    clean_test = test_data.loc[~duplicate_mask].reset_index(drop=True)
    return clean_test, excluded_ids


def assign_length_groups(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Assign the fixed word-count groups used since Day 12."""
    result = dataframe.copy()
    result["length_group"] = pd.cut(
        result["word_count"],
        bins=[-np.inf, 60, 120, 250, np.inf],
        labels=LENGTH_LABELS,
        ordered=True,
    )
    return result


def add_word_counts(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add current word counts and the fixed length groups."""
    result = dataframe.copy().reset_index(drop=True)
    result["word_count"] = [
        int(extract_linguistic_features(row.title, row.content)["word_count"])
        for row in result.itertuples(index=False)
    ]
    return assign_length_groups(result)


def build_leakage_safe_groups(dataframe: pd.DataFrame) -> np.ndarray:
    """Keep pair IDs and exact duplicate texts in the same group."""
    parent = list(range(len(dataframe)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first_index: int, second_index: int) -> None:
        first_root = find(first_index)
        second_root = find(second_index)
        if first_root != second_root:
            parent[second_root] = first_root

    for column in ["pair_id", "model_text"]:
        first_index_by_value: dict[str, int] = {}
        for index, value in enumerate(dataframe[column].astype(str)):
            if value in first_index_by_value:
                union(index, first_index_by_value[value])
            else:
                first_index_by_value[value] = index

    roots = np.asarray([find(index) for index in range(len(dataframe))])
    group_codes, _ = pd.factorize(roots)
    return group_codes


def build_group_safe_folds(
    train: pd.DataFrame,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, list[dict]]:
    """Create five stratified folds without pair/text group leakage."""
    groups = build_leakage_safe_groups(train)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(splitter.split(train["model_text"], train["label"], groups=groups))
    audit: list[dict] = []
    for fold_number, (fit_index, validation_index) in enumerate(folds, start=1):
        overlap = set(groups[fit_index]) & set(groups[validation_index])
        if overlap:
            raise RuntimeError(f"Fold {fold_number} contains leakage groups.")
        validation_labels = train.iloc[validation_index]["label"]
        audit.append(
            {
                "fold": fold_number,
                "fit_rows": int(len(fit_index)),
                "validation_rows": int(len(validation_index)),
                "fit_groups": int(len(np.unique(groups[fit_index]))),
                "validation_groups": int(len(np.unique(groups[validation_index]))),
                "overlapping_groups": 0,
                "validation_real": int(validation_labels.eq(0).sum()),
                "validation_fake": int(validation_labels.eq(1).sum()),
            }
        )
    return folds, groups, audit
