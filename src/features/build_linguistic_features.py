"""Build linguistic feature tables for all articles."""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.features.linguistic_features import extract_features_dataframe
from src.preprocessing.clean_text import prepare_text_dataframe

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DATA_PATH = PROJECT_ROOT / "data" / "interim" / "articles_clean.csv"
FALLBACK_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "articles.csv"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "linguistic_features.csv"
SUMMARY_PATH = PROJECT_ROOT / "reports" / "day3_feature_summary.csv"

LOGGER = logging.getLogger(__name__)


def load_input_dataset() -> pd.DataFrame:
    """Load the clean Day 2 dataset, or rebuild minimal clean columns if needed."""
    data_path = INPUT_DATA_PATH if INPUT_DATA_PATH.exists() else FALLBACK_DATA_PATH
    LOGGER.info("Reading %s", data_path)

    dataframe = pd.read_csv(data_path, encoding="utf-8-sig")
    if "model_text" not in dataframe.columns:
        dataframe = prepare_text_dataframe(dataframe)

    return dataframe


def build_feature_summary(features: pd.DataFrame) -> pd.DataFrame:
    """Compare simple average feature values between real and fake articles."""
    numeric_columns = features.select_dtypes(include="number").columns
    ignored_columns = {"pair_id", "label"}
    feature_columns = [column for column in numeric_columns if column not in ignored_columns]

    summary = features.groupby("label_name")[feature_columns].mean().round(4).T
    summary = summary.reset_index().rename(columns={"index": "feature"})

    if {"real", "fake"}.issubset(summary.columns):
        summary["fake_minus_real"] = (summary["fake"] - summary["real"]).round(4)

    return summary


def build_linguistic_features() -> dict:
    """Extract features for the dataset and save CSV outputs."""
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    dataframe = load_input_dataset()
    features = extract_features_dataframe(dataframe)
    summary = build_feature_summary(features)

    features.to_csv(FEATURES_PATH, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")

    return {
        "input_rows": int(len(dataframe)),
        "feature_rows": int(len(features)),
        "feature_columns": int(len(features.columns)),
        "features_path": FEATURES_PATH,
        "summary_path": SUMMARY_PATH,
        "summary": summary,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    result = build_linguistic_features()

    print("=== Day 3 linguistic features ===")
    print(f"Input rows: {result['input_rows']}")
    print(f"Feature rows: {result['feature_rows']}")
    print(f"Feature columns: {result['feature_columns']}")
    print(f"Features saved: {result['features_path']}")
    print(f"Summary saved: {result['summary_path']}")
    print(result["summary"].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
