"""Build the processed article dataset for Day 1."""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.data.load_dataset import load_dataset
from src.data.validate_dataset import validate_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "alb-fake-news-corpus"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

LOGGER = logging.getLogger(__name__)


def build_dataset(
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
) -> dict[str, Any]:
    """Load, validate, and save the processed Day 1 dataset."""
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading raw dataset from %s", raw_path)
    dataframe = load_dataset(raw_path)

    LOGGER.info("Validating loaded dataset")
    summary = validate_dataset(dataframe, print_report=True)

    csv_path = processed_path / "articles.csv"
    parquet_path = processed_path / "articles.parquet"
    preview_path = processed_path / "articles_preview.csv"

    dataframe.to_csv(csv_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
    dataframe.to_parquet(parquet_path, index=False)

    preview = pd.DataFrame(
        {
            "article_id": dataframe["article_id"],
            "pair_id": dataframe["pair_id"],
            "label": dataframe["label"],
            "label_name": dataframe["label_name"],
            "title": dataframe["title"],
            "content_length": dataframe["content"].fillna("").astype(str).str.len(),
            "raw_text_length": dataframe["raw_text"].fillna("").astype(str).str.len(),
        }
    )
    preview.to_csv(preview_path, index=False, encoding="utf-8-sig")

    LOGGER.info("Saved processed CSV dataset to %s", csv_path)
    LOGGER.info("Saved processed parquet dataset to %s", parquet_path)
    LOGGER.info("Saved CSV preview to %s", preview_path)

    return {
        "dataframe": dataframe,
        "validation_summary": summary,
        "csv_path": csv_path,
        "parquet_path": parquet_path,
        "preview_path": preview_path,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    result = build_dataset()
    print(f"Saved CSV: {result['csv_path']}")
    print(f"Saved parquet: {result['parquet_path']}")
    print(f"Saved preview: {result['preview_path']}")


if __name__ == "__main__":
    main()
