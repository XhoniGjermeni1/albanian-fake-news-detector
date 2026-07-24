"""Load the raw Albanian fake news text files into a pandas DataFrame."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)

DEFAULT_DATASET_DIR = Path("data/raw/alb-fake-news-corpus")

LABELS = {
    "true": (0, "real"),
    "fake": (1, "fake"),
}

EXPECTED_COLUMNS = [
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


def read_text_file(file_path: Path) -> str:
    """Read one article file safely."""
    try:
        return file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        LOGGER.warning("Encoding problem in %s; reading with replacement.", file_path)
        return file_path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        LOGGER.warning("Could not read %s: %s", file_path, error)
        return ""


def extract_pair_id(file_path: Path) -> int | None:
    """Use the numeric file name as pair_id, e.g. 12.txt -> 12."""
    if file_path.stem.isdigit():
        return int(file_path.stem)

    LOGGER.warning("Unexpected file name: %s", file_path.name)
    return None


def split_title_content(raw_text: str) -> tuple[str, str]:
    """Treat the first line as title and the remaining lines as content."""
    lines = raw_text.splitlines()
    if not lines:
        return "", ""

    title = lines[0].strip()
    content = "\n".join(lines[1:]).strip()
    return title, content


def article_to_row(file_path: Path, source_split: str, label: int, label_name: str) -> dict:
    """Convert one raw .txt article into one table row."""
    raw_text = read_text_file(file_path)
    title, content = split_title_content(raw_text)
    pair_id = extract_pair_id(file_path)

    if raw_text.strip() == "":
        LOGGER.warning("Empty article file: %s", file_path)
    if title == "":
        LOGGER.warning("Missing title in: %s", file_path)
    if content == "":
        LOGGER.warning("Missing content in: %s", file_path)

    return {
        "article_id": f"{source_split}_{file_path.stem}",
        "pair_id": pair_id,
        "label": label,
        "label_name": label_name,
        "title": title,
        "content": content,
        "raw_text": raw_text,
        "file_path": file_path.as_posix(),
        "source_split": source_split,
    }


def sort_by_pair_id(file_path: Path) -> tuple[bool, int, str]:
    """Sort numeric files by number and keep unusual names at the end."""
    pair_id = extract_pair_id(file_path)
    return (pair_id is None, pair_id or 0, file_path.name)


def load_dataset(dataset_dir: str | Path = DEFAULT_DATASET_DIR) -> pd.DataFrame:
    """Load true and fake articles from the Albanian Fake News Corpus."""
    dataset_dir = Path(dataset_dir)
    full_texts_dir = dataset_dir if dataset_dir.name == "full_texts" else dataset_dir / "full_texts"

    if not full_texts_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {full_texts_dir}")

    rows = []

    for source_split, (label, label_name) in LABELS.items():
        article_dir = full_texts_dir / source_split
        if not article_dir.exists():
            raise FileNotFoundError(f"Expected folder not found: {article_dir}")

        article_files = sorted(article_dir.glob("*.txt"), key=sort_by_pair_id)
        LOGGER.info("Loading %s files from %s", len(article_files), article_dir)

        for file_path in article_files:
            rows.append(article_to_row(file_path, source_split, label, label_name))

    dataframe = pd.DataFrame(rows, columns=EXPECTED_COLUMNS)
    dataframe["pair_id"] = pd.to_numeric(dataframe["pair_id"], errors="coerce").astype("Int64")
    dataframe["label"] = dataframe["label"].astype(int)

    text_columns = ["article_id", "label_name", "title", "content", "raw_text", "file_path", "source_split"]
    dataframe[text_columns] = dataframe[text_columns].fillna("").astype(str)

    return dataframe


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    df = load_dataset()
    print(df.head())
    print(df["label_name"].value_counts())
