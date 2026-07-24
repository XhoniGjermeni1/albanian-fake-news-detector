from pathlib import Path

import pandas as pd

from src.data.load_dataset import EXPECTED_COLUMNS, load_dataset


def _write_article(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _create_minimal_corpus(base_path: Path) -> Path:
    corpus_path = base_path / "alb-fake-news-corpus"
    _write_article(corpus_path / "full_texts" / "true" / "1.txt", "Titull real\nPermbajtje reale.")
    _write_article(corpus_path / "full_texts" / "fake" / "1.txt", "Titull fake\nPermbajtje fake.")
    return corpus_path


def test_load_dataset_returns_dataframe(tmp_path: Path) -> None:
    corpus_path = _create_minimal_corpus(tmp_path)

    dataframe = load_dataset(corpus_path)

    assert isinstance(dataframe, pd.DataFrame)
    assert len(dataframe) == 2


def test_load_dataset_has_expected_columns(tmp_path: Path) -> None:
    corpus_path = _create_minimal_corpus(tmp_path)

    dataframe = load_dataset(corpus_path)

    assert list(dataframe.columns) == EXPECTED_COLUMNS


def test_load_dataset_labels_are_binary(tmp_path: Path) -> None:
    corpus_path = _create_minimal_corpus(tmp_path)

    dataframe = load_dataset(corpus_path)

    assert set(dataframe["label"]) == {0, 1}
    assert set(dataframe["label_name"]) == {"real", "fake"}


def test_load_dataset_text_fields_are_strings(tmp_path: Path) -> None:
    corpus_path = _create_minimal_corpus(tmp_path)

    dataframe = load_dataset(corpus_path)

    assert dataframe["title"].map(type).eq(str).all()
    assert dataframe["content"].map(type).eq(str).all()
    assert dataframe["raw_text"].map(type).eq(str).all()


def test_load_dataset_extracts_article_and_pair_ids(tmp_path: Path) -> None:
    corpus_path = _create_minimal_corpus(tmp_path)

    dataframe = load_dataset(corpus_path)

    assert dataframe["article_id"].notna().all()
    assert dataframe["pair_id"].notna().all()
    assert set(dataframe["article_id"]) == {"true_1", "fake_1"}
    assert set(dataframe["pair_id"].astype(int)) == {1}


def test_load_dataset_handles_empty_and_malformed_files(tmp_path: Path) -> None:
    corpus_path = _create_minimal_corpus(tmp_path)
    _write_article(corpus_path / "full_texts" / "fake" / "bad_name.txt", "")

    dataframe = load_dataset(corpus_path)
    malformed = dataframe.loc[dataframe["article_id"] == "fake_bad_name"].iloc[0]

    assert len(dataframe) == 3
    assert pd.isna(malformed["pair_id"])
    assert malformed["title"] == ""
    assert malformed["content"] == ""
    assert malformed["raw_text"] == ""
