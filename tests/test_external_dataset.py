from pathlib import Path

import pandas as pd
import pytest

from src.data.validate_external_dataset import (
    ALLOWED_LABELS,
    ALLOWED_TOPICS,
    REQUIRED_COLUMNS,
    build_similarity_review,
    load_corpus_urls,
    validate_external_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "external" / "external_news.csv"


def _valid_row(external_id: str = "EXT-TEST-001") -> dict[str, str]:
    return {
        "external_id": external_id,
        "title": "Titull i dokumentuar për një lajm të jashtëm shqiptar",
        "content": (
            "Ky është një tekst prove me më shumë se njëzet e pesë fjalë, "
            "i shkruar në shqip dhe me informacion të mjaftueshëm për të "
            "kontrolluar në mënyrë të sigurt validatorin e datasetit të jashtëm."
        ),
        "label": "real",
        "source": "Burim zyrtar",
        "url": "https://example.com/article",
        "published_date": "2026-01-15",
        "label_evidence": "Njoftimi është publikuar dhe dokumentuar nga burimi zyrtar përkatës.",
        "evidence_url": "https://example.com/article",
        "topic": "sociale",
        "content_origin": "manual_summary_sq",
        "review_status": "approved",
    }


def test_external_dataset_has_required_structure() -> None:
    dataframe = pd.read_csv(DATASET_PATH, encoding="utf-8")

    assert list(dataframe.columns) == REQUIRED_COLUMNS
    assert len(dataframe) == 40
    assert set(dataframe["label"]) == ALLOWED_LABELS
    assert set(dataframe["topic"]) == ALLOWED_TOPICS


def test_external_dataset_is_balanced_by_label_and_topic() -> None:
    dataframe = pd.read_csv(DATASET_PATH, encoding="utf-8")

    assert dataframe["label"].value_counts().to_dict() == {"real": 20, "fake": 20}
    assert dataframe["topic"].value_counts().eq(8).all()
    assert pd.crosstab(dataframe["topic"], dataframe["label"]).eq(4).all().all()
    assert dataframe["review_status"].eq("approved").all()


def test_validator_accepts_a_clean_row_without_training_data() -> None:
    dataframe = pd.DataFrame([_valid_row()])

    summary, similarity_review = validate_external_dataset(
        dataframe,
        training=None,
        print_report=False,
    )

    assert summary["ready_for_external_evaluation"] is True
    assert summary["blocking_problems"] == []
    assert similarity_review.empty


def test_validator_reports_malformed_and_short_rows() -> None:
    row = _valid_row()
    row["content"] = "Tekst tepër i shkurtër."
    row["label"] = "uncertain"
    row["label_evidence"] = "Pa provë"
    row["url"] = "not-a-url"
    dataframe = pd.DataFrame([row])

    summary, _ = validate_external_dataset(dataframe, print_report=False)

    assert summary["ready_for_external_evaluation"] is False
    assert summary["invalid_labels"] == ["uncertain"]
    assert summary["short_content_ids"] == ["EXT-TEST-001"]
    assert summary["weak_evidence_ids"] == ["EXT-TEST-001"]
    assert summary["invalid_url_ids"] == ["EXT-TEST-001"]


def test_similarity_review_finds_an_exact_training_copy() -> None:
    external = pd.DataFrame([_valid_row()])
    training = pd.DataFrame(
        [
            {
                "article_id": "true_1",
                "title": external.loc[0, "title"],
                "content": external.loc[0, "content"],
            },
            {
                "article_id": "fake_2",
                "title": "Një titull krejt tjetër",
                "content": "Një përmbajtje tjetër që nuk përputhet me tekstin e jashtëm.",
            },
        ]
    )

    review = build_similarity_review(external, training)

    assert bool(review.loc[0, "exact_training_title"]) is True
    assert bool(review.loc[0, "exact_training_content"]) is True
    assert bool(review.loc[0, "exact_training_text"]) is True
    assert review.loc[0, "max_training_similarity"] == pytest.approx(1.0)


def test_validator_finds_a_url_from_raw_metadata(tmp_path: Path) -> None:
    metadata_directory = tmp_path / "true-meta-information"
    metadata_directory.mkdir()
    (metadata_directory / "1.txt").write_text(
        "2026/01/15\nhttps://example.com/share\nhttps://example.com/article\n",
        encoding="utf-8",
    )
    training_urls, metadata_file_count = load_corpus_urls(tmp_path)

    summary, _ = validate_external_dataset(
        pd.DataFrame([_valid_row()]),
        training_urls=training_urls,
        metadata_files_checked=metadata_file_count,
        print_report=False,
    )

    assert summary["training_url_overlap"]["exact_url_matches"] == 1
    assert summary["training_url_overlap"]["exact_url_match_ids"] == ["EXT-TEST-001"]
    assert summary["ready_for_external_evaluation"] is False


def test_validator_rejects_missing_columns() -> None:
    dataframe = pd.DataFrame([_valid_row()]).drop(columns=["label_evidence"])

    with pytest.raises(ValueError, match="label_evidence"):
        validate_external_dataset(dataframe, print_report=False)
