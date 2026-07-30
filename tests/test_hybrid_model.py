import pandas as pd
import pytest

from src.models.predict import build_linguistic_explanation
from src.models.train_hybrid_model import (
    DIRECT_LENGTH_FEATURES,
    exclude_train_duplicates_from_test,
    merge_text_with_features,
)


def _text_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "article_id": ["true_1", "fake_1"],
            "pair_id": [1, 1],
            "label": [0, 1],
            "label_name": ["real", "fake"],
            "model_text": ["Tekst real", "Tekst fake"],
        }
    )


def _feature_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "article_id": ["fake_1", "true_1"],
            "pair_id": [1, 1],
            "label": [1, 0],
            "label_name": ["fake", "real"],
            "word_count": [20, 100],
            "diacritic_ratio": [0.01, 0.05],
        }
    )


def test_merge_uses_article_id_instead_of_row_order() -> None:
    merged = merge_text_with_features(_text_data(), _feature_data())

    assert merged["article_id"].tolist() == ["true_1", "fake_1"]
    assert merged["word_count"].tolist() == [100, 20]


def test_merge_rejects_label_mismatch() -> None:
    features = _feature_data()
    features.loc[features["article_id"] == "true_1", "label"] = 1

    with pytest.raises(ValueError, match="Labels do not match"):
        merge_text_with_features(_text_data(), features)


def test_exact_train_duplicates_are_removed_only_from_test() -> None:
    train = _text_data().iloc[[0]].copy()
    test = _text_data().iloc[[1]].copy()
    test.loc[:, "model_text"] = "Tekst real"

    clean_test, excluded_ids = exclude_train_duplicates_from_test(train, test)

    assert clean_test.empty
    assert excluded_ids == ["fake_1"]
    assert len(train) == 1


def test_length_ablation_has_explicit_feature_list() -> None:
    assert set(DIRECT_LENGTH_FEATURES) == {
        "word_count",
        "sentence_count",
        "character_count",
        "avg_sentence_length",
        "title_length",
        "content_length",
    }


def test_linguistic_explanation_contains_readable_signals() -> None:
    explanation = build_linguistic_explanation(
        "LAJM I FUNDIT!!!",
        "Sipas policia, ky është një lajm tronditës.",
    )

    assert "lajm i fundit" in explanation["sensational_words_found"]
    assert "sipas" in explanation["source_markers_found"]
    assert explanation["exclamation_count"] == 3
    assert explanation["word_count"] > 0
    assert explanation["text_length"] > 0
    assert 0 <= explanation["diacritic_ratio"] <= 1
    assert 0 <= explanation["uppercase_ratio"] <= 1
