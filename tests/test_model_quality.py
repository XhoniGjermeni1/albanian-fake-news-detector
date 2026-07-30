from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.models.analyze_model_quality import (
    build_leakage_safe_groups,
    build_prediction_table,
    evaluate_thresholds,
)
from src.models.predict import classify_probability, predict_news_for_app


class FixedProbabilityModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, texts):
        return np.tile(np.array([[0.45, 0.55]]), (len(texts), 1))


def test_probability_decision_boundaries() -> None:
    assert classify_probability(0.29) == "likely_real"
    assert classify_probability(0.30) == "uncertain"
    assert classify_probability(0.50) == "uncertain"
    assert classify_probability(0.70) == "uncertain"
    assert classify_probability(0.71) == "likely_fake"


def test_probability_decision_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        classify_probability(1.2)
    with pytest.raises(ValueError):
        classify_probability(0.5, real_threshold=0.8, fake_threshold=0.2)


def test_leakage_groups_connect_pairs_and_duplicate_texts() -> None:
    dataframe = pd.DataFrame(
        {
            "pair_id": [1, 1, 2, 3],
            "model_text": ["Teksti A", "Teksti B", "Teksti B", "Teksti C"],
        }
    )

    groups = build_leakage_safe_groups(dataframe)

    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]


def test_threshold_evaluation_counts_uncertain_rows() -> None:
    y_true = pd.Series([0, 0, 1, 1])
    probabilities = np.array([0.10, 0.50, 0.50, 0.90])

    result = evaluate_thresholds(y_true, probabilities)
    variant = result.loc[result["variant"] == "35-65"].iloc[0]

    assert variant["likely_real_count"] == 1
    assert variant["uncertain_count"] == 2
    assert variant["likely_fake_count"] == 1
    assert variant["errors_moved_to_uncertain"] == 1
    assert variant["strong_decision_accuracy"] == 1.0


def test_prediction_table_marks_false_positive_and_false_negative() -> None:
    test_data = pd.DataFrame(
        {
            "article_id": ["true_1", "fake_1"],
            "pair_id": [1, 1],
            "label": [0, 1],
            "label_name": ["real", "fake"],
            "title": ["Titull real", "Titull fake"],
            "content": ["Përmbajtje reale", "Përmbajtje fake"],
            "word_count": [10, 20],
            "source_indicators_found": ["", "sipas"],
            "sensational_found": ["skandal", ""],
            "diacritic_ratio": [0.02, 0.03],
            "uppercase_char_ratio": [0.01, 0.02],
            "exclamation_count": [1, 0],
        }
    )

    table = build_prediction_table(test_data, np.array([0.80, 0.20]))

    assert table["error_type"].tolist() == ["false_positive", "false_negative"]
    assert table["interpretation"].str.contains("TF-IDF").all()


def test_app_prediction_returns_uncertain_and_explanation(tmp_path: Path) -> None:
    model_path = tmp_path / "fixed_model.joblib"
    joblib.dump(FixedProbabilityModel(), model_path)

    result = predict_news_for_app(
        "Lajm i fundit!",
        "Sipas raportit, kjo është përmbajtja.",
        model_path=model_path,
    )

    assert result["decision"] == "uncertain"
    assert result["probability_fake"] == 0.55
    assert "sensational_words_found" in result["linguistic_explanation"]
    assert "nuk është verifikim faktik" in result["notice"]
