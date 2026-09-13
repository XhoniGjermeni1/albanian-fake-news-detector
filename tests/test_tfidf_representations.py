import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from archive.experiments.models.compare_tfidf_representations import (
    CHARACTER_CONFIGS,
    build_representation_pipeline,
    calculate_metrics,
)


def test_representation_pipelines_have_expected_feature_types() -> None:
    config = CHARACTER_CONFIGS[0]

    word = build_representation_pipeline("word_tfidf", config)
    character = build_representation_pipeline("char_tfidf", config)
    combined = build_representation_pipeline("word_char_tfidf", config)

    assert isinstance(word.named_steps["features"], TfidfVectorizer)
    assert word.named_steps["features"].analyzer == "word"
    assert isinstance(character.named_steps["features"], TfidfVectorizer)
    assert character.named_steps["features"].analyzer == "char_wb"
    assert isinstance(combined.named_steps["features"], FeatureUnion)


def test_metrics_use_fake_as_positive_class() -> None:
    table = pd.DataFrame(
        {
            "label": [0, 0, 1, 1],
            "binary_prediction": [0, 1, 0, 1],
            "probability_real": [0.8, 0.4, 0.7, 0.1],
            "probability_fake": [0.2, 0.6, 0.3, 0.9],
            "decision": ["likely_real", "uncertain", "uncertain", "likely_fake"],
        }
    )

    metrics = calculate_metrics(table)

    assert metrics["accuracy"] == 0.5
    assert metrics["f1_fake"] == 0.5
    assert metrics["recall_real"] == 0.5
    assert metrics["recall_fake"] == 0.5
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1

