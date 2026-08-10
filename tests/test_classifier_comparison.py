import unicodedata

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

from src.models.compare_classifiers import (
    FIXED_CHAR_CONFIG,
    add_word_counts,
    build_classifier,
    build_fixed_features,
    build_group_safe_folds,
    classification_metrics,
    fake_decision_scores,
    refresh_model_text,
    select_from_cv,
)


def test_fixed_representation_matches_day13_configuration() -> None:
    features = build_fixed_features(FIXED_CHAR_CONFIG)
    transformers = dict(features.transformer_list)
    word = transformers["word"]
    character = transformers["character"]

    assert word.ngram_range == (1, 2)
    assert word.lowercase is False
    assert word.min_df == 2
    assert word.max_features == 30000
    assert character.analyzer == "char_wb"
    assert character.ngram_range == (3, 5)
    assert character.lowercase is False
    assert character.min_df == 2
    assert character.max_features == 50000


def test_classifier_factory_returns_expected_sparse_classifiers() -> None:
    logistic = build_classifier(
        {
            "classifier": "logistic_regression",
            "parameter_value": 1.0,
        }
    )
    svm = build_classifier(
        {
            "classifier": "linear_svm",
            "parameter_value": 1.0,
        }
    )
    naive_bayes = build_classifier(
        {
            "classifier": "complement_nb",
            "parameter_value": 1.0,
        }
    )

    assert isinstance(logistic, LogisticRegression)
    assert isinstance(svm, LinearSVC)
    assert isinstance(naive_bayes, ComplementNB)


def test_fake_scores_are_oriented_scores_not_svm_probabilities() -> None:
    features = csr_matrix(
        [[1.0, 0.0], [0.8, 0.1], [0.0, 1.0], [0.1, 0.8]]
    )
    labels = np.array([0, 0, 1, 1])
    svm = LinearSVC(random_state=42).fit(features, labels)
    naive_bayes = ComplementNB().fit(features, labels)

    svm_scores = fake_decision_scores(svm, features)
    nb_scores = fake_decision_scores(naive_bayes, features)

    assert not hasattr(svm, "predict_proba")
    assert svm_scores.shape == (4,)
    assert nb_scores.shape == (4,)
    assert svm_scores[:2].mean() < svm_scores[2:].mean()
    assert nb_scores[:2].mean() < nb_scores[2:].mean()


def test_group_safe_folds_keep_pairs_and_duplicate_texts_together() -> None:
    rows = []
    for pair_id in range(12):
        for label in (0, 1):
            rows.append(
                {
                    "pair_id": str(pair_id),
                    "label": label,
                    "model_text": f"tekst i grupit {pair_id}",
                }
            )
    dataframe = pd.DataFrame(rows)
    dataframe.loc[4, "model_text"] = dataframe.loc[0, "model_text"]

    folds, groups, audit = build_group_safe_folds(dataframe)

    assert len(folds) == 5
    assert all(item["overlapping_groups"] == 0 for item in audit)
    for fit_index, validation_index in folds:
        assert not (set(groups[fit_index]) & set(groups[validation_index]))


def test_selection_uses_stability_when_cv_f1_is_within_tolerance() -> None:
    summary = pd.DataFrame(
        [
            {
                "candidate_id": "lr",
                "classifier": "logistic_regression",
                "parameter_name": "C",
                "parameter_value": 1.0,
                "mean_f1_weighted": 0.9000,
                "std_f1_weighted": 0.0100,
                "mean_f1_fake": 0.8990,
                "mean_accuracy": 0.9000,
                "mean_recall_real": 0.9000,
                "mean_recall_fake": 0.9000,
                "mean_training_seconds": 2.0,
            },
            {
                "candidate_id": "svm",
                "classifier": "linear_svm",
                "parameter_name": "C",
                "parameter_value": 1.0,
                "mean_f1_weighted": 0.9015,
                "std_f1_weighted": 0.0200,
                "mean_f1_fake": 0.9010,
                "mean_accuracy": 0.9015,
                "mean_recall_real": 0.9020,
                "mean_recall_fake": 0.9010,
                "mean_training_seconds": 1.0,
            },
            {
                "candidate_id": "nb",
                "classifier": "complement_nb",
                "parameter_name": "alpha",
                "parameter_value": 1.0,
                "mean_f1_weighted": 0.8800,
                "std_f1_weighted": 0.0050,
                "mean_f1_fake": 0.8780,
                "mean_accuracy": 0.8800,
                "mean_recall_real": 0.8900,
                "mean_recall_fake": 0.8700,
                "mean_training_seconds": 0.5,
            },
        ]
    )

    selection = select_from_cv(summary, FIXED_CHAR_CONFIG)

    assert selection["winner_classifier"] == "logistic_regression"
    assert selection["internal_test_used"] is False
    assert selection["external_results_used"] is False
    assert selection["calibration_applied"] is False


def test_metrics_keep_fake_as_positive_class() -> None:
    metrics = classification_metrics([0, 0, 1, 1], [0, 1, 0, 1])

    assert metrics["accuracy"] == 0.5
    assert metrics["f1_fake"] == 0.5
    assert metrics["recall_real"] == 0.5
    assert metrics["recall_fake"] == 0.5
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]


def test_refresh_model_text_applies_current_unicode_normalization() -> None:
    title = unicodedata.normalize("NFD", "Çështja")
    content = unicodedata.normalize("NFD", "Është tekst shqip.")
    dataframe = pd.DataFrame(
        {"title": [title], "content": [content], "model_text": [f"{title}. {content}"]}
    )

    refreshed, stale_rows = refresh_model_text(dataframe)

    assert stale_rows == 1
    assert refreshed.loc[0, "model_text"] == "Çështja. Është tekst shqip."


def test_word_counts_reuse_linguistic_feature_tokenization() -> None:
    dataframe = pd.DataFrame(
        {
            "title": ["Lajmi 2026"],
            "content": ["Është kontrolluar."],
            "model_text": ["Lajmi 2026. Është kontrolluar."],
        }
    )

    counted = add_word_counts(dataframe)

    assert counted.loc[0, "word_count"] == 3
