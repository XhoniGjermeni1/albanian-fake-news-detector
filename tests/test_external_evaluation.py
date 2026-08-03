import pandas as pd

from src.models.evaluate_external_dataset import (
    calculate_binary_metrics,
    calculate_decision_metrics,
    source_group,
    summarize_groups,
)


def sample_predictions() -> pd.DataFrame:
    """Create a small table with one TP, TN, FP, and FN."""
    return pd.DataFrame(
        {
            "true_label": ["real", "real", "fake", "fake"],
            "true_label_number": [0, 0, 1, 1],
            "binary_prediction_number": [0, 1, 0, 1],
            "prediction_correct": [True, False, False, True],
            "error_type": [
                "correct",
                "false_positive",
                "false_negative",
                "correct",
            ],
            "predicted_confidence": [0.80, 0.55, 0.80, 0.90],
            "decision": [
                "likely_real",
                "uncertain",
                "likely_real",
                "likely_fake",
            ],
            "probability_fake": [0.20, 0.55, 0.20, 0.90],
            "word_count": [40, 45, 50, 55],
            "topic": ["politikë", "politikë", "ekonomi", "ekonomi"],
        }
    )


def test_binary_metrics_use_fake_as_positive_class() -> None:
    metrics = calculate_binary_metrics(sample_predictions())

    assert metrics["accuracy"] == 0.5
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["class_fake"]["precision"] == 0.5
    assert metrics["class_fake"]["recall"] == 0.5
    assert metrics["class_fake"]["f1"] == 0.5


def test_three_level_decision_metrics() -> None:
    metrics = calculate_decision_metrics(sample_predictions())

    assert metrics["likely_real"] == 2
    assert metrics["uncertain"] == 1
    assert metrics["likely_fake"] == 1
    assert metrics["strong_decision_count"] == 3
    assert metrics["strong_decision_coverage"] == 0.75
    assert metrics["strong_decision_accuracy"] == 0.6667
    assert metrics["binary_errors_moved_to_uncertain"] == 1


def test_group_summary_counts_errors() -> None:
    summary = summarize_groups(sample_predictions(), "topic").set_index("topic")

    assert summary.loc["politikë", "false_positives"] == 1
    assert summary.loc["ekonomi", "false_negatives"] == 1
    assert summary.loc["ekonomi", "recall_fake"] == 0.5


def test_source_group_recognizes_external_source_types() -> None:
    assert source_group("Këshilli i Ministrave i Shqipërisë") == (
        "institutional_government"
    )
    assert source_group("Banka e Shqipërisë") == "institutional_financial"
    assert source_group("INSTAT") == "institutional_statistics"
    assert source_group("Krypometër / postim në Facebook") == (
        "fact_checked_social_claim"
    )
