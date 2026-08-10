import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from src.models.calibrate_linear_svm import (
    BASELINE_C,
    build_calibrated_svm,
    classify_probability,
    evaluate_threshold_variants,
    expected_calibration_error,
    high_confidence_error_rows,
    select_calibration_method,
    select_thresholds,
    threshold_metrics,
    verify_frozen_day15,
)


def test_expected_calibration_error_is_zero_for_perfect_probabilities() -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.0, 0.0, 1.0, 1.0])

    assert expected_calibration_error(labels, probabilities) == 0.0


def test_uncertain_threshold_boundaries_are_inclusive() -> None:
    assert classify_probability(0.299, 0.30, 0.70) == "likely_real"
    assert classify_probability(0.30, 0.30, 0.70) == "uncertain"
    assert classify_probability(0.70, 0.30, 0.70) == "uncertain"
    assert classify_probability(0.701, 0.30, 0.70) == "likely_fake"


def test_threshold_metrics_count_strong_and_uncertain_errors() -> None:
    labels = np.array([0, 1, 0, 1, 0])
    probabilities = np.array([0.1, 0.9, 0.5, 0.35, 0.8])

    metrics = threshold_metrics(labels, probabilities, 0.30, 0.70)

    assert metrics["likely_real"] == 1
    assert metrics["uncertain"] == 2
    assert metrics["likely_fake"] == 2
    assert metrics["strong_coverage"] == 0.6
    assert metrics["strong_accuracy"] == 2 / 3
    assert metrics["errors_in_uncertain"] == 2
    assert metrics["strong_false_positives"] == 1
    assert metrics["strong_false_negatives"] == 0


def test_calibration_selection_prefers_sigmoid_when_metrics_are_close() -> None:
    comparison = pd.DataFrame(
        [
            {
                "method": "sigmoid",
                "brier_score": 0.080,
                "log_loss": 0.270,
                "ece": 0.025,
                "std_brier_score": 0.005,
            },
            {
                "method": "isotonic",
                "brier_score": 0.079,
                "log_loss": 0.265,
                "ece": 0.020,
                "std_brier_score": 0.009,
            },
        ]
    )

    selection = select_calibration_method(comparison)

    assert selection["selected_method"] == "sigmoid"
    assert "lower_overfitting_risk" in selection["method_selection_reason"]


def test_calibration_selection_uses_isotonic_when_materially_better() -> None:
    comparison = pd.DataFrame(
        [
            {
                "method": "sigmoid",
                "brier_score": 0.090,
                "log_loss": 0.310,
                "ece": 0.030,
                "std_brier_score": 0.005,
            },
            {
                "method": "isotonic",
                "brier_score": 0.080,
                "log_loss": 0.280,
                "ece": 0.025,
                "std_brier_score": 0.008,
            },
        ]
    )

    assert select_calibration_method(comparison)["selected_method"] == "isotonic"


def test_threshold_selection_prefers_coverage_when_accuracy_is_close() -> None:
    comparison = pd.DataFrame(
        [
            {
                "threshold_name": "30_70",
                "lower": 0.30,
                "upper": 0.70,
                "strong_accuracy": 0.950,
                "strong_coverage": 0.75,
                "errors_captured_fraction": 0.70,
                "errors_in_uncertain": 14,
                "strong_false_negatives": 2,
                "strong_false_positives": 3,
            },
            {
                "threshold_name": "35_65",
                "lower": 0.35,
                "upper": 0.65,
                "strong_accuracy": 0.952,
                "strong_coverage": 0.65,
                "errors_captured_fraction": 0.75,
                "errors_in_uncertain": 15,
                "strong_false_negatives": 2,
                "strong_false_positives": 2,
            },
            {
                "threshold_name": "40_60",
                "lower": 0.40,
                "upper": 0.60,
                "strong_accuracy": 0.954,
                "strong_coverage": 0.55,
                "errors_captured_fraction": 0.80,
                "errors_in_uncertain": 16,
                "strong_false_negatives": 1,
                "strong_false_positives": 1,
            },
        ]
    )

    selection = select_thresholds(comparison)

    assert selection["threshold_name"] == "30_70"
    assert selection["lower_threshold"] == 0.30
    assert selection["upper_threshold"] == 0.70


def test_requested_threshold_variants_are_fixed() -> None:
    comparison = evaluate_threshold_variants(
        np.array([0, 1]), np.array([0.1, 0.9])
    )

    assert comparison["threshold_name"].tolist() == ["30_70", "35_65", "40_60"]


def test_calibrated_model_keeps_frozen_linear_svm_configuration() -> None:
    folds = [
        (np.array([2, 3]), np.array([0, 1])),
        (np.array([0, 1]), np.array([2, 3])),
    ]

    model = build_calibrated_svm("sigmoid", folds)

    assert isinstance(model, CalibratedClassifierCV)
    assert model.method == "sigmoid"
    assert model.estimator.named_steps["classifier"].C == BASELINE_C


def test_high_confidence_errors_use_one_common_case_id_column() -> None:
    table = pd.DataFrame(
        {
            "external_id": ["EXT-001", "EXT-002"],
            "label": [0, 1],
            "title": ["Titull real", "Titull fake"],
            "probability_fake": [0.95, 0.85],
            "binary_prediction": [1, 1],
            "confidence": [0.95, 0.85],
        }
    )

    errors = high_confidence_error_rows(
        table, "external", "new_calibrated_svm", "external_id"
    )

    assert errors["case_id"].tolist() == ["EXT-001"]
    assert errors["dataset"].tolist() == ["external"]
    assert errors["binary_prediction"].tolist() == [1]


def test_day15_candidate_is_the_frozen_input() -> None:
    setup = verify_frozen_day15()

    assert setup["representation"] == "word_char_tfidf"
    assert setup["classifier"] == "linear_svm"
    assert setup["c_value"] == 1.0
