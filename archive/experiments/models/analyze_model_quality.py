# Kalibron baseline-in historik TF-IDF + Logistic Regression me folds pa leakage dhe
# mat accuracy, F1, Brier score, log-loss, ECE dhe cilësinë e confidence-it. Krahason
# gjithashtu variante pragjesh për zonën e pasigurt që parapriu kontratën e aplikacionit.

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[3]))

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
)

from src.evaluation.data_utils import (
    build_group_safe_folds,
    exclude_train_duplicates_from_test,
)
from src.evaluation.metrics import classification_metrics
from archive.experiments.models.train_hybrid_model import build_tfidf_model

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAIN_PATH = PROJECT_ROOT / "data" / "interim" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "interim" / "test.csv"
BASELINE_MODEL_PATH = PROJECT_ROOT / "archive" / "models" / "baseline_tfidf_logreg.joblib"
CALIBRATED_MODEL_PATH = PROJECT_ROOT / "archive" / "models" / "calibrated_tfidf_logreg.joblib"

REPORTS_DIR = PROJECT_ROOT / "archive" / "reports"
METRICS_PATH = REPORTS_DIR / "day6_metrics.json"
THRESHOLD_PATH = REPORTS_DIR / "day6_threshold_comparison.csv"

THRESHOLD_VARIANTS = [
    ("35-65", 0.35, 0.65),
    ("40-60", 0.40, 0.60),
    ("30-70", 0.30, 0.70),
]

LOGGER = logging.getLogger(__name__)


def load_day6_inputs() -> tuple[pd.DataFrame, pd.DataFrame, object, dict]:
    required_paths = [TRAIN_PATH, TEST_PATH, BASELINE_MODEL_PATH]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing Day 5 artifacts: {missing_paths}")

    train_data = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig", keep_default_na=False)
    test_data = pd.read_csv(TEST_PATH, encoding="utf-8-sig", keep_default_na=False)
    baseline_model = joblib.load(BASELINE_MODEL_PATH)

    evaluation_test, excluded_ids = exclude_train_duplicates_from_test(train_data, test_data)
    if evaluation_test["model_text"].eq("").any():
        raise ValueError("The evaluation test set contains empty model_text values.")

    checks = {
        "train_rows": int(len(train_data)),
        "original_test_rows": int(len(test_data)),
        "evaluation_test_rows": int(len(evaluation_test)),
        "exact_train_duplicates_excluded": int(len(excluded_ids)),
        "excluded_article_ids": excluded_ids,
        "empty_model_text_rows": 0,
    }
    return train_data, evaluation_test, baseline_model, checks


def build_calibration_folds(dataframe: pd.DataFrame) -> tuple[list[tuple[np.ndarray, np.ndarray]], int]:
    folds, groups, _ = build_group_safe_folds(dataframe)
    return folds, int(len(np.unique(groups)))


def train_calibrated_model(train_data: pd.DataFrame) -> tuple[CalibratedClassifierCV, dict]:
    folds, group_count = build_calibration_folds(train_data)
    model = CalibratedClassifierCV(
        estimator=build_tfidf_model(),
        method="sigmoid",
        cv=folds,
        ensemble=False,
    )
    model.fit(train_data["model_text"], train_data["label"])

    fold_sizes = [
        {
            "fit_rows": int(len(train_index)),
            "calibration_rows": int(len(calibration_index)),
        }
        for train_index, calibration_index in folds
    ]
    return model, {
        "method": "sigmoid",
        "fold_count": len(folds),
        "leakage_group_count": group_count,
        "ensemble": False,
        "fold_sizes": fold_sizes,
    }


def expected_calibration_error(
    y_true: np.ndarray,
    probability_fake: np.ndarray,
    n_bins: int = 10,
) -> float:
    edges = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(probability_fake, edges[1:-1], right=False)
    error = 0.0

    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not mask.any():
            continue
        observed_rate = float(y_true[mask].mean())
        mean_probability = float(probability_fake[mask].mean())
        error += float(mask.mean()) * abs(observed_rate - mean_probability)

    return error


def calculate_probability_metrics(
    y_true: pd.Series,
    probability_fake: np.ndarray,
) -> dict:
    y_array = y_true.to_numpy(dtype=int)
    predictions = (probability_fake >= 0.5).astype(int)
    wrong = predictions != y_array
    confidence = np.maximum(probability_fake, 1 - probability_fake)
    high_confidence = confidence >= 0.90
    classification = classification_metrics(y_array, predictions)

    return {
        "accuracy": round(classification["accuracy"], 4),
        "precision": round(classification["precision_weighted"], 4),
        "recall": round(classification["recall_weighted"], 4),
        "f1": round(classification["f1_weighted"], 4),
        "precision_fake": round(classification["precision_fake"], 4),
        "recall_fake": round(classification["recall_fake"], 4),
        "f1_fake": round(classification["f1_fake"], 4),
        "brier_score": round(float(brier_score_loss(y_array, probability_fake)), 4),
        "log_loss": round(float(log_loss(y_array, probability_fake)), 4),
        "expected_calibration_error": round(
            float(expected_calibration_error(y_array, probability_fake)),
            4,
        ),
        "confusion_matrix": classification["confusion_matrix"],
        "false_positives": classification["false_positives"],
        "false_negatives": classification["false_negatives"],
        "high_confidence_predictions_90": int(high_confidence.sum()),
        "high_confidence_errors_90": int((wrong & high_confidence).sum()),
        "high_confidence_error_rate_90": round(
            float((wrong & high_confidence).sum() / high_confidence.sum())
            if high_confidence.any()
            else 0.0,
            4,
        ),
        "extreme_probability_predictions_10_90": int(
            ((probability_fake <= 0.10) | (probability_fake >= 0.90)).sum()
        ),
        "near_50_predictions_45_55": int(
            ((probability_fake >= 0.45) & (probability_fake <= 0.55)).sum()
        ),
        "near_50_errors_45_55": int(
            (wrong & (probability_fake >= 0.45) & (probability_fake <= 0.55)).sum()
        ),
        "uncertain_predictions_35_65": int(
            ((probability_fake >= 0.35) & (probability_fake <= 0.65)).sum()
        ),
        "uncertain_errors_35_65": int(
            (wrong & (probability_fake >= 0.35) & (probability_fake <= 0.65)).sum()
        ),
        "mean_probability_fake": round(float(probability_fake.mean()), 4),
        "median_probability_fake": round(float(np.median(probability_fake)), 4),
        "minimum_probability_fake": round(float(probability_fake.min()), 4),
        "maximum_probability_fake": round(float(probability_fake.max()), 4),
    }


def evaluate_thresholds(
    y_true: pd.Series,
    probability_fake: np.ndarray,
) -> pd.DataFrame:
    y_array = y_true.to_numpy(dtype=int)
    ordinary_predictions = (probability_fake >= 0.5).astype(int)
    ordinary_errors = ordinary_predictions != y_array
    rows = []

    for variant, real_threshold, fake_threshold in THRESHOLD_VARIANTS:
        likely_real = probability_fake < real_threshold
        likely_fake = probability_fake > fake_threshold
        uncertain = ~(likely_real | likely_fake)
        strong_errors = (likely_real & (y_array == 1)) | (likely_fake & (y_array == 0))
        strong_correct = (likely_real & (y_array == 0)) | (likely_fake & (y_array == 1))
        strong_count = int((~uncertain).sum())

        rows.append(
            {
                "variant": variant,
                "real_threshold": real_threshold,
                "fake_threshold": fake_threshold,
                "likely_real_count": int(likely_real.sum()),
                "uncertain_count": int(uncertain.sum()),
                "likely_fake_count": int(likely_fake.sum()),
                "uncertain_real_count": int((uncertain & (y_array == 0)).sum()),
                "uncertain_fake_count": int((uncertain & (y_array == 1)).sum()),
                "errors_moved_to_uncertain": int((ordinary_errors & uncertain).sum()),
                "strong_decision_errors": int(strong_errors.sum()),
                "strong_false_negatives": int((likely_real & (y_array == 1)).sum()),
                "strong_false_positives": int((likely_fake & (y_array == 0)).sum()),
                "strong_decision_coverage": round(strong_count / len(y_array), 4),
                "strong_decision_accuracy": round(
                    float(strong_correct.sum() / strong_count) if strong_count else 0.0,
                    4,
                ),
            }
        )

    return pd.DataFrame(rows)


def _fake_probabilities(model, model_text: pd.Series) -> np.ndarray:
    classes = list(model.classes_)
    fake_index = classes.index(1)
    return model.predict_proba(model_text)[:, fake_index]


def _relative_path(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def run_model_quality_analysis() -> dict:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CALIBRATED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    train_data, test_data, baseline_model, checks = load_day6_inputs()
    LOGGER.info("Calculating baseline predictions")
    baseline_probability = _fake_probabilities(baseline_model, test_data["model_text"])

    LOGGER.info("Training group-safe sigmoid calibration")
    calibrated_model, calibration_info = train_calibrated_model(train_data)
    joblib.dump(calibrated_model, CALIBRATED_MODEL_PATH)
    calibrated_probability = _fake_probabilities(calibrated_model, test_data["model_text"])

    probability_sets = {
        "uncalibrated": baseline_probability,
        "calibrated_sigmoid": calibrated_probability,
    }
    probability_metrics = {
        model_name: calculate_probability_metrics(test_data["label"], probabilities)
        for model_name, probabilities in probability_sets.items()
    }

    thresholds = evaluate_thresholds(test_data["label"], calibrated_probability)
    thresholds.to_csv(THRESHOLD_PATH, index=False, encoding="utf-8-sig")

    eligible_thresholds = thresholds.loc[thresholds["strong_decision_coverage"].ge(0.80)]
    recommended_threshold = eligible_thresholds.sort_values(
        ["strong_decision_accuracy", "strong_decision_coverage"],
        ascending=False,
    ).iloc[0]

    before = probability_metrics["uncalibrated"]
    after = probability_metrics["calibrated_sigmoid"]
    result = {
        "data_checks": checks,
        "calibration": calibration_info,
        "probability_metrics": probability_metrics,
        "calibration_changes": {
            "accuracy": round(after["accuracy"] - before["accuracy"], 4),
            "f1_fake": round(after["f1_fake"] - before["f1_fake"], 4),
            "brier_score": round(after["brier_score"] - before["brier_score"], 4),
            "log_loss": round(after["log_loss"] - before["log_loss"], 4),
            "expected_calibration_error": round(
                after["expected_calibration_error"] - before["expected_calibration_error"],
                4,
            ),
            "high_confidence_errors_90": (
                after["high_confidence_errors_90"] - before["high_confidence_errors_90"]
            ),
        },
        "threshold_variants": json.loads(thresholds.to_json(orient="records")),
        "recommendation": {
            "prediction_model": "calibrated_tfidf_logreg",
            "calibration_method": "sigmoid",
            "threshold_variant": str(recommended_threshold["variant"]),
            "likely_real_below": float(recommended_threshold["real_threshold"]),
            "likely_fake_above": float(recommended_threshold["fake_threshold"]),
            "uncertain_inclusive": [
                float(recommended_threshold["real_threshold"]),
                float(recommended_threshold["fake_threshold"]),
            ],
            "linguistic_features_role": "explanation_only",
            "warning": "The model analyzes text patterns and does not verify facts.",
        },
        "artifacts": {
            "calibrated_model": _relative_path(CALIBRATED_MODEL_PATH),
            "threshold_comparison": _relative_path(THRESHOLD_PATH),
        },
    }
    METRICS_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    result = run_model_quality_analysis()
    before = result["probability_metrics"]["uncalibrated"]
    after = result["probability_metrics"]["calibrated_sigmoid"]

    print(f"Evaluation rows: {result['data_checks']['evaluation_test_rows']}")
    print(f"False positives: {before['false_positives']}")
    print(f"False negatives: {before['false_negatives']}")
    print(f"Brier before/after: {before['brier_score']} / {after['brier_score']}")
    print(f"Accuracy before/after: {before['accuracy']} / {after['accuracy']}")
    print(f"Recommended thresholds: {result['recommendation']['threshold_variant']}")
    print(f"Metrics saved: {METRICS_PATH}")


if __name__ == "__main__":
    main()
