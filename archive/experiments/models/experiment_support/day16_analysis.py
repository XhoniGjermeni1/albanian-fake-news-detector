"""Configuration and calculations for the Day 16 calibration experiment."""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss as sklearn_log_loss

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.data_utils import (  # noqa: E402
    LENGTH_DISPLAY,
    LENGTH_LABELS,
    add_word_counts,
    build_group_safe_folds,
    exclude_train_duplicates_from_test,
    refresh_model_text,
)
from src.evaluation.experiment_utils import (  # noqa: E402
    escaped_dataframe_to_markdown as dataframe_to_markdown,
    file_sha256,
)
from src.evaluation.metrics import (  # noqa: E402
    classification_metrics,
    rounded_metrics,
)
from src.models.builders import (  # noqa: E402
    FINAL_SVM_C as BASELINE_C,
    build_svm_pipeline,
)


TRAIN_PATH = PROJECT_ROOT / "data" / "interim" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "interim" / "test.csv"
EXTERNAL_PATH = PROJECT_ROOT / "data" / "external" / "external_news.csv"
DAY15_SELECTION_PATH = PROJECT_ROOT / "reports" / "day15_selection.json"
DAY15_LENGTH_BIAS_PATH = PROJECT_ROOT / "reports" / "day15_length_bias_comparison.csv"
CURRENT_APP_MODEL_PATH = PROJECT_ROOT / "models" / "calibrated_tfidf_logreg.joblib"
STREAMLIT_APP_PATH = PROJECT_ROOT / "app" / "streamlit_app.py"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

OOF_PREDICTIONS_PATH = REPORTS_DIR / "day16_oof_calibration_predictions.csv"
CALIBRATION_FOLDS_PATH = REPORTS_DIR / "day16_calibration_fold_metrics.csv"
METHOD_COMPARISON_PATH = REPORTS_DIR / "day16_calibration_method_comparison.csv"
OOF_BINS_PATH = REPORTS_DIR / "day16_oof_calibration_bins.csv"
PROBABILITY_DISTRIBUTION_PATH = REPORTS_DIR / "day16_probability_distribution.csv"
THRESHOLD_COMPARISON_PATH = REPORTS_DIR / "day16_threshold_comparison.csv"
SELECTION_PATH = REPORTS_DIR / "day16_selection.json"
INTERNAL_PREDICTIONS_PATH = REPORTS_DIR / "day16_internal_predictions.csv"
INTERNAL_MODEL_COMPARISON_PATH = REPORTS_DIR / "day16_internal_model_comparison.csv"
INTERNAL_BINS_PATH = REPORTS_DIR / "day16_internal_calibration_bins.csv"
INTERNAL_THRESHOLD_PATH = REPORTS_DIR / "day16_internal_threshold_metrics.csv"
LENGTH_METRICS_PATH = REPORTS_DIR / "day16_length_group_metrics.csv"
SPECIAL_COHORTS_PATH = REPORTS_DIR / "day16_special_cohort_metrics.csv"
LENGTH_BIAS_PATH = REPORTS_DIR / "day16_length_bias.csv"
EXTERNAL_PREDICTIONS_PATH = REPORTS_DIR / "day16_external_predictions.csv"
EXTERNAL_MODEL_COMPARISON_PATH = REPORTS_DIR / "day16_external_model_comparison.csv"
EXTERNAL_THRESHOLD_PATH = REPORTS_DIR / "day16_external_threshold_metrics.csv"
HIGH_CONFIDENCE_ERRORS_PATH = REPORTS_DIR / "day16_high_confidence_errors.csv"
METRICS_PATH = REPORTS_DIR / "day16_metrics.json"
REPORT_PATH = REPORTS_DIR / "day16_calibration_thresholds.md"

OOF_FIGURE_PATH = FIGURES_DIR / "day16_oof_calibration_comparison.png"
THRESHOLD_FIGURE_PATH = FIGURES_DIR / "day16_threshold_comparison.png"
INTERNAL_FIGURE_PATH = FIGURES_DIR / "day16_internal_calibration.png"
LENGTH_FIGURE_PATH = FIGURES_DIR / "day16_length_probability.png"
MODEL_COMPARISON_FIGURE_PATH = FIGURES_DIR / "day16_model_comparison.png"

CALIBRATED_MODEL_PATH = MODELS_DIR / "day16_word_char_linear_svm_calibrated.joblib"

CALIBRATION_METHODS = ["sigmoid", "isotonic"]
THRESHOLD_VARIANTS = [
    {"threshold_name": "30_70", "lower": 0.30, "upper": 0.70},
    {"threshold_name": "35_65", "lower": 0.35, "upper": 0.65},
    {"threshold_name": "40_60", "lower": 0.40, "upper": 0.60},
]
N_CALIBRATION_BINS = 10
METHOD_BRIER_TOLERANCE = 0.002
METHOD_LOG_LOSS_TOLERANCE = 0.01
THRESHOLD_ACCURACY_TOLERANCE = 0.005
HIGH_CONFIDENCE = 0.90
LOGGER = logging.getLogger(__name__)


def verify_frozen_day15() -> dict:
    """Verify that Day 16 starts from the frozen Day 15 candidate."""
    selection = json.loads(DAY15_SELECTION_PATH.read_text(encoding="utf-8"))
    if selection.get("fixed_representation", {}).get("name") != "word_char_tfidf":
        raise ValueError("Day 15 representation is not Word + Character TF-IDF.")
    if selection.get("classifier") != "linear_svm":
        raise ValueError("Day 15 classifier is not Linear SVM.")
    if float(selection.get("selected_c", -1)) != BASELINE_C:
        raise ValueError("Day 15 C is not 1.0.")
    if selection.get("internal_test_used") is not False:
        raise ValueError("Day 15 selection is not train/CV-only.")
    if selection.get("external_results_used") is not False:
        raise ValueError("Day 15 selection used external data.")
    if selection.get("calibration_applied") is not False:
        raise ValueError("Day 15 candidate is already marked as calibrated.")
    return {
        "representation": "word_char_tfidf",
        "classifier": "linear_svm",
        "c_value": BASELINE_C,
        "group_count": int(selection["group_count"]),
    }


def fake_probabilities(model, texts) -> tuple[np.ndarray, np.ndarray]:
    """Return probabilities by class label rather than fixed column positions."""
    probabilities = np.asarray(model.predict_proba(texts), dtype=float)
    classes = list(model.classes_)
    if 0 not in classes or 1 not in classes:
        raise ValueError(f"Expected classes 0 and 1, found {classes}")
    real = probabilities[:, classes.index(0)]
    fake = probabilities[:, classes.index(1)]
    if not np.allclose(real + fake, 1.0, atol=1e-8):
        raise ValueError("Calibrated probabilities do not sum to one.")
    if np.any((fake < 0.0) | (fake > 1.0)):
        raise ValueError("Fake probabilities are outside [0, 1].")
    return real, fake


def calibration_bins(
    y_true,
    probability_fake,
    n_bins: int = N_CALIBRATION_BINS,
) -> pd.DataFrame:
    """Return equal-width bins used by ECE and reliability diagrams."""
    labels = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probability_fake, dtype=float)
    bin_ids = np.minimum((probabilities * n_bins).astype(int), n_bins - 1)
    rows: list[dict] = []
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        count = int(mask.sum())
        mean_probability = float(probabilities[mask].mean()) if count else None
        fraction_fake = float(labels[mask].mean()) if count else None
        gap = (
            abs(mean_probability - fraction_fake)
            if mean_probability is not None and fraction_fake is not None
            else None
        )
        rows.append(
            {
                "bin_id": bin_id,
                "bin_lower": bin_id / n_bins,
                "bin_upper": (bin_id + 1) / n_bins,
                "rows": count,
                "mean_probability_fake": mean_probability,
                "fraction_fake": fraction_fake,
                "absolute_gap": gap,
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(
    y_true,
    probability_fake,
    n_bins: int = N_CALIBRATION_BINS,
) -> float:
    """Calculate weighted equal-width Expected Calibration Error."""
    bins = calibration_bins(y_true, probability_fake, n_bins=n_bins)
    total = int(bins["rows"].sum())
    if total == 0:
        return 0.0
    non_empty = bins.loc[bins["rows"].gt(0)]
    return float(
        (
            non_empty["rows"]
            / total
            * non_empty["absolute_gap"].astype(float)
        ).sum()
    )


def probability_metrics(y_true, probability_fake) -> dict:
    """Calculate discrimination, calibration, and confidence metrics."""
    labels = np.asarray(y_true, dtype=int)
    fake = np.asarray(probability_fake, dtype=float)
    predictions = (fake >= 0.5).astype(int)
    metrics = classification_metrics(labels, predictions)
    confidence = np.maximum(fake, 1.0 - fake)
    high = confidence >= HIGH_CONFIDENCE
    wrong = predictions != labels
    metrics.update(
        {
            "brier_score": float(brier_score_loss(labels, fake)),
            "log_loss": float(
                sklearn_log_loss(
                    labels,
                    np.column_stack([1.0 - fake, fake]),
                    labels=[0, 1],
                )
            ),
            "ece": expected_calibration_error(labels, fake),
            "high_confidence_predictions": int(high.sum()),
            "high_confidence_errors": int((high & wrong).sum()),
            "high_confidence_accuracy": (
                float((~wrong[high]).mean()) if high.any() else None
            ),
            "mean_probability_fake": float(fake.mean()),
            "probability_std": float(fake.std()),
        }
    )
    return metrics


def classify_probability(probability_fake: float, lower: float, upper: float) -> str:
    """Map a calibrated fake probability to one of the three decisions."""
    if probability_fake < lower:
        return "likely_real"
    if probability_fake > upper:
        return "likely_fake"
    return "uncertain"


def threshold_metrics(
    y_true,
    probability_fake,
    lower: float,
    upper: float,
) -> dict:
    """Evaluate one three-level threshold variant."""
    labels = np.asarray(y_true, dtype=int)
    fake = np.asarray(probability_fake, dtype=float)
    binary = (fake >= 0.5).astype(int)
    decisions = np.asarray(
        [classify_probability(value, lower, upper) for value in fake],
        dtype=object,
    )
    strong = decisions != "uncertain"
    strong_correct = (
        ((labels == 0) & (decisions == "likely_real"))
        | ((labels == 1) & (decisions == "likely_fake"))
    )
    binary_wrong = binary != labels
    errors_in_uncertain = int((binary_wrong & ~strong).sum())
    total_errors = int(binary_wrong.sum())
    return {
        "lower_threshold": lower,
        "upper_threshold": upper,
        "likely_real": int(np.sum(decisions == "likely_real")),
        "uncertain": int(np.sum(decisions == "uncertain")),
        "likely_fake": int(np.sum(decisions == "likely_fake")),
        "strong_coverage": float(strong.mean()),
        "strong_accuracy": (
            float(strong_correct[strong].mean()) if strong.any() else 0.0
        ),
        "binary_errors": total_errors,
        "errors_in_uncertain": errors_in_uncertain,
        "errors_captured_fraction": (
            float(errors_in_uncertain / total_errors) if total_errors else 0.0
        ),
        "strong_false_positives": int(
            np.sum((labels == 0) & (decisions == "likely_fake"))
        ),
        "strong_false_negatives": int(
            np.sum((labels == 1) & (decisions == "likely_real"))
        ),
    }


def build_calibrated_svm(
    method: str,
    calibration_folds: list[tuple[np.ndarray, np.ndarray]],
) -> CalibratedClassifierCV:
    """Build the fixed Linear SVM with group-safe calibration folds."""
    if method not in CALIBRATION_METHODS:
        raise ValueError(f"Unknown calibration method: {method}")
    return CalibratedClassifierCV(
        estimator=build_svm_pipeline(BASELINE_C),
        method=method,
        cv=calibration_folds,
        ensemble=False,
        n_jobs=1,
    )


def nested_oof_calibration(
    train: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], int]:
    """Create unbiased OOF probabilities with nested group-safe calibration."""
    outer_folds, outer_groups, outer_audit = build_group_safe_folds(train)
    probability_by_method = {
        method: np.full(len(train), np.nan, dtype=float)
        for method in CALIBRATION_METHODS
    }
    outer_fold_by_row = np.full(len(train), -1, dtype=int)
    fold_rows: list[dict] = []

    for outer_fold, (fit_index, validation_index) in enumerate(outer_folds, start=1):
        outer_fit = train.iloc[fit_index].reset_index(drop=True)
        outer_validation = train.iloc[validation_index]
        inner_folds, _, inner_audit = build_group_safe_folds(outer_fit)
        if any(item["overlapping_groups"] != 0 for item in inner_audit):
            raise RuntimeError(f"Inner leakage detected in outer fold {outer_fold}.")

        for method in CALIBRATION_METHODS:
            LOGGER.info("Outer fold %s/5, calibration=%s", outer_fold, method)
            model = build_calibrated_svm(method, inner_folds)
            started = time.perf_counter()
            model.fit(outer_fit["model_text"], outer_fit["label"])
            training_seconds = time.perf_counter() - started
            _, probability_fake = fake_probabilities(
                model, outer_validation["model_text"]
            )
            probability_by_method[method][validation_index] = probability_fake
            metrics = probability_metrics(
                outer_validation["label"], probability_fake
            )
            fold_rows.append(
                {
                    "method": method,
                    "outer_fold": outer_fold,
                    "fit_rows": int(len(fit_index)),
                    "validation_rows": int(len(validation_index)),
                    "outer_fit_groups": int(len(np.unique(outer_groups[fit_index]))),
                    "outer_validation_groups": int(
                        len(np.unique(outer_groups[validation_index]))
                    ),
                    "outer_overlapping_groups": 0,
                    "inner_folds": int(len(inner_folds)),
                    "inner_overlapping_groups": 0,
                    "training_seconds": training_seconds,
                    **{
                        key: value
                        for key, value in metrics.items()
                        if key != "confusion_matrix"
                    },
                    "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
                }
            )
        outer_fold_by_row[validation_index] = outer_fold

    if np.any(outer_fold_by_row < 0):
        raise RuntimeError("Some train rows did not receive an outer OOF fold.")
    if any(np.isnan(values).any() for values in probability_by_method.values()):
        raise RuntimeError("Some train rows did not receive OOF probabilities.")

    prediction_tables: list[pd.DataFrame] = []
    for method, probability_fake in probability_by_method.items():
        table = train[
            ["article_id", "pair_id", "label", "label_name", "title"]
        ].copy()
        table.insert(0, "method", method)
        table["outer_fold"] = outer_fold_by_row
        table["probability_real"] = 1.0 - probability_fake
        table["probability_fake"] = probability_fake
        table["binary_prediction"] = (probability_fake >= 0.5).astype(int)
        table["confidence"] = np.maximum(probability_fake, 1.0 - probability_fake)
        table["prediction_correct"] = table["label"].eq(
            table["binary_prediction"]
        )
        prediction_tables.append(table)
    return (
        pd.concat(prediction_tables, ignore_index=True),
        pd.DataFrame(fold_rows),
        outer_audit,
        int(len(np.unique(outer_groups))),
    )


def summarize_calibration_methods(
    oof_predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate OOF probability quality and outer-fold stability."""
    rows: list[dict] = []
    for method in CALIBRATION_METHODS:
        table = oof_predictions.loc[oof_predictions["method"].eq(method)]
        metrics = probability_metrics(table["label"], table["probability_fake"])
        folds = fold_metrics.loc[fold_metrics["method"].eq(method)]
        rows.append(
            {
                "method": method,
                **{
                    key: value
                    for key, value in metrics.items()
                    if key != "confusion_matrix"
                },
                "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
                "std_brier_score": float(folds["brier_score"].std(ddof=1)),
                "std_log_loss": float(folds["log_loss"].std(ddof=1)),
                "std_ece": float(folds["ece"].std(ddof=1)),
                "std_f1_weighted": float(folds["f1_weighted"].std(ddof=1)),
                "mean_training_seconds": float(folds["training_seconds"].mean()),
                "total_training_seconds": float(folds["training_seconds"].sum()),
            }
        )
    return pd.DataFrame(rows)


def build_calibration_bin_output(oof_predictions: pd.DataFrame) -> pd.DataFrame:
    tables = []
    for method in CALIBRATION_METHODS:
        subset = oof_predictions.loc[oof_predictions["method"].eq(method)]
        bins = calibration_bins(subset["label"], subset["probability_fake"])
        bins.insert(0, "method", method)
        tables.append(bins)
    return pd.concat(tables, ignore_index=True)


def probability_distribution(oof_predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize probability distributions overall and by true class."""
    rows: list[dict] = []
    for method in CALIBRATION_METHODS:
        method_table = oof_predictions.loc[oof_predictions["method"].eq(method)]
        for label_name, subset in (
            ("all", method_table),
            ("real", method_table.loc[method_table["label"].eq(0)]),
            ("fake", method_table.loc[method_table["label"].eq(1)]),
        ):
            values = subset["probability_fake"].astype(float)
            rows.append(
                {
                    "method": method,
                    "true_label": label_name,
                    "rows": int(len(values)),
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "p10": float(values.quantile(0.10)),
                    "p25": float(values.quantile(0.25)),
                    "median": float(values.median()),
                    "p75": float(values.quantile(0.75)),
                    "p90": float(values.quantile(0.90)),
                    "max": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def select_calibration_method(method_comparison: pd.DataFrame) -> dict:
    """Select calibration from OOF train metrics, preferring sigmoid if close."""
    comparison = method_comparison.copy()
    best_brier = float(comparison["brier_score"].min())
    best_log_loss = float(comparison["log_loss"].min())
    close = comparison.loc[
        comparison["brier_score"].le(best_brier + METHOD_BRIER_TOLERANCE)
        & comparison["log_loss"].le(best_log_loss + METHOD_LOG_LOSS_TOLERANCE)
    ].copy()
    sigmoid_close = close.loc[close["method"].eq("sigmoid")]
    if not sigmoid_close.empty:
        selected = sigmoid_close.iloc[0]
        reason = "sigmoid_within_tolerance_and_lower_overfitting_risk"
    else:
        selected = comparison.sort_values(
            ["brier_score", "log_loss", "ece", "std_brier_score"],
            ascending=[True, True, True, True],
        ).iloc[0]
        reason = "best_oof_probability_metrics"
    return {
        "selected_method": str(selected["method"]),
        "method_selection_reason": reason,
        "brier_tolerance": METHOD_BRIER_TOLERANCE,
        "log_loss_tolerance": METHOD_LOG_LOSS_TOLERANCE,
        "selected_brier_score": float(selected["brier_score"]),
        "selected_log_loss": float(selected["log_loss"]),
        "selected_ece": float(selected["ece"]),
        "selected_std_brier_score": float(selected["std_brier_score"]),
    }


def evaluate_threshold_variants(
    y_true,
    probability_fake,
) -> pd.DataFrame:
    rows = []
    for variant in THRESHOLD_VARIANTS:
        rows.append(
            {
                **variant,
                **threshold_metrics(
                    y_true,
                    probability_fake,
                    variant["lower"],
                    variant["upper"],
                ),
            }
        )
    return pd.DataFrame(rows)


def select_thresholds(threshold_comparison: pd.DataFrame) -> dict:
    """Prefer coverage when strong accuracy is within 0.5 points of the best."""
    best_accuracy = float(threshold_comparison["strong_accuracy"].max())
    finalists = threshold_comparison.loc[
        threshold_comparison["strong_accuracy"].ge(
            best_accuracy - THRESHOLD_ACCURACY_TOLERANCE - 1e-12
        )
    ].copy()
    selected = finalists.sort_values(
        [
            "strong_coverage",
            "errors_captured_fraction",
            "strong_false_negatives",
            "strong_false_positives",
        ],
        ascending=[False, False, True, True],
    ).iloc[0]
    return {
        "threshold_name": str(selected["threshold_name"]),
        "lower_threshold": float(selected["lower"]),
        "upper_threshold": float(selected["upper"]),
        "threshold_selection_rule": (
            "Keep variants within 0.005 strong accuracy of the best, then "
            "prefer higher strong coverage, more captured errors, fewer strong FN/FP."
        ),
        "strong_accuracy_tolerance": THRESHOLD_ACCURACY_TOLERANCE,
        "oof_strong_coverage": float(selected["strong_coverage"]),
        "oof_strong_accuracy": float(selected["strong_accuracy"]),
        "oof_errors_in_uncertain": int(selected["errors_in_uncertain"]),
        "oof_strong_false_positives": int(selected["strong_false_positives"]),
        "oof_strong_false_negatives": int(selected["strong_false_negatives"]),
    }


def high_confidence_error_rows(
    table: pd.DataFrame,
    dataset: str,
    model_name: str,
    id_column: str,
) -> pd.DataFrame:
    """Return wrong predictions whose predicted-class confidence is at least 90%."""
    predictions = table["binary_prediction"].to_numpy(dtype=int)
    labels = table["label"].to_numpy(dtype=int)
    confidence = table["confidence"].to_numpy(dtype=float)
    mask = (predictions != labels) & (confidence >= HIGH_CONFIDENCE)
    columns = [id_column, "label", "title", "probability_fake", "confidence"]
    result = table.loc[mask, columns].copy().rename(columns={id_column: "case_id"})
    result.insert(0, "dataset", dataset)
    result.insert(1, "model", model_name)
    result["binary_prediction"] = predictions[mask]
    return result


def probability_prediction_table(
    dataframe: pd.DataFrame,
    model,
    model_name: str,
    id_column: str,
    lower: float,
    upper: float,
) -> pd.DataFrame:
    """Create probabilities, binary predictions, and frozen three-level decisions."""
    probability_real, probability_fake = fake_probabilities(
        model, dataframe["model_text"]
    )
    table = dataframe.copy().reset_index(drop=True)
    table.insert(0, "model", model_name)
    table["probability_real"] = probability_real
    table["probability_fake"] = probability_fake
    table["binary_prediction"] = (probability_fake >= 0.5).astype(int)
    table["confidence"] = np.maximum(probability_fake, probability_real)
    table["decision"] = [
        classify_probability(value, lower, upper) for value in probability_fake
    ]
    table["prediction_correct"] = table["label"].eq(
        table["binary_prediction"]
    )
    table["error_type"] = np.where(
        table["label"].eq(0) & table["binary_prediction"].eq(1),
        "false_positive",
        np.where(
            table["label"].eq(1) & table["binary_prediction"].eq(0),
            "false_negative",
            "correct",
        ),
    )
    if table[id_column].duplicated().any():
        raise ValueError(f"Duplicate IDs in probability input: {id_column}")
    return table


def verify_selection_hash(expected_hash: str) -> None:
    if not SELECTION_PATH.exists() or file_sha256(SELECTION_PATH) != expected_hash:
        raise RuntimeError("The frozen Day 16 calibration/threshold selection changed.")


def load_internal_test_after_selection(
    train: pd.DataFrame,
    selection_hash: str,
) -> tuple[pd.DataFrame, dict]:
    verify_selection_hash(selection_hash)
    raw_test = pd.read_csv(TEST_PATH, encoding="utf-8-sig", keep_default_na=False)
    test, stale_rows = refresh_model_text(raw_test)
    evaluation_test, excluded_ids = exclude_train_duplicates_from_test(train, test)
    return add_word_counts(evaluation_test), {
        "original_test_rows": int(len(raw_test)),
        "evaluation_test_rows": int(len(evaluation_test)),
        "exact_train_duplicates_excluded": int(len(excluded_ids)),
        "excluded_article_ids": excluded_ids,
        "stale_test_model_text_rows_refreshed_in_memory": stale_rows,
    }


def train_final_calibrated_model(
    train: pd.DataFrame,
    method: str,
) -> tuple[CalibratedClassifierCV, dict]:
    """Fit the selected calibrated model on full train with group-safe folds."""
    folds, groups, audit = build_group_safe_folds(train)
    model = build_calibrated_svm(method, folds)
    started = time.perf_counter()
    model.fit(train["model_text"], train["label"])
    training_seconds = time.perf_counter() - started
    joblib.dump(model, CALIBRATED_MODEL_PATH, compress=3)
    size_bytes = CALIBRATED_MODEL_PATH.stat().st_size
    return model, {
        "method": method,
        "training_seconds": round(training_seconds, 3),
        "model_size_mb": round(size_bytes / (1024 * 1024), 3),
        "model_path": str(CALIBRATED_MODEL_PATH.relative_to(PROJECT_ROOT)),
        "model_sha256": file_sha256(CALIBRATED_MODEL_PATH),
        "calibration_folds": len(folds),
        "group_count": int(len(np.unique(groups))),
        "overlapping_groups": int(sum(item["overlapping_groups"] for item in audit)),
    }


def model_comparison_table(
    dataframe: pd.DataFrame,
    models: dict[str, object],
    dataset_name: str,
    id_column: str,
    lower: float,
    upper: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[pd.DataFrame]]:
    """Evaluate the new model and current app model on exactly the same rows."""
    prediction_tables = []
    metric_rows = []
    high_confidence_tables = []
    for model_name, model in models.items():
        table = probability_prediction_table(
            dataframe, model, model_name, id_column, lower, upper
        )
        metrics = probability_metrics(table["label"], table["probability_fake"])
        thresholds = threshold_metrics(
            table["label"], table["probability_fake"], lower, upper
        )
        metric_rows.append(
            {
                "model": model_name,
                **{
                    key: value
                    for key, value in metrics.items()
                    if key != "confusion_matrix"
                },
                "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
                **{f"threshold_{key}": value for key, value in thresholds.items()},
            }
        )
        prediction_tables.append(table)
        high_confidence_tables.append(
            high_confidence_error_rows(
                table, dataset_name, model_name, id_column
            )
        )
    return (
        pd.DataFrame(metric_rows),
        pd.concat(prediction_tables, ignore_index=True),
        high_confidence_tables,
    )


def evaluate_length_behavior(
    table: pd.DataFrame,
    lower: float,
    upper: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate selected calibrated probabilities across fixed length cohorts."""
    length_rows: list[dict] = []
    for group_name in LENGTH_LABELS:
        subset = table.loc[table["length_group"].astype(str).eq(group_name)]
        metrics = probability_metrics(subset["label"], subset["probability_fake"])
        thresholds = threshold_metrics(
            subset["label"], subset["probability_fake"], lower, upper
        )
        length_rows.append(
            {
                "length_group": group_name,
                "length_description": LENGTH_DISPLAY[group_name],
                **{
                    key: value
                    for key, value in metrics.items()
                    if key != "confusion_matrix"
                },
                "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
                **{f"threshold_{key}": value for key, value in thresholds.items()},
                "mean_probability_fake_real": (
                    float(
                        subset.loc[
                            subset["label"].eq(0), "probability_fake"
                        ].mean()
                    )
                    if subset["label"].eq(0).any()
                    else None
                ),
                "mean_probability_fake_fake": (
                    float(
                        subset.loc[
                            subset["label"].eq(1), "probability_fake"
                        ].mean()
                    )
                    if subset["label"].eq(1).any()
                    else None
                ),
            }
        )

    special_rows = []
    special = {
        "real_30_60": table.loc[
            table["label"].eq(0) & table["word_count"].between(30, 60)
        ],
        "fake_gt_250": table.loc[
            table["label"].eq(1) & table["word_count"].gt(250)
        ],
    }
    for name, subset in special.items():
        metrics = probability_metrics(subset["label"], subset["probability_fake"])
        thresholds = threshold_metrics(
            subset["label"], subset["probability_fake"], lower, upper
        )
        special_rows.append(
            {
                "cohort": name,
                **{
                    key: value
                    for key, value in metrics.items()
                    if key != "confusion_matrix"
                },
                "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
                **{f"threshold_{key}": value for key, value in thresholds.items()},
                "mean_probability_fake": float(subset["probability_fake"].mean()),
            }
        )

    correlations = {}
    for scope, subset in (
        ("all", table),
        ("real", table.loc[table["label"].eq(0)]),
        ("fake", table.loc[table["label"].eq(1)]),
    ):
        result = spearmanr(subset["word_count"], subset["probability_fake"])
        correlations[scope] = (float(result.statistic), float(result.pvalue))
    bias = pd.DataFrame(
        [
            {
                "spearman_all": correlations["all"][0],
                "spearman_all_p": correlations["all"][1],
                "spearman_real": correlations["real"][0],
                "spearman_real_p": correlations["real"][1],
                "spearman_fake": correlations["fake"][0],
                "spearman_fake_p": correlations["fake"][1],
                "mean_absolute_within_label_spearman": (
                    abs(correlations["real"][0]) + abs(correlations["fake"][0])
                )
                / 2,
            }
        ]
    )
    return pd.DataFrame(length_rows), pd.DataFrame(special_rows), bias


def load_external_after_selection(selection_hash: str) -> tuple[pd.DataFrame, int]:
    verify_selection_hash(selection_hash)
    external = pd.read_csv(EXTERNAL_PATH, encoding="utf-8", keep_default_na=False)
    if len(external) != 40 or set(external["label"]) != {"real", "fake"}:
        raise ValueError("External benchmark is not the expected frozen dataset.")
    external["label"] = external["label"].map({"real": 0, "fake": 1})
    external, stale_rows = refresh_model_text(external)
    return add_word_counts(external), stale_rows

