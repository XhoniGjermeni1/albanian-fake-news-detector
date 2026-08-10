from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss as sklearn_log_loss

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.analyze_length_domain_shift import (  # noqa: E402
    LENGTH_DISPLAY,
    LENGTH_LABELS,
)
from src.models.compare_classifiers import (  # noqa: E402
    add_word_counts,
    build_group_safe_folds,
    classification_metrics,
    dataframe_to_markdown,
    file_sha256,
    refresh_model_text,
    rounded_metrics,
)
from src.models.train_hybrid_model import (  # noqa: E402
    exclude_train_duplicates_from_test,
)
from src.models.tune_linear_svm import (  # noqa: E402
    BASELINE_C,
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


def plot_oof_calibration(
    bins: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    colors = {"sigmoid": "#3976A8", "isotonic": "#D76745"}
    for method in CALIBRATION_METHODS:
        method_bins = bins.loc[
            bins["method"].eq(method) & bins["rows"].gt(0)
        ]
        axes[0].plot(
            method_bins["mean_probability_fake"],
            method_bins["fraction_fake"],
            marker="o",
            linewidth=2,
            label=method.capitalize(),
            color=colors[method],
        )
        values = predictions.loc[
            predictions["method"].eq(method), "probability_fake"
        ]
        axes[1].hist(
            values,
            bins=np.linspace(0, 1, 21),
            alpha=0.5,
            label=method.capitalize(),
            color=colors[method],
        )
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#333333")
    axes[0].set_xlabel("Probability fake mesatare")
    axes[0].set_ylabel("Përqindja reale fake")
    axes[0].set_title("Reliability curve, nested OOF train")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.25)
    axes[1].set_xlabel("Probability fake")
    axes[1].set_ylabel("Numri i artikujve")
    axes[1].set_title("Shpërndarja e probabiliteteve")
    axes[1].legend(loc="best")
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(OOF_FIGURE_PATH, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_thresholds(comparison: pd.DataFrame, selected_name: str) -> None:
    data = comparison.copy()
    x = np.arange(len(data))
    width = 0.36
    figure, axis = plt.subplots(figsize=(9, 5.5))
    coverage = axis.bar(
        x - width / 2,
        data["strong_coverage"],
        width,
        label="Strong coverage",
        color="#3976A8",
    )
    accuracy = axis.bar(
        x + width / 2,
        data["strong_accuracy"],
        width,
        label="Strong accuracy",
        color="#2F937F",
    )
    axis.bar_label(coverage, labels=[f"{value:.2f}" for value in data["strong_coverage"]], padding=3)
    axis.bar_label(accuracy, labels=[f"{value:.2f}" for value in data["strong_accuracy"]], padding=3)
    labels = [
        f"{name}\n(selected)" if name == selected_name else name
        for name in data["threshold_name"]
    ]
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("Rezultati")
    axis.set_title("Zgjedhja e zonës uncertain nga OOF train")
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
    )
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(THRESHOLD_FIGURE_PATH, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_internal_calibration(bins: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    colors = {"new_calibrated_svm": "#D76745", "current_app_model": "#3976A8"}
    for model_name, table in bins.groupby("model", sort=False):
        non_empty = table.loc[table["rows"].gt(0)]
        axis.plot(
            non_empty["mean_probability_fake"],
            non_empty["fraction_fake"],
            marker="o",
            linewidth=2,
            label=model_name.replace("_", " ").title(),
            color=colors[model_name],
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color="#333333")
    axis.set_xlabel("Probability fake mesatare")
    axis.set_ylabel("Përqindja reale fake")
    axis.set_title("Calibration në test set-in e brendshëm")
    axis.legend(loc="best")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(INTERNAL_FIGURE_PATH, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_length_probability(length_metrics: pd.DataFrame) -> None:
    data = length_metrics.set_index("length_group").loc[LENGTH_LABELS]
    x = np.arange(len(data))
    width = 0.36
    figure, axis = plt.subplots(figsize=(11, 5.7))
    axis.bar(
        x - width / 2,
        data["mean_probability_fake_real"],
        width,
        label="Label real",
        color="#3976A8",
    )
    axis.bar(
        x + width / 2,
        data["mean_probability_fake_fake"],
        width,
        label="Label fake",
        color="#D76745",
    )
    axis.set_xticks(x, [LENGTH_DISPLAY[name] for name in LENGTH_LABELS], rotation=8)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Probability fake mesatare")
    axis.set_title("Probabiliteti sipas gjatësisë dhe label-it")
    axis.legend(loc="best")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(LENGTH_FIGURE_PATH, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_model_comparison(
    internal: pd.DataFrame,
    external: pd.DataFrame,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    models = ["current_app_model", "new_calibrated_svm"]
    colors = ["#3976A8", "#D76745"]
    x = np.arange(3)
    width = 0.34
    for index, model_name in enumerate(models):
        row = internal.loc[internal["model"].eq(model_name)].iloc[0]
        axes[0].bar(
            x + (index - 0.5) * width,
            [row["accuracy"], row["f1_weighted"], row["f1_fake"]],
            width,
            label=model_name.replace("_", " ").title(),
            color=colors[index],
        )
    axes[0].set_xticks(x, ["Accuracy", "F1 weighted", "F1 fake"])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Test-i i brendshëm")
    axes[0].grid(axis="y", alpha=0.25)

    for index, model_name in enumerate(models):
        row = external.loc[external["model"].eq(model_name)].iloc[0]
        axes[1].bar(
            x + (index - 0.5) * width,
            [row["accuracy"], row["recall_real"], row["recall_fake"]],
            width,
            label=model_name.replace("_", " ").title(),
            color=colors[index],
        )
    axes[1].set_xticks(x, ["Accuracy", "Recall real", "Recall fake"])
    axes[1].set_ylim(0, 1)
    axes[1].set_title("External, vetëm diagnostik")
    axes[1].grid(axis="y", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=2,
    )
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    figure.savefig(MODEL_COMPARISON_FIGURE_PATH, dpi=160, bbox_inches="tight")
    plt.close(figure)


def write_report(
    metrics: dict,
    method_comparison: pd.DataFrame,
    probability_summary: pd.DataFrame,
    threshold_comparison: pd.DataFrame,
    internal_comparison: pd.DataFrame,
    internal_thresholds: pd.DataFrame,
    length_metrics: pd.DataFrame,
    special_cohorts: pd.DataFrame,
    length_bias: pd.DataFrame,
    external_comparison: pd.DataFrame,
    external_thresholds: pd.DataFrame,
) -> None:
    selection = metrics["selection"]
    method = selection["calibration"]["selected_method"]
    lower = selection["thresholds"]["lower_threshold"]
    upper = selection["thresholds"]["upper_threshold"]
    selected_method_row = method_comparison.loc[
        method_comparison["method"].eq(method)
    ].iloc[0]
    new_internal = internal_comparison.loc[
        internal_comparison["model"].eq("new_calibrated_svm")
    ].iloc[0]
    current_internal = internal_comparison.loc[
        internal_comparison["model"].eq("current_app_model")
    ].iloc[0]
    new_external = external_comparison.loc[
        external_comparison["model"].eq("new_calibrated_svm")
    ].iloc[0]
    current_external = external_comparison.loc[
        external_comparison["model"].eq("current_app_model")
    ].iloc[0]
    day15_bias = metrics["length_analysis"]["day15_raw_score_bias"]
    calibrated_bias = float(length_bias.iloc[0]["mean_absolute_within_label_spearman"])

    method_table = dataframe_to_markdown(
        method_comparison,
        [
            "method",
            "brier_score",
            "log_loss",
            "ece",
            "accuracy",
            "f1_weighted",
            "f1_fake",
            "std_brier_score",
            "std_f1_weighted",
            "high_confidence_predictions",
            "high_confidence_errors",
            "mean_training_seconds",
        ],
    )
    probability_table = dataframe_to_markdown(
        probability_summary,
        ["method", "true_label", "rows", "mean", "std", "p10", "median", "p90"],
    )
    threshold_table = dataframe_to_markdown(
        threshold_comparison,
        [
            "threshold_name",
            "likely_real",
            "uncertain",
            "likely_fake",
            "strong_coverage",
            "strong_accuracy",
            "errors_in_uncertain",
            "strong_false_positives",
            "strong_false_negatives",
        ],
    )
    internal_table = dataframe_to_markdown(
        internal_comparison,
        [
            "model",
            "accuracy",
            "f1_weighted",
            "f1_fake",
            "recall_real",
            "recall_fake",
            "brier_score",
            "log_loss",
            "ece",
            "high_confidence_errors",
            "confusion_matrix",
        ],
    )
    internal_threshold_table = dataframe_to_markdown(
        internal_thresholds,
        [
            "model",
            "likely_real",
            "uncertain",
            "likely_fake",
            "strong_coverage",
            "strong_accuracy",
            "errors_in_uncertain",
            "strong_false_positives",
            "strong_false_negatives",
        ],
    )
    length_table = dataframe_to_markdown(
        length_metrics,
        [
            "length_description",
            "rows",
            "accuracy",
            "f1_weighted",
            "recall_real",
            "recall_fake",
            "brier_score",
            "ece",
            "mean_probability_fake_real",
            "mean_probability_fake_fake",
        ],
    )
    special_table = dataframe_to_markdown(
        special_cohorts,
        [
            "cohort",
            "rows",
            "accuracy",
            "recall_real",
            "recall_fake",
            "mean_probability_fake",
            "threshold_uncertain",
            "confusion_matrix",
        ],
    )
    external_table = dataframe_to_markdown(
        external_comparison,
        [
            "model",
            "accuracy",
            "recall_real",
            "recall_fake",
            "brier_score",
            "log_loss",
            "ece",
            "high_confidence_errors",
            "confusion_matrix",
        ],
    )
    external_threshold_table = dataframe_to_markdown(
        external_thresholds,
        [
            "model",
            "likely_real",
            "uncertain",
            "likely_fake",
            "strong_coverage",
            "strong_accuracy",
            "errors_in_uncertain",
            "strong_false_positives",
            "strong_false_negatives",
        ],
    )

    report = f"""# Dita 16 - Calibration dhe pragjet uncertain

## Protokolli

Konfigurimi u mbajt fiks: Word + Character TF-IDF, Linear SVM, `C=1.0`.
Sigmoid dhe isotonic u krahasuan me nested 5x5 group-safe CV vetëm mbi
{metrics['data_audit']['train_rows']} artikujt train dhe
{metrics['data_audit']['group_count']} leakage-groups. Outer folds prodhuan
probabilitete OOF për vlerësim; inner folds trajnuan calibration-in. Në asnjë
nivel nuk pati mbivendosje grupesh.

Metoda dhe pragjet u shkruan te `reports/day16_selection.json` përpara se të
ngarkoheshin test-i i brendshëm, modeli aktual i aplikacionit ose dataset-i i
jashtëm. Streamlit dhe modeli aktual nuk u zëvendësuan.

## Krahasimi i calibration-it në OOF train

{method_table}

U zgjodh **{method}**. Arsyeja e ruajtur ishte
`{selection['calibration']['method_selection_reason']}`. Brier score ishte
{selected_method_row['brier_score']:.4f}, log loss
{selected_method_row['log_loss']:.4f} dhe ECE
{selected_method_row['ece']:.4f}. Sigmoid preferohet ndaj isotonic kur është
brenda tolerancës, sepse ka formë parametrike dhe rrezik më të ulët overfitting.

![OOF calibration](figures/day16_oof_calibration_comparison.png)

### Shpërndarja e probabiliteteve

{probability_table}

## Zgjedhja e pragjeve nga OOF train

{threshold_table}

U zgjodhën pragjet **{lower:.2f}/{upper:.2f}**. Variantet brenda 0.005 strong
accuracy nga më i miri u krahasuan sipas coverage, kapjes së gabimeve dhe
gabimeve të forta. Test set-i nuk u përdor.

![Pragjet](figures/day16_threshold_comparison.png)

## Test set-i i brendshëm

Pas ngrirjes u përjashtuan
{metrics['data_audit']['exact_train_duplicates_excluded']} dublikata ekzakte
dhe mbetën {metrics['data_audit']['evaluation_test_rows']} artikuj.

{internal_table}

Modeli i ri arriti accuracy {new_internal['accuracy']:.4f}, F1 weighted
{new_internal['f1_weighted']:.4f}, F1 fake {new_internal['f1_fake']:.4f},
Brier {new_internal['brier_score']:.4f}, log loss
{new_internal['log_loss']:.4f} dhe ECE {new_internal['ece']:.4f}. Gabimet me
confidence të paktën 90% ishin {int(new_internal['high_confidence_errors'])}.

Me pragjet e ngrira:

{internal_threshold_table}

![Internal calibration](figures/day16_internal_calibration.png)

## Sjellja sipas gjatësisë

{length_table}

{special_table}

Mean absolute within-label Spearman ishte {calibrated_bias:.4f} pas calibration,
kundrejt {day15_bias:.4f} për raw decision score në Ditën 15. Calibration nuk
e zgjidhi bias-in e gjatësisë; ndryshimi interpretohet vetëm si transformim i
score-it në probability.

![Gjatësia](figures/day16_length_probability.png)

## Dataset-i i jashtëm vetëm diagnostik

{external_table}

Për modelin e ri, pragjet e ngrira dhanë:

{external_threshold_table}

Brier/log loss i jashtëm raportohet sepse rastet kanë etiketa të dokumentuara,
por kampioni ka vetëm 40 përmbledhje dhe nuk është calibration set. Asnjë
rezultat i jashtëm nuk ndryshoi metodën ose pragjet.

## Krahasimi me modelin aktual të aplikacionit

- Në test-in e brendshëm, F1 weighted ndryshoi nga
  {current_internal['f1_weighted']:.4f} në {new_internal['f1_weighted']:.4f};
  Brier nga {current_internal['brier_score']:.4f} në
  {new_internal['brier_score']:.4f}.
- Jashtë corpus-it, accuracy ndryshoi nga {current_external['accuracy']:.4f} në
  {new_external['accuracy']:.4f}; recall real/fake i modelit të ri ishte
  {new_external['recall_real']:.4f}/{new_external['recall_fake']:.4f}.
- Modeli i ri fiton përfaqësim character dhe performancë të brendshme më të
  lartë; humbet thjeshtësinë e Logistic Regression dhe mbetet i ekspozuar ndaj
  domain shift-it dhe bias-it të gjatësisë.

![Krahasimi i modeleve](figures/day16_model_comparison.png)

## Rekomandimi për Ditën 17

Rekomandohet **Word + Character TF-IDF + Linear SVM, C=1.0 + {method}** me
pragje **{lower:.2f}/{upper:.2f}** si kandidat për ngrirjen finale. Para
integrimit duhen verifikuar artefakti, funksioni i prediction, versionet e
dependencies dhe testet e regresionit të Streamlit.

## Kufizimet

- Calibration selection përdor nested CV, por ende të njëjtin corpus burimor.
- ECE varet nga 10 bin-et e zgjedhura.
- Isotonic ka mjaft raste, por mund të overfit-ojë më lehtë se sigmoid.
- Calibration nuk korrigjon domain shift, source-label confounding ose bias-in
  e gjatësisë.
- Benchmark-u i jashtëm është i vogël dhe me përmbledhje manuale.
- Modeli i ri nuk është integruar ende në Streamlit.

Modeli eksperimental ruhet te
`models/day16_word_char_linear_svm_calibrated.joblib`.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def run_day16_calibration() -> dict:
    """Run nested calibration selection, threshold selection, and frozen tests."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    frozen_paths = {
        "day15_selection": DAY15_SELECTION_PATH,
        "day15_length_bias": DAY15_LENGTH_BIAS_PATH,
        "current_app_model": CURRENT_APP_MODEL_PATH,
        "streamlit_app": STREAMLIT_APP_PATH,
        "external_dataset": EXTERNAL_PATH,
    }
    required = [TRAIN_PATH, TEST_PATH, *frozen_paths.values()]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Day 16 inputs: {missing}")
    hashes_before = {name: file_sha256(path) for name, path in frozen_paths.items()}

    frozen_setup = verify_frozen_day15()
    raw_train = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig", keep_default_na=False)
    train, stale_train_rows = refresh_model_text(raw_train)
    if train["model_text"].str.strip().eq("").any():
        raise ValueError("Train contains empty model_text values.")

    LOGGER.info("Starting nested OOF calibration comparison")
    oof_predictions, calibration_folds, outer_audit, group_count = (
        nested_oof_calibration(train)
    )
    method_comparison = summarize_calibration_methods(
        oof_predictions, calibration_folds
    )
    oof_bins = build_calibration_bin_output(oof_predictions)
    probability_summary = probability_distribution(oof_predictions)
    calibration_selection = select_calibration_method(method_comparison)
    selected_method = calibration_selection["selected_method"]
    selected_oof = oof_predictions.loc[
        oof_predictions["method"].eq(selected_method)
    ].copy()
    threshold_comparison = evaluate_threshold_variants(
        selected_oof["label"], selected_oof["probability_fake"]
    )
    threshold_selection = select_thresholds(threshold_comparison)

    selection = {
        "selection_scope": "nested_group_safe_oof_train_only",
        "fixed_configuration": {
            "representation": "word_char_tfidf",
            "classifier": "linear_svm",
            "c_value": BASELINE_C,
        },
        "calibration": calibration_selection,
        "thresholds": threshold_selection,
        "internal_test_used": False,
        "current_app_model_used": False,
        "external_results_used": False,
        "outer_fold_audit": outer_audit,
        "group_count": group_count,
    }
    SELECTION_PATH.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    selection_hash = file_sha256(SELECTION_PATH)

    oof_predictions.to_csv(OOF_PREDICTIONS_PATH, index=False, encoding="utf-8")
    calibration_folds.to_csv(CALIBRATION_FOLDS_PATH, index=False, encoding="utf-8")
    method_comparison.to_csv(METHOD_COMPARISON_PATH, index=False, encoding="utf-8")
    oof_bins.to_csv(OOF_BINS_PATH, index=False, encoding="utf-8")
    probability_summary.to_csv(
        PROBABILITY_DISTRIBUTION_PATH, index=False, encoding="utf-8"
    )
    threshold_comparison.to_csv(
        THRESHOLD_COMPARISON_PATH, index=False, encoding="utf-8"
    )
    plot_oof_calibration(oof_bins, oof_predictions)
    plot_thresholds(threshold_comparison, threshold_selection["threshold_name"])

    # Test and current app model remain unopened until both choices are frozen.
    test, test_audit = load_internal_test_after_selection(train, selection_hash)
    calibrated_model, training_metadata = train_final_calibrated_model(
        train, selected_method
    )
    current_app_model = joblib.load(CURRENT_APP_MODEL_PATH)
    lower = threshold_selection["lower_threshold"]
    upper = threshold_selection["upper_threshold"]
    models = {
        "new_calibrated_svm": calibrated_model,
        "current_app_model": current_app_model,
    }
    (
        internal_comparison,
        internal_all_predictions,
        internal_high_confidence,
    ) = model_comparison_table(
        test, models, "internal_test", "article_id", lower, upper
    )
    new_internal = internal_all_predictions.loc[
        internal_all_predictions["model"].eq("new_calibrated_svm")
    ].copy()
    internal_threshold_rows = []
    internal_bin_tables = []
    for model_name, table in internal_all_predictions.groupby("model", sort=False):
        threshold_result = threshold_metrics(
            table["label"], table["probability_fake"], lower, upper
        )
        internal_threshold_rows.append({"model": model_name, **threshold_result})
        bins = calibration_bins(table["label"], table["probability_fake"])
        bins.insert(0, "model", model_name)
        internal_bin_tables.append(bins)
    internal_thresholds = pd.DataFrame(internal_threshold_rows)
    internal_bins = pd.concat(internal_bin_tables, ignore_index=True)
    length_metrics, special_cohorts, length_bias = evaluate_length_behavior(
        new_internal, lower, upper
    )

    internal_columns = [
        "model",
        "article_id",
        "pair_id",
        "label",
        "label_name",
        "title",
        "word_count",
        "length_group",
        "probability_real",
        "probability_fake",
        "binary_prediction",
        "confidence",
        "decision",
        "prediction_correct",
        "error_type",
    ]
    internal_all_predictions[internal_columns].to_csv(
        INTERNAL_PREDICTIONS_PATH, index=False, encoding="utf-8"
    )
    internal_comparison.to_csv(
        INTERNAL_MODEL_COMPARISON_PATH, index=False, encoding="utf-8"
    )
    internal_bins.to_csv(INTERNAL_BINS_PATH, index=False, encoding="utf-8")
    internal_thresholds.to_csv(INTERNAL_THRESHOLD_PATH, index=False, encoding="utf-8")
    length_metrics.to_csv(LENGTH_METRICS_PATH, index=False, encoding="utf-8")
    special_cohorts.to_csv(SPECIAL_COHORTS_PATH, index=False, encoding="utf-8")
    length_bias.to_csv(LENGTH_BIAS_PATH, index=False, encoding="utf-8")
    plot_internal_calibration(internal_bins)
    plot_length_probability(length_metrics)

    # External data is loaded only after calibration and thresholds are immutable.
    external, stale_external_rows = load_external_after_selection(selection_hash)
    (
        external_comparison,
        external_all_predictions,
        external_high_confidence,
    ) = model_comparison_table(
        external, models, "external", "external_id", lower, upper
    )
    new_external = external_all_predictions.loc[
        external_all_predictions["model"].eq("new_calibrated_svm")
    ].copy()
    external_threshold_rows = []
    for model_name, table in external_all_predictions.groupby("model", sort=False):
        external_threshold_rows.append(
            {
                "model": model_name,
                **threshold_metrics(
                    table["label"], table["probability_fake"], lower, upper
                ),
            }
        )
    external_thresholds = pd.DataFrame(external_threshold_rows)
    external_columns = [
        "model",
        "external_id",
        "label",
        "title",
        "topic",
        "source",
        "word_count",
        "probability_real",
        "probability_fake",
        "binary_prediction",
        "confidence",
        "decision",
        "prediction_correct",
        "error_type",
    ]
    external_all_predictions[external_columns].to_csv(
        EXTERNAL_PREDICTIONS_PATH, index=False, encoding="utf-8"
    )
    external_comparison.to_csv(
        EXTERNAL_MODEL_COMPARISON_PATH, index=False, encoding="utf-8"
    )
    external_thresholds.to_csv(
        EXTERNAL_THRESHOLD_PATH, index=False, encoding="utf-8"
    )

    oof_high_confidence = high_confidence_error_rows(
        selected_oof,
        "oof_train",
        f"new_svm_{selected_method}",
        "article_id",
    )
    all_high_confidence = [
        oof_high_confidence,
        *internal_high_confidence,
        *external_high_confidence,
    ]
    non_empty_high_confidence = [
        table for table in all_high_confidence if not table.empty
    ]
    if non_empty_high_confidence:
        pd.concat(non_empty_high_confidence, ignore_index=True).to_csv(
            HIGH_CONFIDENCE_ERRORS_PATH, index=False, encoding="utf-8"
        )
    else:
        pd.DataFrame(
            columns=[
                "dataset",
                "model",
                "case_id",
                "label",
                "title",
                "probability_fake",
                "confidence",
                "binary_prediction",
            ]
        ).to_csv(HIGH_CONFIDENCE_ERRORS_PATH, index=False, encoding="utf-8")

    plot_model_comparison(internal_comparison, external_comparison)

    verify_selection_hash(selection_hash)
    hashes_after = {name: file_sha256(path) for name, path in frozen_paths.items()}
    if hashes_before != hashes_after:
        changed = [
            name for name in frozen_paths if hashes_before[name] != hashes_after[name]
        ]
        raise RuntimeError(f"Frozen artifacts changed during Day 16: {changed}")

    day15_bias_table = pd.read_csv(DAY15_LENGTH_BIAS_PATH)
    day15_raw_bias = float(
        day15_bias_table.loc[
            day15_bias_table["c_value"].eq(BASELINE_C),
            "mean_absolute_within_label_spearman",
        ].iloc[0]
    )
    data_audit = {
        "train_rows": int(len(train)),
        "train_real": int(train["label"].eq(0).sum()),
        "train_fake": int(train["label"].eq(1).sum()),
        "group_count": group_count,
        "stale_train_model_text_rows_refreshed_in_memory": stale_train_rows,
        **test_audit,
        "external_rows": int(len(external)),
        "stale_external_model_text_rows_refreshed_in_memory": stale_external_rows,
    }
    metrics = {
        "status": "completed",
        "protocol": {
            "selection_data": "nested_oof_train_only",
            "outer_cv": "5_fold_stratified_group_safe",
            "inner_calibration_cv": "5_fold_stratified_group_safe",
            "internal_test_used_for_selection": False,
            "current_app_model_used_for_selection": False,
            "external_used_for_calibration_or_thresholds": False,
            "selection_locked_before_internal_test": True,
            "selection_locked_before_external": True,
            "fixed_word_char_tfidf": True,
            "fixed_linear_svm_c": BASELINE_C,
        },
        "frozen_setup": frozen_setup,
        "data_audit": data_audit,
        "selection": selection,
        "calibration_method_metrics": [
            rounded_metrics(row)
            for row in method_comparison.to_dict(orient="records")
        ],
        "threshold_metrics_oof": [
            rounded_metrics(row)
            for row in threshold_comparison.to_dict(orient="records")
        ],
        "training_metadata": training_metadata,
        "internal_model_comparison": [
            rounded_metrics(row)
            for row in internal_comparison.to_dict(orient="records")
        ],
        "internal_threshold_metrics": [
            rounded_metrics(row)
            for row in internal_thresholds.to_dict(orient="records")
        ],
        "length_analysis": {
            "day15_raw_score_bias": day15_raw_bias,
            "calibrated_probability_bias": rounded_metrics(
                length_bias.iloc[0].to_dict()
            ),
        },
        "external_model_comparison": [
            rounded_metrics(row)
            for row in external_comparison.to_dict(orient="records")
        ],
        "external_threshold_metrics": [
            rounded_metrics(row)
            for row in external_thresholds.to_dict(orient="records")
        ],
        "integrity": {
            "hashes_before": hashes_before,
            "hashes_after": hashes_after,
            "all_frozen_artifacts_unchanged": hashes_before == hashes_after,
            "selection_sha256": selection_hash,
            "selection_unchanged_after_internal_and_external": (
                file_sha256(SELECTION_PATH) == selection_hash
            ),
        },
        "artifacts": {
            "model": str(CALIBRATED_MODEL_PATH.relative_to(PROJECT_ROOT)),
            "oof_predictions": str(OOF_PREDICTIONS_PATH.relative_to(PROJECT_ROOT)),
            "method_comparison": str(METHOD_COMPARISON_PATH.relative_to(PROJECT_ROOT)),
            "threshold_comparison": str(
                THRESHOLD_COMPARISON_PATH.relative_to(PROJECT_ROOT)
            ),
            "selection": str(SELECTION_PATH.relative_to(PROJECT_ROOT)),
            "internal_predictions": str(
                INTERNAL_PREDICTIONS_PATH.relative_to(PROJECT_ROOT)
            ),
            "external_predictions": str(
                EXTERNAL_PREDICTIONS_PATH.relative_to(PROJECT_ROOT)
            ),
            "high_confidence_errors": str(
                HIGH_CONFIDENCE_ERRORS_PATH.relative_to(PROJECT_ROOT)
            ),
            "report": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
        },
    }
    METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(
        metrics,
        method_comparison,
        probability_summary,
        threshold_comparison,
        internal_comparison,
        internal_thresholds,
        length_metrics,
        special_cohorts,
        length_bias,
        external_comparison,
        external_thresholds,
    )
    return metrics


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    metrics = run_day16_calibration()
    selection = metrics["selection"]
    print("Day 16 completed.")
    print("Calibration:", selection["calibration"]["selected_method"])
    print(
        "Thresholds:",
        selection["thresholds"]["lower_threshold"],
        selection["thresholds"]["upper_threshold"],
    )
    print("External used for selection:", selection["external_results_used"])
    print("Report:", REPORT_PATH)


if __name__ == "__main__":
    main()
