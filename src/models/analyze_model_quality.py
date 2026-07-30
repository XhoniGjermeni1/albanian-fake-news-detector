"""Run Day 6 error analysis, calibration, and threshold evaluation."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold

from src.models.predict import predict_news, predict_news_for_app
from src.models.train_hybrid_model import (
    build_tfidf_model,
    exclude_train_duplicates_from_test,
    merge_text_with_features,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = PROJECT_ROOT / "data" / "interim" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "interim" / "test.csv"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "linguistic_features.csv"
BASELINE_MODEL_PATH = PROJECT_ROOT / "models" / "baseline_tfidf_logreg.joblib"
HYBRID_MODEL_PATH = PROJECT_ROOT / "models" / "hybrid_tfidf_linguistic_logreg.joblib"
CALIBRATED_MODEL_PATH = PROJECT_ROOT / "models" / "calibrated_tfidf_logreg.joblib"
DAY5_METRICS_PATH = PROJECT_ROOT / "reports" / "day5_metrics.json"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_PATH = REPORTS_DIR / "day6_metrics.json"
ERRORS_PATH = REPORTS_DIR / "day6_error_analysis.csv"
INTERESTING_ERRORS_PATH = REPORTS_DIR / "day6_interesting_errors.csv"
PROBABILITY_SUMMARY_PATH = REPORTS_DIR / "day6_probability_summary.csv"
THRESHOLD_PATH = REPORTS_DIR / "day6_threshold_comparison.csv"
CALIBRATION_BINS_PATH = REPORTS_DIR / "day6_calibration_bins.csv"
CALIBRATION_FIGURE_PATH = FIGURES_DIR / "day6_probability_calibration.png"
REPORT_PATH = REPORTS_DIR / "day6_model_quality.md"

THRESHOLD_VARIANTS = [
    ("35-65", 0.35, 0.65),
    ("40-60", 0.40, 0.60),
    ("30-70", 0.30, 0.70),
]

LOGGER = logging.getLogger(__name__)


def load_day6_inputs() -> tuple[pd.DataFrame, pd.DataFrame, object, dict]:
    """Load and validate all artifacts needed for Day 6."""
    required_paths = [
        TRAIN_PATH,
        TEST_PATH,
        FEATURES_PATH,
        BASELINE_MODEL_PATH,
        HYBRID_MODEL_PATH,
        DAY5_METRICS_PATH,
    ]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing Day 5 artifacts: {missing_paths}")

    train_data = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig", keep_default_na=False)
    test_data = pd.read_csv(TEST_PATH, encoding="utf-8-sig", keep_default_na=False)
    features = pd.read_csv(FEATURES_PATH, encoding="utf-8-sig", keep_default_na=False)
    day5_metrics = json.loads(DAY5_METRICS_PATH.read_text(encoding="utf-8"))
    baseline_model = joblib.load(BASELINE_MODEL_PATH)
    hybrid_model = joblib.load(HYBRID_MODEL_PATH)

    evaluation_test, excluded_ids = exclude_train_duplicates_from_test(train_data, test_data)
    test_with_features = merge_text_with_features(evaluation_test, features)

    if len(evaluation_test) != day5_metrics["data_checks"]["evaluation_test_rows"]:
        raise ValueError("Day 6 evaluation rows do not match the Day 5 report.")
    if evaluation_test["model_text"].eq("").any():
        raise ValueError("The evaluation test set contains empty model_text values.")

    baseline_model.predict_proba(evaluation_test["model_text"].iloc[:1])
    hybrid_model.predict_proba(test_with_features.iloc[:1])
    predict_news(
        evaluation_test.iloc[0]["title"],
        evaluation_test.iloc[0]["content"],
        model_path=BASELINE_MODEL_PATH,
    )

    checks = {
        "day5_outputs_available": True,
        "baseline_model_works": True,
        "hybrid_model_works": True,
        "prediction_function_works": True,
        "train_rows": int(len(train_data)),
        "original_test_rows": int(len(test_data)),
        "evaluation_test_rows": int(len(evaluation_test)),
        "exact_train_duplicates_excluded": int(len(excluded_ids)),
        "excluded_article_ids": excluded_ids,
        "linguistic_feature_rows": int(len(features)),
        "missing_linguistic_feature_rows": 0,
        "empty_model_text_rows": 0,
    }
    return train_data, test_with_features, baseline_model, checks


def build_leakage_safe_groups(dataframe: pd.DataFrame) -> np.ndarray:
    """Keep matching pair IDs and exact duplicate texts in the same CV fold."""
    parent = list(range(len(dataframe)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first_index: int, second_index: int) -> None:
        first_root = find(first_index)
        second_root = find(second_index)
        if first_root != second_root:
            parent[second_root] = first_root

    for column in ["pair_id", "model_text"]:
        first_index_by_value: dict[str, int] = {}
        for index, value in enumerate(dataframe[column].astype(str)):
            if value in first_index_by_value:
                union(index, first_index_by_value[value])
            else:
                first_index_by_value[value] = index

    roots = np.asarray([find(index) for index in range(len(dataframe))])
    group_codes, _ = pd.factorize(roots)
    return group_codes


def build_calibration_folds(dataframe: pd.DataFrame) -> tuple[list[tuple[np.ndarray, np.ndarray]], int]:
    """Create stratified, group-safe folds for out-of-fold calibration."""
    groups = build_leakage_safe_groups(dataframe)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(
        splitter.split(
            dataframe["model_text"],
            dataframe["label"],
            groups=groups,
        )
    )

    for train_index, calibration_index in folds:
        train_groups = set(groups[train_index])
        calibration_groups = set(groups[calibration_index])
        if train_groups & calibration_groups:
            raise ValueError("Calibration folds contain overlapping leakage groups.")

    return folds, int(len(np.unique(groups)))


def train_calibrated_model(train_data: pd.DataFrame) -> tuple[CalibratedClassifierCV, dict]:
    """Calibrate TF-IDF Logistic Regression with group-safe out-of-fold scores."""
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
    """Calculate equal-width expected calibration error."""
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
    """Calculate classification, probability, and confidence metrics."""
    y_array = y_true.to_numpy(dtype=int)
    predictions = (probability_fake >= 0.5).astype(int)
    wrong = predictions != y_array
    confidence = np.maximum(probability_fake, 1 - probability_fake)
    high_confidence = confidence >= 0.90

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_array,
        predictions,
        average="weighted",
        zero_division=0,
    )
    precision_fake, recall_fake, f1_fake, _ = precision_recall_fscore_support(
        y_array,
        predictions,
        average="binary",
        pos_label=1,
        zero_division=0,
    )

    return {
        "accuracy": round(float(accuracy_score(y_array, predictions)), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "precision_fake": round(float(precision_fake), 4),
        "recall_fake": round(float(recall_fake), 4),
        "f1_fake": round(float(f1_fake), 4),
        "brier_score": round(float(brier_score_loss(y_array, probability_fake)), 4),
        "log_loss": round(float(log_loss(y_array, probability_fake)), 4),
        "expected_calibration_error": round(
            float(expected_calibration_error(y_array, probability_fake)),
            4,
        ),
        "confusion_matrix": confusion_matrix(y_array, predictions, labels=[0, 1]).tolist(),
        "false_positives": int(((y_array == 0) & (predictions == 1)).sum()),
        "false_negatives": int(((y_array == 1) & (predictions == 0)).sum()),
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


def build_calibration_bins(
    y_true: pd.Series,
    probability_sets: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Build the points used in the calibration curve."""
    rows = []
    y_array = y_true.to_numpy(dtype=int)
    edges = np.linspace(0, 1, 11)

    for model_name, probabilities in probability_sets.items():
        observed_rates, mean_probabilities = calibration_curve(
            y_array,
            probabilities,
            n_bins=10,
            strategy="uniform",
        )
        bin_ids = np.digitize(probabilities, edges[1:-1], right=False)
        nonempty_bins = [bin_id for bin_id in range(10) if (bin_ids == bin_id).any()]

        for bin_id, mean_probability, observed_rate in zip(
            nonempty_bins,
            mean_probabilities,
            observed_rates,
        ):
            rows.append(
                {
                    "model": model_name,
                    "bin_lower": round(float(edges[bin_id]), 2),
                    "bin_upper": round(float(edges[bin_id + 1]), 2),
                    "sample_count": int((bin_ids == bin_id).sum()),
                    "mean_probability_fake": round(float(mean_probability), 6),
                    "observed_fake_rate": round(float(observed_rate), 6),
                    "absolute_gap": round(float(abs(observed_rate - mean_probability)), 6),
                }
            )

    return pd.DataFrame(rows)


def interpret_error(row: pd.Series, median_word_count: float) -> str:
    """Describe plausible textual signals without claiming factual causation."""
    signals = []
    sensational = str(row["sensational_words_found"]).strip()
    sources = str(row["source_markers_found"]).strip()

    if row["error_type"] == "false_negative":
        if sources:
            signals.append(f"përdor tregues burimi ({sources})")
        if not sensational and int(row["exclamation_count"]) == 0:
            signals.append("ka ton të përmbajtur pa shenja të dukshme clickbait")
        if float(row["word_count"]) >= median_word_count:
            signals.append("është relativisht i gjatë dhe mund të duket formal")
        fallback = "fjalori i tekstit fake mund t'i ngjajë artikujve real"
    else:
        if sensational:
            signals.append(f"përmban shprehje sensacionale ({sensational})")
        if int(row["exclamation_count"]) > 0:
            signals.append("përdor pikëçuditëse")
        if float(row["uppercase_ratio"]) >= 0.05:
            signals.append("ka nivel relativisht të lartë kapitalizimi")
        if float(row["word_count"]) < median_word_count:
            signals.append("është relativisht i shkurtër")
        fallback = "fjalori i tekstit real mund t'i ngjajë artikujve fake"

    explanation = "; ".join(signals) if signals else fallback
    confidence_note = (
        " Modeli gaboi me siguri të lartë."
        if float(row["predicted_confidence"]) >= 0.90
        else ""
    )
    return (
        f"Mund të jetë ngatërruar sepse {explanation}.{confidence_note} "
        "TF-IDF njeh modele fjalësh, jo vërtetësinë faktike."
    )


def build_prediction_table(
    test_data: pd.DataFrame,
    probability_fake: np.ndarray,
) -> pd.DataFrame:
    """Create one analysis row per test article."""
    predictions = (probability_fake >= 0.5).astype(int)
    y_true = test_data["label"].to_numpy(dtype=int)
    confidence = np.where(predictions == 1, probability_fake, 1 - probability_fake)
    error_types = np.where(
        (y_true == 0) & (predictions == 1),
        "false_positive",
        np.where((y_true == 1) & (predictions == 0), "false_negative", "correct"),
    )

    table = pd.DataFrame(
        {
            "article_id": test_data["article_id"].astype(str),
            "pair_id": test_data["pair_id"],
            "true_label": y_true,
            "true_label_name": test_data["label_name"].astype(str),
            "predicted_label": predictions,
            "prediction": np.where(predictions == 1, "fake", "real"),
            "probability_real": np.round(1 - probability_fake, 6),
            "probability_fake": np.round(probability_fake, 6),
            "predicted_confidence": np.round(confidence, 6),
            "error_type": error_types,
            "title": test_data["title"].astype(str),
            "content_excerpt": test_data["content"].astype(str).map(
                lambda text: " ".join(text.split())[:300]
            ),
            "word_count": test_data["word_count"],
            "source_markers_found": test_data["source_indicators_found"].astype(str),
            "sensational_words_found": test_data["sensational_found"].astype(str),
            "diacritic_ratio": test_data["diacritic_ratio"],
            "uppercase_ratio": test_data["uppercase_char_ratio"],
            "exclamation_count": test_data["exclamation_count"],
        }
    )
    table["is_high_confidence_error"] = (
        table["error_type"].ne("correct") & table["predicted_confidence"].ge(0.90)
    )
    table["is_near_50_error"] = (
        table["error_type"].ne("correct")
        & table["probability_fake"].between(0.45, 0.55, inclusive="both")
    )
    table["is_uncertain_35_65"] = table["probability_fake"].between(
        0.35,
        0.65,
        inclusive="both",
    )

    median_word_count = float(test_data["word_count"].median())
    error_mask = table["error_type"].ne("correct")
    table["interpretation"] = ""
    table.loc[error_mask, "interpretation"] = table.loc[error_mask].apply(
        lambda row: interpret_error(row, median_word_count),
        axis=1,
    )
    return table


def select_interesting_errors(predictions: pd.DataFrame) -> pd.DataFrame:
    """Select high-confidence, false-negative, false-positive, and near-50 errors."""
    errors = predictions.loc[predictions["error_type"].ne("correct")].copy()
    reasons: dict[str, list[str]] = {}

    def add_reason(article_ids: pd.Series, reason: str) -> None:
        for article_id in article_ids.astype(str):
            reasons.setdefault(article_id, []).append(reason)

    high_confidence = errors.loc[errors["is_high_confidence_error"]].sort_values(
        "predicted_confidence",
        ascending=False,
    )
    false_negatives = errors.loc[errors["error_type"].eq("false_negative")].nlargest(
        3,
        "predicted_confidence",
    )
    false_positives = errors.loc[errors["error_type"].eq("false_positive")].nlargest(
        3,
        "predicted_confidence",
    )
    near_50 = errors.assign(
        distance_from_50=(errors["probability_fake"] - 0.5).abs()
    ).nsmallest(2, "distance_from_50")

    add_reason(high_confidence["article_id"], "high_confidence_error")
    add_reason(false_negatives["article_id"], "top_false_negative")
    add_reason(false_positives["article_id"], "top_false_positive")
    add_reason(near_50["article_id"], "error_near_50")

    selected = errors.loc[errors["article_id"].astype(str).isin(reasons)].copy()
    selected.insert(
        1,
        "selection_reason",
        selected["article_id"].astype(str).map(lambda value: ", ".join(reasons[value])),
    )
    return selected.sort_values(
        ["is_high_confidence_error", "predicted_confidence"],
        ascending=[False, False],
    ).reset_index(drop=True)


def evaluate_thresholds(
    y_true: pd.Series,
    probability_fake: np.ndarray,
) -> pd.DataFrame:
    """Compare three uncertain-zone variants for the future app."""
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


def build_probability_summary(metrics: dict[str, dict]) -> pd.DataFrame:
    """Convert nested metrics into a compact comparison table."""
    rows = []
    for model_name, model_metrics in metrics.items():
        matrix = model_metrics["confusion_matrix"]
        row = {
            "model": model_name,
            **{
                key: value
                for key, value in model_metrics.items()
                if key != "confusion_matrix"
            },
            "true_real_pred_real": matrix[0][0],
            "true_real_pred_fake": matrix[0][1],
            "true_fake_pred_real": matrix[1][0],
            "true_fake_pred_fake": matrix[1][1],
        }
        rows.append(row)
    return pd.DataFrame(rows)


def plot_probability_analysis(
    calibration_bins: pd.DataFrame,
    probability_sets: dict[str, np.ndarray],
) -> None:
    """Save the calibration curve and probability distributions."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#555555", label="Perfect")
    colors = {"uncalibrated": "#287271", "calibrated_sigmoid": "#E76F51"}
    for model_name, model_bins in calibration_bins.groupby("model", sort=False):
        axes[0].plot(
            model_bins["mean_probability_fake"],
            model_bins["observed_fake_rate"],
            marker="o",
            linewidth=2,
            color=colors[model_name],
            label=model_name,
        )
    axes[0].set_xlabel("Mean predicted fake probability")
    axes[0].set_ylabel("Observed fake rate")
    axes[0].set_title("Calibration curve")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    bins = np.linspace(0, 1, 21)
    for model_name, probabilities in probability_sets.items():
        axes[1].hist(
            probabilities,
            bins=bins,
            alpha=0.58,
            color=colors[model_name],
            label=model_name,
        )
    axes[1].axvspan(0.30, 0.70, color="#F4A261", alpha=0.12, label="uncertain 30-70")
    axes[1].set_xlabel("Predicted fake probability")
    axes[1].set_ylabel("Article count")
    axes[1].set_title("Probability distribution")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(CALIBRATION_FIGURE_PATH, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _fake_probabilities(model, model_text: pd.Series) -> np.ndarray:
    classes = list(model.classes_)
    fake_index = classes.index(1)
    return model.predict_proba(model_text)[:, fake_index]


def _relative_path(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def run_model_quality_analysis() -> dict:
    """Run the complete Day 6 quality analysis and save its artifacts."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
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

    prediction_table = build_prediction_table(test_data, baseline_probability)
    errors = prediction_table.loc[prediction_table["error_type"].ne("correct")].copy()
    interesting_errors = select_interesting_errors(prediction_table)
    thresholds = evaluate_thresholds(test_data["label"], calibrated_probability)
    calibration_bins = build_calibration_bins(test_data["label"], probability_sets)
    probability_summary = build_probability_summary(probability_metrics)

    errors.to_csv(ERRORS_PATH, index=False, encoding="utf-8-sig")
    interesting_errors.to_csv(INTERESTING_ERRORS_PATH, index=False, encoding="utf-8-sig")
    thresholds.to_csv(THRESHOLD_PATH, index=False, encoding="utf-8-sig")
    calibration_bins.to_csv(CALIBRATION_BINS_PATH, index=False, encoding="utf-8-sig")
    probability_summary.to_csv(PROBABILITY_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    plot_probability_analysis(calibration_bins, probability_sets)

    eligible_thresholds = thresholds.loc[thresholds["strong_decision_coverage"].ge(0.80)]
    recommended_threshold = eligible_thresholds.sort_values(
        ["strong_decision_accuracy", "strong_decision_coverage"],
        ascending=False,
    ).iloc[0]

    sample_row = test_data.sort_values(
        ["sensational_count", "source_indicator_count"],
        ascending=False,
    ).iloc[0]
    app_sample = predict_news_for_app(
        sample_row["title"],
        sample_row["content"],
        model_path=CALIBRATED_MODEL_PATH,
        real_threshold=float(recommended_threshold["real_threshold"]),
        fake_threshold=float(recommended_threshold["fake_threshold"]),
    )

    before = probability_metrics["uncalibrated"]
    after = probability_metrics["calibrated_sigmoid"]
    result = {
        "data_checks": checks,
        "baseline_error_analysis": {
            "total_errors": int(len(errors)),
            "false_positives": before["false_positives"],
            "false_negatives": before["false_negatives"],
            "high_confidence_errors_90": before["high_confidence_errors_90"],
            "near_50_errors_45_55": before["near_50_errors_45_55"],
            "uncertain_errors_35_65": before["uncertain_errors_35_65"],
            "interesting_error_count": int(len(interesting_errors)),
            "interesting_article_ids": interesting_errors["article_id"].astype(str).tolist(),
        },
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
        "app_output_sample": {
            "article_id": str(sample_row["article_id"]),
            "true_label": str(sample_row["label_name"]),
            "title": str(sample_row["title"]),
            "output": app_sample,
        },
        "artifacts": {
            "calibrated_model": _relative_path(CALIBRATED_MODEL_PATH),
            "errors": _relative_path(ERRORS_PATH),
            "interesting_errors": _relative_path(INTERESTING_ERRORS_PATH),
            "probability_summary": _relative_path(PROBABILITY_SUMMARY_PATH),
            "threshold_comparison": _relative_path(THRESHOLD_PATH),
            "calibration_bins": _relative_path(CALIBRATION_BINS_PATH),
            "calibration_figure": _relative_path(CALIBRATION_FIGURE_PATH),
            "report": _relative_path(REPORT_PATH),
        },
    }
    METRICS_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    result = run_model_quality_analysis()
    before = result["probability_metrics"]["uncalibrated"]
    after = result["probability_metrics"]["calibrated_sigmoid"]

    print("=== Day 6 model quality ===")
    print(f"Evaluation rows: {result['data_checks']['evaluation_test_rows']}")
    print(f"False positives: {before['false_positives']}")
    print(f"False negatives: {before['false_negatives']}")
    print(f"Brier before/after: {before['brier_score']} / {after['brier_score']}")
    print(f"Accuracy before/after: {before['accuracy']} / {after['accuracy']}")
    print(f"Recommended thresholds: {result['recommendation']['threshold_variant']}")
    print(f"Metrics saved: {METRICS_PATH}")


if __name__ == "__main__":
    main()
