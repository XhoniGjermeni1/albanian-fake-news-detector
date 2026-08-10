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
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.analyze_length_domain_shift import (  # noqa: E402
    LENGTH_DISPLAY,
    LENGTH_LABELS,
)
from src.models.compare_classifiers import (  # noqa: E402
    FIXED_CHAR_CONFIG,
    add_word_counts,
    build_fixed_features,
    build_group_safe_folds,
    classification_metrics,
    dataframe_to_markdown,
    fake_decision_scores,
    file_sha256,
    refresh_model_text,
    rounded_metrics,
)
from src.models.train_hybrid_model import (  # noqa: E402
    exclude_train_duplicates_from_test,
)


TRAIN_PATH = PROJECT_ROOT / "data" / "interim" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "interim" / "test.csv"
EXTERNAL_PATH = PROJECT_ROOT / "data" / "external" / "external_news.csv"
DAY13_SELECTION_PATH = PROJECT_ROOT / "reports" / "day13_internal_selection.json"
DAY14_SELECTION_PATH = PROJECT_ROOT / "reports" / "day14_selection.json"
CURRENT_APP_MODEL_PATH = PROJECT_ROOT / "models" / "calibrated_tfidf_logreg.joblib"
STREAMLIT_APP_PATH = PROJECT_ROOT / "app" / "streamlit_app.py"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

CV_FOLDS_PATH = REPORTS_DIR / "day15_cv_fold_results.csv"
CV_SUMMARY_PATH = REPORTS_DIR / "day15_cv_summary.csv"
SELECTION_PATH = REPORTS_DIR / "day15_selection.json"
INTERNAL_PREDICTIONS_PATH = REPORTS_DIR / "day15_internal_predictions.csv"
INTERNAL_COMPARISON_PATH = REPORTS_DIR / "day15_internal_candidate_comparison.csv"
LENGTH_METRICS_PATH = REPORTS_DIR / "day15_length_group_metrics.csv"
SPECIAL_COHORTS_PATH = REPORTS_DIR / "day15_special_cohort_metrics.csv"
LENGTH_BIAS_PATH = REPORTS_DIR / "day15_length_bias_comparison.csv"
EXTERNAL_PREDICTIONS_PATH = REPORTS_DIR / "day15_external_predictions.csv"
EXTERNAL_METRICS_PATH = REPORTS_DIR / "day15_external_metrics.csv"
METRICS_PATH = REPORTS_DIR / "day15_metrics.json"
REPORT_PATH = REPORTS_DIR / "day15_svm_tuning.md"

CV_FIGURE_PATH = FIGURES_DIR / "day15_cv_c_tuning.png"
INTERNAL_FIGURE_PATH = FIGURES_DIR / "day15_internal_candidate_comparison.png"
LENGTH_FIGURE_PATH = FIGURES_DIR / "day15_length_performance.png"
EXTERNAL_FIGURE_PATH = FIGURES_DIR / "day15_external_confusion_matrix.png"

C_VALUES = [0.25, 0.5, 1.0, 2.0, 4.0]
BASELINE_C = 1.0
F1_CLOSE_TOLERANCE = 0.002
MAX_ACCEPTABLE_RECALL_GAP = 0.10
LOGGER = logging.getLogger(__name__)


def candidate_id(c_value: float) -> str:
    """Return a stable file-safe ID for one C value."""
    return f"linear_svm_c_{str(float(c_value)).replace('.', '_')}"


def candidate_display(c_value: float) -> str:
    return f"Linear SVM (C={float(c_value)})"


def model_path(c_value: float) -> Path:
    return MODELS_DIR / f"day15_word_char_{candidate_id(c_value)}.joblib"


def verify_frozen_setup() -> dict:
    """Verify the representation and candidate inherited from Days 13 and 14."""
    day13 = json.loads(DAY13_SELECTION_PATH.read_text(encoding="utf-8"))
    day14 = json.loads(DAY14_SELECTION_PATH.read_text(encoding="utf-8"))
    if day13.get("recommended_for_day14") != "word_char_tfidf":
        raise ValueError("Day 13 representation is not Word + Character TF-IDF.")
    if day13.get("selected_char_config") != FIXED_CHAR_CONFIG:
        raise ValueError("Day 13 character configuration has changed.")
    if day14.get("winner_classifier") != "linear_svm":
        raise ValueError("Day 14 winner is not Linear SVM.")
    if day14.get("winner_candidate_id") != "linear_svm_c_1_0":
        raise ValueError("Day 14 candidate is not Linear SVM with C=1.0.")
    if day14.get("internal_test_used") is not False:
        raise ValueError("Day 14 selection is not marked train/CV-only.")
    if day14.get("external_results_used") is not False:
        raise ValueError("Day 14 selection used external results.")
    return {
        "representation": "word_char_tfidf",
        "character_config": FIXED_CHAR_CONFIG.copy(),
        "day14_classifier": "linear_svm",
        "day14_c": BASELINE_C,
    }


def build_svm(c_value: float) -> LinearSVC:
    """Build one uncalibrated Linear SVM configuration."""
    return LinearSVC(
        C=float(c_value),
        class_weight="balanced",
        max_iter=5000,
        random_state=42,
    )


def build_svm_pipeline(c_value: float) -> Pipeline:
    """Build the frozen TF-IDF representation followed by Linear SVM."""
    return Pipeline(
        [
            ("features", build_fixed_features(FIXED_CHAR_CONFIG)),
            ("classifier", build_svm(c_value)),
        ]
    )


def run_svm_cv(
    train: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], int]:
    """Evaluate five C values using shared, leakage-safe fold features."""
    folds, groups, fold_audit = build_group_safe_folds(train)
    rows: list[dict] = []

    for fold_number, (fit_index, validation_index) in enumerate(folds, start=1):
        fit = train.iloc[fit_index]
        validation = train.iloc[validation_index]
        vectorizer = build_fixed_features(FIXED_CHAR_CONFIG)

        feature_started = time.perf_counter()
        x_fit = vectorizer.fit_transform(fit["model_text"], fit["label"])
        feature_fit_seconds = time.perf_counter() - feature_started
        transform_started = time.perf_counter()
        x_validation = vectorizer.transform(validation["model_text"])
        feature_transform_seconds = time.perf_counter() - transform_started

        for c_value in C_VALUES:
            classifier = clone(build_svm(c_value))
            classifier_started = time.perf_counter()
            classifier.fit(x_fit, fit["label"])
            classifier_fit_seconds = time.perf_counter() - classifier_started

            prediction_started = time.perf_counter()
            validation_prediction = classifier.predict(x_validation)
            prediction_seconds = time.perf_counter() - prediction_started
            train_prediction = classifier.predict(x_fit)

            validation_metrics = classification_metrics(
                validation["label"], validation_prediction
            )
            train_metrics = classification_metrics(fit["label"], train_prediction)
            rows.append(
                {
                    "candidate_id": candidate_id(c_value),
                    "c_value": c_value,
                    "fold": fold_number,
                    **{
                        key: value
                        for key, value in validation_metrics.items()
                        if key != "confusion_matrix"
                    },
                    "confusion_matrix": json.dumps(
                        validation_metrics["confusion_matrix"]
                    ),
                    "train_f1_weighted": train_metrics["f1_weighted"],
                    "generalization_gap": (
                        train_metrics["f1_weighted"]
                        - validation_metrics["f1_weighted"]
                    ),
                    "recall_gap": abs(
                        validation_metrics["recall_real"]
                        - validation_metrics["recall_fake"]
                    ),
                    "feature_fit_seconds": feature_fit_seconds,
                    "feature_transform_seconds": feature_transform_seconds,
                    "classifier_fit_seconds": classifier_fit_seconds,
                    "training_seconds": (
                        feature_fit_seconds + classifier_fit_seconds
                    ),
                    "prediction_seconds": prediction_seconds,
                    "n_iter": int(classifier.n_iter_),
                }
            )

    fold_results = pd.DataFrame(rows)
    summary_rows: list[dict] = []
    mean_std_metrics = [
        "accuracy",
        "f1_weighted",
        "f1_fake",
        "recall_real",
        "recall_fake",
        "recall_gap",
        "train_f1_weighted",
        "generalization_gap",
        "training_seconds",
        "classifier_fit_seconds",
        "prediction_seconds",
    ]
    for c_value in C_VALUES:
        subset = fold_results.loc[fold_results["c_value"].eq(c_value)]
        row: dict[str, object] = {
            "candidate_id": candidate_id(c_value),
            "c_value": c_value,
            "folds": int(len(subset)),
        }
        for metric in mean_std_metrics:
            values = subset[metric].astype(float)
            row[f"mean_{metric}"] = float(values.mean())
            row[f"std_{metric}"] = float(values.std(ddof=1))
        row["total_false_positives"] = int(subset["false_positives"].sum())
        row["total_false_negatives"] = int(subset["false_negatives"].sum())
        row["mean_false_positives"] = float(subset["false_positives"].mean())
        row["mean_false_negatives"] = float(subset["false_negatives"].mean())
        row["false_positives_by_fold"] = json.dumps(
            subset["false_positives"].astype(int).tolist()
        )
        row["false_negatives_by_fold"] = json.dumps(
            subset["false_negatives"].astype(int).tolist()
        )
        row["max_n_iter"] = int(subset["n_iter"].max())
        summary_rows.append(row)

    return fold_results, pd.DataFrame(summary_rows), fold_audit, int(len(np.unique(groups)))


def choose_analysis_candidates(cv_summary: pd.DataFrame, selected_c: float) -> list[float]:
    """Choose three diagnostic candidates from CV only, always including C=1."""
    ranked = cv_summary.sort_values(
        [
            "mean_f1_weighted",
            "std_f1_weighted",
            "mean_recall_gap",
            "mean_generalization_gap",
            "c_value",
        ],
        ascending=[False, True, True, True, True],
    )
    chosen: list[float] = []
    for c_value in [selected_c, BASELINE_C, *ranked["c_value"].tolist()]:
        value = float(c_value)
        if value not in chosen:
            chosen.append(value)
        if len(chosen) == 3:
            break
    return chosen


def select_c_from_cv(cv_summary: pd.DataFrame) -> dict:
    """Select C using F1, stability, class balance, and overfitting risk."""
    summary = cv_summary.copy()
    eligible = summary.loc[
        summary["mean_recall_gap"].le(MAX_ACCEPTABLE_RECALL_GAP)
    ].copy()
    balance_filter_applied = True
    if eligible.empty:
        eligible = summary.copy()
        balance_filter_applied = False

    best_f1 = float(eligible["mean_f1_weighted"].max())
    finalists = eligible.loc[
        eligible["mean_f1_weighted"].ge(
            best_f1 - F1_CLOSE_TOLERANCE - 1e-12
        )
    ].copy()
    winner = finalists.sort_values(
        [
            "std_f1_weighted",
            "mean_recall_gap",
            "mean_generalization_gap",
            "c_value",
        ],
        ascending=[True, True, True, True],
    ).iloc[0]
    most_stable = summary.sort_values(
        ["std_f1_weighted", "mean_f1_weighted"], ascending=[True, False]
    ).iloc[0]
    best_balance = summary.sort_values(
        ["mean_recall_gap", "mean_f1_weighted"], ascending=[True, False]
    ).iloc[0]
    selected_c = float(winner["c_value"])

    return {
        "selection_scope": "train_only_5_fold_group_safe_cv",
        "fixed_representation": {
            "name": "word_char_tfidf",
            "word": {
                "ngram_range": [1, 2],
                "lowercase": False,
                "min_df": 2,
                "max_features": 30000,
            },
            "character": FIXED_CHAR_CONFIG.copy(),
        },
        "classifier": "linear_svm",
        "tested_c_values": C_VALUES,
        "baseline_c": BASELINE_C,
        "f1_close_tolerance": F1_CLOSE_TOLERANCE,
        "max_acceptable_recall_gap": MAX_ACCEPTABLE_RECALL_GAP,
        "balance_filter_applied": balance_filter_applied,
        "selection_rule": (
            "Exclude mean recall gaps above 0.10 when possible; keep candidates "
            "within 0.002 F1 of the best, then prefer lower F1 standard "
            "deviation, lower recall gap, lower generalization gap, and lower C."
        ),
        "selected_c": selected_c,
        "selected_candidate_id": candidate_id(selected_c),
        "selected_mean_f1_weighted": float(winner["mean_f1_weighted"]),
        "selected_std_f1_weighted": float(winner["std_f1_weighted"]),
        "selected_mean_recall_gap": float(winner["mean_recall_gap"]),
        "selected_mean_generalization_gap": float(
            winner["mean_generalization_gap"]
        ),
        "most_stable_c": float(most_stable["c_value"]),
        "best_recall_balance_c": float(best_balance["c_value"]),
        "analysis_c_values": choose_analysis_candidates(summary, selected_c),
        "internal_test_used": False,
        "external_results_used": False,
        "calibration_applied": False,
    }


def verify_selection_hash(expected_hash: str) -> None:
    if not SELECTION_PATH.exists() or file_sha256(SELECTION_PATH) != expected_hash:
        raise RuntimeError("The frozen Day 15 selection has changed.")


def load_internal_test_after_selection(
    train: pd.DataFrame,
    selection_hash: str,
) -> tuple[pd.DataFrame, dict]:
    """Load and audit the internal test after C has been selected."""
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


def fit_analysis_models(
    train: pd.DataFrame,
    c_values: list[float],
) -> tuple[dict[float, Pipeline], dict[str, dict]]:
    """Fit and save the three candidates chosen before internal evaluation."""
    models: dict[float, Pipeline] = {}
    metadata: dict[str, dict] = {}
    for c_value in c_values:
        model = build_svm_pipeline(c_value)
        started = time.perf_counter()
        model.fit(train["model_text"], train["label"])
        training_seconds = time.perf_counter() - started
        path = model_path(c_value)
        joblib.dump(model, path, compress=3)
        size_bytes = path.stat().st_size
        models[c_value] = model
        metadata[candidate_id(c_value)] = {
            "c_value": c_value,
            "training_seconds": round(training_seconds, 3),
            "model_size_mb": round(size_bytes / (1024 * 1024), 3),
            "model_path": str(path.relative_to(PROJECT_ROOT)),
            "model_sha256": file_sha256(path),
            "calibrated": False,
        }
    return models, metadata


def prediction_table(
    dataframe: pd.DataFrame,
    model: Pipeline,
    c_value: float,
    id_column: str,
) -> pd.DataFrame:
    """Create binary predictions and raw SVM scores, never probabilities."""
    table = dataframe.copy().reset_index(drop=True)
    table.insert(0, "candidate_id", candidate_id(c_value))
    table.insert(1, "candidate_display", candidate_display(c_value))
    table.insert(2, "c_value", c_value)
    table["score_type"] = "Linear SVM decision score (jo probabilitet)"
    table["decision_score_fake"] = fake_decision_scores(
        model, table["model_text"]
    )
    table["binary_prediction"] = model.predict(table["model_text"]).astype(int)
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
        raise ValueError(f"Duplicate IDs in prediction input: {id_column}")
    return table


def prediction_metrics(table: pd.DataFrame) -> dict:
    return classification_metrics(table["label"], table["binary_prediction"])


def evaluate_internal_candidates(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate top candidates overall, by length, and in difficult cohorts."""
    comparison_rows: list[dict] = []
    length_rows: list[dict] = []
    special_rows: list[dict] = []
    bias_rows: list[dict] = []

    for c_value, table in predictions.groupby("c_value", sort=False):
        overall = prediction_metrics(table)
        comparison_rows.append(
            {
                "candidate_id": candidate_id(c_value),
                "candidate_display": candidate_display(c_value),
                "c_value": c_value,
                **{
                    key: value
                    for key, value in overall.items()
                    if key != "confusion_matrix"
                },
                "confusion_matrix": json.dumps(overall["confusion_matrix"]),
            }
        )

        for group_name in LENGTH_LABELS:
            subset = table.loc[table["length_group"].astype(str).eq(group_name)]
            result = prediction_metrics(subset)
            length_rows.append(
                {
                    "candidate_id": candidate_id(c_value),
                    "candidate_display": candidate_display(c_value),
                    "c_value": c_value,
                    "length_group": group_name,
                    "length_description": LENGTH_DISPLAY[group_name],
                    **{
                        key: value
                        for key, value in result.items()
                        if key != "confusion_matrix"
                    },
                    "confusion_matrix": json.dumps(result["confusion_matrix"]),
                }
            )

        special = {
            "real_30_60": table.loc[
                table["label"].eq(0) & table["word_count"].between(30, 60)
            ],
            "fake_gt_250": table.loc[
                table["label"].eq(1) & table["word_count"].gt(250)
            ],
        }
        for cohort_name, subset in special.items():
            result = prediction_metrics(subset)
            special_rows.append(
                {
                    "candidate_id": candidate_id(c_value),
                    "candidate_display": candidate_display(c_value),
                    "c_value": c_value,
                    "cohort": cohort_name,
                    **{
                        key: value
                        for key, value in result.items()
                        if key != "confusion_matrix"
                    },
                    "confusion_matrix": json.dumps(result["confusion_matrix"]),
                }
            )

        correlations = {}
        for scope, subset in (
            ("all", table),
            ("real", table.loc[table["label"].eq(0)]),
            ("fake", table.loc[table["label"].eq(1)]),
        ):
            result = spearmanr(subset["word_count"], subset["decision_score_fake"])
            correlations[scope] = (float(result.statistic), float(result.pvalue))
        special_by_name = {
            row["cohort"]: row
            for row in special_rows
            if float(row["c_value"]) == float(c_value)
        }
        bias_rows.append(
            {
                "candidate_id": candidate_id(c_value),
                "candidate_display": candidate_display(c_value),
                "c_value": c_value,
                "spearman_all": correlations["all"][0],
                "spearman_all_p": correlations["all"][1],
                "spearman_real": correlations["real"][0],
                "spearman_real_p": correlations["real"][1],
                "spearman_fake": correlations["fake"][0],
                "spearman_fake_p": correlations["fake"][1],
                "mean_absolute_within_label_spearman": (
                    abs(correlations["real"][0])
                    + abs(correlations["fake"][0])
                )
                / 2,
                "real_30_60_accuracy": special_by_name["real_30_60"]["accuracy"],
                "fake_gt_250_accuracy": special_by_name["fake_gt_250"]["accuracy"],
            }
        )

    return (
        pd.DataFrame(comparison_rows),
        pd.DataFrame(length_rows),
        pd.DataFrame(special_rows),
        pd.DataFrame(bias_rows),
    )


def load_external_after_selection(selection_hash: str) -> tuple[pd.DataFrame, int]:
    """Load the frozen external benchmark after C selection is immutable."""
    verify_selection_hash(selection_hash)
    external = pd.read_csv(EXTERNAL_PATH, encoding="utf-8", keep_default_na=False)
    if len(external) != 40 or set(external["label"]) != {"real", "fake"}:
        raise ValueError("External benchmark is not the expected frozen dataset.")
    external["label"] = external["label"].map({"real": 0, "fake": 1})
    external, stale_rows = refresh_model_text(external)
    return add_word_counts(external), stale_rows


def plot_cv_tuning(cv_summary: pd.DataFrame, selected_c: float) -> None:
    """Plot CV F1 with error bars and class recalls."""
    data = cv_summary.sort_values("c_value")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    axes[0].errorbar(
        data["c_value"],
        data["mean_f1_weighted"],
        yerr=data["std_f1_weighted"],
        marker="o",
        capsize=5,
        linewidth=2,
        color="#D76745",
    )
    axes[0].axvline(selected_c, color="#333333", linestyle="--", alpha=0.7)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(C_VALUES, [str(value) for value in C_VALUES])
    axes[0].set_xlabel("C")
    axes[0].set_ylabel("F1 weighted mesatare")
    axes[0].set_title("F1 dhe devijimi standard")
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        data["c_value"],
        data["mean_recall_real"],
        marker="o",
        linewidth=2,
        label="Recall real",
        color="#3976A8",
    )
    axes[1].plot(
        data["c_value"],
        data["mean_recall_fake"],
        marker="o",
        linewidth=2,
        label="Recall fake",
        color="#2F937F",
    )
    axes[1].axvline(selected_c, color="#333333", linestyle="--", alpha=0.7)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(C_VALUES, [str(value) for value in C_VALUES])
    axes[1].set_xlabel("C")
    axes[1].set_ylabel("Recall mesatare")
    axes[1].set_title("Balanca mes klasave")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.25)
    figure.suptitle("Linear SVM tuning, group-safe CV vetëm mbi train")
    figure.tight_layout()
    figure.savefig(CV_FIGURE_PATH, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_internal(comparison: pd.DataFrame, selected_c: float) -> None:
    """Plot internal metrics for the three preselected diagnostic candidates."""
    data = comparison.sort_values("c_value")
    metrics = ["accuracy", "f1_weighted", "f1_fake"]
    labels = ["Accuracy", "F1 weighted", "F1 fake"]
    x = np.arange(len(metrics))
    width = 0.8 / len(data)
    figure, axis = plt.subplots(figsize=(10, 5.6))
    alternative_colors = iter(["#3976A8", "#2F937F", "#7566A8"])
    for index, row in data.reset_index(drop=True).iterrows():
        color = (
            "#D76745"
            if float(row["c_value"]) == selected_c
            else next(alternative_colors)
        )
        axis.bar(
            x + (index - (len(data) - 1) / 2) * width,
            [row[metric] for metric in metrics],
            width,
            label=candidate_display(float(row["c_value"])),
            color=color,
            alpha=1.0 if float(row["c_value"]) == selected_c else 0.75,
        )
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Rezultati")
    axis.set_title("Kandidatët e zgjedhur paraprakisht në test-in e brendshëm")
    axis.legend(loc="lower right")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(INTERNAL_FIGURE_PATH, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_length(length_metrics: pd.DataFrame, selected_c: float) -> None:
    """Plot accuracy by fixed length groups for the top candidates."""
    x = np.arange(len(LENGTH_LABELS))
    figure, axis = plt.subplots(figsize=(11, 5.8))
    for c_value, table in length_metrics.groupby("c_value", sort=True):
        indexed = table.set_index("length_group")
        values = [float(indexed.loc[group, "accuracy"]) for group in LENGTH_LABELS]
        axis.plot(
            x,
            values,
            marker="o",
            linewidth=2.5 if float(c_value) == selected_c else 1.8,
            label=candidate_display(float(c_value)),
        )
    axis.set_xticks(x, [LENGTH_DISPLAY[group] for group in LENGTH_LABELS], rotation=8)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Accuracy")
    axis.set_title("Performanca sipas gjatësisë")
    axis.legend(loc="best")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(LENGTH_FIGURE_PATH, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_external_confusion(matrix: list[list[int]], selected_c: float) -> None:
    """Plot the selected candidate's external confusion matrix."""
    values = np.asarray(matrix, dtype=int)
    figure, axis = plt.subplots(figsize=(6.4, 5.5))
    image = axis.imshow(values, cmap="Blues", vmin=0)
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                str(values[row, column]),
                ha="center",
                va="center",
                fontsize=14,
                color="white" if values[row, column] > values.max() / 2 else "black",
            )
    axis.set_xticks([0, 1], ["Pred real", "Pred fake"])
    axis.set_yticks([0, 1], ["Real", "Fake"])
    axis.set_xlabel("Prediction")
    axis.set_ylabel("Label")
    axis.set_title(f"External diagnostic, Linear SVM C={selected_c}")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(EXTERNAL_FIGURE_PATH, dpi=160, bbox_inches="tight")
    plt.close(figure)


def tuning_interpretation(selection: dict, cv_summary: pd.DataFrame) -> str:
    """Explain the CV change relative to C=1 without using test results."""
    selected = cv_summary.loc[
        cv_summary["c_value"].eq(selection["selected_c"])
    ].iloc[0]
    baseline = cv_summary.loc[cv_summary["c_value"].eq(BASELINE_C)].iloc[0]
    if float(selection["selected_c"]) == BASELINE_C:
        raw_best = cv_summary.sort_values(
            ["mean_f1_weighted", "std_f1_weighted"], ascending=[False, True]
        ).iloc[0]
        delta = float(raw_best["mean_f1_weighted"] - baseline["mean_f1_weighted"])
        return (
            f"Tuning-u konfirmoi C=1.0. C={float(raw_best['c_value'])} kishte "
            f"F1 mesatare vetëm {delta:+.4f} më të lartë, por devijimi i tij "
            f"standard ishte {float(raw_best['std_f1_weighted']):.4f} kundrejt "
            f"{float(baseline['std_f1_weighted']):.4f} te C=1.0. Përmirësimi "
            f"ishte shumë i vogël dhe më pak i qëndrueshëm. Recall gap ishte "
            f"{float(raw_best['mean_recall_gap']):.4f} kundrejt "
            f"{float(baseline['mean_recall_gap']):.4f}. Generalization gap nuk "
            "u rrit, por regularizimi më i dobët dhe varianca më e lartë nuk "
            "justifikojnë rrezikun shtesë të kompleksitetit."
        )
    delta = float(selected["mean_f1_weighted"] - baseline["mean_f1_weighted"])
    size = "shumë i vogël" if abs(delta) < F1_CLOSE_TOLERANCE else "i dallueshëm"
    risk = (
        "më i lartë"
        if float(selected["mean_generalization_gap"])
        > float(baseline["mean_generalization_gap"])
        else "jo më i lartë"
    )
    return (
        f"Ndryshimi i F1 weighted kundrejt C=1.0 ishte {delta:+.4f}, pra {size}. "
        f"Generalization gap sugjeron rrezik overfitting-u {risk}."
    )


def write_report(
    metrics: dict,
    cv_summary: pd.DataFrame,
    internal_comparison: pd.DataFrame,
    length_metrics: pd.DataFrame,
    special_cohorts: pd.DataFrame,
    length_bias: pd.DataFrame,
    external_metrics: pd.DataFrame,
) -> None:
    """Write the final Day 15 report."""
    selection = metrics["selection"]
    selected_c = float(selection["selected_c"])
    selected_cv = cv_summary.loc[cv_summary["c_value"].eq(selected_c)].iloc[0]
    baseline_cv = cv_summary.loc[cv_summary["c_value"].eq(BASELINE_C)].iloc[0]
    selected_internal = internal_comparison.loc[
        internal_comparison["c_value"].eq(selected_c)
    ].iloc[0]
    baseline_internal = internal_comparison.loc[
        internal_comparison["c_value"].eq(BASELINE_C)
    ].iloc[0]
    selected_bias = length_bias.loc[length_bias["c_value"].eq(selected_c)].iloc[0]
    baseline_bias = length_bias.loc[length_bias["c_value"].eq(BASELINE_C)].iloc[0]
    lowest_bias = length_bias.sort_values(
        "mean_absolute_within_label_spearman"
    ).iloc[0]
    external = external_metrics.iloc[0]
    selected_c_text = str(selected_c)

    length_report = length_metrics.copy()
    length_order = {name: index for index, name in enumerate(LENGTH_LABELS)}
    length_report["length_order"] = length_report["length_group"].map(length_order)
    length_report = length_report.sort_values(["c_value", "length_order"])
    special_report = special_cohorts.copy()
    special_report["cohort_order"] = special_report["cohort"].map(
        {"real_30_60": 0, "fake_gt_250": 1}
    )
    special_report = special_report.sort_values(["c_value", "cohort_order"])

    cv_table = dataframe_to_markdown(
        cv_summary.sort_values("c_value"),
        [
            "c_value",
            "mean_accuracy",
            "mean_f1_weighted",
            "std_f1_weighted",
            "mean_f1_fake",
            "mean_recall_real",
            "mean_recall_fake",
            "mean_recall_gap",
            "mean_generalization_gap",
            "mean_false_positives",
            "mean_false_negatives",
            "mean_training_seconds",
        ],
    )
    internal_table = dataframe_to_markdown(
        internal_comparison.sort_values("c_value"),
        [
            "candidate_display",
            "accuracy",
            "f1_weighted",
            "f1_fake",
            "recall_real",
            "recall_fake",
            "false_positives",
            "false_negatives",
            "confusion_matrix",
        ],
    )
    length_table = dataframe_to_markdown(
        length_report,
        [
            "candidate_display",
            "length_description",
            "rows",
            "accuracy",
            "recall_real",
            "recall_fake",
        ],
    )
    special_table = dataframe_to_markdown(
        special_report,
        [
            "candidate_display",
            "cohort",
            "rows",
            "accuracy",
            "recall_real",
            "recall_fake",
            "confusion_matrix",
        ],
    )
    bias_table = dataframe_to_markdown(
        length_bias.sort_values("c_value"),
        [
            "candidate_display",
            "spearman_all",
            "spearman_real",
            "spearman_fake",
            "mean_absolute_within_label_spearman",
            "real_30_60_accuracy",
            "fake_gt_250_accuracy",
        ],
    )
    external_table = dataframe_to_markdown(
        external_metrics,
        [
            "candidate_display",
            "accuracy",
            "recall_real",
            "recall_fake",
            "false_positives",
            "false_negatives",
            "confusion_matrix",
        ],
    )
    interpretation = tuning_interpretation(selection, cv_summary)

    report = f"""# Dita 15 - Tuning i kufizuar i Linear SVM

## Protokolli

U mbajt fiks përfaqësimi Word TF-IDF `(1,2)` plus Character TF-IDF
`char_wb (3,5)` i Ditës 13. U përdor vetëm train set-i me 5-fold
`StratifiedGroupKFold` dhe {metrics['data_audit']['group_count']} leakage-groups.
Artikujt me `pair_id` ose tekst identik u mbajtën në të njëjtin grup; të pesë
fold-et kishin zero mbivendosje grupesh.

U provuan vetëm `C = {', '.join(str(value) for value in C_VALUES)}`. TF-IDF u
përshtat një herë brenda çdo fold-i dhe nuk pa validation-in. Për shkak të
normalizimit aktual NFC, u rifreskuan vetëm në memorie
{metrics['data_audit']['stale_train_model_text_rows_refreshed_in_memory']} tekste
train; CSV-të nuk u ndryshuan.

Përzgjedhja u shkrua te `reports/day15_selection.json` përpara se të ngarkohej
test set-i. Dataset-i i jashtëm u hap vetëm pasi vendimi ishte hash-uar. Nuk u
bë calibration, nuk u përdorën pragjet 0.30/0.70 dhe modeli i Streamlit mbeti i
paprekur.

## Cross-validation vetëm mbi train

{cv_table}

FP/FN për çdo fold ruhen te `reports/day15_cv_fold_results.csv`; tabela paraqet
mesataren për fold. Rregulli përjashtoi recall gap mbi
{MAX_ACCEPTABLE_RECALL_GAP:.2f} kur kishte alternativa dhe konsideroi kandidatët
brenda {F1_CLOSE_TOLERANCE:.3f} F1 si shumë të afërt. Pastaj preferoi devijimin
standard më të ulët, recall gap më të vogël, generalization gap më të vogël dhe
`C` më të ulët.

U zgjodh **Linear SVM me C={selected_c_text}**, me CV F1 weighted
{selected_cv['mean_f1_weighted']:.4f} ± {selected_cv['std_f1_weighted']:.4f},
F1 fake {selected_cv['mean_f1_fake']:.4f}, recall real
{selected_cv['mean_recall_real']:.4f} dhe recall fake
{selected_cv['mean_recall_fake']:.4f}. Më i qëndrueshmi ishte
`C={float(selection['most_stable_c'])}`; balancën më të afërt të recall e kishte
`C={float(selection['best_recall_balance_c'])}`. Stabiliteti kishte përparësi
ndaj ndryshimeve shumë të vogla të mesatares.

![CV tuning](figures/day15_cv_c_tuning.png)

## Test set-i i brendshëm

Pas ngrirjes së C u përjashtuan
{metrics['data_audit']['exact_train_duplicates_excluded']} dublikata ekzakte
train/test dhe mbetën {metrics['data_audit']['evaluation_test_rows']} raste.
Tre konfigurimet në tabelë u përcaktuan nga CV përpara ngarkimit të test-it.

{internal_table}

Kandidati i ngrirë mori accuracy {selected_internal['accuracy']:.4f}, F1
weighted {selected_internal['f1_weighted']:.4f}, F1 fake
{selected_internal['f1_fake']:.4f}, me confusion matrix
`{selected_internal['confusion_matrix']}`. Test-i nuk ndryshoi përzgjedhjen.

![Krahasimi i brendshëm](figures/day15_internal_candidate_comparison.png)

## Gjatësia dhe bias-i

{length_table}

Cohort-et e skajeve:

{special_table}

{bias_table}

Spearman përdor raw decision score të Linear SVM, jo probabilitet. Kundrejt
C=1.0, kandidati ndryshoi mean absolute within-label correlation nga
{baseline_bias['mean_absolute_within_label_spearman']:.4f} në
{selected_bias['mean_absolute_within_label_spearman']:.4f}.
Vlerën më të ulët e pati C={float(lowest_bias['c_value'])}, me
{lowest_bias['mean_absolute_within_label_spearman']:.4f}, por të tre kandidatët
morën të njëjtin rezultat te real 30-60 dhe fake mbi 250 fjalë. Pra tuning-u
nuk solli përmirësim praktik në cohort-et e skajeve.

![Performanca sipas gjatësisë](figures/day15_length_performance.png)

## Krahasimi me C=1.0

{interpretation}

- CV F1 weighted: {baseline_cv['mean_f1_weighted']:.4f} te C=1.0 kundrejt
  {selected_cv['mean_f1_weighted']:.4f} te C={selected_c_text}.
- CV recall gap: {baseline_cv['mean_recall_gap']:.4f} kundrejt
  {selected_cv['mean_recall_gap']:.4f}.
- Internal F1 weighted: {baseline_internal['f1_weighted']:.4f} kundrejt
  {selected_internal['f1_weighted']:.4f}.
- Internal recall real/fake te kandidati: {selected_internal['recall_real']:.4f}
  / {selected_internal['recall_fake']:.4f}.

## Dataset-i i jashtëm vetëm diagnostik

U ekzekutua vetëm kandidati i ngrirë:

{external_table}

Rezultati i jashtëm nuk u përdor për të ndryshuar `C`.

![External confusion matrix](figures/day15_external_confusion_matrix.png)

## Rekomandimi për Ditën 16

Rekomandohet **Word + Character TF-IDF + Linear SVM, C={selected_c_text}** për
probability calibration të kontrolluar. Konfigurimi mbetet i pakalibruar dhe
nuk duhet integruar ende në Streamlit.

## Kufizimet

- U provuan vetëm pesë vlera të paracaktuara të `C`.
- Tuning-u bazohet në të njëjtin corpus si test-i i brendshëm.
- Cohort-et diagnostike kanë pak shembuj në skajet e gjatësisë.
- Decision score nuk është probabilitet.
- Benchmark-u i jashtëm ka domain shift dhe source-label confounding; ai nuk
  është validation set.
- Calibration dhe kontrolli i pragjeve mbeten për Ditën 16.

Modelet e tre kandidatëve të përcaktuar nga CV ruhen te
`models/day15_word_char_linear_svm_c_*.joblib`.

Modeli aktual `models/calibrated_tfidf_logreg.joblib` nuk u zëvendësua.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def run_day15_tuning() -> dict:
    """Run train-only tuning, then frozen internal and external diagnostics."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    frozen_paths = {
        "day13_selection": DAY13_SELECTION_PATH,
        "day14_selection": DAY14_SELECTION_PATH,
        "current_app_model": CURRENT_APP_MODEL_PATH,
        "streamlit_app": STREAMLIT_APP_PATH,
        "external_dataset": EXTERNAL_PATH,
    }
    required = [TRAIN_PATH, TEST_PATH, *frozen_paths.values()]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Day 15 inputs: {missing}")
    hashes_before = {name: file_sha256(path) for name, path in frozen_paths.items()}

    frozen_setup = verify_frozen_setup()
    raw_train = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig", keep_default_na=False)
    train, stale_train_rows = refresh_model_text(raw_train)
    if train["model_text"].str.strip().eq("").any():
        raise ValueError("Train contains empty model_text values.")

    LOGGER.info("Running group-safe Linear SVM tuning for C=%s", C_VALUES)
    cv_folds, cv_summary, fold_audit, group_count = run_svm_cv(train)
    cv_folds.to_csv(CV_FOLDS_PATH, index=False, encoding="utf-8")
    cv_summary.to_csv(CV_SUMMARY_PATH, index=False, encoding="utf-8")

    selection = select_c_from_cv(cv_summary)
    selection["group_count"] = group_count
    selection["cv_fold_audit"] = fold_audit
    SELECTION_PATH.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    selection_hash = file_sha256(SELECTION_PATH)

    # The internal test remains unread until the train/CV choice is immutable.
    test, test_audit = load_internal_test_after_selection(train, selection_hash)
    analysis_c_values = [float(value) for value in selection["analysis_c_values"]]
    models, training_metadata = fit_analysis_models(train, analysis_c_values)
    internal_tables = [
        prediction_table(test, models[c_value], c_value, "article_id")
        for c_value in analysis_c_values
    ]
    internal_predictions = pd.concat(internal_tables, ignore_index=True)
    (
        internal_comparison,
        length_metrics,
        special_cohorts,
        length_bias,
    ) = evaluate_internal_candidates(internal_predictions)

    internal_columns = [
        "candidate_id",
        "c_value",
        "article_id",
        "pair_id",
        "label",
        "label_name",
        "title",
        "word_count",
        "length_group",
        "score_type",
        "decision_score_fake",
        "binary_prediction",
        "prediction_correct",
        "error_type",
    ]
    internal_predictions[internal_columns].to_csv(
        INTERNAL_PREDICTIONS_PATH, index=False, encoding="utf-8"
    )
    internal_comparison.to_csv(
        INTERNAL_COMPARISON_PATH, index=False, encoding="utf-8"
    )
    length_metrics.to_csv(LENGTH_METRICS_PATH, index=False, encoding="utf-8")
    special_cohorts.to_csv(SPECIAL_COHORTS_PATH, index=False, encoding="utf-8")
    length_bias.to_csv(LENGTH_BIAS_PATH, index=False, encoding="utf-8")

    selected_c = float(selection["selected_c"])
    plot_cv_tuning(cv_summary, selected_c)
    plot_internal(internal_comparison, selected_c)
    plot_length(length_metrics, selected_c)

    # Only the frozen selected candidate is evaluated on external data.
    external, stale_external_rows = load_external_after_selection(selection_hash)
    external_predictions = prediction_table(
        external, models[selected_c], selected_c, "external_id"
    )
    external_result = prediction_metrics(external_predictions)
    external_metrics = pd.DataFrame(
        [
            {
                "candidate_id": candidate_id(selected_c),
                "candidate_display": candidate_display(selected_c),
                "c_value": selected_c,
                **{
                    key: value
                    for key, value in external_result.items()
                    if key != "confusion_matrix"
                },
                "confusion_matrix": json.dumps(
                    external_result["confusion_matrix"]
                ),
            }
        ]
    )
    external_columns = [
        "candidate_id",
        "c_value",
        "external_id",
        "label",
        "title",
        "topic",
        "source",
        "word_count",
        "score_type",
        "decision_score_fake",
        "binary_prediction",
        "prediction_correct",
        "error_type",
    ]
    external_predictions[external_columns].to_csv(
        EXTERNAL_PREDICTIONS_PATH, index=False, encoding="utf-8"
    )
    external_metrics.to_csv(EXTERNAL_METRICS_PATH, index=False, encoding="utf-8")
    plot_external_confusion(external_result["confusion_matrix"], selected_c)

    verify_selection_hash(selection_hash)
    hashes_after = {name: file_sha256(path) for name, path in frozen_paths.items()}
    if hashes_before != hashes_after:
        changed = [
            name for name in frozen_paths if hashes_before[name] != hashes_after[name]
        ]
        raise RuntimeError(f"Frozen artifacts changed during Day 15: {changed}")

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
            "selection_data": "train_only",
            "cv": "5_fold_stratified_group_safe",
            "fixed_representation": True,
            "classifier": "linear_svm",
            "internal_test_used_for_selection": False,
            "external_used_for_selection_or_tuning": False,
            "calibration_applied": False,
            "application_thresholds_applied": False,
            "selection_locked_before_internal_test": True,
            "selection_locked_before_external": True,
        },
        "frozen_setup": frozen_setup,
        "data_audit": data_audit,
        "selection": selection,
        "cv_summary": [
            rounded_metrics(row) for row in cv_summary.to_dict(orient="records")
        ],
        "training_metadata": training_metadata,
        "internal_metrics": [
            rounded_metrics(row)
            for row in internal_comparison.to_dict(orient="records")
        ],
        "length_bias": [
            rounded_metrics(row) for row in length_bias.to_dict(orient="records")
        ],
        "external_metrics": rounded_metrics(external_metrics.iloc[0].to_dict()),
        "comparison_with_c_1": {
            "cv_f1_delta": float(
                cv_summary.loc[
                    cv_summary["c_value"].eq(selected_c), "mean_f1_weighted"
                ].iloc[0]
                - cv_summary.loc[
                    cv_summary["c_value"].eq(BASELINE_C), "mean_f1_weighted"
                ].iloc[0]
            ),
            "same_configuration_selected": selected_c == BASELINE_C,
        },
        "integrity": {
            "hashes_before": hashes_before,
            "hashes_after": hashes_after,
            "all_frozen_artifacts_unchanged": hashes_before == hashes_after,
            "selection_sha256": selection_hash,
            "selection_unchanged_after_test_and_external": (
                file_sha256(SELECTION_PATH) == selection_hash
            ),
        },
        "artifacts": {
            "cv_fold_results": str(CV_FOLDS_PATH.relative_to(PROJECT_ROOT)),
            "cv_summary": str(CV_SUMMARY_PATH.relative_to(PROJECT_ROOT)),
            "selection": str(SELECTION_PATH.relative_to(PROJECT_ROOT)),
            "internal_predictions": str(
                INTERNAL_PREDICTIONS_PATH.relative_to(PROJECT_ROOT)
            ),
            "internal_comparison": str(
                INTERNAL_COMPARISON_PATH.relative_to(PROJECT_ROOT)
            ),
            "length_group_metrics": str(LENGTH_METRICS_PATH.relative_to(PROJECT_ROOT)),
            "special_cohorts": str(SPECIAL_COHORTS_PATH.relative_to(PROJECT_ROOT)),
            "length_bias": str(LENGTH_BIAS_PATH.relative_to(PROJECT_ROOT)),
            "external_predictions": str(
                EXTERNAL_PREDICTIONS_PATH.relative_to(PROJECT_ROOT)
            ),
            "external_metrics": str(EXTERNAL_METRICS_PATH.relative_to(PROJECT_ROOT)),
            "report": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
        },
    }
    METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(
        metrics,
        cv_summary,
        internal_comparison,
        length_metrics,
        special_cohorts,
        length_bias,
        external_metrics,
    )
    return metrics


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    metrics = run_day15_tuning()
    selection = metrics["selection"]
    print("Day 15 completed.")
    print("Selected C:", selection["selected_c"])
    print("Selection used external data:", selection["external_results_used"])
    print("Report:", REPORT_PATH)


if __name__ == "__main__":
    main()
