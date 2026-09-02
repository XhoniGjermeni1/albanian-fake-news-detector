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
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
    fake_decision_scores,
    rounded_metrics,
)
from src.models.builders import (  # noqa: E402
    FIXED_CHAR_CONFIG,
    build_fixed_features,
)


TRAIN_PATH = PROJECT_ROOT / "data" / "interim" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "interim" / "test.csv"
EXTERNAL_PATH = PROJECT_ROOT / "data" / "external" / "external_news.csv"
DAY13_SELECTION_PATH = PROJECT_ROOT / "reports" / "day13_internal_selection.json"
CURRENT_APP_MODEL_PATH = PROJECT_ROOT / "models" / "calibrated_tfidf_logreg.joblib"
STREAMLIT_APP_PATH = PROJECT_ROOT / "app" / "streamlit_app.py"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

CV_FOLDS_PATH = REPORTS_DIR / "day14_cv_fold_results.csv"
CV_SUMMARY_PATH = REPORTS_DIR / "day14_cv_summary.csv"
SELECTION_PATH = REPORTS_DIR / "day14_selection.json"
INTERNAL_PREDICTIONS_PATH = REPORTS_DIR / "day14_internal_predictions.csv"
INTERNAL_COMPARISON_PATH = REPORTS_DIR / "day14_internal_comparison.csv"
LENGTH_METRICS_PATH = REPORTS_DIR / "day14_length_group_metrics.csv"
COHORT_METRICS_PATH = REPORTS_DIR / "day14_special_cohort_metrics.csv"
LENGTH_BIAS_PATH = REPORTS_DIR / "day14_length_bias_comparison.csv"
EXTERNAL_PREDICTIONS_PATH = REPORTS_DIR / "day14_external_predictions.csv"
EXTERNAL_COMPARISON_PATH = REPORTS_DIR / "day14_external_comparison.csv"
METRICS_PATH = REPORTS_DIR / "day14_metrics.json"
REPORT_PATH = REPORTS_DIR / "day14_classifier_comparison.md"

CV_FIGURE_PATH = FIGURES_DIR / "day14_cv_classifier_comparison.png"
INTERNAL_FIGURE_PATH = FIGURES_DIR / "day14_internal_classifier_comparison.png"
LENGTH_FIGURE_PATH = FIGURES_DIR / "day14_length_performance.png"
EXTERNAL_FIGURE_PATH = FIGURES_DIR / "day14_external_diagnostic.png"

MODEL_PATHS = {
    "logistic_regression": MODELS_DIR / "day14_word_char_logistic_regression.joblib",
    "linear_svm": MODELS_DIR / "day14_word_char_linear_svm.joblib",
    "complement_nb": MODELS_DIR / "day14_word_char_complement_nb.joblib",
}

CLASSIFIER_DISPLAY = {
    "logistic_regression": "Logistic Regression",
    "linear_svm": "Linear SVM",
    "complement_nb": "Complement Naive Bayes",
}
SCORE_DISPLAY = {
    "logistic_regression": "log-odds nga decision_function",
    "linear_svm": "decision score (jo probabilitet)",
    "complement_nb": "diferencë e log-probabiliteteve",
}
COLORS = {
    "logistic_regression": "#3976A8",
    "linear_svm": "#D76745",
    "complement_nb": "#2F937F",
}

CLASSIFIER_CONFIGS = [
    {
        "candidate_id": "logistic_regression_c_0_5",
        "classifier": "logistic_regression",
        "parameter_name": "C",
        "parameter_value": 0.5,
    },
    {
        "candidate_id": "logistic_regression_c_1_0",
        "classifier": "logistic_regression",
        "parameter_name": "C",
        "parameter_value": 1.0,
    },
    {
        "candidate_id": "linear_svm_c_0_5",
        "classifier": "linear_svm",
        "parameter_name": "C",
        "parameter_value": 0.5,
    },
    {
        "candidate_id": "linear_svm_c_1_0",
        "classifier": "linear_svm",
        "parameter_name": "C",
        "parameter_value": 1.0,
    },
    {
        "candidate_id": "complement_nb_alpha_0_5",
        "classifier": "complement_nb",
        "parameter_name": "alpha",
        "parameter_value": 0.5,
    },
    {
        "candidate_id": "complement_nb_alpha_1_0",
        "classifier": "complement_nb",
        "parameter_name": "alpha",
        "parameter_value": 1.0,
    },
]

CV_METRICS = [
    "accuracy",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
    "f1_fake",
    "recall_real",
    "recall_fake",
    "training_seconds",
    "classifier_fit_seconds",
    "prediction_seconds",
]
SELECTION_TOLERANCE = 0.002
LOGGER = logging.getLogger(__name__)


def load_fixed_representation() -> dict:
    """Load and verify the Word + Character representation selected on Day 13."""
    if not DAY13_SELECTION_PATH.exists():
        raise FileNotFoundError(f"Missing Day 13 selection: {DAY13_SELECTION_PATH}")
    selection = json.loads(DAY13_SELECTION_PATH.read_text(encoding="utf-8"))
    selected_config = selection.get("selected_char_config")
    if selection.get("recommended_for_day14") != "word_char_tfidf":
        raise ValueError("Day 13 did not recommend Word + Character TF-IDF.")
    if selected_config != FIXED_CHAR_CONFIG:
        raise ValueError("Day 13 character configuration differs from the frozen setup.")
    if selection.get("external_results_used") is not False:
        raise ValueError("Day 13 selection is not marked as internal-only.")
    return selected_config.copy()


def build_classifier(candidate: dict):
    """Build one uncalibrated classifier candidate."""
    value = float(candidate["parameter_value"])
    classifier = candidate["classifier"]
    if classifier == "logistic_regression":
        return LogisticRegression(
            C=value,
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )
    if classifier == "linear_svm":
        return LinearSVC(
            C=value,
            class_weight="balanced",
            max_iter=5000,
            random_state=42,
        )
    if classifier == "complement_nb":
        return ComplementNB(alpha=value)
    raise ValueError(f"Unknown classifier: {classifier}")


def build_model_pipeline(candidate: dict, char_config: dict) -> Pipeline:
    """Build a complete uncalibrated model for final evaluation."""
    return Pipeline(
        [
            ("features", build_fixed_features(char_config)),
            ("classifier", build_classifier(candidate)),
        ]
    )


def run_group_safe_cv(
    train: pd.DataFrame,
    char_config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], int]:
    """Evaluate the small candidate screen with shared fold-level TF-IDF."""
    folds, groups, fold_audit = build_group_safe_folds(train)
    rows: list[dict] = []

    for fold_number, (fit_index, validation_index) in enumerate(folds, start=1):
        fit = train.iloc[fit_index]
        validation = train.iloc[validation_index]

        vectorizer = build_fixed_features(char_config)
        feature_started = time.perf_counter()
        x_fit = vectorizer.fit_transform(fit["model_text"], fit["label"])
        feature_fit_seconds = time.perf_counter() - feature_started

        transform_started = time.perf_counter()
        x_validation = vectorizer.transform(validation["model_text"])
        feature_transform_seconds = time.perf_counter() - transform_started

        for candidate in CLASSIFIER_CONFIGS:
            classifier = clone(build_classifier(candidate))
            fit_started = time.perf_counter()
            classifier.fit(x_fit, fit["label"])
            classifier_fit_seconds = time.perf_counter() - fit_started

            predict_started = time.perf_counter()
            predictions = classifier.predict(x_validation)
            prediction_seconds = time.perf_counter() - predict_started
            metrics = classification_metrics(validation["label"], predictions)
            rows.append(
                {
                    **candidate,
                    "classifier_display": CLASSIFIER_DISPLAY[
                        candidate["classifier"]
                    ],
                    "fold": fold_number,
                    **{
                        key: value
                        for key, value in metrics.items()
                        if key != "confusion_matrix"
                    },
                    "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
                    "feature_fit_seconds": feature_fit_seconds,
                    "feature_transform_seconds": feature_transform_seconds,
                    "classifier_fit_seconds": classifier_fit_seconds,
                    "training_seconds": (
                        feature_fit_seconds + classifier_fit_seconds
                    ),
                    "prediction_seconds": prediction_seconds,
                }
            )

    fold_results = pd.DataFrame(rows)
    summary_rows: list[dict] = []
    for candidate in CLASSIFIER_CONFIGS:
        candidate_rows = fold_results.loc[
            fold_results["candidate_id"].eq(candidate["candidate_id"])
        ]
        summary_row = {
            **candidate,
            "classifier_display": CLASSIFIER_DISPLAY[candidate["classifier"]],
            "folds": int(len(candidate_rows)),
        }
        for metric in CV_METRICS:
            values = candidate_rows[metric].astype(float)
            summary_row[f"mean_{metric}"] = float(values.mean())
            summary_row[f"std_{metric}"] = float(values.std(ddof=1))
        summary_row["total_classifier_fit_seconds"] = float(
            candidate_rows["classifier_fit_seconds"].sum()
        )
        summary_rows.append(summary_row)

    summary = pd.DataFrame(summary_rows)
    return fold_results, summary, fold_audit, int(len(np.unique(groups)))


def select_from_cv(cv_summary: pd.DataFrame, char_config: dict) -> dict:
    """Select a candidate using only CV F1 and fold stability."""
    ranking_columns = [
        "mean_f1_weighted",
        "std_f1_weighted",
        "mean_f1_fake",
        "mean_accuracy",
        "mean_training_seconds",
    ]
    family_best_rows = []
    for _, family in cv_summary.groupby("classifier", sort=False):
        ranked = family.sort_values(
            ranking_columns,
            ascending=[False, True, False, False, True],
        )
        family_best_rows.append(ranked.iloc[0])
    family_best = pd.DataFrame(family_best_rows).reset_index(drop=True)

    best_mean_f1 = float(family_best["mean_f1_weighted"].max())
    finalists = family_best.loc[
        family_best["mean_f1_weighted"].ge(
            best_mean_f1 - SELECTION_TOLERANCE - 1e-12
        )
    ].copy()
    deployment_preference = {
        "logistic_regression": 0,
        "linear_svm": 1,
        "complement_nb": 2,
    }
    finalists["deployment_preference"] = finalists["classifier"].map(
        deployment_preference
    )
    winner = finalists.sort_values(
        [
            "std_f1_weighted",
            "mean_f1_weighted",
            "mean_f1_fake",
            "deployment_preference",
            "mean_training_seconds",
        ],
        ascending=[True, False, False, True, True],
    ).iloc[0]

    most_stable = family_best.sort_values(
        ["std_f1_weighted", "mean_f1_weighted"],
        ascending=[True, False],
    ).iloc[0]
    balance = family_best.assign(
        recall_gap=lambda frame: (
            frame["mean_recall_real"] - frame["mean_recall_fake"]
        ).abs()
    ).sort_values(
        ["recall_gap", "mean_f1_weighted"],
        ascending=[True, False],
    )

    best_configs = {
        str(row["classifier"]): {
            "candidate_id": str(row["candidate_id"]),
            "parameter_name": str(row["parameter_name"]),
            "parameter_value": float(row["parameter_value"]),
            "mean_f1_weighted": float(row["mean_f1_weighted"]),
            "std_f1_weighted": float(row["std_f1_weighted"]),
        }
        for _, row in family_best.iterrows()
    }
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
            "character": char_config,
        },
        "candidate_count": int(len(cv_summary)),
        "selection_tolerance_f1": SELECTION_TOLERANCE,
        "selection_rule": (
            "Highest mean F1 weighted; candidates within 0.002 are ordered by "
            "lower F1 standard deviation, then F1 fake and deployment suitability."
        ),
        "best_config_by_classifier": best_configs,
        "winner_classifier": str(winner["classifier"]),
        "winner_candidate_id": str(winner["candidate_id"]),
        "winner_mean_f1_weighted": float(winner["mean_f1_weighted"]),
        "winner_std_f1_weighted": float(winner["std_f1_weighted"]),
        "most_stable_classifier": str(most_stable["classifier"]),
        "best_recall_balance_classifier": str(balance.iloc[0]["classifier"]),
        "internal_test_used": False,
        "external_results_used": False,
        "calibration_applied": False,
    }


def candidate_from_selection(selection: dict, classifier: str) -> dict:
    """Return the full candidate dictionary selected for one classifier family."""
    candidate_id = selection["best_config_by_classifier"][classifier][
        "candidate_id"
    ]
    return next(
        candidate.copy()
        for candidate in CLASSIFIER_CONFIGS
        if candidate["candidate_id"] == candidate_id
    )


def verify_selection_hash(expected_hash: str) -> None:
    """Ensure that CV selection has not changed after it was written."""
    if not SELECTION_PATH.exists() or file_sha256(SELECTION_PATH) != expected_hash:
        raise RuntimeError("The frozen Day 14 CV selection has changed.")


def load_internal_test_after_selection(
    train: pd.DataFrame,
    selection_hash: str,
) -> tuple[pd.DataFrame, dict]:
    """Load the internal test only after the train/CV selection is frozen."""
    verify_selection_hash(selection_hash)
    raw_test = pd.read_csv(TEST_PATH, encoding="utf-8-sig", keep_default_na=False)
    test, stale_rows = refresh_model_text(raw_test)
    evaluation_test, excluded_ids = exclude_train_duplicates_from_test(train, test)
    evaluation_test = add_word_counts(evaluation_test)
    return evaluation_test, {
        "original_test_rows": int(len(raw_test)),
        "evaluation_test_rows": int(len(evaluation_test)),
        "exact_train_duplicates_excluded": int(len(excluded_ids)),
        "excluded_article_ids": excluded_ids,
        "stale_test_model_text_rows_refreshed_in_memory": stale_rows,
    }


def fit_selected_family_models(
    train: pd.DataFrame,
    selection: dict,
    char_config: dict,
) -> tuple[dict[str, Pipeline], dict[str, dict]]:
    """Fit and save the best CV configuration from every classifier family."""
    models: dict[str, Pipeline] = {}
    metadata: dict[str, dict] = {}
    for classifier in CLASSIFIER_DISPLAY:
        candidate = candidate_from_selection(selection, classifier)
        model = build_model_pipeline(candidate, char_config)
        started = time.perf_counter()
        model.fit(train["model_text"], train["label"])
        training_seconds = time.perf_counter() - started
        joblib.dump(model, MODEL_PATHS[classifier], compress=3)
        size_bytes = MODEL_PATHS[classifier].stat().st_size
        models[classifier] = model
        metadata[classifier] = {
            "candidate": candidate,
            "training_seconds": round(training_seconds, 3),
            "model_size_bytes": int(size_bytes),
            "model_size_mb": round(size_bytes / (1024 * 1024), 3),
            "model_path": str(MODEL_PATHS[classifier].relative_to(PROJECT_ROOT)),
            "model_sha256": file_sha256(MODEL_PATHS[classifier]),
            "calibrated": False,
        }
    return models, metadata


def prediction_table(
    base_data: pd.DataFrame,
    model: Pipeline,
    classifier: str,
    id_column: str,
) -> pd.DataFrame:
    """Create predictions and raw fake-oriented scores for one classifier."""
    predictions = model.predict(base_data["model_text"]).astype(int)
    scores = fake_decision_scores(model, base_data["model_text"])
    table = base_data.copy().reset_index(drop=True)
    table.insert(0, "classifier", classifier)
    table.insert(1, "classifier_display", CLASSIFIER_DISPLAY[classifier])
    table["score_type"] = SCORE_DISPLAY[classifier]
    table["decision_score_fake"] = scores
    table["binary_prediction"] = predictions
    table["prediction_correct"] = table["label"].eq(predictions)
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


def metrics_from_prediction_table(table: pd.DataFrame) -> dict:
    """Calculate metrics and score summaries from a prediction table."""
    metrics = classification_metrics(table["label"], table["binary_prediction"])
    metrics["mean_score_real"] = (
        float(table.loc[table["label"].eq(0), "decision_score_fake"].mean())
        if table["label"].eq(0).any()
        else None
    )
    metrics["mean_score_fake"] = (
        float(table.loc[table["label"].eq(1), "decision_score_fake"].mean())
        if table["label"].eq(1).any()
        else None
    )
    return metrics


def evaluate_internal_cohorts(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate all length groups and the two requested difficult cohorts."""
    all_rows: list[dict] = []
    for classifier, table in predictions.groupby("classifier", sort=False):
        cohorts = {
            "all_internal_test": table,
            "real_30_60": table.loc[
                table["label"].eq(0) & table["word_count"].between(30, 60)
            ],
            "fake_gt_250": table.loc[
                table["label"].eq(1) & table["word_count"].gt(250)
            ],
        }
        for length_group in LENGTH_LABELS:
            cohorts[f"length_{length_group}"] = table.loc[
                table["length_group"].astype(str).eq(length_group)
            ]

        for cohort_name, cohort in cohorts.items():
            metrics = metrics_from_prediction_table(cohort)
            all_rows.append(
                {
                    "classifier": classifier,
                    "classifier_display": CLASSIFIER_DISPLAY[classifier],
                    "cohort": cohort_name,
                    **{
                        key: value
                        for key, value in metrics.items()
                        if key != "confusion_matrix"
                    },
                    "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
                }
            )

    cohort_metrics = pd.DataFrame(all_rows)
    internal_comparison = cohort_metrics.loc[
        cohort_metrics["cohort"].eq("all_internal_test")
    ].copy()
    length_metrics = cohort_metrics.loc[
        cohort_metrics["cohort"].str.startswith("length_")
    ].copy()
    length_metrics["length_group"] = length_metrics["cohort"].str.replace(
        "length_", "", regex=False
    )
    length_metrics["length_description"] = length_metrics["length_group"].map(
        LENGTH_DISPLAY
    )
    special = cohort_metrics.loc[
        cohort_metrics["cohort"].isin(["real_30_60", "fake_gt_250"])
    ].copy()
    return internal_comparison, length_metrics, special


def length_bias_comparison(predictions: pd.DataFrame) -> pd.DataFrame:
    """Measure rank association between length and each model's raw score."""
    rows: list[dict] = []
    for classifier, table in predictions.groupby("classifier", sort=False):
        correlations: dict[str, tuple[float, float]] = {}
        for scope, subset in (
            ("all", table),
            ("real", table.loc[table["label"].eq(0)]),
            ("fake", table.loc[table["label"].eq(1)]),
        ):
            result = spearmanr(subset["word_count"], subset["decision_score_fake"])
            correlations[scope] = (float(result.statistic), float(result.pvalue))

        real_short = table.loc[
            table["label"].eq(0) & table["word_count"].between(30, 60)
        ]
        fake_long = table.loc[
            table["label"].eq(1) & table["word_count"].gt(250)
        ]
        mean_abs_within_label = (
            abs(correlations["real"][0]) + abs(correlations["fake"][0])
        ) / 2
        rows.append(
            {
                "classifier": classifier,
                "classifier_display": CLASSIFIER_DISPLAY[classifier],
                "score_type": SCORE_DISPLAY[classifier],
                "spearman_all": correlations["all"][0],
                "spearman_all_p": correlations["all"][1],
                "spearman_real": correlations["real"][0],
                "spearman_real_p": correlations["real"][1],
                "spearman_fake": correlations["fake"][0],
                "spearman_fake_p": correlations["fake"][1],
                "mean_absolute_within_label_spearman": mean_abs_within_label,
                "real_30_60_accuracy": float(
                    real_short["prediction_correct"].mean()
                ),
                "fake_gt_250_accuracy": float(
                    fake_long["prediction_correct"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def load_external_after_selection(selection_hash: str) -> tuple[pd.DataFrame, int]:
    """Load the frozen external benchmark only after CV selection is locked."""
    verify_selection_hash(selection_hash)
    external = pd.read_csv(EXTERNAL_PATH, encoding="utf-8", keep_default_na=False)
    if len(external) != 40 or set(external["label"]) != {"real", "fake"}:
        raise ValueError("The external benchmark is not the expected frozen dataset.")
    external["label"] = external["label"].map({"real": 0, "fake": 1})
    external, stale_rows = refresh_model_text(external)
    return add_word_counts(external), stale_rows


def best_cv_rows(cv_summary: pd.DataFrame, selection: dict) -> pd.DataFrame:
    """Return the chosen configuration from every classifier family."""
    ids = [
        details["candidate_id"]
        for details in selection["best_config_by_classifier"].values()
    ]
    order = {candidate_id: index for index, candidate_id in enumerate(ids)}
    result = cv_summary.loc[cv_summary["candidate_id"].isin(ids)].copy()
    result["display_order"] = result["candidate_id"].map(order)
    return result.sort_values("display_order")


def plot_cv(cv_summary: pd.DataFrame, selection: dict) -> None:
    """Plot CV F1 with fold-level standard deviations."""
    data = best_cv_rows(cv_summary, selection)
    names = data["classifier"].tolist()
    values = data["mean_f1_weighted"].to_numpy()
    errors = data["std_f1_weighted"].to_numpy()
    figure, axis = plt.subplots(figsize=(9, 5.4))
    bars = axis.bar(
        [CLASSIFIER_DISPLAY[name] for name in names],
        values,
        yerr=errors,
        capsize=6,
        color=[COLORS[name] for name in names],
    )
    axis.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=4)
    axis.set_ylim(max(0, float(values.min()) - 0.08), min(1.0, float(values.max()) + 0.06))
    axis.set_ylabel("F1 weighted mesatare")
    axis.set_title("Group-safe cross-validation vetëm mbi train")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(CV_FIGURE_PATH, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_internal(comparison: pd.DataFrame) -> None:
    """Plot key internal test metrics."""
    metrics = ["accuracy", "f1_weighted", "f1_fake"]
    labels = ["Accuracy", "F1 weighted", "F1 fake"]
    x = np.arange(len(metrics))
    width = 0.24
    figure, axis = plt.subplots(figsize=(10, 5.6))
    for index, classifier in enumerate(CLASSIFIER_DISPLAY):
        row = comparison.loc[comparison["classifier"].eq(classifier)].iloc[0]
        axis.bar(
            x + (index - 1) * width,
            [row[metric] for metric in metrics],
            width,
            label=CLASSIFIER_DISPLAY[classifier],
            color=COLORS[classifier],
        )
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Rezultati")
    axis.set_title("Vlerësimi në test set-in e brendshëm")
    axis.legend(loc="lower right")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(INTERNAL_FIGURE_PATH, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_length_groups(length_metrics: pd.DataFrame) -> None:
    """Plot accuracy by the fixed Day 12 length groups."""
    x = np.arange(len(LENGTH_LABELS))
    figure, axis = plt.subplots(figsize=(11, 5.8))
    for classifier in CLASSIFIER_DISPLAY:
        table = length_metrics.loc[
            length_metrics["classifier"].eq(classifier)
        ].set_index("length_group")
        values = [float(table.loc[group, "accuracy"]) for group in LENGTH_LABELS]
        axis.plot(
            x,
            values,
            marker="o",
            linewidth=2,
            label=CLASSIFIER_DISPLAY[classifier],
            color=COLORS[classifier],
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


def plot_external(comparison: pd.DataFrame) -> None:
    """Plot the frozen external diagnostic results."""
    metrics = ["accuracy", "recall_real", "recall_fake"]
    labels = ["Accuracy", "Recall real", "Recall fake"]
    x = np.arange(len(metrics))
    width = 0.24
    figure, axis = plt.subplots(figsize=(10, 5.6))
    for index, classifier in enumerate(CLASSIFIER_DISPLAY):
        row = comparison.loc[comparison["classifier"].eq(classifier)].iloc[0]
        axis.bar(
            x + (index - 1) * width,
            [row[metric] for metric in metrics],
            width,
            label=CLASSIFIER_DISPLAY[classifier],
            color=COLORS[classifier],
        )
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Rezultati")
    axis.set_title("Dataset-i i jashtëm, vetëm diagnostik")
    axis.legend(loc="lower right")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(EXTERNAL_FIGURE_PATH, dpi=160, bbox_inches="tight")
    plt.close(figure)


def write_report(
    metrics: dict,
    cv_summary: pd.DataFrame,
    internal_comparison: pd.DataFrame,
    length_metrics: pd.DataFrame,
    special_cohorts: pd.DataFrame,
    length_bias: pd.DataFrame,
    external_comparison: pd.DataFrame,
) -> None:
    """Write the reproducible Day 14 report."""
    selection = metrics["selection"]
    winner = selection["winner_classifier"]
    stable = selection["most_stable_classifier"]
    balanced = selection["best_recall_balance_classifier"]
    cv_best = best_cv_rows(cv_summary, selection)

    cv_table = dataframe_to_markdown(
        cv_best,
        [
            "classifier_display",
            "parameter_name",
            "parameter_value",
            "mean_accuracy",
            "mean_precision_weighted",
            "mean_recall_weighted",
            "mean_f1_weighted",
            "std_f1_weighted",
            "mean_f1_fake",
            "mean_recall_real",
            "mean_recall_fake",
            "mean_training_seconds",
        ],
    )
    internal_table = dataframe_to_markdown(
        internal_comparison,
        [
            "classifier_display",
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
    special_table = dataframe_to_markdown(
        special_cohorts,
        [
            "classifier_display",
            "cohort",
            "rows",
            "accuracy",
            "recall_real",
            "recall_fake",
            "confusion_matrix",
        ],
    )
    length_table = dataframe_to_markdown(
        length_metrics,
        [
            "classifier_display",
            "length_description",
            "rows",
            "accuracy",
            "f1_weighted",
            "recall_real",
            "recall_fake",
        ],
    )
    bias_table = dataframe_to_markdown(
        length_bias,
        [
            "classifier_display",
            "spearman_all",
            "spearman_real",
            "spearman_fake",
            "mean_absolute_within_label_spearman",
            "real_30_60_accuracy",
            "fake_gt_250_accuracy",
        ],
    )
    external_table = dataframe_to_markdown(
        external_comparison,
        [
            "classifier_display",
            "accuracy",
            "recall_real",
            "recall_fake",
            "false_positives",
            "false_negatives",
            "confusion_matrix",
        ],
    )

    logistic_bias = float(
        length_bias.loc[
            length_bias["classifier"].eq("logistic_regression"),
            "mean_absolute_within_label_spearman",
        ].iloc[0]
    )
    lowest_bias_row = length_bias.sort_values(
        "mean_absolute_within_label_spearman"
    ).iloc[0]
    svm_bias_row = length_bias.loc[
        length_bias["classifier"].eq("linear_svm")
    ].iloc[0]
    logistic_long_fake_accuracy = float(
        length_bias.loc[
            length_bias["classifier"].eq("logistic_regression"),
            "fake_gt_250_accuracy",
        ].iloc[0]
    )
    winner_internal = internal_comparison.loc[
        internal_comparison["classifier"].eq(winner)
    ].iloc[0]
    winner_external = external_comparison.loc[
        external_comparison["classifier"].eq(winner)
    ].iloc[0]
    external_best = external_comparison.sort_values(
        ["accuracy", "f1_weighted"], ascending=[False, False]
    ).iloc[0]

    report = f"""# Dita 14 - Krahasimi i classifier-ëve

## Protokolli

Përfaqësimi u mbajt fiks si në Ditën 13: Word TF-IDF `(1, 2)`, maksimumi
30,000 features, plus Character TF-IDF `char_wb (3, 5)`, maksimumi 50,000
features. U përdor i njëjti preprocessing bazë dhe asnjë classifier nuk u
kalibrua.

Përzgjedhja u bë vetëm mbi {metrics['data_audit']['train_rows']} artikujt train
me 5-fold `StratifiedGroupKFold`. `pair_id` i njëjtë dhe tekstet identike u
mbajtën në të njëjtin leakage-group; u gjetën
{metrics['data_audit']['group_count']} grupe dhe zero mbivendosje mes fit dhe
validation. Për shkak të normalizimit aktual NFC, u rindërtuan vetëm në memorie
{metrics['data_audit']['stale_train_model_text_rows_refreshed_in_memory']} vlera
`model_text`; CSV-të nuk u ndryshuan.

Zgjedhja u shkrua te `reports/day14_selection.json` përpara se të ngarkohej
test set-i i brendshëm. Dataset-i i jashtëm u hap vetëm pas kësaj. Test-i,
benchmark-u i jashtëm dhe modeli aktual i Streamlit nuk u përdorën për tuning.

## Konfigurimet e provuara

U provuan dy vlera të arsyeshme për secilën familje: Logistic Regression dhe
Linear SVM me `C=0.5/1.0`, si dhe Complement Naive Bayes me `alpha=0.5/1.0`.
TF-IDF u përshtat një herë për çdo fold dhe u nda mes kandidatëve; koha e
trajnimit në tabelë përfshin atë kosto të përbashkët plus fit-in e classifier-it.

## Cross-validation vetëm mbi train

{cv_table}

Rregulli kryesor ishte F1 weighted mesatare. Kandidatët brenda 0.002 nga vlera
më e mirë u renditën sipas devijimit standard më të ulët, pastaj F1 fake dhe
përshtatshmërisë për calibration/deploy. Fituesi i ngrirë ishte
**{CLASSIFIER_DISPLAY[winner]}** (`{selection['winner_candidate_id']}`), me
F1 weighted {selection['winner_mean_f1_weighted']:.4f} ±
{selection['winner_std_f1_weighted']:.4f}.

Classifier-i më i qëndrueshëm sipas devijimit standard ishte
**{CLASSIFIER_DISPLAY[stable]}**, ndërsa hendekun më të vogël mes recall real
dhe fake e pati **{CLASSIFIER_DISPLAY[balanced]}**. Këto përfundime përdorin
vetëm train/CV.

![Cross-validation](figures/day14_cv_classifier_comparison.png)

## Test set-i i brendshëm

Pas ngrirjes së përzgjedhjes u përjashtuan
{metrics['data_audit']['exact_train_duplicates_excluded']} dublikatat ekzakte
train/test dhe mbetën {metrics['data_audit']['evaluation_test_rows']} artikuj.

{internal_table}

Kandidati i zgjedhur arriti accuracy {winner_internal['accuracy']:.4f}, F1
weighted {winner_internal['f1_weighted']:.4f} dhe F1 fake
{winner_internal['f1_fake']:.4f}. Ky rezultat nuk u përdor për të ndryshuar
zgjedhjen.

![Test-i i brendshëm](figures/day14_internal_classifier_comparison.png)

## Grupet e gjatësisë

{length_table}

Dy cohort-et e kërkuara:

{special_table}

![Performanca sipas gjatësisë](figures/day14_length_performance.png)

## Bias-i i gjatësisë

Për çdo classifier u përdor score-i i tij i pakalibruar, i orientuar drejt
klasës fake. Score-t e classifier-ëve kanë shkallë të ndryshme dhe nuk janë
probabilitete; krahasimi bazohet te Spearman rank correlation.

{bias_table}

Logistic Regression kishte mesataren absolute within-label
{logistic_bias:.4f}. Vlerën më të ulët e pati
**{lowest_bias_row['classifier_display']}** me
{lowest_bias_row['mean_absolute_within_label_spearman']:.4f}. Një vlerë pak më
e ulët nuk do të thotë se bias-i u eliminua; cohort-et e skajeve mbeten të
vogla dhe të vështira.

Linear SVM gjithashtu e uli këtë lidhje nga {logistic_bias:.4f} te
{svm_bias_row['mean_absolute_within_label_spearman']:.4f}. Te 29 lajmet fake
mbi 250 fjalë, accuracy u rrit nga {logistic_long_fake_accuracy:.4f} te
{svm_bias_row['fake_gt_250_accuracy']:.4f}; pra kandidati e zbuti bias-in, por
nuk e eliminoi.

## Dataset-i i jashtëm vetëm diagnostik

{external_table}

Kandidati i ngrirë **{CLASSIFIER_DISPLAY[winner]}** arriti accuracy
{winner_external['accuracy']:.4f}, recall real
{winner_external['recall_real']:.4f} dhe recall fake
{winner_external['recall_fake']:.4f}. Rezultatet e jashtme nuk ndryshuan
classifier-in ose parametrat.

Accuracy më të lartë diagnostike e pati
**{external_best['classifier_display']}** me {external_best['accuracy']:.4f}.
Kjo përmbysje e renditjes kundrejt CV/test-it të brendshëm është provë për
domain shift dhe jo arsye për tuning pas testimit.

![Diagnostika e jashtme](figures/day14_external_diagnostic.png)

## Përfundimi

- Fituesi i cross-validation ishte **{CLASSIFIER_DISPLAY[winner]}**.
- Më i qëndrueshmi mes fold-eve ishte **{CLASSIFIER_DISPLAY[stable]}**.
- Balancën më të afërt recall real/fake e dha
  **{CLASSIFIER_DISPLAY[balanced]}**.
- Lidhjen më të ulët me gjatësinë e pati
  **{lowest_bias_row['classifier_display']}**, por bias-i vazhdon.
- Linear SVM e uli dukshëm bias-in kundrejt Logistic Regression dhe dha
  rezultatin më të mirë te fake të gjatë mes tre kandidatëve.
- Në benchmark-un e jashtëm pati përmbysje renditjeje; kjo nuk ndryshon
  zgjedhjen train/CV.
- Për Ditën 15 rekomandohet **{CLASSIFIER_DISPLAY[winner]}** me përfaqësimin e
  ngrirë Word + Character TF-IDF për tuning të kufizuar dhe calibration të
  kontrolluar.

## Kufizimet

- U provuan vetëm gjashtë konfigurime të paracaktuara.
- CV-ja dhe test-i i brendshëm vijnë nga i njëjti corpus.
- Score-t e Linear SVM dhe modeleve të tjera nuk janë probabilitete dhe nuk u
  aplikuan pragjet 0.30/0.70.
- Cohort-i real 30-60 dhe fake mbi 250 fjalë kanë pak raste.
- Dataset-i i jashtëm ka përmbledhje të shkurtra dhe source-label confounding;
  ai mbetet vetëm benchmark diagnostik.
- Calibration, tuning-u final dhe integrimi në Streamlit janë lënë për ditët e
  ardhshme.

## Modelet eksperimentale

```text
models/day14_word_char_logistic_regression.joblib
models/day14_word_char_linear_svm.joblib
models/day14_word_char_complement_nb.joblib
```

Modeli `models/calibrated_tfidf_logreg.joblib` dhe aplikacioni Streamlit nuk u
zëvendësuan.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def run_day14_comparison() -> dict:
    """Run Day 14 in the strict order train/CV, internal test, external test."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    frozen_paths = {
        "day13_selection": DAY13_SELECTION_PATH,
        "current_app_model": CURRENT_APP_MODEL_PATH,
        "streamlit_app": STREAMLIT_APP_PATH,
        "external_dataset": EXTERNAL_PATH,
    }
    missing = [str(path) for path in [TRAIN_PATH, TEST_PATH, *frozen_paths.values()] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Day 14 input files: {missing}")
    hashes_before = {name: file_sha256(path) for name, path in frozen_paths.items()}

    char_config = load_fixed_representation()
    raw_train = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig", keep_default_na=False)
    train, stale_train_rows = refresh_model_text(raw_train)
    if train["model_text"].str.strip().eq("").any():
        raise ValueError("Train contains empty model_text values.")

    LOGGER.info("Running group-safe CV for %s candidates", len(CLASSIFIER_CONFIGS))
    cv_folds, cv_summary, fold_audit, group_count = run_group_safe_cv(
        train, char_config
    )
    cv_folds.to_csv(CV_FOLDS_PATH, index=False, encoding="utf-8")
    cv_summary.to_csv(CV_SUMMARY_PATH, index=False, encoding="utf-8")

    selection = select_from_cv(cv_summary, char_config)
    selection["cv_fold_audit"] = fold_audit
    selection["group_count"] = group_count
    SELECTION_PATH.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    selection_hash = file_sha256(SELECTION_PATH)

    # Test data is deliberately loaded only after this CV selection is frozen.
    test, test_audit = load_internal_test_after_selection(train, selection_hash)
    models, training_metadata = fit_selected_family_models(
        train, selection, char_config
    )
    internal_tables = [
        prediction_table(test, models[classifier], classifier, "article_id")
        for classifier in CLASSIFIER_DISPLAY
    ]
    internal_predictions = pd.concat(internal_tables, ignore_index=True)
    internal_comparison, length_metrics, special_cohorts = (
        evaluate_internal_cohorts(internal_predictions)
    )
    for classifier, metadata in training_metadata.items():
        mask = internal_comparison["classifier"].eq(classifier)
        internal_comparison.loc[mask, "final_training_seconds"] = metadata[
            "training_seconds"
        ]
        internal_comparison.loc[mask, "model_size_mb"] = metadata["model_size_mb"]
    length_bias = length_bias_comparison(internal_predictions)

    internal_columns = [
        "classifier",
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
    special_cohorts.to_csv(COHORT_METRICS_PATH, index=False, encoding="utf-8")
    length_bias.to_csv(LENGTH_BIAS_PATH, index=False, encoding="utf-8")

    plot_cv(cv_summary, selection)
    plot_internal(internal_comparison)
    plot_length_groups(length_metrics)

    # External data remains unopened until after the train-only choice is locked.
    external, stale_external_rows = load_external_after_selection(selection_hash)
    external_tables = [
        prediction_table(external, models[classifier], classifier, "external_id")
        for classifier in CLASSIFIER_DISPLAY
    ]
    external_predictions = pd.concat(external_tables, ignore_index=True)
    external_rows: list[dict] = []
    for classifier, table in external_predictions.groupby("classifier", sort=False):
        model_metrics = metrics_from_prediction_table(table)
        external_rows.append(
            {
                "classifier": classifier,
                "classifier_display": CLASSIFIER_DISPLAY[classifier],
                **{
                    key: value
                    for key, value in model_metrics.items()
                    if key != "confusion_matrix"
                },
                "confusion_matrix": json.dumps(model_metrics["confusion_matrix"]),
            }
        )
    external_comparison = pd.DataFrame(external_rows)
    external_columns = [
        "classifier",
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
    external_comparison.to_csv(
        EXTERNAL_COMPARISON_PATH, index=False, encoding="utf-8"
    )
    plot_external(external_comparison)

    verify_selection_hash(selection_hash)
    hashes_after = {name: file_sha256(path) for name, path in frozen_paths.items()}
    changed_frozen = [
        name for name in frozen_paths if hashes_before[name] != hashes_after[name]
    ]
    if changed_frozen:
        raise RuntimeError(f"Frozen artifacts changed during Day 14: {changed_frozen}")

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
            "same_fixed_word_char_tfidf": True,
            "internal_test_used_for_selection": False,
            "external_used_for_selection_or_tuning": False,
            "calibration_applied": False,
            "application_thresholds_applied": False,
            "selection_locked_before_internal_test": True,
            "selection_locked_before_external": True,
        },
        "fixed_representation": selection["fixed_representation"],
        "data_audit": data_audit,
        "candidate_configs": CLASSIFIER_CONFIGS,
        "cv_summary": [
            rounded_metrics(row)
            for row in cv_summary.to_dict(orient="records")
        ],
        "selection": selection,
        "final_training": training_metadata,
        "internal_metrics": [
            rounded_metrics(row)
            for row in internal_comparison.to_dict(orient="records")
        ],
        "length_bias": [
            rounded_metrics(row) for row in length_bias.to_dict(orient="records")
        ],
        "external_metrics": [
            rounded_metrics(row)
            for row in external_comparison.to_dict(orient="records")
        ],
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
            "special_cohort_metrics": str(
                COHORT_METRICS_PATH.relative_to(PROJECT_ROOT)
            ),
            "length_bias": str(LENGTH_BIAS_PATH.relative_to(PROJECT_ROOT)),
            "external_predictions": str(
                EXTERNAL_PREDICTIONS_PATH.relative_to(PROJECT_ROOT)
            ),
            "external_comparison": str(
                EXTERNAL_COMPARISON_PATH.relative_to(PROJECT_ROOT)
            ),
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
        external_comparison,
    )
    return metrics


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    metrics = run_day14_comparison()
    selection = metrics["selection"]
    print("Day 14 completed.")
    print("Selected classifier:", CLASSIFIER_DISPLAY[selection["winner_classifier"]])
    print("Selection used external data:", selection["external_results_used"])
    print("Report:", REPORT_PATH)


if __name__ == "__main__":
    main()
