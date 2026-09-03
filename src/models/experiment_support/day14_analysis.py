"""Computation and stable data contracts for the Day 14 classifier comparison."""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.evaluation.data_utils import (
    LENGTH_DISPLAY,
    LENGTH_LABELS,
    add_word_counts,
    build_group_safe_folds,
    exclude_train_duplicates_from_test,
    refresh_model_text,
)
from src.evaluation.experiment_utils import file_sha256
from src.evaluation.metrics import (
    classification_metrics,
    fake_decision_scores,
    rounded_metrics,
)
from src.models.builders import FIXED_CHAR_CONFIG, build_fixed_features


PROJECT_ROOT = Path(__file__).resolve().parents[3]

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

