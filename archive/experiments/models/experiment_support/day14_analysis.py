# Ndërton kandidatët Logistic Regression, Linear SVM dhe SGD mbi përfaqësimin TF-IDF
# tashmë të ngrirë. I mat me të njëjtat folds group-safe dhe zgjedh classifier-in
# duke kombinuar F1 me stabilitetin ndërmjet folds.

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.evaluation.data_utils import build_group_safe_folds
from src.evaluation.metrics import classification_metrics
from src.models.builders import FIXED_CHAR_CONFIG, build_fixed_features

PROJECT_ROOT = Path(__file__).resolve().parents[4]
TRAIN_PATH = PROJECT_ROOT / "data" / "interim" / "train.csv"
DAY13_SELECTION_PATH = (
    PROJECT_ROOT
    / "reports"
    / "experiment_history"
    / "10_tfidf_representations"
    / "selection.json"
)

REPORTS_DIR = (
    PROJECT_ROOT / "reports" / "experiment_history" / "11_classifier_comparison"
)
SELECTION_PATH = REPORTS_DIR / "selection.json"
METRICS_PATH = REPORTS_DIR / "metrics.json"

CLASSIFIER_DISPLAY = {
    "logistic_regression": "Logistic Regression",
    "linear_svm": "Linear SVM",
    "complement_nb": "Complement Naive Bayes",
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
    if not DAY13_SELECTION_PATH.exists():
        raise FileNotFoundError(
            f"Missing TF-IDF representation selection: {DAY13_SELECTION_PATH}"
        )
    selection = json.loads(DAY13_SELECTION_PATH.read_text(encoding="utf-8"))
    selected_config = selection.get("selected_char_config")
    if selection.get("recommended_for_day14") != "word_char_tfidf":
        raise ValueError("The selected representation is not Word + Character TF-IDF.")
    if selected_config != FIXED_CHAR_CONFIG:
        raise ValueError("The selected character configuration is not the frozen setup.")
    if selection.get("external_results_used") is not False:
        raise ValueError("The representation selection is not marked as internal-only.")
    return selected_config.copy()


def build_classifier(candidate: dict):
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
                    "training_seconds": feature_fit_seconds + classifier_fit_seconds,
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
        summary_rows.append(summary_row)

    return (
        fold_results,
        pd.DataFrame(summary_rows),
        fold_audit,
        int(len(np.unique(groups))),
    )


def select_from_cv(cv_summary: pd.DataFrame, char_config: dict) -> dict:
    ranking_columns = [
        "mean_f1_weighted",
        "std_f1_weighted",
        "mean_f1_fake",
        "mean_accuracy",
        "mean_training_seconds",
    ]
    family_best_rows = []
    for _, family in cv_summary.groupby("classifier", sort=False):
        family_best_rows.append(
            family.sort_values(
                ranking_columns,
                ascending=[False, True, False, False, True],
            ).iloc[0]
        )
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
