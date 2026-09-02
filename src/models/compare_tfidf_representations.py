"""Compare word, character, and combined TF-IDF representations."""

from __future__ import annotations

import json
import logging
import sys
import time
import unicodedata
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline

from src.evaluation.data_utils import (
    LENGTH_DISPLAY,
    LENGTH_LABELS,
    add_word_counts,
    build_leakage_safe_groups,
    exclude_train_duplicates_from_test,
)
from src.evaluation.experiment_utils import (
    dataframe_to_markdown,
    file_sha256,
    format_percent as _percent,
)
from src.features.linguistic_features import extract_linguistic_features
from src.models.analyze_length_domain_shift import (
    truncate_to_total_words,
)
from src.models.analyze_model_quality import (
    build_calibration_folds,
)
from src.models.builders import build_char_vectorizer, build_word_vectorizer
from src.models.predict import (
    DEFAULT_FAKE_THRESHOLD,
    DEFAULT_REAL_THRESHOLD,
    classify_probability,
)
from src.preprocessing.clean_text import combine_title_content


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = PROJECT_ROOT / "data" / "interim" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "interim" / "test.csv"
EXTERNAL_PATH = PROJECT_ROOT / "data" / "external" / "external_news.csv"
DAY11_METRICS_PATH = PROJECT_ROOT / "reports" / "day11_external_metrics.json"
DAY12_STABILITY_PATH = (
    PROJECT_ROOT / "reports" / "day12_internal_stability_experiment.csv"
)
CURRENT_BASELINE_PATH = PROJECT_ROOT / "models" / "calibrated_tfidf_logreg.joblib"

MODEL_PATHS = {
    "word_tfidf": PROJECT_ROOT / "models" / "day13_word_tfidf_logreg_calibrated.joblib",
    "char_tfidf": PROJECT_ROOT / "models" / "day13_char_tfidf_logreg_calibrated.joblib",
    "word_char_tfidf": (
        PROJECT_ROOT / "models" / "day13_word_char_tfidf_logreg_calibrated.joblib"
    ),
}

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
CHAR_SCREEN_PATH = REPORTS_DIR / "day13_char_config_screen.csv"
INTERNAL_COMPARISON_PATH = REPORTS_DIR / "day13_internal_model_comparison.csv"
INTERNAL_PREDICTIONS_PATH = REPORTS_DIR / "day13_internal_predictions.csv"
COHORT_METRICS_PATH = REPORTS_DIR / "day13_internal_cohort_metrics.csv"
LENGTH_METRICS_PATH = REPORTS_DIR / "day13_length_group_metrics.csv"
LENGTH_BIAS_PATH = REPORTS_DIR / "day13_length_bias_comparison.csv"
STABILITY_PATH = REPORTS_DIR / "day13_stability_experiment.csv"
STABILITY_SUMMARY_PATH = REPORTS_DIR / "day13_stability_summary.csv"
INTERNAL_SELECTION_PATH = REPORTS_DIR / "day13_internal_selection.json"
EXTERNAL_PREDICTIONS_PATH = REPORTS_DIR / "day13_external_predictions.csv"
EXTERNAL_COMPARISON_PATH = REPORTS_DIR / "day13_external_comparison.csv"
METRICS_PATH = REPORTS_DIR / "day13_metrics.json"
REPORT_PATH = REPORTS_DIR / "day13_tfidf_representation_comparison.md"

INTERNAL_FIGURE_PATH = FIGURES_DIR / "day13_internal_model_comparison.png"
COHORT_FIGURE_PATH = FIGURES_DIR / "day13_cohort_accuracy.png"
LENGTH_BIAS_FIGURE_PATH = FIGURES_DIR / "day13_length_bias.png"
STABILITY_FIGURE_PATH = FIGURES_DIR / "day13_stability.png"
EXTERNAL_FIGURE_PATH = FIGURES_DIR / "day13_external_diagnostic.png"

MODEL_NAMES = ["word_tfidf", "char_tfidf", "word_char_tfidf"]
MODEL_DISPLAY = {
    "word_tfidf": "Word TF-IDF",
    "char_tfidf": "Character TF-IDF",
    "word_char_tfidf": "Word + Character TF-IDF",
}
MODEL_COLORS = {
    "word_tfidf": "#457b9d",
    "char_tfidf": "#e76f51",
    "word_char_tfidf": "#2a9d8f",
}

CHARACTER_CONFIGS = [
    {
        "config_name": "char_wb_3_5",
        "analyzer": "char_wb",
        "ngram_min": 3,
        "ngram_max": 5,
        "min_df": 2,
        "max_features": 50000,
    },
    {
        "config_name": "char_wb_3_6",
        "analyzer": "char_wb",
        "ngram_min": 3,
        "ngram_max": 6,
        "min_df": 2,
        "max_features": 60000,
    },
]

STABILITY_VARIANTS = [
    "full",
    "short_46_words",
    "title_only",
    "without_albanian_diacritics",
    "unicode_nfc_from_nfd",
]
STABILITY_DISPLAY = {
    "full": "Teksti i plotë",
    "short_46_words": "Rreth 46 fjalë",
    "title_only": "Vetëm titulli",
    "without_albanian_diacritics": "Pa ë/ç",
    "unicode_nfc_from_nfd": "Unicode i normalizuar",
}

LOGGER = logging.getLogger(__name__)


def build_classifier() -> LogisticRegression:
    """Return the same classifier for every representation."""
    return LogisticRegression(max_iter=1000, class_weight="balanced")


def build_representation_pipeline(
    representation: str,
    char_config: dict,
) -> Pipeline:
    """Create one of the three comparable TF-IDF pipelines."""
    if representation == "word_tfidf":
        features = build_word_vectorizer()
    elif representation == "char_tfidf":
        features = build_char_vectorizer(char_config)
    elif representation == "word_char_tfidf":
        features = FeatureUnion(
            [
                ("word", build_word_vectorizer()),
                ("character", build_char_vectorizer(char_config)),
            ]
        )
    else:
        raise ValueError(f"Unknown representation: {representation}")

    return Pipeline(
        [
            ("features", features),
            ("classifier", build_classifier()),
        ]
    )


def load_internal_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load the existing split and remove exact train duplicates from test."""
    required = [TRAIN_PATH, TEST_PATH, CURRENT_BASELINE_PATH, DAY12_STABILITY_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Day 13 inputs: {missing}")

    train = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig", keep_default_na=False)
    test = pd.read_csv(TEST_PATH, encoding="utf-8-sig", keep_default_na=False)
    evaluation_test, excluded_ids = exclude_train_duplicates_from_test(train, test)
    return train.reset_index(drop=True), evaluation_test.reset_index(drop=True), excluded_ids


def build_screen_folds(train: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create three train-only group-safe folds for the small char screen."""
    groups = build_leakage_safe_groups(train)
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    folds = list(splitter.split(train["model_text"], train["label"], groups=groups))
    for fit_index, validation_index in folds:
        if set(groups[fit_index]) & set(groups[validation_index]):
            raise RuntimeError("Character screen folds contain group leakage.")
    return folds


def screen_character_configs(train: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Choose between two character settings using train data only."""
    folds = build_screen_folds(train)
    rows: list[dict] = []

    for config in CHARACTER_CONFIGS:
        fold_weighted_f1: list[float] = []
        fold_fake_f1: list[float] = []
        fold_accuracy: list[float] = []
        started = time.perf_counter()
        for fit_index, validation_index in folds:
            model = build_representation_pipeline("char_tfidf", config)
            model.fit(
                train.iloc[fit_index]["model_text"],
                train.iloc[fit_index]["label"],
            )
            y_true = train.iloc[validation_index]["label"].to_numpy()
            y_pred = model.predict(train.iloc[validation_index]["model_text"])
            weighted = precision_recall_fscore_support(
                y_true, y_pred, average="weighted", zero_division=0
            )
            fake = precision_recall_fscore_support(
                y_true,
                y_pred,
                labels=[1],
                average=None,
                zero_division=0,
            )
            fold_accuracy.append(float(accuracy_score(y_true, y_pred)))
            fold_weighted_f1.append(float(weighted[2]))
            fold_fake_f1.append(float(fake[2][0]))

        rows.append(
            {
                **config,
                "screen_scope": "train_only_3_fold_group_safe_cv",
                "mean_accuracy": round(float(np.mean(fold_accuracy)), 4),
                "std_accuracy": round(float(np.std(fold_accuracy)), 4),
                "mean_f1_weighted": round(float(np.mean(fold_weighted_f1)), 4),
                "std_f1_weighted": round(float(np.std(fold_weighted_f1)), 4),
                "mean_f1_fake": round(float(np.mean(fold_fake_f1)), 4),
                "training_seconds": round(time.perf_counter() - started, 3),
            }
        )

    screen = pd.DataFrame(rows).sort_values(
        ["mean_f1_weighted", "mean_f1_fake", "config_name"],
        ascending=[False, False, True],
    )
    selected_name = str(screen.iloc[0]["config_name"])
    selected = next(
        config for config in CHARACTER_CONFIGS if config["config_name"] == selected_name
    )
    return screen.reset_index(drop=True), selected


def train_calibrated_representation(
    train: pd.DataFrame,
    representation: str,
    char_config: dict,
    calibration_folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[CalibratedClassifierCV, float]:
    """Train one representation with identical group-safe calibration."""
    estimator = build_representation_pipeline(representation, char_config)
    model = CalibratedClassifierCV(
        estimator=estimator,
        method="sigmoid",
        cv=calibration_folds,
        ensemble=False,
    )
    started = time.perf_counter()
    model.fit(train["model_text"], train["label"])
    return model, time.perf_counter() - started


def probability_arrays(model, texts: list[str] | pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Return real and fake probabilities using class labels, not positions."""
    probabilities = model.predict_proba(texts)
    classes = list(model.classes_)
    return probabilities[:, classes.index(0)], probabilities[:, classes.index(1)]


def prediction_table(
    base_data: pd.DataFrame,
    model,
    model_name: str,
    id_column: str,
) -> pd.DataFrame:
    """Create a common prediction table for internal or external data."""
    model_texts = [
        combine_title_content(row.title, row.content)
        for row in base_data.itertuples(index=False)
    ]
    probability_real, probability_fake = probability_arrays(model, model_texts)
    table = base_data.copy().reset_index(drop=True)
    table.insert(0, "model", model_name)
    table["probability_real"] = probability_real
    table["probability_fake"] = probability_fake
    table["binary_prediction"] = (probability_fake >= 0.5).astype(int)
    table["decision"] = [
        classify_probability(float(value)) for value in probability_fake
    ]
    table["prediction_correct"] = table["label"].eq(table["binary_prediction"])
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


def calculate_metrics(table: pd.DataFrame) -> dict:
    """Calculate binary, probability, and three-level metrics."""
    y_true = table["label"].to_numpy(dtype=int)
    y_pred = table["binary_prediction"].to_numpy(dtype=int)
    probability_fake = table["probability_fake"].to_numpy(dtype=float)
    probabilities = np.column_stack(
        [table["probability_real"].to_numpy(dtype=float), probability_fake]
    )
    weighted = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    per_class = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    strong = table["decision"].ne("uncertain")
    strong_correct = (
        (table["label"].eq(0) & table["decision"].eq("likely_real"))
        | (table["label"].eq(1) & table["decision"].eq("likely_fake"))
    )
    return {
        "rows": int(len(table)),
        "real_rows": int(table["label"].eq(0).sum()),
        "fake_rows": int(table["label"].eq(1).sum()),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_weighted": round(float(weighted[0]), 4),
        "recall_weighted": round(float(weighted[1]), 4),
        "f1_weighted": round(float(weighted[2]), 4),
        "precision_real": round(float(per_class[0][0]), 4),
        "recall_real": round(float(per_class[1][0]), 4),
        "f1_real": round(float(per_class[2][0]), 4),
        "precision_fake": round(float(per_class[0][1]), 4),
        "recall_fake": round(float(per_class[1][1]), 4),
        "f1_fake": round(float(per_class[2][1]), 4),
        "brier_score": round(float(brier_score_loss(y_true, probability_fake)), 6),
        "log_loss": round(float(log_loss(y_true, probabilities, labels=[0, 1])), 6),
        "confusion_matrix": matrix.tolist(),
        "false_positives": int(matrix[0, 1]),
        "false_negatives": int(matrix[1, 0]),
        "likely_real": int(table["decision"].eq("likely_real").sum()),
        "uncertain": int(table["decision"].eq("uncertain").sum()),
        "likely_fake": int(table["decision"].eq("likely_fake").sum()),
        "strong_coverage": round(float(strong.mean()), 4),
        "strong_accuracy": round(
            float(strong_correct[strong].mean()) if strong.any() else 0.0,
            4,
        ),
        "mean_probability_fake_real": (
            round(float(table.loc[table["label"].eq(0), "probability_fake"].mean()), 4)
            if table["label"].eq(0).any()
            else None
        ),
        "mean_probability_fake_fake": (
            round(float(table.loc[table["label"].eq(1), "probability_fake"].mean()), 4)
            if table["label"].eq(1).any()
            else None
        ),
    }


def build_internal_cohort_metrics(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate requested internal cohorts and every Day 12 length group."""
    rows: list[dict] = []
    for model_name, model_table in predictions.groupby("model", sort=False):
        cohorts = {
            "all_internal_test": model_table,
            "internal_30_60": model_table.loc[
                model_table["word_count"].between(30, 60)
            ],
            "short_real_30_60": model_table.loc[
                model_table["label"].eq(0)
                & model_table["word_count"].between(30, 60)
            ],
            "long_fake_gt_250": model_table.loc[
                model_table["label"].eq(1) & model_table["word_count"].gt(250)
            ],
        }
        for length_group in LENGTH_LABELS:
            cohorts[f"length_{length_group}"] = model_table.loc[
                model_table["length_group"].astype(str).eq(length_group)
            ]

        for cohort_name, cohort in cohorts.items():
            metrics = calculate_metrics(cohort)
            rows.append(
                {
                    "model": model_name,
                    "model_display": MODEL_DISPLAY[model_name],
                    "cohort": cohort_name,
                    **metrics,
                    "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
                }
            )
    cohort_metrics = pd.DataFrame(rows)
    length_metrics = cohort_metrics.loc[
        cohort_metrics["cohort"].str.startswith("length_")
    ].copy()
    length_metrics["length_group"] = length_metrics["cohort"].str.replace(
        "length_", "", regex=False
    )
    length_metrics["length_description"] = length_metrics["length_group"].map(
        LENGTH_DISPLAY
    )
    return cohort_metrics, length_metrics


def build_length_bias_comparison(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize correlation and direction of length bias for each model."""
    rows: list[dict] = []
    for model_name, table in predictions.groupby("model", sort=False):
        correlations = {}
        for scope, group in (
            ("all", table),
            ("real", table.loc[table["label"].eq(0)]),
            ("fake", table.loc[table["label"].eq(1)]),
        ):
            result = spearmanr(group["word_count"], group["probability_fake"])
            correlations[scope] = (float(result.statistic), float(result.pvalue))

        real_short = table.loc[
            table["label"].eq(0) & table["word_count"].between(30, 60)
        ]
        real_long = table.loc[table["label"].eq(0) & table["word_count"].gt(250)]
        fake_short = table.loc[
            table["label"].eq(1) & table["word_count"].between(30, 60)
        ]
        fake_long = table.loc[table["label"].eq(1) & table["word_count"].gt(250)]
        rows.append(
            {
                "model": model_name,
                "model_display": MODEL_DISPLAY[model_name],
                "spearman_all": round(correlations["all"][0], 4),
                "spearman_real": round(correlations["real"][0], 4),
                "spearman_fake": round(correlations["fake"][0], 4),
                "spearman_all_p": correlations["all"][1],
                "real_short_mean_probability_fake": round(
                    float(real_short["probability_fake"].mean()), 4
                ),
                "real_long_mean_probability_fake": round(
                    float(real_long["probability_fake"].mean()), 4
                ),
                "real_probability_gap_short_minus_long": round(
                    float(
                        real_short["probability_fake"].mean()
                        - real_long["probability_fake"].mean()
                    ),
                    4,
                ),
                "fake_short_mean_probability_fake": round(
                    float(fake_short["probability_fake"].mean()), 4
                ),
                "fake_long_mean_probability_fake": round(
                    float(fake_long["probability_fake"].mean()), 4
                ),
                "fake_probability_gap_short_minus_long": round(
                    float(
                        fake_short["probability_fake"].mean()
                        - fake_long["probability_fake"].mean()
                    ),
                    4,
                ),
                "short_real_accuracy": round(
                    float(real_short["prediction_correct"].mean()), 4
                ),
                "long_fake_accuracy": round(
                    float(fake_long["prediction_correct"].mean()), 4
                ),
            }
        )
    return pd.DataFrame(rows)


def remove_albanian_diacritics(text: str) -> str:
    """Remove only Albanian ë/ç while preserving all other characters."""
    return str(text).translate(str.maketrans({"ë": "e", "Ë": "E", "ç": "c", "Ç": "C"}))


def stability_case_ids() -> list[str]:
    """Reuse the eight internally selected diagnostic cases from Day 12."""
    day12 = pd.read_csv(DAY12_STABILITY_PATH, encoding="utf-8", keep_default_na=False)
    full = day12.loc[day12["variant"].eq("full")]
    ids = full["article_id"].drop_duplicates().tolist()
    if len(ids) != 8:
        raise ValueError(f"Expected 8 Day 12 stability IDs, found {len(ids)}")
    return ids


def stability_variants(title: str, content: str) -> dict[str, tuple[str, str]]:
    """Create the five requested diagnostic input variants."""
    return {
        "full": (title, content),
        "short_46_words": (title, truncate_to_total_words(title, content, 46)),
        "title_only": (title, ""),
        "without_albanian_diacritics": (
            remove_albanian_diacritics(title),
            remove_albanian_diacritics(content),
        ),
        "unicode_nfc_from_nfd": (
            unicodedata.normalize("NFD", title),
            unicodedata.normalize("NFD", content),
        ),
    }


def run_stability_experiment(
    test: pd.DataFrame,
    models: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare model probabilities across controlled text variants."""
    selected = test.set_index("article_id").loc[stability_case_ids()].reset_index()
    rows: list[dict] = []

    for article in selected.itertuples(index=False):
        for variant, (variant_title, variant_content) in stability_variants(
            article.title, article.content
        ).items():
            model_text = combine_title_content(variant_title, variant_content)
            word_count = extract_linguistic_features(
                variant_title, variant_content
            )["word_count"]
            for model_name, model in models.items():
                probability_real, probability_fake = probability_arrays(
                    model, [model_text]
                )
                fake_value = float(probability_fake[0])
                rows.append(
                    {
                        "model": model_name,
                        "model_display": MODEL_DISPLAY[model_name],
                        "article_id": article.article_id,
                        "true_label": "real" if int(article.label) == 0 else "fake",
                        "variant": variant,
                        "variant_description": STABILITY_DISPLAY[variant],
                        "word_count": int(word_count),
                        "model_text": model_text,
                        "probability_real": float(probability_real[0]),
                        "probability_fake": fake_value,
                        "binary_prediction": "fake" if fake_value >= 0.5 else "real",
                        "decision": classify_probability(fake_value),
                    }
                )

    results = pd.DataFrame(rows)
    full = results.loc[results["variant"].eq("full")].set_index(
        ["model", "article_id"]
    )
    keys = list(zip(results["model"], results["article_id"]))
    results["full_probability_fake"] = [
        float(full.loc[key, "probability_fake"]) for key in keys
    ]
    results["full_binary_prediction"] = [
        str(full.loc[key, "binary_prediction"]) for key in keys
    ]
    results["full_decision"] = [str(full.loc[key, "decision"]) for key in keys]
    results["delta_probability_fake_from_full"] = (
        results["probability_fake"] - results["full_probability_fake"]
    )
    results["absolute_delta_from_full"] = results[
        "delta_probability_fake_from_full"
    ].abs()
    results["binary_changed_from_full"] = results["binary_prediction"].ne(
        results["full_binary_prediction"]
    )
    results["decision_changed_from_full"] = results["decision"].ne(
        results["full_decision"]
    )
    results["prediction_correct"] = results["true_label"].eq(
        results["binary_prediction"]
    )

    summary_rows: list[dict] = []
    for (model_name, variant), group in results.groupby(
        ["model", "variant"], sort=False
    ):
        summary_rows.append(
            {
                "model": model_name,
                "model_display": MODEL_DISPLAY[model_name],
                "variant": variant,
                "variant_description": STABILITY_DISPLAY[variant],
                "rows": int(len(group)),
                "mean_word_count": round(float(group["word_count"].mean()), 2),
                "mean_probability_fake": round(
                    float(group["probability_fake"].mean()), 4
                ),
                "mean_absolute_delta_from_full": round(
                    float(group["absolute_delta_from_full"].mean()), 6
                ),
                "max_absolute_delta_from_full": round(
                    float(group["absolute_delta_from_full"].max()), 6
                ),
                "binary_accuracy": round(
                    float(group["prediction_correct"].mean()), 4
                ),
                "binary_changes_from_full": int(
                    group["binary_changed_from_full"].sum()
                ),
                "decision_changes_from_full": int(
                    group["decision_changed_from_full"].sum()
                ),
            }
        )
    return results, pd.DataFrame(summary_rows)


def internal_selection(
    comparison: pd.DataFrame,
    cohort_metrics: pd.DataFrame,
    length_bias: pd.DataFrame,
    stability_summary: pd.DataFrame,
    selected_char_config: dict,
) -> dict:
    """Lock recommendations using internal results only."""
    best_internal = comparison.sort_values(
        ["f1_weighted", "f1_fake", "accuracy", "brier_score"],
        ascending=[False, False, False, True],
    ).iloc[0]
    short = cohort_metrics.loc[cohort_metrics["cohort"].eq("internal_30_60")]
    short_stability = stability_summary.loc[
        stability_summary["variant"].eq("short_46_words")
    ][["model", "mean_absolute_delta_from_full"]]
    short_rank = short.merge(short_stability, on="model").sort_values(
        ["accuracy", "f1_weighted", "mean_absolute_delta_from_full"],
        ascending=[False, False, True],
    )
    bias_rank = length_bias.assign(
        mean_absolute_within_label_rho=lambda frame: (
            frame["spearman_real"].abs() + frame["spearman_fake"].abs()
        )
        / 2
    ).sort_values(
        ["mean_absolute_within_label_rho", "short_real_accuracy", "long_fake_accuracy"],
        ascending=[True, False, False],
    )
    return {
        "selection_scope": "internal_only_before_external_load",
        "selected_char_config": selected_char_config,
        "best_internal_model": str(best_internal["model"]),
        "best_internal_f1_weighted": float(best_internal["f1_weighted"]),
        "best_short_cohort_model": str(short_rank.iloc[0]["model"]),
        "lowest_length_association_model": str(bias_rank.iloc[0]["model"]),
        "recommended_for_day14": str(best_internal["model"]),
        "primary_rule": "highest internal F1 weighted, then F1 fake, accuracy, Brier",
        "external_results_used": False,
    }


def load_external_after_selection(selection_hash: str) -> pd.DataFrame:
    """Load external data only after the internal selection artifact exists."""
    if not INTERNAL_SELECTION_PATH.exists():
        raise RuntimeError("Internal selection must be saved before external loading.")
    if file_sha256(INTERNAL_SELECTION_PATH) != selection_hash:
        raise RuntimeError("Internal selection changed before external evaluation.")
    external = pd.read_csv(EXTERNAL_PATH, encoding="utf-8", keep_default_na=False)
    if len(external) != 40 or set(external["label"]) != {"real", "fake"}:
        raise ValueError("Frozen external benchmark is not the expected 40-row dataset.")
    external["label"] = external["label"].map({"real": 0, "fake": 1})
    return add_word_counts(external)


def plot_internal_comparison(comparison: pd.DataFrame) -> None:
    """Plot the main internal metrics."""
    metrics = ["accuracy", "f1_weighted", "f1_fake"]
    x = np.arange(len(metrics))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 6))
    for index, model_name in enumerate(MODEL_NAMES):
        row = comparison.set_index("model").loc[model_name]
        ax.bar(
            x + (index - 1) * width,
            [row[metric] for metric in metrics],
            width,
            label=MODEL_DISPLAY[model_name],
            color=MODEL_COLORS[model_name],
        )
    ax.set_xticks(x, ["Accuracy", "F1 weighted", "F1 fake"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rezultati")
    ax.set_title("Krahasimi i përfaqësimeve në test set-in e brendshëm")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(INTERNAL_FIGURE_PATH, dpi=180)
    plt.close(fig)


def plot_cohort_accuracy(cohort_metrics: pd.DataFrame) -> None:
    """Plot accuracy for the requested diagnostic cohorts."""
    cohorts = [
        "all_internal_test",
        "internal_30_60",
        "short_real_30_60",
        "long_fake_gt_250",
    ]
    labels = ["Të gjithë", "30-60 fjalë", "Real 30-60", "Fake >250"]
    x = np.arange(len(cohorts))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10.5, 6))
    indexed = cohort_metrics.set_index(["model", "cohort"])
    for index, model_name in enumerate(MODEL_NAMES):
        values = [indexed.loc[(model_name, cohort), "accuracy"] for cohort in cohorts]
        ax.bar(
            x + (index - 1) * width,
            values,
            width,
            label=MODEL_DISPLAY[model_name],
            color=MODEL_COLORS[model_name],
        )
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Performanca në cohort-et e gjatësisë")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(COHORT_FIGURE_PATH, dpi=180)
    plt.close(fig)


def plot_length_bias(length_metrics: pd.DataFrame) -> None:
    """Plot mean fake probability by length and true label."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    for axis, label_number, label_name in zip(axes, (0, 1), ("Real", "Fake")):
        for model_name in MODEL_NAMES:
            rows = []
            for length_group in LENGTH_LABELS:
                cohort = f"length_{length_group}"
                metric_row = length_metrics.loc[
                    length_metrics["model"].eq(model_name)
                    & length_metrics["cohort"].eq(cohort)
                ].iloc[0]
                probability_column = (
                    "mean_probability_fake_real"
                    if label_number == 0
                    else "mean_probability_fake_fake"
                )
                rows.append(float(metric_row[probability_column]))
            axis.plot(
                np.arange(len(LENGTH_LABELS)),
                rows,
                marker="o",
                label=MODEL_DISPLAY[model_name],
                color=MODEL_COLORS[model_name],
            )
        axis.set_xticks(
            np.arange(len(LENGTH_LABELS)),
            ["<=60", "61-120", "121-250", ">250"],
        )
        axis.set_title(f"Label real: {label_name}")
        axis.set_xlabel("Numri i fjalëve")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Probability fake mesatare")
    axes[0].set_ylim(0, 1)
    axes[1].legend()
    fig.suptitle("Bias-i i gjatësisë sipas përfaqësimit")
    fig.tight_layout()
    fig.savefig(LENGTH_BIAS_FIGURE_PATH, dpi=180)
    plt.close(fig)


def plot_stability(stability_summary: pd.DataFrame) -> None:
    """Plot mean probability changes for each diagnostic variant."""
    variants = [
        "short_46_words",
        "title_only",
        "without_albanian_diacritics",
        "unicode_nfc_from_nfd",
    ]
    x = np.arange(len(variants))
    width = 0.24
    indexed = stability_summary.set_index(["model", "variant"])
    fig, ax = plt.subplots(figsize=(11, 6))
    for index, model_name in enumerate(MODEL_NAMES):
        values = [
            indexed.loc[(model_name, variant), "mean_absolute_delta_from_full"]
            for variant in variants
        ]
        ax.bar(
            x + (index - 1) * width,
            values,
            width,
            label=MODEL_DISPLAY[model_name],
            color=MODEL_COLORS[model_name],
        )
    ax.set_xticks(x, [STABILITY_DISPLAY[variant] for variant in variants])
    ax.set_ylabel("Ndryshimi absolut mesatar i probability fake")
    ax.set_title("Stabiliteti ndaj varianteve të tekstit")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(STABILITY_FIGURE_PATH, dpi=180)
    plt.close(fig)


def plot_external_comparison(comparison: pd.DataFrame) -> None:
    """Plot external diagnostic metrics and decision counts."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    x = np.arange(3)
    width = 0.24
    indexed = comparison.set_index("model")
    for index, model_name in enumerate(MODEL_NAMES):
        row = indexed.loc[model_name]
        axes[0].bar(
            x + (index - 1) * width,
            [row["accuracy"], row["recall_real"], row["recall_fake"]],
            width,
            label=MODEL_DISPLAY[model_name],
            color=MODEL_COLORS[model_name],
        )
    axes[0].set_xticks(x, ["Accuracy", "Recall real", "Recall fake"])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Metrikat e jashtme")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.2)

    bottom = np.zeros(len(MODEL_NAMES))
    colors = ["#2a9d8f", "#f4a261", "#e76f51"]
    for decision, color in zip(
        ("likely_real", "uncertain", "likely_fake"), colors
    ):
        values = comparison.set_index("model").loc[MODEL_NAMES, decision].to_numpy()
        axes[1].bar(
            [MODEL_DISPLAY[name] for name in MODEL_NAMES],
            values,
            bottom=bottom,
            label=decision,
            color=color,
        )
        bottom += values
    axes[1].set_title("Vendimet me pragjet 0.30/0.70")
    axes[1].set_ylabel("Numri i rasteve")
    axes[1].tick_params(axis="x", rotation=12)
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(EXTERNAL_FIGURE_PATH, dpi=180)
    plt.close(fig)


def write_report(
    metrics: dict,
    char_screen: pd.DataFrame,
    internal_comparison: pd.DataFrame,
    cohort_metrics: pd.DataFrame,
    length_bias: pd.DataFrame,
    stability_summary: pd.DataFrame,
    external_comparison: pd.DataFrame,
) -> None:
    """Write the Day 13 report in Albanian."""
    screen_table = dataframe_to_markdown(
        char_screen,
        [
            "config_name",
            "ngram_min",
            "ngram_max",
            "max_features",
            "mean_accuracy",
            "mean_f1_weighted",
            "mean_f1_fake",
            "training_seconds",
        ],
    )
    internal_table = dataframe_to_markdown(
        internal_comparison,
        [
            "model_display",
            "accuracy",
            "precision_weighted",
            "recall_weighted",
            "f1_weighted",
            "f1_fake",
            "false_positives",
            "false_negatives",
            "brier_score",
            "log_loss",
            "training_seconds",
            "model_size_mb",
        ],
    )
    requested_cohorts = cohort_metrics.loc[
        cohort_metrics["cohort"].isin(
            [
                "internal_30_60",
                "short_real_30_60",
                "long_fake_gt_250",
            ]
        )
    ]
    cohort_table = dataframe_to_markdown(
        requested_cohorts,
        [
            "model_display",
            "cohort",
            "rows",
            "accuracy",
            "f1_weighted",
            "recall_real",
            "recall_fake",
            "mean_probability_fake_real",
            "mean_probability_fake_fake",
        ],
    )
    bias_table = dataframe_to_markdown(
        length_bias,
        [
            "model_display",
            "spearman_all",
            "spearman_real",
            "spearman_fake",
            "real_probability_gap_short_minus_long",
            "fake_probability_gap_short_minus_long",
            "short_real_accuracy",
            "long_fake_accuracy",
        ],
    )
    stability_table = dataframe_to_markdown(
        stability_summary.loc[stability_summary["variant"].ne("full")],
        [
            "model_display",
            "variant_description",
            "mean_absolute_delta_from_full",
            "max_absolute_delta_from_full",
            "binary_changes_from_full",
            "decision_changes_from_full",
        ],
    )
    external_table = dataframe_to_markdown(
        external_comparison,
        [
            "model_display",
            "accuracy",
            "recall_real",
            "recall_fake",
            "false_positives",
            "false_negatives",
            "likely_real",
            "uncertain",
            "likely_fake",
            "mean_probability_fake_real",
            "mean_probability_fake_fake",
        ],
    )

    selection = metrics["internal_selection"]
    best_internal = selection["best_internal_model"]
    best_short = selection["best_short_cohort_model"]
    lowest_bias = selection["lowest_length_association_model"]
    internal_index = internal_comparison.set_index("model")
    bias_index = length_bias.set_index("model")
    stability_index = stability_summary.set_index(["model", "variant"])
    external_index = external_comparison.set_index("model")
    word_external = external_index.loc["word_tfidf"]
    char_external = external_index.loc["char_tfidf"]
    combined_external = external_index.loc["word_char_tfidf"]
    char_external_difference = (
        float(char_external["accuracy"]) - float(word_external["accuracy"])
    )
    combined_external_difference = (
        float(combined_external["accuracy"]) - float(word_external["accuracy"])
    )

    report = f"""# Dita 13 - Krahasimi i përfaqësimeve TF-IDF

## Protokolli

U përdorën të njëjtat `train.csv`/`test.csv`, preprocessing bazë dhe përjashtimi
i 7 dublikatave ekzakte train/test. Të tre variantet përdorën të njëjtën
`LogisticRegression(max_iter=1000, class_weight='balanced')` dhe sigmoid
calibration me 5 fold-e group-safe. Pragjet mbetën 0.30/0.70.

Dataset-i i jashtëm nuk u lexua gjatë character screen, trajnimit, vlerësimit
të brendshëm ose përzgjedhjes. Përzgjedhja u ruajt fillimisht te
`reports/day13_internal_selection.json`; vetëm pas kësaj u hap benchmark-u i
jashtëm. Modeli aktual i aplikacionit dhe dataset-i i jashtëm mbetën të
pandryshuar.

## Character screen vetëm mbi train

U provuan vetëm dy konfigurime të paracaktuara, jo një grid search i madh.

{screen_table}

U zgjodh `{metrics['selected_char_config']['config_name']}` sipas F1 weighted
mesatare në 3-fold group-safe CV. I njëjti konfigurim u përdor te Character dhe
Word + Character.

## Rezultatet e brendshme

{internal_table}

Confusion matrices në rendin `[real, fake]`:

- Word: `{internal_index.loc['word_tfidf', 'confusion_matrix']}`;
- Character: `{internal_index.loc['char_tfidf', 'confusion_matrix']}`;
- Word + Character: `{internal_index.loc['word_char_tfidf', 'confusion_matrix']}`.

Sipas rregullit të ngrirë, varianti më i mirë i brendshëm ishte
**{MODEL_DISPLAY[best_internal]}** me F1 weighted
{_percent(float(internal_index.loc[best_internal, 'f1_weighted']))}. Përzgjedhja
nuk varet nga rezultatet e jashtme.

![Krahasimi i brendshëm](figures/day13_internal_model_comparison.png)

## Cohort-et problematike

{cohort_table}

Cohort-i 30-60 ka vetëm 9 raste, prej të cilave 6 real. Fake mbi 250 fjalë ka
29 raste. Këto rezultate janë diagnostike dhe duhen interpretuar bashkë me
madhësinë e kampionit. Varianti me renditjen më të mirë për cohort-in e shkurtër
ishte **{MODEL_DISPLAY[best_short]}**.

![Accuracy sipas cohort-it](figures/day13_cohort_accuracy.png)

## Bias-i i gjatësisë

{bias_table}

Vlera më afër zeros për lidhjen mes gjatësisë dhe probability fake u arrit nga
**{MODEL_DISPLAY[lowest_bias]}**. Për krahasim, hendeku real short-minus-long
ishte {bias_index.loc['word_tfidf', 'real_probability_gap_short_minus_long']:.4f}
te Word, {bias_index.loc['char_tfidf', 'real_probability_gap_short_minus_long']:.4f}
te Character dhe
{bias_index.loc['word_char_tfidf', 'real_probability_gap_short_minus_long']:.4f}
te kombinimi. Character n-grams
{'e ulën' if abs(bias_index.loc['char_tfidf', 'spearman_real']) < abs(bias_index.loc['word_tfidf', 'spearman_real']) else 'nuk e ulën'}
lidhjen e gjatësisë te lajmet real krahasuar me Word TF-IDF.

![Bias-i i gjatësisë](figures/day13_length_bias.png)

## Stabiliteti i tekstit

U ripërdorën të njëjtat 8 raste të brendshme të Ditës 12. Varianti Unicode u
krijua fillimisht në NFD dhe kaloi në të njëjtin NFC preprocessing; prandaj
duhet të japë rezultat identik me tekstin e plotë.

{stability_table}

Për versionin 46 fjalë, ndryshimi absolut mesatar ishte
{stability_index.loc[('word_tfidf', 'short_46_words'), 'mean_absolute_delta_from_full']:.4f}
te Word,
{stability_index.loc[('char_tfidf', 'short_46_words'), 'mean_absolute_delta_from_full']:.4f}
te Character dhe
{stability_index.loc[('word_char_tfidf', 'short_46_words'), 'mean_absolute_delta_from_full']:.4f}
te kombinimi. Për Unicode të normalizuar, ndryshimi maksimal ishte
{stability_summary.loc[stability_summary['variant'].eq('unicode_nfc_from_nfd'), 'max_absolute_delta_from_full'].max():.8f}.

Character arriti accuracy më të mirë në 9 tekstet natyrshëm 30-60 fjalë, por
ndryshimi i tij mesatar pas shkurtimit artificial ishte më i madh se te Word.
Pra character n-grams nuk dhanë stabilitet uniform. Heqja e `ë/ç` shkaktoi
ndryshime të mëdha te të tre modelet, ndërsa normalizimi Unicode ishte plotësisht
stabil.

![Stabiliteti](figures/day13_stability.png)

## Vlerësimi i jashtëm vetëm diagnostik

Këto rezultate u llogaritën pasi përzgjedhja e brendshme ishte shkruar dhe
hash-i i saj ishte ngrirë. Ato nuk ndryshuan konfigurimet ose rekomandimin.

{external_table}

Confusion matrices:

- Word: `{external_index.loc['word_tfidf', 'confusion_matrix']}`;
- Character: `{external_index.loc['char_tfidf', 'confusion_matrix']}`;
- Word + Character: `{external_index.loc['word_char_tfidf', 'confusion_matrix']}`.

Character ndryshoi accuracy e jashtme me
{100 * char_external_difference:+.2f} pikë përqindjeje dhe kombinimi me
{100 * combined_external_difference:+.2f} pikë përqindjeje kundrejt Word.
Character-only ishte më i miri në këtë benchmark, ndërsa kombinimi dha vetëm
përmirësim të pjesshëm të generalizimit. Ky është vetëm vëzhgim diagnostik dhe
nuk përdoret për model selection.

![Vlerësimi i jashtëm](figures/day13_external_diagnostic.png)

## Përfundimi

- Varianti më i mirë në vlerësimin e brendshëm ishte
  **{MODEL_DISPLAY[best_internal]}**.
- Varianti më i mirë në cohort-in 30-60 ishte **{MODEL_DISPLAY[best_short]}**.
- Lidhjen më të ulët me gjatësinë e pati **{MODEL_DISPLAY[lowest_bias]}**.
- Character n-grams
  {'e përmirësuan' if float(internal_index.loc['char_tfidf', 'f1_weighted']) > float(internal_index.loc['word_tfidf', 'f1_weighted']) else 'nuk e përmirësuan'}
  F1 weighted kundrejt Word baseline.
- Character uli bias-in te real të shkurtër, por jo te fake të gjatë dhe nuk
  ishte më stabil ndaj çdo transformimi diagnostik.
- Word + Character
  {'e përmirësoi' if float(internal_index.loc['word_char_tfidf', 'f1_weighted']) > float(internal_index.loc['word_tfidf', 'f1_weighted']) else 'nuk e përmirësoi'}
  rezultatin e brendshëm kundrejt Word baseline.
- Jashtë corpus-it, Character-only përgjithësoi më mirë se kombinimi; ky rezultat
  nuk ndryshon rekomandimin e ngrirë nga vlerësimi i brendshëm.

Për Ditën 14 rekomandohet **{MODEL_DISPLAY[selection['recommended_for_day14']]}**
si përfaqësim i ngrirë për krahasimin e classifier-ëve. Modelet e tjera ruhen si
eksperimente dhe asnjëri nuk integrohet ende në Streamlit.

## Kufizimet

- Character screen kishte vetëm dy konfigurime dhe u bë vetëm mbi train.
- Cohort-i 30-60 ka 9 raste dhe nuk jep interval të ngushtë besimi.
- Stabiliteti përdor 8 raste të zgjedhura diagnostike, jo gjithë test set-in.
- Të tre modelet mësojnë nga i njëjti corpus ku gjatësia dhe label-i janë të
  lidhura; përfaqësimi i ri nuk e heq automatikisht këtë bias.
- Dataset-i i jashtëm ka përmbledhje manuale dhe source-label confounding;
  rezultatet e tij nuk janë tuning set.
- Përzgjedhja e classifier-it në Ditën 14 duhet bërë me CV mbi train, duke
  ruajtur test set-in dhe benchmark-un e jashtëm për vlerësim.

## Modelet eksperimentale

```text
models/day13_word_tfidf_logreg_calibrated.joblib
models/day13_char_tfidf_logreg_calibrated.joblib
models/day13_word_char_tfidf_logreg_calibrated.joblib
```

Modeli i aplikacionit `models/calibrated_tfidf_logreg.joblib` nuk u zëvendësua.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def run_day13_comparison() -> dict:
    """Run the complete comparison with an internal-first evaluation order."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    baseline_hash_before = file_sha256(CURRENT_BASELINE_PATH)

    train, test, excluded_ids = load_internal_data()
    test = add_word_counts(test)
    char_screen, selected_char_config = screen_character_configs(train)
    char_screen.to_csv(CHAR_SCREEN_PATH, index=False, encoding="utf-8")

    calibration_folds, calibration_group_count = build_calibration_folds(train)
    models: dict[str, object] = {}
    training_metadata: dict[str, dict] = {}
    internal_tables: list[pd.DataFrame] = []

    for model_name in MODEL_NAMES:
        LOGGER.info("Training %s", MODEL_DISPLAY[model_name])
        model, training_seconds = train_calibrated_representation(
            train,
            model_name,
            selected_char_config,
            calibration_folds,
        )
        joblib.dump(model, MODEL_PATHS[model_name], compress=3)
        model_size_bytes = MODEL_PATHS[model_name].stat().st_size
        models[model_name] = model
        training_metadata[model_name] = {
            "training_seconds": round(training_seconds, 3),
            "model_size_bytes": model_size_bytes,
            "model_size_mb": round(model_size_bytes / (1024 * 1024), 3),
            "model_path": str(MODEL_PATHS[model_name].relative_to(PROJECT_ROOT)),
            "model_sha256": file_sha256(MODEL_PATHS[model_name]),
        }
        internal_tables.append(
            prediction_table(test, model, model_name, id_column="article_id")
        )

    internal_predictions = pd.concat(internal_tables, ignore_index=True)
    cohort_metrics, length_metrics = build_internal_cohort_metrics(
        internal_predictions
    )
    internal_comparison = cohort_metrics.loc[
        cohort_metrics["cohort"].eq("all_internal_test")
    ].copy()
    for model_name, metadata in training_metadata.items():
        mask = internal_comparison["model"].eq(model_name)
        internal_comparison.loc[mask, "training_seconds"] = metadata[
            "training_seconds"
        ]
        internal_comparison.loc[mask, "model_size_mb"] = metadata["model_size_mb"]

    length_bias = build_length_bias_comparison(internal_predictions)
    stability, stability_summary = run_stability_experiment(test, models)
    selection = internal_selection(
        internal_comparison,
        cohort_metrics,
        length_bias,
        stability_summary,
        selected_char_config,
    )

    current_baseline = joblib.load(CURRENT_BASELINE_PATH)
    current_model_texts = [
        combine_title_content(row.title, row.content)
        for row in test.itertuples(index=False)
    ]
    current_probability_fake = probability_arrays(
        current_baseline, current_model_texts
    )[1]
    day13_word_probability_fake = internal_predictions.loc[
        internal_predictions["model"].eq("word_tfidf"), "probability_fake"
    ].to_numpy()
    selection["word_vs_current_baseline_max_probability_difference"] = float(
        np.max(np.abs(current_probability_fake - day13_word_probability_fake))
    )
    INTERNAL_SELECTION_PATH.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    selection_hash = file_sha256(INTERNAL_SELECTION_PATH)

    internal_output_columns = [
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
        "decision",
        "prediction_correct",
        "error_type",
    ]
    internal_predictions[internal_output_columns].to_csv(
        INTERNAL_PREDICTIONS_PATH, index=False, encoding="utf-8"
    )
    internal_comparison.to_csv(
        INTERNAL_COMPARISON_PATH, index=False, encoding="utf-8"
    )
    cohort_metrics.to_csv(COHORT_METRICS_PATH, index=False, encoding="utf-8")
    length_metrics.to_csv(LENGTH_METRICS_PATH, index=False, encoding="utf-8")
    length_bias.to_csv(LENGTH_BIAS_PATH, index=False, encoding="utf-8")
    stability.to_csv(STABILITY_PATH, index=False, encoding="utf-8")
    stability_summary.to_csv(
        STABILITY_SUMMARY_PATH, index=False, encoding="utf-8"
    )

    plot_internal_comparison(internal_comparison)
    plot_cohort_accuracy(cohort_metrics)
    plot_length_bias(length_metrics)
    plot_stability(stability_summary)

    # External data is deliberately loaded only after the internal selection is locked.
    external_hash_before = file_sha256(EXTERNAL_PATH)
    external = load_external_after_selection(selection_hash)
    external_tables = [
        prediction_table(external, models[model_name], model_name, "external_id")
        for model_name in MODEL_NAMES
    ]
    external_predictions = pd.concat(external_tables, ignore_index=True)
    external_rows: list[dict] = []
    for model_name, table in external_predictions.groupby("model", sort=False):
        model_metrics = calculate_metrics(table)
        external_rows.append(
            {
                "model": model_name,
                "model_display": MODEL_DISPLAY[model_name],
                **model_metrics,
                "confusion_matrix": json.dumps(model_metrics["confusion_matrix"]),
            }
        )
    external_comparison = pd.DataFrame(external_rows)
    external_output_columns = [
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
        "decision",
        "prediction_correct",
        "error_type",
    ]
    external_predictions[external_output_columns].to_csv(
        EXTERNAL_PREDICTIONS_PATH, index=False, encoding="utf-8"
    )
    external_comparison.to_csv(
        EXTERNAL_COMPARISON_PATH, index=False, encoding="utf-8"
    )
    plot_external_comparison(external_comparison)

    if file_sha256(INTERNAL_SELECTION_PATH) != selection_hash:
        raise RuntimeError("Internal selection changed during external diagnostics.")
    baseline_hash_after = file_sha256(CURRENT_BASELINE_PATH)
    external_hash_after = file_sha256(EXTERNAL_PATH)
    if baseline_hash_before != baseline_hash_after:
        raise RuntimeError("Current application baseline changed during Day 13.")
    if external_hash_before != external_hash_after:
        raise RuntimeError("External benchmark changed during Day 13.")

    metrics = {
        "status": "completed",
        "protocol": {
            "external_used_for_tuning": False,
            "selection_locked_before_external": True,
            "same_train_test_split": True,
            "exact_train_duplicates_excluded_from_test": len(excluded_ids),
            "calibration_method": "sigmoid",
            "calibration_folds": len(calibration_folds),
            "calibration_group_count": calibration_group_count,
            "thresholds": {
                "likely_real_below": DEFAULT_REAL_THRESHOLD,
                "likely_fake_above": DEFAULT_FAKE_THRESHOLD,
            },
        },
        "selected_char_config": selected_char_config,
        "character_screen": char_screen.to_dict(orient="records"),
        "training_metadata": training_metadata,
        "internal_selection": selection,
        "internal_metrics": internal_comparison.to_dict(orient="records"),
        "length_bias": length_bias.to_dict(orient="records"),
        "external_metrics": external_comparison.to_dict(orient="records"),
        "integrity": {
            "current_baseline_sha256_before": baseline_hash_before,
            "current_baseline_sha256_after": baseline_hash_after,
            "current_baseline_unchanged": baseline_hash_before == baseline_hash_after,
            "external_sha256_before": external_hash_before,
            "external_sha256_after": external_hash_after,
            "external_unchanged": external_hash_before == external_hash_after,
            "internal_selection_sha256": selection_hash,
        },
        "artifacts": {
            "report": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
            "internal_selection": str(
                INTERNAL_SELECTION_PATH.relative_to(PROJECT_ROOT)
            ),
            "internal_comparison": str(
                INTERNAL_COMPARISON_PATH.relative_to(PROJECT_ROOT)
            ),
            "cohort_metrics": str(COHORT_METRICS_PATH.relative_to(PROJECT_ROOT)),
            "stability": str(STABILITY_PATH.relative_to(PROJECT_ROOT)),
            "external_comparison": str(
                EXTERNAL_COMPARISON_PATH.relative_to(PROJECT_ROOT)
            ),
        },
    }
    METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(
        metrics,
        char_screen,
        internal_comparison,
        cohort_metrics,
        length_bias,
        stability_summary,
        external_comparison,
    )
    return metrics


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    metrics = run_day13_comparison()
    selection = metrics["internal_selection"]
    LOGGER.info(
        "Recommended representation for Day 14: %s",
        MODEL_DISPLAY[selection["recommended_for_day14"]],
    )
    LOGGER.info(
        "External data used for tuning: %s",
        metrics["protocol"]["external_used_for_tuning"],
    )
    LOGGER.info("Report saved to: %s", REPORT_PATH)


if __name__ == "__main__":
    main()
