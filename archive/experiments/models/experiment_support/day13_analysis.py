"""Computation helpers for the historical Day 13 TF-IDF experiment."""

from __future__ import annotations

import json
import logging
import time
import unicodedata
from pathlib import Path

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
from src.evaluation.experiment_utils import file_sha256
from src.features.linguistic_features import extract_linguistic_features
from src.models.analyze_length_domain_shift import truncate_to_total_words
from src.models.analyze_model_quality import build_calibration_folds
from src.models.builders import build_char_vectorizer, build_word_vectorizer
from src.models.predict import (
    DEFAULT_FAKE_THRESHOLD,
    DEFAULT_REAL_THRESHOLD,
    classify_probability,
)
from src.preprocessing.clean_text import combine_title_content


PROJECT_ROOT = Path(__file__).resolve().parents[3]
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

