# Ndërton dhe vlerëson në kushte identike tre përfaqësime TF-IDF: vetëm fjalë,
# vetëm karaktere dhe bashkimin Word+Character. Përfshin screening-un e konfigurimit
# character, calibration-in e njëjtë dhe përzgjedhjen vetëm nga metrikat e train/CV.

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline

from archive.experiments.models.analyze_model_quality import build_calibration_folds
from archive.experiments.models.predict import (
    DEFAULT_FAKE_THRESHOLD,
    DEFAULT_REAL_THRESHOLD,
    classify_probability,
)
from src.evaluation.data_utils import (
    build_leakage_safe_groups,
    exclude_train_duplicates_from_test,
)
from src.evaluation.metrics import classification_metrics
from src.models.builders import build_char_vectorizer, build_word_vectorizer
from src.preprocessing.clean_text import combine_title_content

PROJECT_ROOT = Path(__file__).resolve().parents[4]
TRAIN_PATH = PROJECT_ROOT / "data" / "interim" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "interim" / "test.csv"

REPORTS_DIR = PROJECT_ROOT / "archive" / "reports"
INTERNAL_SELECTION_PATH = REPORTS_DIR / "day13_internal_selection.json"
METRICS_PATH = REPORTS_DIR / "day13_metrics.json"

MODEL_NAMES = ["word_tfidf", "char_tfidf", "word_char_tfidf"]
MODEL_DISPLAY = {
    "word_tfidf": "Word TF-IDF",
    "char_tfidf": "Character TF-IDF",
    "word_char_tfidf": "Word + Character TF-IDF",
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

LOGGER = logging.getLogger(__name__)


def build_classifier() -> LogisticRegression:
    return LogisticRegression(max_iter=1000, class_weight="balanced")


def build_representation_pipeline(
    representation: str,
    char_config: dict,
) -> Pipeline:
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
    required = [TRAIN_PATH, TEST_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing representation comparison inputs: {missing}")

    train = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig", keep_default_na=False)
    test = pd.read_csv(TEST_PATH, encoding="utf-8-sig", keep_default_na=False)
    evaluation_test, excluded_ids = exclude_train_duplicates_from_test(train, test)
    return train.reset_index(drop=True), evaluation_test.reset_index(drop=True), excluded_ids


def build_screen_folds(train: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = build_leakage_safe_groups(train)
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    folds = list(splitter.split(train["model_text"], train["label"], groups=groups))
    for fit_index, validation_index in folds:
        if set(groups[fit_index]) & set(groups[validation_index]):
            raise RuntimeError("Character screen folds contain group leakage.")
    return folds


def screen_character_configs(train: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    folds = build_screen_folds(train)
    rows: list[dict] = []

    for config in CHARACTER_CONFIGS:
        fold_metrics: list[dict] = []
        for fit_index, validation_index in folds:
            model = build_representation_pipeline("char_tfidf", config)
            model.fit(
                train.iloc[fit_index]["model_text"],
                train.iloc[fit_index]["label"],
            )
            y_true = train.iloc[validation_index]["label"]
            y_pred = model.predict(train.iloc[validation_index]["model_text"])
            fold_metrics.append(classification_metrics(y_true, y_pred))

        rows.append(
            {
                **config,
                "screen_scope": "train_only_3_fold_group_safe_cv",
                "mean_accuracy": round(
                    float(np.mean([item["accuracy"] for item in fold_metrics])),
                    4,
                ),
                "std_accuracy": round(
                    float(np.std([item["accuracy"] for item in fold_metrics])),
                    4,
                ),
                "mean_f1_weighted": round(
                    float(np.mean([item["f1_weighted"] for item in fold_metrics])),
                    4,
                ),
                "std_f1_weighted": round(
                    float(np.std([item["f1_weighted"] for item in fold_metrics])),
                    4,
                ),
                "mean_f1_fake": round(
                    float(np.mean([item["f1_fake"] for item in fold_metrics])),
                    4,
                ),
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
) -> CalibratedClassifierCV:
    model = CalibratedClassifierCV(
        estimator=build_representation_pipeline(representation, char_config),
        method="sigmoid",
        cv=calibration_folds,
        ensemble=False,
    )
    model.fit(train["model_text"], train["label"])
    return model


def probability_arrays(model, texts: list[str] | pd.Series) -> tuple[np.ndarray, np.ndarray]:
    probabilities = model.predict_proba(texts)
    classes = list(model.classes_)
    return probabilities[:, classes.index(0)], probabilities[:, classes.index(1)]


def prediction_table(
    base_data: pd.DataFrame,
    model,
    model_name: str,
    id_column: str,
) -> pd.DataFrame:
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
    y_true = table["label"].to_numpy(dtype=int)
    y_pred = table["binary_prediction"].to_numpy(dtype=int)
    probability_fake = table["probability_fake"].to_numpy(dtype=float)
    probabilities = np.column_stack(
        [table["probability_real"].to_numpy(dtype=float), probability_fake]
    )
    common = classification_metrics(y_true, y_pred)
    strong = table["decision"].ne("uncertain")
    strong_correct = (
        (table["label"].eq(0) & table["decision"].eq("likely_real"))
        | (table["label"].eq(1) & table["decision"].eq("likely_fake"))
    )
    return {
        "rows": common["rows"],
        "real_rows": common["real_rows"],
        "fake_rows": common["fake_rows"],
        "accuracy": round(common["accuracy"], 4),
        "precision_weighted": round(common["precision_weighted"], 4),
        "recall_weighted": round(common["recall_weighted"], 4),
        "f1_weighted": round(common["f1_weighted"], 4),
        "precision_real": round(common["precision_real"], 4),
        "recall_real": round(common["recall_real"], 4),
        "f1_real": round(common["f1_real"], 4),
        "precision_fake": round(common["precision_fake"], 4),
        "recall_fake": round(common["recall_fake"], 4),
        "f1_fake": round(common["f1_fake"], 4),
        "brier_score": round(float(brier_score_loss(y_true, probability_fake)), 6),
        "log_loss": round(float(log_loss(y_true, probabilities, labels=[0, 1])), 6),
        "confusion_matrix": common["confusion_matrix"],
        "false_positives": common["false_positives"],
        "false_negatives": common["false_negatives"],
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


def internal_selection(comparison: pd.DataFrame, selected_char_config: dict) -> dict:
    best_internal = comparison.sort_values(
        ["f1_weighted", "f1_fake", "accuracy", "brier_score"],
        ascending=[False, False, False, True],
    ).iloc[0]
    selected_model = str(best_internal["model"])
    return {
        "selection_scope": "internal_only_before_external_load",
        "selected_char_config": selected_char_config,
        "best_internal_model": selected_model,
        "best_internal_f1_weighted": float(best_internal["f1_weighted"]),
        "recommended_for_day14": selected_model,
        "primary_rule": "highest internal F1 weighted, then F1 fake, accuracy, Brier",
        "external_results_used": False,
    }
