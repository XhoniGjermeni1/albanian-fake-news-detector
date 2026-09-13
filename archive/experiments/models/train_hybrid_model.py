"""Compare the historical text, linguistic, and hybrid Logistic Regression models."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[3]))

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.data_utils import exclude_train_duplicates_from_test
from src.evaluation.metrics import classification_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAIN_PATH = PROJECT_ROOT / "data" / "interim" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "interim" / "test.csv"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "linguistic_features.csv"

MODEL_DIR = PROJECT_ROOT / "archive" / "models"
REPORTS_DIR = PROJECT_ROOT / "archive" / "reports"

BASELINE_MODEL_PATH = MODEL_DIR / "baseline_tfidf_logreg.joblib"
HYBRID_MODEL_PATH = MODEL_DIR / "hybrid_tfidf_linguistic_logreg.joblib"
METRICS_PATH = REPORTS_DIR / "day5_metrics.json"
COMPARISON_PATH = REPORTS_DIR / "day5_model_comparison.csv"

DIRECT_LENGTH_FEATURES = [
    "word_count",
    "sentence_count",
    "character_count",
    "avg_sentence_length",
    "title_length",
    "content_length",
]

MODEL_NAMES = {
    "tfidf_only": "TF-IDF only",
    "linguistic_only": "Linguistic features only",
    "hybrid": "TF-IDF + linguistic features",
    "hybrid_no_length": "Hybrid without length features",
}

LOGGER = logging.getLogger(__name__)


def _require_columns(
    dataframe: pd.DataFrame,
    columns: set[str],
    name: str,
) -> None:
    """Raise a clear error when an input table is incomplete."""
    missing = sorted(columns - set(dataframe.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def numeric_feature_columns(features: pd.DataFrame) -> list[str]:
    """Return numeric linguistic features without identifiers or the label."""
    ignored_columns = {"pair_id", "label"}
    return [
        column
        for column in features.select_dtypes(include="number").columns
        if column not in ignored_columns
    ]


def merge_text_with_features(
    text_data: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Join linguistic features by article_id and verify labels and pair IDs."""
    text_required = {"article_id", "pair_id", "label", "label_name", "model_text"}
    feature_required = {"article_id", "pair_id", "label", "label_name"}
    _require_columns(text_data, text_required, "Text data")
    _require_columns(features, feature_required, "Linguistic features")

    if text_data["article_id"].duplicated().any():
        raise ValueError("Text data contains duplicate article_id values.")
    if features["article_id"].duplicated().any():
        raise ValueError("Linguistic features contain duplicate article_id values.")

    feature_data = features.rename(
        columns={
            "pair_id": "feature_pair_id",
            "label": "feature_label",
            "label_name": "feature_label_name",
        }
    )
    ordered_text = text_data.copy()
    ordered_text["_day5_order"] = range(len(ordered_text))
    merged = ordered_text.merge(
        feature_data,
        on="article_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    missing_feature_rows = merged["_merge"].ne("both")
    if missing_feature_rows.any():
        missing_ids = merged.loc[
            missing_feature_rows,
            "article_id",
        ].astype(str).tolist()
        raise ValueError(f"Missing linguistic features for article IDs: {missing_ids[:10]}")

    pair_mismatch = (
        merged["pair_id"].astype("string")
        != merged["feature_pair_id"].astype("string")
    )
    label_mismatch = merged["label"].astype(int) != merged["feature_label"].astype(int)
    name_mismatch = (
        merged["label_name"].astype(str)
        != merged["feature_label_name"].astype(str)
    )
    if pair_mismatch.any():
        raise ValueError("pair_id values do not match after the feature merge.")
    if label_mismatch.any() or name_mismatch.any():
        raise ValueError("Labels do not match after the feature merge.")

    merged = merged.sort_values("_day5_order").drop(
        columns=[
            "_day5_order",
            "_merge",
            "feature_pair_id",
            "feature_label",
            "feature_label_name",
        ]
    )
    return merged.reset_index(drop=True)


def load_and_check_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict]:
    """Load the frozen split and align it with the linguistic features."""
    required_paths = [TRAIN_PATH, TEST_PATH, FEATURES_PATH]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing required Day 2-3 outputs: {missing_paths}")

    train_data = pd.read_csv(
        TRAIN_PATH,
        encoding="utf-8-sig",
        keep_default_na=False,
    )
    test_data = pd.read_csv(
        TEST_PATH,
        encoding="utf-8-sig",
        keep_default_na=False,
    )
    features = pd.read_csv(
        FEATURES_PATH,
        encoding="utf-8-sig",
        keep_default_na=False,
    )

    dataframes = {
        "Train data": train_data,
        "Test data": test_data,
        "Linguistic features": features,
    }
    identity_columns = {"article_id", "pair_id", "label", "label_name"}
    for name, dataframe in dataframes.items():
        _require_columns(dataframe, identity_columns, name)
        if dataframe["article_id"].duplicated().any():
            raise ValueError(f"{name} contains duplicate article_id values.")

    _require_columns(train_data, {"model_text"}, "Train data")
    _require_columns(test_data, {"model_text"}, "Test data")

    train_ids = set(train_data["article_id"])
    test_ids = set(test_data["article_id"])
    feature_ids = set(features["article_id"])
    if train_ids & test_ids:
        raise ValueError("Train and test contain overlapping article_id values.")
    if feature_ids != train_ids | test_ids:
        raise ValueError("Linguistic feature IDs do not match the frozen split IDs.")

    train_pairs = set(train_data["pair_id"].astype("string"))
    test_pairs = set(test_data["pair_id"].astype("string"))
    if train_pairs & test_pairs:
        raise ValueError("Train and test contain overlapping pair_id values.")

    train_merged = merge_text_with_features(train_data, features)
    full_test_merged = merge_text_with_features(test_data, features)
    test_merged, excluded_ids = exclude_train_duplicates_from_test(
        train_merged,
        full_test_merged,
    )
    feature_columns = numeric_feature_columns(features)

    if not feature_columns:
        raise ValueError("No numeric linguistic feature columns were found.")
    if train_merged[feature_columns].isna().any().any():
        raise ValueError(
            "Train data contains missing linguistic feature values after the merge."
        )
    if test_merged[feature_columns].isna().any().any():
        raise ValueError(
            "Test data contains missing linguistic feature values after the merge."
        )

    checks = {
        "clean_rows": int(len(train_data) + len(test_data)),
        "original_train_rows": int(len(train_data)),
        "original_test_rows": int(len(test_data)),
        "evaluation_test_rows": int(len(test_merged)),
        "train_test_article_id_overlap": 0,
        "train_test_pair_id_overlap": 0,
        "exact_train_texts_excluded_from_test": int(len(excluded_ids)),
        "excluded_article_ids": excluded_ids,
        "linguistic_feature_rows": int(len(features)),
        "numeric_linguistic_feature_count": int(len(feature_columns)),
        "missing_feature_rows": 0,
        "pair_id_mismatches": 0,
        "label_mismatches": 0,
        "empty_train_model_texts": int(train_merged["model_text"].eq("").sum()),
        "empty_test_model_texts": int(test_merged["model_text"].eq("").sum()),
    }
    return train_merged, test_merged, feature_columns, checks


def _tfidf_vectorizer() -> TfidfVectorizer:
    """Use the same TF-IDF configuration as the Day 2 baseline."""
    return TfidfVectorizer(
        lowercase=False,
        ngram_range=(1, 2),
        min_df=2,
        max_features=30000,
    )


def _classifier() -> LogisticRegression:
    """Create the shared classifier used by all Day 5 comparisons."""
    return LogisticRegression(max_iter=1000, class_weight="balanced")


def build_tfidf_model() -> Pipeline:
    """Build the text-only baseline for the fair Day 5 comparison."""
    return Pipeline(
        steps=[
            ("tfidf", _tfidf_vectorizer()),
            ("classifier", _classifier()),
        ]
    )


def build_linguistic_model() -> Pipeline:
    """Build a Logistic Regression model using only numeric features."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", _classifier()),
        ]
    )


def build_hybrid_model(feature_columns: list[str]) -> Pipeline:
    """Combine TF-IDF and standardized linguistic features."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    combined_features = ColumnTransformer(
        transformers=[
            ("tfidf", _tfidf_vectorizer(), "model_text"),
            ("linguistic", numeric_pipeline, feature_columns),
        ]
    )
    return Pipeline(
        steps=[
            ("features", combined_features),
            ("classifier", _classifier()),
        ]
    )


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Calculate the metrics retained in the historical comparison."""
    metrics = classification_metrics(y_true, y_pred)
    return {
        "accuracy": round(metrics["accuracy"], 4),
        "precision": round(metrics["precision_weighted"], 4),
        "recall": round(metrics["recall_weighted"], 4),
        "f1": round(metrics["f1_weighted"], 4),
        "precision_fake": round(metrics["precision_fake"], 4),
        "recall_fake": round(metrics["recall_fake"], 4),
        "f1_fake": round(metrics["f1_fake"], 4),
        "confusion_matrix": metrics["confusion_matrix"],
    }


def build_comparison_table(metrics: dict[str, dict]) -> pd.DataFrame:
    """Create one readable row per evaluated model."""
    rows = []
    for model_key in MODEL_NAMES:
        model_metrics = metrics[model_key]
        matrix = model_metrics["confusion_matrix"]
        rows.append(
            {
                "model_key": model_key,
                "model": MODEL_NAMES[model_key],
                "accuracy": model_metrics["accuracy"],
                "precision": model_metrics["precision"],
                "recall": model_metrics["recall"],
                "f1": model_metrics["f1"],
                "precision_fake": model_metrics["precision_fake"],
                "recall_fake": model_metrics["recall_fake"],
                "f1_fake": model_metrics["f1_fake"],
                "true_real_pred_real": matrix[0][0],
                "true_real_pred_fake": matrix[0][1],
                "true_fake_pred_real": matrix[1][0],
                "true_fake_pred_fake": matrix[1][1],
            }
        )
    return pd.DataFrame(rows)


def _relative_path(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def train_and_compare_models() -> dict:
    """Train the four Day 5 candidates and save their comparison."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train_data, test_data, feature_columns, checks = load_and_check_data()
    no_length_columns = [
        column for column in feature_columns if column not in DIRECT_LENGTH_FEATURES
    ]

    model_specs = {
        "tfidf_only": {
            "model": build_tfidf_model(),
            "train_input": train_data["model_text"],
            "test_input": test_data["model_text"],
            "artifact_path": BASELINE_MODEL_PATH,
        },
        "linguistic_only": {
            "model": build_linguistic_model(),
            "train_input": train_data[feature_columns],
            "test_input": test_data[feature_columns],
            "artifact_path": None,
        },
        "hybrid": {
            "model": build_hybrid_model(feature_columns),
            "train_input": train_data,
            "test_input": test_data,
            "artifact_path": HYBRID_MODEL_PATH,
        },
        "hybrid_no_length": {
            "model": build_hybrid_model(no_length_columns),
            "train_input": train_data,
            "test_input": test_data,
            "artifact_path": None,
        },
    }

    metrics = {}
    for model_key, specification in model_specs.items():
        LOGGER.info("Training %s", MODEL_NAMES[model_key])
        model = specification["model"]
        model.fit(specification["train_input"], train_data["label"])
        predictions = model.predict(specification["test_input"])
        metrics[model_key] = evaluate_predictions(test_data["label"], predictions)
        artifact_path = specification["artifact_path"]
        if artifact_path is not None:
            joblib.dump(model, artifact_path)

    comparison = build_comparison_table(metrics)
    comparison.to_csv(COMPARISON_PATH, index=False, encoding="utf-8-sig")
    best_model_key = max(metrics, key=lambda key: metrics[key]["f1_fake"])

    result = {
        "data_checks": checks,
        "features": {
            "all_numeric_features": feature_columns,
            "all_numeric_feature_count": len(feature_columns),
            "removed_length_features": DIRECT_LENGTH_FEATURES,
            "no_length_feature_count": len(no_length_columns),
        },
        "metrics": metrics,
        "best_model_by_f1_fake": best_model_key,
        "artifacts": {
            "baseline_model": _relative_path(BASELINE_MODEL_PATH),
            "hybrid_model": _relative_path(HYBRID_MODEL_PATH),
            "comparison_table": _relative_path(COMPARISON_PATH),
        },
    }
    METRICS_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    result = train_and_compare_models()

    print("=== Day 5 model comparison ===")
    print(f"Train rows: {result['data_checks']['original_train_rows']}")
    print(f"Evaluation test rows: {result['data_checks']['evaluation_test_rows']}")
    print(
        "Exact train duplicates excluded from test: "
        f"{result['data_checks']['exact_train_texts_excluded_from_test']}"
    )
    for model_key, model_metrics in result["metrics"].items():
        print(
            f"{MODEL_NAMES[model_key]}: accuracy={model_metrics['accuracy']}, "
            f"f1_fake={model_metrics['f1_fake']}"
        )
    print(f"Best model by F1 fake: {MODEL_NAMES[result['best_model_by_f1_fake']]}")
    print(f"Metrics saved: {METRICS_PATH}")
    print(f"Comparison saved: {COMPARISON_PATH}")


if __name__ == "__main__":
    main()
