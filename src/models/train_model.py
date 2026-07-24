"""Train the first minimal TF-IDF + Logistic Regression baseline model."""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from src.preprocessing.clean_text import prepare_text_dataframe
from src.models.predict import predict_news

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "articles.csv"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

CLEAN_DATA_PATH = INTERIM_DIR / "articles_clean.csv"
TRAIN_DATA_PATH = INTERIM_DIR / "train.csv"
TEST_DATA_PATH = INTERIM_DIR / "test.csv"
MODEL_PATH = MODEL_DIR / "baseline_tfidf_logreg.joblib"
METRICS_PATH = REPORTS_DIR / "day2_metrics.json"

LOGGER = logging.getLogger(__name__)


def split_train_test(dataframe: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data by pair_id to reduce leakage between train and test."""
    groups = dataframe["pair_id"].fillna(dataframe["article_id"])
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_index, test_index = next(splitter.split(dataframe, dataframe["label"], groups=groups))

    train_dataframe = dataframe.iloc[train_index].reset_index(drop=True)
    test_dataframe = dataframe.iloc[test_index].reset_index(drop=True)
    return train_dataframe, test_dataframe


def build_baseline_model() -> Pipeline:
    """Create the baseline text classification pipeline."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=False,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=30000,
                ),
            ),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def evaluate_model(model: Pipeline, test_dataframe: pd.DataFrame) -> dict:
    """Calculate basic classification metrics."""
    y_true = test_dataframe["label"]
    y_pred = model.predict(test_dataframe["model_text"])

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label=1,
        zero_division=0,
    )

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_fake": round(float(precision), 4),
        "recall_fake": round(float(recall), 4),
        "f1_fake": round(float(f1), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=["real", "fake"],
            zero_division=0,
            output_dict=True,
        ),
    }


def train_baseline_model() -> dict:
    """Prepare text, split data, train the baseline model, and save artifacts."""
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Reading %s", INPUT_DATA_PATH)
    dataframe = pd.read_csv(INPUT_DATA_PATH, encoding="utf-8-sig")

    dataframe = prepare_text_dataframe(dataframe)
    dataframe = dataframe[dataframe["model_text"].str.len() > 0].reset_index(drop=True)
    dataframe.to_csv(CLEAN_DATA_PATH, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)

    train_dataframe, test_dataframe = split_train_test(dataframe)
    train_dataframe.to_csv(TRAIN_DATA_PATH, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
    test_dataframe.to_csv(TEST_DATA_PATH, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)

    model = build_baseline_model()
    model.fit(train_dataframe["model_text"], train_dataframe["label"])

    metrics = evaluate_model(model, test_dataframe)
    joblib.dump(model, MODEL_PATH)

    sample_row = test_dataframe.iloc[0]
    sample_prediction = predict_news(sample_row["title"], sample_row["content"], MODEL_PATH)

    result = {
        "input_rows": int(len(dataframe)),
        "train_rows": int(len(train_dataframe)),
        "test_rows": int(len(test_dataframe)),
        "train_label_counts": {
            str(label): int(count)
            for label, count in train_dataframe["label_name"].value_counts().sort_index().items()
        },
        "test_label_counts": {
            str(label): int(count)
            for label, count in test_dataframe["label_name"].value_counts().sort_index().items()
        },
        "model_path": str(MODEL_PATH),
        "clean_data_path": str(CLEAN_DATA_PATH),
        "train_data_path": str(TRAIN_DATA_PATH),
        "test_data_path": str(TEST_DATA_PATH),
        "metrics": metrics,
        "sample_prediction": {
            "article_id": sample_row["article_id"],
            "title": sample_row["title"],
            "true_label": sample_row["label_name"],
            "prediction": sample_prediction,
        },
    }

    METRICS_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    result = train_baseline_model()

    print("=== Day 2 baseline model ===")
    print(f"Input rows: {result['input_rows']}")
    print(f"Train rows: {result['train_rows']}")
    print(f"Test rows: {result['test_rows']}")
    print(f"Train label counts: {result['train_label_counts']}")
    print(f"Test label counts: {result['test_label_counts']}")
    print(f"Accuracy: {result['metrics']['accuracy']}")
    print(f"Precision fake: {result['metrics']['precision_fake']}")
    print(f"Recall fake: {result['metrics']['recall_fake']}")
    print(f"F1 fake: {result['metrics']['f1_fake']}")
    print(f"Model saved: {result['model_path']}")
    print(f"Metrics saved: {METRICS_PATH}")
    print(f"Sample prediction: {result['sample_prediction']}")


if __name__ == "__main__":
    main()
