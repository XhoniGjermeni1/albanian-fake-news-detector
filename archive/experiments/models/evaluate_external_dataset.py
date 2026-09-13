# Zbaton kontratën historike të aplikacionit mbi benchmark-un e jashtëm dhe llogarit
# metrikat binare, vendimet me tre nivele, rezultatet sipas klasës/burimit dhe krahasimin
# me testin e brendshëm. U krijua për të matur përgjithësimin jashtë corpus-it të trajnimit.

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from archive.experiments.evaluation.experiment_utils import file_sha256
from archive.experiments.models.evaluate_app_system import (
    evaluate_test_set,
    load_evaluation_data,
)
from archive.experiments.models.predict import (
    DEFAULT_FAKE_THRESHOLD,
    DEFAULT_REAL_THRESHOLD,
    load_model,
    predict_news_for_app,
)
from src.evaluation.metrics import classification_metrics
from src.features.linguistic_features import extract_linguistic_features

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXTERNAL_DATASET_PATH = PROJECT_ROOT / "data" / "external" / "external_news.csv"
MODEL_PATH = PROJECT_ROOT / "archive" / "models" / "calibrated_tfidf_logreg.joblib"
RAW_METADATA_ROOT = (
    PROJECT_ROOT / "data" / "raw" / "alb-fake-news-corpus" / "full_texts"
)

REPORTS_DIR = PROJECT_ROOT / "archive" / "reports"
PREDICTIONS_PATH = REPORTS_DIR / "day11_external_predictions.csv"
METRICS_PATH = REPORTS_DIR / "day11_external_metrics.json"

LOGGER = logging.getLogger(__name__)
LABEL_TO_NUMBER = {"real": 0, "fake": 1}
NUMBER_TO_LABEL = {0: "real", 1: "fake"}
REQUIRED_EXTERNAL_COLUMNS = {
    "external_id",
    "title",
    "content",
    "label",
    "source",
    "topic",
    "published_date",
}


def source_group(source: str) -> str:
    if source.startswith("Këshilli i Ministrave"):
        return "institutional_government"
    if source == "Banka e Shqipërisë":
        return "institutional_financial"
    if source == "INSTAT":
        return "institutional_statistics"
    if source.startswith("Krypometër"):
        return "fact_checked_social_claim"
    return "other"


def load_external_inputs() -> tuple[pd.DataFrame, object]:
    required_paths = [EXTERNAL_DATASET_PATH, MODEL_PATH]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing external evaluation inputs: {missing_paths}")

    external = pd.read_csv(
        EXTERNAL_DATASET_PATH,
        encoding="utf-8",
        keep_default_na=False,
    )
    missing_columns = sorted(REQUIRED_EXTERNAL_COLUMNS - set(external.columns))
    if missing_columns:
        raise ValueError(f"External dataset is missing columns: {missing_columns}")
    if len(external) != 40:
        raise ValueError(f"Expected 40 external rows, found {len(external)}")
    if set(external["label"]) != set(LABEL_TO_NUMBER):
        raise ValueError("External labels must contain only real and fake.")

    return external, load_model(MODEL_PATH)


def _expected_decision(probability_fake: float) -> str:
    if probability_fake < DEFAULT_REAL_THRESHOLD:
        return "likely_real"
    if probability_fake > DEFAULT_FAKE_THRESHOLD:
        return "likely_fake"
    return "uncertain"


def run_external_predictions(external: pd.DataFrame, model) -> pd.DataFrame:
    rows: list[dict] = []
    for article in external.itertuples(index=False):
        result = predict_news_for_app(article.title, article.content, model=model)
        probability_real = float(result["probability_real"])
        probability_fake = float(result["probability_fake"])
        binary_number = int(probability_fake >= 0.50)
        true_number = LABEL_TO_NUMBER[article.label]
        explanation = result["linguistic_explanation"]

        if result["decision"] != _expected_decision(probability_fake):
            raise RuntimeError(f"Threshold mismatch for {article.external_id}")

        error_type = "correct"
        if true_number == 0 and binary_number == 1:
            error_type = "false_positive"
        elif true_number == 1 and binary_number == 0:
            error_type = "false_negative"

        rows.append(
            {
                "external_id": article.external_id,
                "title": article.title,
                "content": article.content,
                "true_label": article.label,
                "true_label_number": true_number,
                "binary_prediction": NUMBER_TO_LABEL[binary_number],
                "binary_prediction_number": binary_number,
                "probability_real": probability_real,
                "probability_fake": probability_fake,
                "probability_sum": round(probability_real + probability_fake, 4),
                "decision": result["decision"],
                "topic": article.topic,
                "source": article.source,
                "source_group": source_group(article.source),
                "published_date": article.published_date,
                "prediction_correct": true_number == binary_number,
                "error_type": error_type,
                "predicted_confidence": max(probability_real, probability_fake),
                "word_count": int(explanation["word_count"]),
                "text_length": int(explanation["text_length"]),
                "exclamation_count": int(explanation["exclamation_count"]),
                "uppercase_ratio": float(explanation["uppercase_ratio"]),
                "diacritic_ratio": float(explanation["diacritic_ratio"]),
                "sensational_words_found": " | ".join(
                    explanation["sensational_words_found"]
                ),
                "source_markers_found": " | ".join(
                    explanation["source_markers_found"]
                ),
                "uncertainty_markers_found": " | ".join(
                    explanation["uncertainty_markers_found"]
                ),
            }
        )

    predictions = pd.DataFrame(rows)
    predictions["length_group"] = pd.cut(
        predictions["word_count"],
        bins=[-np.inf, 44, 47, np.inf],
        labels=["short_38_44", "medium_45_47", "long_48_51"],
    ).astype(str)
    return predictions


def calculate_binary_metrics(predictions: pd.DataFrame) -> dict:
    y_true = predictions["true_label_number"].to_numpy()
    y_pred = predictions["binary_prediction_number"].to_numpy()
    common = classification_metrics(y_true, y_pred)
    macro = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    matrix = common["confusion_matrix"]

    return {
        "rows": common["rows"],
        "accuracy": round(common["accuracy"], 4),
        "precision": round(common["precision_weighted"], 4),
        "recall": round(common["recall_weighted"], 4),
        "f1": round(common["f1_weighted"], 4),
        "averaging": "weighted",
        "precision_macro": round(float(macro[0]), 4),
        "recall_macro": round(float(macro[1]), 4),
        "f1_macro": round(float(macro[2]), 4),
        "class_real": {
            "precision": round(common["precision_real"], 4),
            "recall": round(common["recall_real"], 4),
            "f1": round(common["f1_real"], 4),
            "support": common["real_rows"],
        },
        "class_fake": {
            "precision": round(common["precision_fake"], 4),
            "recall": round(common["recall_fake"], 4),
            "f1": round(common["f1_fake"], 4),
            "support": common["fake_rows"],
        },
        "confusion_matrix_labels": ["real", "fake"],
        "confusion_matrix": matrix,
        "true_negatives": int(matrix[0][0]),
        "false_positives": common["false_positives"],
        "false_negatives": common["false_negatives"],
        "true_positives": int(matrix[1][1]),
        "high_confidence_errors_90": int(
            (
                predictions["error_type"].ne("correct")
                & predictions["predicted_confidence"].ge(0.90)
            ).sum()
        ),
        "balanced_constant_baseline_accuracy": 0.50,
    }


def calculate_decision_metrics(predictions: pd.DataFrame) -> dict:
    strong_mask = predictions["decision"].ne("uncertain")
    strong_correct = (
        (
            predictions["true_label"].eq("real")
            & predictions["decision"].eq("likely_real")
        )
        | (
            predictions["true_label"].eq("fake")
            & predictions["decision"].eq("likely_fake")
        )
    )
    binary_error = predictions["error_type"].ne("correct")
    uncertain_mask = predictions["decision"].eq("uncertain")
    strong_count = int(strong_mask.sum())
    uncertain_count = int(uncertain_mask.sum())
    total_errors = int(binary_error.sum())
    errors_in_uncertain = int((binary_error & uncertain_mask).sum())

    return {
        "thresholds": {
            "likely_real_below": DEFAULT_REAL_THRESHOLD,
            "likely_fake_above": DEFAULT_FAKE_THRESHOLD,
        },
        "likely_real": int(predictions["decision"].eq("likely_real").sum()),
        "uncertain": uncertain_count,
        "likely_fake": int(predictions["decision"].eq("likely_fake").sum()),
        "uncertain_real": int(
            (uncertain_mask & predictions["true_label"].eq("real")).sum()
        ),
        "uncertain_fake": int(
            (uncertain_mask & predictions["true_label"].eq("fake")).sum()
        ),
        "strong_decision_count": strong_count,
        "strong_decision_coverage": round(strong_count / len(predictions), 4),
        "strong_decision_accuracy": round(
            float(strong_correct[strong_mask].mean()) if strong_count else 0.0,
            4,
        ),
        "strong_decision_errors": int((strong_mask & ~strong_correct).sum()),
        "strong_false_positives": int(
            (
                predictions["true_label"].eq("real")
                & predictions["decision"].eq("likely_fake")
            ).sum()
        ),
        "strong_false_negatives": int(
            (
                predictions["true_label"].eq("fake")
                & predictions["decision"].eq("likely_real")
            ).sum()
        ),
        "binary_errors": total_errors,
        "binary_errors_moved_to_uncertain": errors_in_uncertain,
        "binary_error_capture_rate": round(
            errors_in_uncertain / total_errors if total_errors else 0.0,
            4,
        ),
        "uncertain_error_rate": round(
            errors_in_uncertain / uncertain_count if uncertain_count else 0.0,
            4,
        ),
    }


def raw_corpus_date_range() -> dict:
    dates: list[datetime] = []
    for directory_name in ("true-meta-information", "fake-meta-information"):
        directory = RAW_METADATA_ROOT / directory_name
        if not directory.exists():
            continue
        for path in directory.glob("*.txt"):
            with path.open(encoding="utf-8", errors="replace") as metadata_file:
                first_line = metadata_file.readline().strip()
            try:
                dates.append(datetime.strptime(first_line, "%Y/%m/%d, %H:%M:%S"))
            except ValueError:
                continue
    return {
        "valid_metadata_dates": len(dates),
        "minimum": min(dates).date().isoformat() if dates else None,
        "maximum": max(dates).date().isoformat() if dates else None,
    }


def build_internal_comparison(predictions: pd.DataFrame) -> dict:
    internal_test, internal_model, excluded_ids = load_evaluation_data()
    _, internal_summary = evaluate_test_set(internal_test, internal_model)
    internal_words = pd.Series(
        [
            int(extract_linguistic_features(row.title, row.content)["word_count"])
            for row in internal_test.itertuples(index=False)
        ]
    )
    internal_by_label = {}
    for label_number, group in internal_test.assign(word_count=internal_words).groupby(
        "label"
    ):
        words = group["word_count"]
        internal_by_label[NUMBER_TO_LABEL[int(label_number)]] = {
            "rows": int(len(group)),
            "mean_words": round(float(words.mean()), 2),
            "median_words": round(float(words.median()), 2),
        }

    external_dates = pd.to_datetime(predictions["published_date"], errors="coerce")
    return {
        "internal": {
            "rows": int(internal_summary["rows"]),
            "accuracy": float(internal_summary["binary_accuracy"]),
            "false_positives": int(internal_summary["false_positives"]),
            "false_negatives": int(internal_summary["false_negatives"]),
            "strong_decision_coverage": float(
                internal_summary["strong_decision_coverage"]
            ),
            "strong_decision_accuracy": float(
                internal_summary["strong_decision_accuracy"]
            ),
            "uncertain": int(internal_summary["uncertain_count"]),
            "mean_words": round(float(internal_words.mean()), 2),
            "median_words": round(float(internal_words.median()), 2),
            "min_words": int(internal_words.min()),
            "max_words": int(internal_words.max()),
            "words_by_label": internal_by_label,
            "publication_dates": raw_corpus_date_range(),
            "exact_train_duplicates_excluded": len(excluded_ids),
        },
        "external": {
            "rows": int(len(predictions)),
            "accuracy": round(float(predictions["prediction_correct"].mean()), 4),
            "mean_words": round(float(predictions["word_count"].mean()), 2),
            "median_words": round(float(predictions["word_count"].median()), 2),
            "min_words": int(predictions["word_count"].min()),
            "max_words": int(predictions["word_count"].max()),
            "words_by_label": {
                label: {
                    "rows": int(len(group)),
                    "mean_words": round(float(group["word_count"].mean()), 2),
                    "median_words": round(float(group["word_count"].median()), 2),
                }
                for label, group in predictions.groupby("true_label")
            },
            "publication_dates": {
                "minimum": external_dates.min().date().isoformat(),
                "maximum": external_dates.max().date().isoformat(),
            },
            "text_format": "manual summaries",
        },
        "accuracy_difference_external_minus_internal": round(
            float(predictions["prediction_correct"].mean())
            - float(internal_summary["binary_accuracy"]),
            4,
        ),
    }


def run_external_evaluation() -> dict:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model_hash_before = file_sha256(MODEL_PATH)
    dataset_hash_before = file_sha256(EXTERNAL_DATASET_PATH)
    external, model = load_external_inputs()
    predictions = run_external_predictions(external, model)

    maximum_probability_sum_error = float(
        predictions["probability_sum"].sub(1.0).abs().max()
    )
    if maximum_probability_sum_error > 0.0002:
        raise RuntimeError("Real and fake probabilities do not sum to approximately one.")

    binary_metrics = calculate_binary_metrics(predictions)
    decision_metrics = calculate_decision_metrics(predictions)
    internal_comparison = build_internal_comparison(predictions)

    model_hash_after = file_sha256(MODEL_PATH)
    dataset_hash_after = file_sha256(EXTERNAL_DATASET_PATH)
    if model_hash_before != model_hash_after:
        raise RuntimeError("The saved model changed during evaluation.")
    if dataset_hash_before != dataset_hash_after:
        raise RuntimeError("The external dataset changed during evaluation.")

    metrics = {
        "status": "completed",
        "model": MODEL_PATH.name,
        "model_retrained": False,
        "prediction_function": "predict_news_for_app",
        "preprocessing_function": "combine_title_content",
        "model_sha256_before": model_hash_before,
        "model_sha256_after": model_hash_after,
        "dataset_sha256_before": dataset_hash_before,
        "dataset_sha256_after": dataset_hash_after,
        "probability_checks": {
            "outside_0_1": int(
                (
                    predictions["probability_real"].lt(0)
                    | predictions["probability_real"].gt(1)
                    | predictions["probability_fake"].lt(0)
                    | predictions["probability_fake"].gt(1)
                ).sum()
            ),
            "maximum_sum_error": maximum_probability_sum_error,
        },
        "binary_metrics": binary_metrics,
        "decision_metrics": decision_metrics,
        "internal_comparison": internal_comparison,
        "artifacts": {
            "predictions": str(PREDICTIONS_PATH.relative_to(PROJECT_ROOT)),
            "metrics": str(METRICS_PATH.relative_to(PROJECT_ROOT)),
        },
    }

    predictions.to_csv(PREDICTIONS_PATH, index=False, encoding="utf-8")
    METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    metrics = run_external_evaluation()
    binary = metrics["binary_metrics"]
    decision = metrics["decision_metrics"]
    LOGGER.info("External rows: %s", binary["rows"])
    LOGGER.info("Binary accuracy: %.2f%%", binary["accuracy"] * 100)
    LOGGER.info(
        "Fake precision / recall / F1: %.2f%% / %.2f%% / %.2f%%",
        binary["class_fake"]["precision"] * 100,
        binary["class_fake"]["recall"] * 100,
        binary["class_fake"]["f1"] * 100,
    )
    LOGGER.info("Confusion matrix: %s", binary["confusion_matrix"])
    LOGGER.info(
        "Decisions likely_real / uncertain / likely_fake: %s / %s / %s",
        decision["likely_real"],
        decision["uncertain"],
        decision["likely_fake"],
    )
    LOGGER.info("Metrics saved to: %s", METRICS_PATH)


if __name__ == "__main__":
    main()
