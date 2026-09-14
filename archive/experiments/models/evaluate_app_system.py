# Riprodhon testimin end-to-end të modelit historik të kalibruar mbi test split-in pa leakage.
# Kontrollon probabilitetet, vendimet 0.30/0.70, mbulimin e zonës së sigurt dhe përputhjen
# mes helper-it të parashikimit dhe llogaritjes së pavarur të pragjeve.

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[3]))

import joblib
import numpy as np
import pandas as pd

from archive.experiments.models.predict import (
    DEFAULT_FAKE_THRESHOLD,
    DEFAULT_REAL_THRESHOLD,
    classify_probability,
)
from src.evaluation.data_utils import exclude_train_duplicates_from_test
from src.evaluation.metrics import classification_metrics
from src.preprocessing.clean_text import combine_title_content

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAIN_PATH = PROJECT_ROOT / "data" / "interim" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "interim" / "test.csv"
MODEL_PATH = (
    PROJECT_ROOT
    / "reports"
    / "experiment_history"
    / "06_initial_probability_calibration"
    / "artifacts"
    / "calibrated_tfidf_logreg.joblib"
)

REPORTS_DIR = (
    PROJECT_ROOT / "reports" / "experiment_history" / "07_application_contract"
)
METRICS_PATH = REPORTS_DIR / "metrics.json"

LOGGER = logging.getLogger(__name__)


def load_evaluation_data() -> tuple[pd.DataFrame, object, list[str]]:
    required_paths = [TRAIN_PATH, TEST_PATH, MODEL_PATH]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required artifacts: {missing}")

    train_data = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig", keep_default_na=False)
    test_data = pd.read_csv(TEST_PATH, encoding="utf-8-sig", keep_default_na=False)
    evaluation_test, excluded_ids = exclude_train_duplicates_from_test(train_data, test_data)
    model = joblib.load(MODEL_PATH)
    return evaluation_test.reset_index(drop=True), model, excluded_ids


def _count_threshold_mismatches(
    decisions: pd.Series,
    probabilities: pd.Series,
) -> int:
    likely_real = probabilities.lt(DEFAULT_REAL_THRESHOLD) & decisions.eq("likely_real")
    uncertain = probabilities.between(
        DEFAULT_REAL_THRESHOLD,
        DEFAULT_FAKE_THRESHOLD,
        inclusive="both",
    ) & decisions.eq("uncertain")
    likely_fake = probabilities.gt(DEFAULT_FAKE_THRESHOLD) & decisions.eq("likely_fake")
    return int((~(likely_real | uncertain | likely_fake)).sum())


def evaluate_test_set(test_data: pd.DataFrame, model) -> tuple[pd.DataFrame, dict]:
    model_texts = [
        combine_title_content(row.title, row.content)
        for row in test_data.itertuples(index=False)
    ]
    probabilities = model.predict_proba(model_texts)
    classes = list(model.classes_)
    probability_real = probabilities[:, classes.index(0)]
    probability_fake = probabilities[:, classes.index(1)]

    table = test_data.copy()
    table["probability_real"] = probability_real
    table["probability_fake"] = probability_fake
    table["probability_sum"] = probability_real + probability_fake
    table["binary_prediction"] = (probability_fake >= 0.5).astype(int)
    table["decision"] = [classify_probability(float(value)) for value in probability_fake]
    table["predicted_confidence"] = np.maximum(probability_real, probability_fake)
    table["error_type"] = np.where(
        (table["label"] == 0) & (table["binary_prediction"] == 1),
        "false_positive",
        np.where(
            (table["label"] == 1) & (table["binary_prediction"] == 0),
            "false_negative",
            "correct",
        ),
    )

    invalid_probability_rows = int(
        (
            (table["probability_real"] < 0)
            | (table["probability_real"] > 1)
            | (table["probability_fake"] < 0)
            | (table["probability_fake"] > 1)
        ).sum()
    )
    maximum_sum_error = float((table["probability_sum"] - 1).abs().max())
    decision_mismatches = _count_threshold_mismatches(
        table["decision"],
        table["probability_fake"],
    )

    strong_false_positives = int(
        ((table["label"] == 0) & (table["decision"] == "likely_fake")).sum()
    )
    strong_false_negatives = int(
        ((table["label"] == 1) & (table["decision"] == "likely_real")).sum()
    )
    strong_mask = table["decision"] != "uncertain"
    strong_correct = (
        ((table["label"] == 0) & (table["decision"] == "likely_real"))
        | ((table["label"] == 1) & (table["decision"] == "likely_fake"))
    )
    strong_count = int(strong_mask.sum())
    binary_error_mask = table["label"] != table["binary_prediction"]
    errors_moved_to_uncertain = int(
        (binary_error_mask & (table["decision"] == "uncertain")).sum()
    )
    metrics = classification_metrics(table["label"], table["binary_prediction"])

    summary = {
        "rows": metrics["rows"],
        "real_rows": metrics["real_rows"],
        "fake_rows": metrics["fake_rows"],
        "invalid_probability_rows": invalid_probability_rows,
        "maximum_probability_sum_error": maximum_sum_error,
        "decision_threshold_mismatches": decision_mismatches,
        "minimum_probability_fake": round(float(table["probability_fake"].min()), 6),
        "maximum_probability_fake": round(float(table["probability_fake"].max()), 6),
        "likely_real_count": int((table["decision"] == "likely_real").sum()),
        "uncertain_count": int((table["decision"] == "uncertain").sum()),
        "uncertain_real_count": int(
            ((table["decision"] == "uncertain") & (table["label"] == 0)).sum()
        ),
        "uncertain_fake_count": int(
            ((table["decision"] == "uncertain") & (table["label"] == 1)).sum()
        ),
        "likely_fake_count": int((table["decision"] == "likely_fake").sum()),
        "strong_decision_count": strong_count,
        "strong_decision_errors": strong_false_positives + strong_false_negatives,
        "strong_decision_coverage": round(strong_count / len(table), 4),
        "strong_decision_accuracy": round(
            float(strong_correct.sum() / strong_count) if strong_count else 0.0,
            4,
        ),
        "binary_correct": int((table["label"] == table["binary_prediction"]).sum()),
        "binary_errors_moved_to_uncertain": errors_moved_to_uncertain,
        "binary_accuracy": round(metrics["accuracy"], 4),
        "false_positives": metrics["false_positives"],
        "false_negatives": metrics["false_negatives"],
        "confusion_matrix": metrics["confusion_matrix"],
        "high_confidence_errors_90": int(
            (
                (table["error_type"] != "correct")
                & (table["predicted_confidence"] >= 0.90)
            ).sum()
        ),
        "strong_false_positives": strong_false_positives,
        "strong_false_negatives": strong_false_negatives,
    }
    return table, summary


def run_system_evaluation() -> dict:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    test_data, model, excluded_ids = load_evaluation_data()

    LOGGER.info("Checking the complete leakage-safe test set")
    _, test_summary = evaluate_test_set(test_data, model)
    invariant_checks_passed = (
        test_summary["invalid_probability_rows"] == 0
        and test_summary["maximum_probability_sum_error"] <= 1e-12
        and test_summary["decision_threshold_mismatches"] == 0
    )
    result = {
        "status": "passed" if invariant_checks_passed else "failed",
        "model": MODEL_PATH.name,
        "model_retrained": False,
        "test_set": {
            **test_summary,
            "exact_train_duplicates_excluded": int(len(excluded_ids)),
            "excluded_article_ids": excluded_ids,
        },
        "artifacts": {
            "metrics": METRICS_PATH.relative_to(PROJECT_ROOT).as_posix(),
        },
    }
    METRICS_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    result = run_system_evaluation()
    print("=== Historical Day 9 test-set evaluation ===")
    print(f"Status: {result['status']}")
    print(f"Test rows: {result['test_set']['rows']}")
    print(f"Binary accuracy: {result['test_set']['binary_accuracy']:.2%}")
    print(
        "Decisions: "
        f"{result['test_set']['likely_real_count']} likely_real, "
        f"{result['test_set']['uncertain_count']} uncertain, "
        f"{result['test_set']['likely_fake_count']} likely_fake"
    )
    print(f"Metrics saved: {METRICS_PATH}")

    if result["status"] != "passed":
        raise RuntimeError("One or more historical evaluation checks failed.")


if __name__ == "__main__":
    main()
