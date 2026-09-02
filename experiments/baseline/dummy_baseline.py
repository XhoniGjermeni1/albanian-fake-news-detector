"""Evaluate a most-frequent DummyClassifier on the frozen internal split."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.data_utils import (  # noqa: E402
    exclude_train_duplicates_from_test,
    refresh_model_text,
)
from src.evaluation.metrics import classification_metrics  # noqa: E402

TRAIN_PATH = PROJECT_ROOT / "data" / "interim" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "interim" / "test.csv"
FINAL_COMPARISON_PATH = PROJECT_ROOT / "reports" / "day17_final_model_comparison.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "experiments" / "baseline"
METRICS_PATH = OUTPUT_DIR / "dummy_baseline_metrics.json"
COMPARISON_PATH = OUTPUT_DIR / "dummy_baseline_comparison.csv"
REPORT_PATH = OUTPUT_DIR / "dummy_baseline.md"


def load_frozen_split() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load train and leakage-clean internal test data."""
    train = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig", keep_default_na=False)
    test = pd.read_csv(TEST_PATH, encoding="utf-8-sig", keep_default_na=False)
    train, _ = refresh_model_text(train)
    test, _ = refresh_model_text(test)
    clean_test, excluded_ids = exclude_train_duplicates_from_test(train, test)
    return train, clean_test, excluded_ids


def evaluate_dummy_baseline() -> tuple[dict, pd.DataFrame]:
    """Fit the trivial baseline and compare it with the frozen models."""
    train, test, excluded_ids = load_frozen_split()
    classifier = DummyClassifier(strategy="most_frequent", random_state=42)
    classifier.fit(train[["model_text"]], train["label"])
    predictions = classifier.predict(test[["model_text"]])
    metrics = classification_metrics(test["label"], predictions)
    metrics.update(
        {
            "model": "dummy_most_frequent",
            "strategy": "most_frequent",
            "majority_class": int(classifier.classes_[classifier.class_prior_.argmax()]),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "test_rows_before_duplicate_exclusion": 799,
            "excluded_train_test_duplicates": int(len(excluded_ids)),
            "excluded_article_ids": excluded_ids,
            "selection_role": "academic_trivial_baseline_only",
        }
    )

    final_models = pd.read_csv(FINAL_COMPARISON_PATH, keep_default_na=False)
    columns = [
        "model",
        "accuracy",
        "f1_weighted",
        "f1_fake",
        "recall_real",
        "recall_fake",
        "false_positives",
        "false_negatives",
    ]
    dummy_row = pd.DataFrame([{column: metrics[column] for column in columns}])
    comparison = pd.concat([dummy_row, final_models[columns]], ignore_index=True)
    return metrics, comparison


def save_outputs(metrics: dict, comparison: pd.DataFrame) -> None:
    """Save the baseline without changing any frozen report."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    comparison.to_csv(COMPARISON_PATH, index=False, encoding="utf-8-sig")

    dummy = comparison.loc[comparison["model"].eq("dummy_most_frequent")].iloc[0]
    logistic = comparison.loc[comparison["model"].eq("baseline_word_logreg")].iloc[0]
    final = comparison.loc[comparison["model"].eq("final_word_char_svm")].iloc[0]
    REPORT_PATH.write_text(
        f"""# Trivial Dummy Baseline

Ky eksperiment përdor `DummyClassifier(strategy=\"most_frequent\")` mbi të
njëjtin train set dhe test set të brendshëm prej 792 artikujsh. Shtatë kopjet
ekzakte train-test përjashtohen me të njëjtën logjikë si vlerësimi final.

| Modeli | Accuracy | F1 weighted | F1 fake |
|---|---:|---:|---:|
| Dummy most-frequent | {float(dummy['accuracy']):.4f} | {float(dummy['f1_weighted']):.4f} | {float(dummy['f1_fake']):.4f} |
| Word TF-IDF + Logistic Regression | {float(logistic['accuracy']):.4f} | {float(logistic['f1_weighted']):.4f} | {float(logistic['f1_fake']):.4f} |
| Word + Character TF-IDF + Linear SVM | {float(final['accuracy']):.4f} | {float(final['f1_weighted']):.4f} | {float(final['f1_fake']):.4f} |

Dummy baseline parashikon vetëm klasën shumicë dhe nuk është kandidat për
aplikacionin. Ai tregon se modelet reale mësojnë sinjal përtej shpërndarjes së
klasave.
""",
        encoding="utf-8",
    )


def main() -> None:
    metrics, comparison = evaluate_dummy_baseline()
    save_outputs(metrics, comparison)
    print(
        f"Dummy accuracy={metrics['accuracy']:.4f}, "
        f"F1 weighted={metrics['f1_weighted']:.4f}, "
        f"F1 fake={metrics['f1_fake']:.4f}"
    )


if __name__ == "__main__":
    main()
