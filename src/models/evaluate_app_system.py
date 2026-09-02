"""Run the Day 9 end-to-end robustness tests for the Streamlit prediction system."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np
import pandas as pd

from app.streamlit_app import EXAMPLES, FACT_CHECK_WARNING, validate_news_input
from src.models.predict import (
    build_linguistic_explanation,
    classify_probability,
    predict_news_for_app,
)
from src.evaluation.data_utils import exclude_train_duplicates_from_test
from src.preprocessing.clean_text import combine_title_content

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = PROJECT_ROOT / "data" / "interim" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "interim" / "test.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "calibrated_tfidf_logreg.joblib"

REPORTS_DIR = PROJECT_ROOT / "reports"
CASE_RESULTS_PATH = REPORTS_DIR / "day9_system_test_cases.csv"
DEMO_EXAMPLES_PATH = REPORTS_DIR / "day9_demo_examples.csv"
METRICS_PATH = REPORTS_DIR / "day9_system_test_metrics.json"

LOGGER = logging.getLogger(__name__)


def build_system_cases() -> list[dict]:
    """Create representative valid, invalid, and unusual user inputs."""
    official = EXAMPLES["Raport institucional"]
    uncertain = EXAMPLES["Njoftim i shkurtër"]
    clickbait = EXAMPLES["Titull sensacional"]
    normal_content = (
        "Komuna njoftoi se punimet në rrugën kryesore do të fillojnë të hënën. "
        "Qarkullimi do të devijohet përkohësisht dhe banorët u kërkua të ndjekin "
        "sinjalistikën e vendosur gjatë punimeve."
    )
    long_content = (
        "Sipas raportit zyrtar, institucioni konfirmoi të dhënat e publikuara. " * 350
    )

    return [
        {
            "case_name": "empty_input",
            "category": "validation",
            "title": "",
            "content": "",
            "expect_prediction": False,
            "error_fragment": "të paktën titullin",
        },
        {
            "case_name": "punctuation_only",
            "category": "validation",
            "title": "!!!",
            "content": "🚨 ... ??? \u200b",
            "expect_prediction": False,
            "error_fragment": "shkronjë ose numër",
        },
        {
            "case_name": "title_only",
            "category": "partial_input",
            "title": "Njoftim nga komuna për punimet",
            "content": "",
            "expect_prediction": True,
            "warning_fragment": "vetëm titulli",
        },
        {
            "case_name": "content_only",
            "category": "partial_input",
            "title": "",
            "content": normal_content,
            "expect_prediction": True,
        },
        {
            "case_name": "very_short",
            "category": "length",
            "title": "Njoftim",
            "content": "Takimi mbahet nesër.",
            "expect_prediction": True,
            "warning_fragment": "më pak se 20 fjalë",
        },
        {
            "case_name": "normal_text",
            "category": "style",
            "title": "Njoftim për punimet në rrugë",
            "content": normal_content,
            "expect_prediction": True,
        },
        {
            "case_name": "very_long",
            "category": "length",
            "title": "Raport i zgjeruar",
            "content": long_content,
            "expect_prediction": True,
            "warning_fragment": "shumë i gjatë",
        },
        {
            "case_name": "above_maximum_length",
            "category": "validation",
            "title": "",
            "content": "lajm " * 20_001,
            "expect_prediction": False,
            "error_fragment": "Kufiri i analizës",
        },
        {
            "case_name": "uppercase_text",
            "category": "style",
            "title": "NJOFTIM SHUMË I RËNDËSISHËM",
            "content": " ".join(["INSTITUCIONI PUBLIKOI NJË NJOFTIM ZYRTAR"] * 6),
            "expect_prediction": True,
            "feature_check": "uppercase",
        },
        {
            "case_name": "many_exclamations",
            "category": "style",
            "title": "LAJM!!! NJOFTIM!!!",
            "content": "Ky tekst përmban shumë pikëçuditëse!!!!!!!! dhe fjalë të mjaftueshme për analizën e sistemit.",
            "expect_prediction": True,
            "warning_fragment": "më pak se 20 fjalë",
            "feature_check": "exclamations",
        },
        {
            "case_name": "clickbait_style",
            "category": "style",
            **clickbait,
            "expect_prediction": True,
            "expected_decision": "likely_fake",
            "feature_check": "sensational",
        },
        {
            "case_name": "official_style",
            "category": "style",
            **official,
            "expect_prediction": True,
            "expected_decision": "likely_real",
        },
        {
            "case_name": "source_markers",
            "category": "language_signals",
            "title": "Institucioni publikoi raportin",
            "content": (
                "Sipas raportit zyrtar, ministria deklaroi se të dhënat u kontrolluan. "
                "Policia konfirmoi informacionin dhe institucioni bëri të ditur se "
                "raporti i plotë do të publikohet pas përfundimit të verifikimit."
            ),
            "expect_prediction": True,
            "feature_check": "sources",
        },
        {
            "case_name": "without_diacritics",
            "category": "unicode",
            "title": "Njoftim per qytetaret",
            "content": (
                "Sipas raportit zyrtar institucioni konfirmoi se te dhenat do te "
                "publikohen neser dhe qytetaret mund te lexojne dokumentin e plote."
            ),
            "expect_prediction": True,
            "feature_check": "no_diacritics",
        },
        {
            "case_name": "unusual_unicode",
            "category": "unicode",
            "title": "C\u0327fare\u0308 e\u0308shte\u0308 ky njoftim?",
            "content": (
                "“Institucioni” bëri të ditur se të dhënat janë kontrolluar — më 3 gusht. "
                "Teksti përmban emoji ✅, thonjëza tipografike dhe shkronja të kombinuara."
            ),
            "expect_prediction": True,
            "feature_check": "unicode",
        },
        {
            "case_name": "uncertain_example",
            "category": "decision",
            **uncertain,
            "expect_prediction": True,
            "expected_decision": "uncertain",
        },
    ]


def check_expected_features(check_name: str, explanation: dict) -> bool:
    """Check the linguistic signal expected from a synthetic case."""
    if not check_name:
        return True
    if check_name == "uppercase":
        return explanation["uppercase_ratio"] >= 0.80
    if check_name == "exclamations":
        return explanation["exclamation_count"] >= 10
    if check_name == "sensational":
        return bool(explanation["sensational_words_found"])
    if check_name == "sources":
        required = {"sipas", "deklaroi", "konfirmoi"}
        return required.issubset(set(explanation["source_markers_found"]))
    if check_name == "no_diacritics":
        return explanation["diacritic_ratio"] == 0
    if check_name == "unicode":
        return explanation["diacritic_ratio"] > 0
    raise ValueError(f"Unknown feature check: {check_name}")


def evaluate_system_case(case: dict, model) -> dict:
    """Run validation, prediction, and output checks for one input case."""
    title = case["title"]
    content = case["content"]
    errors, warnings = validate_news_input(title, content)
    prediction_allowed = not errors
    issues: list[str] = []

    if prediction_allowed != case["expect_prediction"]:
        issues.append("validation behavior did not match expectation")

    expected_error = case.get("error_fragment")
    if expected_error and not any(expected_error in error for error in errors):
        issues.append("expected validation error was not shown")

    expected_warning = case.get("warning_fragment")
    if expected_warning and not any(expected_warning in warning for warning in warnings):
        issues.append("expected warning was not shown")

    result = None
    if prediction_allowed:
        try:
            result = predict_news_for_app(title, content, model=model)
        except Exception as error:  # The exception text is recorded in the CSV report.
            issues.append(f"prediction crashed: {type(error).__name__}: {error}")

    probability_real = None
    probability_fake = None
    probability_sum = None
    decision = "blocked"
    probability_check = None
    threshold_check = None
    feature_check = None

    if result is not None:
        probability_real = result["probability_real"]
        probability_fake = result["probability_fake"]
        probability_sum = probability_real + probability_fake
        decision = result["decision"]
        probability_check = (
            0 <= probability_real <= 1
            and 0 <= probability_fake <= 1
            and abs(probability_sum - 1) <= 0.0002
        )
        threshold_check = decision == classify_probability(probability_fake)
        feature_check = check_expected_features(
            case.get("feature_check", ""),
            result["linguistic_explanation"],
        )

        if not probability_check:
            issues.append("probabilities were outside valid bounds or did not sum to one")
        if not threshold_check:
            issues.append("decision did not match the 30/70 thresholds")
        if not feature_check:
            issues.append("linguistic feature check failed")
        if case.get("expected_decision") and decision != case["expected_decision"]:
            issues.append("decision did not match the expected demonstration outcome")

    combined_text = f"{title} {content}".strip()
    return {
        "case_name": case["case_name"],
        "category": case["category"],
        "input_characters": len(combined_text),
        "input_words": len(combined_text.split()),
        "title": title[:180],
        "content_excerpt": " ".join(content.split())[:240],
        "validation_status": "allowed" if prediction_allowed else "blocked",
        "errors": " | ".join(errors),
        "warnings": " | ".join(warnings),
        "decision": decision,
        "probability_real": probability_real,
        "probability_fake": probability_fake,
        "probability_sum": probability_sum,
        "probabilities_valid": probability_check,
        "decision_matches_thresholds": threshold_check,
        "linguistic_feature_check": feature_check,
        "passed": not issues,
        "issues": " | ".join(issues),
    }


def load_evaluation_data() -> tuple[pd.DataFrame, object, list[str]]:
    """Load the calibrated model and leakage-safe test data."""
    required_paths = [TRAIN_PATH, TEST_PATH, MODEL_PATH]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required artifacts: {missing}")

    train_data = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig", keep_default_na=False)
    test_data = pd.read_csv(TEST_PATH, encoding="utf-8-sig", keep_default_na=False)
    evaluation_test, excluded_ids = exclude_train_duplicates_from_test(train_data, test_data)
    model = joblib.load(MODEL_PATH)
    return evaluation_test.reset_index(drop=True), model, excluded_ids


def evaluate_test_set(test_data: pd.DataFrame, model) -> tuple[pd.DataFrame, dict]:
    """Evaluate probability and decision invariants on every test article."""
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
    decision_mismatches = int(
        sum(
            decision != classify_probability(float(probability))
            for decision, probability in zip(table["decision"], table["probability_fake"])
        )
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
    true_negatives = int(
        ((table["label"] == 0) & (table["binary_prediction"] == 0)).sum()
    )
    true_positives = int(
        ((table["label"] == 1) & (table["binary_prediction"] == 1)).sum()
    )

    summary = {
        "rows": int(len(table)),
        "real_rows": int((table["label"] == 0).sum()),
        "fake_rows": int((table["label"] == 1).sum()),
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
        "binary_accuracy": round(
            float((table["label"] == table["binary_prediction"]).mean()),
            4,
        ),
        "false_positives": int((table["error_type"] == "false_positive").sum()),
        "false_negatives": int((table["error_type"] == "false_negative").sum()),
        "confusion_matrix": [
            [true_negatives, int((table["error_type"] == "false_positive").sum())],
            [int((table["error_type"] == "false_negative").sum()), true_positives],
        ],
        "high_confidence_errors_90": int(
            ((table["error_type"] != "correct") & (table["predicted_confidence"] >= 0.90)).sum()
        ),
        "strong_false_positives": strong_false_positives,
        "strong_false_negatives": strong_false_negatives,
    }
    return table, summary


def describe_demo(demo_type: str, row: pd.Series, explanation: dict) -> str:
    """Create a short, non-causal interpretation for a demo article."""
    signals = []
    if explanation["source_markers_found"]:
        signals.append("përdor tregues burimi")
    if explanation["sensational_words_found"]:
        signals.append("përmban shprehje sensacionale")
    if explanation["exclamation_count"]:
        signals.append(f"ka {explanation['exclamation_count']} pikëçuditëse")
    if explanation["word_count"] >= 500:
        signals.append("është relativisht i gjatë dhe formal")
    if explanation["word_count"] < 20:
        signals.append("është shumë i shkurtër")
    if explanation["diacritic_ratio"] == 0:
        signals.append("nuk përdor ë/ç")

    signal_text = ", ".join(signals) if signals else "nuk ka një sinjal të vetëm dominues"
    if demo_type in {"false_positive", "false_negative", "high_confidence_error"}:
        return (
            f"Modeli gaboi edhe pse teksti {signal_text}. Kjo tregon se TF-IDF njeh "
            "ngjashmëri fjalori dhe stili, jo vërtetësinë faktike."
        )
    if demo_type == "uncertain":
        return (
            f"Probabiliteti është pranë mesit dhe teksti {signal_text}; modeli nuk ka "
            "siguri të mjaftueshme."
        )
    return f"Shembull demonstrues ku teksti {signal_text}."


def select_demo_examples(predictions: pd.DataFrame, model) -> pd.DataFrame:
    """Select correct decisions and representative calibrated model errors."""
    likely_real_pool = predictions.loc[
        (predictions["label"] == 0) & (predictions["decision"] == "likely_real")
    ].copy()
    likely_real_pool["distance"] = (likely_real_pool["probability_fake"] - 0.15).abs()

    likely_fake_pool = predictions.loc[
        (predictions["label"] == 1) & (predictions["decision"] == "likely_fake")
    ].copy()
    likely_fake_pool["distance"] = (likely_fake_pool["probability_fake"] - 0.85).abs()

    uncertain_pool = predictions.loc[predictions["decision"] == "uncertain"].copy()
    uncertain_pool["distance"] = (uncertain_pool["probability_fake"] - 0.5).abs()

    false_positive = predictions.loc[
        predictions["error_type"] == "false_positive"
    ].nlargest(1, "probability_fake")
    false_negative = predictions.loc[
        predictions["error_type"] == "false_negative"
    ].nsmallest(1, "probability_fake")
    chosen_error_ids = set(
        pd.concat([false_positive, false_negative])["article_id"].astype(str)
    )
    remaining_errors = predictions.loc[
        (predictions["error_type"] != "correct")
        & (~predictions["article_id"].astype(str).isin(chosen_error_ids))
    ]
    high_confidence_error = remaining_errors.nlargest(1, "predicted_confidence")

    selections = [
        ("likely_real", likely_real_pool.nsmallest(1, "distance")),
        ("likely_fake", likely_fake_pool.nsmallest(1, "distance")),
        ("uncertain", uncertain_pool.nsmallest(1, "distance")),
        ("false_positive", false_positive),
        ("false_negative", false_negative),
        ("high_confidence_error", high_confidence_error),
    ]

    rows = []
    for demo_type, selection in selections:
        if selection.empty:
            raise ValueError(f"No article available for demo type: {demo_type}")
        source_row = selection.iloc[0]
        app_result = predict_news_for_app(
            source_row["title"],
            source_row["content"],
            model=model,
        )
        explanation = app_result["linguistic_explanation"]
        binary_prediction = "fake" if app_result["probability_fake"] >= 0.5 else "real"
        true_label = str(source_row["label_name"])
        error_type = (
            "false_positive"
            if true_label == "real" and binary_prediction == "fake"
            else "false_negative"
            if true_label == "fake" and binary_prediction == "real"
            else "correct_or_uncertain"
        )
        rows.append(
            {
                "demo_type": demo_type,
                "article_id": str(source_row["article_id"]),
                "pair_id": int(source_row["pair_id"]),
                "true_label": true_label,
                "title": str(source_row["title"]),
                "content_excerpt": " ".join(str(source_row["content"]).split())[:350],
                "app_decision": app_result["decision"],
                "binary_prediction": binary_prediction,
                "probability_real": app_result["probability_real"],
                "probability_fake": app_result["probability_fake"],
                "predicted_confidence": round(
                    max(app_result["probability_real"], app_result["probability_fake"]),
                    4,
                ),
                "error_type": error_type,
                "word_count": explanation["word_count"],
                "text_length": explanation["text_length"],
                "exclamation_count": explanation["exclamation_count"],
                "uppercase_ratio": round(explanation["uppercase_ratio"], 6),
                "diacritic_ratio": round(explanation["diacritic_ratio"], 6),
                "sensational_words_found": ", ".join(
                    explanation["sensational_words_found"]
                ),
                "source_markers_found": ", ".join(explanation["source_markers_found"]),
                "uncertainty_markers_found": ", ".join(
                    explanation["uncertainty_markers_found"]
                ),
                "interpretation": describe_demo(demo_type, source_row, explanation),
            }
        )
    return pd.DataFrame(rows)


def run_system_evaluation() -> dict:
    """Run all Day 9 checks and save reproducible result files."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    test_data, model, excluded_ids = load_evaluation_data()

    LOGGER.info("Running synthetic and edge-case tests")
    case_results = pd.DataFrame(
        [evaluate_system_case(case, model) for case in build_system_cases()]
    )

    LOGGER.info("Checking the complete leakage-safe test set")
    prediction_table, test_summary = evaluate_test_set(test_data, model)

    LOGGER.info("Selecting demonstration and error examples")
    demo_examples = select_demo_examples(prediction_table, model)

    case_results.to_csv(CASE_RESULTS_PATH, index=False, encoding="utf-8-sig")
    demo_examples.to_csv(DEMO_EXAMPLES_PATH, index=False, encoding="utf-8-sig")

    invariant_checks_passed = (
        test_summary["invalid_probability_rows"] == 0
        and test_summary["maximum_probability_sum_error"] <= 1e-12
        and test_summary["decision_threshold_mismatches"] == 0
    )
    result = {
        "status": "passed"
        if bool(case_results["passed"].all()) and invariant_checks_passed
        else "failed",
        "model": "calibrated_tfidf_logreg.joblib",
        "model_retrained": False,
        "fact_check_warning_configured": "nuk verifikon faktet" in FACT_CHECK_WARNING,
        "system_cases": {
            "total": int(len(case_results)),
            "passed": int(case_results["passed"].sum()),
            "failed": int((~case_results["passed"]).sum()),
            "blocked_as_expected": int(
                ((case_results["validation_status"] == "blocked") & case_results["passed"]).sum()
            ),
            "predicted_without_crash": int(
                ((case_results["validation_status"] == "allowed") & case_results["passed"]).sum()
            ),
        },
        "test_set": {
            **test_summary,
            "exact_train_duplicates_excluded": int(len(excluded_ids)),
            "excluded_article_ids": excluded_ids,
        },
        "demo_examples": demo_examples[
            [
                "demo_type",
                "article_id",
                "true_label",
                "app_decision",
                "probability_real",
                "probability_fake",
            ]
        ].to_dict(orient="records"),
        "artifacts": {
            "case_results": CASE_RESULTS_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "demo_examples": DEMO_EXAMPLES_PATH.relative_to(PROJECT_ROOT).as_posix(),
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
    print("=== Day 9 system evaluation ===")
    print(f"Status: {result['status']}")
    print(
        "System cases: "
        f"{result['system_cases']['passed']}/{result['system_cases']['total']} passed"
    )
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
        raise RuntimeError("One or more Day 9 system checks failed.")


if __name__ == "__main__":
    main()
