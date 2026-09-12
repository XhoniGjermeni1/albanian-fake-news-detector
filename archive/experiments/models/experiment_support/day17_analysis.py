"""Verify and freeze the Day 16 candidate without retraining it."""

from __future__ import annotations

import json
import logging
import platform
import shutil
import sys
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.linguistic_features import (  # noqa: E402
    extract_linguistic_features,
)
from src.evaluation.data_utils import (  # noqa: E402
    LENGTH_DISPLAY,
    LENGTH_LABELS,
    add_word_counts,
    exclude_train_duplicates_from_test,
    refresh_model_text,
)
from src.evaluation.experiment_utils import (  # noqa: E402
    escaped_dataframe_to_markdown as dataframe_to_markdown,
    file_sha256,
)
from src.evaluation.metrics import (  # noqa: E402
    rounded_metrics,
)
from src.models.experiment_support.day16_analysis import (  # noqa: E402
    evaluate_length_behavior,
    high_confidence_error_rows,
    model_comparison_table,
)
from src.models.prediction_utils import (  # noqa: E402
    DEFAULT_FAKE_THRESHOLD,
    DEFAULT_REAL_THRESHOLD,
    classify_probability,
)
from src.models.predict_final import (  # noqa: E402
    FINAL_FAKE_THRESHOLD,
    FINAL_MANIFEST_PATH,
    FINAL_MODEL_ID,
    FINAL_MODEL_PATH,
    FINAL_MODEL_VERSION,
    FINAL_REAL_THRESHOLD,
    predict_final_news,
    prepare_final_model_text,
)


TRAIN_PATH = PROJECT_ROOT / "data" / "interim" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "interim" / "test.csv"
EXTERNAL_PATH = PROJECT_ROOT / "data" / "external" / "external_news.csv"
SOURCE_MODEL_PATH = (
    PROJECT_ROOT / "models" / "day16_word_char_linear_svm_calibrated.joblib"
)
BASELINE_MODEL_PATH = PROJECT_ROOT / "models" / "calibrated_tfidf_logreg.joblib"
DAY16_SELECTION_PATH = PROJECT_ROOT / "reports" / "day16_selection.json"
DAY16_METRICS_PATH = PROJECT_ROOT / "reports" / "day16_metrics.json"
DAY16_FOLDS_PATH = PROJECT_ROOT / "reports" / "day16_calibration_fold_metrics.csv"
DAY16_INTERNAL_PREDICTIONS_PATH = (
    PROJECT_ROOT / "reports" / "day16_internal_predictions.csv"
)
DAY16_EXTERNAL_PREDICTIONS_PATH = (
    PROJECT_ROOT / "reports" / "day16_external_predictions.csv"
)
STREAMLIT_APP_PATH = PROJECT_ROOT / "app" / "streamlit_app.py"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
VERIFICATION_PATH = REPORTS_DIR / "day17_artifact_verification.json"
REGRESSION_PATH = REPORTS_DIR / "day17_regression_checks.csv"
INTERNAL_COMPARISON_PATH = REPORTS_DIR / "day17_final_model_comparison.csv"
INTERNAL_PREDICTIONS_PATH = REPORTS_DIR / "day17_final_internal_predictions.csv"
EXTERNAL_COMPARISON_PATH = REPORTS_DIR / "day17_final_external_evaluation.csv"
EXTERNAL_PREDICTIONS_PATH = REPORTS_DIR / "day17_final_external_predictions.csv"
LENGTH_METRICS_PATH = REPORTS_DIR / "day17_final_length_metrics.csv"
SPECIAL_COHORTS_PATH = REPORTS_DIR / "day17_final_special_cohorts.csv"
DEMO_CASES_PATH = REPORTS_DIR / "day17_final_demo_cases.csv"
HIGH_CONFIDENCE_ERRORS_PATH = REPORTS_DIR / "day17_final_high_confidence_errors.csv"
METRICS_PATH = REPORTS_DIR / "day17_final_metrics.json"
REPORT_PATH = REPORTS_DIR / "day17_final_model.md"
MODEL_COMPARISON_FIGURE_PATH = FIGURES_DIR / "day17_final_model_comparison.png"
LENGTH_FIGURE_PATH = FIGURES_DIR / "day17_final_length_performance.png"

FINAL_MODEL_NAME = "final_word_char_svm"
BASELINE_MODEL_NAME = "baseline_word_logreg"
EXPECTED_TRAIN_ROWS = 3195
EXPECTED_TEST_ROWS = 792
EXPECTED_EXTERNAL_ROWS = 40
HIGH_CONFIDENCE = 0.90
LOGGER = logging.getLogger(__name__)

REGRESSION_CASES = [
    {
        "case_id": "official_style",
        "title": "Institucioni publikoi njoftimin zyrtar",
        "content": (
            "Ministria njoftoi se vendimi u miratua sot dhe dokumenti i plote "
            "u publikua ne faqen zyrtare per konsultim publik."
        ),
    },
    {
        "case_id": "clickbait_style",
        "title": "E PABESUESHME! Ky lajm po trondit rrjetin",
        "content": (
            "Shperndajeni menjehere kete lajm qe mediat nuk duan ta tregojne. "
            "Zbulimi i fundit ka habitur te gjithe!"
        ),
    },
    {
        "case_id": "title_only",
        "title": "Kuvendi miraton projektligjin ne seance plenare",
        "content": "",
    },
    {
        "case_id": "content_only",
        "title": "",
        "content": (
            "Sipas njoftimit, sherbimi do te jete i disponueshem nga dita e hene "
            "per te gjithe qytetaret."
        ),
    },
    {
        "case_id": "unicode_nfc",
        "title": "Çështja u diskutua në mbledhje",
        "content": "Është publikuar një përmbledhje me të dhënat kryesore.",
    },
    {
        "case_id": "unicode_nfd",
        "title": unicodedata.normalize("NFD", "Çështja u diskutua në mbledhje"),
        "content": unicodedata.normalize(
            "NFD", "Është publikuar një përmbledhje me të dhënat kryesore."
        ),
    },
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_frozen_selection() -> tuple[dict, dict]:
    """Verify the Day 16 decision before touching the final artifact."""
    selection = load_json(DAY16_SELECTION_PATH)
    metrics = load_json(DAY16_METRICS_PATH)
    fixed = selection.get("fixed_configuration", {})
    calibration = selection.get("calibration", {})
    thresholds = selection.get("thresholds", {})

    checks = {
        "representation_is_word_char": fixed.get("representation")
        == "word_char_tfidf",
        "classifier_is_linear_svm": fixed.get("classifier") == "linear_svm",
        "c_is_1": float(fixed.get("c_value", -1)) == 1.0,
        "calibration_is_sigmoid": calibration.get("selected_method") == "sigmoid",
        "lower_threshold_is_030": float(thresholds.get("lower_threshold", -1))
        == FINAL_REAL_THRESHOLD,
        "upper_threshold_is_070": float(thresholds.get("upper_threshold", -1))
        == FINAL_FAKE_THRESHOLD,
        "external_not_used_for_selection": selection.get("external_results_used")
        is False,
        "day16_integrity_passed": metrics.get("integrity", {}).get(
            "all_frozen_artifacts_unchanged"
        )
        is True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Day 16 frozen selection failed checks: {failed}")
    return selection, metrics


def verify_model_configuration(model) -> dict:
    """Inspect the fitted sklearn object and require the exact final setup."""
    if not isinstance(model, CalibratedClassifierCV):
        raise TypeError("Final candidate is not CalibratedClassifierCV.")
    if model.method != "sigmoid" or model.ensemble is not False:
        raise ValueError("Final candidate does not use non-ensemble sigmoid calibration.")
    if list(model.classes_) != [0, 1]:
        raise ValueError(f"Unexpected classes: {model.classes_}")
    if len(model.calibrated_classifiers_) != 1:
        raise ValueError("Expected one fitted calibrated classifier.")

    calibrated = model.calibrated_classifiers_[0]
    pipeline = calibrated.estimator
    if not isinstance(pipeline, Pipeline):
        raise TypeError("Calibrated estimator is not an sklearn Pipeline.")
    features = pipeline.named_steps.get("features")
    classifier = pipeline.named_steps.get("classifier")
    if not isinstance(features, FeatureUnion):
        raise TypeError("Final representation is not a FeatureUnion.")
    if not isinstance(classifier, LinearSVC):
        raise TypeError("Final classifier is not LinearSVC.")

    vectorizers = dict(features.transformer_list)
    if set(vectorizers) != {"word", "character"}:
        raise ValueError(f"Unexpected feature branches: {set(vectorizers)}")
    word = vectorizers["word"]
    character = vectorizers["character"]
    if not isinstance(word, TfidfVectorizer) or not isinstance(
        character, TfidfVectorizer
    ):
        raise TypeError("Both final feature branches must be TfidfVectorizer.")

    expected_word = {
        "analyzer": "word",
        "lowercase": False,
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_features": 30000,
    }
    expected_character = {
        "analyzer": "char_wb",
        "lowercase": False,
        "ngram_range": (3, 5),
        "min_df": 2,
        "max_features": 50000,
    }
    for parameter, expected in expected_word.items():
        if getattr(word, parameter) != expected:
            raise ValueError(f"Unexpected Word TF-IDF {parameter}.")
    for parameter, expected in expected_character.items():
        if getattr(character, parameter) != expected:
            raise ValueError(f"Unexpected Character TF-IDF {parameter}.")
    if float(classifier.C) != 1.0:
        raise ValueError("Final Linear SVM C is not 1.0.")
    if classifier.class_weight != "balanced":
        raise ValueError("Final Linear SVM class_weight changed.")

    calibrator_names = [item.__class__.__name__ for item in calibrated.calibrators]
    if calibrator_names != ["_SigmoidCalibration"]:
        raise ValueError(f"Unexpected fitted calibrators: {calibrator_names}")
    return {
        "sklearn_object": model.__class__.__name__,
        "calibration_method": model.method,
        "calibration_ensemble": bool(model.ensemble),
        "calibration_folds": len(model.cv),
        "fitted_calibrators": calibrator_names,
        "feature_union_branches": list(vectorizers),
        "word_tfidf": expected_word,
        "character_tfidf": expected_character,
        "classifier": {
            "name": classifier.__class__.__name__,
            "C": float(classifier.C),
            "class_weight": classifier.class_weight,
            "max_iter": int(classifier.max_iter),
            "random_state": int(classifier.random_state),
        },
        "classes": [int(value) for value in model.classes_],
    }


def verify_preprocessing_contract() -> dict:
    nfc_title = "Çështja për ëndrrën"
    nfc_content = "Është një përmbledhje e shkurtër."
    nfd_title = unicodedata.normalize("NFD", nfc_title)
    nfd_content = unicodedata.normalize("NFD", nfc_content)
    prepared_nfc = prepare_final_model_text(nfc_title, nfc_content)
    prepared_nfd = prepare_final_model_text(nfd_title, nfd_content)
    if prepared_nfc != prepared_nfd:
        raise ValueError("NFC and NFD inputs do not produce identical model text.")
    if not unicodedata.is_normalized("NFC", prepared_nfd):
        raise ValueError("Final preprocessing output is not Unicode NFC.")
    if DEFAULT_REAL_THRESHOLD != FINAL_REAL_THRESHOLD:
        raise ValueError("Application real threshold differs from final threshold.")
    if DEFAULT_FAKE_THRESHOLD != FINAL_FAKE_THRESHOLD:
        raise ValueError("Application fake threshold differs from final threshold.")
    return {
        "function": "src.preprocessing.clean_text.combine_title_content",
        "unicode_normalization": "NFC",
        "whitespace": "collapsed",
        "title_content_separator": ". ",
        "lowercase_removed": False,
        "punctuation_removed": False,
        "nfc_nfd_equivalence_passed": True,
        "application_thresholds_match": True,
    }


def freeze_artifact(source_hash: str) -> tuple[object, dict]:
    """Copy the candidate byte-for-byte and refuse to overwrite another v1."""
    FINAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if FINAL_MODEL_PATH.exists():
        existing_hash = file_sha256(FINAL_MODEL_PATH)
        if existing_hash != source_hash:
            raise FileExistsError(
                "Final v1 already exists with a different hash; refusing overwrite."
            )
        copied_now = False
    else:
        shutil.copy2(SOURCE_MODEL_PATH, FINAL_MODEL_PATH)
        copied_now = True

    final_hash = file_sha256(FINAL_MODEL_PATH)
    if final_hash != source_hash:
        raise RuntimeError("Final artifact is not byte-identical to Day 16 candidate.")
    model = joblib.load(FINAL_MODEL_PATH)
    return model, {
        "source_path": str(SOURCE_MODEL_PATH.relative_to(PROJECT_ROOT)),
        "final_path": str(FINAL_MODEL_PATH.relative_to(PROJECT_ROOT)),
        "source_sha256": source_hash,
        "final_sha256": final_hash,
        "byte_identical": True,
        "copied_during_this_run": copied_now,
        "size_bytes": FINAL_MODEL_PATH.stat().st_size,
        "size_mb": round(FINAL_MODEL_PATH.stat().st_size / (1024 * 1024), 3),
    }


def load_evaluation_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    raw_train = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig", keep_default_na=False)
    raw_test = pd.read_csv(TEST_PATH, encoding="utf-8-sig", keep_default_na=False)
    train, stale_train = refresh_model_text(raw_train)
    test, stale_test = refresh_model_text(raw_test)
    test, excluded_ids = exclude_train_duplicates_from_test(train, test)
    test = add_word_counts(test)

    external = pd.read_csv(EXTERNAL_PATH, encoding="utf-8", keep_default_na=False)
    if set(external["label"]) != {"real", "fake"}:
        raise ValueError("Unexpected external labels.")
    external["label"] = external["label"].map({"real": 0, "fake": 1})
    external, stale_external = refresh_model_text(external)
    external = add_word_counts(external)

    if len(train) != EXPECTED_TRAIN_ROWS:
        raise ValueError(f"Expected {EXPECTED_TRAIN_ROWS} train rows, found {len(train)}.")
    if len(test) != EXPECTED_TEST_ROWS:
        raise ValueError(f"Expected {EXPECTED_TEST_ROWS} test rows, found {len(test)}.")
    if len(external) != EXPECTED_EXTERNAL_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_EXTERNAL_ROWS} external rows, found {len(external)}."
        )
    return train, test, external, {
        "train_rows": int(len(train)),
        "train_real": int(train["label"].eq(0).sum()),
        "train_fake": int(train["label"].eq(1).sum()),
        "internal_test_rows": int(len(test)),
        "internal_test_real": int(test["label"].eq(0).sum()),
        "internal_test_fake": int(test["label"].eq(1).sum()),
        "external_rows": int(len(external)),
        "external_real": int(external["label"].eq(0).sum()),
        "external_fake": int(external["label"].eq(1).sum()),
        "excluded_exact_train_test_duplicates": int(len(excluded_ids)),
        "excluded_article_ids": excluded_ids,
        "stale_train_model_text_refreshed": stale_train,
        "stale_test_model_text_refreshed": stale_test,
        "stale_external_model_text_refreshed": stale_external,
    }


def run_regression_checks(
    source_model,
    final_model,
    reloaded_model,
    evaluation_test: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    cases = [dict(case) for case in REGRESSION_CASES]
    regression_anchors = {
        "known_likely_real": ("true_1594", "likely_real"),
        "known_uncertain": ("true_585", "uncertain"),
        "known_likely_fake": ("fake_531", "likely_fake"),
    }
    for case_id, (article_id, expected_decision) in regression_anchors.items():
        matches = evaluation_test.loc[evaluation_test["article_id"].eq(article_id)]
        if len(matches) != 1:
            raise RuntimeError(f"Regression anchor {article_id} is unavailable.")
        article = matches.iloc[0]
        cases.append(
            {
                "case_id": case_id,
                "title": article["title"],
                "content": article["content"],
                "expected_decision": expected_decision,
            }
        )

    rows = []
    for case in cases:
        source = predict_final_news(case["title"], case["content"], model=source_model)
        final = predict_final_news(case["title"], case["content"], model=final_model)
        reloaded = predict_final_news(
            case["title"], case["content"], model=reloaded_model
        )
        model_text = prepare_final_model_text(case["title"], case["content"])
        sum_error = abs(final["probability_real"] + final["probability_fake"] - 1.0)
        expected_decision = classify_probability(
            final["probability_fake"],
            FINAL_REAL_THRESHOLD,
            FINAL_FAKE_THRESHOLD,
        )
        rows.append(
            {
                "case_id": case["case_id"],
                "probability_real": final["probability_real"],
                "probability_fake": final["probability_fake"],
                "probability_sum_error": sum_error,
                "decision": final["decision"],
                "expected_decision": case.get("expected_decision", "not_fixed"),
                "matches_expected_anchor": (
                    final["decision"] == case["expected_decision"]
                    if "expected_decision" in case
                    else True
                ),
                "decision_matches_thresholds": final["decision"]
                == expected_decision,
                "source_final_max_difference": max(
                    abs(source["probability_real"] - final["probability_real"]),
                    abs(source["probability_fake"] - final["probability_fake"]),
                ),
                "reload_max_difference": max(
                    abs(reloaded["probability_real"] - final["probability_real"]),
                    abs(reloaded["probability_fake"] - final["probability_fake"]),
                ),
                "model_text_is_nfc": unicodedata.is_normalized("NFC", model_text),
            }
        )
    regression = pd.DataFrame(rows)

    expected_model_text = pd.Series(
        [
            prepare_final_model_text(row.title, row.content)
            for row in evaluation_test.itertuples(index=False)
        ],
        index=evaluation_test.index,
    )
    preprocessing_mismatches = int(
        expected_model_text.ne(evaluation_test["model_text"]).sum()
    )
    nfc_mismatches = int(
        sum(
            not unicodedata.is_normalized("NFC", value)
            for value in expected_model_text
        )
    )
    unicode_rows = regression.loc[
        regression["case_id"].isin(["unicode_nfc", "unicode_nfd"])
    ]
    unicode_probability_difference = float(
        unicode_rows["probability_fake"].max()
        - unicode_rows["probability_fake"].min()
    )
    summary = {
        "cases": int(len(regression)),
        "probabilities_in_range": bool(
            regression["probability_real"].between(0, 1).all()
            and regression["probability_fake"].between(0, 1).all()
        ),
        "maximum_probability_sum_error": float(
            regression["probability_sum_error"].max()
        ),
        "all_decisions_match_thresholds": bool(
            regression["decision_matches_thresholds"].all()
        ),
        "all_anchor_decisions_stable": bool(
            regression["matches_expected_anchor"].all()
        ),
        "decision_levels_covered": sorted(regression["decision"].unique().tolist()),
        "maximum_source_final_difference": float(
            regression["source_final_max_difference"].max()
        ),
        "maximum_reload_difference": float(
            regression["reload_max_difference"].max()
        ),
        "unicode_nfc_nfd_probability_difference": unicode_probability_difference,
        "evaluation_prediction_preprocessing_mismatches": preprocessing_mismatches,
        "evaluation_model_text_non_nfc_rows": nfc_mismatches,
    }
    if not summary["probabilities_in_range"]:
        raise RuntimeError("Regression probabilities are outside [0, 1].")
    if summary["maximum_probability_sum_error"] > 1e-12:
        raise RuntimeError("Regression probabilities do not sum to one.")
    if not summary["all_decisions_match_thresholds"]:
        raise RuntimeError("Regression decision does not match frozen thresholds.")
    if not summary["all_anchor_decisions_stable"]:
        raise RuntimeError("A known regression anchor changed decision.")
    if set(summary["decision_levels_covered"]) != {
        "likely_real",
        "uncertain",
        "likely_fake",
    }:
        raise RuntimeError("Regression checks do not cover all decision levels.")
    if summary["maximum_source_final_difference"] != 0.0:
        raise RuntimeError("Copied final artifact differs from source predictions.")
    if summary["maximum_reload_difference"] != 0.0:
        raise RuntimeError("Final artifact prediction changes after reload.")
    if unicode_probability_difference != 0.0:
        raise RuntimeError("NFC and NFD inputs produce different predictions.")
    if preprocessing_mismatches or nfc_mismatches:
        raise RuntimeError("Evaluation and final prediction preprocessing differ.")
    return regression, summary


def maximum_day16_probability_difference(
    predictions: pd.DataFrame,
    day16_path: Path,
    id_column: str,
) -> float:
    day16 = pd.read_csv(day16_path)
    day16 = day16.loc[day16["model"].eq("new_calibrated_svm")]
    final = predictions.loc[predictions["model"].eq(FINAL_MODEL_NAME)]
    merged = final[[id_column, "probability_fake"]].merge(
        day16[[id_column, "probability_fake"]],
        on=id_column,
        suffixes=("_final", "_day16"),
        validate="one_to_one",
    )
    if len(merged) != len(final):
        raise RuntimeError("Day 16 and Day 17 prediction IDs do not align.")
    return float(
        (
            merged["probability_fake_final"]
            - merged["probability_fake_day16"]
        )
        .abs()
        .max()
    )


def observed_signal_summary(features: dict) -> str:
    signals = [f"gjatesia {int(features['word_count'])} fjale"]
    if features["sensational_found"]:
        signals.append(f"markues sensacional: {features['sensational_found']}")
    if features["source_indicators_found"]:
        signals.append(f"tregues burimi: {features['source_indicators_found']}")
    if int(features["exclamation_count"]):
        signals.append(f"{int(features['exclamation_count'])} pikëçuditëse")
    signals.append(f"uppercase ratio {float(features['uppercase_char_ratio']):.3f}")
    signals.append(f"diacritic ratio {float(features['diacritic_ratio']):.3f}")
    return "; ".join(signals)


def demo_explanation(demo_type: str, row: pd.Series, features: dict) -> str:
    probability = float(row["probability_fake"])
    if demo_type == "likely_real_correct":
        opening = "Vendimi likely_real përputhet me label-in real."
    elif demo_type == "likely_fake_correct":
        opening = "Vendimi likely_fake përputhet me label-in fake."
    elif demo_type == "uncertain":
        opening = (
            f"Probability fake {probability:.3f} bie brenda zonës 0.30-0.70."
        )
    elif demo_type == "false_positive":
        opening = "Artikulli real u shty gabimisht drejt fake."
    elif demo_type == "false_negative":
        opening = "Artikulli fake u shty gabimisht drejt real."
    else:
        opening = (
            "Ky është një gabim i rëndësishëm sepse confidence kalon 90%."
        )
    return (
        f"{opening} Sinjale të vëzhguara: {observed_signal_summary(features)}. "
        "Këto sinjale përshkruajnë sjelljen e modelit, jo vërtetësinë faktike."
    )


def select_demo_cases(final_internal: pd.DataFrame) -> pd.DataFrame:
    """Select six distinct and reproducible thesis demonstration cases."""
    data = final_internal.copy()
    data["confidence"] = data[["probability_real", "probability_fake"]].max(axis=1)
    used: set[str] = set()
    selected: list[tuple[str, pd.Series]] = []

    def choose(demo_type: str, subset: pd.DataFrame, sort_columns, ascending) -> None:
        available = subset.loc[~subset["article_id"].isin(used)].sort_values(
            sort_columns, ascending=ascending
        )
        if available.empty:
            raise RuntimeError(f"No candidate found for demo type {demo_type}.")
        row = available.iloc[0]
        used.add(str(row["article_id"]))
        selected.append((demo_type, row))

    medium_length = data["word_count"].between(61, 250)
    choose(
        "likely_real_correct",
        data.loc[
            data["label"].eq(0)
            & data["decision"].eq("likely_real")
            & medium_length
        ],
        ["confidence", "article_id"],
        [False, True],
    )
    choose(
        "likely_fake_correct",
        data.loc[
            data["label"].eq(1)
            & data["decision"].eq("likely_fake")
            & medium_length
        ],
        ["confidence", "article_id"],
        [False, True],
    )
    uncertain = data.loc[data["decision"].eq("uncertain")].copy()
    uncertain["distance_from_half"] = (
        uncertain["probability_fake"] - 0.5
    ).abs()
    choose(
        "uncertain",
        uncertain,
        ["distance_from_half", "article_id"],
        [True, True],
    )
    choose(
        "false_positive",
        data.loc[
            data["error_type"].eq("false_positive")
            & data["confidence"].lt(HIGH_CONFIDENCE)
        ],
        ["confidence", "article_id"],
        [False, True],
    )
    choose(
        "false_negative",
        data.loc[
            data["error_type"].eq("false_negative")
            & data["confidence"].lt(HIGH_CONFIDENCE)
        ],
        ["confidence", "article_id"],
        [False, True],
    )
    choose(
        "high_confidence_error",
        data.loc[
            data["error_type"].ne("correct")
            & data["confidence"].ge(HIGH_CONFIDENCE)
        ],
        ["confidence", "article_id"],
        [False, True],
    )

    rows = []
    for demo_type, row in selected:
        features = extract_linguistic_features(row["title"], row["content"])
        rows.append(
            {
                "demo_type": demo_type,
                "article_id": row["article_id"],
                "title": row["title"],
                "content_excerpt": str(row["content"])[:240].replace("\n", " "),
                "true_label": "fake" if int(row["label"]) == 1 else "real",
                "binary_prediction": (
                    "fake" if int(row["binary_prediction"]) == 1 else "real"
                ),
                "decision": row["decision"],
                "probability_real": float(row["probability_real"]),
                "probability_fake": float(row["probability_fake"]),
                "confidence": float(row["confidence"]),
                "word_count": int(features["word_count"]),
                "sensational_words": features["sensational_found"],
                "source_markers": features["source_indicators_found"],
                "exclamation_count": int(features["exclamation_count"]),
                "uppercase_ratio": float(features["uppercase_char_ratio"]),
                "diacritic_ratio": float(features["diacritic_ratio"]),
                "explanation": demo_explanation(demo_type, row, features),
            }
        )
    return pd.DataFrame(rows)

