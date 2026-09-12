"""Verify and freeze the Day 16 candidate without retraining it."""

from __future__ import annotations

import json
import logging
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.experiment_support.day17_analysis import (  # noqa: E402
    BASELINE_MODEL_NAME,
    BASELINE_MODEL_PATH,
    DAY16_EXTERNAL_PREDICTIONS_PATH,
    DAY16_FOLDS_PATH,
    DAY16_INTERNAL_PREDICTIONS_PATH,
    DAY16_METRICS_PATH,
    DAY16_SELECTION_PATH,
    DEFAULT_FAKE_THRESHOLD,
    DEFAULT_REAL_THRESHOLD,
    DEMO_CASES_PATH,
    EXPECTED_EXTERNAL_ROWS,
    EXPECTED_TEST_ROWS,
    EXPECTED_TRAIN_ROWS,
    EXTERNAL_COMPARISON_PATH,
    EXTERNAL_PATH,
    EXTERNAL_PREDICTIONS_PATH,
    FIGURES_DIR,
    FINAL_FAKE_THRESHOLD,
    FINAL_MANIFEST_PATH,
    FINAL_MODEL_ID,
    FINAL_MODEL_NAME,
    FINAL_MODEL_PATH,
    FINAL_MODEL_VERSION,
    FINAL_REAL_THRESHOLD,
    HIGH_CONFIDENCE,
    HIGH_CONFIDENCE_ERRORS_PATH,
    INTERNAL_COMPARISON_PATH,
    INTERNAL_PREDICTIONS_PATH,
    LENGTH_DISPLAY,
    LENGTH_METRICS_PATH,
    LENGTH_LABELS,
    METRICS_PATH,
    PROJECT_ROOT,
    REGRESSION_CASES,
    REGRESSION_PATH,
    REPORTS_DIR,
    REPORT_PATH,
    SOURCE_MODEL_PATH,
    SPECIAL_COHORTS_PATH,
    STREAMLIT_APP_PATH,
    TEST_PATH,
    TRAIN_PATH,
    VERIFICATION_PATH,
    add_word_counts,
    classify_probability,
    demo_explanation,
    evaluate_length_behavior,
    exclude_train_duplicates_from_test,
    extract_linguistic_features,
    file_sha256,
    freeze_artifact,
    high_confidence_error_rows,
    load_evaluation_data,
    load_json,
    maximum_day16_probability_difference,
    model_comparison_table,
    observed_signal_summary,
    predict_final_news,
    prepare_final_model_text,
    refresh_model_text,
    rounded_metrics,
    run_regression_checks,
    select_demo_cases,
    verify_frozen_selection,
    verify_model_configuration,
    verify_preprocessing_contract,
)
from src.models.experiment_support.day17_outputs import (  # noqa: E402
    plot_length_results,
    plot_model_comparison,
    write_report,
)


def run_finalization() -> dict:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    required = [
        TRAIN_PATH,
        TEST_PATH,
        EXTERNAL_PATH,
        SOURCE_MODEL_PATH,
        BASELINE_MODEL_PATH,
        DAY16_SELECTION_PATH,
        DAY16_METRICS_PATH,
        DAY16_FOLDS_PATH,
        DAY16_INTERNAL_PREDICTIONS_PATH,
        DAY16_EXTERNAL_PREDICTIONS_PATH,
        STREAMLIT_APP_PATH,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Day 17 inputs: {missing}")

    protected_paths = {
        "day16_candidate": SOURCE_MODEL_PATH,
        "baseline_model": BASELINE_MODEL_PATH,
        "day16_selection": DAY16_SELECTION_PATH,
        "external_dataset": EXTERNAL_PATH,
        "streamlit_app": STREAMLIT_APP_PATH,
    }
    hashes_before = {name: file_sha256(path) for name, path in protected_paths.items()}
    selection, day16_metrics = verify_frozen_selection()
    source_hash = file_sha256(SOURCE_MODEL_PATH)
    expected_source_hash = day16_metrics["training_metadata"]["model_sha256"]
    if source_hash != expected_source_hash:
        raise RuntimeError("Day 16 model hash differs from its recorded metrics.")

    source_model = joblib.load(SOURCE_MODEL_PATH)
    model_configuration = verify_model_configuration(source_model)
    preprocessing = verify_preprocessing_contract()
    folds = pd.read_csv(DAY16_FOLDS_PATH)
    if len(folds) != 10:
        raise ValueError("Expected five sigmoid and five isotonic outer-fold rows.")
    if folds[["outer_overlapping_groups", "inner_overlapping_groups"]].max().max() != 0:
        raise RuntimeError("Day 16 calibration fold audit contains leakage.")

    final_model, artifact = freeze_artifact(source_hash)
    final_configuration = verify_model_configuration(final_model)
    if final_configuration != model_configuration:
        raise RuntimeError("Final artifact configuration differs after copy.")
    reloaded_model = joblib.load(FINAL_MODEL_PATH)

    train, test, external, data_audit = load_evaluation_data()
    regression, regression_summary = run_regression_checks(
        source_model, final_model, reloaded_model, test
    )
    regression.to_csv(REGRESSION_PATH, index=False, encoding="utf-8")

    models = {
        BASELINE_MODEL_NAME: joblib.load(BASELINE_MODEL_PATH),
        FINAL_MODEL_NAME: final_model,
    }
    internal_comparison, internal_predictions, internal_high = model_comparison_table(
        test,
        models,
        "internal_test",
        "article_id",
        FINAL_REAL_THRESHOLD,
        FINAL_FAKE_THRESHOLD,
    )
    final_internal = internal_predictions.loc[
        internal_predictions["model"].eq(FINAL_MODEL_NAME)
    ].copy()
    length_metrics, special_cohorts, length_bias = evaluate_length_behavior(
        final_internal,
        FINAL_REAL_THRESHOLD,
        FINAL_FAKE_THRESHOLD,
    )

    # The external benchmark is evaluated only after the final artifact is frozen.
    external_comparison, external_predictions, external_high = model_comparison_table(
        external,
        models,
        "external_pilot",
        "external_id",
        FINAL_REAL_THRESHOLD,
        FINAL_FAKE_THRESHOLD,
    )
    internal_day16_difference = maximum_day16_probability_difference(
        internal_predictions,
        DAY16_INTERNAL_PREDICTIONS_PATH,
        "article_id",
    )
    external_day16_difference = maximum_day16_probability_difference(
        external_predictions,
        DAY16_EXTERNAL_PREDICTIONS_PATH,
        "external_id",
    )
    if internal_day16_difference > 1e-15 or external_day16_difference > 1e-15:
        raise RuntimeError("Final predictions differ from frozen Day 16 results.")

    demos = select_demo_cases(final_internal)
    high_confidence_errors = pd.concat(
        [*internal_high, *external_high], ignore_index=True
    )
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
        "confidence",
        "decision",
        "prediction_correct",
        "error_type",
    ]
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
        "confidence",
        "decision",
        "prediction_correct",
        "error_type",
    ]
    internal_comparison.to_csv(INTERNAL_COMPARISON_PATH, index=False, encoding="utf-8")
    internal_predictions[internal_output_columns].to_csv(
        INTERNAL_PREDICTIONS_PATH, index=False, encoding="utf-8"
    )
    external_comparison.to_csv(EXTERNAL_COMPARISON_PATH, index=False, encoding="utf-8")
    external_predictions[external_output_columns].to_csv(
        EXTERNAL_PREDICTIONS_PATH, index=False, encoding="utf-8"
    )
    length_metrics.to_csv(LENGTH_METRICS_PATH, index=False, encoding="utf-8")
    special_cohorts.to_csv(SPECIAL_COHORTS_PATH, index=False, encoding="utf-8")
    demos.to_csv(DEMO_CASES_PATH, index=False, encoding="utf-8")
    high_confidence_errors.to_csv(
        HIGH_CONFIDENCE_ERRORS_PATH, index=False, encoding="utf-8"
    )
    plot_model_comparison(internal_comparison, external_comparison)
    plot_length_results(length_metrics)

    final_internal_metrics = internal_comparison.loc[
        internal_comparison["model"].eq(FINAL_MODEL_NAME)
    ].iloc[0]
    final_external_metrics = external_comparison.loc[
        external_comparison["model"].eq(FINAL_MODEL_NAME)
    ].iloc[0]
    hashes_after = {name: file_sha256(path) for name, path in protected_paths.items()}
    if hashes_before != hashes_after:
        changed = [name for name in hashes_before if hashes_before[name] != hashes_after[name]]
        raise RuntimeError(f"Protected Day 17 inputs changed: {changed}")

    verification = {
        "all_checks_passed": True,
        "model_load_passed": True,
        "configuration": model_configuration,
        "preprocessing": preprocessing,
        "probabilities_valid": regression_summary["probabilities_in_range"],
        "threshold_logic_valid": regression_summary[
            "all_decisions_match_thresholds"
        ],
        "reload_is_deterministic": regression_summary[
            "maximum_reload_difference"
        ]
        == 0.0,
        "evaluation_prediction_preprocessing_identical": regression_summary[
            "evaluation_prediction_preprocessing_mismatches"
        ]
        == 0,
        "day16_internal_prediction_max_difference": internal_day16_difference,
        "day16_external_prediction_max_difference": external_day16_difference,
        "calibration_group_overlap": 0,
        "retraining_performed": False,
        "tuning_performed": False,
        "streamlit_modified": False,
        "protected_hashes_before": hashes_before,
        "protected_hashes_after": hashes_after,
    }
    VERIFICATION_PATH.write_text(
        json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    official_internal = final_internal_metrics[
        [
            "accuracy",
            "f1_weighted",
            "f1_fake",
            "recall_real",
            "recall_fake",
            "confusion_matrix",
            "brier_score",
            "log_loss",
            "ece",
            "high_confidence_errors",
            "threshold_strong_coverage",
            "threshold_strong_accuracy",
        ]
    ].to_dict()
    external_pilot = final_external_metrics[
        [
            "accuracy",
            "recall_real",
            "recall_fake",
            "confusion_matrix",
            "brier_score",
            "log_loss",
            "high_confidence_errors",
            "threshold_likely_real",
            "threshold_uncertain",
            "threshold_likely_fake",
            "threshold_strong_coverage",
            "threshold_strong_accuracy",
        ]
    ].to_dict()
    metrics = {
        "status": "final_frozen",
        "model_id": FINAL_MODEL_ID,
        "model_version": FINAL_MODEL_VERSION,
        "protocol": {
            "source": "frozen_day16_candidate",
            "retraining": False,
            "tuning": False,
            "configuration_changed": False,
            "external_used_for_model_decisions": False,
            "streamlit_integration": "integrated_day18",
        },
        "artifact": artifact,
        "configuration": model_configuration,
        "preprocessing": preprocessing,
        "thresholds": {
            "likely_real_below": FINAL_REAL_THRESHOLD,
            "uncertain_inclusive": [FINAL_REAL_THRESHOLD, FINAL_FAKE_THRESHOLD],
            "likely_fake_above": FINAL_FAKE_THRESHOLD,
        },
        "data": data_audit,
        "regression_checks": regression_summary,
        "official_internal_metrics": rounded_metrics(official_internal),
        "external_pilot_metrics": rounded_metrics(external_pilot),
        "internal_model_comparison": [
            rounded_metrics(row)
            for row in internal_comparison.to_dict(orient="records")
        ],
        "external_model_comparison": [
            rounded_metrics(row)
            for row in external_comparison.to_dict(orient="records")
        ],
        "length_bias": rounded_metrics(length_bias.iloc[0].to_dict()),
        "demo_case_ids": demos[["demo_type", "article_id"]].to_dict(
            orient="records"
        ),
        "verification": verification,
    }
    METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    manifest = {
        "schema_version": 1,
        "status": "final_frozen",
        "model_id": FINAL_MODEL_ID,
        "model_version": FINAL_MODEL_VERSION,
        "artifact": artifact,
        "configuration": model_configuration,
        "preprocessing": preprocessing,
        "thresholds": metrics["thresholds"],
        "training_data": {
            "path": str(TRAIN_PATH.relative_to(PROJECT_ROOT)),
            "rows": data_audit["train_rows"],
            "real": data_audit["train_real"],
            "fake": data_audit["train_fake"],
            "leakage_groups": int(selection["group_count"]),
        },
        "official_internal_metrics": metrics["official_internal_metrics"],
        "external_evaluation_role": "pilot_only_not_used_for_model_decisions",
        "runtime": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
        "prediction_function": "src.models.predict_final.predict_final_news",
        "fact_checking": False,
        "streamlit_integration": "integrated_day18",
        "streamlit_runtime": {
            "app_path": "app/streamlit_app.py",
            "model_loader": "src.models.predict_final.load_final_model",
            "prediction_function": "src.models.predict_final.predict_final_news",
            "model_cache": "streamlit.cache_resource",
            "linguistic_features_role": "explanation_only",
        },
        "limitations": [
            "length_bias",
            "source_label_confounding",
            "external_domain_shift",
            "short_text_instability",
            "linguistic_classification_not_fact_checking",
        ],
    }
    FINAL_MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(
        metrics,
        internal_comparison,
        external_comparison,
        length_metrics,
        special_cohorts,
        demos,
        regression,
        high_confidence_errors,
    )
    return metrics


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    metrics = run_finalization()
    print("Day 17 completed without retraining.")
    print("Status:", metrics["status"])
    print("Final artifact:", metrics["artifact"]["final_path"])
    print("SHA-256:", metrics["artifact"]["final_sha256"])
    print("Report:", REPORT_PATH)


if __name__ == "__main__":
    main()
