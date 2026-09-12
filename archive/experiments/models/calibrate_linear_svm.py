"""Run the Day 16 calibration and threshold-selection experiment."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.experiment_support.day16_analysis import (  # noqa: E402
    BASELINE_C,
    CALIBRATED_MODEL_PATH,
    CALIBRATION_FOLDS_PATH,
    CALIBRATION_METHODS,
    CURRENT_APP_MODEL_PATH,
    DAY15_LENGTH_BIAS_PATH,
    DAY15_SELECTION_PATH,
    EXTERNAL_MODEL_COMPARISON_PATH,
    EXTERNAL_PATH,
    EXTERNAL_PREDICTIONS_PATH,
    EXTERNAL_THRESHOLD_PATH,
    FIGURES_DIR,
    HIGH_CONFIDENCE,
    HIGH_CONFIDENCE_ERRORS_PATH,
    INTERNAL_BINS_PATH,
    INTERNAL_MODEL_COMPARISON_PATH,
    INTERNAL_PREDICTIONS_PATH,
    INTERNAL_THRESHOLD_PATH,
    LENGTH_BIAS_PATH,
    LENGTH_DISPLAY,
    LENGTH_METRICS_PATH,
    LENGTH_LABELS,
    LOGGER,
    METHOD_COMPARISON_PATH,
    METRICS_PATH,
    MODELS_DIR,
    N_CALIBRATION_BINS,
    OOF_BINS_PATH,
    OOF_PREDICTIONS_PATH,
    PROBABILITY_DISTRIBUTION_PATH,
    PROJECT_ROOT,
    REPORTS_DIR,
    REPORT_PATH,
    SELECTION_PATH,
    SPECIAL_COHORTS_PATH,
    STREAMLIT_APP_PATH,
    TEST_PATH,
    THRESHOLD_COMPARISON_PATH,
    THRESHOLD_VARIANTS,
    TRAIN_PATH,
    add_word_counts,
    build_calibrated_svm,
    build_calibration_bin_output,
    build_group_safe_folds,
    build_svm_pipeline,
    calibration_bins,
    classification_metrics,
    classify_probability,
    evaluate_length_behavior,
    evaluate_threshold_variants,
    exclude_train_duplicates_from_test,
    expected_calibration_error,
    fake_probabilities,
    file_sha256,
    high_confidence_error_rows,
    load_external_after_selection,
    load_internal_test_after_selection,
    model_comparison_table,
    nested_oof_calibration,
    probability_distribution,
    probability_metrics,
    probability_prediction_table,
    refresh_model_text,
    rounded_metrics,
    select_calibration_method,
    select_thresholds,
    summarize_calibration_methods,
    threshold_metrics,
    train_final_calibrated_model,
    verify_frozen_day15,
    verify_selection_hash,
)
from src.models.experiment_support.day16_outputs import (  # noqa: E402
    plot_internal_calibration,
    plot_length_probability,
    plot_model_comparison,
    plot_oof_calibration,
    plot_thresholds,
    write_report,
)


def run_day16_calibration() -> dict:
    """Run nested calibration selection, threshold selection, and frozen tests."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    frozen_paths = {
        "day15_selection": DAY15_SELECTION_PATH,
        "day15_length_bias": DAY15_LENGTH_BIAS_PATH,
        "current_app_model": CURRENT_APP_MODEL_PATH,
        "streamlit_app": STREAMLIT_APP_PATH,
        "external_dataset": EXTERNAL_PATH,
    }
    required = [TRAIN_PATH, TEST_PATH, *frozen_paths.values()]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Day 16 inputs: {missing}")
    hashes_before = {name: file_sha256(path) for name, path in frozen_paths.items()}

    frozen_setup = verify_frozen_day15()
    raw_train = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig", keep_default_na=False)
    train, stale_train_rows = refresh_model_text(raw_train)
    if train["model_text"].str.strip().eq("").any():
        raise ValueError("Train contains empty model_text values.")

    LOGGER.info("Starting nested OOF calibration comparison")
    oof_predictions, calibration_folds, outer_audit, group_count = (
        nested_oof_calibration(train)
    )
    method_comparison = summarize_calibration_methods(
        oof_predictions, calibration_folds
    )
    oof_bins = build_calibration_bin_output(oof_predictions)
    probability_summary = probability_distribution(oof_predictions)
    calibration_selection = select_calibration_method(method_comparison)
    selected_method = calibration_selection["selected_method"]
    selected_oof = oof_predictions.loc[
        oof_predictions["method"].eq(selected_method)
    ].copy()
    threshold_comparison = evaluate_threshold_variants(
        selected_oof["label"], selected_oof["probability_fake"]
    )
    threshold_selection = select_thresholds(threshold_comparison)

    selection = {
        "selection_scope": "nested_group_safe_oof_train_only",
        "fixed_configuration": {
            "representation": "word_char_tfidf",
            "classifier": "linear_svm",
            "c_value": BASELINE_C,
        },
        "calibration": calibration_selection,
        "thresholds": threshold_selection,
        "internal_test_used": False,
        "current_app_model_used": False,
        "external_results_used": False,
        "outer_fold_audit": outer_audit,
        "group_count": group_count,
    }
    SELECTION_PATH.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    selection_hash = file_sha256(SELECTION_PATH)

    oof_predictions.to_csv(OOF_PREDICTIONS_PATH, index=False, encoding="utf-8")
    calibration_folds.to_csv(CALIBRATION_FOLDS_PATH, index=False, encoding="utf-8")
    method_comparison.to_csv(METHOD_COMPARISON_PATH, index=False, encoding="utf-8")
    oof_bins.to_csv(OOF_BINS_PATH, index=False, encoding="utf-8")
    probability_summary.to_csv(
        PROBABILITY_DISTRIBUTION_PATH, index=False, encoding="utf-8"
    )
    threshold_comparison.to_csv(
        THRESHOLD_COMPARISON_PATH, index=False, encoding="utf-8"
    )
    plot_oof_calibration(oof_bins, oof_predictions)
    plot_thresholds(threshold_comparison, threshold_selection["threshold_name"])

    # Test and current app model remain unopened until both choices are frozen.
    test, test_audit = load_internal_test_after_selection(train, selection_hash)
    calibrated_model, training_metadata = train_final_calibrated_model(
        train, selected_method
    )
    current_app_model = joblib.load(CURRENT_APP_MODEL_PATH)
    lower = threshold_selection["lower_threshold"]
    upper = threshold_selection["upper_threshold"]
    models = {
        "new_calibrated_svm": calibrated_model,
        "current_app_model": current_app_model,
    }
    (
        internal_comparison,
        internal_all_predictions,
        internal_high_confidence,
    ) = model_comparison_table(
        test, models, "internal_test", "article_id", lower, upper
    )
    new_internal = internal_all_predictions.loc[
        internal_all_predictions["model"].eq("new_calibrated_svm")
    ].copy()
    internal_threshold_rows = []
    internal_bin_tables = []
    for model_name, table in internal_all_predictions.groupby("model", sort=False):
        threshold_result = threshold_metrics(
            table["label"], table["probability_fake"], lower, upper
        )
        internal_threshold_rows.append({"model": model_name, **threshold_result})
        bins = calibration_bins(table["label"], table["probability_fake"])
        bins.insert(0, "model", model_name)
        internal_bin_tables.append(bins)
    internal_thresholds = pd.DataFrame(internal_threshold_rows)
    internal_bins = pd.concat(internal_bin_tables, ignore_index=True)
    length_metrics, special_cohorts, length_bias = evaluate_length_behavior(
        new_internal, lower, upper
    )

    internal_columns = [
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
    internal_all_predictions[internal_columns].to_csv(
        INTERNAL_PREDICTIONS_PATH, index=False, encoding="utf-8"
    )
    internal_comparison.to_csv(
        INTERNAL_MODEL_COMPARISON_PATH, index=False, encoding="utf-8"
    )
    internal_bins.to_csv(INTERNAL_BINS_PATH, index=False, encoding="utf-8")
    internal_thresholds.to_csv(INTERNAL_THRESHOLD_PATH, index=False, encoding="utf-8")
    length_metrics.to_csv(LENGTH_METRICS_PATH, index=False, encoding="utf-8")
    special_cohorts.to_csv(SPECIAL_COHORTS_PATH, index=False, encoding="utf-8")
    length_bias.to_csv(LENGTH_BIAS_PATH, index=False, encoding="utf-8")
    plot_internal_calibration(internal_bins)
    plot_length_probability(length_metrics)

    # External data is loaded only after calibration and thresholds are immutable.
    external, stale_external_rows = load_external_after_selection(selection_hash)
    (
        external_comparison,
        external_all_predictions,
        external_high_confidence,
    ) = model_comparison_table(
        external, models, "external", "external_id", lower, upper
    )
    new_external = external_all_predictions.loc[
        external_all_predictions["model"].eq("new_calibrated_svm")
    ].copy()
    external_threshold_rows = []
    for model_name, table in external_all_predictions.groupby("model", sort=False):
        external_threshold_rows.append(
            {
                "model": model_name,
                **threshold_metrics(
                    table["label"], table["probability_fake"], lower, upper
                ),
            }
        )
    external_thresholds = pd.DataFrame(external_threshold_rows)
    external_columns = [
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
    external_all_predictions[external_columns].to_csv(
        EXTERNAL_PREDICTIONS_PATH, index=False, encoding="utf-8"
    )
    external_comparison.to_csv(
        EXTERNAL_MODEL_COMPARISON_PATH, index=False, encoding="utf-8"
    )
    external_thresholds.to_csv(
        EXTERNAL_THRESHOLD_PATH, index=False, encoding="utf-8"
    )

    oof_high_confidence = high_confidence_error_rows(
        selected_oof,
        "oof_train",
        f"new_svm_{selected_method}",
        "article_id",
    )
    all_high_confidence = [
        oof_high_confidence,
        *internal_high_confidence,
        *external_high_confidence,
    ]
    non_empty_high_confidence = [
        table for table in all_high_confidence if not table.empty
    ]
    if non_empty_high_confidence:
        pd.concat(non_empty_high_confidence, ignore_index=True).to_csv(
            HIGH_CONFIDENCE_ERRORS_PATH, index=False, encoding="utf-8"
        )
    else:
        pd.DataFrame(
            columns=[
                "dataset",
                "model",
                "case_id",
                "label",
                "title",
                "probability_fake",
                "confidence",
                "binary_prediction",
            ]
        ).to_csv(HIGH_CONFIDENCE_ERRORS_PATH, index=False, encoding="utf-8")

    plot_model_comparison(internal_comparison, external_comparison)

    verify_selection_hash(selection_hash)
    hashes_after = {name: file_sha256(path) for name, path in frozen_paths.items()}
    if hashes_before != hashes_after:
        changed = [
            name for name in frozen_paths if hashes_before[name] != hashes_after[name]
        ]
        raise RuntimeError(f"Frozen artifacts changed during Day 16: {changed}")

    day15_bias_table = pd.read_csv(DAY15_LENGTH_BIAS_PATH)
    day15_raw_bias = float(
        day15_bias_table.loc[
            day15_bias_table["c_value"].eq(BASELINE_C),
            "mean_absolute_within_label_spearman",
        ].iloc[0]
    )
    data_audit = {
        "train_rows": int(len(train)),
        "train_real": int(train["label"].eq(0).sum()),
        "train_fake": int(train["label"].eq(1).sum()),
        "group_count": group_count,
        "stale_train_model_text_rows_refreshed_in_memory": stale_train_rows,
        **test_audit,
        "external_rows": int(len(external)),
        "stale_external_model_text_rows_refreshed_in_memory": stale_external_rows,
    }
    metrics = {
        "status": "completed",
        "protocol": {
            "selection_data": "nested_oof_train_only",
            "outer_cv": "5_fold_stratified_group_safe",
            "inner_calibration_cv": "5_fold_stratified_group_safe",
            "internal_test_used_for_selection": False,
            "current_app_model_used_for_selection": False,
            "external_used_for_calibration_or_thresholds": False,
            "selection_locked_before_internal_test": True,
            "selection_locked_before_external": True,
            "fixed_word_char_tfidf": True,
            "fixed_linear_svm_c": BASELINE_C,
        },
        "frozen_setup": frozen_setup,
        "data_audit": data_audit,
        "selection": selection,
        "calibration_method_metrics": [
            rounded_metrics(row)
            for row in method_comparison.to_dict(orient="records")
        ],
        "threshold_metrics_oof": [
            rounded_metrics(row)
            for row in threshold_comparison.to_dict(orient="records")
        ],
        "training_metadata": training_metadata,
        "internal_model_comparison": [
            rounded_metrics(row)
            for row in internal_comparison.to_dict(orient="records")
        ],
        "internal_threshold_metrics": [
            rounded_metrics(row)
            for row in internal_thresholds.to_dict(orient="records")
        ],
        "length_analysis": {
            "day15_raw_score_bias": day15_raw_bias,
            "calibrated_probability_bias": rounded_metrics(
                length_bias.iloc[0].to_dict()
            ),
        },
        "external_model_comparison": [
            rounded_metrics(row)
            for row in external_comparison.to_dict(orient="records")
        ],
        "external_threshold_metrics": [
            rounded_metrics(row)
            for row in external_thresholds.to_dict(orient="records")
        ],
        "integrity": {
            "hashes_before": hashes_before,
            "hashes_after": hashes_after,
            "all_frozen_artifacts_unchanged": hashes_before == hashes_after,
            "selection_sha256": selection_hash,
            "selection_unchanged_after_internal_and_external": (
                file_sha256(SELECTION_PATH) == selection_hash
            ),
        },
        "artifacts": {
            "model": str(CALIBRATED_MODEL_PATH.relative_to(PROJECT_ROOT)),
            "oof_predictions": str(OOF_PREDICTIONS_PATH.relative_to(PROJECT_ROOT)),
            "method_comparison": str(METHOD_COMPARISON_PATH.relative_to(PROJECT_ROOT)),
            "threshold_comparison": str(
                THRESHOLD_COMPARISON_PATH.relative_to(PROJECT_ROOT)
            ),
            "selection": str(SELECTION_PATH.relative_to(PROJECT_ROOT)),
            "internal_predictions": str(
                INTERNAL_PREDICTIONS_PATH.relative_to(PROJECT_ROOT)
            ),
            "external_predictions": str(
                EXTERNAL_PREDICTIONS_PATH.relative_to(PROJECT_ROOT)
            ),
            "high_confidence_errors": str(
                HIGH_CONFIDENCE_ERRORS_PATH.relative_to(PROJECT_ROOT)
            ),
            "report": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
        },
    }
    METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(
        metrics,
        method_comparison,
        probability_summary,
        threshold_comparison,
        internal_comparison,
        internal_thresholds,
        length_metrics,
        special_cohorts,
        length_bias,
        external_comparison,
        external_thresholds,
    )
    return metrics


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    metrics = run_day16_calibration()
    selection = metrics["selection"]
    print("Day 16 completed.")
    print("Calibration:", selection["calibration"]["selected_method"])
    print(
        "Thresholds:",
        selection["thresholds"]["lower_threshold"],
        selection["thresholds"]["upper_threshold"],
    )
    print("External used for selection:", selection["external_results_used"])
    print("Report:", REPORT_PATH)


if __name__ == "__main__":
    main()
