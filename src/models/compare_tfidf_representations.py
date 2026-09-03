"""Compare word, character, and combined TF-IDF representations."""

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

from src.models.experiment_support.day13_analysis import (
    CHARACTER_CONFIGS,
    CHAR_SCREEN_PATH,
    COHORT_FIGURE_PATH,
    COHORT_METRICS_PATH,
    CURRENT_BASELINE_PATH,
    DAY11_METRICS_PATH,
    DAY12_STABILITY_PATH,
    DEFAULT_FAKE_THRESHOLD,
    DEFAULT_REAL_THRESHOLD,
    EXTERNAL_COMPARISON_PATH,
    EXTERNAL_FIGURE_PATH,
    EXTERNAL_PATH,
    EXTERNAL_PREDICTIONS_PATH,
    FIGURES_DIR,
    INTERNAL_COMPARISON_PATH,
    INTERNAL_FIGURE_PATH,
    INTERNAL_PREDICTIONS_PATH,
    INTERNAL_SELECTION_PATH,
    LENGTH_BIAS_FIGURE_PATH,
    LENGTH_BIAS_PATH,
    LENGTH_DISPLAY,
    LENGTH_LABELS,
    LENGTH_METRICS_PATH,
    METRICS_PATH,
    MODEL_COLORS,
    MODEL_DISPLAY,
    MODEL_NAMES,
    MODEL_PATHS,
    PROJECT_ROOT,
    REPORT_PATH,
    REPORTS_DIR,
    STABILITY_DISPLAY,
    STABILITY_FIGURE_PATH,
    STABILITY_PATH,
    STABILITY_SUMMARY_PATH,
    STABILITY_VARIANTS,
    TEST_PATH,
    TRAIN_PATH,
    add_word_counts,
    build_calibration_folds,
    build_char_vectorizer,
    build_classifier,
    build_internal_cohort_metrics,
    build_leakage_safe_groups,
    build_length_bias_comparison,
    build_representation_pipeline,
    build_screen_folds,
    build_word_vectorizer,
    calculate_metrics,
    classify_probability,
    combine_title_content,
    exclude_train_duplicates_from_test,
    extract_linguistic_features,
    file_sha256,
    internal_selection,
    load_external_after_selection,
    load_internal_data,
    prediction_table,
    probability_arrays,
    remove_albanian_diacritics,
    run_stability_experiment,
    screen_character_configs,
    stability_case_ids,
    stability_variants,
    train_calibrated_representation,
    truncate_to_total_words,
)
from src.models.experiment_support.day13_outputs import (
    plot_cohort_accuracy,
    plot_external_comparison,
    plot_internal_comparison,
    plot_length_bias,
    plot_stability,
    write_report,
)


LOGGER = logging.getLogger(__name__)


def run_day13_comparison() -> dict:
    """Run the complete comparison with an internal-first evaluation order."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    baseline_hash_before = file_sha256(CURRENT_BASELINE_PATH)

    train, test, excluded_ids = load_internal_data()
    test = add_word_counts(test)
    char_screen, selected_char_config = screen_character_configs(train)
    char_screen.to_csv(CHAR_SCREEN_PATH, index=False, encoding="utf-8")

    calibration_folds, calibration_group_count = build_calibration_folds(train)
    models: dict[str, object] = {}
    training_metadata: dict[str, dict] = {}
    internal_tables: list[pd.DataFrame] = []

    for model_name in MODEL_NAMES:
        LOGGER.info("Training %s", MODEL_DISPLAY[model_name])
        model, training_seconds = train_calibrated_representation(
            train,
            model_name,
            selected_char_config,
            calibration_folds,
        )
        joblib.dump(model, MODEL_PATHS[model_name], compress=3)
        model_size_bytes = MODEL_PATHS[model_name].stat().st_size
        models[model_name] = model
        training_metadata[model_name] = {
            "training_seconds": round(training_seconds, 3),
            "model_size_bytes": model_size_bytes,
            "model_size_mb": round(model_size_bytes / (1024 * 1024), 3),
            "model_path": str(MODEL_PATHS[model_name].relative_to(PROJECT_ROOT)),
            "model_sha256": file_sha256(MODEL_PATHS[model_name]),
        }
        internal_tables.append(
            prediction_table(test, model, model_name, id_column="article_id")
        )

    internal_predictions = pd.concat(internal_tables, ignore_index=True)
    cohort_metrics, length_metrics = build_internal_cohort_metrics(
        internal_predictions
    )
    internal_comparison = cohort_metrics.loc[
        cohort_metrics["cohort"].eq("all_internal_test")
    ].copy()
    for model_name, metadata in training_metadata.items():
        mask = internal_comparison["model"].eq(model_name)
        internal_comparison.loc[mask, "training_seconds"] = metadata[
            "training_seconds"
        ]
        internal_comparison.loc[mask, "model_size_mb"] = metadata["model_size_mb"]

    length_bias = build_length_bias_comparison(internal_predictions)
    stability, stability_summary = run_stability_experiment(test, models)
    selection = internal_selection(
        internal_comparison,
        cohort_metrics,
        length_bias,
        stability_summary,
        selected_char_config,
    )

    current_baseline = joblib.load(CURRENT_BASELINE_PATH)
    current_model_texts = [
        combine_title_content(row.title, row.content)
        for row in test.itertuples(index=False)
    ]
    current_probability_fake = probability_arrays(
        current_baseline, current_model_texts
    )[1]
    day13_word_probability_fake = internal_predictions.loc[
        internal_predictions["model"].eq("word_tfidf"), "probability_fake"
    ].to_numpy()
    selection["word_vs_current_baseline_max_probability_difference"] = float(
        np.max(np.abs(current_probability_fake - day13_word_probability_fake))
    )
    INTERNAL_SELECTION_PATH.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    selection_hash = file_sha256(INTERNAL_SELECTION_PATH)

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
        "decision",
        "prediction_correct",
        "error_type",
    ]
    internal_predictions[internal_output_columns].to_csv(
        INTERNAL_PREDICTIONS_PATH, index=False, encoding="utf-8"
    )
    internal_comparison.to_csv(
        INTERNAL_COMPARISON_PATH, index=False, encoding="utf-8"
    )
    cohort_metrics.to_csv(COHORT_METRICS_PATH, index=False, encoding="utf-8")
    length_metrics.to_csv(LENGTH_METRICS_PATH, index=False, encoding="utf-8")
    length_bias.to_csv(LENGTH_BIAS_PATH, index=False, encoding="utf-8")
    stability.to_csv(STABILITY_PATH, index=False, encoding="utf-8")
    stability_summary.to_csv(
        STABILITY_SUMMARY_PATH, index=False, encoding="utf-8"
    )

    plot_internal_comparison(internal_comparison)
    plot_cohort_accuracy(cohort_metrics)
    plot_length_bias(length_metrics)
    plot_stability(stability_summary)

    # External data is deliberately loaded only after the internal selection is locked.
    external_hash_before = file_sha256(EXTERNAL_PATH)
    external = load_external_after_selection(selection_hash)
    external_tables = [
        prediction_table(external, models[model_name], model_name, "external_id")
        for model_name in MODEL_NAMES
    ]
    external_predictions = pd.concat(external_tables, ignore_index=True)
    external_rows: list[dict] = []
    for model_name, table in external_predictions.groupby("model", sort=False):
        model_metrics = calculate_metrics(table)
        external_rows.append(
            {
                "model": model_name,
                "model_display": MODEL_DISPLAY[model_name],
                **model_metrics,
                "confusion_matrix": json.dumps(model_metrics["confusion_matrix"]),
            }
        )
    external_comparison = pd.DataFrame(external_rows)
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
        "decision",
        "prediction_correct",
        "error_type",
    ]
    external_predictions[external_output_columns].to_csv(
        EXTERNAL_PREDICTIONS_PATH, index=False, encoding="utf-8"
    )
    external_comparison.to_csv(
        EXTERNAL_COMPARISON_PATH, index=False, encoding="utf-8"
    )
    plot_external_comparison(external_comparison)

    if file_sha256(INTERNAL_SELECTION_PATH) != selection_hash:
        raise RuntimeError("Internal selection changed during external diagnostics.")
    baseline_hash_after = file_sha256(CURRENT_BASELINE_PATH)
    external_hash_after = file_sha256(EXTERNAL_PATH)
    if baseline_hash_before != baseline_hash_after:
        raise RuntimeError("Current application baseline changed during Day 13.")
    if external_hash_before != external_hash_after:
        raise RuntimeError("External benchmark changed during Day 13.")

    metrics = {
        "status": "completed",
        "protocol": {
            "external_used_for_tuning": False,
            "selection_locked_before_external": True,
            "same_train_test_split": True,
            "exact_train_duplicates_excluded_from_test": len(excluded_ids),
            "calibration_method": "sigmoid",
            "calibration_folds": len(calibration_folds),
            "calibration_group_count": calibration_group_count,
            "thresholds": {
                "likely_real_below": DEFAULT_REAL_THRESHOLD,
                "likely_fake_above": DEFAULT_FAKE_THRESHOLD,
            },
        },
        "selected_char_config": selected_char_config,
        "character_screen": char_screen.to_dict(orient="records"),
        "training_metadata": training_metadata,
        "internal_selection": selection,
        "internal_metrics": internal_comparison.to_dict(orient="records"),
        "length_bias": length_bias.to_dict(orient="records"),
        "external_metrics": external_comparison.to_dict(orient="records"),
        "integrity": {
            "current_baseline_sha256_before": baseline_hash_before,
            "current_baseline_sha256_after": baseline_hash_after,
            "current_baseline_unchanged": baseline_hash_before == baseline_hash_after,
            "external_sha256_before": external_hash_before,
            "external_sha256_after": external_hash_after,
            "external_unchanged": external_hash_before == external_hash_after,
            "internal_selection_sha256": selection_hash,
        },
        "artifacts": {
            "report": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
            "internal_selection": str(
                INTERNAL_SELECTION_PATH.relative_to(PROJECT_ROOT)
            ),
            "internal_comparison": str(
                INTERNAL_COMPARISON_PATH.relative_to(PROJECT_ROOT)
            ),
            "cohort_metrics": str(COHORT_METRICS_PATH.relative_to(PROJECT_ROOT)),
            "stability": str(STABILITY_PATH.relative_to(PROJECT_ROOT)),
            "external_comparison": str(
                EXTERNAL_COMPARISON_PATH.relative_to(PROJECT_ROOT)
            ),
        },
    }
    METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(
        metrics,
        char_screen,
        internal_comparison,
        cohort_metrics,
        length_bias,
        stability_summary,
        external_comparison,
    )
    return metrics


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    metrics = run_day13_comparison()
    selection = metrics["internal_selection"]
    LOGGER.info(
        "Recommended representation for Day 14: %s",
        MODEL_DISPLAY[selection["recommended_for_day14"]],
    )
    LOGGER.info(
        "External data used for tuning: %s",
        metrics["protocol"]["external_used_for_tuning"],
    )
    LOGGER.info("Report saved to: %s", REPORT_PATH)


if __name__ == "__main__":
    main()
