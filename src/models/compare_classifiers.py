"""Run the Day 14 classifier comparison in its leakage-safe frozen order."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.data_utils import (  # noqa: E402
    LENGTH_DISPLAY,
    LENGTH_LABELS,
    add_word_counts,
    build_group_safe_folds,
    exclude_train_duplicates_from_test,
    refresh_model_text,
)
from src.evaluation.experiment_utils import (  # noqa: E402
    escaped_dataframe_to_markdown as dataframe_to_markdown,
    file_sha256,
)
from src.evaluation.metrics import (  # noqa: E402
    classification_metrics,
    fake_decision_scores,
    rounded_metrics,
)
from src.models.builders import FIXED_CHAR_CONFIG, build_fixed_features  # noqa: E402
from src.models.experiment_support.day14_analysis import (  # noqa: E402
    CLASSIFIER_CONFIGS,
    CLASSIFIER_DISPLAY,
    COHORT_METRICS_PATH,
    COLORS,
    CURRENT_APP_MODEL_PATH,
    CV_FIGURE_PATH,
    CV_FOLDS_PATH,
    CV_METRICS,
    CV_SUMMARY_PATH,
    DAY13_SELECTION_PATH,
    EXTERNAL_COMPARISON_PATH,
    EXTERNAL_FIGURE_PATH,
    EXTERNAL_PATH,
    EXTERNAL_PREDICTIONS_PATH,
    FIGURES_DIR,
    INTERNAL_COMPARISON_PATH,
    INTERNAL_FIGURE_PATH,
    INTERNAL_PREDICTIONS_PATH,
    LENGTH_BIAS_PATH,
    LENGTH_FIGURE_PATH,
    LENGTH_METRICS_PATH,
    METRICS_PATH,
    MODEL_PATHS,
    MODELS_DIR,
    REPORT_PATH,
    REPORTS_DIR,
    SCORE_DISPLAY,
    SELECTION_PATH,
    SELECTION_TOLERANCE,
    STREAMLIT_APP_PATH,
    TEST_PATH,
    TRAIN_PATH,
    best_cv_rows,
    build_classifier,
    build_model_pipeline,
    candidate_from_selection,
    evaluate_internal_cohorts,
    fit_selected_family_models,
    length_bias_comparison,
    load_external_after_selection,
    load_fixed_representation,
    load_internal_test_after_selection,
    metrics_from_prediction_table,
    prediction_table,
    run_group_safe_cv,
    select_from_cv,
    verify_selection_hash,
)
from src.models.experiment_support.day14_outputs import (  # noqa: E402
    plot_cv,
    plot_external,
    plot_internal,
    plot_length_groups,
    write_report,
)


LOGGER = logging.getLogger(__name__)


def run_day14_comparison() -> dict:
    """Run Day 14 in the strict order train/CV, internal test, external test."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    frozen_paths = {
        "day13_selection": DAY13_SELECTION_PATH,
        "current_app_model": CURRENT_APP_MODEL_PATH,
        "streamlit_app": STREAMLIT_APP_PATH,
        "external_dataset": EXTERNAL_PATH,
    }
    missing = [str(path) for path in [TRAIN_PATH, TEST_PATH, *frozen_paths.values()] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Day 14 input files: {missing}")
    hashes_before = {name: file_sha256(path) for name, path in frozen_paths.items()}

    char_config = load_fixed_representation()
    raw_train = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig", keep_default_na=False)
    train, stale_train_rows = refresh_model_text(raw_train)
    if train["model_text"].str.strip().eq("").any():
        raise ValueError("Train contains empty model_text values.")

    LOGGER.info("Running group-safe CV for %s candidates", len(CLASSIFIER_CONFIGS))
    cv_folds, cv_summary, fold_audit, group_count = run_group_safe_cv(
        train, char_config
    )
    cv_folds.to_csv(CV_FOLDS_PATH, index=False, encoding="utf-8")
    cv_summary.to_csv(CV_SUMMARY_PATH, index=False, encoding="utf-8")

    selection = select_from_cv(cv_summary, char_config)
    selection["cv_fold_audit"] = fold_audit
    selection["group_count"] = group_count
    SELECTION_PATH.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    selection_hash = file_sha256(SELECTION_PATH)

    # Test data is deliberately loaded only after this CV selection is frozen.
    test, test_audit = load_internal_test_after_selection(train, selection_hash)
    models, training_metadata = fit_selected_family_models(
        train, selection, char_config
    )
    internal_tables = [
        prediction_table(test, models[classifier], classifier, "article_id")
        for classifier in CLASSIFIER_DISPLAY
    ]
    internal_predictions = pd.concat(internal_tables, ignore_index=True)
    internal_comparison, length_metrics, special_cohorts = (
        evaluate_internal_cohorts(internal_predictions)
    )
    for classifier, metadata in training_metadata.items():
        mask = internal_comparison["classifier"].eq(classifier)
        internal_comparison.loc[mask, "final_training_seconds"] = metadata[
            "training_seconds"
        ]
        internal_comparison.loc[mask, "model_size_mb"] = metadata["model_size_mb"]
    length_bias = length_bias_comparison(internal_predictions)

    internal_columns = [
        "classifier",
        "article_id",
        "pair_id",
        "label",
        "label_name",
        "title",
        "word_count",
        "length_group",
        "score_type",
        "decision_score_fake",
        "binary_prediction",
        "prediction_correct",
        "error_type",
    ]
    internal_predictions[internal_columns].to_csv(
        INTERNAL_PREDICTIONS_PATH, index=False, encoding="utf-8"
    )
    internal_comparison.to_csv(
        INTERNAL_COMPARISON_PATH, index=False, encoding="utf-8"
    )
    length_metrics.to_csv(LENGTH_METRICS_PATH, index=False, encoding="utf-8")
    special_cohorts.to_csv(COHORT_METRICS_PATH, index=False, encoding="utf-8")
    length_bias.to_csv(LENGTH_BIAS_PATH, index=False, encoding="utf-8")

    plot_cv(cv_summary, selection)
    plot_internal(internal_comparison)
    plot_length_groups(length_metrics)

    # External data remains unopened until after the train-only choice is locked.
    external, stale_external_rows = load_external_after_selection(selection_hash)
    external_tables = [
        prediction_table(external, models[classifier], classifier, "external_id")
        for classifier in CLASSIFIER_DISPLAY
    ]
    external_predictions = pd.concat(external_tables, ignore_index=True)
    external_rows: list[dict] = []
    for classifier, table in external_predictions.groupby("classifier", sort=False):
        model_metrics = metrics_from_prediction_table(table)
        external_rows.append(
            {
                "classifier": classifier,
                "classifier_display": CLASSIFIER_DISPLAY[classifier],
                **{
                    key: value
                    for key, value in model_metrics.items()
                    if key != "confusion_matrix"
                },
                "confusion_matrix": json.dumps(model_metrics["confusion_matrix"]),
            }
        )
    external_comparison = pd.DataFrame(external_rows)
    external_columns = [
        "classifier",
        "external_id",
        "label",
        "title",
        "topic",
        "source",
        "word_count",
        "score_type",
        "decision_score_fake",
        "binary_prediction",
        "prediction_correct",
        "error_type",
    ]
    external_predictions[external_columns].to_csv(
        EXTERNAL_PREDICTIONS_PATH, index=False, encoding="utf-8"
    )
    external_comparison.to_csv(
        EXTERNAL_COMPARISON_PATH, index=False, encoding="utf-8"
    )
    plot_external(external_comparison)

    verify_selection_hash(selection_hash)
    hashes_after = {name: file_sha256(path) for name, path in frozen_paths.items()}
    changed_frozen = [
        name for name in frozen_paths if hashes_before[name] != hashes_after[name]
    ]
    if changed_frozen:
        raise RuntimeError(f"Frozen artifacts changed during Day 14: {changed_frozen}")

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
            "selection_data": "train_only",
            "cv": "5_fold_stratified_group_safe",
            "same_fixed_word_char_tfidf": True,
            "internal_test_used_for_selection": False,
            "external_used_for_selection_or_tuning": False,
            "calibration_applied": False,
            "application_thresholds_applied": False,
            "selection_locked_before_internal_test": True,
            "selection_locked_before_external": True,
        },
        "fixed_representation": selection["fixed_representation"],
        "data_audit": data_audit,
        "candidate_configs": CLASSIFIER_CONFIGS,
        "cv_summary": [
            rounded_metrics(row)
            for row in cv_summary.to_dict(orient="records")
        ],
        "selection": selection,
        "final_training": training_metadata,
        "internal_metrics": [
            rounded_metrics(row)
            for row in internal_comparison.to_dict(orient="records")
        ],
        "length_bias": [
            rounded_metrics(row) for row in length_bias.to_dict(orient="records")
        ],
        "external_metrics": [
            rounded_metrics(row)
            for row in external_comparison.to_dict(orient="records")
        ],
        "integrity": {
            "hashes_before": hashes_before,
            "hashes_after": hashes_after,
            "all_frozen_artifacts_unchanged": hashes_before == hashes_after,
            "selection_sha256": selection_hash,
            "selection_unchanged_after_test_and_external": (
                file_sha256(SELECTION_PATH) == selection_hash
            ),
        },
        "artifacts": {
            "cv_fold_results": str(CV_FOLDS_PATH.relative_to(PROJECT_ROOT)),
            "cv_summary": str(CV_SUMMARY_PATH.relative_to(PROJECT_ROOT)),
            "selection": str(SELECTION_PATH.relative_to(PROJECT_ROOT)),
            "internal_predictions": str(
                INTERNAL_PREDICTIONS_PATH.relative_to(PROJECT_ROOT)
            ),
            "internal_comparison": str(
                INTERNAL_COMPARISON_PATH.relative_to(PROJECT_ROOT)
            ),
            "length_group_metrics": str(LENGTH_METRICS_PATH.relative_to(PROJECT_ROOT)),
            "special_cohort_metrics": str(
                COHORT_METRICS_PATH.relative_to(PROJECT_ROOT)
            ),
            "length_bias": str(LENGTH_BIAS_PATH.relative_to(PROJECT_ROOT)),
            "external_predictions": str(
                EXTERNAL_PREDICTIONS_PATH.relative_to(PROJECT_ROOT)
            ),
            "external_comparison": str(
                EXTERNAL_COMPARISON_PATH.relative_to(PROJECT_ROOT)
            ),
            "report": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
        },
    }
    METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(
        metrics,
        cv_summary,
        internal_comparison,
        length_metrics,
        special_cohorts,
        length_bias,
        external_comparison,
    )
    return metrics


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    metrics = run_day14_comparison()
    selection = metrics["selection"]
    print("Day 14 completed.")
    print("Selected classifier:", CLASSIFIER_DISPLAY[selection["winner_classifier"]])
    print("Selection used external data:", selection["external_results_used"])
    print("Report:", REPORT_PATH)


if __name__ == "__main__":
    main()
