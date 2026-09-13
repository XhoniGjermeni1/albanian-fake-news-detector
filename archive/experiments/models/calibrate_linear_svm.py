# Ekzekuton fazën e calibration-it për kandidatin Word+Character Linear SVM: krahason
# sigmoid me isotonic përmes probabiliteteve OOF, zgjedh pragjet e vendimit vetëm nga train
# dhe krijon kandidatin e kalibruar bashkë me snapshot-in e nevojshëm për finalizim.

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from archive.experiments.models.experiment_support.day16_analysis import (
    BASELINE_C,
    CALIBRATED_MODEL_PATH,
    CALIBRATION_FOLDS_PATH,
    DAY15_SELECTION_PATH,
    EXTERNAL_PATH,
    EXTERNAL_PREDICTIONS_PATH,
    INTERNAL_PREDICTIONS_PATH,
    LOGGER,
    METRICS_PATH,
    MODELS_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    SELECTION_PATH,
    TEST_PATH,
    TRAIN_PATH,
    build_calibrated_svm,
    classify_probability,
    evaluate_threshold_variants,
    expected_calibration_error,
    file_sha256,
    load_external_after_selection,
    load_internal_test_after_selection,
    nested_oof_calibration,
    probability_metrics,
    probability_prediction_table,
    refresh_model_text,
    select_calibration_method,
    select_thresholds,
    summarize_calibration_methods,
    threshold_metrics,
    train_final_calibrated_model,
    verify_frozen_day15,
    verify_selection_hash,
)
from src.evaluation.metrics import rounded_metrics


def _prediction_snapshot(
    dataframe: pd.DataFrame,
    model,
    id_column: str,
    lower: float,
    upper: float,
) -> tuple[pd.DataFrame, dict]:
    table = probability_prediction_table(
        dataframe,
        model,
        "new_calibrated_svm",
        id_column,
        lower,
        upper,
    )
    metrics = probability_metrics(table["label"], table["probability_fake"])
    thresholds = threshold_metrics(
        table["label"], table["probability_fake"], lower, upper
    )
    return table, {
        "classification": rounded_metrics(metrics),
        "thresholds": rounded_metrics(thresholds),
    }


def run_calibration() -> dict:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    input_paths = {
        "train": TRAIN_PATH,
        "internal_test": TEST_PATH,
        "external_dataset": EXTERNAL_PATH,
        "svm_selection": DAY15_SELECTION_PATH,
    }
    missing = [str(path) for path in input_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing calibration inputs: {missing}")
    hashes_before = {
        name: file_sha256(path) for name, path in input_paths.items()
    }

    frozen_setup = verify_frozen_day15()
    raw_train = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig", keep_default_na=False)
    train, stale_train_rows = refresh_model_text(raw_train)
    if train["model_text"].str.strip().eq("").any():
        raise ValueError("Train contains empty model_text values.")

    LOGGER.info("Comparing sigmoid and isotonic calibration with nested OOF CV")
    oof_predictions, calibration_folds, outer_audit, group_count = (
        nested_oof_calibration(train)
    )
    method_comparison = summarize_calibration_methods(
        oof_predictions, calibration_folds
    )
    calibration_selection = select_calibration_method(method_comparison)
    selected_method = calibration_selection["selected_method"]

    selected_oof = oof_predictions.loc[
        oof_predictions["method"].eq(selected_method)
    ]
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

    # Finalizimi kontrollon të dhjetë rreshtat e folds të calibration-it për group leakage.
    calibration_folds.to_csv(
        CALIBRATION_FOLDS_PATH, index=False, encoding="utf-8"
    )

    # Të dhënat test dhe external hapen vetëm pasi të dyja zgjedhjet janë ngrirë.
    test, test_audit = load_internal_test_after_selection(train, selection_hash)
    calibrated_model, training_metadata = train_final_calibrated_model(
        train, selected_method
    )
    lower = float(threshold_selection["lower_threshold"])
    upper = float(threshold_selection["upper_threshold"])

    internal_predictions, internal_metrics = _prediction_snapshot(
        test, calibrated_model, "article_id", lower, upper
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
    internal_predictions[internal_columns].to_csv(
        INTERNAL_PREDICTIONS_PATH, index=False, encoding="utf-8"
    )

    external, stale_external_rows = load_external_after_selection(selection_hash)
    external_predictions, external_metrics = _prediction_snapshot(
        external, calibrated_model, "external_id", lower, upper
    )
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
    external_predictions[external_columns].to_csv(
        EXTERNAL_PREDICTIONS_PATH, index=False, encoding="utf-8"
    )

    verify_selection_hash(selection_hash)
    hashes_after = {
        name: file_sha256(path) for name, path in input_paths.items()
    }
    if hashes_before != hashes_after:
        changed = [
            name for name in input_paths if hashes_before[name] != hashes_after[name]
        ]
        raise RuntimeError(f"Calibration inputs changed during the experiment: {changed}")

    metrics = {
        "status": "completed",
        "protocol": {
            "selection_data": "nested_oof_train_only",
            "outer_cv": "5_fold_stratified_group_safe",
            "inner_calibration_cv": "5_fold_stratified_group_safe",
            "internal_test_used_for_selection": False,
            "external_used_for_calibration_or_thresholds": False,
            "fixed_word_char_tfidf": True,
            "fixed_linear_svm_c": BASELINE_C,
        },
        "frozen_setup": frozen_setup,
        "data_audit": {
            "train_rows": int(len(train)),
            "train_real": int(train["label"].eq(0).sum()),
            "train_fake": int(train["label"].eq(1).sum()),
            "group_count": group_count,
            "stale_train_model_text_rows_refreshed_in_memory": stale_train_rows,
            **test_audit,
            "external_rows": int(len(external)),
            "stale_external_model_text_rows_refreshed_in_memory": (
                stale_external_rows
            ),
        },
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
        "internal_metrics": internal_metrics,
        "external_metrics": external_metrics,
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
            "calibration_folds": str(
                CALIBRATION_FOLDS_PATH.relative_to(PROJECT_ROOT)
            ),
            "selection": str(SELECTION_PATH.relative_to(PROJECT_ROOT)),
            "internal_predictions": str(
                INTERNAL_PREDICTIONS_PATH.relative_to(PROJECT_ROOT)
            ),
            "external_predictions": str(
                EXTERNAL_PREDICTIONS_PATH.relative_to(PROJECT_ROOT)
            ),
            "metrics": str(METRICS_PATH.relative_to(PROJECT_ROOT)),
        },
    }
    METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


# Emri publik historik ruhet për thirrjet dhe notebook-et ekzistuese.
run_day16_calibration = run_calibration


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    selection = run_calibration()["selection"]
    print("Calibration:", selection["calibration"]["selected_method"])
    print(
        "Thresholds:",
        selection["thresholds"]["lower_threshold"],
        selection["thresholds"]["upper_threshold"],
    )
    print("Selection saved to:", SELECTION_PATH)


if __name__ == "__main__":
    main()
