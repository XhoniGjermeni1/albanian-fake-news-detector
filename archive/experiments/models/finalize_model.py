# Finalizon kandidatin e zgjedhur pa ritrajnim: verifikon konfigurimin, calibration-in,
# metrikat dhe prediction anchors, kopjon artefaktin byte-for-byte dhe ndërton manifestin.
# Ky file garanton që modeli i publikuar është pikërisht ai që fitoi eksperimentet.

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

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from archive.experiments.models.experiment_support.day16_analysis import (
    probability_metrics,
    probability_prediction_table,
    threshold_metrics,
)
from archive.experiments.models.experiment_support.day17_analysis import (
    DAY16_EXTERNAL_PREDICTIONS_PATH,
    DAY16_FOLDS_PATH,
    DAY16_INTERNAL_PREDICTIONS_PATH,
    DAY16_METRICS_PATH,
    DAY16_SELECTION_PATH,
    EXTERNAL_PATH,
    FINAL_FAKE_THRESHOLD,
    FINAL_MANIFEST_PATH,
    FINAL_MODEL_ID,
    FINAL_MODEL_NAME,
    FINAL_MODEL_PATH,
    FINAL_MODEL_VERSION,
    FINAL_REAL_THRESHOLD,
    METRICS_PATH,
    PROJECT_ROOT,
    REPORTS_DIR,
    SOURCE_MODEL_PATH,
    TEST_PATH,
    TRAIN_PATH,
    file_sha256,
    freeze_artifact,
    load_evaluation_data,
    maximum_day16_probability_difference,
    run_regression_checks,
    verify_frozen_selection,
    verify_model_configuration,
    verify_preprocessing_contract,
)
from src.evaluation.metrics import rounded_metrics


def _evaluation_snapshot(
    dataframe: pd.DataFrame,
    model,
    id_column: str,
) -> tuple[pd.DataFrame, dict]:
    predictions = probability_prediction_table(
        dataframe,
        model,
        FINAL_MODEL_NAME,
        id_column,
        FINAL_REAL_THRESHOLD,
        FINAL_FAKE_THRESHOLD,
    )
    binary = probability_metrics(
        predictions["label"], predictions["probability_fake"]
    )
    thresholds = threshold_metrics(
        predictions["label"],
        predictions["probability_fake"],
        FINAL_REAL_THRESHOLD,
        FINAL_FAKE_THRESHOLD,
    )
    return predictions, {
        **binary,
        **{f"threshold_{key}": value for key, value in thresholds.items()},
    }


def _official_internal_metrics(metrics: dict) -> dict:
    keys = [
        "accuracy",
        "f1_weighted",
        "f1_fake",
        "recall_real",
        "recall_fake",
        "brier_score",
        "log_loss",
        "ece",
        "high_confidence_errors",
        "threshold_strong_coverage",
        "threshold_strong_accuracy",
    ]
    selected = {key: metrics[key] for key in keys}
    selected["confusion_matrix"] = json.dumps(metrics["confusion_matrix"])
    return rounded_metrics(selected)


def _external_pilot_metrics(metrics: dict) -> dict:
    keys = [
        "accuracy",
        "recall_real",
        "recall_fake",
        "brier_score",
        "log_loss",
        "high_confidence_errors",
        "threshold_likely_real",
        "threshold_uncertain",
        "threshold_likely_fake",
        "threshold_strong_coverage",
        "threshold_strong_accuracy",
    ]
    selected = {key: metrics[key] for key in keys}
    selected["confusion_matrix"] = json.dumps(metrics["confusion_matrix"])
    return rounded_metrics(selected)


def run_finalization() -> dict:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    input_paths = {
        "train": TRAIN_PATH,
        "internal_test": TEST_PATH,
        "external_dataset": EXTERNAL_PATH,
        "calibrated_candidate": SOURCE_MODEL_PATH,
        "calibration_selection": DAY16_SELECTION_PATH,
        "calibration_metrics": DAY16_METRICS_PATH,
        "calibration_folds": DAY16_FOLDS_PATH,
        "internal_prediction_snapshot": DAY16_INTERNAL_PREDICTIONS_PATH,
        "external_prediction_snapshot": DAY16_EXTERNAL_PREDICTIONS_PATH,
    }
    missing = [str(path) for path in input_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing finalization inputs: {missing}")
    hashes_before = {
        name: file_sha256(path) for name, path in input_paths.items()
    }

    selection, calibration_metrics = verify_frozen_selection()
    source_hash = file_sha256(SOURCE_MODEL_PATH)
    expected_hash = calibration_metrics["training_metadata"]["model_sha256"]
    if source_hash != expected_hash:
        raise RuntimeError("Calibrated model hash differs from recorded metrics.")

    source_model = joblib.load(SOURCE_MODEL_PATH)
    model_configuration = verify_model_configuration(source_model)
    preprocessing = verify_preprocessing_contract()

    folds = pd.read_csv(DAY16_FOLDS_PATH)
    if len(folds) != 10:
        raise ValueError("Expected five sigmoid and five isotonic outer-fold rows.")
    overlap_columns = ["outer_overlapping_groups", "inner_overlapping_groups"]
    if folds[overlap_columns].max().max() != 0:
        raise RuntimeError("Calibration fold audit contains group leakage.")

    final_model, artifact = freeze_artifact(source_hash)
    if verify_model_configuration(final_model) != model_configuration:
        raise RuntimeError("Final model configuration changed during the copy.")
    reloaded_model = joblib.load(FINAL_MODEL_PATH)

    _, test, external, data_audit = load_evaluation_data()
    _, regression_summary = run_regression_checks(
        source_model, final_model, reloaded_model, test
    )

    internal_predictions, internal_metrics = _evaluation_snapshot(
        test, final_model, "article_id"
    )
    external_predictions, external_metrics = _evaluation_snapshot(
        external, final_model, "external_id"
    )
    internal_difference = maximum_day16_probability_difference(
        internal_predictions, DAY16_INTERNAL_PREDICTIONS_PATH, "article_id"
    )
    external_difference = maximum_day16_probability_difference(
        external_predictions, DAY16_EXTERNAL_PREDICTIONS_PATH, "external_id"
    )
    if internal_difference > 1e-15 or external_difference > 1e-15:
        raise RuntimeError("Final predictions differ from calibrated snapshots.")

    hashes_after = {
        name: file_sha256(path) for name, path in input_paths.items()
    }
    if hashes_before != hashes_after:
        changed = [
            name for name in input_paths if hashes_before[name] != hashes_after[name]
        ]
        raise RuntimeError(f"Finalization inputs changed: {changed}")

    verification = {
        "all_checks_passed": True,
        "model_load_passed": True,
        "configuration": model_configuration,
        "preprocessing": preprocessing,
        "probabilities_valid": regression_summary["probabilities_in_range"],
        "threshold_logic_valid": regression_summary[
            "all_decisions_match_thresholds"
        ],
        "reload_is_deterministic": (
            regression_summary["maximum_reload_difference"] == 0.0
        ),
        "evaluation_prediction_preprocessing_identical": (
            regression_summary["evaluation_prediction_preprocessing_mismatches"]
            == 0
        ),
        "day16_internal_prediction_max_difference": internal_difference,
        "day16_external_prediction_max_difference": external_difference,
        "calibration_group_overlap": 0,
        "retraining_performed": False,
        "tuning_performed": False,
        "protected_hashes_before": hashes_before,
        "protected_hashes_after": hashes_after,
    }
    official_internal = _official_internal_metrics(internal_metrics)
    external_pilot = _external_pilot_metrics(external_metrics)

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
        "official_internal_metrics": official_internal,
        "external_pilot_metrics": external_pilot,
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
        "official_internal_metrics": official_internal,
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
            "app_path": "app\\streamlit_app.py",
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
    return metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    metrics = run_finalization()
    print("Finalized without retraining:", metrics["artifact"]["final_path"])
    print("SHA-256:", metrics["artifact"]["final_sha256"])


if __name__ == "__main__":
    main()
