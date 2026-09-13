# Krahason Word TF-IDF, Character TF-IDF dhe bashkimin e tyre me të njëjtin classifier
# dhe të njëjtin split. Përzgjedh vetëm nga të dhënat e brendshme përfaqësimin që më pas
# përdoret në krahasimin e classifier-ave dhe në modelin final.

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[3]))

import pandas as pd

from archive.experiments.evaluation.experiment_utils import file_sha256
from archive.experiments.models.experiment_support.day13_analysis import (
    CHARACTER_CONFIGS,
    DEFAULT_FAKE_THRESHOLD,
    DEFAULT_REAL_THRESHOLD,
    INTERNAL_SELECTION_PATH,
    METRICS_PATH,
    MODEL_DISPLAY,
    MODEL_NAMES,
    PROJECT_ROOT,
    REPORTS_DIR,
    build_calibration_folds,
    build_representation_pipeline,
    calculate_metrics,
    internal_selection,
    load_internal_data,
    prediction_table,
    screen_character_configs,
    train_calibrated_representation,
)

LOGGER = logging.getLogger(__name__)


def run_representation_comparison() -> dict:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    train, test, excluded_ids = load_internal_data()
    character_screen, selected_char_config = screen_character_configs(train)
    calibration_folds, calibration_group_count = build_calibration_folds(train)

    comparison_rows: list[dict] = []
    for model_name in MODEL_NAMES:
        LOGGER.info("Training %s", MODEL_DISPLAY[model_name])
        model = train_calibrated_representation(
            train,
            model_name,
            selected_char_config,
            calibration_folds,
        )
        predictions = prediction_table(
            test,
            model,
            model_name,
            id_column="article_id",
        )
        comparison_rows.append(
            {
                "model": model_name,
                "model_display": MODEL_DISPLAY[model_name],
                **calculate_metrics(predictions),
            }
        )

    internal_comparison = pd.DataFrame(comparison_rows)
    selection = internal_selection(internal_comparison, selected_char_config)
    INTERNAL_SELECTION_PATH.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    selection_hash = file_sha256(INTERNAL_SELECTION_PATH)

    metrics = {
        "status": "completed",
        "protocol": {
            "external_used_for_tuning": False,
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
        "character_screen": character_screen.to_dict(orient="records"),
        "internal_selection": selection,
        "internal_metrics": internal_comparison.to_dict(orient="records"),
        "integrity": {
            "internal_selection_sha256": selection_hash,
        },
        "artifacts": {
            "internal_selection": str(
                INTERNAL_SELECTION_PATH.relative_to(PROJECT_ROOT)
            ),
            "metrics": str(METRICS_PATH.relative_to(PROJECT_ROOT)),
        },
    }
    METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics


# Emri publik historik ruhet për thirrjet dhe notebook-et ekzistuese.
run_day13_comparison = run_representation_comparison


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    metrics = run_representation_comparison()
    selection = metrics["internal_selection"]
    LOGGER.info(
        "Recommended representation: %s",
        MODEL_DISPLAY[selection["recommended_for_day14"]],
    )
    LOGGER.info(
        "External data used for selection: %s",
        metrics["protocol"]["external_used_for_tuning"],
    )
    LOGGER.info("Selection saved to: %s", INTERNAL_SELECTION_PATH)


if __name__ == "__main__":
    main()
