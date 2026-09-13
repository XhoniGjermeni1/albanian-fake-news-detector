"""Compare classifier families with one fixed TF-IDF representation."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from archive.experiments.evaluation.experiment_utils import file_sha256  # noqa: E402
from archive.experiments.models.experiment_support.day14_analysis import (  # noqa: E402
    CLASSIFIER_CONFIGS,
    CLASSIFIER_DISPLAY,
    DAY13_SELECTION_PATH,
    METRICS_PATH,
    REPORTS_DIR,
    SELECTION_PATH,
    TRAIN_PATH,
    build_classifier,
    load_fixed_representation,
    run_group_safe_cv,
    select_from_cv,
)
from src.evaluation.data_utils import (  # noqa: E402
    add_word_counts,
    build_group_safe_folds,
    refresh_model_text,
)
from src.evaluation.metrics import (  # noqa: E402
    classification_metrics,
    fake_decision_scores,
    rounded_metrics,
)
from src.models.builders import FIXED_CHAR_CONFIG, build_fixed_features  # noqa: E402

LOGGER = logging.getLogger(__name__)


def _without_timing(row: dict) -> dict:
    """Keep model-quality results while omitting machine-dependent timings."""
    return {
        key: value
        for key, value in rounded_metrics(row).items()
        if "seconds" not in key
    }


def run_classifier_comparison() -> dict:
    """Select the classifier using train-only group-safe cross-validation."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    required_paths = [TRAIN_PATH, DAY13_SELECTION_PATH]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing classifier comparison inputs: {missing}")

    input_hashes_before = {
        "train": file_sha256(TRAIN_PATH),
        "representation_selection": file_sha256(DAY13_SELECTION_PATH),
    }
    char_config = load_fixed_representation()
    raw_train = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig", keep_default_na=False)
    train, stale_train_rows = refresh_model_text(raw_train)
    if train["model_text"].str.strip().eq("").any():
        raise ValueError("Train contains empty model_text values.")

    LOGGER.info("Running group-safe CV for %s candidates", len(CLASSIFIER_CONFIGS))
    _, cv_summary, fold_audit, group_count = run_group_safe_cv(train, char_config)
    selection = select_from_cv(cv_summary, char_config)
    selection["cv_fold_audit"] = fold_audit
    selection["group_count"] = group_count
    SELECTION_PATH.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    selection_hash = file_sha256(SELECTION_PATH)

    input_hashes_after = {
        "train": file_sha256(TRAIN_PATH),
        "representation_selection": file_sha256(DAY13_SELECTION_PATH),
    }
    if input_hashes_before != input_hashes_after:
        raise RuntimeError("Classifier comparison inputs changed during evaluation.")

    metrics = {
        "status": "completed",
        "protocol": {
            "selection_data": "train_only",
            "cv": "5_fold_stratified_group_safe",
            "same_fixed_word_char_tfidf": True,
            "internal_test_used_for_selection": False,
            "external_used_for_selection_or_tuning": False,
            "calibration_applied": False,
        },
        "fixed_representation": selection["fixed_representation"],
        "data_audit": {
            "train_rows": int(len(train)),
            "train_real": int(train["label"].eq(0).sum()),
            "train_fake": int(train["label"].eq(1).sum()),
            "group_count": group_count,
            "stale_train_model_text_rows_refreshed_in_memory": stale_train_rows,
        },
        "candidate_configs": CLASSIFIER_CONFIGS,
        "cv_summary": [
            _without_timing(row) for row in cv_summary.to_dict(orient="records")
        ],
        "selection": selection,
        "integrity": {
            "input_hashes_before": input_hashes_before,
            "input_hashes_after": input_hashes_after,
            "all_inputs_unchanged": input_hashes_before == input_hashes_after,
            "selection_sha256": selection_hash,
        },
        "artifacts": {
            "selection": str(SELECTION_PATH.relative_to(PROJECT_ROOT)),
            "metrics": str(METRICS_PATH.relative_to(PROJECT_ROOT)),
        },
    }
    METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics


# Historical public name retained for callers and notebooks.
run_day14_comparison = run_classifier_comparison


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    metrics = run_classifier_comparison()
    selection = metrics["selection"]
    print("Selected classifier:", CLASSIFIER_DISPLAY[selection["winner_classifier"]])
    print("Selection used external data:", selection["external_results_used"])
    print("Selection saved to:", SELECTION_PATH)


if __name__ == "__main__":
    main()
