# Krahason C={0.25, 0.5, 1.0, 2.0, 4.0} për Linear SVM mbi të njëjtat folds pa leakage.
# Zgjedh C nga F1, stabiliteti ndërmjet folds, ekuilibri i recall-it dhe rreziku i overfitting;
# ky eksperiment fiksoi C=1.0 për kandidatin final.

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from archive.experiments.evaluation.experiment_utils import file_sha256
from src.evaluation.data_utils import (
    build_group_safe_folds,
    refresh_model_text,
)
from src.evaluation.metrics import classification_metrics
from src.models.builders import (
    FINAL_SVM_C,
    FIXED_CHAR_CONFIG,
    build_fixed_features,
    build_svm,
)


TRAIN_PATH = PROJECT_ROOT / "data" / "interim" / "train.csv"
DAY13_SELECTION_PATH = (
    PROJECT_ROOT
    / "reports"
    / "experiment_history"
    / "10_tfidf_representations"
    / "selection.json"
)
DAY14_SELECTION_PATH = (
    PROJECT_ROOT
    / "reports"
    / "experiment_history"
    / "11_classifier_comparison"
    / "selection.json"
)

REPORTS_DIR = PROJECT_ROOT / "reports" / "experiment_history" / "12_svm_tuning"
SELECTION_PATH = REPORTS_DIR / "selection.json"
METRICS_PATH = REPORTS_DIR / "metrics.json"

C_VALUES = [0.25, 0.5, 1.0, 2.0, 4.0]
BASELINE_C = FINAL_SVM_C
F1_CLOSE_TOLERANCE = 0.002
MAX_ACCEPTABLE_RECALL_GAP = 0.10
LOGGER = logging.getLogger(__name__)


def candidate_id(c_value: float) -> str:
    return f"linear_svm_c_{str(float(c_value)).replace('.', '_')}"


def verify_frozen_setup() -> dict:
    day13 = json.loads(DAY13_SELECTION_PATH.read_text(encoding="utf-8"))
    day14 = json.loads(DAY14_SELECTION_PATH.read_text(encoding="utf-8"))

    if day13.get("recommended_for_day14") != "word_char_tfidf":
        raise ValueError("The selected representation is not Word + Character TF-IDF.")
    if day13.get("selected_char_config") != FIXED_CHAR_CONFIG:
        raise ValueError("The selected character configuration has changed.")
    if day14.get("winner_classifier") != "linear_svm":
        raise ValueError("The selected classifier is not Linear SVM.")
    if day14.get("winner_candidate_id") != "linear_svm_c_1_0":
        raise ValueError("The selected classifier candidate is not Linear SVM C=1.0.")
    if day14.get("internal_test_used") is not False:
        raise ValueError("The classifier selection is not marked train/CV-only.")
    if day14.get("external_results_used") is not False:
        raise ValueError("The classifier selection used external results.")

    return {
        "representation": "word_char_tfidf",
        "character_config": FIXED_CHAR_CONFIG.copy(),
        "day14_classifier": "linear_svm",
        "day14_c": BASELINE_C,
    }


def run_svm_cv(
    train: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], int]:
    folds, groups, fold_audit = build_group_safe_folds(train)
    rows: list[dict] = []

    for fold_number, (fit_index, validation_index) in enumerate(folds, start=1):
        fit = train.iloc[fit_index]
        validation = train.iloc[validation_index]

        vectorizer = build_fixed_features(FIXED_CHAR_CONFIG)
        x_fit = vectorizer.fit_transform(fit["model_text"], fit["label"])
        x_validation = vectorizer.transform(validation["model_text"])

        for c_value in C_VALUES:
            classifier = clone(build_svm(c_value))
            classifier.fit(x_fit, fit["label"])
            validation_prediction = classifier.predict(x_validation)
            train_prediction = classifier.predict(x_fit)

            validation_metrics = classification_metrics(
                validation["label"], validation_prediction
            )
            train_metrics = classification_metrics(fit["label"], train_prediction)
            rows.append(
                {
                    "candidate_id": candidate_id(c_value),
                    "c_value": c_value,
                    "fold": fold_number,
                    "accuracy": validation_metrics["accuracy"],
                    "f1_weighted": validation_metrics["f1_weighted"],
                    "f1_fake": validation_metrics["f1_fake"],
                    "recall_real": validation_metrics["recall_real"],
                    "recall_fake": validation_metrics["recall_fake"],
                    "false_positives": validation_metrics["false_positives"],
                    "false_negatives": validation_metrics["false_negatives"],
                    "train_f1_weighted": train_metrics["f1_weighted"],
                    "generalization_gap": (
                        train_metrics["f1_weighted"]
                        - validation_metrics["f1_weighted"]
                    ),
                    "recall_gap": abs(
                        validation_metrics["recall_real"]
                        - validation_metrics["recall_fake"]
                    ),
                }
            )

    fold_results = pd.DataFrame(rows)
    summary_rows: list[dict] = []
    summarized_metrics = [
        "accuracy",
        "f1_weighted",
        "f1_fake",
        "recall_real",
        "recall_fake",
        "recall_gap",
        "train_f1_weighted",
        "generalization_gap",
    ]

    for c_value in C_VALUES:
        subset = fold_results.loc[fold_results["c_value"].eq(c_value)]
        summary: dict[str, object] = {
            "candidate_id": candidate_id(c_value),
            "c_value": c_value,
            "folds": int(len(subset)),
        }
        for metric in summarized_metrics:
            values = subset[metric].astype(float)
            summary[f"mean_{metric}"] = float(values.mean())
            summary[f"std_{metric}"] = float(values.std(ddof=1))
        summary["total_false_positives"] = int(subset["false_positives"].sum())
        summary["total_false_negatives"] = int(subset["false_negatives"].sum())
        summary_rows.append(summary)

    return (
        fold_results,
        pd.DataFrame(summary_rows),
        fold_audit,
        int(len(np.unique(groups))),
    )


def select_c_from_cv(cv_summary: pd.DataFrame) -> dict:
    summary = cv_summary.copy()
    eligible = summary.loc[
        summary["mean_recall_gap"].le(MAX_ACCEPTABLE_RECALL_GAP)
    ].copy()
    balance_filter_applied = True
    if eligible.empty:
        eligible = summary.copy()
        balance_filter_applied = False

    best_f1 = float(eligible["mean_f1_weighted"].max())
    finalists = eligible.loc[
        eligible["mean_f1_weighted"].ge(
            best_f1 - F1_CLOSE_TOLERANCE - 1e-12
        )
    ].copy()
    winner = finalists.sort_values(
        [
            "std_f1_weighted",
            "mean_recall_gap",
            "mean_generalization_gap",
            "c_value",
        ],
        ascending=[True, True, True, True],
    ).iloc[0]
    most_stable = summary.sort_values(
        ["std_f1_weighted", "mean_f1_weighted"], ascending=[True, False]
    ).iloc[0]
    best_balance = summary.sort_values(
        ["mean_recall_gap", "mean_f1_weighted"], ascending=[True, False]
    ).iloc[0]
    selected_c = float(winner["c_value"])

    return {
        "selection_scope": "train_only_5_fold_group_safe_cv",
        "fixed_representation": {
            "name": "word_char_tfidf",
            "word": {
                "ngram_range": [1, 2],
                "lowercase": False,
                "min_df": 2,
                "max_features": 30000,
            },
            "character": FIXED_CHAR_CONFIG.copy(),
        },
        "classifier": "linear_svm",
        "tested_c_values": C_VALUES,
        "baseline_c": BASELINE_C,
        "f1_close_tolerance": F1_CLOSE_TOLERANCE,
        "max_acceptable_recall_gap": MAX_ACCEPTABLE_RECALL_GAP,
        "balance_filter_applied": balance_filter_applied,
        "selection_rule": (
            "Exclude mean recall gaps above 0.10 when possible; keep candidates "
            "within 0.002 F1 of the best, then prefer lower F1 standard "
            "deviation, lower recall gap, lower generalization gap, and lower C."
        ),
        "selected_c": selected_c,
        "selected_candidate_id": candidate_id(selected_c),
        "selected_mean_f1_weighted": float(winner["mean_f1_weighted"]),
        "selected_std_f1_weighted": float(winner["std_f1_weighted"]),
        "selected_mean_recall_gap": float(winner["mean_recall_gap"]),
        "selected_mean_generalization_gap": float(
            winner["mean_generalization_gap"]
        ),
        "most_stable_c": float(most_stable["c_value"]),
        "best_recall_balance_c": float(best_balance["c_value"]),
        "internal_test_used": False,
        "external_results_used": False,
        "calibration_applied": False,
    }


def run_svm_tuning() -> dict:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    required_paths = [TRAIN_PATH, DAY13_SELECTION_PATH, DAY14_SELECTION_PATH]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing SVM tuning inputs: {missing}")

    hashes_before = {
        "train": file_sha256(TRAIN_PATH),
        "representation_selection": file_sha256(DAY13_SELECTION_PATH),
        "classifier_selection": file_sha256(DAY14_SELECTION_PATH),
    }
    frozen_setup = verify_frozen_setup()
    raw_train = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig", keep_default_na=False)
    train, stale_train_rows = refresh_model_text(raw_train)
    if train["model_text"].str.strip().eq("").any():
        raise ValueError("Train contains empty model_text values.")

    LOGGER.info("Running group-safe Linear SVM tuning for C=%s", C_VALUES)
    _, cv_summary, fold_audit, group_count = run_svm_cv(train)
    selection = select_c_from_cv(cv_summary)
    selection["group_count"] = group_count
    selection["cv_fold_audit"] = fold_audit
    SELECTION_PATH.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    hashes_after = {
        "train": file_sha256(TRAIN_PATH),
        "representation_selection": file_sha256(DAY13_SELECTION_PATH),
        "classifier_selection": file_sha256(DAY14_SELECTION_PATH),
    }
    if hashes_before != hashes_after:
        raise RuntimeError("SVM tuning inputs changed during cross-validation.")

    metrics = {
        "status": "completed",
        "protocol": {
            "selection_data": "train_only",
            "cv": "5_fold_stratified_group_safe",
            "fixed_representation": True,
            "classifier": "linear_svm",
            "internal_test_used_for_selection": False,
            "external_used_for_selection_or_tuning": False,
            "calibration_applied": False,
        },
        "frozen_setup": frozen_setup,
        "data_audit": {
            "train_rows": int(len(train)),
            "train_real": int(train["label"].eq(0).sum()),
            "train_fake": int(train["label"].eq(1).sum()),
            "group_count": group_count,
            "stale_train_model_text_rows_refreshed_in_memory": stale_train_rows,
        },
        "selection": selection,
        "cv_summary": cv_summary.to_dict(orient="records"),
        "integrity": {
            "input_hashes_before": hashes_before,
            "input_hashes_after": hashes_after,
            "all_inputs_unchanged": hashes_before == hashes_after,
            "selection_sha256": file_sha256(SELECTION_PATH),
        },
        "artifacts": {
            "selection": str(SELECTION_PATH.relative_to(PROJECT_ROOT)),
            "metrics": str(METRICS_PATH.relative_to(PROJECT_ROOT)),
        },
    }
    METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


# Emri publik historik ruhet për thirrjet dhe notebook-et ekzistuese.
run_day15_tuning = run_svm_tuning


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    metrics = run_svm_tuning()
    selection = metrics["selection"]
    print("Selected C:", selection["selected_c"])
    print("Selection used external data:", selection["external_results_used"])
    print("Selection saved to:", SELECTION_PATH)


if __name__ == "__main__":
    main()
