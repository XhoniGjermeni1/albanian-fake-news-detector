import pandas as pd
from sklearn.svm import LinearSVC

from src.models.tune_linear_svm import (
    BASELINE_C,
    C_VALUES,
    build_svm,
    candidate_id,
    choose_analysis_candidates,
    select_c_from_cv,
    verify_frozen_setup,
)


def cv_row(
    c_value: float,
    f1: float,
    std: float,
    recall_gap: float,
    generalization_gap: float,
) -> dict:
    return {
        "candidate_id": candidate_id(c_value),
        "c_value": c_value,
        "mean_f1_weighted": f1,
        "std_f1_weighted": std,
        "mean_recall_gap": recall_gap,
        "mean_generalization_gap": generalization_gap,
    }


def test_day15_uses_only_requested_c_values() -> None:
    assert C_VALUES == [0.25, 0.5, 1.0, 2.0, 4.0]
    assert BASELINE_C == 1.0
    assert [candidate_id(value) for value in C_VALUES] == [
        "linear_svm_c_0_25",
        "linear_svm_c_0_5",
        "linear_svm_c_1_0",
        "linear_svm_c_2_0",
        "linear_svm_c_4_0",
    ]


def test_svm_factory_preserves_day14_classifier_settings() -> None:
    model = build_svm(2.0)

    assert isinstance(model, LinearSVC)
    assert model.C == 2.0
    assert model.class_weight == "balanced"
    assert model.max_iter == 5000
    assert model.random_state == 42


def test_frozen_setup_matches_days_13_and_14() -> None:
    setup = verify_frozen_setup()

    assert setup["representation"] == "word_char_tfidf"
    assert setup["day14_classifier"] == "linear_svm"
    assert setup["day14_c"] == 1.0
    assert setup["character_config"]["config_name"] == "char_wb_3_5"


def test_selection_prefers_stability_when_f1_is_very_close() -> None:
    summary = pd.DataFrame(
        [
            cv_row(0.25, 0.8900, 0.0040, 0.050, 0.030),
            cv_row(0.5, 0.8990, 0.0030, 0.050, 0.035),
            cv_row(1.0, 0.9100, 0.0020, 0.050, 0.040),
            cv_row(2.0, 0.9115, 0.0060, 0.070, 0.050),
            cv_row(4.0, 0.9050, 0.0070, 0.080, 0.070),
        ]
    )

    selection = select_c_from_cv(summary)

    assert selection["selected_c"] == 1.0
    assert selection["internal_test_used"] is False
    assert selection["external_results_used"] is False
    assert selection["calibration_applied"] is False


def test_selection_rejects_large_recall_gap_when_alternative_exists() -> None:
    summary = pd.DataFrame(
        [
            cv_row(0.25, 0.880, 0.004, 0.060, 0.030),
            cv_row(0.5, 0.890, 0.003, 0.050, 0.035),
            cv_row(1.0, 0.900, 0.002, 0.040, 0.040),
            cv_row(2.0, 0.920, 0.002, 0.150, 0.050),
            cv_row(4.0, 0.910, 0.005, 0.140, 0.070),
        ]
    )

    selection = select_c_from_cv(summary)

    assert selection["balance_filter_applied"] is True
    assert selection["selected_c"] == 1.0


def test_analysis_candidates_include_selected_and_c1() -> None:
    summary = pd.DataFrame(
        [
            cv_row(0.25, 0.880, 0.004, 0.060, 0.030),
            cv_row(0.5, 0.890, 0.003, 0.050, 0.035),
            cv_row(1.0, 0.900, 0.002, 0.040, 0.040),
            cv_row(2.0, 0.910, 0.003, 0.050, 0.050),
            cv_row(4.0, 0.905, 0.005, 0.080, 0.070),
        ]
    )

    candidates = choose_analysis_candidates(summary, selected_c=2.0)

    assert len(candidates) == 3
    assert candidates[0] == 2.0
    assert 1.0 in candidates
