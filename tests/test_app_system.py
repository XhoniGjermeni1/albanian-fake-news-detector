import pytest

from archive.experiments.models.evaluate_app_system import (
    evaluate_test_set,
    load_evaluation_data,
)


@pytest.fixture(scope="module")
def evaluation_inputs():
    return load_evaluation_data()


def test_evaluation_data_excludes_train_duplicates(evaluation_inputs) -> None:
    test_data, _, excluded_ids = evaluation_inputs

    assert len(test_data) == 792
    assert len(excluded_ids) == 7


def test_full_test_set_invariants(evaluation_inputs) -> None:
    test_data, model, excluded_ids = evaluation_inputs
    predictions, summary = evaluate_test_set(test_data, model)

    assert len(excluded_ids) == 7
    assert len(predictions) == 792
    assert summary["rows"] == 792
    assert summary["invalid_probability_rows"] == 0
    assert summary["maximum_probability_sum_error"] <= 1e-12
    assert summary["decision_threshold_mismatches"] == 0
