import pytest

from src.models.evaluate_app_system import (
    build_system_cases,
    evaluate_system_case,
    evaluate_test_set,
    load_evaluation_data,
    select_demo_examples,
)


@pytest.fixture(scope="module")
def evaluation_inputs():
    return load_evaluation_data()


def test_system_case_suite_covers_required_input_types() -> None:
    case_names = {case["case_name"] for case in build_system_cases()}

    assert {
        "empty_input",
        "title_only",
        "content_only",
        "very_short",
        "normal_text",
        "very_long",
        "uppercase_text",
        "many_exclamations",
        "clickbait_style",
        "official_style",
        "source_markers",
        "without_diacritics",
        "unusual_unicode",
        "uncertain_example",
    }.issubset(case_names)


def test_all_system_cases_pass_without_unexpected_crashes(evaluation_inputs) -> None:
    _, model, _ = evaluation_inputs
    results = [evaluate_system_case(case, model) for case in build_system_cases()]

    assert len(results) == 16
    assert all(result["passed"] for result in results)
    assert sum(result["validation_status"] == "blocked" for result in results) == 3


def test_full_test_set_invariants_and_demo_selection(evaluation_inputs) -> None:
    test_data, model, excluded_ids = evaluation_inputs
    predictions, summary = evaluate_test_set(test_data, model)
    demos = select_demo_examples(predictions, model)

    assert len(excluded_ids) == 7
    assert summary["rows"] == 792
    assert summary["invalid_probability_rows"] == 0
    assert summary["maximum_probability_sum_error"] <= 1e-12
    assert summary["decision_threshold_mismatches"] == 0
    assert set(demos["demo_type"]) == {
        "likely_real",
        "likely_fake",
        "uncertain",
        "false_positive",
        "false_negative",
        "high_confidence_error",
    }
