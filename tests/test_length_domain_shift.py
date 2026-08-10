import pandas as pd
import pytest

from src.features.linguistic_features import get_words
from src.models.analyze_length_domain_shift import (
    LENGTH_LABELS,
    assign_length_groups,
    summarize_group,
    truncate_to_total_words,
    validate_expansions,
)


def test_length_group_boundaries_are_fixed() -> None:
    data = pd.DataFrame({"word_count": [0, 60, 61, 120, 121, 250, 251]})

    grouped = assign_length_groups(data)

    assert grouped["length_group"].astype(str).tolist() == [
        LENGTH_LABELS[0],
        LENGTH_LABELS[0],
        LENGTH_LABELS[1],
        LENGTH_LABELS[1],
        LENGTH_LABELS[2],
        LENGTH_LABELS[2],
        LENGTH_LABELS[3],
    ]


def test_group_summary_counts_false_positives_and_negatives() -> None:
    data = pd.DataFrame(
        {
            "label": [0, 0, 1, 1],
            "binary_prediction": [0, 1, 0, 1],
            "prediction_correct": [True, False, False, True],
            "error_type": [
                "correct",
                "false_positive",
                "false_negative",
                "correct",
            ],
            "probability_fake": [0.1, 0.8, 0.2, 0.9],
            "decision": ["likely_real", "likely_fake", "likely_real", "likely_fake"],
            "word_count": [40, 45, 50, 55],
        }
    )

    summary = summarize_group(data, "sample")

    assert summary["accuracy"] == 0.5
    assert summary["false_positives"] == 1
    assert summary["false_negatives"] == 1
    assert summary["predicted_fake_rate"] == 0.5
    assert summary["likely_real"] == 2
    assert summary["likely_fake"] == 2


def test_text_truncation_respects_total_target() -> None:
    title = "Titull me tri fjalë"
    content = "një dy tre katër pesë gjashtë shtatë tetë nëntë dhjetë"

    truncated = truncate_to_total_words(title, content, target_words=8)

    assert len(get_words(f"{title} {truncated}")) == 8
    assert truncated


def test_expansion_validation_rejects_label_leakage() -> None:
    external = pd.DataFrame(
        {
            "external_id": ["EXT-1"],
            "url": ["https://example.com/source"],
        }
    )
    valid = pd.DataFrame(
        {
            "external_id": ["EXT-1"],
            "expanded_content": ["Përmbajtje neutrale nga burimi."],
            "source_url": ["https://example.com/source"],
            "selection_reason": ["diagnostic"],
            "construction_notes": ["source only"],
        }
    )
    validate_expansions(valid, external)

    invalid = valid.copy()
    invalid.loc[0, "expanded_content"] = "Ky është një lajm i rremë."

    with pytest.raises(ValueError, match="leaks verdict terms"):
        validate_expansions(invalid, external)
