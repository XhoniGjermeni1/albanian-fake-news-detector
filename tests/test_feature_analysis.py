import pandas as pd

from archive.experiments.features.analyze_linguistic_features import compare_features


def _sample_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "article_id": ["real_1", "fake_1"],
            "pair_id": [1, 1],
            "label": [0, 1],
            "label_name": ["real", "fake"],
            "word_count": [100, 50],
            "sentence_count": [5, 2],
            "avg_sentence_length": [20.0, 25.0],
            "exclamation_count": [0, 2],
            "question_count": [0, 1],
            "uppercase_word_ratio": [0.01, 0.03],
            "sensational_count": [0, 1],
            "sensational_ratio": [0.0, 0.02],
            "source_indicator_count": [2, 0],
            "source_indicator_ratio": [0.02, 0.0],
            "uncertainty_count": [1, 0],
            "uncertainty_ratio": [0.01, 0.0],
            "diacritic_ratio": [0.07, 0.05],
            "title_length": [60, 80],
            "content_length": [900, 400],
            "sensational_found": ["", "skandal"],
            "source_indicators_found": ["sipas", ""],
            "uncertainty_found": ["thuhet", ""],
            "possible_missing_diacritic_words": ["", ""],
        }
    )


def test_compare_features_returns_statistical_analysis() -> None:
    comparison = compare_features(_sample_features(), feature_names=["word_count", "sensational_count"])

    assert set(comparison["feature"]) == {"word_count", "sensational_count"}
    assert set(comparison.columns) == {
        "feature",
        "mann_whitney_p",
        "ttest_p",
        "cohens_d_fake_minus_real",
    }
