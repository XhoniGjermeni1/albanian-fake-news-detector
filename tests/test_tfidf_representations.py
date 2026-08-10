import unicodedata

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from src.models.compare_tfidf_representations import (
    CHARACTER_CONFIGS,
    build_representation_pipeline,
    calculate_metrics,
    remove_albanian_diacritics,
    stability_variants,
)
from src.preprocessing.clean_text import combine_title_content


def test_representation_pipelines_have_expected_feature_types() -> None:
    config = CHARACTER_CONFIGS[0]

    word = build_representation_pipeline("word_tfidf", config)
    character = build_representation_pipeline("char_tfidf", config)
    combined = build_representation_pipeline("word_char_tfidf", config)

    assert isinstance(word.named_steps["features"], TfidfVectorizer)
    assert word.named_steps["features"].analyzer == "word"
    assert isinstance(character.named_steps["features"], TfidfVectorizer)
    assert character.named_steps["features"].analyzer == "char_wb"
    assert isinstance(combined.named_steps["features"], FeatureUnion)


def test_metrics_use_fake_as_positive_class() -> None:
    table = pd.DataFrame(
        {
            "label": [0, 0, 1, 1],
            "binary_prediction": [0, 1, 0, 1],
            "probability_real": [0.8, 0.4, 0.7, 0.1],
            "probability_fake": [0.2, 0.6, 0.3, 0.9],
            "decision": ["likely_real", "uncertain", "uncertain", "likely_fake"],
        }
    )

    metrics = calculate_metrics(table)

    assert metrics["accuracy"] == 0.5
    assert metrics["f1_fake"] == 0.5
    assert metrics["recall_real"] == 0.5
    assert metrics["recall_fake"] == 0.5
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1


def test_diacritic_variant_removes_only_albanian_diacritics() -> None:
    assert remove_albanian_diacritics("Është çështje në Tiranë.") == (
        "Eshte ceshtje ne Tirane."
    )


def test_unicode_variant_normalizes_back_to_same_model_text() -> None:
    title = "Çështja e ditës"
    content = "Është një përmbajtje në gjuhën shqipe."
    variants = stability_variants(title, content)
    nfd_title, nfd_content = variants["unicode_nfc_from_nfd"]

    assert unicodedata.normalize("NFC", nfd_title) == title
    assert combine_title_content(nfd_title, nfd_content) == combine_title_content(
        title, content
    )
