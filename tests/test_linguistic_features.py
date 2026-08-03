import pandas as pd

from src.features.linguistic_features import extract_features_dataframe, extract_linguistic_features


def test_extract_linguistic_features_counts_basic_structure() -> None:
    features = extract_linguistic_features(
        title="LAJM I FUNDIT!!!",
        content="Ja çfarë ndodhi... Sipas policia, thuhet se është tronditëse?",
    )

    assert features["word_count"] > 0
    assert features["sentence_count"] > 0
    assert features["title_length"] == len("LAJM I FUNDIT!!!")
    assert features["exclamation_count"] == 3
    assert features["question_count"] == 1
    assert features["comma_count"] == 1
    assert features["ellipsis_count"] == 1


def test_extract_linguistic_features_keeps_albanian_signals() -> None:
    features = extract_linguistic_features(
        title="Çfarë është kjo?",
        content="Ky është një lajm për Shqipërinë.",
    )

    assert features["e_count"] >= 2
    assert features["c_count"] >= 1
    assert features["diacritic_count"] == features["e_count"] + features["c_count"]


def test_extract_linguistic_features_normalizes_decomposed_diacritics() -> None:
    features = extract_linguistic_features(
        title="C\u0327fare\u0308 e\u0308shte\u0308 kjo?",
        content="Nje\u0308 pe\u0308rmbajtje me shkronja shqipe.",
    )

    assert features["c_count"] >= 1
    assert features["e_count"] >= 3


def test_extract_linguistic_features_detects_phrase_groups() -> None:
    features = extract_linguistic_features(
        title="Lajmi i fundit - Ekskluzive",
        content="Ja çfarë ndodhi. Sipas policia, thuhet se ka gjasa të ketë zhvillime.",
    )

    assert features["sensational_count"] >= 3
    assert "lajmi i fundit" in features["sensational_found"]
    assert "ekskluzive" in features["sensational_found"]
    assert "ja çfarë ndodhi" in features["sensational_found"]
    assert features["source_indicator_count"] >= 2
    assert "sipas" in features["source_indicators_found"]
    assert "policia" in features["source_indicators_found"]
    assert features["uncertainty_count"] >= 2
    assert "thuhet" in features["uncertainty_found"]
    assert "ka gjasa" in features["uncertainty_found"]


def test_extract_features_dataframe_adds_article_identifiers() -> None:
    dataframe = pd.DataFrame(
        {
            "article_id": ["real_1", "fake_1"],
            "pair_id": [1, 1],
            "label": [0, 1],
            "label_name": ["real", "fake"],
            "title": ["Titull real", "Titull fake"],
            "content": ["Sipas raporti, ky është tekst.", "Skandal! Nuk do ta besoni."],
        }
    )

    features = extract_features_dataframe(dataframe)

    assert len(features) == 2
    assert "article_id" in features.columns
    assert "word_count" in features.columns
    assert "sensational_count" in features.columns
