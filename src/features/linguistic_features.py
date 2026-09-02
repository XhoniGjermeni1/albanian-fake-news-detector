"""Extract simple linguistic and stylistic features from Albanian news text."""

from __future__ import annotations

import re

import pandas as pd

from src.preprocessing.clean_text import normalize_spaces

WORD_PATTERN = re.compile(r"[^\W\d_]+(?:[-'][^\W\d_]+)?", re.UNICODE)
SENTENCE_PATTERN = re.compile(r"[.!?]+")

SENSATIONAL_PHRASES = [
    "tronditëse",
    "skandal",
    "shokuese",
    "urgjente",
    "e pabesueshme",
    "nuk do ta besoni",
    "lajm i fundit",
    "lajmi i fundit",
    "ekskluzive",
    "bombë",
    "alarm",
    "ja çfarë ndodhi",
]

SOURCE_PHRASES = [
    "sipas",
    "deklaroi",
    "konfirmoi",
    "raportoi",
    "bëri të ditur",
    "burime zyrtare",
    "ministria",
    "policia",
    "institucioni",
    "raporti",
    "studimi",
]

UNCERTAINTY_PHRASES = [
    "thuhet",
    "mendohet",
    "dyshohet",
    "ndoshta",
    "mund të",
    "ka gjasa",
    "supozohet",
    "burime pranë",
]

POSSIBLE_MISSING_DIACRITIC_WORDS = [
    "eshte",
    "cfare",
    "shqiperi",
    "kosove",
    "per",
    "nje",
    "behet",
    "kerkon",
]


def get_words(text: str) -> list[str]:
    """Return word-like tokens, preserving Albanian letters."""
    return WORD_PATTERN.findall(text)


def count_sentences(text: str, word_count: int) -> int:
    """Count sentence-ending punctuation marks, with a fallback for non-empty text."""
    sentence_count = len([part for part in SENTENCE_PATTERN.split(text) if part.strip()])
    if sentence_count == 0 and word_count > 0:
        return 1
    return sentence_count


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    """Avoid division by zero in feature ratios."""
    if denominator == 0:
        return 0.0

    return round(float(numerator) / float(denominator), 6)


def find_phrases(text: str, phrases: list[str]) -> list[str]:
    """Find phrases with case-insensitive matching."""
    normalized_text = normalize_spaces(text).casefold()
    found_phrases = []

    for phrase in phrases:
        normalized_phrase = phrase.casefold()
        pattern = rf"(?<!\w){re.escape(normalized_phrase)}(?!\w)"
        if re.search(pattern, normalized_text):
            found_phrases.append(phrase)

    return found_phrases


def count_phrase_occurrences(text: str, phrases: list[str]) -> int:
    """Count total occurrences of a phrase list."""
    normalized_text = normalize_spaces(text).casefold()
    total = 0

    for phrase in phrases:
        normalized_phrase = phrase.casefold()
        pattern = rf"(?<!\w){re.escape(normalized_phrase)}(?!\w)"
        total += len(re.findall(pattern, normalized_text))

    return total


def extract_linguistic_features(title: str, content: str) -> dict:
    """Extract simple linguistic features for one article."""
    title = normalize_spaces(title)
    content = normalize_spaces(content)
    full_text = normalize_spaces(f"{title} {content}")

    words = get_words(full_text)
    word_count = len(words)
    sentence_count = count_sentences(full_text, word_count)
    character_count = len(full_text)

    word_lengths = [len(word) for word in words]
    uppercase_words = [word for word in words if word.isupper() and len(word) > 1]
    letters = [char for char in full_text if char.isalpha()]
    uppercase_letters = [char for char in letters if char.isupper()]

    sensational_found = find_phrases(full_text, SENSATIONAL_PHRASES)
    source_found = find_phrases(full_text, SOURCE_PHRASES)
    uncertainty_found = find_phrases(full_text, UNCERTAINTY_PHRASES)
    possible_missing_diacritics = find_phrases(full_text, POSSIBLE_MISSING_DIACRITIC_WORDS)

    exclamation_count = full_text.count("!")
    question_count = full_text.count("?")
    comma_count = full_text.count(",")
    quote_count = sum(full_text.count(mark) for mark in ['"', "'", "“", "”", "„", "«", "»"])
    ellipsis_count = full_text.count("...") + full_text.count("…")
    e_count = full_text.casefold().count("ë")
    c_count = full_text.casefold().count("ç")

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "character_count": character_count,
        "avg_word_length": round(sum(word_lengths) / word_count, 2) if word_count else 0.0,
        "avg_sentence_length": round(word_count / sentence_count, 2) if sentence_count else 0.0,
        "title_length": len(title),
        "content_length": len(content),
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "comma_count": comma_count,
        "quote_count": quote_count,
        "ellipsis_count": ellipsis_count,
        "exclamation_ratio": safe_ratio(exclamation_count, word_count),
        "question_ratio": safe_ratio(question_count, word_count),
        "uppercase_word_count": len(uppercase_words),
        "uppercase_word_ratio": safe_ratio(len(uppercase_words), word_count),
        "uppercase_char_ratio": safe_ratio(len(uppercase_letters), len(letters)),
        "title_excessive_uppercase": int(safe_ratio(sum(char.isupper() for char in title), len([char for char in title if char.isalpha()])) > 0.6),
        "e_count": e_count,
        "c_count": c_count,
        "diacritic_count": e_count + c_count,
        "diacritic_ratio": safe_ratio(e_count + c_count, max(character_count, 1)),
        "possible_missing_diacritic_count": len(possible_missing_diacritics),
        "possible_missing_diacritic_words": ", ".join(possible_missing_diacritics),
        "sensational_count": count_phrase_occurrences(full_text, SENSATIONAL_PHRASES),
        "sensational_ratio": safe_ratio(count_phrase_occurrences(full_text, SENSATIONAL_PHRASES), word_count),
        "sensational_found": ", ".join(sensational_found),
        "source_indicator_count": count_phrase_occurrences(full_text, SOURCE_PHRASES),
        "source_indicator_ratio": safe_ratio(count_phrase_occurrences(full_text, SOURCE_PHRASES), word_count),
        "source_indicators_found": ", ".join(source_found),
        "uncertainty_count": count_phrase_occurrences(full_text, UNCERTAINTY_PHRASES),
        "uncertainty_ratio": safe_ratio(count_phrase_occurrences(full_text, UNCERTAINTY_PHRASES), word_count),
        "uncertainty_found": ", ".join(uncertainty_found),
    }


def extract_features_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Extract linguistic features for every article in a DataFrame."""
    feature_rows = [
        extract_linguistic_features(row["title"], row["content"])
        for _, row in dataframe.iterrows()
    ]

    features = pd.DataFrame(feature_rows)
    identifier_columns = ["article_id", "pair_id", "label", "label_name"]
    return pd.concat([dataframe[identifier_columns].reset_index(drop=True), features], axis=1)
