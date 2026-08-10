"""Shared helpers for probability decisions and linguistic explanations."""

from __future__ import annotations

from src.features.linguistic_features import extract_linguistic_features


DEFAULT_REAL_THRESHOLD = 0.30
DEFAULT_FAKE_THRESHOLD = 0.70


def _marker_list(value: str) -> list[str]:
    """Convert a comma-separated marker value into a clean list."""
    return [marker.strip() for marker in str(value).split(",") if marker.strip()]


def build_linguistic_explanation(title: str, content: str) -> dict:
    """Return observable language signals without affecting prediction."""
    features = extract_linguistic_features(title, content)
    return {
        "sensational_words_found": _marker_list(features["sensational_found"]),
        "source_markers_found": _marker_list(features["source_indicators_found"]),
        "uncertainty_markers_found": _marker_list(features["uncertainty_found"]),
        "exclamation_count": int(features["exclamation_count"]),
        "word_count": int(features["word_count"]),
        "text_length": int(features["character_count"]),
        "diacritic_ratio": float(features["diacritic_ratio"]),
        "uppercase_ratio": float(features["uppercase_char_ratio"]),
    }


def classify_probability(
    probability_fake: float,
    real_threshold: float = DEFAULT_REAL_THRESHOLD,
    fake_threshold: float = DEFAULT_FAKE_THRESHOLD,
) -> str:
    """Convert fake probability into the three application decision levels."""
    if not 0 <= probability_fake <= 1:
        raise ValueError("probability_fake must be between 0 and 1.")
    if not 0 <= real_threshold < fake_threshold <= 1:
        raise ValueError("Thresholds must satisfy 0 <= real < fake <= 1.")

    if probability_fake < real_threshold:
        return "likely_real"
    if probability_fake > fake_threshold:
        return "likely_fake"
    return "uncertain"
