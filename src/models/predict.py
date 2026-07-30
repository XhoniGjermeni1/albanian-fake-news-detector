"""Prediction helper for the baseline fake news model."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.features.linguistic_features import extract_linguistic_features
from src.preprocessing.clean_text import combine_title_content

DEFAULT_MODEL_PATH = Path("models/baseline_tfidf_logreg.joblib")
DEFAULT_HYBRID_MODEL_PATH = Path("models/hybrid_tfidf_linguistic_logreg.joblib")
DEFAULT_CALIBRATED_MODEL_PATH = Path("models/calibrated_tfidf_logreg.joblib")
DEFAULT_REAL_THRESHOLD = 0.30
DEFAULT_FAKE_THRESHOLD = 0.70


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH):
    """Load a saved sklearn model pipeline."""
    return joblib.load(model_path)


def predict_news(title: str, content: str, model_path: str | Path = DEFAULT_MODEL_PATH) -> dict:
    """Predict whether one news article is real or fake."""
    model = load_model(model_path)
    model_text = combine_title_content(title, content)

    predicted_label = int(model.predict([model_text])[0])
    probabilities = model.predict_proba([model_text])[0]

    probability_real = float(probabilities[0])
    probability_fake = float(probabilities[1])

    return {
        "label": predicted_label,
        "label_name": "fake" if predicted_label == 1 else "real",
        "probability_real": round(probability_real, 4),
        "probability_fake": round(probability_fake, 4),
    }


def _marker_list(value: str) -> list[str]:
    """Convert a comma-separated marker value into a clean list."""
    return [marker.strip() for marker in str(value).split(",") if marker.strip()]


def build_linguistic_explanation(title: str, content: str) -> dict:
    """Return simple observable language signals for one news article."""
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
    """Convert fake probability into the three app decision levels."""
    if not 0 <= probability_fake <= 1:
        raise ValueError("probability_fake must be between 0 and 1.")
    if not 0 <= real_threshold < fake_threshold <= 1:
        raise ValueError("Thresholds must satisfy 0 <= real < fake <= 1.")

    if probability_fake < real_threshold:
        return "likely_real"
    if probability_fake > fake_threshold:
        return "likely_fake"
    return "uncertain"


def predict_news_for_app(
    title: str,
    content: str,
    model_path: str | Path = DEFAULT_CALIBRATED_MODEL_PATH,
    real_threshold: float = DEFAULT_REAL_THRESHOLD,
    fake_threshold: float = DEFAULT_FAKE_THRESHOLD,
    model=None,
) -> dict:
    """Return the calibrated prediction structure expected by the future app."""
    prediction_model = model if model is not None else load_model(model_path)
    model_text = combine_title_content(title, content)
    probabilities = prediction_model.predict_proba([model_text])[0]
    probability_by_label = {
        int(label): float(probability)
        for label, probability in zip(prediction_model.classes_, probabilities)
    }
    probability_real = probability_by_label.get(0, 0.0)
    probability_fake = probability_by_label.get(1, 0.0)

    return {
        "decision": classify_probability(
            probability_fake,
            real_threshold=real_threshold,
            fake_threshold=fake_threshold,
        ),
        "probability_real": round(probability_real, 4),
        "probability_fake": round(probability_fake, 4),
        "thresholds": {
            "likely_real_below": real_threshold,
            "likely_fake_above": fake_threshold,
        },
        "linguistic_explanation": build_linguistic_explanation(title, content),
        "notice": (
            "Rezultati bazohet në modelin statistikor dhe sinjale gjuhësore; "
            "nuk është verifikim faktik i lajmit."
        ),
    }


def predict_hybrid_news(
    title: str,
    content: str,
    model_path: str | Path = DEFAULT_HYBRID_MODEL_PATH,
) -> dict:
    """Predict with the hybrid model and include a simple language summary."""
    model = load_model(model_path)
    linguistic_features = extract_linguistic_features(title, content)
    model_row = pd.DataFrame(
        [{"model_text": combine_title_content(title, content), **linguistic_features}]
    )

    predicted_label = int(model.predict(model_row)[0])
    probabilities = model.predict_proba(model_row)[0]
    classes = model.named_steps["classifier"].classes_
    probability_by_label = {
        int(label): float(probability)
        for label, probability in zip(classes, probabilities)
    }

    return {
        "label": predicted_label,
        "label_name": "fake" if predicted_label == 1 else "real",
        "probability_real": round(probability_by_label.get(0, 0.0), 4),
        "probability_fake": round(probability_by_label.get(1, 0.0), 4),
        "linguistic_explanation": build_linguistic_explanation(title, content),
        "notice": (
            "Ky është probabilitet sipas modelit dhe karakteristikave gjuhësore; "
            "nuk zëvendëson verifikimin faktik."
        ),
    }
