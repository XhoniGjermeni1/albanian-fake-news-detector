"""Prediction helper for the baseline fake news model."""

from __future__ import annotations

from pathlib import Path

import joblib

from src.preprocessing.clean_text import combine_title_content

DEFAULT_MODEL_PATH = Path("models/baseline_tfidf_logreg.joblib")


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
