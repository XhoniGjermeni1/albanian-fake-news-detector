# Përkufizon rrugën e vetme zyrtare të parashikimit: përgatit tekstin, ngarkon artefaktin,
# verifikon probabilitetet dhe zbaton pragjet 0.30/0.70. Rezultati përfshin vendimin
# determinist dhe sinjalet gjuhësore që shfaqen në aplikacion.

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from src.models.prediction_utils import (
    DEFAULT_FAKE_THRESHOLD,
    DEFAULT_REAL_THRESHOLD,
    build_linguistic_explanation,
    classify_probability,
)
from src.preprocessing.clean_text import combine_title_content


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINAL_MODEL_ID = "albanian_fake_news_word_char_svm_sigmoid_v1"
FINAL_MODEL_VERSION = "1.0.0"
FINAL_MODEL_PATH = (
    PROJECT_ROOT / "final_model" / "final_word_char_linear_svm_calibrated_v1.joblib"
)
FINAL_MANIFEST_PATH = PROJECT_ROOT / "final_model" / "final_model_v1_manifest.json"
FINAL_REAL_THRESHOLD = DEFAULT_REAL_THRESHOLD
FINAL_FAKE_THRESHOLD = DEFAULT_FAKE_THRESHOLD
FINAL_NOTICE = (
    "Rezultati është probabilitet sipas modelit dhe sinjaleve gjuhësore; "
    "nuk është verifikim faktik i lajmit."
)


def prepare_final_model_text(title: str, content: str) -> str:
    return combine_title_content(title, content)


def load_final_model(model_path: str | Path = FINAL_MODEL_PATH):
    return joblib.load(model_path)


def _predict_probabilities(prediction_model, model_text: str) -> tuple[float, float]:
    probabilities = np.asarray(
        prediction_model.predict_proba([model_text])[0], dtype=float
    )
    probability_by_label = {
        int(label): float(probability)
        for label, probability in zip(prediction_model.classes_, probabilities)
    }
    if set(probability_by_label) != {0, 1}:
        raise ValueError(
            f"Final model must expose classes 0 and 1, found {prediction_model.classes_}."
        )

    probability_real = probability_by_label[0]
    probability_fake = probability_by_label[1]
    if not np.all(np.isfinite([probability_real, probability_fake])):
        raise ValueError("Final model returned non-finite probabilities.")
    if not np.allclose(probability_real + probability_fake, 1.0, atol=1e-12):
        raise ValueError("Final model probabilities do not sum to one.")
    if not 0.0 <= probability_real <= 1.0 or not 0.0 <= probability_fake <= 1.0:
        raise ValueError("Final model probabilities are outside [0, 1].")
    return probability_real, probability_fake


def _build_prediction_result(
    title: str,
    content: str,
    probability_real: float,
    probability_fake: float,
) -> dict:
    binary_prediction = int(probability_fake >= 0.5)
    return {
        "model_id": FINAL_MODEL_ID,
        "model_version": FINAL_MODEL_VERSION,
        "decision": classify_probability(
            probability_fake,
            real_threshold=FINAL_REAL_THRESHOLD,
            fake_threshold=FINAL_FAKE_THRESHOLD,
        ),
        "binary_prediction": binary_prediction,
        "label_name": "fake" if binary_prediction == 1 else "real",
        "probability_real": probability_real,
        "probability_fake": probability_fake,
        "thresholds": {
            "likely_real_below": FINAL_REAL_THRESHOLD,
            "likely_fake_above": FINAL_FAKE_THRESHOLD,
        },
        "linguistic_explanation": build_linguistic_explanation(title, content),
        "notice": FINAL_NOTICE,
    }


def predict_final_news(
    title: str,
    content: str,
    model=None,
    model_path: str | Path = FINAL_MODEL_PATH,
) -> dict:
    prediction_model = model if model is not None else load_final_model(model_path)
    model_text = prepare_final_model_text(title, content)
    probability_real, probability_fake = _predict_probabilities(
        prediction_model, model_text
    )
    return _build_prediction_result(
        title, content, probability_real, probability_fake
    )
