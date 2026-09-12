"""Integrity checks for the frozen final model and preprocessing contract."""

from __future__ import annotations

import hashlib
import unicodedata
from pathlib import Path

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from src.models.predict_final import (
    FINAL_FAKE_THRESHOLD,
    FINAL_REAL_THRESHOLD,
    prepare_final_model_text,
)
from src.models.prediction_utils import DEFAULT_FAKE_THRESHOLD, DEFAULT_REAL_THRESHOLD


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 fingerprint of one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model_configuration(model) -> dict:
    """Require the exact fitted Word+Char Linear SVM calibration setup."""
    if not isinstance(model, CalibratedClassifierCV):
        raise TypeError("Final model is not CalibratedClassifierCV.")
    if model.method != "sigmoid" or model.ensemble is not False:
        raise ValueError("Final model does not use non-ensemble sigmoid calibration.")
    if list(model.classes_) != [0, 1]:
        raise ValueError(f"Unexpected classes: {model.classes_}")
    if len(model.calibrated_classifiers_) != 1:
        raise ValueError("Expected one fitted calibrated classifier.")

    calibrated = model.calibrated_classifiers_[0]
    pipeline = calibrated.estimator
    if not isinstance(pipeline, Pipeline):
        raise TypeError("Calibrated estimator is not an sklearn Pipeline.")
    features = pipeline.named_steps.get("features")
    classifier = pipeline.named_steps.get("classifier")
    if not isinstance(features, FeatureUnion):
        raise TypeError("Final representation is not a FeatureUnion.")
    if not isinstance(classifier, LinearSVC):
        raise TypeError("Final classifier is not LinearSVC.")

    vectorizers = dict(features.transformer_list)
    if set(vectorizers) != {"word", "character"}:
        raise ValueError(f"Unexpected feature branches: {set(vectorizers)}")
    word = vectorizers["word"]
    character = vectorizers["character"]
    if not isinstance(word, TfidfVectorizer) or not isinstance(
        character, TfidfVectorizer
    ):
        raise TypeError("Both final feature branches must be TfidfVectorizer.")

    expected_word = {
        "analyzer": "word",
        "lowercase": False,
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_features": 30000,
    }
    expected_character = {
        "analyzer": "char_wb",
        "lowercase": False,
        "ngram_range": (3, 5),
        "min_df": 2,
        "max_features": 50000,
    }
    for parameter, expected in expected_word.items():
        if getattr(word, parameter) != expected:
            raise ValueError(f"Unexpected Word TF-IDF {parameter}.")
    for parameter, expected in expected_character.items():
        if getattr(character, parameter) != expected:
            raise ValueError(f"Unexpected Character TF-IDF {parameter}.")
    if float(classifier.C) != 1.0:
        raise ValueError("Final Linear SVM C is not 1.0.")
    if classifier.class_weight != "balanced":
        raise ValueError("Final Linear SVM class_weight changed.")

    calibrator_names = [item.__class__.__name__ for item in calibrated.calibrators]
    if calibrator_names != ["_SigmoidCalibration"]:
        raise ValueError(f"Unexpected fitted calibrators: {calibrator_names}")
    return {
        "sklearn_object": model.__class__.__name__,
        "calibration_method": model.method,
        "calibration_ensemble": bool(model.ensemble),
        "calibration_folds": len(model.cv),
        "fitted_calibrators": calibrator_names,
        "feature_union_branches": list(vectorizers),
        "word_tfidf": expected_word,
        "character_tfidf": expected_character,
        "classifier": {
            "name": classifier.__class__.__name__,
            "C": float(classifier.C),
            "class_weight": classifier.class_weight,
            "max_iter": int(classifier.max_iter),
            "random_state": int(classifier.random_state),
        },
        "classes": [int(value) for value in model.classes_],
    }


def verify_preprocessing_contract() -> dict:
    """Verify Unicode normalization and application thresholds."""
    nfc_title = "Çështja për ëndrrën"
    nfc_content = "Është një përmbledhje e shkurtër."
    prepared_nfc = prepare_final_model_text(nfc_title, nfc_content)
    prepared_nfd = prepare_final_model_text(
        unicodedata.normalize("NFD", nfc_title),
        unicodedata.normalize("NFD", nfc_content),
    )
    if prepared_nfc != prepared_nfd:
        raise ValueError("NFC and NFD inputs do not produce identical model text.")
    if not unicodedata.is_normalized("NFC", prepared_nfd):
        raise ValueError("Final preprocessing output is not Unicode NFC.")
    if DEFAULT_REAL_THRESHOLD != FINAL_REAL_THRESHOLD:
        raise ValueError("Application real threshold differs from final threshold.")
    if DEFAULT_FAKE_THRESHOLD != FINAL_FAKE_THRESHOLD:
        raise ValueError("Application fake threshold differs from final threshold.")
    return {
        "function": "src.preprocessing.clean_text.combine_title_content",
        "unicode_normalization": "NFC",
        "whitespace": "collapsed",
        "title_content_separator": ". ",
        "lowercase_removed": False,
        "punctuation_removed": False,
        "nfc_nfd_equivalence_passed": True,
        "application_thresholds_match": True,
    }
