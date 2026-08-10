import json
import unicodedata
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import src.models.predict_final as predict_final_module

from src.models.compare_classifiers import file_sha256, refresh_model_text
from src.models.finalize_model import (
    BASELINE_MODEL_PATH,
    FINAL_MANIFEST_PATH,
    SOURCE_MODEL_PATH,
    STREAMLIT_APP_PATH,
    verify_model_configuration,
    verify_preprocessing_contract,
)
from src.models.predict import DEFAULT_FAKE_THRESHOLD, DEFAULT_REAL_THRESHOLD
from src.models.predict_final import (
    FINAL_FAKE_THRESHOLD,
    FINAL_MODEL_PATH,
    FINAL_MODEL_VERSION,
    FINAL_REAL_THRESHOLD,
    predict_final_news,
    prepare_final_model_text,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def final_model():
    return joblib.load(FINAL_MODEL_PATH)


def test_final_artifact_is_byte_identical_to_day16_candidate() -> None:
    assert SOURCE_MODEL_PATH.exists()
    assert FINAL_MODEL_PATH.exists()
    assert file_sha256(FINAL_MODEL_PATH) == file_sha256(SOURCE_MODEL_PATH)


def test_final_pipeline_has_the_frozen_configuration() -> None:
    configuration = verify_model_configuration(final_model())

    assert configuration["calibration_method"] == "sigmoid"
    assert configuration["calibration_ensemble"] is False
    assert configuration["feature_union_branches"] == ["word", "character"]
    assert configuration["word_tfidf"]["ngram_range"] == (1, 2)
    assert configuration["character_tfidf"]["analyzer"] == "char_wb"
    assert configuration["character_tfidf"]["ngram_range"] == (3, 5)
    assert configuration["classifier"]["name"] == "LinearSVC"
    assert configuration["classifier"]["C"] == 1.0


def test_final_and_application_threshold_contracts_match() -> None:
    assert FINAL_REAL_THRESHOLD == DEFAULT_REAL_THRESHOLD == 0.30
    assert FINAL_FAKE_THRESHOLD == DEFAULT_FAKE_THRESHOLD == 0.70


def test_final_prediction_probabilities_and_decision_are_valid() -> None:
    result = predict_final_news(
        "Njoftim zyrtar",
        "Institucioni publikoi sot informacionin e plote.",
        model=final_model(),
    )

    assert result["model_version"] == FINAL_MODEL_VERSION
    assert result["model_id"] == "albanian_fake_news_word_char_svm_sigmoid_v1"
    assert 0.0 <= result["probability_real"] <= 1.0
    assert 0.0 <= result["probability_fake"] <= 1.0
    assert np.isclose(
        result["probability_real"] + result["probability_fake"], 1.0
    )
    assert result["decision"] in {"likely_real", "uncertain", "likely_fake"}


def test_prediction_is_identical_after_model_reload() -> None:
    title = "Kuvendi zhvilloi seancen plenare"
    content = "Njoftimi u publikua pas perfundimit te mbledhjes."
    first = predict_final_news(title, content, model=final_model())
    reloaded = predict_final_news(title, content, model=joblib.load(FINAL_MODEL_PATH))

    assert first["probability_real"] == reloaded["probability_real"]
    assert first["probability_fake"] == reloaded["probability_fake"]
    assert first["decision"] == reloaded["decision"]


def test_nfc_and_nfd_inputs_have_identical_predictions() -> None:
    title = "Çështja u diskutua në mbledhje"
    content = "Është publikuar një përmbledhje me të dhënat kryesore."
    nfc = predict_final_news(title, content, model=final_model())
    nfd = predict_final_news(
        unicodedata.normalize("NFD", title),
        unicodedata.normalize("NFD", content),
        model=final_model(),
    )

    assert nfc["probability_fake"] == nfd["probability_fake"]
    assert nfc["decision"] == nfd["decision"]
    assert unicodedata.is_normalized("NFC", prepare_final_model_text(title, content))


def test_linguistic_explanation_does_not_change_model_probabilities(
    monkeypatch,
) -> None:
    title = "Titull për kontroll"
    content = "Përmbajtje e njëjtë për të dyja prediction-et."
    before = predict_final_news(title, content, model=final_model())
    monkeypatch.setattr(
        predict_final_module,
        "build_linguistic_explanation",
        lambda _title, _content: {"changed": True},
    )
    after = predict_final_news(title, content, model=final_model())

    assert before["probability_real"] == after["probability_real"]
    assert before["probability_fake"] == after["probability_fake"]
    assert before["decision"] == after["decision"]
    assert after["linguistic_explanation"] == {"changed": True}


def test_evaluation_and_prediction_use_the_same_preprocessing() -> None:
    raw_test = pd.read_csv(
        PROJECT_ROOT / "data" / "interim" / "test.csv",
        encoding="utf-8-sig",
        keep_default_na=False,
    )
    refreshed, _ = refresh_model_text(raw_test)
    prediction_text = pd.Series(
        [
            prepare_final_model_text(row.title, row.content)
            for row in refreshed.itertuples(index=False)
        ]
    )

    assert prediction_text.tolist() == refreshed["model_text"].tolist()
    assert all(unicodedata.is_normalized("NFC", value) for value in prediction_text)
    assert verify_preprocessing_contract()["nfc_nfd_equivalence_passed"] is True


def test_final_manifest_documents_version_hash_and_limitations() -> None:
    manifest = json.loads(FINAL_MANIFEST_PATH.read_text(encoding="utf-8"))

    assert manifest["status"] == "final_frozen"
    assert manifest["model_version"] == FINAL_MODEL_VERSION
    assert manifest["artifact"]["final_sha256"] == file_sha256(FINAL_MODEL_PATH)
    assert manifest["fact_checking"] is False
    assert manifest["streamlit_integration"] == "integrated_day18"
    assert manifest["streamlit_runtime"]["app_path"] == "app\\streamlit_app.py"
    assert (
        manifest["streamlit_runtime"]["prediction_function"]
        == "src.models.predict_final.predict_final_news"
    )
    assert manifest["streamlit_runtime"]["linguistic_features_role"] == (
        "explanation_only"
    )
    assert "length_bias" in manifest["limitations"]


def test_day17_does_not_train_or_replace_the_baseline() -> None:
    finalizer_source = (
        PROJECT_ROOT / "src" / "models" / "finalize_model.py"
    ).read_text(encoding="utf-8")

    assert ".fit(" not in finalizer_source
    assert BASELINE_MODEL_PATH.exists()
    assert BASELINE_MODEL_PATH != FINAL_MODEL_PATH


def test_streamlit_integration_uses_only_the_final_contract() -> None:
    app_source = STREAMLIT_APP_PATH.read_text(encoding="utf-8")
    final_prediction_source = (
        PROJECT_ROOT / "src" / "models" / "predict_final.py"
    ).read_text(encoding="utf-8")

    assert "predict_final_news" in app_source
    assert "FINAL_MODEL_PATH" in app_source
    assert "DEFAULT_CALIBRATED_MODEL_PATH" not in app_source
    assert "predict_news_for_app" not in app_source
    assert "from src.models.predict import" not in final_prediction_source
