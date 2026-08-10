from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app.streamlit_app import (
    EXAMPLES,
    FACT_CHECK_WARNING,
    MANIFEST_PATH,
    MODEL_PATH,
    build_human_explanations,
    get_cached_model,
    inspect_model_assets,
    predict_with_final_model,
    validate_news_input,
)
from src.models.predict_final import FINAL_MANIFEST_PATH, FINAL_MODEL_PATH

APP_PATH = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
PROJECT_ROOT = APP_PATH.parents[1]


@pytest.fixture(scope="module")
def frozen_demo_inputs() -> pd.DataFrame:
    demos = pd.read_csv(PROJECT_ROOT / "reports" / "day17_final_demo_cases.csv")
    test = pd.read_csv(
        PROJECT_ROOT / "data" / "interim" / "test.csv",
        encoding="utf-8-sig",
        keep_default_na=False,
    )
    return demos.merge(
        test[["article_id", "title", "content"]],
        on=["article_id", "title"],
        how="left",
        validate="one_to_one",
    )


def test_validation_rejects_empty_input() -> None:
    errors, warnings = validate_news_input("  ", "\n")

    assert errors == ["Vendos të paktën titullin ose përmbajtjen e lajmit."]
    assert warnings == []


@pytest.mark.parametrize("content", ["!!!...", "🚨🚨🚨", "\u200b\u200c"])
def test_validation_rejects_input_without_letters_or_numbers(content: str) -> None:
    errors, warnings = validate_news_input("", content)

    assert errors == ["Teksti duhet të përmbajë të paktën një shkronjë ose numër."]
    assert warnings == []


def test_validation_warns_for_title_only() -> None:
    errors, warnings = validate_news_input("Vetëm titull", "")

    assert errors == []
    assert "vetëm titulli" in warnings[0]


def test_validation_warns_for_short_content() -> None:
    errors, warnings = validate_news_input("", "Tekst shumë i shkurtër.")

    assert errors == []
    assert "më pak se 20 fjalë" in warnings[0]


def test_validation_accepts_content_without_title() -> None:
    content = " ".join(["fjalë"] * 25)
    errors, warnings = validate_news_input("", content)

    assert errors == []
    assert warnings == []


def test_validation_warns_for_long_content() -> None:
    content = " ".join(["fjalë"] * 25)
    errors, warnings = validate_news_input(
        "",
        content,
        long_text_characters=50,
    )

    assert errors == []
    assert any("shumë i gjatë" in warning for warning in warnings)


def test_validation_rejects_content_above_maximum() -> None:
    errors, warnings = validate_news_input(
        "",
        "tekst tepër i gjatë",
        max_input_characters=10,
    )

    assert any("Kufiri i analizës" in error for error in errors)
    assert warnings == []


def test_validation_uses_real_long_and_maximum_limits() -> None:
    long_content = " ".join(["fjalë"] * 4_000)
    errors, warnings = validate_news_input("", long_content)

    assert errors == []
    assert any("shumë i gjatë" in warning for warning in warnings)

    errors, warnings = validate_news_input("", "a" * 100_001)
    assert any("Kufiri i analizës" in error for error in errors)
    assert warnings == []


def test_app_points_only_to_final_model_and_manifest() -> None:
    assert MODEL_PATH == FINAL_MODEL_PATH
    assert MANIFEST_PATH == FINAL_MANIFEST_PATH

    manifest, errors = inspect_model_assets()
    assert errors == []
    assert manifest is not None
    assert manifest["model_version"] == "1.0.0"
    assert get_cached_model(str(MODEL_PATH)) is get_cached_model(str(MODEL_PATH))


def test_missing_final_assets_return_clear_errors(tmp_path: Path) -> None:
    manifest, errors = inspect_model_assets(
        tmp_path / "missing-model.joblib",
        tmp_path / "missing-manifest.json",
    )

    assert manifest is None
    assert len(errors) == 2
    assert "Artefakti final" in errors[0]
    assert "Manifesti" in errors[1]


def test_human_explanation_interprets_language_signals() -> None:
    observations = build_human_explanations(
        {
            "word_count": 12,
            "text_length": 90,
            "exclamation_count": 4,
            "uppercase_ratio": 0.20,
            "diacritic_ratio": 0.0,
            "sensational_words_found": ["skandal"],
            "source_markers_found": ["sipas"],
            "uncertainty_markers_found": ["mund të"],
        }
    )
    explanation_text = " ".join(observations)

    assert "shumë i shkurtër" in explanation_text
    assert "4 pikëçuditëse" in explanation_text
    assert "shkronjave të mëdha duket i lartë" in explanation_text
    assert "Nuk u gjetën shkronjat shqipe ë/ç" in explanation_text
    assert "skandal" in explanation_text
    assert "sipas" in explanation_text
    assert "mund të" in explanation_text


def test_streamlit_initial_view_and_empty_submission() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()

    assert not app.exception
    assert app.title[0].value == "Detektuesi i lajmeve në shqip"
    assert app.text_input[0].label == "Titulli i lajmit"
    assert app.text_area[0].label == "Përmbajtja e lajmit"
    markdown_text = " ".join(item.value for item in app.markdown)
    assert "Word + Character TF-IDF" in markdown_text
    assert "Linear SVM" in markdown_text
    assert "Kalibrim sigmoid" in markdown_text
    assert "versioni `1.0.0`" in markdown_text

    app.button[0].click().run()

    assert not app.exception
    assert any("të paktën titullin" in error.value for error in app.error)


def test_app_prediction_function_matches_all_frozen_day17_cases(
    frozen_demo_inputs: pd.DataFrame,
) -> None:
    model = get_cached_model(str(MODEL_PATH))

    for row in frozen_demo_inputs.itertuples(index=False):
        result = predict_with_final_model(row.title, row.content, model)
        assert result["decision"] == row.decision
        assert np.isclose(result["probability_real"], row.probability_real)
        assert np.isclose(result["probability_fake"], row.probability_fake)
        assert result["thresholds"] == {
            "likely_real_below": 0.30,
            "likely_fake_above": 0.70,
        }
        assert result["model_version"] == "1.0.0"


def test_streamlit_matches_all_six_frozen_day17_cases(
    frozen_demo_inputs: pd.DataFrame,
) -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()

    for row in frozen_demo_inputs.itertuples(index=False):
        app.text_input[0].input(row.title)
        app.text_area[0].input(row.content)
        app.button[0].click().run()

        assert not app.exception
        assert any(
            f"`{row.decision}`" in caption.value for caption in app.caption
        )
        metric_values = {metric.label: metric.value for metric in app.metric}
        assert metric_values["Real sipas modelit"] == f"{row.probability_real:.1%}"
        assert metric_values["Fake sipas modelit"] == f"{row.probability_fake:.1%}"
        assert any("versioni `1.0.0`" in caption.value for caption in app.caption)


def test_streamlit_blocks_punctuation_only_input() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    app.text_input[0].input("!!!")
    app.text_area[0].input("🚨 ... ??? \u200b")
    app.button[0].click().run()

    assert not app.exception
    assert any("shkronjë ose numër" in error.value for error in app.error)
    assert not any("Rezultati teknik" in caption.value for caption in app.caption)


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Run Day 17 to freeze the final model.")
def test_streamlit_handles_short_input() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    app.text_input[0].input("Titull")
    app.text_area[0].input("Tekst i shkurtër.")
    app.button[0].click().run()

    assert not app.exception
    assert any("më pak se 20 fjalë" in warning.value for warning in app.warning)
    assert any("Rezultati teknik" in caption.value for caption in app.caption)


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Run Day 17 to freeze the final model.")
def test_streamlit_examples_cover_all_decisions() -> None:
    expected_decisions = {
        "Raport institucional": "likely_real",
        "Njoftim i shkurtër": "uncertain",
        "Titull sensacional": "likely_fake",
    }
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()

    for example_name, expected_decision in expected_decisions.items():
        example = EXAMPLES[example_name]
        app.text_input[0].input(example["title"])
        app.text_area[0].input(example["content"])
        app.button[0].click().run()

        assert not app.exception
        assert any(
            f"`{expected_decision}`" in caption.value
            for caption in app.caption
        )
        assert len(app.metric) == 7
        assert any("karakteristika të vëzhguara" in info.value for info in app.info)


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Run Day 17 to freeze the final model.")
def test_streamlit_explains_uncertain_result_and_fact_check_limit() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    example = EXAMPLES["Njoftim i shkurtër"]
    app.text_input[0].input(example["title"])
    app.text_area[0].input(example["content"])
    app.button[0].click().run()

    assert not app.exception
    assert any("zonës së pasigurt" in info.value for info in app.info)
    assert any("nuk do të thotë" in info.value.lower() for info in app.info)
    assert sum(FACT_CHECK_WARNING in warning.value for warning in app.warning) >= 2


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Run Day 17 to freeze the final model.")
@pytest.mark.parametrize(
    ("title", "content", "expected_warning"),
    [
        ("Vetëm titull", "", "vetëm titulli"),
        ("", " ".join(["Përmbajtje"] * 25), None),
    ],
)
def test_streamlit_accepts_title_or_content_separately(
    title: str,
    content: str,
    expected_warning: str | None,
) -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    app.text_input[0].input(title)
    app.text_area[0].input(content)
    app.button[0].click().run()

    assert not app.exception
    assert any("Rezultati teknik" in caption.value for caption in app.caption)
    if expected_warning:
        assert any(expected_warning in warning.value for warning in app.warning)


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Run Day 17 to freeze the final model.")
def test_streamlit_handles_unusual_unicode_and_language_markers() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    app.text_input[0].input("C\u0327fare\u0308 njoftimi është ky?!!!")
    app.text_area[0].input(
        "Sipas raportit, ministria deklaroi dhe policia konfirmoi të dhënat!!! "
        "Institucioni bëri të ditur se dokumenti është kontrolluar. ✅"
    )
    app.button[0].click().run()

    assert not app.exception
    assert len(app.metric) == 7
    assert any("Rezultati teknik" in caption.value for caption in app.caption)
    assert any("sipas" in markdown.value.lower() for markdown in app.markdown)
    assert any(FACT_CHECK_WARNING in warning.value for warning in app.warning)


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Run Day 17 to freeze the final model.")
def test_streamlit_handles_very_long_allowed_input() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    app.text_input[0].input("Tekst i gjatë")
    app.text_area[0].input(" ".join(["përmbajtje"] * 2_500))
    app.button[0].click().run()

    assert not app.exception
    assert any("shumë i gjatë" in warning.value for warning in app.warning)
    assert any("Rezultati teknik" in caption.value for caption in app.caption)
