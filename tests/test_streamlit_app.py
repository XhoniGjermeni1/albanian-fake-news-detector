from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app.streamlit_app import EXAMPLES, MODEL_PATH, validate_news_input

APP_PATH = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def test_validation_rejects_empty_input() -> None:
    errors, warnings = validate_news_input("  ", "\n")

    assert errors == ["Vendos të paktën titullin ose përmbajtjen e lajmit."]
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


def test_streamlit_initial_view_and_empty_submission() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()

    assert not app.exception
    assert app.title[0].value == "Detektuesi i lajmeve në shqip"
    assert app.text_input[0].label == "Titulli i lajmit"
    assert app.text_area[0].label == "Përmbajtja e lajmit"

    app.button[0].click().run()

    assert not app.exception
    assert any("të paktën titullin" in error.value for error in app.error)


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Run Day 6 to generate the calibrated model.")
def test_streamlit_handles_short_input() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    app.text_input[0].input("Titull")
    app.text_area[0].input("Tekst i shkurtër.")
    app.button[0].click().run()

    assert not app.exception
    assert any("më pak se 20 fjalë" in warning.value for warning in app.warning)
    assert any("Rezultati teknik" in caption.value for caption in app.caption)


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Run Day 6 to generate the calibrated model.")
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
