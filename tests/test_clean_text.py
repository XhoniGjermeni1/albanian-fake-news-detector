import pandas as pd

from src.preprocessing.clean_text import combine_title_content, normalize_spaces, prepare_text_dataframe


def test_normalize_spaces_keeps_albanian_letters_case_and_punctuation() -> None:
    text = "  Çfarë   ndodhi?\nËshtë LAJM!  "

    cleaned = normalize_spaces(text)

    assert cleaned == "Çfarë ndodhi? Është LAJM!"


def test_normalize_spaces_converts_decomposed_unicode_to_nfc() -> None:
    decomposed_text = "C\u0327fare\u0308 e\u0308shte\u0308"

    assert normalize_spaces(decomposed_text) == "Çfarë është"


def test_combine_title_content_handles_empty_values() -> None:
    assert combine_title_content("Titull", "Përmbajtje") == "Titull. Përmbajtje"
    assert combine_title_content("", "Vetëm përmbajtje") == "Vetëm përmbajtje"
    assert combine_title_content("Vetëm titull", "") == "Vetëm titull"


def test_prepare_text_dataframe_adds_clean_columns() -> None:
    dataframe = pd.DataFrame(
        {
            "title": ["  Titull   TEST  "],
            "content": [" Përmbajtje\nme   hapësira. "],
        }
    )

    result = prepare_text_dataframe(dataframe)

    assert result.loc[0, "title_clean"] == "Titull TEST"
    assert result.loc[0, "content_clean"] == "Përmbajtje me hapësira."
    assert result.loc[0, "model_text"] == "Titull TEST. Përmbajtje me hapësira."
