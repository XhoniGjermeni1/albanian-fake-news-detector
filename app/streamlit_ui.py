"""Validation and rendering helpers for the Streamlit interface."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from src.models.predict_final import (
    FINAL_MANIFEST_PATH,
    FINAL_MODEL_ID,
    FINAL_MODEL_PATH,
    FINAL_MODEL_VERSION,
)

MIN_CONTENT_WORDS = 20
LONG_TEXT_CHARACTERS = 20_000
MAX_INPUT_CHARACTERS = 100_000
STYLE_PATH = Path(__file__).with_name("style.css")

FACT_CHECK_WARNING = (
    "Ky aplikacion nuk verifikon faktet e lajmit. Rezultati bazohet vetëm në "
    "modele gjuhësore dhe statistikore dhe duhet interpretuar me kujdes."
)

EXAMPLES = {
    "Raport institucional": {
        "title": "Masat kundër pandemisë, qeveria merr vendim edhe për çerdhet",
        "content": (
            "Qeveria e Kosovës sot ka miratuar 28 masat e reja kundër përhapjes së "
            "koronavirusit. Përveç të tjerash, qeveria Hoti ka marrë vendim edhe për "
            "çerdhet publike dhe ato private. Në vendim thuhet se ushtrimi i "
            "veprimtarisë së çerdheve publike dhe private në tërë territorin e "
            "Republikës së Kosovës lejohet pas vlerësimit dhe mbikëqyrjes nga "
            "autoritetet komunale, në përputhje me Manualin për mbrojtjen nga "
            "përhapja e COVID-19."
        ),
    },
    "Njoftim i shkurtër": {
        "title": "Njoftim nga universiteti",
        "content": (
            "Universiteti bëri të ditur se regjistrimet për semestrin e ri do të "
            "hapen të hënën. Sipas njoftimit zyrtar, studentët duhet të dorëzojnë "
            "dokumentet brenda afatit."
        ),
    },
    "Titull sensacional": {
        "title": "LAJM I FUNDIT! Zbulimi që tronditi vendin",
        "content": (
            "Skandal! Nuk do ta besoni çfarë ndodhi. Lajmi po shpërndahet me "
            "shpejtësi dhe ka shkaktuar alarm te qytetarët. Shumë persona e kanë "
            "ndarë pretendimin pa treguar burimin fillestar."
        ),
    },
}

DECISION_TEXT = {
    "likely_real": "Lajmi duket më shumë si i vërtetë sipas modelit",
    "uncertain": "Modeli nuk ka siguri të mjaftueshme",
    "likely_fake": "Lajmi duket më shumë si i pavërtetë sipas modelit",
}


def validate_news_input(
    title: str | None,
    content: str | None,
    min_content_words: int = MIN_CONTENT_WORDS,
    long_text_characters: int = LONG_TEXT_CHARACTERS,
    max_input_characters: int = MAX_INPUT_CHARACTERS,
) -> tuple[list[str], list[str]]:
    """Return blocking errors and non-blocking warnings for the form input."""
    clean_title = "" if title is None else str(title).strip()
    clean_content = "" if content is None else str(content).strip()
    combined_text = f"{clean_title} {clean_content}".strip()
    total_characters = len(combined_text)
    total_words = len(combined_text.split())
    errors: list[str] = []
    warnings: list[str] = []

    if not combined_text:
        errors.append("Vendos të paktën titullin ose përmbajtjen e lajmit.")
        return errors, warnings

    if not any(character.isalnum() for character in combined_text):
        errors.append("Teksti duhet të përmbajë të paktën një shkronjë ose numër.")
        return errors, warnings

    if total_characters > max_input_characters:
        errors.append(
            f"Teksti është shumë i gjatë. Kufiri i analizës është {max_input_characters:,} karaktere."
        )
        return errors, warnings

    if clean_title and not clean_content:
        warnings.append(
            "Po analizohet vetëm titulli. Pa përmbajtjen, rezultati mund të jetë më pak i besueshëm."
        )
    elif total_words < min_content_words:
        warnings.append(
            f"Teksti ka më pak se {min_content_words} fjalë. Edhe nëse modeli shfaq "
            "një përqindje të lartë, rezultati mund të jetë më pak i besueshëm."
        )

    if total_characters > long_text_characters:
        warnings.append(
            "Teksti është shumë i gjatë, ndaj analiza mund të kërkojë pak më shumë kohë."
        )

    return errors, warnings


def inspect_model_assets(
    model_path: str | Path = FINAL_MODEL_PATH,
    manifest_path: str | Path = FINAL_MANIFEST_PATH,
) -> tuple[dict | None, list[str]]:
    """Return final model metadata and clear blocking asset errors."""
    model_file = Path(model_path)
    manifest_file = Path(manifest_path)
    errors: list[str] = []
    if not model_file.exists():
        errors.append(
            f"Artefakti final i modelit mungon: {model_file}. "
            "Ekzekuto python src\\models\\finalize_model.py."
        )
    if not manifest_file.exists():
        errors.append(
            f"Manifesti i modelit final mungon: {manifest_file}. "
            "Ekzekuto python src\\models\\finalize_model.py."
        )
    if errors:
        return None, errors

    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return None, [f"Manifesti i modelit final nuk mund të lexohet: {error}"]

    if manifest.get("model_id") != FINAL_MODEL_ID:
        errors.append("Manifesti nuk përputhet me ID-në e modelit final.")
    if manifest.get("model_version") != FINAL_MODEL_VERSION:
        errors.append("Manifesti nuk përputhet me versionin final 1.0.0.")
    manifest_artifact = str(manifest.get("artifact", {}).get("final_path", ""))
    manifest_name = Path(manifest_artifact.replace("\\", "/")).name
    if manifest_name != model_file.name:
        errors.append("Manifesti referon një artefakt tjetër nga modeli final.")
    return (manifest if not errors else None), errors


def apply_page_style() -> None:
    """Load the static stylesheet used by the application."""
    css = STYLE_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def initialize_state() -> None:
    st.session_state.setdefault("news_title", "")
    st.session_state.setdefault("news_content", "")
    st.session_state.setdefault("analysis_result", None)
    st.session_state.setdefault("analysis_warnings", [])


def render_sidebar(manifest: dict | None) -> None:
    with st.sidebar:
        st.subheader("Shembuj testimi")
        selected_example = st.selectbox(
            "Zgjidh një shembull",
            options=list(EXAMPLES),
            index=None,
            placeholder="Zgjidh tekstin",
        )
        if st.button(
            "Ngarko shembullin",
            use_container_width=True,
            disabled=selected_example is None,
        ):
            example = EXAMPLES[selected_example]
            st.session_state["news_title"] = example["title"]
            st.session_state["news_content"] = example["content"]
            st.session_state["analysis_result"] = None
            st.session_state["analysis_warnings"] = []
            st.rerun()

        st.divider()
        st.subheader("Rreth modelit")
        model_version = (
            manifest.get("model_version", FINAL_MODEL_VERSION)
            if manifest
            else FINAL_MODEL_VERSION
        )
        st.markdown(
            f"""
            Modeli final është trajnuar mbi një korpus shqiptar me etiketa real/fake.

            - Word + Character TF-IDF
            - Linear SVM
            - Kalibrim sigmoid
            - versioni `{model_version}`
            """
        )
        st.caption(
            "Zona `uncertain` është 30%-70%. Karakteristikat gjuhësore përdoren "
            "vetëm për shpjegim; modeli nuk kontrollon burimet ose faktet."
        )


def _format_markers(markers: list[str]) -> str:
    return ", ".join(markers) if markers else "Asnjë e gjetur"


def build_human_explanations(explanation: dict) -> list[str]:
    """Turn numeric language features into cautious, readable observations."""
    word_count = int(explanation["word_count"])
    text_length = int(explanation["text_length"])
    exclamation_count = int(explanation["exclamation_count"])
    uppercase_ratio = float(explanation["uppercase_ratio"])
    diacritic_ratio = float(explanation["diacritic_ratio"])

    if word_count < MIN_CONTENT_WORDS:
        length_description = "Teksti është shumë i shkurtër për një vlerësim të qëndrueshëm."
    elif word_count < 100:
        length_description = "Teksti ka gjatësi të moderuar."
    else:
        length_description = "Teksti është relativisht i gjatë."

    if exclamation_count == 0:
        exclamation_description = "Nuk u gjetën pikëçuditëse."
    elif exclamation_count <= 2:
        exclamation_description = (
            f"U gjetën {exclamation_count} pikëçuditëse; përdorimi i tyre është i kufizuar."
        )
    else:
        exclamation_description = (
            f"U gjetën {exclamation_count} pikëçuditëse, pra teksti i përdor shpesh."
        )

    if uppercase_ratio == 0:
        uppercase_description = "Nuk u gjetën shkronja të mëdha."
    elif uppercase_ratio <= 0.05:
        uppercase_description = "Përdorimi i shkronjave të mëdha duket i ulët."
    elif uppercase_ratio <= 0.15:
        uppercase_description = "Përdorimi i shkronjave të mëdha duket i moderuar."
    else:
        uppercase_description = "Përdorimi i shkronjave të mëdha duket i lartë."

    if diacritic_ratio == 0:
        diacritic_description = "Nuk u gjetën shkronjat shqipe ë/ç në këtë tekst."
    elif diacritic_ratio < 0.01:
        diacritic_description = "Përdorimi i shkronjave shqipe ë/ç duket i ulët."
    elif diacritic_ratio <= 0.04:
        diacritic_description = "Përdorimi i shkronjave shqipe ë/ç duket i moderuar."
    else:
        diacritic_description = "Shkronjat shqipe ë/ç përdoren relativisht shpesh."

    sensational_markers = explanation["sensational_words_found"]
    source_markers = explanation["source_markers_found"]
    uncertainty_markers = explanation["uncertainty_markers_found"]

    sensational_description = (
        f"U gjetën fjalë ose shprehje sensacionale: {_format_markers(sensational_markers)}."
        if sensational_markers
        else "Nuk u gjetën fjalë nga lista e shprehjeve sensacionale."
    )
    source_description = (
        f"U gjetën tregues që referojnë një burim: {_format_markers(source_markers)}."
        if source_markers
        else "Nuk u gjetën tregues burimi nga lista e përdorur."
    )
    uncertainty_description = (
        f"U gjetën shprehje pasigurie: {_format_markers(uncertainty_markers)}."
        if uncertainty_markers
        else "Nuk u gjetën shprehje pasigurie nga lista e përdorur."
    )

    return [
        f"Teksti ka {word_count} fjalë dhe {text_length} karaktere. {length_description}",
        exclamation_description,
        uppercase_description,
        diacritic_description,
        sensational_description,
        source_description,
        uncertainty_description,
    ]


def render_decision(result: dict) -> None:
    decision = result["decision"]
    message = DECISION_TEXT[decision]
    probability_fake = result["probability_fake"]
    real_threshold = result["thresholds"]["likely_real_below"]
    fake_threshold = result["thresholds"]["likely_fake_above"]

    if decision == "likely_real":
        st.success(message)
        st.write(
            f"Probabiliteti Fake ({probability_fake:.1%}) është nën pragun "
            f"{real_threshold:.0%} të modelit."
        )
    elif decision == "likely_fake":
        st.error(message)
        st.write(
            f"Probabiliteti Fake ({probability_fake:.1%}) është mbi pragun "
            f"{fake_threshold:.0%} të modelit."
        )
    else:
        st.warning(message)
        st.info(
            f"Probabiliteti Fake ({probability_fake:.1%}) ndodhet brenda zonës së "
            f"pasigurt {real_threshold:.0%}-{fake_threshold:.0%}. Kjo nuk do të "
            "thotë se lajmi është gjysmë real dhe gjysmë fake; do të thotë se "
            "modeli nuk ka siguri të mjaftueshme. Kontrolloje lajmin me burime të jashtme."
        )

    st.caption(f"Rezultati teknik: `{decision}`")
    st.warning(FACT_CHECK_WARNING)


def render_result(result: dict, warnings: list[str]) -> None:
    for warning in warnings:
        st.warning(warning)

    st.divider()
    st.subheader("Rezultati")
    st.caption(
        f"Modeli `{result['model_id']}` · versioni `{result['model_version']}`"
    )
    render_decision(result)

    st.subheader("Probabilitetet")
    real_column, fake_column = st.columns(2)
    real_column.metric("Real sipas modelit", f"{result['probability_real']:.1%}")
    fake_column.metric("Fake sipas modelit", f"{result['probability_fake']:.1%}")
    real_column.progress(
        result["probability_real"],
        text=f"Real: {result['probability_real']:.1%}",
    )
    fake_column.progress(
        result["probability_fake"],
        text=f"Fake: {result['probability_fake']:.1%}",
    )
    st.caption("Përqindjet janë probabilitete të modelit, jo prova faktike.")

    explanation = result["linguistic_explanation"]
    st.subheader("Karakteristika të vëzhguara në tekst")
    words_column, length_column, punctuation_column = st.columns(3)
    words_column.metric("Fjalë", explanation["word_count"])
    length_column.metric("Gjatësia", explanation["text_length"], help="Numri i karaktereve")
    punctuation_column.metric("Pikëçuditëse", explanation["exclamation_count"])

    uppercase_column, diacritic_column = st.columns(2)
    uppercase_column.metric("Shkronja të mëdha", f"{explanation['uppercase_ratio']:.2%}")
    diacritic_column.metric("Shkronja ë/ç", f"{explanation['diacritic_ratio']:.2%}")

    observations = build_human_explanations(explanation)
    st.markdown("\n".join(f"- {observation}" for observation in observations))
    st.info(
        "Këto janë karakteristika të vëzhguara në tekst. Asnjëra prej tyre nuk provon vetë nëse lajmi është real ose fake."
    )
