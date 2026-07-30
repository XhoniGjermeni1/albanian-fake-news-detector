"""Minimal Streamlit interface for the calibrated Albanian news model."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import streamlit as st

from src.models.predict import (
    DEFAULT_CALIBRATED_MODEL_PATH,
    load_model,
    predict_news_for_app,
)

MODEL_PATH = PROJECT_ROOT / DEFAULT_CALIBRATED_MODEL_PATH
MIN_CONTENT_WORDS = 20
LOGGER = logging.getLogger(__name__)

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
) -> tuple[list[str], list[str]]:
    """Return blocking errors and non-blocking warnings for the form input."""
    clean_title = "" if title is None else str(title).strip()
    clean_content = "" if content is None else str(content).strip()
    errors: list[str] = []
    warnings: list[str] = []

    if not clean_title and not clean_content:
        errors.append("Vendos të paktën titullin ose përmbajtjen e lajmit.")
        return errors, warnings

    if clean_title and not clean_content:
        warnings.append(
            "Po analizohet vetëm titulli. Pa përmbajtjen, rezultati mund të jetë më pak i besueshëm."
        )
    elif clean_content and len(clean_content.split()) < min_content_words:
        warnings.append(
            f"Përmbajtja ka më pak se {min_content_words} fjalë. Rezultati mund të jetë i pasigurt."
        )

    return errors, warnings


@st.cache_resource(show_spinner=False)
def get_cached_model(model_path: str):
    """Load the model once for the lifetime of the Streamlit process."""
    return load_model(model_path)


def apply_page_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            --app-ink: #17211f;
            --app-muted: #56635f;
            --app-surface: #ffffff;
            --app-border: #d8dfdc;
            --app-accent: #176b62;
        }
        .stApp {
            background: #f5f7f5;
            color: var(--app-ink);
        }
        [data-testid="stHeader"] {
            background: rgba(245, 247, 245, 0.94);
        }
        [data-testid="stSidebar"] {
            background: #e9eeec;
            border-right: 1px solid var(--app-border);
        }
        .block-container {
            max-width: 980px;
            padding-top: 2.25rem;
            padding-bottom: 3rem;
        }
        h1, h2, h3, p, label, button {
            letter-spacing: 0 !important;
        }
        h1 {
            color: #173c37;
        }
        [data-testid="stMetric"] {
            background: var(--app-surface);
            border: 1px solid var(--app-border);
            border-radius: 6px;
            padding: 0.85rem 1rem;
            min-height: 108px;
        }
        .stButton button, .stFormSubmitButton button {
            border-radius: 6px;
            min-height: 2.7rem;
        }
        .stFormSubmitButton button {
            background: var(--app-accent);
            border-color: var(--app-accent);
        }
        textarea, input {
            border-radius: 6px !important;
        }
        [data-testid="stCaptionContainer"] {
            color: var(--app-muted);
        }
        @media (max-width: 640px) {
            .block-container {
                padding-top: 1.25rem;
                padding-left: 1rem;
                padding-right: 1rem;
            }
            [data-testid="stMetric"] {
                min-height: 96px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_state() -> None:
    st.session_state.setdefault("news_title", "")
    st.session_state.setdefault("news_content", "")
    st.session_state.setdefault("analysis_result", None)
    st.session_state.setdefault("analysis_warnings", [])


def render_sidebar() -> None:
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
        st.markdown(
            """
            Modeli është trajnuar mbi një dataset shqiptar me lajme real/fake.

            - TF-IDF + Logistic Regression
            - probabilitete me sigmoid calibration
            - linguistic features vetëm për shpjegim
            - pragje 30% dhe 70%
            """
        )
        st.caption(
            "Rezultati duhet interpretuar me kujdes. Modeli analizon tekstin dhe nuk kontrollon burimet ose faktet."
        )


def _format_markers(markers: list[str]) -> str:
    return ", ".join(markers) if markers else "Asnjë e gjetur"


def render_decision(result: dict) -> None:
    decision = result["decision"]
    message = DECISION_TEXT[decision]

    if decision == "likely_real":
        st.success(message)
    elif decision == "likely_fake":
        st.error(message)
    else:
        st.warning(message)

    st.caption(f"Rezultati teknik: `{decision}`")


def render_result(result: dict, warnings: list[str]) -> None:
    for warning in warnings:
        st.warning(warning)

    st.divider()
    st.subheader("Rezultati")
    render_decision(result)

    st.subheader("Probabilitetet")
    real_column, fake_column = st.columns(2)
    real_column.metric("Real sipas modelit", f"{result['probability_real']:.1%}")
    fake_column.metric("Fake sipas modelit", f"{result['probability_fake']:.1%}")
    st.caption("Përqindjet janë probabilitete të modelit, jo prova faktike.")

    explanation = result["linguistic_explanation"]
    st.subheader("Karakteristika të vëzhguara në tekst")
    words_column, length_column, punctuation_column = st.columns(3)
    words_column.metric("Fjalë", explanation["word_count"])
    length_column.metric("Gjatësia", explanation["text_length"], help="Numri i karaktereve")
    punctuation_column.metric("Pikëçuditëse", explanation["exclamation_count"])

    uppercase_column, diacritic_column = st.columns(2)
    uppercase_column.metric("Uppercase ratio", f"{explanation['uppercase_ratio']:.2%}")
    diacritic_column.metric("Diacritic ratio", f"{explanation['diacritic_ratio']:.2%}")

    st.markdown(
        f"**Fjalë sensacionale:** {_format_markers(explanation['sensational_words_found'])}"
    )
    st.markdown(
        f"**Tregues burimi:** {_format_markers(explanation['source_markers_found'])}"
    )
    st.markdown(
        f"**Shprehje pasigurie:** {_format_markers(explanation['uncertainty_markers_found'])}"
    )
    st.info(
        "Këto janë karakteristika të vëzhguara në tekst. Asnjëra prej tyre nuk provon vetë nëse lajmi është real ose fake."
    )
    st.warning(result["notice"])


def main() -> None:
    st.set_page_config(
        page_title="Detektuesi i lajmeve në shqip",
        layout="centered",
        initial_sidebar_state="auto",
    )
    apply_page_style()
    initialize_state()
    render_sidebar()

    st.title("Detektuesi i lajmeve në shqip")
    st.write(
        "Vendos titullin dhe përmbajtjen për të marrë një vlerësim gjuhësor nga modeli."
    )
    st.warning(
        "Ky mjet nuk bën verifikim faktesh. Rezultati është probabilitet sipas modelit."
    )

    with st.form("news_analysis_form"):
        title = st.text_input(
            "Titulli i lajmit",
            key="news_title",
            placeholder="Shkruaj titullin",
        )
        content = st.text_area(
            "Përmbajtja e lajmit",
            key="news_content",
            placeholder="Shkruaj ose vendos përmbajtjen e lajmit",
            height=260,
        )
        submitted = st.form_submit_button(
            "Analizo lajmin",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        errors, warnings = validate_news_input(title, content)
        st.session_state["analysis_result"] = None
        st.session_state["analysis_warnings"] = warnings

        if errors:
            for error in errors:
                st.error(error)
        else:
            try:
                with st.spinner("Po analizohet teksti..."):
                    model = get_cached_model(str(MODEL_PATH))
                    st.session_state["analysis_result"] = predict_news_for_app(
                        title,
                        content,
                        model_path=MODEL_PATH,
                        model=model,
                    )
            except FileNotFoundError:
                LOGGER.exception("Calibrated model file was not found")
                st.error(
                    "Modeli i kalibruar mungon. Ekzekuto fillimisht analizën e Ditës 6."
                )
            except Exception:
                LOGGER.exception("Prediction failed")
                st.error("Analiza nuk mund të përfundohej. Kontrollo input-in dhe provo përsëri.")

    if st.session_state["analysis_result"] is not None:
        render_result(
            st.session_state["analysis_result"],
            st.session_state["analysis_warnings"],
        )


if __name__ == "__main__":
    main()
