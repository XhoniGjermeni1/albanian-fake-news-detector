"""Streamlit entrypoint for the frozen Albanian news model."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import streamlit as st

from app.streamlit_ui import (
    EXAMPLES,
    FACT_CHECK_WARNING,
    apply_page_style,
    build_human_explanations,
    initialize_state,
    inspect_model_assets,
    render_result,
    render_sidebar,
    validate_news_input,
)
from src.models.predict_final import (
    FINAL_MANIFEST_PATH,
    FINAL_MODEL_PATH,
    load_final_model,
    predict_final_news,
)

MODEL_PATH = FINAL_MODEL_PATH
MANIFEST_PATH = FINAL_MANIFEST_PATH
LOGGER = logging.getLogger(__name__)

__all__ = [
    "EXAMPLES",
    "FACT_CHECK_WARNING",
    "MANIFEST_PATH",
    "MODEL_PATH",
    "build_human_explanations",
    "get_cached_model",
    "inspect_model_assets",
    "main",
    "predict_with_final_model",
    "validate_news_input",
]


@st.cache_resource(show_spinner=False)
def get_cached_model(model_path: str):
    """Load the model once for the lifetime of the Streamlit process."""
    return load_final_model(model_path)


def predict_with_final_model(title: str, content: str, model) -> dict:
    """Use the single frozen prediction path shared with evaluation."""
    return predict_final_news(title, content, model=model)


def main() -> None:
    st.set_page_config(
        page_title="Detektuesi i lajmeve në shqip",
        layout="centered",
        initial_sidebar_state="auto",
    )
    apply_page_style()
    initialize_state()
    manifest, asset_errors = inspect_model_assets(MODEL_PATH, MANIFEST_PATH)
    render_sidebar(manifest)

    st.title("Detektuesi i lajmeve në shqip")
    st.write(
        "Vendos titullin dhe përmbajtjen për të marrë një vlerësim gjuhësor nga modeli."
    )
    st.warning(FACT_CHECK_WARNING)
    for asset_error in asset_errors:
        st.error(asset_error)

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
            disabled=bool(asset_errors),
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
                    st.session_state["analysis_result"] = predict_with_final_model(
                        title,
                        content,
                        model,
                    )
            except FileNotFoundError:
                LOGGER.exception("Final model file was not found")
                st.error(
                    "Modeli final mungon. Ekzekuto python "
                    "src\\models\\finalize_model.py."
                )
            except Exception:
                LOGGER.exception("Prediction failed")
                st.error(
                    "Modeli nuk mund të ngarkohej ose analiza dështoi. "
                    "Kontrollo skedarin e modelit dhe provo përsëri."
                )

    if st.session_state["analysis_result"] is not None:
        render_result(
            st.session_state["analysis_result"],
            st.session_state["analysis_warnings"],
        )


if __name__ == "__main__":
    main()
