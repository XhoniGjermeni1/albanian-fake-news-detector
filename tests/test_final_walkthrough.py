import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "02_final_walkthrough.ipynb"
DEMO_CASES_PATH = PROJECT_ROOT / "reports" / "day19_demo_cases.csv"
FROZEN_DEMOS_PATH = PROJECT_ROOT / "reports" / "day17_final_demo_cases.csv"


def load_notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def test_final_walkthrough_is_valid_and_uses_frozen_outputs() -> None:
    notebook = load_notebook()
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) >= 20

    markdown = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    for section in [
        "Ngarkimi dhe kontrolli bazë",
        "Preprocessing bazë",
        "Word + Character TF-IDF",
        "Karakteristikat gjuhësore",
        "Prediction-et e demonstrimit",
        "Metrikat finale dhe confusion matrix",
        "Vlerësimi i jashtëm pilot",
        "Shembull error analysis",
    ]:
        assert section in markdown

    assert "load_final_model" in code
    assert "predict_final_news" in code
    assert "day17_final_metrics.json" in code
    assert "day17_final_external_predictions.csv" in code
    assert ".fit(" not in code
    assert "run_finalization(" not in code
    assert "train_test_split(" not in code


def test_day19_demo_cases_are_complete_and_match_day17() -> None:
    demo_cases = pd.read_csv(DEMO_CASES_PATH, encoding="utf-8-sig")
    frozen = pd.read_csv(FROZEN_DEMOS_PATH)
    expected_types = {
        "likely_real_correct",
        "likely_fake_correct",
        "uncertain",
        "false_positive",
        "false_negative",
    }

    assert set(demo_cases["demo_type"]) == expected_types
    assert demo_cases["article_id"].is_unique
    assert demo_cases["title"].fillna("").str.strip().ne("").all()
    assert demo_cases["content"].fillna("").str.strip().ne("").all()
    assert demo_cases["expected_decision"].isin(
        ["likely_real", "uncertain", "likely_fake"]
    ).all()
    assert np.allclose(
        demo_cases["probability_real"] + demo_cases["probability_fake"], 1.0
    )

    comparison = demo_cases.merge(
        frozen[
            [
                "article_id",
                "decision",
                "probability_real",
                "probability_fake",
            ]
        ],
        on="article_id",
        suffixes=("_day19", "_day17"),
        validate="one_to_one",
    )
    assert (comparison["expected_decision"] == comparison["decision"]).all()
    assert np.allclose(
        comparison["probability_real_day19"], comparison["probability_real_day17"]
    )
    assert np.allclose(
        comparison["probability_fake_day19"], comparison["probability_fake_day17"]
    )


def test_clean_computer_instructions_include_final_runtime() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    requirements = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")

    assert "py -3.11 -m venv .venv" in readme
    assert "python -m pip install -r requirements.txt" in readme
    assert "python -m streamlit run app\\streamlit_app.py" in readme
    assert "python -m pytest -q" in readme
    assert "02_final_walkthrough.ipynb" in readme
    assert "final_word_char_linear_svm_calibrated_v1.joblib" in readme
    assert "scikit-learn==1.8.0" in requirements
    assert "jupyterlab" in requirements
