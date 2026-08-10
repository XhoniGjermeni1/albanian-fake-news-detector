from configparser import ConfigParser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_dataset_submodule_is_declared() -> None:
    config = ConfigParser()
    config.read(PROJECT_ROOT / ".gitmodules", encoding="utf-8")
    section = 'submodule "data/raw/alb-fake-news-corpus"'

    assert config.has_section(section)
    assert config.get(section, "path") == "data/raw/alb-fake-news-corpus"
    assert config.get(section, "url") == (
        "https://github.com/rexshijaku/alb-fake-news-corpus.git"
    )


def test_gitignore_excludes_experiments_but_keeps_final_model() -> None:
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")

    assert ".venv/" in gitignore
    assert "__pycache__/" in gitignore
    assert "models/*.joblib" in gitignore
    assert "!models/final_word_char_linear_svm_calibrated_v1.joblib" in gitignore


def test_final_documentation_and_dependencies_are_reproducible() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    requirements = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")

    for required_text in [
        "Modeli Final",
        "Pipeline-i",
        "Rezultatet Kryesore",
        "Kufizimet",
        "python -m streamlit run app\\streamlit_app.py",
        "python -m jupyter lab notebooks\\02_final_walkthrough.ipynb",
        "python -m pytest -q",
        "nuk zëvendëson fact-checking-un",
    ]:
        assert required_text in readme

    for requirement in [
        "pandas==3.0.1",
        "numpy==2.4.3",
        "pyarrow==23.0.1",
        "scikit-learn==1.8.0",
        "scipy==1.17.1",
        "joblib==1.5.3",
        "streamlit==1.60.0",
        "pytest==9.1.1",
        "jupyterlab==4.6.2",
        "ipykernel==7.2.0",
    ]:
        assert requirement in requirements

    assert "plotly" not in requirements
    assert "statsmodels" not in requirements
