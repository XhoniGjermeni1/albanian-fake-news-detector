"""Simple builders for the frozen Word + Character Linear SVM design."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

FIXED_CHAR_CONFIG = {
    "config_name": "char_wb_3_5",
    "analyzer": "char_wb",
    "ngram_min": 3,
    "ngram_max": 5,
    "min_df": 2,
    "max_features": 50000,
}
FINAL_SVM_C = 1.0


def build_word_vectorizer() -> TfidfVectorizer:
    """Return the frozen word TF-IDF configuration."""
    return TfidfVectorizer(
        lowercase=False,
        ngram_range=(1, 2),
        min_df=2,
        max_features=30000,
    )


def build_char_vectorizer(config: dict) -> TfidfVectorizer:
    """Build one predeclared character TF-IDF configuration."""
    return TfidfVectorizer(
        lowercase=False,
        analyzer=config["analyzer"],
        ngram_range=(config["ngram_min"], config["ngram_max"]),
        min_df=config["min_df"],
        max_features=config["max_features"],
    )


def build_fixed_features(char_config: dict = FIXED_CHAR_CONFIG) -> FeatureUnion:
    """Build the frozen Word + Character TF-IDF representation."""
    return FeatureUnion(
        [
            ("word", build_word_vectorizer()),
            ("character", build_char_vectorizer(char_config)),
        ]
    )


def build_svm(c_value: float = FINAL_SVM_C) -> LinearSVC:
    """Build the selected Linear SVM classifier."""
    return LinearSVC(
        C=float(c_value),
        class_weight="balanced",
        max_iter=5000,
        random_state=42,
    )


def build_svm_pipeline(c_value: float = FINAL_SVM_C) -> Pipeline:
    """Build the frozen TF-IDF representation followed by Linear SVM."""
    return Pipeline(
        [
            ("features", build_fixed_features(FIXED_CHAR_CONFIG)),
            ("classifier", build_svm(c_value)),
        ]
    )
