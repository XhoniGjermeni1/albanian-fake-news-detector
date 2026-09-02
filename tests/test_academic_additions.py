import joblib
import numpy as np

from experiments.baseline.dummy_baseline import evaluate_dummy_baseline
from experiments.model_interpretability.linear_feature_coefficients import (
    extract_linear_coefficients,
    select_top_features,
)
from src.evaluation.data_utils import add_word_counts
from src.evaluation.metrics import classification_metrics
from src.models.builders import (
    FINAL_SVM_C,
    FIXED_CHAR_CONFIG,
    build_fixed_features,
    build_svm,
)
from src.models.compare_classifiers import (
    add_word_counts as historical_add_word_counts,
    classification_metrics as historical_classification_metrics,
)
from src.models.predict_final import FINAL_MODEL_PATH


def test_shared_utilities_keep_historical_imports() -> None:
    assert historical_add_word_counts is add_word_counts
    assert historical_classification_metrics is classification_metrics


def test_frozen_model_builders_keep_selected_configuration() -> None:
    features = build_fixed_features()
    word = dict(features.transformer_list)["word"]
    character = dict(features.transformer_list)["character"]
    classifier = build_svm()

    assert word.ngram_range == (1, 2)
    assert word.lowercase is False
    assert character.analyzer == FIXED_CHAR_CONFIG["analyzer"] == "char_wb"
    assert character.ngram_range == (3, 5)
    assert classifier.C == FINAL_SVM_C == 1.0
    assert classifier.class_weight == "balanced"


def test_dummy_baseline_uses_clean_frozen_test_split() -> None:
    metrics, comparison = evaluate_dummy_baseline()

    assert metrics["train_rows"] == 3195
    assert metrics["test_rows"] == 792
    assert metrics["excluded_train_test_duplicates"] == 7
    assert metrics["confusion_matrix"] == [[399, 0], [393, 0]]
    assert metrics["f1_fake"] == 0.0
    assert set(comparison["model"]) == {
        "dummy_most_frequent",
        "baseline_word_logreg",
        "final_word_char_svm",
    }


def test_frozen_linear_coefficients_are_aligned() -> None:
    model = joblib.load(FINAL_MODEL_PATH)
    coefficients = extract_linear_coefficients(model)
    top = select_top_features(coefficients)

    assert len(coefficients) == 80000
    assert set(coefficients["branch"]) == {"word", "character"}
    assert np.isfinite(coefficients["coefficient"]).all()
    assert len(top) == 100
    assert set(top["direction"]) == {"real", "fake"}
