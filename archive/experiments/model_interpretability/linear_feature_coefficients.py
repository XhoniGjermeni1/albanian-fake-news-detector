"""Report global feature coefficients from the frozen linear classifier."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.predict_final import FINAL_MODEL_PATH  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "archive" / "reports" / "experiments" / "interpretability"
FEATURES_PATH = OUTPUT_DIR / "top_linear_features.csv"
TOP_FEATURES_PER_DIRECTION = 25


def extract_linear_coefficients(model) -> pd.DataFrame:
    """Return feature names and class-1-oriented Linear SVM coefficients."""
    calibrated = getattr(model, "calibrated_classifiers_", None)
    if not calibrated:
        raise TypeError("Expected a fitted CalibratedClassifierCV artifact.")
    if len(calibrated) != 1:
        raise ValueError("This report expects the frozen ensemble=False artifact.")

    estimator = calibrated[0].estimator
    features = estimator.named_steps["features"]
    classifier = estimator.named_steps["classifier"]
    classes = list(classifier.classes_)
    if classes != [0, 1]:
        raise ValueError(f"Expected class order [0, 1], found {classes}.")

    names = features.get_feature_names_out()
    coefficients = np.asarray(classifier.coef_[0], dtype=float)
    if len(names) != len(coefficients):
        raise RuntimeError("Feature names and coefficients are misaligned.")

    table = pd.DataFrame(
        {
            "feature_name": names,
            "coefficient": coefficients,
        }
    )
    parts = table["feature_name"].str.split("__", n=1, expand=True)
    table["branch"] = parts[0].map({"word": "word", "character": "character"})
    table["ngram"] = parts[1]
    table["direction"] = np.where(table["coefficient"].ge(0), "fake", "real")
    return table[
        [
            "branch",
            "direction",
            "ngram",
            "coefficient",
        ]
    ]


def select_top_features(
    coefficients: pd.DataFrame,
    top_n: int = TOP_FEATURES_PER_DIRECTION,
) -> pd.DataFrame:
    """Select the strongest positive and negative features per TF-IDF branch."""
    selected: list[pd.DataFrame] = []
    for branch in ["word", "character"]:
        branch_rows = coefficients.loc[coefficients["branch"].eq(branch)]
        fake = branch_rows.nlargest(top_n, "coefficient").copy()
        real = branch_rows.nsmallest(top_n, "coefficient").copy()
        fake["rank"] = range(1, len(fake) + 1)
        real["rank"] = range(1, len(real) + 1)
        selected.extend([fake, real])
    result = pd.concat(selected, ignore_index=True)
    return result[
        ["branch", "direction", "rank", "ngram", "coefficient"]
    ].sort_values(["branch", "direction", "rank"])


def save_outputs(top_features: pd.DataFrame) -> None:
    """Save the strongest global Word and Character TF-IDF features."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    top_features.to_csv(FEATURES_PATH, index=False, encoding="utf-8-sig")


def main() -> None:
    model = joblib.load(FINAL_MODEL_PATH)
    coefficients = extract_linear_coefficients(model)
    top_features = select_top_features(coefficients)
    save_outputs(top_features)
    print(
        f"Extracted {len(coefficients):,} coefficients and saved "
        f"{len(top_features)} top-feature rows."
    )


if __name__ == "__main__":
    main()
