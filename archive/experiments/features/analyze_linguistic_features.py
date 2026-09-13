"""Run the historical statistical analysis of linguistic features."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "linguistic_features.csv"
REPORTS_DIR = PROJECT_ROOT / "archive" / "reports"

COMPARISON_PATH = REPORTS_DIR / "day4_feature_comparison.csv"

KEY_FEATURES = [
    "word_count",
    "sentence_count",
    "avg_sentence_length",
    "exclamation_count",
    "question_count",
    "uppercase_word_ratio",
    "sensational_count",
    "sensational_ratio",
    "source_indicator_count",
    "source_indicator_ratio",
    "uncertainty_count",
    "uncertainty_ratio",
    "diacritic_ratio",
    "title_length",
    "content_length",
]

def load_features() -> pd.DataFrame:
    """Load the linguistic feature table."""
    return pd.read_csv(FEATURES_PATH, encoding="utf-8-sig", keep_default_na=False)


def cohens_d(fake_values: pd.Series, real_values: pd.Series) -> float:
    """Calculate a simple Cohen's d effect size."""
    fake = fake_values.dropna().astype(float)
    real = real_values.dropna().astype(float)
    pooled_std = np.sqrt((fake.var(ddof=1) + real.var(ddof=1)) / 2)

    if pooled_std == 0 or np.isnan(pooled_std):
        return 0.0

    return float((fake.mean() - real.mean()) / pooled_std)


def compare_features(features: pd.DataFrame, feature_names: list[str] = KEY_FEATURES) -> pd.DataFrame:
    """Run the historical statistical tests for each linguistic feature."""
    rows = []

    for feature in feature_names:
        fake = features.loc[features["label_name"] == "fake", feature].astype(float)
        real = features.loc[features["label_name"] == "real", feature].astype(float)

        try:
            mann_whitney_p = float(mannwhitneyu(fake, real, alternative="two-sided").pvalue)
        except ValueError:
            mann_whitney_p = np.nan

        try:
            ttest_p = float(ttest_ind(fake, real, equal_var=False, nan_policy="omit").pvalue)
        except ValueError:
            ttest_p = np.nan

        rows.append(
            {
                "feature": feature,
                "mann_whitney_p": mann_whitney_p,
                "ttest_p": ttest_p,
                "cohens_d_fake_minus_real": round(cohens_d(fake, real), 6),
            }
        )

    return pd.DataFrame(rows)


def run_analysis() -> pd.DataFrame:
    """Run and save the historical Day 4 statistical analysis."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    features = load_features()
    comparison = compare_features(features)

    comparison.to_csv(COMPARISON_PATH, index=False, encoding="utf-8-sig")
    return comparison


def main() -> None:
    comparison = run_analysis()

    print("=== Day 4 linguistic feature analysis ===")
    print(f"Features analyzed: {len(comparison)}")
    print(f"Comparison saved: {COMPARISON_PATH}")


if __name__ == "__main__":
    main()
