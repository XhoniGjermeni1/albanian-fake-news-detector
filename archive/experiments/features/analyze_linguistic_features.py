"""Analyze linguistic features and run a small feature-only baseline."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "linguistic_features.csv"
TRAIN_PATH = PROJECT_ROOT / "data" / "interim" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "interim" / "test.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODEL_PATH = PROJECT_ROOT / "models" / "linguistic_features_logreg.joblib"

QUALITY_PATH = REPORTS_DIR / "day4_feature_quality.json"
COMPARISON_PATH = REPORTS_DIR / "day4_feature_comparison.csv"
MODEL_METRICS_PATH = REPORTS_DIR / "day4_linguistic_only_model_metrics.json"

LOGGER = logging.getLogger(__name__)

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

TEXT_MARKER_COLUMNS = [
    "possible_missing_diacritic_words",
    "sensational_found",
    "source_indicators_found",
    "uncertainty_found",
]


def load_features() -> pd.DataFrame:
    """Load the linguistic feature table."""
    return pd.read_csv(FEATURES_PATH, encoding="utf-8-sig", keep_default_na=False)


def numeric_feature_columns(features: pd.DataFrame) -> list[str]:
    """Return numeric feature columns, excluding identifiers and labels."""
    ignored_columns = {"pair_id", "label"}
    return [
        column
        for column in features.select_dtypes(include="number").columns
        if column not in ignored_columns
    ]


def quality_checks(features: pd.DataFrame) -> dict:
    """Check whether the feature dataset is usable for analysis."""
    numeric = features.select_dtypes(include="number")
    ratio_columns = [column for column in numeric.columns if column.endswith("_ratio")]
    count_like_columns = [
        column
        for column in numeric.columns
        if column.endswith("_count") or column.endswith("_length") or column in {"sentence_count", "character_count"}
    ]

    ratio_issues = {
        column: int(((features[column] < 0) | (features[column] > 1)).sum())
        for column in ratio_columns
    }
    negative_issues = {
        column: int((features[column] < 0).sum())
        for column in count_like_columns
    }

    constant_numeric_features = [
        column
        for column in numeric_feature_columns(features)
        if features[column].nunique(dropna=False) <= 1
    ]

    expected_empty_marker_values = {
        column: int(features[column].astype(str).str.strip().eq("").sum())
        for column in TEXT_MARKER_COLUMNS
        if column in features.columns
    }

    return {
        "rows": int(len(features)),
        "columns": int(len(features.columns)),
        "label_counts": {
            str(label): int(count)
            for label, count in features["label_name"].value_counts().sort_index().items()
        },
        "duplicate_article_ids": int(features["article_id"].duplicated().sum()),
        "duplicate_rows": int(features.duplicated().sum()),
        "missing_values_total": int(features.isna().sum().sum()),
        "numeric_missing_values_total": int(numeric.isna().sum().sum()),
        "infinite_values_total": int(np.isinf(numeric.to_numpy()).sum()),
        "ratio_values_outside_0_1": ratio_issues,
        "negative_count_values": negative_issues,
        "articles_with_word_count_below_20": int((features["word_count"] < 20).sum()),
        "constant_numeric_features": constant_numeric_features,
        "expected_empty_marker_values": expected_empty_marker_values,
    }


def cohens_d(fake_values: pd.Series, real_values: pd.Series) -> float:
    """Calculate a simple Cohen's d effect size."""
    fake = fake_values.dropna().astype(float)
    real = real_values.dropna().astype(float)
    pooled_std = np.sqrt((fake.var(ddof=1) + real.var(ddof=1)) / 2)

    if pooled_std == 0 or np.isnan(pooled_std):
        return 0.0

    return float((fake.mean() - real.mean()) / pooled_std)


def compare_features(features: pd.DataFrame, feature_names: list[str] = KEY_FEATURES) -> pd.DataFrame:
    """Create descriptive statistics and simple statistical tests by label."""
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

        fake_mean = float(fake.mean())
        real_mean = float(real.mean())

        rows.append(
            {
                "feature": feature,
                "fake_mean": round(fake_mean, 6),
                "real_mean": round(real_mean, 6),
                "fake_median": round(float(fake.median()), 6),
                "real_median": round(float(real.median()), 6),
                "fake_min": round(float(fake.min()), 6),
                "real_min": round(float(real.min()), 6),
                "fake_max": round(float(fake.max()), 6),
                "real_max": round(float(real.max()), 6),
                "fake_std": round(float(fake.std()), 6),
                "real_std": round(float(real.std()), 6),
                "fake_minus_real_mean": round(fake_mean - real_mean, 6),
                "higher_average_label": "fake" if fake_mean > real_mean else "real",
                "mann_whitney_p": mann_whitney_p,
                "ttest_p": ttest_p,
                "cohens_d_fake_minus_real": round(cohens_d(fake, real), 6),
            }
        )

    return pd.DataFrame(rows)


def plot_word_count_distribution(features: pd.DataFrame, output_path: Path) -> None:
    """Plot text length distributions by label."""
    plt.figure(figsize=(8, 5))
    upper_limit = features["word_count"].quantile(0.98)

    for label_name, color in [("real", "#4c78a8"), ("fake", "#f58518")]:
        values = features.loc[features["label_name"] == label_name, "word_count"]
        plt.hist(values.clip(upper=upper_limit), bins=35, alpha=0.65, label=label_name, color=color)

    plt.title("Word count distribution by label")
    plt.xlabel("Word count, clipped at 98th percentile")
    plt.ylabel("Articles")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_marker_means(comparison: pd.DataFrame, output_path: Path) -> None:
    """Plot mean values for marker/count features."""
    selected = comparison[
        comparison["feature"].isin(
            [
                "sensational_count",
                "source_indicator_count",
                "uncertainty_count",
                "exclamation_count",
                "question_count",
            ]
        )
    ].copy()

    x = np.arange(len(selected))
    width = 0.38

    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, selected["real_mean"], width, label="real", color="#4c78a8")
    plt.bar(x + width / 2, selected["fake_mean"], width, label="fake", color="#f58518")
    plt.xticks(x, selected["feature"], rotation=25, ha="right")
    plt.title("Mean marker counts by label")
    plt.ylabel("Mean count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_ratio_means(comparison: pd.DataFrame, output_path: Path) -> None:
    """Plot mean ratio features."""
    selected = comparison[
        comparison["feature"].isin(
            [
                "uppercase_word_ratio",
                "sensational_ratio",
                "source_indicator_ratio",
                "uncertainty_ratio",
                "diacritic_ratio",
            ]
        )
    ].copy()

    x = np.arange(len(selected))
    width = 0.38

    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, selected["real_mean"], width, label="real", color="#4c78a8")
    plt.bar(x + width / 2, selected["fake_mean"], width, label="fake", color="#f58518")
    plt.xticks(x, selected["feature"], rotation=25, ha="right")
    plt.title("Mean ratio features by label")
    plt.ylabel("Mean ratio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_top_effect_sizes(comparison: pd.DataFrame, output_path: Path) -> None:
    """Plot the largest absolute Cohen's d values."""
    selected = comparison.copy()
    selected["abs_effect"] = selected["cohens_d_fake_minus_real"].abs()
    selected = selected.sort_values("abs_effect", ascending=False).head(10).sort_values("abs_effect")

    colors = ["#f58518" if value > 0 else "#4c78a8" for value in selected["cohens_d_fake_minus_real"]]

    plt.figure(figsize=(9, 5))
    plt.barh(selected["feature"], selected["cohens_d_fake_minus_real"], color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Top feature differences, Cohen's d")
    plt.xlabel("Fake minus real effect size")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def create_plots(features: pd.DataFrame, comparison: pd.DataFrame) -> list[str]:
    """Create Day 4 figures and return their paths."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    figure_paths = [
        FIGURES_DIR / "day4_word_count_distribution.png",
        FIGURES_DIR / "day4_marker_count_means.png",
        FIGURES_DIR / "day4_ratio_feature_means.png",
        FIGURES_DIR / "day4_top_effect_sizes.png",
    ]

    plot_word_count_distribution(features, figure_paths[0])
    plot_marker_means(comparison, figure_paths[1])
    plot_ratio_means(comparison, figure_paths[2])
    plot_top_effect_sizes(comparison, figure_paths[3])

    return [str(path) for path in figure_paths]


def train_linguistic_only_model(features: pd.DataFrame) -> dict:
    """Train a small model using only linguistic features and the Day 2 split."""
    train_ids = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig", usecols=["article_id"])
    test_ids = pd.read_csv(TEST_PATH, encoding="utf-8-sig", usecols=["article_id"])

    train_data = features.merge(train_ids, on="article_id", how="inner")
    test_data = features.merge(test_ids, on="article_id", how="inner")

    feature_columns = numeric_feature_columns(features)
    x_train = train_data[feature_columns]
    y_train = train_data["label"]
    x_test = test_data[feature_columns]
    y_test = test_data["label"]

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="binary",
        pos_label=1,
        zero_division=0,
    )

    metrics = {
        "model": "LogisticRegression with linguistic numeric features only",
        "train_rows": int(len(train_data)),
        "test_rows": int(len(test_data)),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision_fake": round(float(precision), 4),
        "recall_fake": round(float(recall), 4),
        "f1_fake": round(float(f1), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }

    joblib.dump(model, MODEL_PATH)
    MODEL_METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    return metrics


def run_analysis() -> dict:
    """Run the complete Day 4 analysis."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    features = load_features()
    quality = quality_checks(features)
    comparison = compare_features(features)
    figure_paths = create_plots(features, comparison)
    model_metrics = train_linguistic_only_model(features)

    QUALITY_PATH.write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")
    comparison.to_csv(COMPARISON_PATH, index=False, encoding="utf-8-sig")

    return {
        "quality": quality,
        "comparison": comparison,
        "figure_paths": figure_paths,
        "model_metrics": model_metrics,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    result = run_analysis()

    print("=== Day 4 linguistic feature analysis ===")
    print(f"Rows: {result['quality']['rows']}")
    print(f"Numeric missing values: {result['quality']['numeric_missing_values_total']}")
    print(f"Infinite values: {result['quality']['infinite_values_total']}")
    print(f"Articles with word_count below 20: {result['quality']['articles_with_word_count_below_20']}")
    print(f"Comparison saved: {COMPARISON_PATH}")
    print(f"Quality report saved: {QUALITY_PATH}")
    print(f"Figures: {result['figure_paths']}")
    print(f"Linguistic-only accuracy: {result['model_metrics']['accuracy']}")
    print(f"Linguistic-only F1 fake: {result['model_metrics']['f1_fake']}")


if __name__ == "__main__":
    main()
