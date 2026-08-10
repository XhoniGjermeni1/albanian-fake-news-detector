"""Diagnose length effects and domain shift without changing the saved model."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from src.features.linguistic_features import (
    extract_linguistic_features,
    get_words,
)
from src.models.evaluate_app_system import evaluate_test_set, load_evaluation_data
from src.models.predict import (
    DEFAULT_FAKE_THRESHOLD,
    DEFAULT_REAL_THRESHOLD,
    predict_news_for_app,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_DATASET_PATH = PROJECT_ROOT / "data" / "external" / "external_news.csv"
EXPANSIONS_PATH = PROJECT_ROOT / "data" / "interim" / "day12_external_expansions.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "calibrated_tfidf_logreg.joblib"
DAY11_PREDICTIONS_PATH = PROJECT_ROOT / "reports" / "day11_external_predictions.csv"
DAY11_METRICS_PATH = PROJECT_ROOT / "reports" / "day11_external_metrics.json"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
INTERNAL_PREDICTIONS_PATH = REPORTS_DIR / "day12_internal_predictions.csv"
LENGTH_GROUPS_PATH = REPORTS_DIR / "day12_internal_length_groups.csv"
LABEL_LENGTH_PATH = REPORTS_DIR / "day12_label_length_summary.csv"
MATCHED_CASES_PATH = REPORTS_DIR / "day12_internal_30_60_cases.csv"
MATCHED_COMPARISON_PATH = REPORTS_DIR / "day12_matched_length_comparison.csv"
CORRELATIONS_PATH = REPORTS_DIR / "day12_length_correlations.csv"
STABILITY_PATH = REPORTS_DIR / "day12_internal_stability_experiment.csv"
STABILITY_SUMMARY_PATH = REPORTS_DIR / "day12_internal_stability_summary.csv"
EXTERNAL_EXPANSION_PATH = REPORTS_DIR / "day12_external_expansion_experiment.csv"
DOMAIN_SHIFT_PATH = REPORTS_DIR / "day12_domain_shift_summary.csv"
METRICS_PATH = REPORTS_DIR / "day12_metrics.json"
REPORT_PATH = REPORTS_DIR / "day12_length_domain_shift.md"

LENGTH_FIGURE_PATH = FIGURES_DIR / "day12_internal_length_performance.png"
CORRELATION_FIGURE_PATH = FIGURES_DIR / "day12_probability_vs_length.png"
STABILITY_FIGURE_PATH = FIGURES_DIR / "day12_internal_stability.png"
EXPANSION_FIGURE_PATH = FIGURES_DIR / "day12_external_expansion.png"
DOMAIN_FIGURE_PATH = FIGURES_DIR / "day12_domain_shift.png"

LENGTH_LABELS = [
    "very_short_le_60",
    "short_61_120",
    "medium_121_250",
    "long_gt_250",
]
LENGTH_DISPLAY = {
    "very_short_le_60": "Shumë të shkurtër (<=60)",
    "short_61_120": "Të shkurtër (61-120)",
    "medium_121_250": "Mesatarë (121-250)",
    "long_gt_250": "Të gjatë (>250)",
}
VARIANT_ORDER = [
    "full",
    "title_plus_first_paragraph_proxy",
    "short_46_words",
    "title_only",
]
VARIANT_DISPLAY = {
    "full": "Teksti i plotë",
    "title_plus_first_paragraph_proxy": "Titull + 120 fjalët e para",
    "short_46_words": "Versioni rreth 46 fjalë",
    "title_only": "Vetëm titulli",
}

DAY11_FROZEN_PATHS = [
    EXTERNAL_DATASET_PATH,
    MODEL_PATH,
    REPORTS_DIR / "day11_external_predictions.csv",
    REPORTS_DIR / "day11_external_metrics.json",
    REPORTS_DIR / "day11_external_by_topic.csv",
    REPORTS_DIR / "day11_external_by_label.csv",
    REPORTS_DIR / "day11_external_by_length.csv",
    REPORTS_DIR / "day11_external_by_source.csv",
    REPORTS_DIR / "day11_external_errors.csv",
    REPORTS_DIR / "day11_external_interesting_cases.csv",
    REPORTS_DIR / "day11_external_confusion_matrix.csv",
    FIGURES_DIR / "day11_external_confusion_matrix.png",
    REPORTS_DIR / "day11_external_evaluation.md",
]

LOGGER = logging.getLogger(__name__)


def file_sha256(path: Path) -> str:
    """Return the SHA-256 fingerprint of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_hashes() -> dict[str, str]:
    """Fingerprint every frozen Day 11 input and output."""
    missing = [str(path) for path in DAY11_FROZEN_PATHS if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing frozen Day 11 artifacts: {missing}")
    return {
        str(path.relative_to(PROJECT_ROOT)): file_sha256(path)
        for path in DAY11_FROZEN_PATHS
    }


def assign_length_groups(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Assign fixed, interpretable word-count groups."""
    result = dataframe.copy()
    result["length_group"] = pd.cut(
        result["word_count"],
        bins=[-np.inf, 60, 120, 250, np.inf],
        labels=LENGTH_LABELS,
        ordered=True,
    )
    return result


def add_linguistic_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add the same observable linguistic features used by the app."""
    features = pd.DataFrame(
        [
            extract_linguistic_features(row.title, row.content)
            for row in dataframe.itertuples(index=False)
        ]
    )
    result = dataframe.reset_index(drop=True).copy()
    for column in features.columns:
        result[column] = features[column]
    return result


def prepare_internal_predictions() -> tuple[pd.DataFrame, object, list[str], dict]:
    """Load the frozen model and reproduce leakage-safe internal predictions."""
    test_data, model, excluded_ids = load_evaluation_data()
    predictions, overall_metrics = evaluate_test_set(test_data, model)
    predictions = add_linguistic_features(predictions)
    predictions["label_name"] = predictions["label"].map({0: "real", 1: "fake"})
    predictions["prediction_correct"] = predictions["label"].eq(
        predictions["binary_prediction"]
    )
    predictions = assign_length_groups(predictions)
    return predictions, model, excluded_ids, overall_metrics


def summarize_group(group: pd.DataFrame, name: str) -> dict:
    """Summarize predictions for one cohort."""
    if group.empty:
        return {
            "cohort": name,
            "rows": 0,
            "real_rows": 0,
            "fake_rows": 0,
            "accuracy": None,
            "false_positives": 0,
            "false_negatives": 0,
            "mean_probability_fake": None,
            "mean_probability_fake_real": None,
            "mean_probability_fake_fake": None,
            "predicted_fake_rate": None,
            "likely_real": 0,
            "uncertain": 0,
            "likely_fake": 0,
            "mean_word_count": None,
            "median_word_count": None,
            "min_word_count": None,
            "max_word_count": None,
        }

    real = group.loc[group["label"].eq(0)]
    fake = group.loc[group["label"].eq(1)]
    return {
        "cohort": name,
        "rows": int(len(group)),
        "real_rows": int(len(real)),
        "fake_rows": int(len(fake)),
        "accuracy": round(float(group["prediction_correct"].mean()), 4),
        "false_positives": int(group["error_type"].eq("false_positive").sum()),
        "false_negatives": int(group["error_type"].eq("false_negative").sum()),
        "mean_probability_fake": round(float(group["probability_fake"].mean()), 4),
        "mean_probability_fake_real": (
            round(float(real["probability_fake"].mean()), 4) if len(real) else None
        ),
        "mean_probability_fake_fake": (
            round(float(fake["probability_fake"].mean()), 4) if len(fake) else None
        ),
        "predicted_fake_rate": round(
            float(group["binary_prediction"].eq(1).mean()), 4
        ),
        "likely_real": int(group["decision"].eq("likely_real").sum()),
        "uncertain": int(group["decision"].eq("uncertain").sum()),
        "likely_fake": int(group["decision"].eq("likely_fake").sum()),
        "mean_word_count": round(float(group["word_count"].mean()), 2),
        "median_word_count": round(float(group["word_count"].median()), 2),
        "min_word_count": int(group["word_count"].min()),
        "max_word_count": int(group["word_count"].max()),
    }


def summarize_length_groups(predictions: pd.DataFrame) -> pd.DataFrame:
    """Calculate the requested metrics for every fixed length group."""
    rows: list[dict] = []
    for length_group in LENGTH_LABELS:
        group = predictions.loc[
            predictions["length_group"].astype(str).eq(length_group)
        ]
        summary = summarize_group(group, length_group)
        summary["length_description"] = LENGTH_DISPLAY[length_group]
        rows.append(summary)
    return pd.DataFrame(rows)


def summarize_label_by_length(predictions: pd.DataFrame) -> pd.DataFrame:
    """Separate the length relationship for real and fake articles."""
    rows: list[dict] = []
    for label_number, label_name in ((0, "real"), (1, "fake")):
        for length_group in LENGTH_LABELS:
            group = predictions.loc[
                predictions["label"].eq(label_number)
                & predictions["length_group"].astype(str).eq(length_group)
            ]
            summary = summarize_group(group, f"{label_name}_{length_group}")
            rows.append(
                {
                    "label": label_name,
                    "length_group": length_group,
                    "length_description": LENGTH_DISPLAY[length_group],
                    "rows": summary["rows"],
                    "accuracy": summary["accuracy"],
                    "mean_probability_fake": summary["mean_probability_fake"],
                    "predicted_fake_rate": summary["predicted_fake_rate"],
                    "false_positives": summary["false_positives"],
                    "false_negatives": summary["false_negatives"],
                    "likely_real": summary["likely_real"],
                    "uncertain": summary["uncertain"],
                    "likely_fake": summary["likely_fake"],
                }
            )
    return pd.DataFrame(rows)


def prepare_external_predictions() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load the unchanged external data and frozen Day 11 predictions."""
    external = pd.read_csv(EXTERNAL_DATASET_PATH, encoding="utf-8", keep_default_na=False)
    day11 = pd.read_csv(DAY11_PREDICTIONS_PATH, encoding="utf-8", keep_default_na=False)
    metrics = json.loads(DAY11_METRICS_PATH.read_text(encoding="utf-8"))

    if len(external) != 40 or len(day11) != 40:
        raise ValueError("Day 12 expects the same 40 external rows used on Day 11.")
    if set(external["external_id"]) != set(day11["external_id"]):
        raise ValueError("External IDs do not match the frozen Day 11 predictions.")

    result = day11.copy()
    result["label"] = result["true_label_number"].astype(int)
    result["binary_prediction"] = result["binary_prediction_number"].astype(int)
    result["prediction_correct"] = result["label"].eq(result["binary_prediction"])
    return external, result, metrics


def build_matched_length_comparison(
    internal: pd.DataFrame,
    external_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare the external benchmark with similar-length internal cohorts."""
    internal_30_60 = internal.loc[internal["word_count"].between(30, 60)].copy()
    internal_exact = internal.loc[internal["word_count"].between(38, 51)].copy()
    rows = [
        summarize_group(internal_30_60, "internal_30_60"),
        summarize_group(internal_exact, "internal_external_range_38_51"),
        summarize_group(external_predictions, "external_day11_38_51"),
    ]
    return internal_30_60, pd.DataFrame(rows)


def calculate_correlations(predictions: pd.DataFrame) -> pd.DataFrame:
    """Measure simple length-probability associations overall and by label."""
    rows: list[dict] = []
    for scope, group in (
        ("all", predictions),
        ("real", predictions.loc[predictions["label"].eq(0)]),
        ("fake", predictions.loc[predictions["label"].eq(1)]),
    ):
        spearman = spearmanr(group["word_count"], group["probability_fake"])
        pearson = pearsonr(group["word_count"], group["probability_fake"])
        rows.append(
            {
                "scope": scope,
                "rows": int(len(group)),
                "spearman_rho": round(float(spearman.statistic), 4),
                "spearman_p_value": float(spearman.pvalue),
                "pearson_r": round(float(pearson.statistic), 4),
                "pearson_p_value": float(pearson.pvalue),
            }
        )
    return pd.DataFrame(rows)


def truncate_to_total_words(title: str, content: str, target_words: int) -> str:
    """Keep content tokens until title plus content reaches the target length."""
    if target_words <= len(get_words(title)):
        return ""

    selected_tokens: list[str] = []
    for token in str(content).split():
        selected_tokens.append(token)
        candidate = " ".join(selected_tokens)
        if len(get_words(f"{title} {candidate}")) >= target_words:
            break
    return " ".join(selected_tokens)


def select_stability_cases(predictions: pd.DataFrame) -> pd.DataFrame:
    """Select four cases per label spanning the model's probability range."""
    selections: list[pd.DataFrame] = []
    for label_number in (0, 1):
        candidates = predictions.loc[
            predictions["label"].eq(label_number)
            & predictions["word_count"].ge(180)
        ].sort_values(["probability_fake", "article_id"])
        positions = np.linspace(0, len(candidates) - 1, 4).round().astype(int)
        selected = candidates.iloc[positions].copy()
        selected["selection_position"] = [
            "low_probability_fake",
            "mid_low_probability_fake",
            "mid_high_probability_fake",
            "high_probability_fake",
        ]
        selections.append(selected)
    return pd.concat(selections, ignore_index=True)


def run_internal_stability_experiment(
    predictions: pd.DataFrame,
    model,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict full and shortened versions of selected internal articles."""
    selected = select_stability_cases(predictions)
    rows: list[dict] = []

    for article in selected.itertuples(index=False):
        variants = {
            "full": article.content,
            "title_plus_first_paragraph_proxy": truncate_to_total_words(
                article.title, article.content, 120
            ),
            "short_46_words": truncate_to_total_words(
                article.title, article.content, 46
            ),
            "title_only": "",
        }
        for variant, variant_content in variants.items():
            result = predict_news_for_app(
                article.title,
                variant_content,
                model=model,
            )
            probability_fake = float(result["probability_fake"])
            rows.append(
                {
                    "article_id": article.article_id,
                    "true_label": article.label_name,
                    "selection_position": article.selection_position,
                    "variant": variant,
                    "variant_description": VARIANT_DISPLAY[variant],
                    "title": article.title,
                    "variant_content": variant_content,
                    "word_count": int(
                        result["linguistic_explanation"]["word_count"]
                    ),
                    "probability_real": float(result["probability_real"]),
                    "probability_fake": probability_fake,
                    "binary_prediction": "fake" if probability_fake >= 0.5 else "real",
                    "decision": result["decision"],
                }
            )

    results = pd.DataFrame(rows)
    full_values = results.loc[results["variant"].eq("full")].set_index("article_id")
    results["full_probability_fake"] = results["article_id"].map(
        full_values["probability_fake"]
    )
    results["full_binary_prediction"] = results["article_id"].map(
        full_values["binary_prediction"]
    )
    results["full_decision"] = results["article_id"].map(full_values["decision"])
    results["delta_probability_fake_from_full"] = (
        results["probability_fake"] - results["full_probability_fake"]
    ).round(4)
    results["binary_changed_from_full"] = results["binary_prediction"].ne(
        results["full_binary_prediction"]
    )
    results["decision_changed_from_full"] = results["decision"].ne(
        results["full_decision"]
    )
    results["prediction_correct"] = (
        results["true_label"].eq(results["binary_prediction"])
    )
    results["variant"] = pd.Categorical(
        results["variant"], categories=VARIANT_ORDER, ordered=True
    )
    results = results.sort_values(["article_id", "variant"]).reset_index(drop=True)

    summary_rows: list[dict] = []
    for (variant, true_label), group in results.groupby(
        ["variant", "true_label"], observed=True, sort=False
    ):
        summary_rows.append(
            {
                "variant": str(variant),
                "variant_description": VARIANT_DISPLAY[str(variant)],
                "true_label": true_label,
                "rows": int(len(group)),
                "mean_word_count": round(float(group["word_count"].mean()), 2),
                "mean_probability_fake": round(
                    float(group["probability_fake"].mean()), 4
                ),
                "mean_delta_probability_fake_from_full": round(
                    float(group["delta_probability_fake_from_full"].mean()), 4
                ),
                "binary_accuracy": round(
                    float(group["prediction_correct"].mean()), 4
                ),
                "predicted_fake_rate": round(
                    float(group["binary_prediction"].eq("fake").mean()), 4
                ),
                "binary_changes_from_full": int(
                    group["binary_changed_from_full"].sum()
                ),
                "decision_changes_from_full": int(
                    group["decision_changed_from_full"].sum()
                ),
            }
        )
    return results, pd.DataFrame(summary_rows)


def validate_expansions(expansions: pd.DataFrame, external: pd.DataFrame) -> None:
    """Ensure diagnostic expansions are linked and contain no explicit verdict."""
    required = {
        "external_id",
        "expanded_content",
        "source_url",
        "selection_reason",
        "construction_notes",
    }
    missing = required - set(expansions.columns)
    if missing:
        raise ValueError(f"Expansion CSV is missing columns: {sorted(missing)}")
    if expansions["external_id"].duplicated().any():
        raise ValueError("External diagnostic IDs must be unique.")

    source_urls = external.set_index("external_id")["url"]
    for row in expansions.itertuples(index=False):
        if row.external_id not in source_urls.index:
            raise ValueError(f"Unknown external ID: {row.external_id}")
        if row.source_url != source_urls.loc[row.external_id]:
            raise ValueError(f"Source URL mismatch for {row.external_id}")

    forbidden_terms = [
        "rrenë",
        "rremë",
        "mashtrim",
        "manipuluar",
        "fact-check",
        "verifikim faktik",
        "përgënjeshtrim",
    ]
    for row in expansions.itertuples(index=False):
        lowered = row.expanded_content.casefold()
        found = [term for term in forbidden_terms if term in lowered]
        if found:
            raise ValueError(
                f"Expanded model text for {row.external_id} leaks verdict terms: {found}"
            )


def run_external_expansion_experiment(
    external: pd.DataFrame,
    day11_predictions: pd.DataFrame,
    model,
) -> pd.DataFrame:
    """Compare frozen short summaries with separate source-based expansions."""
    expansions = pd.read_csv(EXPANSIONS_PATH, encoding="utf-8", keep_default_na=False)
    validate_expansions(expansions, external)
    external_index = external.set_index("external_id")
    day11_index = day11_predictions.set_index("external_id")
    rows: list[dict] = []

    for expansion in expansions.itertuples(index=False):
        article = external_index.loc[expansion.external_id]
        frozen = day11_index.loc[expansion.external_id]
        short_result = predict_news_for_app(article.title, article.content, model=model)
        expanded_result = predict_news_for_app(
            article.title,
            expansion.expanded_content,
            model=model,
        )
        if abs(float(short_result["probability_fake"]) - float(frozen["probability_fake"])) > 0.0001:
            raise RuntimeError(
                f"Day 11 probability could not be reproduced for {expansion.external_id}"
            )

        short_fake = float(short_result["probability_fake"])
        expanded_fake = float(expanded_result["probability_fake"])
        rows.append(
            {
                "external_id": expansion.external_id,
                "true_label": article.label,
                "selection_reason": expansion.selection_reason,
                "title": article.title,
                "source_url": expansion.source_url,
                "construction_notes": expansion.construction_notes,
                "short_content": article.content,
                "expanded_content": expansion.expanded_content,
                "short_word_count": int(
                    short_result["linguistic_explanation"]["word_count"]
                ),
                "expanded_word_count": int(
                    expanded_result["linguistic_explanation"]["word_count"]
                ),
                "short_probability_fake": short_fake,
                "expanded_probability_fake": expanded_fake,
                "delta_probability_fake": round(expanded_fake - short_fake, 4),
                "short_binary_prediction": "fake" if short_fake >= 0.5 else "real",
                "expanded_binary_prediction": (
                    "fake" if expanded_fake >= 0.5 else "real"
                ),
                "short_decision": short_result["decision"],
                "expanded_decision": expanded_result["decision"],
                "binary_changed": (short_fake >= 0.5) != (expanded_fake >= 0.5),
                "decision_changed": short_result["decision"]
                != expanded_result["decision"],
            }
        )
    return pd.DataFrame(rows)


def domain_summary_rows(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> list[dict]:
    """Summarize non-label and label-specific domain characteristics."""
    rows: list[dict] = []
    for scope, group in (
        ("all", dataframe),
        ("real", dataframe.loc[dataframe["label"].eq(0)]),
        ("fake", dataframe.loc[dataframe["label"].eq(1)]),
    ):
        rows.append(
            {
                "dataset": dataset_name,
                "scope": scope,
                "rows": int(len(group)),
                "mean_word_count": round(float(group["word_count"].mean()), 4),
                "median_word_count": round(float(group["word_count"].median()), 4),
                "mean_sentence_count": round(
                    float(group["sentence_count"].mean()), 4
                ),
                "mean_avg_sentence_length": round(
                    float(group["avg_sentence_length"].mean()), 4
                ),
                "mean_diacritic_ratio": round(
                    float(group["diacritic_ratio"].mean()), 6
                ),
                "mean_uppercase_ratio": round(
                    float(group["uppercase_char_ratio"].mean()), 6
                ),
                "mean_source_marker_count": round(
                    float(group["source_indicator_count"].mean()), 4
                ),
                "source_marker_prevalence": round(
                    float(group["source_indicator_count"].gt(0).mean()), 4
                ),
                "mean_sensational_marker_count": round(
                    float(group["sensational_count"].mean()), 4
                ),
                "sensational_marker_prevalence": round(
                    float(group["sensational_count"].gt(0).mean()), 4
                ),
                "mean_exclamation_count": round(
                    float(group["exclamation_count"].mean()), 4
                ),
                "exclamation_prevalence": round(
                    float(group["exclamation_count"].gt(0).mean()), 4
                ),
            }
        )
    return rows


def build_domain_shift_summary(
    internal: pd.DataFrame,
    external: pd.DataFrame,
) -> pd.DataFrame:
    """Compare observable linguistic distributions across domains."""
    external_features = add_linguistic_features(external)
    external_features["label"] = external_features["label"].map(
        {"real": 0, "fake": 1}
    )
    rows = domain_summary_rows(internal, "internal_test")
    rows.extend(domain_summary_rows(external_features, "external_day10"))
    return pd.DataFrame(rows)


def plot_length_performance(length_summary: pd.DataFrame) -> None:
    """Plot counts and performance across internal length groups."""
    labels = [LENGTH_DISPLAY[value] for value in length_summary["cohort"]]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].bar(x, length_summary["real_rows"], label="Real", color="#2a9d8f")
    axes[0].bar(
        x,
        length_summary["fake_rows"],
        bottom=length_summary["real_rows"],
        label="Fake",
        color="#e76f51",
    )
    axes[0].set_ylabel("Numri i artikujve")
    axes[0].set_title("Test set-i i brendshëm sipas gjatësisë")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].plot(x, length_summary["accuracy"], marker="o", label="Accuracy")
    axes[1].plot(
        x,
        length_summary["mean_probability_fake"],
        marker="s",
        label="Probabiliteti mesatar fake",
    )
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Vlera")
    axes[1].set_xticks(x, labels, rotation=12, ha="right")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(LENGTH_FIGURE_PATH, dpi=180)
    plt.close(fig)


def plot_probability_vs_length(predictions: pd.DataFrame) -> None:
    """Plot fake probability against word count for each true label."""
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for label_number, label_name, color in (
        (0, "Real", "#2a9d8f"),
        (1, "Fake", "#e76f51"),
    ):
        group = predictions.loc[predictions["label"].eq(label_number)]
        ax.scatter(
            group["word_count"],
            group["probability_fake"],
            s=24,
            alpha=0.48,
            label=label_name,
            color=color,
            edgecolors="none",
        )
        log_words = np.log10(group["word_count"].clip(lower=1))
        slope, intercept = np.polyfit(log_words, group["probability_fake"], 1)
        x_values = np.logspace(log_words.min(), log_words.max(), 120)
        y_values = slope * np.log10(x_values) + intercept
        ax.plot(x_values, np.clip(y_values, 0, 1), color=color, linewidth=2)

    for boundary in (60, 120, 250):
        ax.axvline(boundary, color="#555555", linestyle=":", linewidth=1)
    ax.axhline(0.5, color="#222222", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Numri i fjalëve (shkallë logaritmike)")
    ax.set_ylabel("Probability fake")
    ax.set_title("Lidhja mes gjatësisë dhe probabilitetit fake")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(CORRELATION_FIGURE_PATH, dpi=180)
    plt.close(fig)


def plot_internal_stability(stability: pd.DataFrame) -> None:
    """Plot probability changes across shortened internal variants."""
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    x = np.arange(len(VARIANT_ORDER))
    for article_id, group in stability.groupby("article_id"):
        ordered = group.set_index("variant").loc[VARIANT_ORDER]
        color = "#2a9d8f" if ordered["true_label"].iloc[0] == "real" else "#e76f51"
        ax.plot(
            x,
            ordered["probability_fake"],
            marker="o",
            alpha=0.75,
            color=color,
            label=article_id,
        )
    for threshold, style in ((0.3, ":"), (0.5, "--"), (0.7, ":")):
        ax.axhline(threshold, color="#444444", linestyle=style, linewidth=1)
    ax.set_xticks(x, [VARIANT_DISPLAY[value] for value in VARIANT_ORDER], rotation=12)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability fake")
    ax.set_title("Stabiliteti kur hiqet përmbajtja (8 artikuj diagnostikë)")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(ncol=2, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(STABILITY_FIGURE_PATH, dpi=180)
    plt.close(fig)


def plot_external_expansion(expansion: pd.DataFrame) -> None:
    """Plot short versus expanded external probabilities."""
    fig, ax = plt.subplots(figsize=(9, 6))
    for row in expansion.itertuples(index=False):
        color = "#2a9d8f" if row.true_label == "real" else "#e76f51"
        ax.plot(
            [0, 1],
            [row.short_probability_fake, row.expanded_probability_fake],
            marker="o",
            color=color,
            linewidth=2,
            label=row.external_id,
        )
    for threshold, style in ((0.3, ":"), (0.5, "--"), (0.7, ":")):
        ax.axhline(threshold, color="#444444", linestyle=style, linewidth=1)
    ax.set_xticks([0, 1], ["Përmbledhja e Ditës 11", "Versioni i zgjeruar"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability fake")
    ax.set_title("Eksperimenti diagnostik me pesë raste të jashtme")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(EXPANSION_FIGURE_PATH, dpi=180)
    plt.close(fig)


def plot_domain_shift(domain: pd.DataFrame) -> None:
    """Plot selected internal/external distribution differences."""
    overall = domain.loc[domain["scope"].eq("all")].set_index("dataset")
    datasets = ["internal_test", "external_day10"]
    colors = ["#457b9d", "#f4a261"]
    labels = ["Internal", "External"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].bar(labels, overall.loc[datasets, "median_word_count"], color=colors)
    axes[0, 0].set_title("Mediana e fjalëve")
    axes[0, 0].set_ylabel("Fjalë")

    ratio_values = np.array(
        [
            overall.loc[datasets, "mean_diacritic_ratio"].to_numpy(),
            overall.loc[datasets, "mean_uppercase_ratio"].to_numpy(),
        ]
    ).T
    x = np.arange(2)
    axes[0, 1].bar(x - 0.18, ratio_values[0] * 100, 0.36, label="Internal")
    axes[0, 1].bar(x + 0.18, ratio_values[1] * 100, 0.36, label="External")
    axes[0, 1].set_xticks(x, ["Diakritika", "Uppercase"])
    axes[0, 1].set_title("Raportet mesatare")
    axes[0, 1].set_ylabel("Përqindje")
    axes[0, 1].legend()

    marker_values = np.array(
        [
            overall.loc[datasets, "source_marker_prevalence"].to_numpy(),
            overall.loc[datasets, "sensational_marker_prevalence"].to_numpy(),
        ]
    ).T
    axes[1, 0].bar(x - 0.18, marker_values[0] * 100, 0.36, label="Internal")
    axes[1, 0].bar(x + 0.18, marker_values[1] * 100, 0.36, label="External")
    axes[1, 0].set_xticks(x, ["Source markers", "Sensational markers"])
    axes[1, 0].set_title("Prevalenca e marker-ave")
    axes[1, 0].set_ylabel("Artikuj me të paktën një marker (%)")
    axes[1, 0].legend()

    sentence_values = overall.loc[datasets, "mean_avg_sentence_length"]
    axes[1, 1].bar(labels, sentence_values, color=colors)
    axes[1, 1].set_title("Gjatësia mesatare e fjalisë")
    axes[1, 1].set_ylabel("Fjalë")

    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Domain shift: test set-i i brendshëm kundrejt datasetit të jashtëm")
    fig.tight_layout()
    fig.savefig(DOMAIN_FIGURE_PATH, dpi=180)
    plt.close(fig)


def _markdown_value(value: object) -> str:
    if pd.isna(value):
        return "n/a"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value).replace("|", "/").replace("\n", " ")


def dataframe_to_markdown(dataframe: pd.DataFrame, columns: list[str]) -> str:
    """Render a small DataFrame without an optional dependency."""
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(_markdown_value(row[column]) for column in columns) + " |"
        for _, row in dataframe[columns].iterrows()
    ]
    return "\n".join([header, separator, *rows])


def _percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def write_report(
    metrics: dict,
    length_summary: pd.DataFrame,
    label_length: pd.DataFrame,
    matched: pd.DataFrame,
    correlations: pd.DataFrame,
    stability: pd.DataFrame,
    stability_summary: pd.DataFrame,
    expansion: pd.DataFrame,
    domain: pd.DataFrame,
) -> None:
    """Write the Day 12 interpretation in Albanian."""
    length_table = dataframe_to_markdown(
        length_summary,
        [
            "length_description",
            "rows",
            "real_rows",
            "fake_rows",
            "accuracy",
            "false_positives",
            "false_negatives",
            "mean_probability_fake",
            "likely_real",
            "uncertain",
            "likely_fake",
        ],
    )
    matched_table = dataframe_to_markdown(
        matched,
        [
            "cohort",
            "rows",
            "real_rows",
            "fake_rows",
            "accuracy",
            "false_positives",
            "false_negatives",
            "mean_probability_fake_real",
            "mean_probability_fake_fake",
            "predicted_fake_rate",
        ],
    )
    label_table = dataframe_to_markdown(
        label_length,
        [
            "label",
            "length_description",
            "rows",
            "accuracy",
            "mean_probability_fake",
            "predicted_fake_rate",
            "likely_real",
            "uncertain",
            "likely_fake",
        ],
    )
    correlation_report = correlations.copy()
    correlation_report["spearman_p_value"] = correlation_report[
        "spearman_p_value"
    ].map(lambda value: f"{value:.2e}")
    correlation_table = dataframe_to_markdown(
        correlation_report,
        ["scope", "rows", "spearman_rho", "spearman_p_value", "pearson_r"],
    )
    stability_table = dataframe_to_markdown(
        stability_summary,
        [
            "variant_description",
            "true_label",
            "rows",
            "mean_word_count",
            "mean_probability_fake",
            "mean_delta_probability_fake_from_full",
            "binary_accuracy",
            "binary_changes_from_full",
            "decision_changes_from_full",
        ],
    )
    expansion_table = dataframe_to_markdown(
        expansion,
        [
            "external_id",
            "true_label",
            "short_word_count",
            "expanded_word_count",
            "short_probability_fake",
            "expanded_probability_fake",
            "delta_probability_fake",
            "short_decision",
            "expanded_decision",
        ],
    )
    domain_overall = domain.loc[domain["scope"].eq("all")]
    domain_table = dataframe_to_markdown(
        domain_overall,
        [
            "dataset",
            "rows",
            "mean_word_count",
            "median_word_count",
            "mean_avg_sentence_length",
            "mean_diacritic_ratio",
            "mean_uppercase_ratio",
            "source_marker_prevalence",
            "sensational_marker_prevalence",
        ],
    )

    very_short = length_summary.set_index("cohort").loc["very_short_le_60"]
    long_group = length_summary.set_index("cohort").loc["long_gt_250"]
    by_label = label_length.set_index(["label", "length_group"])
    real_very_short = by_label.loc[("real", "very_short_le_60")]
    real_long = by_label.loc[("real", "long_gt_250")]
    fake_very_short = by_label.loc[("fake", "very_short_le_60")]
    fake_long = by_label.loc[("fake", "long_gt_250")]
    matched_index = matched.set_index("cohort")
    internal_matched = matched_index.loc["internal_30_60"]
    external_matched = matched_index.loc["external_day11_38_51"]
    correlation_index = correlations.set_index("scope")

    short_variants = stability.loc[stability["variant"].astype(str).eq("short_46_words")]
    increased_after_shortening = int(
        short_variants["delta_probability_fake_from_full"].gt(0).sum()
    )
    real_expansion = expansion.loc[expansion["true_label"].eq("real")]
    fake_expansion = expansion.loc[expansion["true_label"].eq("fake")]
    expansion_decreases = int(expansion["delta_probability_fake"].lt(0).sum())
    expansion_decision_changes = int(expansion["decision_changed"].sum())
    context = metrics["domain_context"]

    report = f"""# Dita 12 - Analiza e gjatësisë dhe domain shift-it

## Integriteti i eksperimentit

Analiza përdori modelin ekzistues `calibrated_tfidf_logreg.joblib`, të njëjtin
preprocessing dhe pragjet e pandryshuara 0.30/0.70. Nuk u thirr `fit`, nuk u
ruajt model i ri dhe `data/external/external_news.csv` nuk u ndryshua. Hash-et e
modelit, datasetit të jashtëm dhe të gjitha output-eve të Ditës 11 u kontrolluan
para dhe pas analizës: **{'Po' if metrics['frozen_integrity']['all_unchanged'] else 'Jo'}**.

Test set-i i brendshëm përmban {metrics['internal_overall']['rows']} artikuj pas
përjashtimit të {metrics['excluded_train_duplicates']} dublikatave ekzakte me
train set-in.

## Performanca e brendshme sipas gjatësisë

Intervalet janë fikse dhe të interpretueshme; ato nuk u zgjodhën për të
optimizuar metrikat.

{length_table}

Vetëm {int(very_short['rows'])} artikuj të brendshëm kishin deri në 60 fjalë,
ndërsa grupi 61-120 dominohej nga fake. Probabiliteti mesatar fake ra nga
{_percent(float(very_short['mean_probability_fake']))} në grupin shumë të
shkurtër në {_percent(float(long_group['mean_probability_fake']))} te artikujt
mbi 250 fjalë. Kjo përzierje e label-it me gjatësinë është një sinjal i fortë
se modeli ka mësuar edhe dallime të corpus-it, jo vetëm dallime të përgjithshme
mes lajmeve real dhe fake.

![Performanca sipas gjatësisë](figures/day12_internal_length_performance.png)

## Krahasimi me gjatësi të përafërt

{matched_table}

Cohort-i i brendshëm 30-60 fjalë ka vetëm {int(internal_matched['rows'])} raste,
prandaj nuk jep një vlerësim të qëndrueshëm. Megjithatë, ai arriti
{_percent(float(internal_matched['accuracy']))} accuracy dhe gaboi
{int(internal_matched['false_positives'])} nga
{int(internal_matched['real_rows'])} rastet real. Dataset-i i jashtëm gaboi
{int(external_matched['false_positives'])} nga
{int(external_matched['real_rows'])} rastet real. Pra shkurtësia rrit prirjen
drejt fake, por **nuk e riprodhon e vetme dështimin 19/20** të jashtëm.

Intervali ekzakt 38-51 fjalë ka vetëm
{int(matched_index.loc['internal_external_range_38_51', 'rows'])} raste të
brendshme, ndaj përdoret vetëm si kontroll përshkrues.

## Ndikimi veçmas për real dhe fake

{label_table}

- Për real, probability fake mesatare ishte
  {_percent(float(real_very_short['mean_probability_fake']))} deri në 60 fjalë
  dhe {_percent(float(real_long['mean_probability_fake']))} mbi 250 fjalë.
- Për fake, probability fake mesatare ishte
  {_percent(float(fake_very_short['mean_probability_fake']))} deri në 60 fjalë,
  por vetëm {_percent(float(fake_long['mean_probability_fake']))} mbi 250 fjalë.
- Accuracy për fake të gjatë ishte {_percent(float(fake_long['accuracy']))};
  kjo tregon problemin simetrik: fake të gjatë shtyhen drejt real.

{correlation_table}

Spearman rho ishte {correlation_index.loc['all', 'spearman_rho']:.4f} në total,
{correlation_index.loc['real', 'spearman_rho']:.4f} vetëm te real dhe
{correlation_index.loc['fake', 'spearman_rho']:.4f} vetëm te fake. Lidhja
negative mbetet brenda secilës klasë, ndaj nuk shpjegohet vetëm nga përzierja e
label-eve. Kjo është lidhje statistikore dhe jo provë e vetme shkakësie.

![Probability fake kundrejt gjatësisë](figures/day12_probability_vs_length.png)

## Eksperimenti i stabilitetit të brendshëm

U përzgjodhën katër artikuj për secilën klasë, me të paktën 180 fjalë dhe në
pozicione të ndryshme të shpërndarjes së probability fake. Ky është kampion
diagnostik, jo metrikë e re testimi. Corpus-i i përpunuar nuk ruan kufij
paragrafësh; prandaj `title_plus_first_paragraph_proxy` është operacionalizuar
si titulli plus deri në 120 fjalët e para.

{stability_table}

Në {increased_after_shortening} nga {len(short_variants)} rastet, versioni rreth
46 fjalë mori probability fake më të lartë se teksti i plotë. Ndryshimet binare
dhe të vendimit raportohen për çdo variant në CSV. Heqja e përmbajtjes ndryshon
edhe fjalorin dhe peshat TF-IDF, prandaj eksperimenti tregon ndjeshmëri ndaj
shkurtimit, jo një efekt të izoluar mekanik të numrit të fjalëve.

![Eksperimenti i stabilitetit](figures/day12_internal_stability.png)

## Eksperimenti diagnostik me raste të jashtme

Pesë raste problematike të Ditës 11 u zgjeruan vetëm me informacion nga URL-ja
e tyre burimore: tri real me gabim të fortë dhe dy fake të humbura. Tekstet u
ruajtën veçmas te `data/interim/day12_external_expansions.csv`. Verdikti,
përgënjeshtrimi dhe provat e fact-check-ut nuk iu dhanë modelit. Ky eksperiment
nuk ndryshon benchmark-un dhe nuk llogaritet si rezultat i ri i jashtëm.

{expansion_table}

Për tri rastet real, ndryshimi mesatar i probability fake ishte
{real_expansion['delta_probability_fake'].mean():+.4f}; për dy rastet fake ishte
{fake_expansion['delta_probability_fake'].mean():+.4f}. Në
{expansion_decreases} nga {len(expansion)} rastet, zgjerimi e uli probability
fake dhe ndryshoi {expansion_decision_changes} vendime. Kjo ndihmoi dy raste
real të kalonin nga `likely_fake` në `uncertain`, por e shtyu edhe
`EXT-F-012` nga `uncertain` në `likely_real`. Pra drejtimi lidhet me zgjerimin,
jo me saktësinë e label-it. Me vetëm pesë raste dhe me zgjerime të kuruara
manualisht, rezultati përdoret si kontroll stabiliteti, jo si provë
përfundimtare.

![Zgjerimi i rasteve të jashtme](figures/day12_external_expansion.png)

## Domain shift përtej gjatësisë

{domain_table}

Ndryshimet e dokumentuara janë:

- **Periudha:** corpus-i i brendshëm mbulon
  {context['internal_period']['minimum']} deri
  {context['internal_period']['maximum']}; rastet e jashtme
  {context['external_period']['minimum']} deri
  {context['external_period']['maximum']}.
- **Stili:** të brendshmet janë artikuj corpus-i, ndërsa të jashtmet janë
  përmbledhje manuale uniforme. Kjo ndryshon fjalorin, strukturën dhe dendësinë
  e informacionit.
- **Temat:** dataset-i i jashtëm ka pesë tema të balancuara me dorë
  ({', '.join(context['external_topics'])}). Corpus-i i brendshëm nuk ka label
  teme, prandaj diferenca tematike nuk mund të matet drejt.
- **Burimet:** në datasetin e jashtëm, real vijnë nga burime institucionale dhe
  fake nga pretendime sociale të dokumentuara nga fact-check. Burimi nuk i
  jepet modelit, por kjo ndërthurje pengon ndarjen e efektit të stilit nga label-i.
- **Forma gjuhësore:** diakritikat, uppercase ratio, source markers dhe
  sensational markers kanë shpërndarje të ndryshme në tabelë. Këto janë
  kandidatë për domain shift, jo prova shkakësie.

![Përmbledhja e domain shift-it](figures/day12_domain_shift.png)

## Përfundimi

Bias-i i lidhur me gjatësinë është **i fortë**. Ai shfaqet në grupet e
brendshme, në të dyja klasat, në korrelacionet negative dhe në eksperimentin e
shkurtimit. Modeli TF-IDF nuk merr `word_count` si kolonë numerike; sinjali vjen
në mënyrë indirekte nga fjalori, sasia e kontekstit dhe shpërndarja e gjatësisë
në corpus.

Gjatësia shpjegon një pjesë të rëndësishme, por **jo pjesën e plotë të dështimit
të jashtëm**. Lajmet real të brendshme me 30-60 fjalë nuk u sollën aq keq sa 20
lajmet real të jashtme. Periudha e re, përmbledhja manuale, temat, burimet dhe
ndryshimet në marker-at gjuhësorë tregojnë domain shift shtesë.

Modeli aktual mund të ruhet si baseline dhe si pjesë e analizës së diplomës,
por jo të konsiderohet detektor i besueshëm për përmbledhje të shkurtra jashtë
corpus-it. Rezultatet e dobëta nuk duhen fshehur; ato janë një gjetje e vlefshme
mbi kufijtë e përgjithësimit.

## Rekomandimi për Ditën 13

Të krahasohen në të njëjtën ndarje pa leakage:

1. Word TF-IDF;
2. Character TF-IDF;
3. Word + Character TF-IDF.

Krahasimi duhet të ruajë modelin dhe benchmark-un aktual, të raportojë veçmas
test set-in e brendshëm, cohort-in 30-60 fjalë dhe datasetin e jashtëm. Dataset-i
i jashtëm nuk duhet përdorur për tuning ose zgjedhje pragjesh.

## Output-et

```text
reports/day12_internal_predictions.csv
reports/day12_internal_length_groups.csv
reports/day12_label_length_summary.csv
reports/day12_internal_30_60_cases.csv
reports/day12_matched_length_comparison.csv
reports/day12_length_correlations.csv
reports/day12_internal_stability_experiment.csv
reports/day12_internal_stability_summary.csv
reports/day12_external_expansion_experiment.csv
reports/day12_domain_shift_summary.csv
reports/day12_metrics.json
reports/day12_length_domain_shift.md
reports/figures/day12_*.png
```
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def run_day12_analysis() -> dict:
    """Run all Day 12 diagnostics without training or changing frozen files."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    hashes_before = frozen_hashes()

    internal, model, excluded_ids, internal_overall = prepare_internal_predictions()
    external, external_predictions, day11_metrics = prepare_external_predictions()
    length_summary = summarize_length_groups(internal)
    label_length = summarize_label_by_length(internal)
    matched_cases, matched_comparison = build_matched_length_comparison(
        internal, external_predictions
    )
    correlations = calculate_correlations(internal)
    stability, stability_summary = run_internal_stability_experiment(internal, model)
    expansion = run_external_expansion_experiment(
        external, external_predictions, model
    )
    domain = build_domain_shift_summary(internal, external)

    internal_columns = [
        "article_id",
        "pair_id",
        "label",
        "label_name",
        "title",
        "word_count",
        "sentence_count",
        "avg_sentence_length",
        "diacritic_ratio",
        "uppercase_char_ratio",
        "source_indicator_count",
        "sensational_count",
        "exclamation_count",
        "probability_real",
        "probability_fake",
        "binary_prediction",
        "decision",
        "prediction_correct",
        "error_type",
        "length_group",
    ]
    internal[internal_columns].to_csv(
        INTERNAL_PREDICTIONS_PATH, index=False, encoding="utf-8"
    )
    length_summary.to_csv(LENGTH_GROUPS_PATH, index=False, encoding="utf-8")
    label_length.to_csv(LABEL_LENGTH_PATH, index=False, encoding="utf-8")
    matched_cases[internal_columns].to_csv(
        MATCHED_CASES_PATH, index=False, encoding="utf-8"
    )
    matched_comparison.to_csv(
        MATCHED_COMPARISON_PATH, index=False, encoding="utf-8"
    )
    correlations.to_csv(CORRELATIONS_PATH, index=False, encoding="utf-8")
    stability.to_csv(STABILITY_PATH, index=False, encoding="utf-8")
    stability_summary.to_csv(
        STABILITY_SUMMARY_PATH, index=False, encoding="utf-8"
    )
    expansion.to_csv(EXTERNAL_EXPANSION_PATH, index=False, encoding="utf-8")
    domain.to_csv(DOMAIN_SHIFT_PATH, index=False, encoding="utf-8")

    plot_length_performance(length_summary)
    plot_probability_vs_length(internal)
    plot_internal_stability(stability)
    plot_external_expansion(expansion)
    plot_domain_shift(domain)

    hashes_after = frozen_hashes()
    all_unchanged = hashes_before == hashes_after
    if not all_unchanged:
        changed = [
            path
            for path in hashes_before
            if hashes_before[path] != hashes_after.get(path)
        ]
        raise RuntimeError(f"Frozen Day 11 artifacts changed: {changed}")

    internal_context = day11_metrics["internal_comparison"]["internal"]
    external_context = day11_metrics["internal_comparison"]["external"]
    metrics = {
        "status": "completed",
        "model_retrained": False,
        "thresholds_changed": False,
        "external_dataset_changed": False,
        "prediction_function": "predict_news_for_app / vectorized equivalent",
        "thresholds": {
            "likely_real_below": DEFAULT_REAL_THRESHOLD,
            "likely_fake_above": DEFAULT_FAKE_THRESHOLD,
        },
        "frozen_integrity": {
            "all_unchanged": all_unchanged,
            "hashes_before": hashes_before,
            "hashes_after": hashes_after,
        },
        "excluded_train_duplicates": len(excluded_ids),
        "internal_overall": internal_overall,
        "length_groups": length_summary.to_dict(orient="records"),
        "matched_length_comparison": matched_comparison.to_dict(orient="records"),
        "length_correlations": correlations.to_dict(orient="records"),
        "stability_summary": stability_summary.to_dict(orient="records"),
        "external_expansion_cases": expansion[
            [
                "external_id",
                "true_label",
                "short_word_count",
                "expanded_word_count",
                "short_probability_fake",
                "expanded_probability_fake",
                "delta_probability_fake",
                "binary_changed",
                "decision_changed",
            ]
        ].to_dict(orient="records"),
        "domain_context": {
            "internal_period": internal_context["publication_dates"],
            "external_period": external_context["publication_dates"],
            "internal_text_style": "full corpus articles",
            "external_text_style": "manual summaries",
            "internal_topics_available": False,
            "external_topics": sorted(external["topic"].unique().tolist()),
            "external_source_label_confounding": True,
        },
        "artifacts": {
            "report": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
            "internal_predictions": str(
                INTERNAL_PREDICTIONS_PATH.relative_to(PROJECT_ROOT)
            ),
            "length_groups": str(LENGTH_GROUPS_PATH.relative_to(PROJECT_ROOT)),
            "matched_comparison": str(
                MATCHED_COMPARISON_PATH.relative_to(PROJECT_ROOT)
            ),
            "correlations": str(CORRELATIONS_PATH.relative_to(PROJECT_ROOT)),
            "stability": str(STABILITY_PATH.relative_to(PROJECT_ROOT)),
            "external_expansion": str(
                EXTERNAL_EXPANSION_PATH.relative_to(PROJECT_ROOT)
            ),
            "domain_shift": str(DOMAIN_SHIFT_PATH.relative_to(PROJECT_ROOT)),
        },
    }
    METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(
        metrics,
        length_summary,
        label_length,
        matched_comparison,
        correlations,
        stability,
        stability_summary,
        expansion,
        domain,
    )
    return metrics


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    metrics = run_day12_analysis()
    LOGGER.info("Internal rows: %s", metrics["internal_overall"]["rows"])
    LOGGER.info(
        "Spearman rho (all/real/fake): %s / %s / %s",
        *[
            row["spearman_rho"]
            for row in metrics["length_correlations"]
        ],
    )
    LOGGER.info(
        "Frozen Day 11 artifacts unchanged: %s",
        metrics["frozen_integrity"]["all_unchanged"],
    )
    LOGGER.info("Report saved to: %s", REPORT_PATH)


if __name__ == "__main__":
    main()
