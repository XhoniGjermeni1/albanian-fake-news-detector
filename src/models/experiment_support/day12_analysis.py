"""Analysis helpers and frozen paths for the historical Day 12 experiment."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from src.evaluation.data_utils import (
    LENGTH_DISPLAY,
    LENGTH_LABELS,
    assign_length_groups,
)
from src.evaluation.experiment_utils import file_sha256
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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
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

LOGGER = logging.getLogger("src.models.analyze_length_domain_shift")


def frozen_hashes() -> dict[str, str]:
    """Fingerprint every frozen Day 11 input and output."""
    missing = [str(path) for path in DAY11_FROZEN_PATHS if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing frozen Day 11 artifacts: {missing}")
    return {
        str(path.relative_to(PROJECT_ROOT)): file_sha256(path)
        for path in DAY11_FROZEN_PATHS
    }


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

