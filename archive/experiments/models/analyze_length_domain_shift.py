"""Stable entry point for the historical Day 12 length/domain-shift analysis."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.experiment_support.day12_analysis import (
    CORRELATION_FIGURE_PATH,
    CORRELATIONS_PATH,
    DAY11_FROZEN_PATHS,
    DAY11_METRICS_PATH,
    DAY11_PREDICTIONS_PATH,
    DEFAULT_FAKE_THRESHOLD,
    DEFAULT_REAL_THRESHOLD,
    DOMAIN_FIGURE_PATH,
    DOMAIN_SHIFT_PATH,
    EXPANSION_FIGURE_PATH,
    EXPANSIONS_PATH,
    EXTERNAL_DATASET_PATH,
    EXTERNAL_EXPANSION_PATH,
    FIGURES_DIR,
    INTERNAL_PREDICTIONS_PATH,
    LABEL_LENGTH_PATH,
    LENGTH_DISPLAY,
    LENGTH_FIGURE_PATH,
    LENGTH_GROUPS_PATH,
    LENGTH_LABELS,
    LOGGER,
    MATCHED_CASES_PATH,
    MATCHED_COMPARISON_PATH,
    METRICS_PATH,
    MODEL_PATH,
    PROJECT_ROOT,
    REPORT_PATH,
    REPORTS_DIR,
    STABILITY_FIGURE_PATH,
    STABILITY_PATH,
    STABILITY_SUMMARY_PATH,
    VARIANT_DISPLAY,
    VARIANT_ORDER,
    add_linguistic_features,
    assign_length_groups,
    build_domain_shift_summary,
    build_matched_length_comparison,
    calculate_correlations,
    domain_summary_rows,
    evaluate_test_set,
    extract_linguistic_features,
    file_sha256,
    frozen_hashes,
    get_words,
    load_evaluation_data,
    predict_news_for_app,
    prepare_external_predictions,
    prepare_internal_predictions,
    run_external_expansion_experiment,
    run_internal_stability_experiment,
    select_stability_cases,
    summarize_group,
    summarize_label_by_length,
    summarize_length_groups,
    truncate_to_total_words,
    validate_expansions,
)
from src.models.experiment_support.day12_outputs import (
    dataframe_to_markdown,
    plot_domain_shift,
    plot_external_expansion,
    plot_internal_stability,
    plot_length_performance,
    plot_probability_vs_length,
    write_report,
)

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
