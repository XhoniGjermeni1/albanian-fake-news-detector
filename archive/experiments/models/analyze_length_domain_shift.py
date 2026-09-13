# Drejton analizën e gjatësisë dhe domain shift-it duke përdorur modelin dhe dataset-et
# e ngrira. Mat sjelljen sipas grupeve të gjatësisë, stabilitetin pas shkurtimit/zgjerimit
# dhe dallimet gjuhësore mes të dhënave të brendshme dhe benchmark-ut të jashtëm.

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from archive.experiments.models.experiment_support.day12_analysis import (
    DEFAULT_FAKE_THRESHOLD,
    DEFAULT_REAL_THRESHOLD,
    LENGTH_LABELS,
    LOGGER,
    METRICS_PATH,
    PROJECT_ROOT,
    assign_length_groups,
    build_domain_shift_summary,
    build_matched_length_comparison,
    calculate_correlations,
    frozen_hashes,
    prepare_external_predictions,
    prepare_internal_predictions,
    run_external_expansion_experiment,
    run_internal_stability_experiment,
    summarize_group,
    summarize_length_groups,
    truncate_to_total_words,
    validate_expansions,
)


def run_length_domain_shift_analysis() -> dict:
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    hashes_before = frozen_hashes()

    internal, model, excluded_ids, internal_overall = prepare_internal_predictions()
    external, external_predictions, domain_periods = prepare_external_predictions()

    length_summary = summarize_length_groups(internal)
    _, matched_comparison = build_matched_length_comparison(
        internal,
        external_predictions,
    )
    correlations = calculate_correlations(internal)
    _, stability_summary = run_internal_stability_experiment(internal, model)
    expansion = run_external_expansion_experiment(
        external,
        external_predictions,
        model,
    )
    domain_summary = build_domain_shift_summary(internal, external)

    hashes_after = frozen_hashes()
    if hashes_before != hashes_after:
        changed = [
            path
            for path in hashes_before
            if hashes_before[path] != hashes_after.get(path)
        ]
        raise RuntimeError(f"Frozen inputs changed during analysis: {changed}")

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
            "all_unchanged": True,
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
        "domain_shift_summary": domain_summary.to_dict(orient="records"),
        "domain_context": {
            "internal_period": domain_periods["internal"],
            "external_period": domain_periods["external"],
            "internal_text_style": "full corpus articles",
            "external_text_style": "manual summaries",
            "internal_topics_available": False,
            "external_topics": sorted(external["topic"].unique().tolist()),
            "external_source_label_confounding": True,
        },
        "artifacts": {
            "metrics": str(METRICS_PATH.relative_to(PROJECT_ROOT)),
        },
    }
    METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics


# Emri publik historik ruhet për thirrjet dhe notebook-et ekzistuese.
run_day12_analysis = run_length_domain_shift_analysis


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    metrics = run_length_domain_shift_analysis()
    LOGGER.info("Internal rows: %s", metrics["internal_overall"]["rows"])
    LOGGER.info(
        "Spearman rho (all/real/fake): %s / %s / %s",
        *[row["spearman_rho"] for row in metrics["length_correlations"]],
    )
    LOGGER.info(
        "Frozen inputs unchanged: %s",
        metrics["frozen_integrity"]["all_unchanged"],
    )
    LOGGER.info("Metrics saved to: %s", METRICS_PATH)


if __name__ == "__main__":
    main()
