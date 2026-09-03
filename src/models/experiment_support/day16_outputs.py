"""Plots and Markdown report for the Day 16 calibration experiment."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.experiment_utils import (
    escaped_dataframe_to_markdown as dataframe_to_markdown,
)
from src.models.experiment_support.day16_analysis import (
    CALIBRATION_METHODS,
    INTERNAL_FIGURE_PATH,
    LENGTH_DISPLAY,
    LENGTH_FIGURE_PATH,
    LENGTH_LABELS,
    MODEL_COMPARISON_FIGURE_PATH,
    OOF_FIGURE_PATH,
    REPORT_PATH,
    THRESHOLD_FIGURE_PATH,
)


def plot_oof_calibration(
    bins: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    colors = {"sigmoid": "#3976A8", "isotonic": "#D76745"}
    for method in CALIBRATION_METHODS:
        method_bins = bins.loc[
            bins["method"].eq(method) & bins["rows"].gt(0)
        ]
        axes[0].plot(
            method_bins["mean_probability_fake"],
            method_bins["fraction_fake"],
            marker="o",
            linewidth=2,
            label=method.capitalize(),
            color=colors[method],
        )
        values = predictions.loc[
            predictions["method"].eq(method), "probability_fake"
        ]
        axes[1].hist(
            values,
            bins=np.linspace(0, 1, 21),
            alpha=0.5,
            label=method.capitalize(),
            color=colors[method],
        )
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#333333")
    axes[0].set_xlabel("Probability fake mesatare")
    axes[0].set_ylabel("Përqindja reale fake")
    axes[0].set_title("Reliability curve, nested OOF train")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.25)
    axes[1].set_xlabel("Probability fake")
    axes[1].set_ylabel("Numri i artikujve")
    axes[1].set_title("Shpërndarja e probabiliteteve")
    axes[1].legend(loc="best")
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(OOF_FIGURE_PATH, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_thresholds(comparison: pd.DataFrame, selected_name: str) -> None:
    data = comparison.copy()
    x = np.arange(len(data))
    width = 0.36
    figure, axis = plt.subplots(figsize=(9, 5.5))
    coverage = axis.bar(
        x - width / 2,
        data["strong_coverage"],
        width,
        label="Strong coverage",
        color="#3976A8",
    )
    accuracy = axis.bar(
        x + width / 2,
        data["strong_accuracy"],
        width,
        label="Strong accuracy",
        color="#2F937F",
    )
    axis.bar_label(coverage, labels=[f"{value:.2f}" for value in data["strong_coverage"]], padding=3)
    axis.bar_label(accuracy, labels=[f"{value:.2f}" for value in data["strong_accuracy"]], padding=3)
    labels = [
        f"{name}\n(selected)" if name == selected_name else name
        for name in data["threshold_name"]
    ]
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("Rezultati")
    axis.set_title("Zgjedhja e zonës uncertain nga OOF train")
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
    )
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(THRESHOLD_FIGURE_PATH, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_internal_calibration(bins: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    colors = {"new_calibrated_svm": "#D76745", "current_app_model": "#3976A8"}
    for model_name, table in bins.groupby("model", sort=False):
        non_empty = table.loc[table["rows"].gt(0)]
        axis.plot(
            non_empty["mean_probability_fake"],
            non_empty["fraction_fake"],
            marker="o",
            linewidth=2,
            label=model_name.replace("_", " ").title(),
            color=colors[model_name],
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color="#333333")
    axis.set_xlabel("Probability fake mesatare")
    axis.set_ylabel("Përqindja reale fake")
    axis.set_title("Calibration në test set-in e brendshëm")
    axis.legend(loc="best")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(INTERNAL_FIGURE_PATH, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_length_probability(length_metrics: pd.DataFrame) -> None:
    data = length_metrics.set_index("length_group").loc[LENGTH_LABELS]
    x = np.arange(len(data))
    width = 0.36
    figure, axis = plt.subplots(figsize=(11, 5.7))
    axis.bar(
        x - width / 2,
        data["mean_probability_fake_real"],
        width,
        label="Label real",
        color="#3976A8",
    )
    axis.bar(
        x + width / 2,
        data["mean_probability_fake_fake"],
        width,
        label="Label fake",
        color="#D76745",
    )
    axis.set_xticks(x, [LENGTH_DISPLAY[name] for name in LENGTH_LABELS], rotation=8)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Probability fake mesatare")
    axis.set_title("Probabiliteti sipas gjatësisë dhe label-it")
    axis.legend(loc="best")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(LENGTH_FIGURE_PATH, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_model_comparison(
    internal: pd.DataFrame,
    external: pd.DataFrame,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    models = ["current_app_model", "new_calibrated_svm"]
    colors = ["#3976A8", "#D76745"]
    x = np.arange(3)
    width = 0.34
    for index, model_name in enumerate(models):
        row = internal.loc[internal["model"].eq(model_name)].iloc[0]
        axes[0].bar(
            x + (index - 0.5) * width,
            [row["accuracy"], row["f1_weighted"], row["f1_fake"]],
            width,
            label=model_name.replace("_", " ").title(),
            color=colors[index],
        )
    axes[0].set_xticks(x, ["Accuracy", "F1 weighted", "F1 fake"])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Test-i i brendshëm")
    axes[0].grid(axis="y", alpha=0.25)

    for index, model_name in enumerate(models):
        row = external.loc[external["model"].eq(model_name)].iloc[0]
        axes[1].bar(
            x + (index - 0.5) * width,
            [row["accuracy"], row["recall_real"], row["recall_fake"]],
            width,
            label=model_name.replace("_", " ").title(),
            color=colors[index],
        )
    axes[1].set_xticks(x, ["Accuracy", "Recall real", "Recall fake"])
    axes[1].set_ylim(0, 1)
    axes[1].set_title("External, vetëm diagnostik")
    axes[1].grid(axis="y", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=2,
    )
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    figure.savefig(MODEL_COMPARISON_FIGURE_PATH, dpi=160, bbox_inches="tight")
    plt.close(figure)


def write_report(
    metrics: dict,
    method_comparison: pd.DataFrame,
    probability_summary: pd.DataFrame,
    threshold_comparison: pd.DataFrame,
    internal_comparison: pd.DataFrame,
    internal_thresholds: pd.DataFrame,
    length_metrics: pd.DataFrame,
    special_cohorts: pd.DataFrame,
    length_bias: pd.DataFrame,
    external_comparison: pd.DataFrame,
    external_thresholds: pd.DataFrame,
) -> None:
    selection = metrics["selection"]
    method = selection["calibration"]["selected_method"]
    lower = selection["thresholds"]["lower_threshold"]
    upper = selection["thresholds"]["upper_threshold"]
    selected_method_row = method_comparison.loc[
        method_comparison["method"].eq(method)
    ].iloc[0]
    new_internal = internal_comparison.loc[
        internal_comparison["model"].eq("new_calibrated_svm")
    ].iloc[0]
    current_internal = internal_comparison.loc[
        internal_comparison["model"].eq("current_app_model")
    ].iloc[0]
    new_external = external_comparison.loc[
        external_comparison["model"].eq("new_calibrated_svm")
    ].iloc[0]
    current_external = external_comparison.loc[
        external_comparison["model"].eq("current_app_model")
    ].iloc[0]
    day15_bias = metrics["length_analysis"]["day15_raw_score_bias"]
    calibrated_bias = float(length_bias.iloc[0]["mean_absolute_within_label_spearman"])

    method_table = dataframe_to_markdown(
        method_comparison,
        [
            "method",
            "brier_score",
            "log_loss",
            "ece",
            "accuracy",
            "f1_weighted",
            "f1_fake",
            "std_brier_score",
            "std_f1_weighted",
            "high_confidence_predictions",
            "high_confidence_errors",
            "mean_training_seconds",
        ],
    )
    probability_table = dataframe_to_markdown(
        probability_summary,
        ["method", "true_label", "rows", "mean", "std", "p10", "median", "p90"],
    )
    threshold_table = dataframe_to_markdown(
        threshold_comparison,
        [
            "threshold_name",
            "likely_real",
            "uncertain",
            "likely_fake",
            "strong_coverage",
            "strong_accuracy",
            "errors_in_uncertain",
            "strong_false_positives",
            "strong_false_negatives",
        ],
    )
    internal_table = dataframe_to_markdown(
        internal_comparison,
        [
            "model",
            "accuracy",
            "f1_weighted",
            "f1_fake",
            "recall_real",
            "recall_fake",
            "brier_score",
            "log_loss",
            "ece",
            "high_confidence_errors",
            "confusion_matrix",
        ],
    )
    internal_threshold_table = dataframe_to_markdown(
        internal_thresholds,
        [
            "model",
            "likely_real",
            "uncertain",
            "likely_fake",
            "strong_coverage",
            "strong_accuracy",
            "errors_in_uncertain",
            "strong_false_positives",
            "strong_false_negatives",
        ],
    )
    length_table = dataframe_to_markdown(
        length_metrics,
        [
            "length_description",
            "rows",
            "accuracy",
            "f1_weighted",
            "recall_real",
            "recall_fake",
            "brier_score",
            "ece",
            "mean_probability_fake_real",
            "mean_probability_fake_fake",
        ],
    )
    special_table = dataframe_to_markdown(
        special_cohorts,
        [
            "cohort",
            "rows",
            "accuracy",
            "recall_real",
            "recall_fake",
            "mean_probability_fake",
            "threshold_uncertain",
            "confusion_matrix",
        ],
    )
    external_table = dataframe_to_markdown(
        external_comparison,
        [
            "model",
            "accuracy",
            "recall_real",
            "recall_fake",
            "brier_score",
            "log_loss",
            "ece",
            "high_confidence_errors",
            "confusion_matrix",
        ],
    )
    external_threshold_table = dataframe_to_markdown(
        external_thresholds,
        [
            "model",
            "likely_real",
            "uncertain",
            "likely_fake",
            "strong_coverage",
            "strong_accuracy",
            "errors_in_uncertain",
            "strong_false_positives",
            "strong_false_negatives",
        ],
    )

    report = f"""# Dita 16 - Calibration dhe pragjet uncertain

## Protokolli

Konfigurimi u mbajt fiks: Word + Character TF-IDF, Linear SVM, `C=1.0`.
Sigmoid dhe isotonic u krahasuan me nested 5x5 group-safe CV vetëm mbi
{metrics['data_audit']['train_rows']} artikujt train dhe
{metrics['data_audit']['group_count']} leakage-groups. Outer folds prodhuan
probabilitete OOF për vlerësim; inner folds trajnuan calibration-in. Në asnjë
nivel nuk pati mbivendosje grupesh.

Metoda dhe pragjet u shkruan te `reports/day16_selection.json` përpara se të
ngarkoheshin test-i i brendshëm, modeli aktual i aplikacionit ose dataset-i i
jashtëm. Streamlit dhe modeli aktual nuk u zëvendësuan.

## Krahasimi i calibration-it në OOF train

{method_table}

U zgjodh **{method}**. Arsyeja e ruajtur ishte
`{selection['calibration']['method_selection_reason']}`. Brier score ishte
{selected_method_row['brier_score']:.4f}, log loss
{selected_method_row['log_loss']:.4f} dhe ECE
{selected_method_row['ece']:.4f}. Sigmoid preferohet ndaj isotonic kur është
brenda tolerancës, sepse ka formë parametrike dhe rrezik më të ulët overfitting.

![OOF calibration](figures/day16_oof_calibration_comparison.png)

### Shpërndarja e probabiliteteve

{probability_table}

## Zgjedhja e pragjeve nga OOF train

{threshold_table}

U zgjodhën pragjet **{lower:.2f}/{upper:.2f}**. Variantet brenda 0.005 strong
accuracy nga më i miri u krahasuan sipas coverage, kapjes së gabimeve dhe
gabimeve të forta. Test set-i nuk u përdor.

![Pragjet](figures/day16_threshold_comparison.png)

## Test set-i i brendshëm

Pas ngrirjes u përjashtuan
{metrics['data_audit']['exact_train_duplicates_excluded']} dublikata ekzakte
dhe mbetën {metrics['data_audit']['evaluation_test_rows']} artikuj.

{internal_table}

Modeli i ri arriti accuracy {new_internal['accuracy']:.4f}, F1 weighted
{new_internal['f1_weighted']:.4f}, F1 fake {new_internal['f1_fake']:.4f},
Brier {new_internal['brier_score']:.4f}, log loss
{new_internal['log_loss']:.4f} dhe ECE {new_internal['ece']:.4f}. Gabimet me
confidence të paktën 90% ishin {int(new_internal['high_confidence_errors'])}.

Me pragjet e ngrira:

{internal_threshold_table}

![Internal calibration](figures/day16_internal_calibration.png)

## Sjellja sipas gjatësisë

{length_table}

{special_table}

Mean absolute within-label Spearman ishte {calibrated_bias:.4f} pas calibration,
kundrejt {day15_bias:.4f} për raw decision score në Ditën 15. Calibration nuk
e zgjidhi bias-in e gjatësisë; ndryshimi interpretohet vetëm si transformim i
score-it në probability.

![Gjatësia](figures/day16_length_probability.png)

## Dataset-i i jashtëm vetëm diagnostik

{external_table}

Për modelin e ri, pragjet e ngrira dhanë:

{external_threshold_table}

Brier/log loss i jashtëm raportohet sepse rastet kanë etiketa të dokumentuara,
por kampioni ka vetëm 40 përmbledhje dhe nuk është calibration set. Asnjë
rezultat i jashtëm nuk ndryshoi metodën ose pragjet.

## Krahasimi me modelin aktual të aplikacionit

- Në test-in e brendshëm, F1 weighted ndryshoi nga
  {current_internal['f1_weighted']:.4f} në {new_internal['f1_weighted']:.4f};
  Brier nga {current_internal['brier_score']:.4f} në
  {new_internal['brier_score']:.4f}.
- Jashtë corpus-it, accuracy ndryshoi nga {current_external['accuracy']:.4f} në
  {new_external['accuracy']:.4f}; recall real/fake i modelit të ri ishte
  {new_external['recall_real']:.4f}/{new_external['recall_fake']:.4f}.
- Modeli i ri fiton përfaqësim character dhe performancë të brendshme më të
  lartë; humbet thjeshtësinë e Logistic Regression dhe mbetet i ekspozuar ndaj
  domain shift-it dhe bias-it të gjatësisë.

![Krahasimi i modeleve](figures/day16_model_comparison.png)

## Rekomandimi për Ditën 17

Rekomandohet **Word + Character TF-IDF + Linear SVM, C=1.0 + {method}** me
pragje **{lower:.2f}/{upper:.2f}** si kandidat për ngrirjen finale. Para
integrimit duhen verifikuar artefakti, funksioni i prediction, versionet e
dependencies dhe testet e regresionit të Streamlit.

## Kufizimet

- Calibration selection përdor nested CV, por ende të njëjtin corpus burimor.
- ECE varet nga 10 bin-et e zgjedhura.
- Isotonic ka mjaft raste, por mund të overfit-ojë më lehtë se sigmoid.
- Calibration nuk korrigjon domain shift, source-label confounding ose bias-in
  e gjatësisë.
- Benchmark-u i jashtëm është i vogël dhe me përmbledhje manuale.
- Modeli i ri nuk është integruar ende në Streamlit.

Modeli eksperimental ruhet te
`models/day16_word_char_linear_svm_calibrated.joblib`.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


