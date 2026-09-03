"""Plots and report rendering for the historical Day 13 experiment."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.experiment_utils import (
    dataframe_to_markdown,
    format_percent as _percent,
)
from src.models.experiment_support.day13_analysis import (
    COHORT_FIGURE_PATH,
    EXTERNAL_FIGURE_PATH,
    INTERNAL_FIGURE_PATH,
    LENGTH_BIAS_FIGURE_PATH,
    LENGTH_LABELS,
    MODEL_COLORS,
    MODEL_DISPLAY,
    MODEL_NAMES,
    REPORT_PATH,
    STABILITY_DISPLAY,
    STABILITY_FIGURE_PATH,
)


def plot_internal_comparison(comparison: pd.DataFrame) -> None:
    """Plot the main internal metrics."""
    metrics = ["accuracy", "f1_weighted", "f1_fake"]
    x = np.arange(len(metrics))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 6))
    for index, model_name in enumerate(MODEL_NAMES):
        row = comparison.set_index("model").loc[model_name]
        ax.bar(
            x + (index - 1) * width,
            [row[metric] for metric in metrics],
            width,
            label=MODEL_DISPLAY[model_name],
            color=MODEL_COLORS[model_name],
        )
    ax.set_xticks(x, ["Accuracy", "F1 weighted", "F1 fake"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rezultati")
    ax.set_title("Krahasimi i përfaqësimeve në test set-in e brendshëm")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(INTERNAL_FIGURE_PATH, dpi=180)
    plt.close(fig)


def plot_cohort_accuracy(cohort_metrics: pd.DataFrame) -> None:
    """Plot accuracy for the requested diagnostic cohorts."""
    cohorts = [
        "all_internal_test",
        "internal_30_60",
        "short_real_30_60",
        "long_fake_gt_250",
    ]
    labels = ["Të gjithë", "30-60 fjalë", "Real 30-60", "Fake >250"]
    x = np.arange(len(cohorts))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10.5, 6))
    indexed = cohort_metrics.set_index(["model", "cohort"])
    for index, model_name in enumerate(MODEL_NAMES):
        values = [indexed.loc[(model_name, cohort), "accuracy"] for cohort in cohorts]
        ax.bar(
            x + (index - 1) * width,
            values,
            width,
            label=MODEL_DISPLAY[model_name],
            color=MODEL_COLORS[model_name],
        )
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Performanca në cohort-et e gjatësisë")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(COHORT_FIGURE_PATH, dpi=180)
    plt.close(fig)


def plot_length_bias(length_metrics: pd.DataFrame) -> None:
    """Plot mean fake probability by length and true label."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    for axis, label_number, label_name in zip(axes, (0, 1), ("Real", "Fake")):
        for model_name in MODEL_NAMES:
            rows = []
            for length_group in LENGTH_LABELS:
                cohort = f"length_{length_group}"
                metric_row = length_metrics.loc[
                    length_metrics["model"].eq(model_name)
                    & length_metrics["cohort"].eq(cohort)
                ].iloc[0]
                probability_column = (
                    "mean_probability_fake_real"
                    if label_number == 0
                    else "mean_probability_fake_fake"
                )
                rows.append(float(metric_row[probability_column]))
            axis.plot(
                np.arange(len(LENGTH_LABELS)),
                rows,
                marker="o",
                label=MODEL_DISPLAY[model_name],
                color=MODEL_COLORS[model_name],
            )
        axis.set_xticks(
            np.arange(len(LENGTH_LABELS)),
            ["<=60", "61-120", "121-250", ">250"],
        )
        axis.set_title(f"Label real: {label_name}")
        axis.set_xlabel("Numri i fjalëve")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Probability fake mesatare")
    axes[0].set_ylim(0, 1)
    axes[1].legend()
    fig.suptitle("Bias-i i gjatësisë sipas përfaqësimit")
    fig.tight_layout()
    fig.savefig(LENGTH_BIAS_FIGURE_PATH, dpi=180)
    plt.close(fig)


def plot_stability(stability_summary: pd.DataFrame) -> None:
    """Plot mean probability changes for each diagnostic variant."""
    variants = [
        "short_46_words",
        "title_only",
        "without_albanian_diacritics",
        "unicode_nfc_from_nfd",
    ]
    x = np.arange(len(variants))
    width = 0.24
    indexed = stability_summary.set_index(["model", "variant"])
    fig, ax = plt.subplots(figsize=(11, 6))
    for index, model_name in enumerate(MODEL_NAMES):
        values = [
            indexed.loc[(model_name, variant), "mean_absolute_delta_from_full"]
            for variant in variants
        ]
        ax.bar(
            x + (index - 1) * width,
            values,
            width,
            label=MODEL_DISPLAY[model_name],
            color=MODEL_COLORS[model_name],
        )
    ax.set_xticks(x, [STABILITY_DISPLAY[variant] for variant in variants])
    ax.set_ylabel("Ndryshimi absolut mesatar i probability fake")
    ax.set_title("Stabiliteti ndaj varianteve të tekstit")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(STABILITY_FIGURE_PATH, dpi=180)
    plt.close(fig)


def plot_external_comparison(comparison: pd.DataFrame) -> None:
    """Plot external diagnostic metrics and decision counts."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    x = np.arange(3)
    width = 0.24
    indexed = comparison.set_index("model")
    for index, model_name in enumerate(MODEL_NAMES):
        row = indexed.loc[model_name]
        axes[0].bar(
            x + (index - 1) * width,
            [row["accuracy"], row["recall_real"], row["recall_fake"]],
            width,
            label=MODEL_DISPLAY[model_name],
            color=MODEL_COLORS[model_name],
        )
    axes[0].set_xticks(x, ["Accuracy", "Recall real", "Recall fake"])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Metrikat e jashtme")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.2)

    bottom = np.zeros(len(MODEL_NAMES))
    colors = ["#2a9d8f", "#f4a261", "#e76f51"]
    for decision, color in zip(
        ("likely_real", "uncertain", "likely_fake"), colors
    ):
        values = comparison.set_index("model").loc[MODEL_NAMES, decision].to_numpy()
        axes[1].bar(
            [MODEL_DISPLAY[name] for name in MODEL_NAMES],
            values,
            bottom=bottom,
            label=decision,
            color=color,
        )
        bottom += values
    axes[1].set_title("Vendimet me pragjet 0.30/0.70")
    axes[1].set_ylabel("Numri i rasteve")
    axes[1].tick_params(axis="x", rotation=12)
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(EXTERNAL_FIGURE_PATH, dpi=180)
    plt.close(fig)


def write_report(
    metrics: dict,
    char_screen: pd.DataFrame,
    internal_comparison: pd.DataFrame,
    cohort_metrics: pd.DataFrame,
    length_bias: pd.DataFrame,
    stability_summary: pd.DataFrame,
    external_comparison: pd.DataFrame,
) -> None:
    """Write the Day 13 report in Albanian."""
    screen_table = dataframe_to_markdown(
        char_screen,
        [
            "config_name",
            "ngram_min",
            "ngram_max",
            "max_features",
            "mean_accuracy",
            "mean_f1_weighted",
            "mean_f1_fake",
            "training_seconds",
        ],
    )
    internal_table = dataframe_to_markdown(
        internal_comparison,
        [
            "model_display",
            "accuracy",
            "precision_weighted",
            "recall_weighted",
            "f1_weighted",
            "f1_fake",
            "false_positives",
            "false_negatives",
            "brier_score",
            "log_loss",
            "training_seconds",
            "model_size_mb",
        ],
    )
    requested_cohorts = cohort_metrics.loc[
        cohort_metrics["cohort"].isin(
            [
                "internal_30_60",
                "short_real_30_60",
                "long_fake_gt_250",
            ]
        )
    ]
    cohort_table = dataframe_to_markdown(
        requested_cohorts,
        [
            "model_display",
            "cohort",
            "rows",
            "accuracy",
            "f1_weighted",
            "recall_real",
            "recall_fake",
            "mean_probability_fake_real",
            "mean_probability_fake_fake",
        ],
    )
    bias_table = dataframe_to_markdown(
        length_bias,
        [
            "model_display",
            "spearman_all",
            "spearman_real",
            "spearman_fake",
            "real_probability_gap_short_minus_long",
            "fake_probability_gap_short_minus_long",
            "short_real_accuracy",
            "long_fake_accuracy",
        ],
    )
    stability_table = dataframe_to_markdown(
        stability_summary.loc[stability_summary["variant"].ne("full")],
        [
            "model_display",
            "variant_description",
            "mean_absolute_delta_from_full",
            "max_absolute_delta_from_full",
            "binary_changes_from_full",
            "decision_changes_from_full",
        ],
    )
    external_table = dataframe_to_markdown(
        external_comparison,
        [
            "model_display",
            "accuracy",
            "recall_real",
            "recall_fake",
            "false_positives",
            "false_negatives",
            "likely_real",
            "uncertain",
            "likely_fake",
            "mean_probability_fake_real",
            "mean_probability_fake_fake",
        ],
    )

    selection = metrics["internal_selection"]
    best_internal = selection["best_internal_model"]
    best_short = selection["best_short_cohort_model"]
    lowest_bias = selection["lowest_length_association_model"]
    internal_index = internal_comparison.set_index("model")
    bias_index = length_bias.set_index("model")
    stability_index = stability_summary.set_index(["model", "variant"])
    external_index = external_comparison.set_index("model")
    word_external = external_index.loc["word_tfidf"]
    char_external = external_index.loc["char_tfidf"]
    combined_external = external_index.loc["word_char_tfidf"]
    char_external_difference = (
        float(char_external["accuracy"]) - float(word_external["accuracy"])
    )
    combined_external_difference = (
        float(combined_external["accuracy"]) - float(word_external["accuracy"])
    )

    report = f"""# Dita 13 - Krahasimi i përfaqësimeve TF-IDF

## Protokolli

U përdorën të njëjtat `train.csv`/`test.csv`, preprocessing bazë dhe përjashtimi
i 7 dublikatave ekzakte train/test. Të tre variantet përdorën të njëjtën
`LogisticRegression(max_iter=1000, class_weight='balanced')` dhe sigmoid
calibration me 5 fold-e group-safe. Pragjet mbetën 0.30/0.70.

Dataset-i i jashtëm nuk u lexua gjatë character screen, trajnimit, vlerësimit
të brendshëm ose përzgjedhjes. Përzgjedhja u ruajt fillimisht te
`reports/day13_internal_selection.json`; vetëm pas kësaj u hap benchmark-u i
jashtëm. Modeli aktual i aplikacionit dhe dataset-i i jashtëm mbetën të
pandryshuar.

## Character screen vetëm mbi train

U provuan vetëm dy konfigurime të paracaktuara, jo një grid search i madh.

{screen_table}

U zgjodh `{metrics['selected_char_config']['config_name']}` sipas F1 weighted
mesatare në 3-fold group-safe CV. I njëjti konfigurim u përdor te Character dhe
Word + Character.

## Rezultatet e brendshme

{internal_table}

Confusion matrices në rendin `[real, fake]`:

- Word: `{internal_index.loc['word_tfidf', 'confusion_matrix']}`;
- Character: `{internal_index.loc['char_tfidf', 'confusion_matrix']}`;
- Word + Character: `{internal_index.loc['word_char_tfidf', 'confusion_matrix']}`.

Sipas rregullit të ngrirë, varianti më i mirë i brendshëm ishte
**{MODEL_DISPLAY[best_internal]}** me F1 weighted
{_percent(float(internal_index.loc[best_internal, 'f1_weighted']))}. Përzgjedhja
nuk varet nga rezultatet e jashtme.

![Krahasimi i brendshëm](figures/day13_internal_model_comparison.png)

## Cohort-et problematike

{cohort_table}

Cohort-i 30-60 ka vetëm 9 raste, prej të cilave 6 real. Fake mbi 250 fjalë ka
29 raste. Këto rezultate janë diagnostike dhe duhen interpretuar bashkë me
madhësinë e kampionit. Varianti me renditjen më të mirë për cohort-in e shkurtër
ishte **{MODEL_DISPLAY[best_short]}**.

![Accuracy sipas cohort-it](figures/day13_cohort_accuracy.png)

## Bias-i i gjatësisë

{bias_table}

Vlera më afër zeros për lidhjen mes gjatësisë dhe probability fake u arrit nga
**{MODEL_DISPLAY[lowest_bias]}**. Për krahasim, hendeku real short-minus-long
ishte {bias_index.loc['word_tfidf', 'real_probability_gap_short_minus_long']:.4f}
te Word, {bias_index.loc['char_tfidf', 'real_probability_gap_short_minus_long']:.4f}
te Character dhe
{bias_index.loc['word_char_tfidf', 'real_probability_gap_short_minus_long']:.4f}
te kombinimi. Character n-grams
{'e ulën' if abs(bias_index.loc['char_tfidf', 'spearman_real']) < abs(bias_index.loc['word_tfidf', 'spearman_real']) else 'nuk e ulën'}
lidhjen e gjatësisë te lajmet real krahasuar me Word TF-IDF.

![Bias-i i gjatësisë](figures/day13_length_bias.png)

## Stabiliteti i tekstit

U ripërdorën të njëjtat 8 raste të brendshme të Ditës 12. Varianti Unicode u
krijua fillimisht në NFD dhe kaloi në të njëjtin NFC preprocessing; prandaj
duhet të japë rezultat identik me tekstin e plotë.

{stability_table}

Për versionin 46 fjalë, ndryshimi absolut mesatar ishte
{stability_index.loc[('word_tfidf', 'short_46_words'), 'mean_absolute_delta_from_full']:.4f}
te Word,
{stability_index.loc[('char_tfidf', 'short_46_words'), 'mean_absolute_delta_from_full']:.4f}
te Character dhe
{stability_index.loc[('word_char_tfidf', 'short_46_words'), 'mean_absolute_delta_from_full']:.4f}
te kombinimi. Për Unicode të normalizuar, ndryshimi maksimal ishte
{stability_summary.loc[stability_summary['variant'].eq('unicode_nfc_from_nfd'), 'max_absolute_delta_from_full'].max():.8f}.

Character arriti accuracy më të mirë në 9 tekstet natyrshëm 30-60 fjalë, por
ndryshimi i tij mesatar pas shkurtimit artificial ishte më i madh se te Word.
Pra character n-grams nuk dhanë stabilitet uniform. Heqja e `ë/ç` shkaktoi
ndryshime të mëdha te të tre modelet, ndërsa normalizimi Unicode ishte plotësisht
stabil.

![Stabiliteti](figures/day13_stability.png)

## Vlerësimi i jashtëm vetëm diagnostik

Këto rezultate u llogaritën pasi përzgjedhja e brendshme ishte shkruar dhe
hash-i i saj ishte ngrirë. Ato nuk ndryshuan konfigurimet ose rekomandimin.

{external_table}

Confusion matrices:

- Word: `{external_index.loc['word_tfidf', 'confusion_matrix']}`;
- Character: `{external_index.loc['char_tfidf', 'confusion_matrix']}`;
- Word + Character: `{external_index.loc['word_char_tfidf', 'confusion_matrix']}`.

Character ndryshoi accuracy e jashtme me
{100 * char_external_difference:+.2f} pikë përqindjeje dhe kombinimi me
{100 * combined_external_difference:+.2f} pikë përqindjeje kundrejt Word.
Character-only ishte më i miri në këtë benchmark, ndërsa kombinimi dha vetëm
përmirësim të pjesshëm të generalizimit. Ky është vetëm vëzhgim diagnostik dhe
nuk përdoret për model selection.

![Vlerësimi i jashtëm](figures/day13_external_diagnostic.png)

## Përfundimi

- Varianti më i mirë në vlerësimin e brendshëm ishte
  **{MODEL_DISPLAY[best_internal]}**.
- Varianti më i mirë në cohort-in 30-60 ishte **{MODEL_DISPLAY[best_short]}**.
- Lidhjen më të ulët me gjatësinë e pati **{MODEL_DISPLAY[lowest_bias]}**.
- Character n-grams
  {'e përmirësuan' if float(internal_index.loc['char_tfidf', 'f1_weighted']) > float(internal_index.loc['word_tfidf', 'f1_weighted']) else 'nuk e përmirësuan'}
  F1 weighted kundrejt Word baseline.
- Character uli bias-in te real të shkurtër, por jo te fake të gjatë dhe nuk
  ishte më stabil ndaj çdo transformimi diagnostik.
- Word + Character
  {'e përmirësoi' if float(internal_index.loc['word_char_tfidf', 'f1_weighted']) > float(internal_index.loc['word_tfidf', 'f1_weighted']) else 'nuk e përmirësoi'}
  rezultatin e brendshëm kundrejt Word baseline.
- Jashtë corpus-it, Character-only përgjithësoi më mirë se kombinimi; ky rezultat
  nuk ndryshon rekomandimin e ngrirë nga vlerësimi i brendshëm.

Për Ditën 14 rekomandohet **{MODEL_DISPLAY[selection['recommended_for_day14']]}**
si përfaqësim i ngrirë për krahasimin e classifier-ëve. Modelet e tjera ruhen si
eksperimente dhe asnjëri nuk integrohet ende në Streamlit.

## Kufizimet

- Character screen kishte vetëm dy konfigurime dhe u bë vetëm mbi train.
- Cohort-i 30-60 ka 9 raste dhe nuk jep interval të ngushtë besimi.
- Stabiliteti përdor 8 raste të zgjedhura diagnostike, jo gjithë test set-in.
- Të tre modelet mësojnë nga i njëjti corpus ku gjatësia dhe label-i janë të
  lidhura; përfaqësimi i ri nuk e heq automatikisht këtë bias.
- Dataset-i i jashtëm ka përmbledhje manuale dhe source-label confounding;
  rezultatet e tij nuk janë tuning set.
- Përzgjedhja e classifier-it në Ditën 14 duhet bërë me CV mbi train, duke
  ruajtur test set-in dhe benchmark-un e jashtëm për vlerësim.

## Modelet eksperimentale

```text
models/day13_word_tfidf_logreg_calibrated.joblib
models/day13_char_tfidf_logreg_calibrated.joblib
models/day13_word_char_tfidf_logreg_calibrated.joblib
```

Modeli i aplikacionit `models/calibrated_tfidf_logreg.joblib` nuk u zëvendësua.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")

