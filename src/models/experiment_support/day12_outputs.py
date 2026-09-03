"""Plots and Markdown rendering for the historical Day 12 experiment."""

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
from src.models.experiment_support.day12_analysis import (
    CORRELATION_FIGURE_PATH,
    DOMAIN_FIGURE_PATH,
    EXPANSION_FIGURE_PATH,
    LENGTH_DISPLAY,
    LENGTH_FIGURE_PATH,
    REPORT_PATH,
    STABILITY_FIGURE_PATH,
    VARIANT_DISPLAY,
    VARIANT_ORDER,
)

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

