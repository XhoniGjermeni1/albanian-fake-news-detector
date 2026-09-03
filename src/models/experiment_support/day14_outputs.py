"""Figures and Markdown rendering for the Day 14 classifier comparison."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.data_utils import LENGTH_DISPLAY, LENGTH_LABELS
from src.evaluation.experiment_utils import (
    escaped_dataframe_to_markdown as dataframe_to_markdown,
)
from src.models.experiment_support.day14_analysis import (
    CLASSIFIER_DISPLAY,
    COLORS,
    CV_FIGURE_PATH,
    EXTERNAL_FIGURE_PATH,
    INTERNAL_FIGURE_PATH,
    LENGTH_FIGURE_PATH,
    REPORT_PATH,
    best_cv_rows,
)


def plot_cv(cv_summary: pd.DataFrame, selection: dict) -> None:
    """Plot CV F1 with fold-level standard deviations."""
    data = best_cv_rows(cv_summary, selection)
    names = data["classifier"].tolist()
    values = data["mean_f1_weighted"].to_numpy()
    errors = data["std_f1_weighted"].to_numpy()
    figure, axis = plt.subplots(figsize=(9, 5.4))
    bars = axis.bar(
        [CLASSIFIER_DISPLAY[name] for name in names],
        values,
        yerr=errors,
        capsize=6,
        color=[COLORS[name] for name in names],
    )
    axis.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=4)
    axis.set_ylim(max(0, float(values.min()) - 0.08), min(1.0, float(values.max()) + 0.06))
    axis.set_ylabel("F1 weighted mesatare")
    axis.set_title("Group-safe cross-validation vetëm mbi train")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(CV_FIGURE_PATH, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_internal(comparison: pd.DataFrame) -> None:
    """Plot key internal test metrics."""
    metrics = ["accuracy", "f1_weighted", "f1_fake"]
    labels = ["Accuracy", "F1 weighted", "F1 fake"]
    x = np.arange(len(metrics))
    width = 0.24
    figure, axis = plt.subplots(figsize=(10, 5.6))
    for index, classifier in enumerate(CLASSIFIER_DISPLAY):
        row = comparison.loc[comparison["classifier"].eq(classifier)].iloc[0]
        axis.bar(
            x + (index - 1) * width,
            [row[metric] for metric in metrics],
            width,
            label=CLASSIFIER_DISPLAY[classifier],
            color=COLORS[classifier],
        )
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Rezultati")
    axis.set_title("Vlerësimi në test set-in e brendshëm")
    axis.legend(loc="lower right")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(INTERNAL_FIGURE_PATH, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_length_groups(length_metrics: pd.DataFrame) -> None:
    """Plot accuracy by the fixed Day 12 length groups."""
    x = np.arange(len(LENGTH_LABELS))
    figure, axis = plt.subplots(figsize=(11, 5.8))
    for classifier in CLASSIFIER_DISPLAY:
        table = length_metrics.loc[
            length_metrics["classifier"].eq(classifier)
        ].set_index("length_group")
        values = [float(table.loc[group, "accuracy"]) for group in LENGTH_LABELS]
        axis.plot(
            x,
            values,
            marker="o",
            linewidth=2,
            label=CLASSIFIER_DISPLAY[classifier],
            color=COLORS[classifier],
        )
    axis.set_xticks(x, [LENGTH_DISPLAY[group] for group in LENGTH_LABELS], rotation=8)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Accuracy")
    axis.set_title("Performanca sipas gjatësisë")
    axis.legend(loc="best")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(LENGTH_FIGURE_PATH, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_external(comparison: pd.DataFrame) -> None:
    """Plot the frozen external diagnostic results."""
    metrics = ["accuracy", "recall_real", "recall_fake"]
    labels = ["Accuracy", "Recall real", "Recall fake"]
    x = np.arange(len(metrics))
    width = 0.24
    figure, axis = plt.subplots(figsize=(10, 5.6))
    for index, classifier in enumerate(CLASSIFIER_DISPLAY):
        row = comparison.loc[comparison["classifier"].eq(classifier)].iloc[0]
        axis.bar(
            x + (index - 1) * width,
            [row[metric] for metric in metrics],
            width,
            label=CLASSIFIER_DISPLAY[classifier],
            color=COLORS[classifier],
        )
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Rezultati")
    axis.set_title("Dataset-i i jashtëm, vetëm diagnostik")
    axis.legend(loc="lower right")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(EXTERNAL_FIGURE_PATH, dpi=160, bbox_inches="tight")
    plt.close(figure)


def write_report(
    metrics: dict,
    cv_summary: pd.DataFrame,
    internal_comparison: pd.DataFrame,
    length_metrics: pd.DataFrame,
    special_cohorts: pd.DataFrame,
    length_bias: pd.DataFrame,
    external_comparison: pd.DataFrame,
) -> None:
    """Write the reproducible Day 14 report."""
    selection = metrics["selection"]
    winner = selection["winner_classifier"]
    stable = selection["most_stable_classifier"]
    balanced = selection["best_recall_balance_classifier"]
    cv_best = best_cv_rows(cv_summary, selection)

    cv_table = dataframe_to_markdown(
        cv_best,
        [
            "classifier_display",
            "parameter_name",
            "parameter_value",
            "mean_accuracy",
            "mean_precision_weighted",
            "mean_recall_weighted",
            "mean_f1_weighted",
            "std_f1_weighted",
            "mean_f1_fake",
            "mean_recall_real",
            "mean_recall_fake",
            "mean_training_seconds",
        ],
    )
    internal_table = dataframe_to_markdown(
        internal_comparison,
        [
            "classifier_display",
            "accuracy",
            "f1_weighted",
            "f1_fake",
            "recall_real",
            "recall_fake",
            "false_positives",
            "false_negatives",
            "confusion_matrix",
        ],
    )
    special_table = dataframe_to_markdown(
        special_cohorts,
        [
            "classifier_display",
            "cohort",
            "rows",
            "accuracy",
            "recall_real",
            "recall_fake",
            "confusion_matrix",
        ],
    )
    length_table = dataframe_to_markdown(
        length_metrics,
        [
            "classifier_display",
            "length_description",
            "rows",
            "accuracy",
            "f1_weighted",
            "recall_real",
            "recall_fake",
        ],
    )
    bias_table = dataframe_to_markdown(
        length_bias,
        [
            "classifier_display",
            "spearman_all",
            "spearman_real",
            "spearman_fake",
            "mean_absolute_within_label_spearman",
            "real_30_60_accuracy",
            "fake_gt_250_accuracy",
        ],
    )
    external_table = dataframe_to_markdown(
        external_comparison,
        [
            "classifier_display",
            "accuracy",
            "recall_real",
            "recall_fake",
            "false_positives",
            "false_negatives",
            "confusion_matrix",
        ],
    )

    logistic_bias = float(
        length_bias.loc[
            length_bias["classifier"].eq("logistic_regression"),
            "mean_absolute_within_label_spearman",
        ].iloc[0]
    )
    lowest_bias_row = length_bias.sort_values(
        "mean_absolute_within_label_spearman"
    ).iloc[0]
    svm_bias_row = length_bias.loc[
        length_bias["classifier"].eq("linear_svm")
    ].iloc[0]
    logistic_long_fake_accuracy = float(
        length_bias.loc[
            length_bias["classifier"].eq("logistic_regression"),
            "fake_gt_250_accuracy",
        ].iloc[0]
    )
    winner_internal = internal_comparison.loc[
        internal_comparison["classifier"].eq(winner)
    ].iloc[0]
    winner_external = external_comparison.loc[
        external_comparison["classifier"].eq(winner)
    ].iloc[0]
    external_best = external_comparison.sort_values(
        ["accuracy", "f1_weighted"], ascending=[False, False]
    ).iloc[0]

    report = f"""# Dita 14 - Krahasimi i classifier-ëve

## Protokolli

Përfaqësimi u mbajt fiks si në Ditën 13: Word TF-IDF `(1, 2)`, maksimumi
30,000 features, plus Character TF-IDF `char_wb (3, 5)`, maksimumi 50,000
features. U përdor i njëjti preprocessing bazë dhe asnjë classifier nuk u
kalibrua.

Përzgjedhja u bë vetëm mbi {metrics['data_audit']['train_rows']} artikujt train
me 5-fold `StratifiedGroupKFold`. `pair_id` i njëjtë dhe tekstet identike u
mbajtën në të njëjtin leakage-group; u gjetën
{metrics['data_audit']['group_count']} grupe dhe zero mbivendosje mes fit dhe
validation. Për shkak të normalizimit aktual NFC, u rindërtuan vetëm në memorie
{metrics['data_audit']['stale_train_model_text_rows_refreshed_in_memory']} vlera
`model_text`; CSV-të nuk u ndryshuan.

Zgjedhja u shkrua te `reports/day14_selection.json` përpara se të ngarkohej
test set-i i brendshëm. Dataset-i i jashtëm u hap vetëm pas kësaj. Test-i,
benchmark-u i jashtëm dhe modeli aktual i Streamlit nuk u përdorën për tuning.

## Konfigurimet e provuara

U provuan dy vlera të arsyeshme për secilën familje: Logistic Regression dhe
Linear SVM me `C=0.5/1.0`, si dhe Complement Naive Bayes me `alpha=0.5/1.0`.
TF-IDF u përshtat një herë për çdo fold dhe u nda mes kandidatëve; koha e
trajnimit në tabelë përfshin atë kosto të përbashkët plus fit-in e classifier-it.

## Cross-validation vetëm mbi train

{cv_table}

Rregulli kryesor ishte F1 weighted mesatare. Kandidatët brenda 0.002 nga vlera
më e mirë u renditën sipas devijimit standard më të ulët, pastaj F1 fake dhe
përshtatshmërisë për calibration/deploy. Fituesi i ngrirë ishte
**{CLASSIFIER_DISPLAY[winner]}** (`{selection['winner_candidate_id']}`), me
F1 weighted {selection['winner_mean_f1_weighted']:.4f} ±
{selection['winner_std_f1_weighted']:.4f}.

Classifier-i më i qëndrueshëm sipas devijimit standard ishte
**{CLASSIFIER_DISPLAY[stable]}**, ndërsa hendekun më të vogël mes recall real
dhe fake e pati **{CLASSIFIER_DISPLAY[balanced]}**. Këto përfundime përdorin
vetëm train/CV.

![Cross-validation](figures/day14_cv_classifier_comparison.png)

## Test set-i i brendshëm

Pas ngrirjes së përzgjedhjes u përjashtuan
{metrics['data_audit']['exact_train_duplicates_excluded']} dublikatat ekzakte
train/test dhe mbetën {metrics['data_audit']['evaluation_test_rows']} artikuj.

{internal_table}

Kandidati i zgjedhur arriti accuracy {winner_internal['accuracy']:.4f}, F1
weighted {winner_internal['f1_weighted']:.4f} dhe F1 fake
{winner_internal['f1_fake']:.4f}. Ky rezultat nuk u përdor për të ndryshuar
zgjedhjen.

![Test-i i brendshëm](figures/day14_internal_classifier_comparison.png)

## Grupet e gjatësisë

{length_table}

Dy cohort-et e kërkuara:

{special_table}

![Performanca sipas gjatësisë](figures/day14_length_performance.png)

## Bias-i i gjatësisë

Për çdo classifier u përdor score-i i tij i pakalibruar, i orientuar drejt
klasës fake. Score-t e classifier-ëve kanë shkallë të ndryshme dhe nuk janë
probabilitete; krahasimi bazohet te Spearman rank correlation.

{bias_table}

Logistic Regression kishte mesataren absolute within-label
{logistic_bias:.4f}. Vlerën më të ulët e pati
**{lowest_bias_row['classifier_display']}** me
{lowest_bias_row['mean_absolute_within_label_spearman']:.4f}. Një vlerë pak më
e ulët nuk do të thotë se bias-i u eliminua; cohort-et e skajeve mbeten të
vogla dhe të vështira.

Linear SVM gjithashtu e uli këtë lidhje nga {logistic_bias:.4f} te
{svm_bias_row['mean_absolute_within_label_spearman']:.4f}. Te 29 lajmet fake
mbi 250 fjalë, accuracy u rrit nga {logistic_long_fake_accuracy:.4f} te
{svm_bias_row['fake_gt_250_accuracy']:.4f}; pra kandidati e zbuti bias-in, por
nuk e eliminoi.

## Dataset-i i jashtëm vetëm diagnostik

{external_table}

Kandidati i ngrirë **{CLASSIFIER_DISPLAY[winner]}** arriti accuracy
{winner_external['accuracy']:.4f}, recall real
{winner_external['recall_real']:.4f} dhe recall fake
{winner_external['recall_fake']:.4f}. Rezultatet e jashtme nuk ndryshuan
classifier-in ose parametrat.

Accuracy më të lartë diagnostike e pati
**{external_best['classifier_display']}** me {external_best['accuracy']:.4f}.
Kjo përmbysje e renditjes kundrejt CV/test-it të brendshëm është provë për
domain shift dhe jo arsye për tuning pas testimit.

![Diagnostika e jashtme](figures/day14_external_diagnostic.png)

## Përfundimi

- Fituesi i cross-validation ishte **{CLASSIFIER_DISPLAY[winner]}**.
- Më i qëndrueshmi mes fold-eve ishte **{CLASSIFIER_DISPLAY[stable]}**.
- Balancën më të afërt recall real/fake e dha
  **{CLASSIFIER_DISPLAY[balanced]}**.
- Lidhjen më të ulët me gjatësinë e pati
  **{lowest_bias_row['classifier_display']}**, por bias-i vazhdon.
- Linear SVM e uli dukshëm bias-in kundrejt Logistic Regression dhe dha
  rezultatin më të mirë te fake të gjatë mes tre kandidatëve.
- Në benchmark-un e jashtëm pati përmbysje renditjeje; kjo nuk ndryshon
  zgjedhjen train/CV.
- Për Ditën 15 rekomandohet **{CLASSIFIER_DISPLAY[winner]}** me përfaqësimin e
  ngrirë Word + Character TF-IDF për tuning të kufizuar dhe calibration të
  kontrolluar.

## Kufizimet

- U provuan vetëm gjashtë konfigurime të paracaktuara.
- CV-ja dhe test-i i brendshëm vijnë nga i njëjti corpus.
- Score-t e Linear SVM dhe modeleve të tjera nuk janë probabilitete dhe nuk u
  aplikuan pragjet 0.30/0.70.
- Cohort-i real 30-60 dhe fake mbi 250 fjalë kanë pak raste.
- Dataset-i i jashtëm ka përmbledhje të shkurtra dhe source-label confounding;
  ai mbetet vetëm benchmark diagnostik.
- Calibration, tuning-u final dhe integrimi në Streamlit janë lënë për ditët e
  ardhshme.

## Modelet eksperimentale

```text
models/day14_word_char_logistic_regression.joblib
models/day14_word_char_linear_svm.joblib
models/day14_word_char_complement_nb.joblib
```

Modeli `models/calibrated_tfidf_logreg.joblib` dhe aplikacioni Streamlit nuk u
zëvendësuan.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")

