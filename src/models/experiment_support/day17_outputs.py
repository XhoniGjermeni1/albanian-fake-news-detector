"""Plots and Markdown report for the Day 17 finalization run."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.evaluation.experiment_utils import (
    escaped_dataframe_to_markdown as dataframe_to_markdown,
)
from src.models.experiment_support.day17_analysis import (
    BASELINE_MODEL_NAME,
    FINAL_MODEL_ID,
    FINAL_MODEL_NAME,
    FINAL_MODEL_VERSION,
    LENGTH_DISPLAY,
    LENGTH_FIGURE_PATH,
    LENGTH_LABELS,
    MODEL_COMPARISON_FIGURE_PATH,
    REPORT_PATH,
)


def plot_model_comparison(internal: pd.DataFrame, external: pd.DataFrame) -> None:
    display = {
        BASELINE_MODEL_NAME: "Word LR baseline",
        FINAL_MODEL_NAME: "Word + Char SVM final",
    }
    colors = {BASELINE_MODEL_NAME: "#3976A8", FINAL_MODEL_NAME: "#D76745"}
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    width = 0.35

    internal_metrics = ["accuracy", "f1_weighted", "f1_fake"]
    x_internal = np.arange(len(internal_metrics))
    for index, model_name in enumerate([BASELINE_MODEL_NAME, FINAL_MODEL_NAME]):
        row = internal.loc[internal["model"].eq(model_name)].iloc[0]
        axes[0].bar(
            x_internal + (index - 0.5) * width,
            [row[name] for name in internal_metrics],
            width,
            label=display[model_name],
            color=colors[model_name],
        )
    axes[0].set_xticks(x_internal, ["Accuracy", "F1 weighted", "F1 fake"])
    axes[0].set_title("Test set-i i brendshëm")
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis="y", alpha=0.25)

    external_metrics = ["accuracy", "recall_real", "recall_fake"]
    x_external = np.arange(len(external_metrics))
    for index, model_name in enumerate([BASELINE_MODEL_NAME, FINAL_MODEL_NAME]):
        row = external.loc[external["model"].eq(model_name)].iloc[0]
        axes[1].bar(
            x_external + (index - 0.5) * width,
            [row[name] for name in external_metrics],
            width,
            label=display[model_name],
            color=colors[model_name],
        )
    axes[1].set_xticks(x_external, ["Accuracy", "Recall real", "Recall fake"])
    axes[1].set_title("Benchmark-u i jashtëm pilot")
    axes[1].set_ylim(0, 1)
    axes[1].grid(axis="y", alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2)
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    figure.savefig(MODEL_COMPARISON_FIGURE_PATH, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_length_results(length_metrics: pd.DataFrame) -> None:
    data = length_metrics.set_index("length_group").loc[LENGTH_LABELS]
    x = np.arange(len(data))
    width = 0.25
    figure, axis = plt.subplots(figsize=(11, 5.8))
    axis.bar(x - width, data["accuracy"], width, label="Accuracy", color="#3976A8")
    axis.bar(x, data["recall_real"], width, label="Recall real", color="#2F937F")
    axis.bar(
        x + width,
        data["recall_fake"],
        width,
        label="Recall fake",
        color="#D76745",
    )
    axis.set_xticks(
        x,
        [LENGTH_DISPLAY[name] for name in LENGTH_LABELS],
        rotation=8,
    )
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("Rezultati")
    axis.set_title("Modeli final sipas gjatësisë")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(LENGTH_FIGURE_PATH, dpi=160, bbox_inches="tight")
    plt.close(figure)


def write_report(
    metrics: dict,
    internal: pd.DataFrame,
    external: pd.DataFrame,
    length_metrics: pd.DataFrame,
    special_cohorts: pd.DataFrame,
    demos: pd.DataFrame,
    regression: pd.DataFrame,
    high_confidence_errors: pd.DataFrame,
) -> None:
    final_internal = internal.loc[internal["model"].eq(FINAL_MODEL_NAME)].iloc[0]
    final_external = external.loc[external["model"].eq(FINAL_MODEL_NAME)].iloc[0]
    internal_table = dataframe_to_markdown(
        internal,
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
            "threshold_strong_coverage",
            "threshold_strong_accuracy",
            "confusion_matrix",
        ],
    )
    external_table = dataframe_to_markdown(
        external,
        [
            "model",
            "accuracy",
            "recall_real",
            "recall_fake",
            "brier_score",
            "log_loss",
            "high_confidence_errors",
            "threshold_likely_real",
            "threshold_uncertain",
            "threshold_likely_fake",
            "threshold_strong_coverage",
            "threshold_strong_accuracy",
            "confusion_matrix",
        ],
    )
    length_table = dataframe_to_markdown(
        length_metrics,
        [
            "length_description",
            "rows",
            "real_rows",
            "fake_rows",
            "accuracy",
            "recall_real",
            "recall_fake",
            "mean_probability_fake_real",
            "mean_probability_fake_fake",
            "confusion_matrix",
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
    demo_table = dataframe_to_markdown(
        demos,
        [
            "demo_type",
            "article_id",
            "title",
            "true_label",
            "binary_prediction",
            "decision",
            "probability_real",
            "probability_fake",
            "explanation",
        ],
    )
    regression_table = dataframe_to_markdown(
        regression,
        [
            "case_id",
            "probability_real",
            "probability_fake",
            "decision",
            "expected_decision",
            "matches_expected_anchor",
            "probability_sum_error",
            "source_final_max_difference",
            "reload_max_difference",
            "model_text_is_nfc",
        ],
    )
    external_high_confidence = high_confidence_errors.loc[
        high_confidence_errors["dataset"].eq("external_pilot")
        & high_confidence_errors["model"].eq(FINAL_MODEL_NAME)
    ]
    if external_high_confidence.empty:
        external_high_confidence_table = "Nuk pati gabime të jashtme me confidence >=90%."
    else:
        external_high_confidence_table = dataframe_to_markdown(
            external_high_confidence,
            [
                "case_id",
                "label",
                "binary_prediction",
                "title",
                "probability_fake",
                "confidence",
            ],
        )
    report = f"""# Dita 17 - Modeli final klasik

## Vendimi

Modeli **është gati të konsiderohet final** për pipeline-in klasik të projektit:

`Word + Character TF-IDF + Linear SVM (C=1.0) + sigmoid calibration`.

Pragjet janë ngrirë në `0.30/0.70`. Nuk u bë tuning, trajnim, ndryshim
classifier-i, ndryshim TF-IDF ose zgjedhje nga benchmark-u i jashtëm.
Artefakti final është kopje byte-for-byte e kandidatit të Ditës 16.

## Artefakti

- Versioni: `{FINAL_MODEL_VERSION}`
- Model ID: `{FINAL_MODEL_ID}`
- Path: `{metrics['artifact']['final_path']}`
- SHA-256: `{metrics['artifact']['final_sha256']}`
- Madhësia: {metrics['artifact']['size_mb']:.3f} MB
- Manifesti: `models/final_model_v1_manifest.json`

## Konfigurimi i ngrirë

- Train: {metrics['data']['train_rows']} artikuj, me
  {metrics['data']['train_real']} real dhe {metrics['data']['train_fake']} fake.
- Preprocessing: Unicode NFC, normalizim hapësirash, bashkim
  `title + ". " + content`; ruhen kapitalizimi, pikësimi dhe ë/ç.
- Word TF-IDF: n-grams 1-2, `min_df=2`, `max_features=30000`, pa lowercase.
- Character TF-IDF: `char_wb` n-grams 3-5, `min_df=2`,
  `max_features=50000`, pa lowercase.
- Classifier: Linear SVM, `C=1.0`, `class_weight="balanced"`.
- Calibration: sigmoid me 5 fold-e group-safe, pa overlap grupesh.
- Vendimi: `<0.30 likely_real`, `0.30-0.70 uncertain`, `>0.70 likely_fake`.

## Regression checks

{regression_table}

Të gjitha probabilitetet ishin në `[0,1]`, shuma ishte 1, vendimet ndoqën
pragjet, reload-i dha rezultate identike dhe preprocessing-u i prediction ishte
identik me evaluation për të {metrics['data']['internal_test_rows']} rreshtat.

## Metrikat zyrtare të brendshme

{internal_table}

Modeli final arriti accuracy {final_internal['accuracy']:.4f}, F1 weighted
{final_internal['f1_weighted']:.4f}, F1 fake {final_internal['f1_fake']:.4f},
Brier {final_internal['brier_score']:.4f}, log loss
{final_internal['log_loss']:.4f} dhe ECE {final_internal['ece']:.4f}.
Këto janë metrikat zyrtare që duhen përdorur në diplomë.

![Krahasimi final](figures/day17_final_model_comparison.png)

## Benchmark-u i jashtëm pilot

{external_table}

Modeli final arriti accuracy {final_external['accuracy']:.4f}, recall real
{final_external['recall_real']:.4f} dhe recall fake
{final_external['recall_fake']:.4f}. Ky benchmark ka 40 përmbledhje të shkurtra,
domain shift në gjatësi, stil, periudhë dhe lloj burimi. Përdoret vetëm si
vlerësim pilot; nuk u përdor për tuning, calibration, pragje ose ngrirjen e
modelit.

Gabimet e modelit final me confidence të paktën 90%:

{external_high_confidence_table}

## Rezultatet sipas gjatësisë

{length_table}

{special_table}

![Gjatësia](figures/day17_final_length_performance.png)

Calibration nuk e zgjidh bias-in e gjatësisë. Veçanërisht, fake mbi 250 fjalë
mbeten një grup i vështirë, ndërsa grupi 30-60 fjalë ka shumë pak raste të
brendshme dhe duhet interpretuar me kujdes.

## Rastet finale të demonstrimit

{demo_table}

Shpjegimet janë përshkruese dhe bazohen në sinjale të vëzhgueshme; ato nuk janë
shpjegime shkakësore ose fact-checking.

## Kufizimet

- Modeli analizon ngjashmëri tekstuale dhe stilistike, jo fakte, URL, autorë,
  prova ose burime reale.
- Corpus-i ka lidhje mes label-it, gjatësisë dhe llojit të burimit.
- Tekstet shumë të shkurtra dhe fake shumë të gjata mbeten problematike.
- Benchmark-u i jashtëm është i vogël dhe me përmbledhje manuale.
- Probability calibration nuk garanton calibration të njëjtë pas domain shift-it.
- Rezultati duhet paraqitur si probabilitet sipas modelit dhe duhet shoqëruar me
  paralajmërim për verifikim faktik.

## Dita 18

Dita 18 duhet vetëm të kalojë Streamlit te `predict_final_news()`, të përdorë
artefaktin final dhe manifestin, të ruajë pragjet 0.30/0.70, dhe të ekzekutojë
testet e regresionit të UI-së. Modeli klasik nuk duhet ndryshuar më, përveç një
bug-u real të dokumentuar.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


