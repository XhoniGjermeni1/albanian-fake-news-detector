"""Evaluate the frozen calibrated model on the Day 10 external dataset."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.evaluation.data_utils import exclude_train_duplicates_from_test
from src.evaluation.experiment_utils import (
    dataframe_to_markdown,
    file_sha256,
    format_percent as _percent,
)
from src.features.linguistic_features import extract_linguistic_features
from src.models.predict import (
    DEFAULT_FAKE_THRESHOLD,
    DEFAULT_REAL_THRESHOLD,
    classify_probability,
    load_model,
    predict_news_for_app,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_DATASET_PATH = PROJECT_ROOT / "data" / "external" / "external_news.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "calibrated_tfidf_logreg.joblib"
TRAIN_PATH = PROJECT_ROOT / "data" / "interim" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "interim" / "test.csv"
DAY9_METRICS_PATH = PROJECT_ROOT / "reports" / "day9_system_test_metrics.json"
RAW_METADATA_ROOT = (
    PROJECT_ROOT / "data" / "raw" / "alb-fake-news-corpus" / "full_texts"
)

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
PREDICTIONS_PATH = REPORTS_DIR / "day11_external_predictions.csv"
METRICS_PATH = REPORTS_DIR / "day11_external_metrics.json"
TOPIC_RESULTS_PATH = REPORTS_DIR / "day11_external_by_topic.csv"
LABEL_RESULTS_PATH = REPORTS_DIR / "day11_external_by_label.csv"
LENGTH_RESULTS_PATH = REPORTS_DIR / "day11_external_by_length.csv"
SOURCE_RESULTS_PATH = REPORTS_DIR / "day11_external_by_source.csv"
ERRORS_PATH = REPORTS_DIR / "day11_external_errors.csv"
INTERESTING_CASES_PATH = REPORTS_DIR / "day11_external_interesting_cases.csv"
CONFUSION_CSV_PATH = REPORTS_DIR / "day11_external_confusion_matrix.csv"
CONFUSION_FIGURE_PATH = FIGURES_DIR / "day11_external_confusion_matrix.png"
REPORT_PATH = REPORTS_DIR / "day11_external_evaluation.md"

LOGGER = logging.getLogger(__name__)
LABEL_TO_NUMBER = {"real": 0, "fake": 1}
NUMBER_TO_LABEL = {0: "real", 1: "fake"}
REQUIRED_EXTERNAL_COLUMNS = {
    "external_id",
    "title",
    "content",
    "label",
    "source",
    "topic",
    "published_date",
}


def source_group(source: str) -> str:
    """Create a coarse source group for the bias analysis."""
    if source.startswith("Këshilli i Ministrave"):
        return "institutional_government"
    if source == "Banka e Shqipërisë":
        return "institutional_financial"
    if source == "INSTAT":
        return "institutional_statistics"
    if source.startswith("Krypometër"):
        return "fact_checked_social_claim"
    return "other"


def load_external_inputs() -> tuple[pd.DataFrame, object, dict]:
    """Load the frozen model, external data, and internal comparison inputs."""
    required_paths = [
        EXTERNAL_DATASET_PATH,
        MODEL_PATH,
        TRAIN_PATH,
        TEST_PATH,
        DAY9_METRICS_PATH,
    ]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing required Day 11 inputs: {missing_paths}")

    external = pd.read_csv(
        EXTERNAL_DATASET_PATH,
        encoding="utf-8",
        keep_default_na=False,
    )
    missing_columns = sorted(REQUIRED_EXTERNAL_COLUMNS - set(external.columns))
    if missing_columns:
        raise ValueError(f"External dataset is missing columns: {missing_columns}")
    if len(external) != 40:
        raise ValueError(f"Expected 40 external rows, found {len(external)}")
    if set(external["label"]) != set(LABEL_TO_NUMBER):
        raise ValueError("External labels must contain only real and fake.")

    model = load_model(MODEL_PATH)
    day9_metrics = json.loads(DAY9_METRICS_PATH.read_text(encoding="utf-8"))
    return external, model, day9_metrics


def run_external_predictions(external: pd.DataFrame, model) -> pd.DataFrame:
    """Call predict_news_for_app for every external article."""
    rows: list[dict] = []
    for article in external.itertuples(index=False):
        result = predict_news_for_app(article.title, article.content, model=model)
        probability_real = float(result["probability_real"])
        probability_fake = float(result["probability_fake"])
        binary_number = int(probability_fake >= 0.50)
        true_number = LABEL_TO_NUMBER[article.label]
        explanation = result["linguistic_explanation"]

        if result["decision"] != classify_probability(probability_fake):
            raise RuntimeError(f"Threshold mismatch for {article.external_id}")
        if result["thresholds"] != {
            "likely_real_below": DEFAULT_REAL_THRESHOLD,
            "likely_fake_above": DEFAULT_FAKE_THRESHOLD,
        }:
            raise RuntimeError(f"Unexpected thresholds for {article.external_id}")

        error_type = "correct"
        if true_number == 0 and binary_number == 1:
            error_type = "false_positive"
        elif true_number == 1 and binary_number == 0:
            error_type = "false_negative"

        rows.append(
            {
                "external_id": article.external_id,
                "title": article.title,
                "content": article.content,
                "true_label": article.label,
                "true_label_number": true_number,
                "binary_prediction": NUMBER_TO_LABEL[binary_number],
                "binary_prediction_number": binary_number,
                "probability_real": probability_real,
                "probability_fake": probability_fake,
                "probability_sum": round(probability_real + probability_fake, 4),
                "decision": result["decision"],
                "topic": article.topic,
                "source": article.source,
                "source_group": source_group(article.source),
                "published_date": article.published_date,
                "prediction_correct": true_number == binary_number,
                "error_type": error_type,
                "predicted_confidence": max(probability_real, probability_fake),
                "word_count": int(explanation["word_count"]),
                "text_length": int(explanation["text_length"]),
                "exclamation_count": int(explanation["exclamation_count"]),
                "uppercase_ratio": float(explanation["uppercase_ratio"]),
                "diacritic_ratio": float(explanation["diacritic_ratio"]),
                "sensational_words_found": " | ".join(
                    explanation["sensational_words_found"]
                ),
                "source_markers_found": " | ".join(
                    explanation["source_markers_found"]
                ),
                "uncertainty_markers_found": " | ".join(
                    explanation["uncertainty_markers_found"]
                ),
            }
        )

    predictions = pd.DataFrame(rows)
    predictions["length_group"] = pd.cut(
        predictions["word_count"],
        bins=[-np.inf, 44, 47, np.inf],
        labels=["short_38_44", "medium_45_47", "long_48_51"],
    ).astype(str)
    return predictions


def calculate_binary_metrics(predictions: pd.DataFrame) -> dict:
    """Calculate overall and per-class binary classification metrics."""
    y_true = predictions["true_label_number"].to_numpy()
    y_pred = predictions["binary_prediction_number"].to_numpy()
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    weighted = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    macro = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    per_class = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        average=None,
        zero_division=0,
    )

    return {
        "rows": int(len(predictions)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(weighted[0]), 4),
        "recall": round(float(weighted[1]), 4),
        "f1": round(float(weighted[2]), 4),
        "averaging": "weighted",
        "precision_macro": round(float(macro[0]), 4),
        "recall_macro": round(float(macro[1]), 4),
        "f1_macro": round(float(macro[2]), 4),
        "class_real": {
            "precision": round(float(per_class[0][0]), 4),
            "recall": round(float(per_class[1][0]), 4),
            "f1": round(float(per_class[2][0]), 4),
            "support": int(per_class[3][0]),
        },
        "class_fake": {
            "precision": round(float(per_class[0][1]), 4),
            "recall": round(float(per_class[1][1]), 4),
            "f1": round(float(per_class[2][1]), 4),
            "support": int(per_class[3][1]),
        },
        "confusion_matrix_labels": ["real", "fake"],
        "confusion_matrix": matrix.tolist(),
        "true_negatives": int(matrix[0, 0]),
        "false_positives": int(matrix[0, 1]),
        "false_negatives": int(matrix[1, 0]),
        "true_positives": int(matrix[1, 1]),
        "high_confidence_errors_90": int(
            (
                predictions["error_type"].ne("correct")
                & predictions["predicted_confidence"].ge(0.90)
            ).sum()
        ),
        "balanced_constant_baseline_accuracy": 0.50,
    }


def calculate_decision_metrics(predictions: pd.DataFrame) -> dict:
    """Evaluate the unchanged 30/70 three-level decision policy."""
    strong_mask = predictions["decision"].ne("uncertain")
    strong_correct = (
        (
            predictions["true_label"].eq("real")
            & predictions["decision"].eq("likely_real")
        )
        | (
            predictions["true_label"].eq("fake")
            & predictions["decision"].eq("likely_fake")
        )
    )
    binary_error = predictions["error_type"].ne("correct")
    uncertain_mask = predictions["decision"].eq("uncertain")
    strong_count = int(strong_mask.sum())
    uncertain_count = int(uncertain_mask.sum())
    total_errors = int(binary_error.sum())
    errors_in_uncertain = int((binary_error & uncertain_mask).sum())

    return {
        "thresholds": {
            "likely_real_below": DEFAULT_REAL_THRESHOLD,
            "likely_fake_above": DEFAULT_FAKE_THRESHOLD,
        },
        "likely_real": int(predictions["decision"].eq("likely_real").sum()),
        "uncertain": uncertain_count,
        "likely_fake": int(predictions["decision"].eq("likely_fake").sum()),
        "uncertain_real": int(
            (uncertain_mask & predictions["true_label"].eq("real")).sum()
        ),
        "uncertain_fake": int(
            (uncertain_mask & predictions["true_label"].eq("fake")).sum()
        ),
        "strong_decision_count": strong_count,
        "strong_decision_coverage": round(strong_count / len(predictions), 4),
        "strong_decision_accuracy": round(
            float(strong_correct[strong_mask].mean()) if strong_count else 0.0,
            4,
        ),
        "strong_decision_errors": int((strong_mask & ~strong_correct).sum()),
        "strong_false_positives": int(
            (
                predictions["true_label"].eq("real")
                & predictions["decision"].eq("likely_fake")
            ).sum()
        ),
        "strong_false_negatives": int(
            (
                predictions["true_label"].eq("fake")
                & predictions["decision"].eq("likely_real")
            ).sum()
        ),
        "binary_errors": total_errors,
        "binary_errors_moved_to_uncertain": errors_in_uncertain,
        "binary_error_capture_rate": round(
            errors_in_uncertain / total_errors if total_errors else 0.0,
            4,
        ),
        "uncertain_error_rate": round(
            errors_in_uncertain / uncertain_count if uncertain_count else 0.0,
            4,
        ),
    }


def summarize_groups(predictions: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """Build a common summary for topic, label, length, or source groups."""
    rows: list[dict] = []
    for group_value, group in predictions.groupby(group_column, sort=True, observed=True):
        strong_mask = group["decision"].ne("uncertain")
        strong_correct = (
            (group["true_label"].eq("real") & group["decision"].eq("likely_real"))
            | (group["true_label"].eq("fake") & group["decision"].eq("likely_fake"))
        )
        y_true = group["true_label_number"].to_numpy()
        y_pred = group["binary_prediction_number"].to_numpy()
        fake_precision, fake_recall, fake_f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[1],
            average=None,
            zero_division=0,
        )
        rows.append(
            {
                group_column: str(group_value),
                "rows": int(len(group)),
                "real_rows": int(group["true_label"].eq("real").sum()),
                "fake_rows": int(group["true_label"].eq("fake").sum()),
                "correct": int(group["prediction_correct"].sum()),
                "accuracy": round(float(group["prediction_correct"].mean()), 4),
                "precision_fake": round(float(fake_precision[0]), 4),
                "recall_fake": round(float(fake_recall[0]), 4),
                "f1_fake": round(float(fake_f1[0]), 4),
                "false_positives": int(group["error_type"].eq("false_positive").sum()),
                "false_negatives": int(group["error_type"].eq("false_negative").sum()),
                "likely_real": int(group["decision"].eq("likely_real").sum()),
                "uncertain": int(group["decision"].eq("uncertain").sum()),
                "likely_fake": int(group["decision"].eq("likely_fake").sum()),
                "strong_coverage": round(float(strong_mask.mean()), 4),
                "strong_accuracy": round(
                    float(strong_correct[strong_mask].mean()) if strong_mask.any() else 0.0,
                    4,
                ),
                "mean_probability_fake": round(float(group["probability_fake"].mean()), 4),
                "mean_word_count": round(float(group["word_count"].mean()), 2),
                "min_word_count": int(group["word_count"].min()),
                "max_word_count": int(group["word_count"].max()),
            }
        )
    return pd.DataFrame(rows)


def raw_corpus_date_range() -> dict:
    """Read the publication-date range recorded in the raw corpus metadata."""
    dates: list[datetime] = []
    for directory_name in ("true-meta-information", "fake-meta-information"):
        directory = RAW_METADATA_ROOT / directory_name
        if not directory.exists():
            continue
        for path in directory.glob("*.txt"):
            with path.open(encoding="utf-8", errors="replace") as metadata_file:
                first_line = metadata_file.readline().strip()
            try:
                dates.append(datetime.strptime(first_line, "%Y/%m/%d, %H:%M:%S"))
            except ValueError:
                continue
    return {
        "valid_metadata_dates": len(dates),
        "minimum": min(dates).date().isoformat() if dates else None,
        "maximum": max(dates).date().isoformat() if dates else None,
    }


def build_internal_comparison(
    predictions: pd.DataFrame,
    day9_metrics: dict,
) -> dict:
    """Collect comparable Day 9 metrics and text-length context."""
    train = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig", keep_default_na=False)
    test = pd.read_csv(TEST_PATH, encoding="utf-8-sig", keep_default_na=False)
    internal_test, excluded_ids = exclude_train_duplicates_from_test(train, test)
    internal_test = internal_test.reset_index(drop=True)
    internal_words = pd.Series(
        [
            int(extract_linguistic_features(row.title, row.content)["word_count"])
            for row in internal_test.itertuples(index=False)
        ]
    )
    internal_by_label = {}
    for label_number, group in internal_test.assign(word_count=internal_words).groupby("label"):
        words = group["word_count"]
        internal_by_label[NUMBER_TO_LABEL[int(label_number)]] = {
            "rows": int(len(group)),
            "mean_words": round(float(words.mean()), 2),
            "median_words": round(float(words.median()), 2),
        }

    external_dates = pd.to_datetime(predictions["published_date"], errors="coerce")
    internal_summary = day9_metrics["test_set"]
    return {
        "internal": {
            "rows": int(internal_summary["rows"]),
            "accuracy": float(internal_summary["binary_accuracy"]),
            "false_positives": int(internal_summary["false_positives"]),
            "false_negatives": int(internal_summary["false_negatives"]),
            "strong_decision_coverage": float(
                internal_summary["strong_decision_coverage"]
            ),
            "strong_decision_accuracy": float(
                internal_summary["strong_decision_accuracy"]
            ),
            "uncertain": int(internal_summary["uncertain_count"]),
            "mean_words": round(float(internal_words.mean()), 2),
            "median_words": round(float(internal_words.median()), 2),
            "min_words": int(internal_words.min()),
            "max_words": int(internal_words.max()),
            "words_by_label": internal_by_label,
            "publication_dates": raw_corpus_date_range(),
            "exact_train_duplicates_excluded": len(excluded_ids),
        },
        "external": {
            "rows": int(len(predictions)),
            "accuracy": round(float(predictions["prediction_correct"].mean()), 4),
            "mean_words": round(float(predictions["word_count"].mean()), 2),
            "median_words": round(float(predictions["word_count"].median()), 2),
            "min_words": int(predictions["word_count"].min()),
            "max_words": int(predictions["word_count"].max()),
            "words_by_label": {
                label: {
                    "rows": int(len(group)),
                    "mean_words": round(float(group["word_count"].mean()), 2),
                    "median_words": round(float(group["word_count"].median()), 2),
                }
                for label, group in predictions.groupby("true_label")
            },
            "publication_dates": {
                "minimum": external_dates.min().date().isoformat(),
                "maximum": external_dates.max().date().isoformat(),
            },
            "text_format": "manual summaries",
        },
        "accuracy_difference_external_minus_internal": round(
            float(predictions["prediction_correct"].mean())
            - float(internal_summary["binary_accuracy"]),
            4,
        ),
    }


def interpret_case(row: pd.Series, internal_real_median_words: float = 199.0) -> str:
    """Create a cautious, evidence-based interpretation for an error or uncertain case."""
    reasons: list[str] = []
    if row["error_type"] == "false_positive":
        reasons.append(
            "Rasti real u shty drejt fake; përmbledhja shumë e shkurtër mund të "
            f"ketë ngjarë me anën e shkurtër të corpus-it ({row['word_count']} fjalë "
            f"kundrejt medianës {internal_real_median_words:.0f} për real në test set)."
        )
        reasons.append(
            "Burimi institucional është vetëm metadata dhe modeli nuk e sheh atë si feature."
        )
    elif row["error_type"] == "false_negative":
        reasons.append(
            "Pretendimi fake është përmbledhur me stil neutral; prova e fact-check-ut "
            "ruhet veçmas dhe nuk i jepet modelit, prandaj formulimi mund të duket si lajm real."
        )
    else:
        reasons.append(
            "Probabiliteti ra në zonën 0.30-0.70, duke treguar se sinjalet tekstuale "
            "nuk mbështetën një vendim të fortë."
        )

    sensational = str(row["sensational_words_found"]).strip()
    source_markers = str(row["source_markers_found"]).strip()
    if sensational:
        reasons.append(f"U gjetën markerë sensacionalë: {sensational}.")
    elif row["error_type"] == "false_negative":
        reasons.append("Nuk u gjet asnjë marker nga lista e kufizuar sensacionale.")
    if source_markers:
        reasons.append(f"U gjetën markerë burimi: {source_markers}.")
    if row["predicted_confidence"] >= 0.90 and row["error_type"] != "correct":
        reasons.append(
            "Gabimi mbeti me të paktën 90% siguri; calibration nuk garanton "
            "saktësi për një rast individual jashtë shpërndarjes së trajnimit."
        )
    if row["decision"] == "uncertain" and row["error_type"] != "correct":
        reasons.append("Pragjet e zhvendosën këtë gabim binar në zonën uncertain.")
    return " ".join(reasons)


def build_case_tables(predictions: pd.DataFrame, internal_comparison: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return every binary error and the union of errors with uncertain cases."""
    median_real_words = internal_comparison["internal"]["words_by_label"]["real"][
        "median_words"
    ]
    table = predictions.copy()
    table["interpretation"] = [
        interpret_case(row, median_real_words) for _, row in table.iterrows()
    ]

    errors = table.loc[table["error_type"].ne("correct")].copy()
    interesting_mask = table["error_type"].ne("correct") | table["decision"].eq(
        "uncertain"
    )
    interesting = table.loc[interesting_mask].copy()

    def tags(row: pd.Series) -> str:
        values: list[str] = []
        if row["error_type"] != "correct":
            values.append(str(row["error_type"]))
        if row["decision"] == "uncertain":
            values.append("uncertain")
        if row["predicted_confidence"] >= 0.90 and row["error_type"] != "correct":
            values.append("high_confidence_error")
        if row["word_count"] <= 44:
            values.append("short_input")
        return " | ".join(values)

    interesting.insert(1, "case_tags", [tags(row) for _, row in interesting.iterrows()])
    return errors, interesting


def plot_confusion(matrix: list[list[int]]) -> None:
    """Save a simple confusion matrix for the external evaluation."""
    values = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    image = ax.imshow(values, cmap="Blues", vmin=0, vmax=max(int(values.max()), 1))
    ax.set_xticks([0, 1], labels=["Real", "Fake"])
    ax.set_yticks([0, 1], labels=["Real", "Fake"])
    ax.set_xlabel("Prediction binar")
    ax.set_ylabel("Label-i i dokumentuar")
    ax.set_title("Confusion matrix: dataseti i jashtëm (n=40)")

    threshold = values.max() / 2
    for row in range(2):
        for column in range(2):
            ax.text(
                column,
                row,
                str(values[row, column]),
                ha="center",
                va="center",
                color="white" if values[row, column] > threshold else "black",
                fontsize=15,
                fontweight="bold",
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(CONFUSION_FIGURE_PATH, dpi=180)
    plt.close(fig)


def write_report(
    metrics: dict,
    predictions: pd.DataFrame,
    by_topic: pd.DataFrame,
    by_label: pd.DataFrame,
    by_length: pd.DataFrame,
    by_source: pd.DataFrame,
    errors: pd.DataFrame,
) -> None:
    """Write the complete Day 11 evaluation report in Albanian."""
    binary = metrics["binary_metrics"]
    decision = metrics["decision_metrics"]
    comparison = metrics["internal_comparison"]
    high_confidence = errors.loc[errors["predicted_confidence"].ge(0.90)]
    uncertain = predictions.loc[predictions["decision"].eq("uncertain")]
    baseline_difference = binary["accuracy"] - binary["balanced_constant_baseline_accuracy"]
    worst_topic = by_topic.sort_values("accuracy").iloc[0]
    best_topic_accuracy = float(by_topic["accuracy"].max())
    best_topics = ", ".join(
        by_topic.loc[by_topic["accuracy"].eq(best_topic_accuracy), "topic"].tolist()
    )
    one_sided_topics = ", ".join(
        by_topic.loc[
            by_topic["false_positives"].eq(by_topic["real_rows"])
            & by_topic["recall_fake"].eq(1.0),
            "topic",
        ].tolist()
    )

    topic_table = dataframe_to_markdown(
        by_topic,
        [
            "topic",
            "rows",
            "accuracy",
            "precision_fake",
            "recall_fake",
            "false_positives",
            "false_negatives",
            "uncertain",
            "strong_accuracy",
        ],
    )
    label_table = dataframe_to_markdown(
        by_label,
        [
            "true_label",
            "rows",
            "correct",
            "accuracy",
            "likely_real",
            "uncertain",
            "likely_fake",
            "mean_probability_fake",
            "mean_word_count",
        ],
    )
    length_table = dataframe_to_markdown(
        by_length,
        [
            "length_group",
            "rows",
            "accuracy",
            "false_positives",
            "false_negatives",
            "uncertain",
            "mean_probability_fake",
        ],
    )
    source_table = dataframe_to_markdown(
        by_source,
        [
            "source",
            "rows",
            "real_rows",
            "fake_rows",
            "accuracy",
            "uncertain",
            "mean_probability_fake",
        ],
    )
    uncertain_table = dataframe_to_markdown(
        uncertain,
        [
            "external_id",
            "true_label",
            "binary_prediction",
            "probability_real",
            "probability_fake",
            "topic",
            "word_count",
            "title",
        ],
    )

    error_sections: list[str] = []
    for _, row in errors.sort_values(
        ["error_type", "predicted_confidence"], ascending=[True, False]
    ).iterrows():
        markers = (
            f"sensacionalë={row['sensational_words_found'] or 'asnjë'}; "
            f"burimi={row['source_markers_found'] or 'asnjë'}; "
            f"!={row['exclamation_count']}; uppercase={_percent(row['uppercase_ratio'])}; "
            f"diakritika={_percent(row['diacritic_ratio'])}"
        )
        error_sections.append(
            f"### {row['external_id']} - {row['error_type']}\n\n"
            f"- Titulli: {row['title']}\n"
            f"- Label / prediction: `{row['true_label']}` / `{row['binary_prediction']}`\n"
            f"- Probabiliteti Real / Fake: {_percent(row['probability_real'])} / "
            f"{_percent(row['probability_fake'])}\n"
            f"- Vendimi / tema / fjalët: `{row['decision']}` / `{row['topic']}` / "
            f"{row['word_count']}\n"
            f"- Sinjale: {markers}\n"
            f"- Interpretim: {row['interpretation']}"
        )

    high_confidence_ids = ", ".join(high_confidence["external_id"].tolist()) or "asnjë"
    internal = comparison["internal"]
    external = comparison["external"]
    comparison_table = (
        "| Treguesi | Test set-i i brendshëm | Dataseti i jashtëm |\n"
        "| --- | ---: | ---: |\n"
        f"| Raste | {internal['rows']} | {external['rows']} |\n"
        f"| Accuracy binare | {_percent(internal['accuracy'])} | "
        f"{_percent(external['accuracy'])} |\n"
        f"| Coverage e fortë | {_percent(internal['strong_decision_coverage'])} | "
        f"{_percent(decision['strong_decision_coverage'])} |\n"
        f"| Accuracy e fortë | {_percent(internal['strong_decision_accuracy'])} | "
        f"{_percent(decision['strong_decision_accuracy'])} |\n"
        f"| Uncertain | {internal['uncertain']} | {decision['uncertain']} |\n"
        f"| Mesatarja e fjalëve | {internal['mean_words']:.2f} | "
        f"{external['mean_words']:.2f} |\n"
        f"| Mediana e fjalëve | {internal['median_words']:.2f} | "
        f"{external['median_words']:.2f} |"
    )
    error_text = "\n\n".join(error_sections)

    report = f"""# Dita 11 - Vlerësimi në datasetin e jashtëm

## Qëllimi dhe integriteti i eksperimentit

Modeli ekzistues `models/calibrated_tfidf_logreg.joblib` u ngarkua vetëm për
inference. Çdo rresht u analizua me `predict_news_for_app()`, i cili përdor të
njëjtin `combine_title_content()` dhe të njëjtat pragje si aplikacioni:

- `probability_fake < 0.30`: `likely_real`;
- `0.30 <= probability_fake <= 0.70`: `uncertain`;
- `probability_fake > 0.70`: `likely_fake`.

Nuk u thirr asnjë metodë `fit`, modeli nuk u ritrajnua, dataset-i nuk u ndryshua
dhe fingerprint-et SHA-256 të modelit dhe datasetit mbetën të pandryshuara para
dhe pas vlerësimit.

## Metrikat binare

Prediction-i binar përdor pragun standard `probability_fake >= 0.50` për `fake`.
Precision, recall dhe F1 kryesore janë mesatare të ponderuara, në përputhje me
raportet e mëparshme të projektit.

| Metrika | Rezultati |
| --- | ---: |
| Accuracy | {_percent(binary['accuracy'])} |
| Precision weighted | {_percent(binary['precision'])} |
| Recall weighted | {_percent(binary['recall'])} |
| F1 weighted | {_percent(binary['f1'])} |
| Precision macro | {_percent(binary['precision_macro'])} |
| Recall macro | {_percent(binary['recall_macro'])} |
| F1 macro | {_percent(binary['f1_macro'])} |
| Precision fake | {_percent(binary['class_fake']['precision'])} |
| Recall fake | {_percent(binary['class_fake']['recall'])} |
| F1 fake | {_percent(binary['class_fake']['f1'])} |
| Precision real | {_percent(binary['class_real']['precision'])} |
| Recall real | {_percent(binary['class_real']['recall'])} |
| F1 real | {_percent(binary['class_real']['f1'])} |

Confusion matrix, me rreshta `true` dhe kolona `predicted` në rendin
`[real, fake]`:

```text
{binary['confusion_matrix']}
```

- True real: **{binary['true_negatives']}**;
- false positives, real të klasifikuara fake: **{binary['false_positives']}**;
- false negatives, fake të klasifikuara real: **{binary['false_negatives']}**;
- true fake: **{binary['true_positives']}**.

Dataset-i është i balancuar, prandaj një klasifikues konstant do të kishte 50%
accuracy. Rezultati {_percent(binary['accuracy'])} është
{abs(100 * baseline_difference):.2f} pikë përqindjeje
{'mbi' if baseline_difference >= 0 else 'nën'} këtë baseline të thjeshtë. Modeli
dalloi {binary['true_positives']} nga {binary['class_fake']['support']} rastet fake,
por vetëm {binary['true_negatives']} nga {binary['class_real']['support']} rastet
real. Pra recall-i i mirë për fake vjen së bashku me një numër
shumë të lartë false positives.

## Vendimet me tri nivele

- `likely_real`: **{decision['likely_real']}**;
- `uncertain`: **{decision['uncertain']}**;
- `likely_fake`: **{decision['likely_fake']}**;
- coverage e vendimeve të forta: **{_percent(decision['strong_decision_coverage'])}**;
- accuracy e vendimeve të forta: **{_percent(decision['strong_decision_accuracy'])}**;
- gabime binare të zhvendosura në `uncertain`: **{decision['binary_errors_moved_to_uncertain']}**
  nga {decision['binary_errors']} ({_percent(decision['binary_error_capture_rate'])});
- gabime që mbetën vendime të forta: **{decision['strong_decision_errors']}**.

Zona `uncertain` ishte e dobishme si sinjal paralajmërues:
{decision['binary_errors_moved_to_uncertain']} nga {decision['uncertain']} rastet
e saj ishin gabime binare. Megjithatë, ajo nuk e zgjidhi zhvendosjen e
përgjithshme drejt klasës fake; {decision['strong_decision_errors']} gabime
mbetën vendime të forta.

## Sipas label-it

{label_table}

Të gjitha burimet institucionale i përkasin klasës real dhe të gjitha pretendimet
e dokumentuara nga Krypometër klasës fake. Për këtë arsye ndikimi i burimit nuk
mund të ndahet statistikisht nga ndikimi i label-it në këtë dataset.

## Sipas temës

{topic_table}

Tema me rezultatin më të dobët ishte `{worst_topic['topic']}` me
{_percent(float(worst_topic['accuracy']))}. Rezultatin më të lartë
{_percent(best_topic_accuracy)} e arritën: {best_topics}. Kjo nuk nënkupton
domosdoshmërisht balancë të mirë. Në temat {one_sided_topics or 'asnjë'}, modeli
gjeti të gjitha rastet fake dhe humbi të gjitha rastet real.

## Sipas gjatësisë

Grupet u përcaktuan mbi tekstin e plotë që pa modeli: 38-44 fjalë, 45-47 fjalë
dhe 48-51 fjalë.

{length_table}

Të gjitha tekstet e jashtme janë pranë kufirit minimal të test set-it të
brendshëm. Gjatësia nuk ndahet qartë mes real/fake në datasetin e jashtëm, ndërsa
në test set-in e brendshëm mediana ishte
{internal['words_by_label']['real']['median_words']:.0f} fjalë për real dhe
{internal['words_by_label']['fake']['median_words']:.0f} për fake. Kjo e bën
shkurtësinë një shpjegim të mundshëm për prirjen e fortë drejt fake.

## Sipas burimit

{source_table}

Burimi nuk futet në model. Kjo tabelë analizon sjelljen pas prediction-it dhe nuk
provon shkakësi. Rezultati tregon megjithatë se formati i shkurtër institucional
nuk u trajtua si artikujt real të gjatë të corpus-it.

## Rastet uncertain

{uncertain_table}

## Gabimet me probabilitet të lartë

U gjetën **{binary['high_confidence_errors_90']}** gabime me të paktën 90% siguri:
`{high_confidence_ids}`. Kjo tregon se probabiliteti i kalibruar është besimi i
modelit brenda sinjaleve që ka mësuar, jo garanci faktike ose garanci
përgjithësimi jashtë shpërndarjes.

## Analiza e çdo gabimi

Interpretimet e mëposhtme janë hipoteza të kujdesshme mbi stilin dhe sinjalet që
modeli sheh; ato nuk provojnë shkakun e saktë të çdo prediction-i.

{error_text}

## Krahasimi me test set-in e brendshëm

{comparison_table}

Krahasimi nuk është eksperiment i barabartë:

- test set-i i brendshëm ka artikuj të plotë dhe 792 raste; dataseti i jashtëm
  ka 40 përmbledhje manuale;
- mesatarja ra nga {internal['mean_words']:.2f} në {external['mean_words']:.2f}
  fjalë;
- corpus-i i brendshëm mbulon periudhën
  {internal['publication_dates']['minimum']} deri
  {internal['publication_dates']['maximum']}, ndërsa rastet e jashtme periudhën
  {external['publication_dates']['minimum']} deri
  {external['publication_dates']['maximum']};
- temat e jashtme janë të balancuara me dorë;
- burimet dhe label-et e jashtme janë të ndërthurura: institucionale për real dhe
  pretendime sociale të fact-check-uara për fake;
- provat e etiketimit nuk iu dhanë modelit, sepse aplikacioni analizon vetëm
  titullin dhe përmbajtjen.

Rënia e accuracy ishte
**{100 * abs(comparison['accuracy_difference_external_minus_internal']):.2f}**
pikë përqindjeje. Për shkak të këtyre ndryshimeve, kjo nuk mat vetëm cilësinë e
modelit; mat edhe domain shift-in mes artikujve të corpus-it dhe përmbledhjeve të
shkurtra të jashtme.

## Përfundimi

Në këtë vlerësim modeli përgjithësoi **dobët**. Ai ruajti recall
{_percent(binary['class_fake']['recall'])} për fake, por accuracy totale
{_percent(binary['accuracy'])}, recall {_percent(binary['class_real']['recall'])}
për real dhe {binary['false_positives']} false positives tregojnë se nuk e ndan
në mënyrë të besueshme dy klasat jashtë corpus-it. Probabilitetet dhe `uncertain`
ndihmuan të shënohen {decision['binary_errors_moved_to_uncertain']} gabime, por
modeli dha ende {decision['strong_decision_errors']} vendime të forta të gabuara
dhe {binary['high_confidence_errors_90']} gabime mbi 90% siguri.

Ky rezultat nuk duhet përdorur për të ndryshuar datasetin ose për të zgjedhur
pragje të reja pas shikimit të përgjigjeve. Dataset-i i Ditës 10 duhet të mbetet
i ngrirë si kontroll i jashtëm.

## Rekomandimi për Ditën 12

1. Të analizohet në mënyrë të kontrolluar bias-i i gjatësisë, duke krahasuar
   artikuj të brendshëm dhe të jashtëm me gjatësi të ngjashme, pa ndryshuar këtë
   benchmark.
2. Të mblidhen më vonë artikuj të jashtëm më të plotë dhe burime të kryqëzuara,
   ku edhe real edhe fake vijnë nga disa lloje burimesh.
3. Të kontrollohet stabiliteti i probabiliteteve ndaj versionit të shkurtër dhe
   të zgjeruar të të njëjtit lajm.
4. Vetëm pas dokumentimit të këtyre analizave të vendoset nëse duhet një model i
   përmirësuar, linguistic features, ribalancim ose ndryshim i politikës
   `uncertain`. Vlerësimi i sotëm duhet të ruhet i pandryshuar.

## Output-et

```text
reports/day11_external_predictions.csv
reports/day11_external_metrics.json
reports/day11_external_by_topic.csv
reports/day11_external_by_label.csv
reports/day11_external_by_length.csv
reports/day11_external_by_source.csv
reports/day11_external_errors.csv
reports/day11_external_interesting_cases.csv
reports/day11_external_confusion_matrix.csv
reports/figures/day11_external_confusion_matrix.png
reports/day11_external_evaluation.md
```
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def run_external_evaluation() -> dict:
    """Run the complete frozen-model external evaluation and save artifacts."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    model_hash_before = file_sha256(MODEL_PATH)
    dataset_hash_before = file_sha256(EXTERNAL_DATASET_PATH)
    external, model, day9_metrics = load_external_inputs()
    predictions = run_external_predictions(external, model)

    maximum_probability_sum_error = float(
        predictions["probability_sum"].sub(1.0).abs().max()
    )
    if maximum_probability_sum_error > 0.0002:
        raise RuntimeError("Real and fake probabilities do not sum to approximately one.")

    binary_metrics = calculate_binary_metrics(predictions)
    decision_metrics = calculate_decision_metrics(predictions)
    internal_comparison = build_internal_comparison(predictions, day9_metrics)

    by_topic = summarize_groups(predictions, "topic")
    by_label = summarize_groups(predictions, "true_label")
    by_length = summarize_groups(predictions, "length_group")
    by_source = summarize_groups(predictions, "source")
    errors, interesting = build_case_tables(predictions, internal_comparison)

    model_hash_after = file_sha256(MODEL_PATH)
    dataset_hash_after = file_sha256(EXTERNAL_DATASET_PATH)
    if model_hash_before != model_hash_after:
        raise RuntimeError("The saved model changed during evaluation.")
    if dataset_hash_before != dataset_hash_after:
        raise RuntimeError("The external dataset changed during evaluation.")

    metrics = {
        "status": "completed",
        "model": MODEL_PATH.name,
        "model_retrained": False,
        "prediction_function": "predict_news_for_app",
        "preprocessing_function": "combine_title_content",
        "model_sha256_before": model_hash_before,
        "model_sha256_after": model_hash_after,
        "dataset_sha256_before": dataset_hash_before,
        "dataset_sha256_after": dataset_hash_after,
        "probability_checks": {
            "outside_0_1": int(
                (
                    predictions["probability_real"].lt(0)
                    | predictions["probability_real"].gt(1)
                    | predictions["probability_fake"].lt(0)
                    | predictions["probability_fake"].gt(1)
                ).sum()
            ),
            "maximum_sum_error": maximum_probability_sum_error,
        },
        "binary_metrics": binary_metrics,
        "decision_metrics": decision_metrics,
        "internal_comparison": internal_comparison,
        "artifacts": {
            "predictions": str(PREDICTIONS_PATH.relative_to(PROJECT_ROOT)),
            "metrics": str(METRICS_PATH.relative_to(PROJECT_ROOT)),
            "by_topic": str(TOPIC_RESULTS_PATH.relative_to(PROJECT_ROOT)),
            "by_label": str(LABEL_RESULTS_PATH.relative_to(PROJECT_ROOT)),
            "by_length": str(LENGTH_RESULTS_PATH.relative_to(PROJECT_ROOT)),
            "by_source": str(SOURCE_RESULTS_PATH.relative_to(PROJECT_ROOT)),
            "errors": str(ERRORS_PATH.relative_to(PROJECT_ROOT)),
            "interesting_cases": str(INTERESTING_CASES_PATH.relative_to(PROJECT_ROOT)),
            "confusion_csv": str(CONFUSION_CSV_PATH.relative_to(PROJECT_ROOT)),
            "confusion_figure": str(CONFUSION_FIGURE_PATH.relative_to(PROJECT_ROOT)),
            "report": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
        },
    }

    predictions.to_csv(PREDICTIONS_PATH, index=False, encoding="utf-8")
    by_topic.to_csv(TOPIC_RESULTS_PATH, index=False, encoding="utf-8")
    by_label.to_csv(LABEL_RESULTS_PATH, index=False, encoding="utf-8")
    by_length.to_csv(LENGTH_RESULTS_PATH, index=False, encoding="utf-8")
    by_source.to_csv(SOURCE_RESULTS_PATH, index=False, encoding="utf-8")
    errors.to_csv(ERRORS_PATH, index=False, encoding="utf-8")
    interesting.to_csv(INTERESTING_CASES_PATH, index=False, encoding="utf-8")

    confusion_values = pd.DataFrame(
        binary_metrics["confusion_matrix"],
        index=["true_real", "true_fake"],
        columns=["predicted_real", "predicted_fake"],
    )
    confusion_values.to_csv(CONFUSION_CSV_PATH, encoding="utf-8")
    plot_confusion(binary_metrics["confusion_matrix"])
    METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(
        metrics,
        predictions,
        by_topic,
        by_label,
        by_length,
        by_source,
        errors,
    )
    return metrics


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    metrics = run_external_evaluation()
    binary = metrics["binary_metrics"]
    decision = metrics["decision_metrics"]
    LOGGER.info("External rows: %s", binary["rows"])
    LOGGER.info("Binary accuracy: %.2f%%", binary["accuracy"] * 100)
    LOGGER.info(
        "Fake precision / recall / F1: %.2f%% / %.2f%% / %.2f%%",
        binary["class_fake"]["precision"] * 100,
        binary["class_fake"]["recall"] * 100,
        binary["class_fake"]["f1"] * 100,
    )
    LOGGER.info("Confusion matrix: %s", binary["confusion_matrix"])
    LOGGER.info(
        "Decisions likely_real / uncertain / likely_fake: %s / %s / %s",
        decision["likely_real"],
        decision["uncertain"],
        decision["likely_fake"],
    )
    LOGGER.info("Report saved to: %s", REPORT_PATH)


if __name__ == "__main__":
    main()
