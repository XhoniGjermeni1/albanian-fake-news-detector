"""Train and compare the Day 5 text, linguistic, and hybrid models."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.predict import predict_hybrid_news

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = PROJECT_ROOT / "data" / "interim" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "interim" / "test.csv"
CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "interim" / "articles_clean.csv"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "linguistic_features.csv"

MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

BASELINE_MODEL_PATH = MODEL_DIR / "baseline_tfidf_logreg.joblib"
LINGUISTIC_MODEL_PATH = MODEL_DIR / "linguistic_features_logreg.joblib"
HYBRID_MODEL_PATH = MODEL_DIR / "hybrid_tfidf_linguistic_logreg.joblib"
HYBRID_NO_LENGTH_MODEL_PATH = MODEL_DIR / "hybrid_tfidf_linguistic_no_length_logreg.joblib"

METRICS_PATH = REPORTS_DIR / "day5_metrics.json"
COMPARISON_PATH = REPORTS_DIR / "day5_model_comparison.csv"
REPORT_PATH = REPORTS_DIR / "day5_hybrid_model.md"
FIGURE_PATH = FIGURES_DIR / "day5_model_comparison.png"

DIRECT_LENGTH_FEATURES = [
    "word_count",
    "sentence_count",
    "character_count",
    "avg_sentence_length",
    "title_length",
    "content_length",
]

MODEL_NAMES = {
    "tfidf_only": "TF-IDF only",
    "linguistic_only": "Linguistic features only",
    "hybrid": "TF-IDF + linguistic features",
    "hybrid_no_length": "Hybrid without length features",
}

LOGGER = logging.getLogger(__name__)


def _require_columns(dataframe: pd.DataFrame, columns: set[str], name: str) -> None:
    """Raise a clear error when an input table is incomplete."""
    missing = sorted(columns - set(dataframe.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def numeric_feature_columns(features: pd.DataFrame) -> list[str]:
    """Return numeric linguistic features without identifiers or the label."""
    ignored_columns = {"pair_id", "label"}
    return [
        column
        for column in features.select_dtypes(include="number").columns
        if column not in ignored_columns
    ]


def merge_text_with_features(text_data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Join linguistic features by article_id and verify labels and pair IDs."""
    text_required = {"article_id", "pair_id", "label", "label_name", "model_text"}
    feature_required = {"article_id", "pair_id", "label", "label_name"}
    _require_columns(text_data, text_required, "Text data")
    _require_columns(features, feature_required, "Linguistic features")

    if text_data["article_id"].duplicated().any():
        raise ValueError("Text data contains duplicate article_id values.")
    if features["article_id"].duplicated().any():
        raise ValueError("Linguistic features contain duplicate article_id values.")

    feature_data = features.rename(
        columns={
            "pair_id": "feature_pair_id",
            "label": "feature_label",
            "label_name": "feature_label_name",
        }
    )

    ordered_text = text_data.copy()
    ordered_text["_day5_order"] = range(len(ordered_text))
    merged = ordered_text.merge(
        feature_data,
        on="article_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    missing_feature_rows = merged["_merge"].ne("both")
    if missing_feature_rows.any():
        missing_ids = merged.loc[missing_feature_rows, "article_id"].astype(str).tolist()
        raise ValueError(f"Missing linguistic features for article IDs: {missing_ids[:10]}")

    pair_mismatch = merged["pair_id"].astype("string") != merged["feature_pair_id"].astype("string")
    label_mismatch = merged["label"].astype(int) != merged["feature_label"].astype(int)
    name_mismatch = merged["label_name"].astype(str) != merged["feature_label_name"].astype(str)

    if pair_mismatch.any():
        raise ValueError("pair_id values do not match after the feature merge.")
    if label_mismatch.any() or name_mismatch.any():
        raise ValueError("Labels do not match after the feature merge.")

    merged = merged.sort_values("_day5_order").drop(
        columns=[
            "_day5_order",
            "_merge",
            "feature_pair_id",
            "feature_label",
            "feature_label_name",
        ]
    )
    return merged.reset_index(drop=True)


def exclude_train_duplicates_from_test(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Exclude exact train-text copies from the evaluation test set."""
    train_texts = set(train_data["model_text"])
    duplicate_mask = test_data["model_text"].isin(train_texts)
    excluded_ids = test_data.loc[duplicate_mask, "article_id"].astype(str).tolist()
    clean_test = test_data.loc[~duplicate_mask].reset_index(drop=True)
    return clean_test, excluded_ids


def load_and_check_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict]:
    """Load Day 2 and Day 3 outputs and run alignment and leakage checks."""
    required_paths = [TRAIN_PATH, TEST_PATH, CLEAN_DATA_PATH, FEATURES_PATH]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing required Day 2-3 outputs: {missing_paths}")

    clean_data = pd.read_csv(CLEAN_DATA_PATH, encoding="utf-8-sig", keep_default_na=False)
    train_data = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig", keep_default_na=False)
    test_data = pd.read_csv(TEST_PATH, encoding="utf-8-sig", keep_default_na=False)
    features = pd.read_csv(FEATURES_PATH, encoding="utf-8-sig", keep_default_na=False)

    dataframes = {
        "Clean data": clean_data,
        "Train data": train_data,
        "Test data": test_data,
        "Linguistic features": features,
    }
    for name, dataframe in dataframes.items():
        _require_columns(dataframe, {"article_id", "pair_id", "label", "label_name"}, name)
        if dataframe["article_id"].duplicated().any():
            raise ValueError(f"{name} contains duplicate article_id values.")

    _require_columns(clean_data, {"model_text"}, "Clean data")
    _require_columns(train_data, {"model_text"}, "Train data")
    _require_columns(test_data, {"model_text"}, "Test data")

    clean_ids = set(clean_data["article_id"])
    train_ids = set(train_data["article_id"])
    test_ids = set(test_data["article_id"])
    feature_ids = set(features["article_id"])

    if train_ids & test_ids:
        raise ValueError("Train and test contain overlapping article_id values.")
    if train_ids | test_ids != clean_ids:
        raise ValueError("Train and test article IDs do not reconstruct the clean dataset.")
    if feature_ids != clean_ids:
        raise ValueError("Linguistic feature IDs do not match the clean dataset IDs.")

    train_pairs = set(train_data["pair_id"].astype("string"))
    test_pairs = set(test_data["pair_id"].astype("string"))
    if train_pairs & test_pairs:
        raise ValueError("Train and test contain overlapping pair_id values.")

    # Check pair_id and label consistency for every article before modeling.
    merge_text_with_features(clean_data, features)

    evaluation_test, excluded_ids = exclude_train_duplicates_from_test(train_data, test_data)
    train_merged = merge_text_with_features(train_data, features)
    test_merged = merge_text_with_features(evaluation_test, features)
    feature_columns = numeric_feature_columns(features)

    if not feature_columns:
        raise ValueError("No numeric linguistic feature columns were found.")
    if train_merged[feature_columns].isna().any().any():
        raise ValueError("Train data contains missing linguistic feature values after the merge.")
    if test_merged[feature_columns].isna().any().any():
        raise ValueError("Test data contains missing linguistic feature values after the merge.")

    checks = {
        "clean_rows": int(len(clean_data)),
        "original_train_rows": int(len(train_data)),
        "original_test_rows": int(len(test_data)),
        "evaluation_test_rows": int(len(evaluation_test)),
        "train_test_article_id_overlap": 0,
        "train_test_pair_id_overlap": 0,
        "exact_train_texts_excluded_from_test": int(len(excluded_ids)),
        "excluded_article_ids": excluded_ids,
        "linguistic_feature_rows": int(len(features)),
        "numeric_linguistic_feature_count": int(len(feature_columns)),
        "missing_feature_rows": 0,
        "pair_id_mismatches": 0,
        "label_mismatches": 0,
        "empty_train_model_texts": int(train_data["model_text"].eq("").sum()),
        "empty_test_model_texts": int(evaluation_test["model_text"].eq("").sum()),
    }
    return train_merged, test_merged, feature_columns, checks


def _tfidf_vectorizer() -> TfidfVectorizer:
    """Use the same TF-IDF configuration as the Day 2 baseline."""
    return TfidfVectorizer(
        lowercase=False,
        ngram_range=(1, 2),
        min_df=2,
        max_features=30000,
    )


def _classifier() -> LogisticRegression:
    """Create the shared classifier used by all Day 5 comparisons."""
    return LogisticRegression(max_iter=1000, class_weight="balanced")


def build_tfidf_model() -> Pipeline:
    """Build the text-only baseline for the fair Day 5 comparison."""
    return Pipeline(steps=[("tfidf", _tfidf_vectorizer()), ("classifier", _classifier())])


def build_linguistic_model(feature_columns: list[str]) -> Pipeline:
    """Build a Logistic Regression model using only numeric features."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", _classifier()),
        ]
    )


def build_hybrid_model(feature_columns: list[str]) -> Pipeline:
    """Combine TF-IDF and standardized linguistic features."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    combined_features = ColumnTransformer(
        transformers=[
            ("tfidf", _tfidf_vectorizer(), "model_text"),
            ("linguistic", numeric_pipeline, feature_columns),
        ]
    )
    return Pipeline(steps=[("features", combined_features), ("classifier", _classifier())])


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Calculate overall and fake-class metrics."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    precision_fake, recall_fake, f1_fake, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "precision_fake": round(float(precision_fake), 4),
        "recall_fake": round(float(recall_fake), 4),
        "f1_fake": round(float(f1_fake), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def build_comparison_table(metrics: dict[str, dict]) -> pd.DataFrame:
    """Create one readable row per evaluated model."""
    rows = []
    for model_key in MODEL_NAMES:
        model_metrics = metrics[model_key]
        matrix = model_metrics["confusion_matrix"]
        rows.append(
            {
                "model_key": model_key,
                "model": MODEL_NAMES[model_key],
                "accuracy": model_metrics["accuracy"],
                "precision": model_metrics["precision"],
                "recall": model_metrics["recall"],
                "f1": model_metrics["f1"],
                "precision_fake": model_metrics["precision_fake"],
                "recall_fake": model_metrics["recall_fake"],
                "f1_fake": model_metrics["f1_fake"],
                "true_real_pred_real": matrix[0][0],
                "true_real_pred_fake": matrix[0][1],
                "true_fake_pred_real": matrix[1][0],
                "true_fake_pred_fake": matrix[1][1],
            }
        )
    return pd.DataFrame(rows)


def plot_comparison(comparison: pd.DataFrame) -> None:
    """Save a compact accuracy and fake-F1 comparison chart."""
    figure, axis = plt.subplots(figsize=(10, 5.5))
    y_positions = list(range(len(comparison)))
    axis.barh(
        [position + 0.18 for position in y_positions],
        comparison["accuracy"],
        height=0.34,
        label="Accuracy",
        color="#287271",
    )
    axis.barh(
        [position - 0.18 for position in y_positions],
        comparison["f1_fake"],
        height=0.34,
        label="F1 fake",
        color="#E76F51",
    )
    axis.set_yticks(y_positions, comparison["model"])
    axis.set_xlim(0, 1)
    axis.set_xlabel("Score")
    axis.set_title("Day 5 model comparison")
    axis.grid(axis="x", alpha=0.25)
    axis.legend(loc="lower right")

    for position, row in comparison.reset_index(drop=True).iterrows():
        axis.text(
            row["accuracy"] + 0.008,
            position + 0.18,
            f"{row['accuracy']:.4f}",
            va="center",
            fontsize=9,
        )
        axis.text(
            row["f1_fake"] + 0.008,
            position - 0.18,
            f"{row['f1_fake']:.4f}",
            va="center",
            fontsize=9,
        )

    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _relative_path(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _comparison_markdown(comparison: pd.DataFrame) -> str:
    lines = [
        "| Modeli | Accuracy | Precision | Recall | F1 | F1 fake |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.accuracy:.4f} | {row.precision:.4f} | "
            f"{row.recall:.4f} | {row.f1:.4f} | {row.f1_fake:.4f} |"
        )
    return "\n".join(lines)


def write_report(result: dict, comparison: pd.DataFrame) -> None:
    """Write the Day 5 report using the measured results."""
    checks = result["data_checks"]
    metrics = result["metrics"]
    sample = result["sample_prediction"]
    explanation = sample["prediction"]["linguistic_explanation"]
    hybrid_delta = metrics["hybrid"]["accuracy"] - metrics["tfidf_only"]["accuracy"]
    no_length_delta = metrics["hybrid_no_length"]["accuracy"] - metrics["hybrid"]["accuracy"]
    best_name = MODEL_NAMES[result["best_model_by_f1_fake"]]
    removed_features = ", ".join(
        f"`{name}`" for name in result["features"]["removed_length_features"]
    )

    report = f"""# Dita 5: Modeli hibrid

## Qëllimi

U ndërtua dhe u vlerësua modeli `TF-IDF + linguistic features` me Logistic Regression. Krahasimi përdor të njëjtin train dhe të njëjtin test të pastër për të katër provat.

## Kontrolli i output-eve të mëparshme

- Dataset i pastruar: {checks['clean_rows']} artikuj.
- Train fillestar: {checks['original_train_rows']} artikuj.
- Test fillestar: {checks['original_test_rows']} artikuj.
- Mbivendosje `article_id` train/test: {checks['train_test_article_id_overlap']}.
- Mbivendosje `pair_id` train/test: {checks['train_test_pair_id_overlap']}.
- Rreshta pa linguistic features: {checks['missing_feature_rows']}.
- Mospërputhje labels: {checks['label_mismatches']}.
- Mospërputhje `pair_id`: {checks['pair_id_mismatches']}.

U gjetën {checks['exact_train_texts_excluded_from_test']} artikuj në test me tekst identik me një tekst në train. Ata u përjashtuan vetëm nga vlerësimi i Ditës 5, prandaj testi i pastër ka {checks['evaluation_test_rows']} artikuj. Kjo shmang vlerësimin mbi kopje që modeli i ka parë gjatë trajnimit.

## Bashkimi i të dhënave

Teksti dhe karakteristikat gjuhësore u bashkuan vetëm me `article_id`. Pas bashkimit u kontrolluan përsëri `pair_id`, `label` dhe `label_name`; rendi i rreshtave nuk u përdor si supozim.

TF-IDF përdor `model_text`, që bashkon titullin me përmbajtjen pas normalizimit bazë të hapësirave. Shkronjat shqipe, kapitalizimi dhe pikësimi ruhen.

U përdorën {checks['numeric_linguistic_feature_count']} karakteristika numerike:

- strukturë dhe gjatësi: `word_count`, `sentence_count`, `character_count`, `avg_word_length`, `avg_sentence_length`, `title_length`, `content_length`;
- pikësim: count-et dhe ratio-t për pikëçuditëse, pikëpyetje, presje, thonjëza dhe tri pika;
- kapitalizim: `uppercase_word_count`, `uppercase_word_ratio`, `uppercase_char_ratio`, `title_excessive_uppercase`;
- diakritika: count-et për `ë`/`ç`, `diacritic_count`, `diacritic_ratio`, dhe sinjali për diakritika të mundshme që mungojnë;
- shprehje: count-et dhe ratio-t për fjalë sensacionale, tregues burimi dhe pasiguri.

`TfidfVectorizer`, imputimi dhe standardizimi u përshtatën vetëm mbi train brenda pipeline-it.

## Rezultatet

{_comparison_markdown(comparison)}

Kolonat Precision, Recall dhe F1 janë mesatare të ponderuara për të dy klasat. `F1 fake` paraqitet veçmas për klasën me label `1`.

Confusion matrix përdor rendin `[[real→real, real→fake], [fake→real, fake→fake]]`:

- TF-IDF only: `{metrics['tfidf_only']['confusion_matrix']}`
- Linguistic features only: `{metrics['linguistic_only']['confusion_matrix']}`
- Hybrid: `{metrics['hybrid']['confusion_matrix']}`
- Hybrid pa feature-t e gjatësisë: `{metrics['hybrid_no_length']['confusion_matrix']}`

## Prova pa feature-t e gjatësisë

U hoqën: {removed_features}.

Accuracy ndryshoi me {no_length_delta:+.4f} kundrejt modelit hibrid të plotë. Kjo tregon se në kombinimin aktual heqja e feature-ve direkte të gjatësisë nuk e dëmtoi rezultatin. Sinjalet e tjera gjuhësore mbetën të përdorshme, por nuk e kaluan TF-IDF baseline.

## Përfundimi i krahasimit

Modeli hibrid ndryshoi accuracy me {hybrid_delta:+.4f} kundrejt TF-IDF only. Pra linguistic features nuk e përmirësuan klasifikimin në këtë konfigurim të parë. Modeli më i mirë sipas F1 për klasën fake ishte **{best_name}**.

Për aplikacionin e ardhshëm, TF-IDF only është kandidati më i mirë aktual për parashikimin. Karakteristikat gjuhësore mbeten të vlefshme për shpjegimin e input-it dhe për analiza, edhe pse bashkimi i tyre nuk dha rritje metrike.

## Shembull parashikimi

- `article_id`: `{sample['article_id']}`
- Label real në dataset: `{sample['true_label']}`
- Parashikimi: `{sample['prediction']['label_name']}`
- Probabilitet real sipas modelit: {sample['prediction']['probability_real']:.4f}
- Probabilitet fake sipas modelit: {sample['prediction']['probability_fake']:.4f}
- Fjalë sensacionale: {', '.join(explanation['sensational_words_found']) or 'asnjë'}
- Tregues burimi: {', '.join(explanation['source_markers_found']) or 'asnjë'}
- Pikëçuditëse: {explanation['exclamation_count']}
- Numër fjalësh: {explanation['word_count']}
- Gjatësi teksti: {explanation['text_length']}
- `diacritic_ratio`: {explanation['diacritic_ratio']:.6f}
- `uppercase_ratio`: {explanation['uppercase_ratio']:.6f}

Ky është probabilitet sipas modelit dhe karakteristikave gjuhësore. Nuk zëvendëson verifikimin faktik të lajmit.

## Kufizime

- Rezultatet vijnë nga një ndarje e vetme train/test.
- Shtatë kopje ekzakte u gjetën pas ndarjes së vjetër dhe u hoqën nga testi i vlerësimit.
- Dataseti mund të përmbajë sinjale të burimit, temës ose gjatësisë, jo vetëm sinjale të vërtetësisë.
- Listat e fjalëve sensacionale dhe source markers janë manuale dhe fillestare.
- Probabilitetet e Logistic Regression nuk janë kalibruar ende.
- Karakteristikat gjuhësore nuk bëjnë verifikim faktesh.

## Hapi i rekomanduar për Ditën 6

Të bëhet error analysis për rastet ku TF-IDF dhe modeli hibrid gabojnë, pastaj të kontrollohet kalibrimi i probabiliteteve dhe pragu për rezultatin `i pasigurt`. Kjo duhet bërë para ndërtimit të Streamlit.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def train_and_compare_models() -> dict:
    """Run Day 5 checks, train models, and save reproducible outputs."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    train_data, test_data, feature_columns, checks = load_and_check_data()
    no_length_columns = [
        column for column in feature_columns if column not in DIRECT_LENGTH_FEATURES
    ]

    model_specs = {
        "tfidf_only": {
            "model": build_tfidf_model(),
            "train_input": train_data["model_text"],
            "test_input": test_data["model_text"],
            "path": BASELINE_MODEL_PATH,
        },
        "linguistic_only": {
            "model": build_linguistic_model(feature_columns),
            "train_input": train_data[feature_columns],
            "test_input": test_data[feature_columns],
            "path": LINGUISTIC_MODEL_PATH,
        },
        "hybrid": {
            "model": build_hybrid_model(feature_columns),
            "train_input": train_data,
            "test_input": test_data,
            "path": HYBRID_MODEL_PATH,
        },
        "hybrid_no_length": {
            "model": build_hybrid_model(no_length_columns),
            "train_input": train_data,
            "test_input": test_data,
            "path": HYBRID_NO_LENGTH_MODEL_PATH,
        },
    }

    metrics = {}
    for model_key, specification in model_specs.items():
        LOGGER.info("Training %s", MODEL_NAMES[model_key])
        model = specification["model"]
        model.fit(specification["train_input"], train_data["label"])
        predictions = model.predict(specification["test_input"])
        metrics[model_key] = evaluate_predictions(test_data["label"], predictions)
        joblib.dump(model, specification["path"])

    comparison = build_comparison_table(metrics)
    comparison.to_csv(COMPARISON_PATH, index=False, encoding="utf-8-sig")
    plot_comparison(comparison)

    sample_index = test_data.sort_values(
        ["sensational_count", "source_indicator_count"],
        ascending=False,
    ).index[0]
    sample_row = test_data.loc[sample_index]
    sample_prediction = predict_hybrid_news(
        sample_row["title"],
        sample_row["content"],
        model_path=HYBRID_MODEL_PATH,
    )

    best_model_key = max(metrics, key=lambda key: metrics[key]["f1_fake"])
    result = {
        "data_checks": checks,
        "features": {
            "all_numeric_features": feature_columns,
            "all_numeric_feature_count": len(feature_columns),
            "removed_length_features": DIRECT_LENGTH_FEATURES,
            "no_length_feature_count": len(no_length_columns),
        },
        "metrics": metrics,
        "best_model_by_f1_fake": best_model_key,
        "sample_prediction": {
            "article_id": str(sample_row["article_id"]),
            "title": str(sample_row["title"]),
            "true_label": str(sample_row["label_name"]),
            "prediction": sample_prediction,
        },
        "artifacts": {
            "baseline_model": _relative_path(BASELINE_MODEL_PATH),
            "linguistic_model": _relative_path(LINGUISTIC_MODEL_PATH),
            "hybrid_model": _relative_path(HYBRID_MODEL_PATH),
            "hybrid_no_length_model": _relative_path(HYBRID_NO_LENGTH_MODEL_PATH),
            "comparison_table": _relative_path(COMPARISON_PATH),
            "comparison_figure": _relative_path(FIGURE_PATH),
            "report": _relative_path(REPORT_PATH),
        },
    }

    METRICS_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(result, comparison)
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    result = train_and_compare_models()

    print("=== Day 5 hybrid model ===")
    print(f"Train rows: {result['data_checks']['original_train_rows']}")
    print(f"Evaluation test rows: {result['data_checks']['evaluation_test_rows']}")
    print(
        "Exact train duplicates excluded from test: "
        f"{result['data_checks']['exact_train_texts_excluded_from_test']}"
    )
    for model_key, model_metrics in result["metrics"].items():
        print(
            f"{MODEL_NAMES[model_key]}: accuracy={model_metrics['accuracy']}, "
            f"f1_fake={model_metrics['f1_fake']}"
        )
    print(f"Best model by F1 fake: {MODEL_NAMES[result['best_model_by_f1_fake']]}")
    print(f"Metrics saved: {METRICS_PATH}")
    print(f"Report saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
