"""Quality and train-overlap checks for the Day 10 external dataset."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "external" / "external_news.csv"
DEFAULT_TRAINING_PATH = PROJECT_ROOT / "data" / "interim" / "articles_clean.csv"
DEFAULT_METADATA_ROOT = (
    PROJECT_ROOT / "data" / "raw" / "alb-fake-news-corpus" / "full_texts"
)
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "day10_external_dataset_audit.json"
DEFAULT_SIMILARITY_PATH = PROJECT_ROOT / "reports" / "day10_external_similarity_review.csv"

REQUIRED_COLUMNS = [
    "external_id",
    "title",
    "content",
    "label",
    "source",
    "url",
    "published_date",
    "label_evidence",
    "evidence_url",
    "topic",
    "content_origin",
    "review_status",
]

TEXT_COLUMNS = ["title", "content", "source", "label_evidence"]
ALLOWED_LABELS = {"real", "fake"}
ALLOWED_TOPICS = {"politikë", "shëndetësi", "ekonomi", "sociale", "teknologji"}


def normalize_for_comparison(value: object) -> str:
    """Normalize text for exact duplicate checks without changing the stored data."""
    text = unicodedata.normalize("NFC", str(value or "")).casefold()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def combine_text(dataframe: pd.DataFrame) -> pd.Series:
    """Combine title and content in the same order used by the project."""
    title = dataframe["title"].fillna("").astype(str).str.strip()
    content = dataframe["content"].fillna("").astype(str).str.strip()
    return (title + " " + content).str.strip()


def is_http_url(value: object) -> bool:
    """Return True for a syntactically valid HTTP or HTTPS URL."""
    parsed = urlsplit(str(value).strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def normalize_url(value: object) -> str:
    """Normalize an HTTP URL for exact comparisons."""
    parsed = urlsplit(str(value).strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit(
        (parsed.scheme.lower(), parsed.netloc.lower(), path, parsed.query, "")
    )


def load_corpus_urls(metadata_root: Path) -> tuple[set[str], int]:
    """Read source URLs from the true/fake metadata files in the raw corpus."""
    paths: list[Path] = []
    for directory_name in ("true-meta-information", "fake-meta-information"):
        directory = metadata_root / directory_name
        if directory.exists():
            paths.extend(directory.glob("*.txt"))

    # Thousands of small metadata files are considerably faster to read in parallel on Windows.
    with ThreadPoolExecutor(max_workers=16) as executor:
        url_groups = executor.map(_read_urls_from_metadata_file, paths)
        urls = {url for group in url_groups for url in group}
    return urls, len(paths)


def _read_urls_from_metadata_file(path: Path) -> set[str]:
    """Extract normalized URLs from one raw metadata file."""
    urls: set[str] = set()
    with path.open(encoding="utf-8", errors="replace") as metadata_file:
        for line in metadata_file:
            normalized = normalize_url(line)
            if normalized:
                urls.add(normalized)
    return urls


def duplicate_summary(values: pd.Series) -> dict[str, int]:
    """Count rows and groups involved in non-empty duplicates."""
    normalized = values.fillna("").astype(str).map(normalize_for_comparison)
    normalized = normalized[normalized.ne("")]
    counts = normalized.value_counts()
    return {
        "rows": int(normalized.duplicated(keep=False).sum()),
        "groups": int(counts.gt(1).sum()),
    }


def _nearest_similarities(
    external_text: pd.Series,
    training_text: pd.Series,
    max_features: int = 50_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return nearest training and external text similarities using char TF-IDF."""
    all_text = pd.concat([external_text, training_text], ignore_index=True)
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=max_features,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(all_text)
    external_matrix = matrix[: len(external_text)]
    training_matrix = matrix[len(external_text) :]

    train_similarity = linear_kernel(external_matrix, training_matrix)
    nearest_training_index = train_similarity.argmax(axis=1)
    nearest_training_score = train_similarity[
        np.arange(len(external_text)), nearest_training_index
    ]

    external_similarity = linear_kernel(external_matrix, external_matrix)
    np.fill_diagonal(external_similarity, -1.0)
    nearest_external_index = external_similarity.argmax(axis=1)
    nearest_external_score = external_similarity[
        np.arange(len(external_text)), nearest_external_index
    ]
    return (
        nearest_training_index,
        nearest_training_score,
        nearest_external_index,
        nearest_external_score,
    )


def build_similarity_review(
    external: pd.DataFrame,
    training: pd.DataFrame,
    similarity_threshold: float = 0.90,
) -> pd.DataFrame:
    """Compare external rows with training rows and with each other."""
    required_training = {"article_id", "title", "content"}
    missing_training = sorted(required_training - set(training.columns))
    if missing_training:
        raise ValueError(f"Training dataset is missing columns: {missing_training}")

    external_text = combine_text(external)
    training_text = combine_text(training)

    external_title_normalized = external["title"].map(normalize_for_comparison)
    external_content_normalized = external["content"].map(normalize_for_comparison)
    external_text_normalized = external_text.map(normalize_for_comparison)
    training_title_normalized = set(training["title"].map(normalize_for_comparison))
    training_content_normalized = set(training["content"].map(normalize_for_comparison))
    training_text_normalized = set(training_text.map(normalize_for_comparison))

    (
        nearest_training_index,
        nearest_training_score,
        nearest_external_index,
        nearest_external_score,
    ) = _nearest_similarities(external_text, training_text)

    nearest_training = training.iloc[nearest_training_index].reset_index(drop=True)
    nearest_external_ids = external.iloc[nearest_external_index]["external_id"].reset_index(drop=True)

    review = external[["external_id", "label", "topic", "title"]].copy()
    review["exact_training_title"] = external_title_normalized.isin(
        training_title_normalized
    )
    review["exact_training_content"] = external_content_normalized.isin(
        training_content_normalized
    )
    review["exact_training_text"] = external_text_normalized.isin(training_text_normalized)
    review["max_training_similarity"] = np.round(nearest_training_score, 6)
    review["nearest_training_article_id"] = nearest_training["article_id"].astype(str)
    review["nearest_training_title"] = nearest_training["title"].fillna("").astype(str)
    review["high_similarity_to_training"] = (
        review["max_training_similarity"] >= similarity_threshold
    )
    review["max_external_similarity"] = np.round(nearest_external_score, 6)
    review["nearest_external_id"] = nearest_external_ids
    review["high_similarity_within_external"] = (
        review["max_external_similarity"] >= similarity_threshold
    )
    return review


def validate_external_dataset(
    external: pd.DataFrame,
    training: pd.DataFrame | None = None,
    training_urls: set[str] | None = None,
    metadata_files_checked: int = 0,
    short_content_words: int = 25,
    similarity_threshold: float = 0.90,
    print_report: bool = True,
) -> tuple[dict, pd.DataFrame]:
    """Validate the external dataset and return a summary plus similarity details."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in external.columns]
    if missing_columns:
        raise ValueError(f"External dataset is missing columns: {missing_columns}")

    data = external.copy()
    for column in REQUIRED_COLUMNS:
        data[column] = data[column].fillna("")

    missing_values = {
        column: int(data[column].astype(str).str.strip().eq("").sum())
        for column in REQUIRED_COLUMNS
    }
    missing_values = {column: count for column, count in missing_values.items() if count}

    invalid_labels = sorted(set(data["label"].astype(str)) - ALLOWED_LABELS)
    invalid_topics = sorted(set(data["topic"].astype(str)) - ALLOWED_TOPICS)
    unapproved_ids = data.loc[
        data["review_status"].astype(str).ne("approved"), "external_id"
    ].astype(str).tolist()

    parsed_dates = pd.to_datetime(data["published_date"], format="%Y-%m-%d", errors="coerce")
    invalid_date_ids = data.loc[parsed_dates.isna(), "external_id"].astype(str).tolist()
    invalid_url_ids = data.loc[~data["url"].map(is_http_url), "external_id"].astype(str).tolist()
    invalid_evidence_url_ids = data.loc[
        ~data["evidence_url"].map(is_http_url), "external_id"
    ].astype(str).tolist()

    url_overlap_summary: dict[str, object] = {"checked": False}
    if training_urls is not None:
        normalized_training_urls = {normalize_url(url) for url in training_urls}
        normalized_training_urls.discard("")
        source_url_match = data["url"].map(normalize_url).isin(normalized_training_urls)
        evidence_url_match = data["evidence_url"].map(normalize_url).isin(
            normalized_training_urls
        )
        url_match = source_url_match | evidence_url_match
        url_overlap_summary = {
            "checked": True,
            "metadata_files_checked": int(metadata_files_checked),
            "unique_training_urls": int(len(normalized_training_urls)),
            "exact_url_matches": int(url_match.sum()),
            "exact_url_match_ids": data.loc[url_match, "external_id"].astype(str).tolist(),
        }

    content_word_count = data["content"].astype(str).str.split().str.len()
    short_ids = data.loc[
        content_word_count.lt(short_content_words), "external_id"
    ].astype(str).tolist()
    weak_evidence_ids = data.loc[
        data["label_evidence"].astype(str).str.split().str.len().lt(8), "external_id"
    ].astype(str).tolist()

    non_nfc_ids: set[str] = set()
    replacement_character_ids: set[str] = set()
    for column in TEXT_COLUMNS:
        for external_id, value in zip(data["external_id"], data[column].astype(str)):
            if value != unicodedata.normalize("NFC", value):
                non_nfc_ids.add(str(external_id))
            if "\ufffd" in value:
                replacement_character_ids.add(str(external_id))

    duplicate_ids = duplicate_summary(data["external_id"])
    duplicate_urls = duplicate_summary(data["url"])
    duplicate_titles = duplicate_summary(data["title"])
    duplicate_contents = duplicate_summary(data["content"])

    similarity_review = pd.DataFrame()
    overlap_summary: dict[str, object] = {"checked": False}
    if training is not None and not training.empty:
        similarity_review = build_similarity_review(
            data,
            training,
            similarity_threshold=similarity_threshold,
        )
        exact_mask = similarity_review[
            ["exact_training_title", "exact_training_content", "exact_training_text"]
        ].any(axis=1)
        high_training = similarity_review["high_similarity_to_training"]
        high_external = similarity_review["high_similarity_within_external"]
        overlap_summary = {
            "checked": True,
            "training_articles": int(len(training)),
            "similarity_threshold": similarity_threshold,
            "exact_training_matches": int(exact_mask.sum()),
            "exact_training_match_ids": similarity_review.loc[
                exact_mask, "external_id"
            ].tolist(),
            "high_similarity_to_training": int(high_training.sum()),
            "high_similarity_to_training_ids": similarity_review.loc[
                high_training, "external_id"
            ].tolist(),
            "high_similarity_within_external": int(high_external.sum()),
            "high_similarity_within_external_ids": similarity_review.loc[
                high_external, "external_id"
            ].tolist(),
            "max_training_similarity": round(
                float(similarity_review["max_training_similarity"].max()), 6
            ),
            "mean_max_training_similarity": round(
                float(similarity_review["max_training_similarity"].mean()), 6
            ),
        }

    label_counts = data["label"].value_counts().sort_index()
    topic_counts = data["topic"].value_counts().sort_index()
    topic_label_table = pd.crosstab(data["topic"], data["label"])

    blockers: list[str] = []
    checks = {
        "missing required values": bool(missing_values),
        "invalid labels": bool(invalid_labels),
        "invalid topics": bool(invalid_topics),
        "unapproved rows": bool(unapproved_ids),
        "invalid dates": bool(invalid_date_ids),
        "invalid source URLs": bool(invalid_url_ids),
        "invalid evidence URLs": bool(invalid_evidence_url_ids),
        "short content": bool(short_ids),
        "weak labeling evidence": bool(weak_evidence_ids),
        "non-NFC Unicode": bool(non_nfc_ids),
        "Unicode replacement characters": bool(replacement_character_ids),
        "duplicate IDs": duplicate_ids["rows"] > 0,
        "duplicate URLs": duplicate_urls["rows"] > 0,
        "duplicate titles": duplicate_titles["rows"] > 0,
        "duplicate contents": duplicate_contents["rows"] > 0,
    }
    if overlap_summary.get("checked"):
        checks["exact matches with training data"] = bool(
            overlap_summary["exact_training_matches"]
        )
        checks["high similarity with training data"] = bool(
            overlap_summary["high_similarity_to_training"]
        )
        checks["high similarity inside external data"] = bool(
            overlap_summary["high_similarity_within_external"]
        )
    if url_overlap_summary.get("checked"):
        checks["exact URL matches with raw corpus metadata"] = bool(
            url_overlap_summary["exact_url_matches"]
        )
    blockers.extend(name for name, failed in checks.items() if failed)

    summary = {
        "total_articles": int(len(data)),
        "label_counts": {str(label): int(count) for label, count in label_counts.items()},
        "topic_counts": {str(topic): int(count) for topic, count in topic_counts.items()},
        "topic_label_counts": {
            str(topic): {
                str(label): int(topic_label_table.loc[topic, label])
                for label in topic_label_table.columns
            }
            for topic in topic_label_table.index
        },
        "source_counts": {
            str(source): int(count)
            for source, count in data["source"].value_counts().items()
        },
        "missing_values": missing_values,
        "invalid_labels": invalid_labels,
        "invalid_topics": invalid_topics,
        "unapproved_ids": unapproved_ids,
        "invalid_date_ids": invalid_date_ids,
        "invalid_url_ids": invalid_url_ids,
        "invalid_evidence_url_ids": invalid_evidence_url_ids,
        "short_content_threshold_words": short_content_words,
        "short_content_ids": short_ids,
        "weak_evidence_ids": weak_evidence_ids,
        "unicode": {
            "non_nfc_ids": sorted(non_nfc_ids),
            "replacement_character_ids": sorted(replacement_character_ids),
            "albanian_diacritic_rows": int(
                combine_text(data).str.contains(r"[ëçËÇ]", regex=True).sum()
            ),
        },
        "duplicates": {
            "external_id": duplicate_ids,
            "url": duplicate_urls,
            "title": duplicate_titles,
            "content": duplicate_contents,
        },
        "content_word_stats": {
            "min": int(content_word_count.min()),
            "max": int(content_word_count.max()),
            "mean": round(float(content_word_count.mean()), 2),
            "median": round(float(content_word_count.median()), 2),
        },
        "training_overlap": overlap_summary,
        "training_url_overlap": url_overlap_summary,
        "blocking_problems": blockers,
        "ready_for_external_evaluation": not blockers,
    }

    if print_report:
        print_validation_report(summary)
    return summary, similarity_review


def print_validation_report(summary: dict) -> None:
    """Print a compact, readable quality report."""
    print("=== Day 10 external dataset validation ===")
    print(f"Articles: {summary['total_articles']}")
    print(f"Labels: {summary['label_counts']}")
    print(f"Topics: {summary['topic_counts']}")
    print(f"Missing values: {summary['missing_values']}")
    print(f"Short content IDs: {summary['short_content_ids']}")
    print(f"Duplicate checks: {summary['duplicates']}")
    overlap = summary["training_overlap"]
    if overlap.get("checked"):
        print(f"Exact matches with training data: {overlap['exact_training_matches']}")
        print(
            "Near matches with training data: "
            f"{overlap['high_similarity_to_training']} "
            f"(threshold={overlap['similarity_threshold']})"
        )
        print(f"Highest training similarity: {overlap['max_training_similarity']:.4f}")
    url_overlap = summary["training_url_overlap"]
    if url_overlap.get("checked"):
        print(
            "Exact URL matches with raw corpus metadata: "
            f"{url_overlap['exact_url_matches']} "
            f"({url_overlap['unique_training_urls']} URLs checked)"
        )
    print(f"Blocking problems: {summary['blocking_problems']}")
    print(f"Ready for external evaluation: {summary['ready_for_external_evaluation']}")


def parse_args() -> argparse.Namespace:
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--training", type=Path, default=DEFAULT_TRAINING_PATH)
    parser.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--similarity-report", type=Path, default=DEFAULT_SIMILARITY_PATH)
    parser.add_argument("--short-content-words", type=int, default=25)
    parser.add_argument("--similarity-threshold", type=float, default=0.90)
    return parser.parse_args()


def main() -> int:
    """Run validation and save the Day 10 audit artifacts."""
    args = parse_args()
    if not args.dataset.exists():
        raise FileNotFoundError(f"External dataset not found: {args.dataset}")
    if not args.training.exists():
        raise FileNotFoundError(f"Training dataset not found: {args.training}")
    if not args.metadata_root.exists():
        raise FileNotFoundError(f"Raw metadata directory not found: {args.metadata_root}")

    external = pd.read_csv(args.dataset, encoding="utf-8")
    training = pd.read_csv(args.training, encoding="utf-8")
    training_urls, metadata_files_checked = load_corpus_urls(args.metadata_root)
    summary, similarity_review = validate_external_dataset(
        external,
        training,
        training_urls=training_urls,
        metadata_files_checked=metadata_files_checked,
        short_content_words=args.short_content_words,
        similarity_threshold=args.similarity_threshold,
    )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    similarity_review.to_csv(args.similarity_report, index=False, encoding="utf-8")
    print(f"Audit saved to: {args.report}")
    print(f"Similarity details saved to: {args.similarity_report}")
    return 0 if summary["ready_for_external_evaluation"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
