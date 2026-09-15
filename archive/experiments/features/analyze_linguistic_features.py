# Analizon në mënyrë përshkruese ndryshimet gjuhësore ndërmjet lajmeve real dhe fake.
# Krahason madhësinë, strukturën, pikësimin dhe marker-at gjuhësorë, pastaj ruan
# një tabelë të thjeshtë dhe grafikë në shqip pa ndikuar te trajnimi i modeleve.

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[3]))

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "linguistic_features.csv"
REPORTS_DIR = (
    PROJECT_ROOT / "reports" / "experiment_history" / "04_linguistic_features"
)
FIGURES_DIR = REPORTS_DIR / "figures"
COMPARISON_PATH = REPORTS_DIR / "linguistic_comparison.csv"
PUNCTUATION_PATH = REPORTS_DIR / "punctuation_comparison.csv"

LABELS = ["real", "fake"]
LABEL_NAMES = {"real": "Real", "fake": "Fake"}
COLORS = {"real": "#4C78A8", "fake": "#F58518"}

FEATURE_LABELS = {
    "word_count": "Numri i fjalëve",
    "sentence_count": "Numri i fjalive",
    "avg_sentence_length": "Fjalë mesatare për fjali",
    "title_length": "Gjatësia e titullit",
    "content_length": "Gjatësia e përmbajtjes",
    "exclamation_count": "Pikëçuditëse",
    "question_count": "Pikëpyetje",
    "comma_count": "Presje",
    "quote_count": "Thonjëza",
    "ellipsis_count": "Tri pika",
    "sensational_count": "Shprehje sensacionale",
    "source_indicator_count": "Tregues burimi",
    "uncertainty_count": "Shprehje pasigurie",
    "uppercase_word_ratio": "Fjalë me shkronja të mëdha",
    "uppercase_char_ratio": "Shkronja të mëdha",
    "diacritic_ratio": "Përdorimi i ë/ç",
}

KEY_FEATURES = list(FEATURE_LABELS)
PUNCTUATION_COLUMNS = [
    "comma_count",
    "quote_count",
    "exclamation_count",
    "question_count",
    "ellipsis_count",
]


def load_features() -> pd.DataFrame:
    features = pd.read_csv(
        FEATURES_PATH,
        encoding="utf-8-sig",
        keep_default_na=False,
    )
    required_columns = {"label_name", *KEY_FEATURES}
    missing_columns = sorted(required_columns.difference(features.columns))
    if missing_columns:
        raise ValueError(f"Mungojnë kolonat e analizës: {missing_columns}")
    if set(features["label_name"]) != set(LABELS):
        raise ValueError("Dataset-i duhet të përmbajë klasat real dhe fake.")
    return features


def compare_features(
    features: pd.DataFrame,
    feature_names: list[str] = KEY_FEATURES,
) -> pd.DataFrame:
    rows = []
    for feature in feature_names:
        real = features.loc[features["label_name"].eq("real"), feature].astype(float)
        fake = features.loc[features["label_name"].eq("fake"), feature].astype(float)
        real_mean = float(real.mean())
        fake_mean = float(fake.mean())
        difference = fake_mean - real_mean
        higher_class = "fake" if difference > 0 else "real" if difference < 0 else "barabartë"
        rows.append(
            {
                "feature": feature,
                "pershkrimi": FEATURE_LABELS.get(feature, feature),
                "mesatarja_te_lajmet_real": round(real_mean, 4),
                "mesatarja_te_lajmet_fake": round(fake_mean, 4),
                "diferenca_fake_minus_real": round(difference, 4),
                "mesatarja_me_e_larte_te": higher_class,
            }
        )
    return pd.DataFrame(rows)


def compare_punctuation(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in PUNCTUATION_COLUMNS:
        values = {}
        for label in LABELS:
            subset = features.loc[features["label_name"].eq(label)]
            values[label] = float(
                subset[column].sum() / subset["word_count"].sum() * 100
            )
        difference = values["fake"] - values["real"]
        higher_class = "fake" if difference > 0 else "real" if difference < 0 else "barabartë"
        rows.append(
            {
                "shenja": FEATURE_LABELS[column],
                "per_100_fjale_real": round(values["real"], 4),
                "per_100_fjale_fake": round(values["fake"], 4),
                "diferenca_fake_minus_real": round(difference, 4),
                "me_e_shpeshte_te": higher_class,
            }
        )
    return pd.DataFrame(rows)


def _save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_class_distribution(features: pd.DataFrame) -> None:
    counts = features["label_name"].value_counts().reindex(LABELS)
    total = int(counts.sum())
    figure, axis = plt.subplots(figsize=(7, 5))
    bars = axis.bar(
        [LABEL_NAMES[label] for label in LABELS],
        counts,
        color=[COLORS[label] for label in LABELS],
    )
    axis.set_title("Dataset-i është i balancuar mes lajmeve real dhe fake")
    axis.set_xlabel("Lloji i lajmit")
    axis.set_ylabel("Numri i artikujve")
    axis.bar_label(
        bars,
        labels=[f"{count:,}\n({count / total:.1%})" for count in counts],
        padding=4,
    )
    axis.set_ylim(0, counts.max() * 1.15)
    figure.savefig(
        FIGURES_DIR / "shperndarja_lajmeve_real_fake.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_word_count_distribution(features: pd.DataFrame) -> None:
    upper_limit = float(features["word_count"].quantile(0.98))
    plt.figure(figsize=(10, 6))
    for label in LABELS:
        values = features.loc[features["label_name"].eq(label), "word_count"].clip(
            upper=upper_limit
        )
        plt.hist(
            values,
            bins=35,
            alpha=0.58,
            label=LABEL_NAMES[label],
            color=COLORS[label],
        )
        mean_value = float(
            features.loc[features["label_name"].eq(label), "word_count"].mean()
        )
        plt.axvline(
            mean_value,
            color=COLORS[label],
            linestyle="--",
            linewidth=2,
            label=f"Mesatarja {LABEL_NAMES[label]}: {mean_value:.1f}",
        )
    plt.title("Lajmet real përmbajnë mesatarisht më shumë fjalë")
    plt.xlabel("Numri i fjalëve (i kufizuar te percentili 98)")
    plt.ylabel("Numri i artikujve")
    plt.legend(title="Lloji i lajmit")
    _save_figure(FIGURES_DIR / "shperndarja_gjatesise_lajmeve.png")


def plot_length_comparison(features: pd.DataFrame) -> None:
    specifications = [
        ("word_count", "Numri mesatar i fjalëve për artikull", "Fjalë"),
        ("sentence_count", "Numri mesatar i fjalive për artikull", "Fjali"),
        ("title_length", "Gjatësia mesatare e titullit", "Karaktere"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for axis, (feature, title, ylabel) in zip(axes, specifications):
        values = features.groupby("label_name")[feature].mean().reindex(LABELS)
        bars = axis.bar(
            [LABEL_NAMES[label] for label in LABELS],
            values,
            color=[COLORS[label] for label in LABELS],
        )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.bar_label(bars, fmt="%.1f", padding=3)
        axis.set_ylim(0, values.max() * 1.18)
    figure.suptitle(
        "Lajmet real janë më të gjata, ndërsa lajmet fake kanë tituj më të gjatë",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(
        FIGURES_DIR / "krahasimi_gjatesise_dhe_struktures.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_marker_frequency(features: pd.DataFrame) -> None:
    marker_columns = [
        "sensational_count",
        "source_indicator_count",
        "uncertainty_count",
    ]
    marker_labels = [FEATURE_LABELS[column] for column in marker_columns]
    positions = range(len(marker_columns))
    width = 0.36

    plt.figure(figsize=(11, 6))
    for index, label in enumerate(LABELS):
        subset = features.loc[features["label_name"].eq(label)]
        frequencies = [
            subset[column].sum() / subset["word_count"].sum() * 100
            for column in marker_columns
        ]
        offsets = [position + (index - 0.5) * width for position in positions]
        bars = plt.bar(
            offsets,
            frequencies,
            width=width,
            label=LABEL_NAMES[label],
            color=COLORS[label],
        )
        plt.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    plt.title("Burimet dhe pasiguria përdoren më shpesh te lajmet real")
    plt.xlabel("Sinjali gjuhësor")
    plt.ylabel("Përdorime për 100 fjalë")
    plt.xticks(list(positions), marker_labels, rotation=18, ha="right")
    plt.legend(title="Lloji i lajmit")
    _save_figure(FIGURES_DIR / "krahasimi_sinjaleve_gjuhesore.png")


def plot_punctuation_frequency(features: pd.DataFrame) -> None:
    groups = [
        (["comma_count", "quote_count"], "Presjet dhe thonjëzat"),
        (
            ["exclamation_count", "question_count", "ellipsis_count"],
            "Pikësimi shprehës",
        ),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    width = 0.36

    for axis, (columns, title) in zip(axes, groups):
        positions = range(len(columns))
        for index, label in enumerate(LABELS):
            subset = features.loc[features["label_name"].eq(label)]
            frequencies = [
                subset[column].sum() / subset["word_count"].sum() * 100
                for column in columns
            ]
            offsets = [position + (index - 0.5) * width for position in positions]
            bars = axis.bar(
                offsets,
                frequencies,
                width=width,
                label=LABEL_NAMES[label],
                color=COLORS[label],
            )
            axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
        axis.set_title(title)
        axis.set_xlabel("Shenja e pikësimit")
        axis.set_ylabel("Përdorime për 100 fjalë")
        axis.set_xticks(list(positions), [FEATURE_LABELS[column] for column in columns])
        axis.legend(title="Lloji i lajmit")

    figure.suptitle(
        "Real përdor më shumë presje e thonjëza; fake më shumë pikësim shprehës",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(
        FIGURES_DIR / "krahasimi_shenjave_te_pikesimit.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_style_ratios(features: pd.DataFrame) -> None:
    ratio_columns = [
        "uppercase_word_ratio",
        "uppercase_char_ratio",
        "diacritic_ratio",
    ]
    ratio_labels = [FEATURE_LABELS[column] for column in ratio_columns]
    positions = range(len(ratio_columns))
    width = 0.36

    plt.figure(figsize=(9, 5.5))
    for index, label in enumerate(LABELS):
        values = (
            features.loc[features["label_name"].eq(label), ratio_columns].mean()
            * 100
        )
        offsets = [position + (index - 0.5) * width for position in positions]
        bars = plt.bar(
            offsets,
            values,
            width=width,
            label=LABEL_NAMES[label],
            color=COLORS[label],
        )
        plt.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    plt.title("Fake përdor më shumë kapitalizim; real përdor më shumë ë/ç")
    plt.xlabel("Karakteristika gjuhësore")
    plt.ylabel("Përqindja mesatare")
    plt.xticks(list(positions), ratio_labels, rotation=12, ha="right")
    plt.legend(title="Lloji i lajmit")
    _save_figure(FIGURES_DIR / "krahasimi_kapitalizimit_dhe_diacritikave.png")


def create_figures(features: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_class_distribution(features)
    plot_word_count_distribution(features)
    plot_length_comparison(features)
    plot_marker_frequency(features)
    plot_punctuation_frequency(features)
    plot_style_ratios(features)


def run_analysis() -> pd.DataFrame:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    features = load_features()
    comparison = compare_features(features)
    punctuation = compare_punctuation(features)
    comparison.to_csv(COMPARISON_PATH, index=False, encoding="utf-8-sig")
    punctuation.to_csv(PUNCTUATION_PATH, index=False, encoding="utf-8-sig")
    create_figures(features)
    return comparison


def main() -> None:
    comparison = run_analysis()
    print("=== Analiza gjuhësore e dataset-it ===")
    print(f"Artikuj të analizuar: {len(load_features()):,}")
    print(f"Karakteristika të krahasuara: {len(comparison)}")
    print(f"Tabela u ruajt te: {COMPARISON_PATH}")
    print(f"Krahasimi i pikësimit u ruajt te: {PUNCTUATION_PATH}")
    print(f"Grafikët u ruajtën te: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
