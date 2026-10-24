"""Exploratory Data Analysis (EDA) Module.

Generates statistical summaries, checks class distributions and lengths,
identifies common vocabulary patterns, and saves publication-grade figures.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless / script execution
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocessing import clean_html, normalize_whitespace
from src.data_loader import load_processed_splits, LABEL_MAPPING

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("eda")

# Global aesthetic style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "figure.titlesize": 14
})


def compute_text_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Computes character length and word count for each review.

    Args:
        df: DataFrame containing a 'review' column.

    Returns:
        DataFrame augmented with 'char_length' and 'word_count' columns.
    """
    df = df.copy()
    cleaned_series = df["review"].astype(str).apply(lambda t: normalize_whitespace(clean_html(t)))
    df["char_length"] = cleaned_series.str.len()
    df["word_count"] = cleaned_series.str.split().apply(len)
    return df


def plot_class_distribution(
    df: pd.DataFrame,
    output_path: Path,
    title_suffix: str = "Training Set"
) -> None:
    """Plots and saves class frequency bar and percentage distribution.

    Args:
        df: DataFrame containing 'label' column.
        output_path: File path to save the generated figure.
        title_suffix: Title label identifying partition.
    """
    counts = df["label"].value_counts().rename(index=LABEL_MAPPING)
    total = len(df)

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(counts.index, counts.values, color=["#d9534f", "#5cb85c"], width=0.5, edgecolor="black")

    for bar in bars:
        height = bar.get_height()
        pct = (height / total) * 100
        ax.annotate(
            f"{height:,}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center", va="bottom",
            fontweight="bold"
        )

    ax.set_title(f"Sentiment Class Distribution ({title_suffix})", pad=15)
    ax.set_xlabel("Sentiment Class")
    ax.set_ylabel("Number of Reviews")
    ax.set_ylim(0, max(counts.values) * 1.18)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved class distribution plot to {output_path}.")


def plot_review_length_distribution(
    df: pd.DataFrame,
    output_path: Path
) -> None:
    """Plots KDE and histogram of review word counts across sentiment classes.

    Args:
        df: DataFrame with 'word_count' and 'label' columns.
        output_path: File path to save the figure.
    """
    df_plot = df.copy()
    df_plot["Sentiment"] = df_plot["label"].map(LABEL_MAPPING)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Word Count Distribution (Log scale or truncated at 99th percentile for clarity)
    p99 = df_plot["word_count"].quantile(0.99)
    sns.histplot(
        data=df_plot[df_plot["word_count"] <= p99],
        x="word_count",
        hue="Sentiment",
        kde=True,
        element="step",
        palette={"Negative": "#d9534f", "Positive": "#5cb85c"},
        ax=axes[0]
    )
    axes[0].set_title("Word Count Distribution by Sentiment (<= 99th percentile)")
    axes[0].set_xlabel("Word Count per Review")
    axes[0].set_ylabel("Count")

    # 2. Box Plot of Lengths
    sns.boxplot(
        data=df_plot,
        x="Sentiment",
        y="word_count",
        palette={"Negative": "#d9534f", "Positive": "#5cb85c"},
        showfliers=False,
        ax=axes[1]
    )
    axes[1].set_title("Word Count Box Plot (Excluding Outliers)")
    axes[1].set_xlabel("Sentiment")
    axes[1].set_ylabel("Word Count")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved review length distribution plot to {output_path}.")


def plot_top_words_by_class(
    df: pd.DataFrame,
    output_path: Path,
    top_n: int = 15
) -> None:
    """Computes and plots the most frequent words for Negative vs Positive reviews.

    Args:
        df: DataFrame with 'review' and 'label' columns.
        output_path: File path to save the generated plot.
        top_n: Number of top words to visualize.
    """
    # Simple list of English stop words to reveal meaningful sentiment vocabulary
    common_stopwords = {
        "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with",
        "is", "was", "are", "were", "it", "this", "that", "as", "by", "from", "be",
        "have", "has", "had", "i", "you", "he", "she", "they", "we", "movie", "film",
        "one", "all", "so", "there", "his", "her", "my", "about", "who", "which",
        "what", "when", "just", "if", "out", "up", "even", "more", "would", "some"
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, (label_val, label_name, color) in enumerate([
        (0, "Negative", "#d9534f"),
        (1, "Positive", "#5cb85c")
    ]):
        class_reviews = df[df["label"] == label_val]["review"].astype(str)
        tokens = []
        for rev in class_reviews:
            words = normalize_whitespace(clean_html(rev.lower())).split()
            tokens.extend([w.strip(".,!?:;\"'()[]{}") for w in words if len(w) > 2])

        filtered_tokens = [w for w in tokens if w not in common_stopwords and w.isalpha()]
        most_common = Counter(filtered_tokens).most_common(top_n)

        words, freqs = zip(*most_common) if most_common else ([], [])
        y_pos = np.arange(len(words))

        axes[idx].barh(y_pos, freqs, color=color, edgecolor="black", alpha=0.85)
        axes[idx].set_yticks(y_pos)
        axes[idx].set_yticklabels(words)
        axes[idx].invert_yaxis()  # Top word at the top
        axes[idx].set_title(f"Top {top_n} Characteristic Words: {label_name}")
        axes[idx].set_xlabel("Frequency Count")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved top words plot to {output_path}.")


def run_eda(
    data_dir: str = "data/processed",
    figures_dir: str = "reports/figures"
) -> Dict[str, Any]:
    """Executes full Exploratory Data Analysis and saves summary charts.

    Args:
        data_dir: Path to processed dataset splits.
        figures_dir: Path to save generated charts.

    Returns:
        Dictionary of EDA summary metrics.
    """
    logger.info(f"Initiating EDA using splits from {data_dir}...")
    train_df, val_df, test_df = load_processed_splits(data_dir)

    train_df = compute_text_metrics(train_df)
    fig_path = Path(figures_dir)
    fig_path.mkdir(parents=True, exist_ok=True)

    # 1. Class Distribution
    plot_class_distribution(train_df, fig_path / "class_distribution.png", title_suffix="Train Split")

    # 2. Length Distribution
    plot_review_length_distribution(train_df, fig_path / "review_length_distribution.png")

    # 3. Top Words
    plot_top_words_by_class(train_df, fig_path / "top_words_by_sentiment.png", top_n=15)

    # Summary Statistics
    summary = {
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "mean_word_count_train": float(train_df["word_count"].mean()),
        "median_word_count_train": float(train_df["word_count"].median()),
        "std_word_count_train": float(train_df["word_count"].std()),
        "max_word_count_train": int(train_df["word_count"].max()),
        "min_word_count_train": int(train_df["word_count"].min()),
        "negative_class_pct": float((train_df["label"] == 0).mean() * 100),
        "positive_class_pct": float((train_df["label"] == 1).mean() * 100),
        "mean_words_negative": float(train_df[train_df["label"] == 0]["word_count"].mean()),
        "mean_words_positive": float(train_df[train_df["label"] == 1]["word_count"].mean()),
    }

    logger.info(
        f"EDA Complete:\n"
        f"  Total Training Reviews: {summary['train_samples']:,}\n"
        f"  Balance: {summary['negative_class_pct']:.1f}% Negative / {summary['positive_class_pct']:.1f}% Positive\n"
        f"  Mean Word Count: {summary['mean_word_count_train']:.1f} (Neg: {summary['mean_words_negative']:.1f}, Pos: {summary['mean_words_positive']:.1f})\n"
        f"  Median Word Count: {summary['median_word_count_train']:.1f}\n"
        f"  Saved figures to: {fig_path.resolve()}"
    )

    return summary


if __name__ == "__main__":
    run_eda()
