from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config import LABELS, OUTPUT_DIR
from data_utils import load_project_data


def plot_label_distribution(df, out_path):
    counts = df["label"].value_counts().reindex(LABELS)
    ax = counts.plot(kind="bar", figsize=(9, 4), title="CampusCare Big Dataset — Label Distribution")
    ax.set_xlabel("Label")
    ax.set_ylabel("Number of messages")
    for p in ax.patches:
        ax.annotate(str(int(p.get_height())), (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="bottom", fontsize=8)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_text_length(df, out_path):
    ax = df.boxplot(column="word_count", by="label", figsize=(9, 4), grid=False)
    ax.set_title("Word Count Distribution by Label")
    ax.set_xlabel("Label")
    ax.set_ylabel("Word count")
    plt.suptitle("")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_results(out_path):
    files = [OUTPUT_DIR / "baseline_results_summary.csv", OUTPUT_DIR / "deep_results_summary.csv"]
    frames = [pd.read_csv(f) for f in files if f.exists()]
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)
    test = df[df["split"] == "test"]
    ax = test.set_index("model")[["accuracy", "macro_f1"]].plot(kind="bar", figsize=(9, 4), title="Model Comparison on Test Split")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="bottom", fontsize=8)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_dataset_profile(df):
    profile = pd.DataFrame({
        "metric": ["rows", "labels", "avg_words", "median_words", "min_words", "max_words"],
        "value": [
            len(df),
            df["label"].nunique(),
            round(df["word_count"].mean(), 2),
            round(df["word_count"].median(), 2),
            int(df["word_count"].min()),
            int(df["word_count"].max()),
        ],
    })
    profile.to_csv(OUTPUT_DIR / "dataset_profile.csv", index=False)
    df["label"].value_counts().reindex(LABELS).rename_axis("label").reset_index(name="count").to_csv(
        OUTPUT_DIR / "label_counts.csv", index=False
    )


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = load_project_data(use_full_if_available=True)
    save_dataset_profile(df)
    plot_label_distribution(df, OUTPUT_DIR / "label_distribution.png")
    plot_text_length(df, OUTPUT_DIR / "word_count_by_label.png")
    plot_results(OUTPUT_DIR / "model_comparison.png")
