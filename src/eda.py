"""Exploratory Data Analysis for UCI Air Quality Dataset.

Generates summary statistics and publication-quality visualisation plots
saved to the ``figures/`` directory.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.data_loader import get_dataset
from src.preprocessing import TARGET, FEATURE_FIELDS, clean_dataset

FIGURES_DIR = ROOT / "figures"


def summary_statistics(df: pd.DataFrame) -> None:
    """Print dataset summary statistics to stdout."""
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Shape: {df.shape[0]} samples Ã— {df.shape[1]} features")
    print(f"\nMissing values per column:\n{df.isnull().sum()}")
    print(f"\nDescriptive statistics:\n{df[FEATURE_FIELDS + [TARGET]].describe().round(2)}")


def plot_correlation_heatmap(df: pd.DataFrame, out: Path) -> None:
    """Pearson correlation heat-map of sensor and weather features."""
    cols = FEATURE_FIELDS + [TARGET]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "correlation_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  âœ“ Saved {out / 'correlation_heatmap.png'}")


def plot_target_distribution(df: pd.DataFrame, out: Path) -> None:
    """Histogram + KDE of the target variable CO(GT)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[TARGET].dropna(), bins=50, kde=True, color="#4C72B0", ax=ax)
    ax.set_xlabel("CO(GT) Concentration (mg/mÂ³)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Distribution of Target Variable â€” CO(GT)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "target_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  âœ“ Saved {out / 'target_distribution.png'}")


def plot_time_series(df: pd.DataFrame, out: Path) -> None:
    """Time-series plot of CO(GT) concentration."""
    if "Date_Time" not in df.columns:
        print("  âš  Skipping time-series plot (no Date_Time column)")
        return
    ts = df.dropna(subset=[TARGET, "Date_Time"]).sort_values("Date_Time")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ts["Date_Time"], ts[TARGET], linewidth=0.4, color="#4C72B0", alpha=0.7)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("CO(GT) mg/mÂ³", fontsize=12)
    ax.set_title("CO(GT) Concentration Over Time", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "co_time_series.png", dpi=150)
    plt.close(fig)
    print(f"  âœ“ Saved {out / 'co_time_series.png'}")


def plot_sensor_boxplots(df: pd.DataFrame, out: Path) -> None:
    """Box-plots for all sensor features (normalised for readability)."""
    from sklearn.preprocessing import StandardScaler
    sensor_cols = [c for c in FEATURE_FIELDS if c.startswith("PT08")]
    data = df[sensor_cols].dropna()
    scaled = pd.DataFrame(StandardScaler().fit_transform(data), columns=sensor_cols)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=scaled, palette="Set2", ax=ax)
    ax.set_title("Sensor Feature Distributions (Standardised)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Standardised Value", fontsize=12)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out / "sensor_boxplots.png", dpi=150)
    plt.close(fig)
    print(f"  âœ“ Saved {out / 'sensor_boxplots.png'}")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading dataset â€¦")
    df = get_dataset(ROOT / "data")
    df = clean_dataset(df)
    summary_statistics(df)
    print("\nGenerating figures:")
    plot_correlation_heatmap(df, FIGURES_DIR)
    plot_target_distribution(df, FIGURES_DIR)
    plot_time_series(df, FIGURES_DIR)
    plot_sensor_boxplots(df, FIGURES_DIR)
    print("\nEDA complete âœ“")


if __name__ == "__main__":
    main()
