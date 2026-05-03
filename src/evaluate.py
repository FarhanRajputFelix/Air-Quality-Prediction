"""Model Evaluation & Visualisation.

Loads the saved model and preprocessor, evaluates on a held-out test split,
and generates diagnostic plots saved to ``figures/``.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_loader import get_dataset
from src.preprocessing import preprocess_dataset
from src.model import load_model, load_preprocessor

FIGURES_DIR = ROOT / "figures"


def evaluate_model(y_true, y_pred):
    """Return a dict of standard regression metrics."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "RÂ²": r2_score(y_true, y_pred),
    }


def plot_actual_vs_predicted(y_true, y_pred, out: Path) -> None:
    """Scatter plot: actual vs predicted with identity line."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=8, alpha=0.4, color="#4C72B0")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "--", color="red", linewidth=1.2, label="Ideal (y = x)")
    ax.set_xlabel("Actual CO(GT)", fontsize=12)
    ax.set_ylabel("Predicted CO(GT)", fontsize=12)
    ax.set_title("Actual vs Predicted", fontsize=14, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "actual_vs_predicted.png", dpi=150)
    plt.close(fig)
    print(f"  âœ“ Saved actual_vs_predicted.png")


def plot_residuals(y_true, y_pred, out: Path) -> None:
    """Residual distribution and residual vs predicted."""
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram of residuals
    axes[0].hist(residuals, bins=50, color="#4C72B0", edgecolor="white", alpha=0.85)
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Residual", fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].set_title("Residual Distribution", fontsize=13, fontweight="bold")

    # Residual vs predicted
    axes[1].scatter(y_pred, residuals, s=8, alpha=0.4, color="#4C72B0")
    axes[1].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Predicted CO(GT)", fontsize=12)
    axes[1].set_ylabel("Residual", fontsize=12)
    axes[1].set_title("Residuals vs Predicted", fontsize=13, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out / "residual_analysis.png", dpi=150)
    plt.close(fig)
    print(f"  âœ“ Saved residual_analysis.png")


def plot_feature_importance(model, feature_names, out: Path) -> None:
    """Horizontal bar chart of feature importances from tree-based model."""
    importances = model.feature_importances_
    indices = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(indices)), importances[indices], color="#4C72B0", edgecolor="white")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=10)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title("Feature Importance (Random Forest)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "feature_importance.png", dpi=150)
    plt.close(fig)
    print(f"  âœ“ Saved feature_importance.png")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading data and model â€¦")
    df = get_dataset(ROOT / "data")
    preprocessor, X, y = preprocess_dataset(df, fit=True)

    # Use same split as training
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = load_model()
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test.values, y_pred)

    print("\n" + "=" * 40)
    print("TEST SET EVALUATION")
    print("=" * 40)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Determine feature names from preprocessing
    from src.preprocessing import FEATURE_FIELDS, prepare_features, clean_dataset
    df_clean = clean_dataset(df)
    feature_df = prepare_features(df_clean)
    feature_names = list(feature_df.columns)

    print("\nGenerating evaluation figures:")
    plot_actual_vs_predicted(y_test.values, y_pred, FIGURES_DIR)
    plot_residuals(y_test.values, y_pred, FIGURES_DIR)
    plot_feature_importance(model, feature_names, FIGURES_DIR)
    print("\nEvaluation complete âœ“")


if __name__ == "__main__":
    main()
