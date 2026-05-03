"""Training workflow with multi-model comparison.

Compares Random Forest and Gradient Boosting regressors on the UCI Air
Quality dataset and saves the best-performing model to ``models/``.
"""

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import numpy as np

from src.data_loader import get_dataset
from src.preprocessing import preprocess_dataset
from src.model import create_model, create_gradient_boosting_model, save_artifacts


def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def run_training(data_dir: Path, test_size: float, random_state: int) -> None:
    df = get_dataset(data_dir)
    preprocessor, X, y = preprocess_dataset(df, fit=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
    )

    # ---------- Model candidates ----------
    candidates = {
        "Random Forest": create_model(),
        "Gradient Boosting": create_gradient_boosting_model(),
    }

    results = {}
    best_name, best_model, best_r2 = None, None, -float("inf")

    for name, model in candidates.items():
        print(f"\nTraining {name} ...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = evaluate(y_test, preds)
        results[name] = metrics
        if metrics["R2"] > best_r2:
            best_name, best_model, best_r2 = name, model, metrics["R2"]

    # ---------- Results table ----------
    print("\n" + "=" * 55)
    print(f"{'Model':<25} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
    print("-" * 55)
    for name, m in results.items():
        tag = " *" if name == best_name else ""
        print(f"{name:<25} {m['MAE']:>8.4f} {m['RMSE']:>8.4f} {m['R2']:>8.4f}{tag}")
    print("=" * 55)
    print(f"\nBest model: {best_name}")

    save_artifacts(best_model, preprocessor)


def main():
    parser = argparse.ArgumentParser(description="Train air quality regression model")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Data directory")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    run_training(args.data_dir, args.test_size, args.random_state)


if __name__ == "__main__":
    main()
