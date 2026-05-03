import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd
from src.model import load_model, load_preprocessor
from src.preprocessing import FEATURE_FIELDS, prepare_features


def load_input_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [col for col in FEATURE_FIELDS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required features in CSV: {missing}")
    return df


def predict_from_csv(csv_path: Path) -> pd.Series:
    preprocessor = load_preprocessor()
    model = load_model()
    df = load_input_csv(csv_path)
    input_df = prepare_features(df)
    X = preprocessor.transform(input_df)
    predictions = model.predict(X)
    return pd.Series(predictions, index=df.index)


def main():
    parser = argparse.ArgumentParser(description="Predict air quality from CSV")
    parser.add_argument("--csv", type=Path, required=True, help="CSV file with input features")
    parser.add_argument("--output", type=Path, default=None, help="Optional output CSV path")
    args = parser.parse_args()

    predictions = predict_from_csv(args.csv)
    print("Predictions:")
    print(predictions.to_string(index=False))

    if args.output:
        output_df = pd.read_csv(args.csv).copy()
        output_df["predicted_CO_GT"] = predictions.values
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(args.output, index=False)
        print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
