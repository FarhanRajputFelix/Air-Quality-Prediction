from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib

MODEL_FILE = Path("models") / "air_quality_model.joblib"
PREPROCESSOR_FILE = Path("models") / "preprocessor.joblib"


def create_model() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )


def create_gradient_boosting_model() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
    )


def save_artifacts(model, preprocessor, model_path: Path = MODEL_FILE, preprocessor_path: Path = PREPROCESSOR_FILE) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Saved model to {model_path}")
    print(f"Saved preprocessor to {preprocessor_path}")


def load_model(model_path: Path = MODEL_FILE):
    return joblib.load(model_path)


def load_preprocessor(preprocessor_path: Path = PREPROCESSOR_FILE):
    return joblib.load(preprocessor_path)
