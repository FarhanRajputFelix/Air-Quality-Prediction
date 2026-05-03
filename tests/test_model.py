"""Unit tests for src.model module."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tempfile
import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from src.model import (
    create_model,
    create_gradient_boosting_model,
    save_artifacts,
    load_model,
    load_preprocessor,
)
from src.preprocessing import build_pipeline


class TestCreateModel:
    def test_returns_random_forest(self):
        model = create_model()
        assert isinstance(model, RandomForestRegressor)

    def test_model_params(self):
        model = create_model()
        assert model.n_estimators == 200
        assert model.random_state == 42

    def test_gradient_boosting_model(self):
        model = create_gradient_boosting_model()
        assert isinstance(model, GradientBoostingRegressor)


class TestSaveLoadArtifacts:
    def test_save_and_load_roundtrip(self):
        model = create_model()
        X = np.random.rand(50, 5)
        y = np.random.rand(50)
        model.fit(X, y)

        pipeline = build_pipeline()
        pipeline.fit(X)

        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.joblib"
            prep_path = Path(tmp) / "prep.joblib"
            save_artifacts(model, pipeline, model_path, prep_path)

            loaded_model = load_model(model_path)
            loaded_prep = load_preprocessor(prep_path)

            preds_original = model.predict(X)
            preds_loaded = loaded_model.predict(X)
            np.testing.assert_allclose(preds_original, preds_loaded, rtol=1e-14)
