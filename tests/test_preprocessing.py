"""Unit tests for src.preprocessing module."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    TARGET,
    FEATURE_FIELDS,
    clean_dataset,
    prepare_features,
    build_pipeline,
    preprocess_dataset,
)


@pytest.fixture
def sample_df():
    """Create a minimal DataFrame that mimics the UCI Air Quality structure."""
    np.random.seed(42)
    n = 100
    data = {col: np.random.uniform(0, 500, n) for col in FEATURE_FIELDS}
    data[TARGET] = np.random.uniform(0, 10, n)
    data["Date_Time"] = pd.date_range("2004-03-10", periods=n, freq="h")
    data["Unnamed: 15"] = np.nan  # junk column
    return pd.DataFrame(data)


class TestCleanDataset:
    def test_drops_unnamed_columns(self, sample_df):
        result = clean_dataset(sample_df)
        assert not any(col.startswith("Unnamed") for col in result.columns)

    def test_drops_rows_without_target(self, sample_df):
        df = sample_df.copy()
        df.loc[0, TARGET] = np.nan
        result = clean_dataset(df)
        assert result[TARGET].isna().sum() == 0

    def test_output_has_numeric_features(self, sample_df):
        result = clean_dataset(sample_df)
        for col in FEATURE_FIELDS:
            assert pd.api.types.is_numeric_dtype(result[col])


class TestPrepareFeatures:
    def test_output_columns_include_features(self, sample_df):
        df = clean_dataset(sample_df)
        features = prepare_features(df)
        for col in FEATURE_FIELDS:
            assert col in features.columns

    def test_date_features_extracted(self, sample_df):
        df = clean_dataset(sample_df)
        features = prepare_features(df)
        assert "hour" in features.columns
        assert "month" in features.columns


class TestBuildPipeline:
    def test_pipeline_fit_transform(self, sample_df):
        df = clean_dataset(sample_df)
        X = prepare_features(df)
        pipeline = build_pipeline()
        X_transformed = pipeline.fit_transform(X)
        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] == X.shape[1]

    def test_pipeline_output_scaled(self, sample_df):
        df = clean_dataset(sample_df)
        X = prepare_features(df)
        pipeline = build_pipeline()
        X_transformed = pipeline.fit_transform(X)
        # StandardScaler produces approx zero mean
        assert abs(X_transformed.mean()) < 0.5


class TestPreprocessDataset:
    def test_returns_three_elements(self, sample_df):
        result = preprocess_dataset(sample_df)
        assert len(result) == 3

    def test_preprocessor_is_fitted(self, sample_df):
        preprocessor, X, y = preprocess_dataset(sample_df)
        # Fitted pipeline should have named_steps
        assert hasattr(preprocessor, "named_steps")

    def test_target_shape(self, sample_df):
        _, X, y = preprocess_dataset(sample_df)
        assert X.shape[0] == len(y)
