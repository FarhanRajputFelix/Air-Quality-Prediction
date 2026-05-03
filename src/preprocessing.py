from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "CO(GT)"
FEATURE_FIELDS = [
    "PT08.S1(CO)",
    "NMHC(GT)",
    "C6H6(GT)",
    "PT08.S2(NMHC)",
    "NOx(GT)",
    "PT08.S3(NOx)",
    "NO2(GT)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "T",
    "RH",
    "AH",
]


class DateFeatureExtractor(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        if "Date_Time" in df.columns:
            df["hour"] = df["Date_Time"].dt.hour
            df["month"] = df["Date_Time"].dt.month
            return df[["hour", "month"]]
        raise ValueError("Date_Time column not found for feature extraction")


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=[col for col in df.columns if col.startswith("Unnamed")], errors="ignore")
    df = df.dropna(subset=[TARGET])
    df[FEATURE_FIELDS] = df[FEATURE_FIELDS].apply(pd.to_numeric, errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=FEATURE_FIELDS)
    return df


def build_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return Pipeline(steps=[("numeric", numeric_pipeline)])


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Date_Time" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["Date_Time"]):
            df["Date_Time"] = pd.to_datetime(df["Date_Time"], errors="coerce")
        df["hour"] = df["Date_Time"].dt.hour
        df["month"] = df["Date_Time"].dt.month
    feature_df = df[FEATURE_FIELDS].copy()
    if "hour" in df.columns and "month" in df.columns:
        feature_df["hour"] = df["hour"]
        feature_df["month"] = df["month"]
    return feature_df


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df = clean_dataset(df)
    X = prepare_features(df)
    y = df[TARGET].copy()
    return X, y


def preprocess_dataset(
    df: pd.DataFrame,
    preprocessor: Optional[Pipeline] = None,
    fit: bool = True,
) -> Tuple[Pipeline, pd.DataFrame, pd.Series]:
    df = clean_dataset(df)
    X = prepare_features(df)
    y = df[TARGET].copy()
    if preprocessor is None:
        preprocessor = build_pipeline()
    if fit:
        X_processed = preprocessor.fit_transform(X)
    else:
        X_processed = preprocessor.transform(X)
    return preprocessor, X_processed, y
