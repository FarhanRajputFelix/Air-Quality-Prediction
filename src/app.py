from pathlib import Path
from typing import List
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.model import load_model, load_preprocessor
from src.preprocessing import FEATURE_FIELDS, prepare_features

app = FastAPI(
    title="Air Quality Prediction API",
    description="Predict CO(GT) concentration from air quality sensors and weather features.",
    version="1.0.0",
)

MODEL = None
PREPROCESSOR = None

class PredictionRequest(BaseModel):
    data: List[dict] = Field(
        ...,
        description="A list of rows containing all required input features",
    )

class PredictionResponse(BaseModel):
    predictions: List[float]


@app.on_event("startup")
def load_artifacts():
    global MODEL, PREPROCESSOR
    try:
        PREPROCESSOR = load_preprocessor()
        MODEL = load_model()
    except Exception as exc:
        raise RuntimeError(f"Failed to load artifacts: {exc}") from exc


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if MODEL is None or PREPROCESSOR is None:
        raise HTTPException(status_code=500, detail="Model artifacts not loaded")

    df = None
    try:
        df = prepare_features(pd.DataFrame(request.data))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid input data: {exc}") from exc

    X = PREPROCESSOR.transform(df)
    values = MODEL.predict(X).tolist()
    return PredictionResponse(predictions=values)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
