import pickle
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist

app = FastAPI(
    title="Breast Cancer Detection API",
    description="FastAPI backend for XGBoost breast cancer prediction.",
    version="1.0.0",
)

MODEL_PATH = "xgb_model.pkl"
SCALER_PATH = "scaler.pkl"
FEATURE_COUNT = 30


class PredictionInput(BaseModel):
    features: conlist(float, min_items=FEATURE_COUNT, max_items=FEATURE_COUNT)
    feature_names: Optional[List[str]] = None


class PredictionResponse(BaseModel):
    prediction: str
    label: int
    probabilities: dict
    feature_count: int


def load_model():
    try:
        with open(MODEL_PATH, "rb") as model_file:
            model = pickle.load(model_file)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Could not load model from {MODEL_PATH}") from exc

    try:
        with open(SCALER_PATH, "rb") as scaler_file:
            scaler = pickle.load(scaler_file)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Could not load scaler from {SCALER_PATH}") from exc

    return model, scaler


model, scaler = load_model()


@app.get("/", tags=["Health"])
def root():
    return {
        "status": "ok",
        "message": "Breast Cancer Detection API is running.",
        "documentation": "/docs",
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(data: PredictionInput):
    try:
        input_array = [data.features]
        input_scaled = scaler.transform(input_array)
        prediction_label = int(model.predict(input_scaled)[0])
        probabilities = model.predict_proba(input_scaled)[0].tolist()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "prediction": "Benign" if prediction_label == 1 else "Malignant",
        "label": prediction_label,
        "probabilities": {
            "malignant": probabilities[0],
            "benign": probabilities[1],
        },
        "feature_count": len(data.features),
    }
