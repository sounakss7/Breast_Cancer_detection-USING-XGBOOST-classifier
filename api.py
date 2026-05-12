from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from typing import List

app = FastAPI(title="Breast Cancer Detection API", description="API for predicting breast cancer using XGBoost")

# Load saved model and scaler
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

class BreastCancerFeatures(BaseModel):
    mean_radius: float
    mean_texture: float
    mean_perimeter: float
    mean_area: float
    mean_smoothness: float
    mean_compactness: float
    mean_concavity: float
    mean_concave_points: float
    mean_symmetry: float
    mean_fractal_dimension: float
    radius_error: float
    texture_error: float
    perimeter_error: float
    area_error: float
    smoothness_error: float
    compactness_error: float
    concavity_error: float
    concave_points_error: float
    symmetry_error: float
    fractal_dimension_error: float
    worst_radius: float
    worst_texture: float
    worst_perimeter: float
    worst_area: float
    worst_smoothness: float
    worst_compactness: float
    worst_concavity: float
    worst_concave_points: float
    worst_symmetry: float
    worst_fractal_dimension: float

@app.post("/predict")
def predict(features: BreastCancerFeatures):
    # Convert input to numpy array
    feature_values = [
        features.mean_radius, features.mean_texture, features.mean_perimeter, features.mean_area, features.mean_smoothness,
        features.mean_compactness, features.mean_concavity, features.mean_concave_points, features.mean_symmetry, features.mean_fractal_dimension,
        features.radius_error, features.texture_error, features.perimeter_error, features.area_error, features.smoothness_error,
        features.compactness_error, features.concavity_error, features.concave_points_error, features.symmetry_error, features.fractal_dimension_error,
        features.worst_radius, features.worst_texture, features.worst_perimeter, features.worst_area, features.worst_smoothness,
        features.worst_compactness, features.worst_concavity, features.worst_concave_points, features.worst_symmetry, features.worst_fractal_dimension
    ]
    
    input_array = np.array(feature_values).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = int(model.predict(input_scaled)[0])
    prob = model.predict_proba(input_scaled)[0].tolist()
    
    result = "Benign" if prediction == 1 else "Malignant"
    confidence = prob[1] if prediction == 1 else prob[0]
    
    return {
        "prediction": result,
        "prediction_class": prediction,
        "confidence": confidence,
        "probabilities": {"malignant": prob[0], "benign": prob[1]}
    }

@app.get("/")
def root():
    return {"message": "Welcome to the Breast Cancer Detection API. Go to /docs to test the API."}
