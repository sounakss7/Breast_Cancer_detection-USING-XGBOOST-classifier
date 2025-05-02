import streamlit as st
import pickle
import numpy as np

# Load saved model and scaler
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# UI
st.set_page_config(page_title="Breast Cancer Detector", layout="wide")
st.title("🔬 Breast Cancer Detection using XGBoost")
st.markdown("Enter the feature values below to predict if the tumor is **benign (1)** or **malignant (0)**.")

# Input fields based on the 30 features in the dataset
feature_names = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
    'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
    'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

# Organize inputs into 3 columns
cols = st.columns(3)
user_inputs = []

for idx, feature in enumerate(feature_names):
    with cols[idx % 3]:
        val = st.number_input(f"{feature}", min_value=0.0, format="%.4f")
        user_inputs.append(val)

if st.button("🔍 Predict"):
    input_array = np.array(user_inputs).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0]

    if prediction == 1:
        st.success(f"🎉 Prediction: **Benign** (Confidence: {prob[1]*100:.2f}%)")
    else:
        st.error(f"⚠️ Prediction: **Malignant** (Confidence: {prob[0]*100:.2f}%)")
