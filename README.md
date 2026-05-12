---

# Breast Cancer ML Diagnostic System (Pro) 🔬

![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-red?style=for-the-badge&logo=streamlit)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green?style=for-the-badge&logo=python)
![SHAP](https://img.shields.io/badge/Explainable%20AI-SHAP-blue?style=for-the-badge)

## 🔗 Live Demo

Access the live Streamlit app here:
👉 [Click to Launch Web App](https://breastcancerdetection-using-xgboost-classifier-jjxxgerkbhvtgip.streamlit.app/) *(Note: This link may point to the legacy version depending on deployment status).*

---

## 📌 Project Overview

This project is a **professional-grade machine learning diagnostic tool** designed to detect breast cancer using the **XGBoost classifier**. It uses the Wisconsin Breast Cancer Dataset and provides a premium web interface built using **Streamlit**. 

Unlike standard ML projects, this system incorporates **Explainable AI (XAI)** to eliminate the "black-box" problem, providing clinicians and recruiters with deep insights into exactly *why* a specific prediction was made.

---

## 🚀 Advanced Features

* **Premium UI/UX:** Features a sleek dark-mode interface with glassmorphism effects, dynamic color-coding, and animated gradient buttons.
* **Effortless Testing:** Includes a **"🎲 Load Random Patient"** button that instantly populates all 30 input features with a real case from the dataset, allowing for instant testing without manual data entry.
* **Explainable AI (SHAP):** Once a prediction is made, the app generates a detailed SHAP waterfall chart breaking down the exact impact each feature had on the model's final decision.
* **3D Visualizations:** Includes fully interactive 3D scatter plots allowing users to map out and explore the clustering of benign vs. malignant cases.
* **Technical Model Performance:** A dedicated dashboard evaluating the underlying XGBoost model using Confusion Matrices and an interactive ROC-AUC Curve.

---

## 🧠 Technologies Used

* **Core Frameworks:** Python, Streamlit
* **Machine Learning:** Scikit-learn, XGBoost
* **Explainable AI (XAI):** SHAP
* **Data Visualization:** Plotly (Express & Graph Objects), Matplotlib, Seaborn
* **Data Processing:** Pandas, NumPy

---

## 📊 Dataset

* **Name**: Breast Cancer Wisconsin (Diagnostic) Data Set
* **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
* **Features**: 30 real-valued input features + diagnosis label (M = Malignant, B = Benign)

---

## 🔧 How to Run Locally

1. **Clone the repository**

   ```bash
   git clone https://github.com/sounakss7/Breast_Cancer_detection-USING-XGBOOST-classifier.git
   cd Breast_Cancer_detection-USING-XGBOOST-classifier
   ```

2. **Install dependencies**
   It is recommended to use a virtual environment.

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**

   ```bash
   python -m streamlit run app.py
   ```
   *(The app will automatically open in your default browser at `http://localhost:8501`)*

   Inside the Streamlit sidebar, choose **API backend** to send the prediction request to `http://localhost:8000/predict` or your custom backend URL.

4. **Run the FastAPI backend**

   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```
   Then open `http://localhost:8000/docs` for API docs.

---

## 🐳 Docker Usage

Build the Docker image:

```bash
docker build -t breast-cancer-api .
```

Run the backend container:

```bash
docker run --rm -p 8000:8000 breast-cancer-api
```

Run both services with Docker Compose:

```bash
docker compose up --build
```

The backend will be available at `http://localhost:8000` and the Streamlit UI at `http://localhost:8501`.

When running with Docker Compose, the Streamlit app automatically uses the backend service host `http://backend:8000/predict` via the `BACKEND_URL` environment variable.

---

## ✅ Model Performance

* **Classifier**: XGBoost
* **Accuracy**: ~98%
* **Evaluation Metrics Provided**: Confusion Matrix, ROC-AUC Curve

---

## 📂 Project Structure

```
├── app.py                     # Main Streamlit web application with advanced UI
├── main.py                    # FastAPI backend for prediction requests
├── Dockerfile                 # Container image definition for backend and UI
├── docker-compose.yml         # Compose setup for backend and Streamlit services
├── xgb_model.pkl              # Trained XGBoost model
├── scaler.pkl                 # StandardScaler for feature normalization
├── requirements.txt           # Python dependencies (Streamlit, XGBoost, SHAP, FastAPI, etc.)
├── XGBOOST.ipynb              # Original training and analysis notebook
└── README.md                  # Project documentation
```

---

## 📸 Screenshots

### Confusion Matrix
*(Legacy Screenshot)*
![Confusion Matrix](output.png)

---

## 📬 Contact

Created with ❤️ by [**Sounak**](https://github.com/sounakss7)
Feel free to reach out via GitHub for issues or suggestions.

## 📊 Project Reach

Here’s a quick snapshot of how many people have shown interest in my work:

| Repository | Total Clones | Unique Cloners |
|------------|--------------|----------------|
| [Breast_Cancer_detection-USING-XGBOOST-classifier](https://github.com/sounakss7/Breast_Cancer_detection-USING-XGBOOST-classifier) | 64 | 41 |

> 🚀 Thank you to everyone who has explored and cloned my projects!  
> I’m continuously working to share more impactful open-source work.
