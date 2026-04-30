import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_curve, auc, confusion_matrix
import shap
import random

# ----------------- Configuration & CSS -----------------
st.set_page_config(page_title="Breast Cancer ML System (Pro)", layout="wide", page_icon="🔬")

# Custom CSS for Premium Look
st.markdown("""
<style>
    /* Glassmorphism for containers */
    .stApp {
        background-color: #0e1117;
    }
    div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] {
        background: rgba(25, 25, 30, 0.6);
        border-radius: 15px;
        padding: 15px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    /* Stylish Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #ff4b4b, #ff8b8b);
        color: white;
        border-radius: 8px;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(255, 75, 75, 0.3);
        font-weight: bold;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(255, 75, 75, 0.5);
    }
    /* Metric Cards styling */
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 800;
        color: #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Data & Models -----------------
@st.cache_resource
def load_models():
    with open("xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

@st.cache_data
def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    df['target_name'] = df['target'].map({0: 'Malignant', 1: 'Benign'})
    return df, data.feature_names, data.target

try:
    model, scaler = load_models()
except FileNotFoundError:
    st.error("Error: Model files not found.")
    st.stop()

df, feature_names, targets = load_data()

mean_features = [f for f in feature_names if 'mean' in f]
error_features = [f for f in feature_names if 'error' in f]
worst_features = [f for f in feature_names if 'worst' in f]

# ----------------- UI State & Sidebar -----------------
if 'user_inputs' not in st.session_state:
    st.session_state.user_inputs = {f: float(df[f].mean()) for f in feature_names}
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3209/3209088.png", width=100)
st.sidebar.title("ML Diagnostic System")
st.sidebar.markdown("Powered by **XGBoost** and **SHAP** Explainable AI.")
st.sidebar.markdown("---")

st.sidebar.subheader("Quick Testing")
st.sidebar.markdown("Use the button below to instantly load a random patient from the dataset.")

if st.sidebar.button("🎲 Load Random Patient", type="primary"):
    rand_idx = random.randint(0, len(df)-1)
    for f in feature_names:
        st.session_state.user_inputs[f] = df.loc[rand_idx, f]
    st.sidebar.success(f"Loaded Patient #{rand_idx} ({df.loc[rand_idx, 'target_name']})")
    st.session_state.prediction_made = False

# ----------------- Main Layout -----------------
st.title("🔬 Breast Cancer ML System (Pro)")
st.markdown("A portfolio-ready predictive tool powered by the **XGBoost Algorithm**. Enter features manually or use the sidebar to load a random patient.")

tab_predict, tab_xai, tab_visualize, tab_performance = st.tabs([
    "🔍 Diagnostics Dashboard", "🧠 Explainable AI (SHAP)", "🌌 3D Visualizations", "📈 Model Performance"
])

# ------------- TAB 1: PREDICT -------------
with tab_predict:
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Mean Features")
        for f in mean_features[:10]:
            st.session_state.user_inputs[f] = st.number_input(f.title(), value=float(st.session_state.user_inputs[f]), format="%.4f", key=f"in_{f}")
    with col2:
        st.subheader("Error Features")
        for f in error_features[:10]:
            st.session_state.user_inputs[f] = st.number_input(f.title(), value=float(st.session_state.user_inputs[f]), format="%.4f", key=f"in_{f}")
    with col3:
        st.subheader("Worst Features")
        for f in worst_features[:10]:
            st.session_state.user_inputs[f] = st.number_input(f.title(), value=float(st.session_state.user_inputs[f]), format="%.4f", key=f"in_{f}")
            
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Run Full Diagnostics", use_container_width=True):
        input_array = np.array([st.session_state.user_inputs[f] for f in feature_names]).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0]
        
        st.session_state.prediction = pred
        st.session_state.prob_benign = prob[1]
        st.session_state.prob_malignant = prob[0]
        st.session_state.input_scaled = input_scaled
        st.session_state.prediction_made = True
        
    if st.session_state.prediction_made:
        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 1.5])
        
        with res_col1:
            st.subheader("Diagnostic Outcome")
            if st.session_state.prediction == 1:
                st.success("## 🎉 BENIGN\nThe model is confident the tumor is benign.")
            else:
                st.error("## ⚠️ MALIGNANT\nHigh risk of malignancy detected.")
            
            st.metric("Confidence Score", f"{max(st.session_state.prob_benign, st.session_state.prob_malignant)*100:.2f}%")
            
        with res_col2:
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = st.session_state.prob_malignant * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Malignancy Probability", 'font': {'size': 24}},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#ff4b4b"},
                    'bgcolor': "rgba(255,255,255,0.1)",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 40], 'color': "rgba(0, 255, 0, 0.2)"},
                        {'range': [40, 60], 'color': "rgba(255, 255, 0, 0.2)"},
                        {'range': [60, 100], 'color': "rgba(255, 0, 0, 0.2)"}],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 50}
                }
            ))
            fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)", font_color="white")
            st.plotly_chart(fig_gauge, use_container_width=True)

# ------------- TAB 2: EXPLAINABLE AI (SHAP) -------------
with tab_xai:
    st.header("🧠 SHAP: Why did the model make this decision?")
    if not st.session_state.prediction_made:
        st.info("Run a prediction first to see the AI's explanation.")
    else:
        st.markdown("This chart breaks down the exact impact each feature had on the model's final prediction.")
        
        with st.spinner("Calculating SHAP values..."):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(st.session_state.input_scaled)
            
            # Handle potential list output from shap_values
            sv = shap_values[0] if isinstance(shap_values, list) else shap_values[0]
            
            shap_df = pd.DataFrame({
                'Feature': feature_names,
                'SHAP Value': sv,
                'Patient Value': [st.session_state.user_inputs[f] for f in feature_names]
            })
            
            shap_df['Abs SHAP'] = shap_df['SHAP Value'].abs()
            top_shap = shap_df.sort_values(by='Abs SHAP', ascending=False).head(12)
            
            fig_shap = px.bar(top_shap, x='SHAP Value', y='Feature', orientation='h',
                              color='SHAP Value', color_continuous_scale=px.colors.diverging.Tealrose,
                              title="Top Features Driving This Specific Prediction",
                              text='Patient Value')
            fig_shap.update_layout(yaxis={'categoryorder':'total ascending'}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_shap, use_container_width=True)
            
            st.markdown("""
            * **Positive SHAP Values:** Push the model towards Malignant.
            * **Negative SHAP Values:** Push the model towards Benign.
            """)

# ------------- TAB 3: 3D VISUALIZATIONS -------------
with tab_visualize:
    st.header("🌌 3D Data Exploration")
    st.markdown("Explore the clustering of benign vs malignant cases in a 3D feature space.")
    
    col_x, col_y, col_z = st.columns(3)
    with col_x: x_axis = st.selectbox("X-Axis", feature_names, index=0)
    with col_y: y_axis = st.selectbox("Y-Axis", feature_names, index=1)
    with col_z: z_axis = st.selectbox("Z-Axis", feature_names, index=2)
    
    fig_3d = px.scatter_3d(df, x=x_axis, y=y_axis, z=z_axis, color="target_name",
                           color_discrete_map={"Benign": "#00ffcc", "Malignant": "#ff4b4b"},
                           opacity=0.7, title=f"3D Mapping: {x_axis.title()} vs {y_axis.title()} vs {z_axis.title()}")
    fig_3d.update_layout(scene=dict(bgcolor="rgba(0,0,0,0)"), paper_bgcolor="rgba(0,0,0,0)", height=600, margin=dict(l=0, r=0, b=0, t=40))
    st.plotly_chart(fig_3d, use_container_width=True)

# ------------- TAB 4: MODEL PERFORMANCE -------------
with tab_performance:
    st.header("📈 Technical Model Evaluation")
    st.markdown("A breakdown of the XGBoost model's performance")
    
    perf_col1, perf_col2 = st.columns(2)
    
    # Calculate whole dataset metrics for portfolio demonstration purposes
    X = scaler.transform(df[feature_names])
    y_true = df['target']
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1] 
    
    with perf_col1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                           labels=dict(x="Predicted Label", y="True Label"),
                           x=['Malignant (0)', 'Benign (1)'], y=['Malignant (0)', 'Benign (1)'])
        fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_cm, use_container_width=True)
        
    with perf_col2:
        st.subheader("ROC Curve & AUC")
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ROC curve (AUC = {roc_auc:.4f})', line=dict(color='#ff4b4b', width=3)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color='gray', width=2, dash='dash'), showlegend=False))
        fig_roc.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_roc, use_container_width=True)
