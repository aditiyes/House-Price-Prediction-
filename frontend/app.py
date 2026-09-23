import os
import sys
import requests
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="House Price Prediction System | IBM SkillsBuild",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design Aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 20px;
    }
    .badge-container {
        background: linear-gradient(135deg, #1E40AF, #3B82F6);
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        font-weight: 600;
        margin-bottom: 25px;
    }
    .price-display {
        font-size: 2.5rem;
        color: #166534;
        font-weight: 800;
    }
</style>
""", unsafe_allow_html=True)

# Application Title & Academic Branding
st.markdown('<div class="main-header">🏠 HOUSE PRICE PREDICTION SYSTEM</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Machine Learning-Based Property Valuation using Linear Regression, Flask REST API & Streamlit</div>', unsafe_allow_html=True)
st.markdown('<div class="badge-container">🎓 AICTE | IBM SkillsBuild Data Analytics with AI Academic Internship Program 2026 | BharatCares</div>', unsafe_allow_html=True)

# Helper Function to Load Dataset
@st.cache_data
def load_dataset():
    data_path = "house_price_regression_dataset.csv"
    if not os.path.exists(data_path):
        data_path = os.path.join("data", "house_prices.csv")
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None

df_data = load_dataset()

# Helper Function to Load Local Model (Fallback)
@st.cache_resource
def load_local_model():
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "model.pkl")
    if not os.path.exists(model_path):
        model_path = os.path.join("models", "model.pkl")
    if not os.path.exists(model_path):
        model_path = "model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model_pipeline = load_local_model()

# Sidebar Setup - User Inputs
st.sidebar.header("📋 Property Specifications")
st.sidebar.markdown("Specify the details of the target property below:")

sq_ft = st.sidebar.number_input("Square Footage (sq.ft)", min_value=500, max_value=5000, value=2500, step=50)
bedrooms = st.sidebar.slider("Number of Bedrooms", min_value=1, max_value=5, value=3)
bathrooms = st.sidebar.slider("Number of Bathrooms", min_value=1, max_value=3, value=2)
year_built = st.sidebar.number_input("Year Built", min_value=1950, max_value=2024, value=2005, step=1)
lot_size = st.sidebar.number_input("Lot Size (Acres)", min_value=0.1, max_value=5.0, value=2.5, step=0.1)
garage_size = st.sidebar.slider("Garage Size (Cars)", min_value=0, max_value=2, value=1)
neighborhood = st.sidebar.slider("Neighborhood Quality (1-10)", min_value=1, max_value=10, value=6)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Backend Architecture")
backend_mode = st.sidebar.radio(
    "Prediction Mode",
    ["Flask REST API Endpoint", "Direct Local Pipeline Model"],
    index=0
)
api_url = st.sidebar.text_input("Flask API URL", value="http://127.0.0.1:5000/predict")

# Main Content Layout
tab1, tab2, tab3 = st.tabs(["🔮 Real-Time Prediction", "📊 Dataset Analytics", "📈 Model Evaluation & Details"])

with tab1:
    col_input, col_result = st.columns([1, 1])

    with col_input:
        st.subheader("Selected Property Inputs")
        input_summary = pd.DataFrame([{
            "Feature": "Square Footage", "Value": f"{sq_ft:,} sq.ft."
        }, {
            "Feature": "Bedrooms", "Value": f"{bedrooms}"
        }, {
            "Feature": "Bathrooms", "Value": f"{bathrooms}"
        }, {
            "Feature": "Year Built", "Value": f"{year_built}"
        }, {
            "Feature": "Lot Size", "Value": f"{lot_size} acres"
        }, {
            "Feature": "Garage Size", "Value": f"{garage_size} car(s)"
        }, {
            "Feature": "Neighborhood Quality", "Value": f"{neighborhood} / 10"
        }])
        st.table(input_summary)

        predict_btn = st.button("🚀 Predict House Price", use_container_width=True, type="primary")

    with col_result:
        st.subheader("Model Estimated Valuation")

        if predict_btn:
            payload = {
                "Square_Footage": sq_ft,
                "Num_Bedrooms": bedrooms,
                "Num_Bathrooms": bathrooms,
                "Year_Built": year_built,
                "Lot_Size": lot_size,
                "Garage_Size": garage_size,
                "Neighborhood_Quality": neighborhood
            }

            predicted_price = None
            prediction_source = ""

            if backend_mode == "Flask REST API Endpoint":
                with st.spinner("Contacting Flask REST API (/predict)..."):
                    try:
                        res = requests.post(api_url, json=payload, timeout=5)
                        if res.status_code == 200:
                            data = res.json()
                            predicted_price = data.get("predicted_price")
                            prediction_source = "Flask REST API (http://127.0.0.1:5000)"
                        else:
                            st.error(f"Flask API returned status code {res.status_code}: {res.text}")
                    except Exception as err:
                        st.warning(f"Could not connect to Flask API ({err}). Falling back to local model.")
                        if model_pipeline:
                            input_df = pd.DataFrame([payload])
                            predicted_price = float(model_pipeline.predict(input_df)[0])
                            prediction_source = "Local Pipeline Model (Fallback)"
            else:
                if model_pipeline:
                    input_df = pd.DataFrame([payload])
                    predicted_price = float(model_pipeline.predict(input_df)[0])
                    prediction_source = "Direct Local Pipeline Model"
                else:
                    st.error("Local model file `model.pkl` could not be found.")

            if predicted_price is not None:
                predicted_price = max(predicted_price, 10000.0)

                st.success("✅ Prediction Successfully Generated!")
                st.markdown(f'<div class="price-display">${predicted_price:,.2f}</div>', unsafe_allow_html=True)
                
                price_per_sqft = predicted_price / sq_ft
                st.write(f"**Estimated Rate:** ${price_per_sqft:,.2f} per sq.ft.")
                st.caption(f"Prediction Source: `{prediction_source}`")
        else:
            st.info("👈 Adjust property features in sidebar and click **Predict House Price** to get an estimate.")

with tab2:
    st.subheader("Exploratory Data Analysis (EDA)")
    if df_data is not None:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Distribution of House Prices")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(df_data['House_Price'] / 1000, kde=True, ax=ax, color='#2563EB')
            ax.set_xlabel("Price ($ Thousands)")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

        with c2:
            st.markdown("#### Square Footage vs Price")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(data=df_data, x='Square_Footage', y=df_data['House_Price']/1000, hue='Neighborhood_Quality', palette='viridis', ax=ax)
            ax.set_xlabel("Square Footage")
            ax.set_ylabel("Price ($ Thousands)")
            st.pyplot(fig)

        st.markdown("#### Feature Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(df_data.corr(), annot=True, fmt='.2f', cmap='Blues', ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Dataset file not available for display.")

with tab3:
    st.subheader("Model Performance & Technical Summary")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("R² Score (Accuracy)", "0.9984", "99.84% Precision")
    m2.metric("Mean Absolute Error (MAE)", "$8,174.58", "Low Error")
    m3.metric("Root Mean Squared Error (RMSE)", "$10,071.48", "Standard Error")

    st.markdown("---")
    st.markdown("""
    ### 🔬 Methodology & Architecture
    - **Model Type:** Multiple Linear Regression with Standard Scaler Preprocessing.
    - **Dataset:** `house_price_regression_dataset.csv` (1,000 property records).
    - **Mathematical Form:** $y = \\beta_0 + \\beta_1 x_1 + \\beta_2 x_2 + \\dots + \\beta_n x_n$
    - **Pipeline Integration:** `scikit-learn` Pipeline saved via `joblib`.
    - **Deployment:** Flask REST API Backend + Streamlit Interactive Dashboard Frontend.
    
    > **Academic Disclaimer:** This application is designed as an analytical decision-support demonstration tool under the AICTE | IBM SkillsBuild Internship Program.
    """)
