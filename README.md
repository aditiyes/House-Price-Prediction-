# 🏠 House Price Prediction System

> **Academic Internship Project Submission**  
> **Program:** AICTE | IBM SkillsBuild Data Analytics with AI Academic Internship Program 2026  
> **Partner Organization:** BharatCares  
> **Student Name:** Vinayak Rathi  

---

## 📌 Project Overview
The **House Price Prediction System** is an end-to-end Machine Learning web application designed to estimate residential property market values based on property attributes (`house_price_regression_dataset.csv`). The project implements a complete data-to-deployment pipeline featuring data preprocessing, exploratory data analysis (EDA), feature scaling, model training using **Linear Regression**, a scikit-learn Pipeline saved via `joblib`, a **Flask REST API** backend, and an interactive **Streamlit Dashboard** frontend.

---

## 🎯 Objectives
- Analyze key real estate attributes (`Square_Footage`, `Num_Bedrooms`, `Num_Bathrooms`, `Year_Built`, `Lot_Size`, `Garage_Size`, `Neighborhood_Quality`) influencing market valuation.
- Implement a reproducible Machine Learning pipeline with standardized scaling and multiple linear regression.
- Evaluate model prediction performance using empirical metrics ($R^2$, MAE, RMSE).
- Deploy the model via a RESTful Flask API (`POST /predict` and `GET /health`).
- Develop a user-friendly Streamlit web dashboard for real-time interactive property estimation and visual data analytics.
- Produce academic submission artifacts including executable Jupyter notebooks, DOCX report, and structured documentation.

---

## ⚙️ System Architecture

```
                ┌─────────────────────────┐
                │   Dataset (CSV File)    │
                │ house_price_regression_ │
                │       dataset.csv       │
                └────────────┬────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │ Preprocessing & Scaling │
                │     StandardScaler      │
                └────────────┬────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │ Linear Regression Model │
                │     train_model.py      │
                └────────────┬────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │ Saved Pipeline Artifact │
                │    models/model.pkl     │
                └────────────┬────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │ Flask REST API Backend  │
                │    backend/app.py       │
                └────────────┬────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │   Streamlit Frontend    │
                │     frontend/app.py     │
                └────────────┬────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │ Estimated Price Output  │
                │   + Visual Analytics    │
                └─────────────────────────┘
```

---

## 📊 Empirical Model Evaluation Results

The model was evaluated on an 80/20 train-test split (`random_state=42`) containing 200 holdout test properties from `house_price_regression_dataset.csv`.

| Evaluation Metric | Empirical Value | Description |
| :--- | :---: | :--- |
| **$R^2$ Score** | **0.9984** | Explains 99.84% of total variance in property prices. |
| **Mean Absolute Error (MAE)** | **$8,174.58** | Average magnitude of absolute prediction error. |
| **Root Mean Squared Error (RMSE)** | **$10,071.48** | Standard deviation of residual errors. |

### Feature Coefficients Breakdown (Standardized)
- `Square_Footage`: +$249,787.91 per standard deviation increase (Primary Price Driver)
- `Year_Built`: +$20,662.12 (Newer construction premium)
- `Lot_Size`: +$19,088.11 per std deviation
- `Num_Bedrooms`: +$14,524.73
- `Num_Bathrooms`: +$6,695.91
- `Garage_Size`: +$4,219.45
- `Neighborhood_Quality`: +$335.25
- `Intercept`: $618,576.05

---

## 🛠️ Project Structure

```
house/
├── house_price_regression_dataset.csv        # Dataset containing 1,000 property records
├── data/
│   └── house_prices.csv                      # Synced dataset copy
├── models/
│   └── model.pkl                             # Saved trained scikit-learn pipeline object
├── backend/
│   ├── app.py                                # Flask REST API server (/health, /predict)
│   └── test_api.py                           # Automated API unit testing suite
├── frontend/
│   └── app.py                                # Streamlit interactive web dashboard
├── notebooks/
│   └── VinayakRathi_HousePricePrediction.ipynb  # Executable 22-section Jupyter Notebook
├── reports/
│   └── VinayakRathi_HousePricePredictionReport.docx # Comprehensive academic report
├── train_model.py                             # Training & validation execution script
├── generate_report.py                         # Automated DOCX report generation script
├── app.py                                     # Streamlit entry point wrapper
├── VinayakRathi_HousePricePrediction.ipynb        # Main submission notebook
├── VinayakRathi_HousePricePredictionReport.docx   # Main submission report
├── requirements.txt                           # Python runtime dependencies
└── README.md                                  # Project documentation
```

---

## 🚀 Quickstart & Installation

### 1. Prerequisites & Virtual Environment
```bash
python -m venv venv
```

Activate environment:
- **Windows PowerShell:**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- **Linux/macOS:**
  ```bash
  source venv/bin/activate
  ```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Model Training & Validation
To re-train the linear regression model and generate `models/model.pkl`:
```bash
python train_model.py
```

### 4. Run Automated API Tests
To run unit tests for Flask REST API endpoints:
```bash
python backend/test_api.py
```

### 5. Launch Flask REST API Server
```bash
python backend/app.py
```
*API will run on `http://127.0.0.1:5000`.*

### 6. Launch Streamlit Web Dashboard
In a separate terminal window:
```bash
streamlit run app.py
```
*Dashboard will launch automatically in your web browser at `http://localhost:8501`.*

---

## 📡 REST API Usage & Testing

### Health Check Endpoint
- **URL:** `GET /health`
- **Response Example:**
```json
{
  "status": "healthy",
  "service": "House Price Prediction REST API",
  "model_loaded": true,
  "features_required": [
    "Square_Footage", "Num_Bedrooms", "Num_Bathrooms", 
    "Year_Built", "Lot_Size", "Garage_Size", "Neighborhood_Quality"
  ]
}
```

### Prediction Endpoint
- **URL:** `POST /predict`
- **Headers:** `Content-Type: application/json`
- **Request Body Example:**
```json
{
  "Square_Footage": 2500,
  "Num_Bedrooms": 3,
  "Num_Bathrooms": 2,
  "Year_Built": 2005,
  "Lot_Size": 3.5,
  "Garage_Size": 2,
  "Neighborhood_Quality": 8
}
```
- **Response Example:**
```json
{
  "success": true,
  "predicted_price": 590661.89,
  "formatted_price": "$590,661.89",
  "currency": "USD",
  "inputs": {
    "Square_Footage": 2500.0,
    "Num_Bedrooms": 3.0,
    "Num_Bathrooms": 2.0,
    "Year_Built": 2005.0,
    "Lot_Size": 3.5,
    "Garage_Size": 2.0,
    "Neighborhood_Quality": 8.0
  }
}
```

---

## 📜 Limitations & Responsible AI Use
1. **Linear Relationship Assumption:** Linear Regression models continuous linear features; non-linear interactions may require non-linear ensemble algorithms.
2. **Advisory Tool:** The application generates statistical point estimations and is not intended to replace certified human real estate appraisers or legal appraisals.

---

## 🎓 Academic Attribution & Acknowledgements
This project was developed as part of the **AICTE | IBM SkillsBuild Data Analytics with AI Academic Internship Program 2026** in collaboration with **BharatCares**.
