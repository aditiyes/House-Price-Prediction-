import sys
import os
import joblib
import numpy as np
import pandas as pd

# Reconfigure stdout for UTF-8 encoding on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def train_and_save_model():
    print("=" * 60)
    print("HOUSE PRICE PREDICTION - MODEL TRAINING & VALIDATION")
    print("=" * 60)

    # 1. Load Dataset
    data_path = "house_price_regression_dataset.csv"
    if not os.path.exists(data_path):
        data_path = os.path.join("data", "house_prices.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    print(f"[*] Dataset loaded successfully from {data_path}. Shape: {df.shape}")

    # Data Quality Checks
    print(f"[*] Missing values count:\n{df.isnull().sum()}")
    print(f"[*] Duplicate rows count: {df.duplicated().sum()}")

    # 2. Features and Target
    feature_cols = [
        'Square_Footage', 
        'Num_Bedrooms', 
        'Num_Bathrooms', 
        'Year_Built', 
        'Lot_Size', 
        'Garage_Size',
        'Neighborhood_Quality'
    ]
    target_col = 'House_Price'

    X = df[feature_cols]
    y = df[target_col]

    # 3. Train-Test Split (80% Train, 20% Test, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"[*] Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    # 4. Build Pipeline (StandardScaler + LinearRegression)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])

    # 5. Train Model
    pipeline.fit(X_train, y_train)
    print("[*] Linear Regression pipeline trained successfully.")

    # 6. Evaluate Model
    y_pred = pipeline.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("-" * 60)
    print("EVALUATION METRICS ON TEST SET:")
    print(f"   R² Score : {r2:.4f}")
    print(f"   MAE      : ${mae:,.2f}")
    print(f"   RMSE     : ${rmse:,.2f}")
    print("-" * 60)

    # 7. Model Coefficients Breakdown
    regressor = pipeline.named_steps['regressor']
    scaler = pipeline.named_steps['scaler']
    print("MODEL COEFFICIENTS (Standardized):")
    for feature, coef in zip(feature_cols, regressor.coef_):
        print(f"   {feature:<30}: {coef:>15,.2f}")
    print(f"   {'Intercept':<30}: {regressor.intercept_:>15,.2f}")
    print("-" * 60)

    # 8. Save Model Pipeline
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "model.pkl")
    joblib.dump(pipeline, model_path)
    joblib.dump(pipeline, "model.pkl")
    print(f"[*] Model saved successfully to '{model_path}' and 'model.pkl'.")

    # 9. Reload & Validate Saved Model
    print("[*] Reloading saved model to validate inference...")
    reloaded_pipeline = joblib.load(model_path)

    sample_input = pd.DataFrame([{
        'Square_Footage': 2500,
        'Num_Bedrooms': 3,
        'Num_Bathrooms': 2,
        'Year_Built': 2005,
        'Lot_Size': 3.5,
        'Garage_Size': 2,
        'Neighborhood_Quality': 8
    }])

    predicted_price = reloaded_pipeline.predict(sample_input)[0]
    print(f"[*] Validation Sample Input:\n{sample_input.iloc[0].to_dict()}")
    print(f"[*] Predicted House Price: ${predicted_price:,.2f}")
    print("=" * 60)
    print("TRAINING & VALIDATION COMPLETE SUCCESSFUL!")
    print("=" * 60)

    return {
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'coefficients': dict(zip(feature_cols, regressor.coef_)),
        'intercept': regressor.intercept_
    }

if __name__ == '__main__':
    train_and_save_model()
