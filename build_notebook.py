import json
import os

def preserve_outputs(cells, notebook_path):
    if not os.path.exists(notebook_path):
        return

    with open(notebook_path, encoding="utf-8") as f:
        previous_cells = json.load(f).get("cells", [])

    outputs_by_source = {
        "".join(cell.get("source", [])): cell
        for cell in previous_cells
        if cell.get("cell_type") == "code" and cell.get("outputs")
    }
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        previous_cell = outputs_by_source.get("".join(cell.get("source", [])))
        if previous_cell is not None:
            cell["execution_count"] = previous_cell.get("execution_count")
            cell["outputs"] = previous_cell["outputs"]

def create_notebook():
    cells = [
        # Section 1
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🏠 House Price Prediction System\n",
                "## AICTE | IBM SkillsBuild Data Analytics with AI Academic Internship Program 2026 | BharatCares\n",
                "\n",
                "**Student Name:** Aditi Kongre  \n",
                "**Domain:** Machine Learning & Data Analytics  \n",
                "**Dataset:** `house_price_regression_dataset.csv`  \n",
                "**Algorithm:** Linear Regression  \n",
                "**Deployment Stack:** Flask REST API & Streamlit Interactive Dashboard  \n",
                "\n",
                "---"
            ]
        },
        # Section 2 & 3
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1. Project Introduction & Problem Statement\n",
                "Real estate valuation is a critical process for home buyers, sellers, property investors, and financial lenders. Traditional manual appraisal methods rely heavily on subjective judgment, which can introduce inconsistency and valuation bias.\n",
                "\n",
                "### 2. Business Objective\n",
                "The objective of this project is to construct a fully transparent Machine Learning pipeline that predicts property prices (`House_Price`) based on property dimensions, age, lot size, and location quality using **Linear Regression**. The model is encapsulated in a scikit-learn Pipeline, exposed via a **Flask REST API**, and presented through an interactive **Streamlit Dashboard**."
            ]
        },
        # Section 4
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3. Dataset Description\n",
                "The project utilizes `house_price_regression_dataset.csv` containing 1,000 property records with 7 predictive features and 1 target variable (`House_Price`):\n",
                "- `Square_Footage`: Living area of the house in square feet (503 - 4,999 sq.ft.)\n",
                "- `Num_Bedrooms`: Total number of bedrooms (1 - 5)\n",
                "- `Num_Bathrooms`: Total number of bathrooms (1 - 3)\n",
                "- `Year_Built`: Year of construction (1950 - 2020)\n",
                "- `Lot_Size`: Total lot size in acres (0.5 - 5.0 acres)\n",
                "- `Garage_Size`: Number of garage parking spaces (0 - 2 cars)\n",
                "- `Neighborhood_Quality`: Rating of neighborhood quality (1 - 10 scale)\n",
                "- `House_Price`: Final property transaction price ($111,626 - $1,108,237)"
            ]
        },
        # Section 5: Import Libraries
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Section 5: Import Libraries\n",
                "import os\n",
                "import joblib\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "from sklearn.model_selection import train_test_split\n",
                "from sklearn.pipeline import Pipeline\n",
                "from sklearn.preprocessing import StandardScaler\n",
                "from sklearn.linear_model import LinearRegression\n",
                "from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error\n",
                "\n",
                "# Set plot styles\n",
                "plt.style.use('seaborn-v0_8-whitegrid')\n",
                "plt.rcParams['figure.figsize'] = (10, 6)\n",
                "print(\"[*] Libraries imported successfully.\")"
            ]
        },
        # Section 6 & 7: Load Dataset & Data Understanding
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Section 6 & 7: Load Dataset & Data Understanding\n",
                "data_path = \"house_price_regression_dataset.csv\"\n",
                "if not os.path.exists(data_path):\n",
                "    data_path = os.path.join(\"data\", \"house_prices.csv\")\n",
                "\n",
                "df = pd.read_csv(data_path)\n",
                "print(f\"[*] Dataset Shape: {df.shape}\")\n",
                "display(df.head())\n",
                "print(\"\\n[*] Dataset Info:\")\n",
                "df.info()\n",
                "print(\"\\n[*] Descriptive Statistics:\")\n",
                "display(df.describe())"
            ]
        },
        # Section 8, 9, 10, 11: Data Cleaning, Missing Values, Duplicates, Outliers
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Section 8 - 11: Data Cleaning & Integrity Analysis\n",
                "print(\"[*] Missing Values Count per Feature:\")\n",
                "print(df.isnull().sum())\n",
                "\n",
                "print(f\"\\n[*] Total Duplicate Rows: {df.duplicated().sum()}\")\n",
                "\n",
                "# Outlier Boxplots\n",
                "plt.figure(figsize=(14, 6))\n",
                "plt.subplot(1, 2, 1)\n",
                "sns.boxplot(y=df['House_Price'] / 1000, color='#3B82F6')\n",
                "plt.title(\"Boxplot of House Prices ($ Thousands)\")\n",
                "plt.ylabel(\"Price ($ Thousands)\")\n",
                "\n",
                "plt.subplot(1, 2, 2)\n",
                "sns.boxplot(y=df['Square_Footage'], color='#10B981')\n",
                "plt.title(\"Boxplot of Square Footage\")\n",
                "plt.ylabel(\"Square Feet\")\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        # Section 12: EDA
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Section 12: Exploratory Data Analysis (EDA)\n",
                "fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n",
                "\n",
                "# 1. Price Distribution\n",
                "sns.histplot(df['House_Price'] / 1000, kde=True, ax=axes[0, 0], color='#2563EB')\n",
                "axes[0, 0].set_title(\"Distribution of House Prices ($ Thousands)\")\n",
                "axes[0, 0].set_xlabel(\"Price ($ Thousands)\")\n",
                "\n",
                "# 2. Square Footage vs Price\n",
                "sns.scatterplot(data=df, x='Square_Footage', y=df['House_Price']/1000, hue='Neighborhood_Quality', palette='viridis', ax=axes[0, 1])\n",
                "axes[0, 1].set_title(\"Square Footage vs Price (by Neighborhood Quality)\")\n",
                "axes[0, 1].set_ylabel(\"Price ($ Thousands)\")\n",
                "\n",
                "# 3. Neighborhood Quality vs Price\n",
                "sns.boxplot(data=df, x='Neighborhood_Quality', y=df['House_Price']/1000, ax=axes[1, 0], palette='Blues')\n",
                "axes[1, 0].set_title(\"Neighborhood Quality vs Price\")\n",
                "axes[1, 0].set_ylabel(\"Price ($ Thousands)\")\n",
                "\n",
                "# 4. Feature Correlation Matrix\n",
                "sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='Blues', ax=axes[1, 1])\n",
                "axes[1, 1].set_title(\"Feature Correlation Heatmap\")\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        # Section 13, 14, 15: Feature Selection, Feature Engineering, Train-Test Split
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Section 13 - 15: Feature Selection, Engineering & Train-Test Split\n",
                "feature_cols = [\n",
                "    'Square_Footage', \n",
                "    'Num_Bedrooms', \n",
                "    'Num_Bathrooms', \n",
                "    'Year_Built', \n",
                "    'Lot_Size', \n",
                "    'Garage_Size',\n",
                "    'Neighborhood_Quality'\n",
                "]\n",
                "target_col = 'House_Price'\n",
                "\n",
                "X = df[feature_cols]\n",
                "y = df[target_col]\n",
                "\n",
                "# Train-Test Split (80% Train, 20% Test, random_state=42)\n",
                "X_train, X_test, y_train, y_test = train_test_split(\n",
                "    X, y, test_size=0.2, random_state=42\n",
                ")\n",
                "\n",
                "print(f\"[*] Training set shape: {X_train.shape}\")\n",
                "print(f\"[*] Test set shape: {X_test.shape}\")"
            ]
        },
        # Section 16 & 17: Model Training & Evaluation
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Section 16 & 17: Model Training & Evaluation\n",
                "pipeline = Pipeline([\n",
                "    ('scaler', StandardScaler()),\n",
                "    ('regressor', LinearRegression())\n",
                "])\n",
                "\n",
                "# Train Model\n",
                "pipeline.fit(X_train, y_train)\n",
                "\n",
                "# Model Evaluation\n",
                "y_pred = pipeline.predict(X_test)\n",
                "\n",
                "r2 = r2_score(y_test, y_pred)\n",
                "mae = mean_absolute_error(y_test, y_pred)\n",
                "rmse = np.sqrt(mean_squared_error(y_test, y_pred))\n",
                "\n",
                "print(\"=\" * 50)\n",
                "print(\"MODEL EVALUATION RESULTS (TEST SET):\")\n",
                "print(f\"   R² Score (Accuracy) : {r2:.4f}\")\n",
                "print(f\"   MAE                 : ${mae:,.2f}\")\n",
                "print(f\"   RMSE                : ${rmse:,.2f}\")\n",
                "print(\"=\" * 50)"
            ]
        },
        # Section 18, 19, 20: Predictions, Visualizations & Residuals
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Section 18 - 20: Actual vs Predicted & Residual Analysis\n",
                "residuals = y_test - y_pred\n",
                "\n",
                "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
                "\n",
                "# Actual vs Predicted Scatter Plot\n",
                "axes[0].scatter(y_test / 1000, y_pred / 1000, alpha=0.7, color='#2563EB')\n",
                "axes[0].plot([y_test.min()/1000, y_test.max()/1000], [y_test.min()/1000, y_test.max()/1000], 'r--', lw=2)\n",
                "axes[0].set_xlabel(\"Actual Price ($ Thousands)\")\n",
                "axes[0].set_ylabel(\"Predicted Price ($ Thousands)\")\n",
                "axes[0].set_title(\"Actual vs Predicted House Prices\")\n",
                "\n",
                "# Residuals Histogram\n",
                "sns.histplot(residuals / 1000, kde=True, ax=axes[1], color='#DC2626')\n",
                "axes[1].set_xlabel(\"Residual Error ($ Thousands)\")\n",
                "axes[1].set_ylabel(\"Frequency\")\n",
                "axes[1].set_title(\"Distribution of Residual Errors\")\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        # Section 21 & 22: Model Saving & Conclusion
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Section 21 & 22: Model Persistence & Final Validation\n",
                "os.makedirs(\"models\", exist_ok=True)\n",
                "joblib.dump(pipeline, os.path.join(\"models\", \"model.pkl\"))\n",
                "joblib.dump(pipeline, \"model.pkl\")\n",
                "print(\"[*] Model saved successfully to 'models/model.pkl' and 'model.pkl'.\")\n",
                "\n",
                "# Perform sample inference\n",
                "sample_data = pd.DataFrame([{\n",
                "    'Square_Footage': 2500,\n",
                "    'Num_Bedrooms': 3,\n",
                "    'Num_Bathrooms': 2,\n",
                "    'Year_Built': 2005,\n",
                "    'Lot_Size': 3.5,\n",
                "    'Garage_Size': 2,\n",
                "    'Neighborhood_Quality': 8\n",
                "}])\n",
                "predicted_val = pipeline.predict(sample_data)[0]\n",
                "print(f\"[*] Sample Prediction for 2500 sq.ft property: ${predicted_val:,.2f}\")"
            ]
        }
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    os.makedirs("notebooks", exist_ok=True)
    nb_path_1 = "AditiKongre_HousePricePrediction.ipynb"
    nb_path_2 = os.path.join("notebooks", "AditiKongre_HousePricePrediction.ipynb")
    preserve_outputs(cells, nb_path_1)
    preserve_outputs(cells, nb_path_2)

    with open(nb_path_1, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    with open(nb_path_2, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print(f"[*] Jupyter Notebook updated at '{nb_path_1}' and '{nb_path_2}'.")

if __name__ == '__main__':
    create_notebook()
