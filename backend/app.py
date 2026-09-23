import os
import sys
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for Streamlit frontend

# Path to the saved model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model.pkl")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model.pkl")

# Load model globally on startup
try:
    model_pipeline = joblib.load(MODEL_PATH)
    print(f"[*] Flask Backend: Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    model_pipeline = None
    print(f"[!] Flask Backend Warning: Model failed to load: {e}")

FEATURE_COLS = [
    'Square_Footage', 
    'Num_Bedrooms', 
    'Num_Bathrooms', 
    'Year_Built', 
    'Lot_Size', 
    'Garage_Size',
    'Neighborhood_Quality'
]

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify backend status and model readiness."""
    status = {
        "status": "healthy" if model_pipeline is not None else "degraded",
        "service": "House Price Prediction REST API",
        "model_loaded": model_pipeline is not None,
        "features_required": FEATURE_COLS
    }
    return jsonify(status), 200 if model_pipeline is not None else 503

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint that accepts property characteristics and returns estimated price."""
    if model_pipeline is None:
        return jsonify({
            "success": False,
            "error": "Model is not loaded on server. Please check backend setup."
        }), 500

    if not request.is_json:
        return jsonify({
            "success": False,
            "error": "Invalid request content-type. Expected application/json."
        }), 400

    data = request.get_json()
    if not data:
        return jsonify({
            "success": False,
            "error": "Request body is empty."
        }), 400

    # Input Validation
    missing_fields = [col for col in FEATURE_COLS if col not in data]
    if missing_fields:
        return jsonify({
            "success": False,
            "error": f"Missing required feature fields: {missing_fields}"
        }), 400

    # Type & Range Validations
    try:
        sq_ft = float(data['Square_Footage'])
        beds = float(data['Num_Bedrooms'])
        baths = float(data['Num_Bathrooms'])
        year = float(data['Year_Built'])
        lot = float(data['Lot_Size'])
        garage = float(data['Garage_Size'])
        neighborhood = float(data['Neighborhood_Quality'])
    except (ValueError, TypeError):
        return jsonify({
            "success": False,
            "error": "All input features must be numerical values."
        }), 400

    # Value sanity checks
    if sq_ft <= 0:
        return jsonify({"success": False, "error": "Square_Footage must be greater than 0."}), 400
    if beds < 0 or baths < 0 or lot < 0 or garage < 0:
        return jsonify({"success": False, "error": "Bedrooms, bathrooms, lot size, and garage size cannot be negative."}), 400
    if neighborhood < 1 or neighborhood > 10:
        return jsonify({"success": False, "error": "Neighborhood_Quality must be between 1 and 10."}), 400

    # Construct DataFrame for Model Pipeline
    input_df = pd.DataFrame([{
        'Square_Footage': sq_ft,
        'Num_Bedrooms': beds,
        'Num_Bathrooms': baths,
        'Year_Built': year,
        'Lot_Size': lot,
        'Garage_Size': garage,
        'Neighborhood_Quality': neighborhood
    }])

    try:
        prediction = float(model_pipeline.predict(input_df)[0])
        prediction = max(prediction, 10000.0)

        response = {
            "success": True,
            "predicted_price": round(prediction, 2),
            "formatted_price": f"${prediction:,.2f}",
            "currency": "USD",
            "inputs": {
                'Square_Footage': sq_ft,
                'Num_Bedrooms': beds,
                'Num_Bathrooms': baths,
                'Year_Built': year,
                'Lot_Size': lot,
                'Garage_Size': garage,
                'Neighborhood_Quality': neighborhood
            }
        }
        return jsonify(response), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": "An error occurred while generating the prediction.",
            "details": str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"[*] Starting Flask REST API on http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
