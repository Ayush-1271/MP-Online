"""
app.py
------
Flask REST API that loads the trained heart-disease model and serves
predictions.

Endpoints:
    GET  /            -> simple HTML form (optional, for manual testing)
    GET  /health       -> health check, returns {"status": "ok"}
    POST /predict       -> accepts patient details as JSON, returns a
                            JSON prediction

Example request body for /predict:
{
    "age": 52,
    "sex": 1,
    "cp": 0,
    "trestbps": 125,
    "chol": 212,
    "fbs": 0,
    "restecg": 1,
    "thalach": 168,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 2,
    "ca": 2,
    "thal": 3
}

Example response:
{
    "prediction": "Heart Disease Detected",
    "prediction_label": 1,
    "probability": 0.83
}
"""

import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ---------------------------------------------------------------
# Load the trained model at startup
# ---------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
FEATURES = artifact["features"]


@app.route("/")
def home():
    """Simple index page (optional UI)."""
    try:
        return render_template("index.html")
    except Exception:
        return jsonify({
            "message": "Heart Disease Prediction API is running.",
            "usage": "POST patient details as JSON to /predict",
            "required_fields": FEATURES,
        })


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True)

    if not data:
        return jsonify({"error": "No input JSON provided"}), 400

    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    try:
        input_df = pd.DataFrame([{f: data[f] for f in FEATURES}])
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][int(pred)]

        result = {
            "prediction": "Heart Disease Detected" if pred == 1 else "No Heart Disease Detected",
            "prediction_label": int(pred),
            "probability": round(float(proba), 4),
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
