"""
Flask API for the heart disease model.

Loads model.pkl (trained in train_model.py / the notebook) and exposes:
    GET  /         - a simple form so I can test predictions in the browser
    GET  /health   - basic health check for render/uptime checks
    POST /predict  - takes patient data as JSON, returns a prediction

Sample input for /predict:
{
    "age": 52, "sex": 1, "cp": 0, "trestbps": 125, "chol": 212,
    "fbs": 0, "restecg": 1, "thalach": 168, "exang": 0,
    "oldpeak": 1.0, "slope": 2, "ca": 2, "thal": 3
}
"""

import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# load model once when the app starts (not per-request, that'd be slow)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
FEATURES = artifact["features"]  # order the model was trained on


@app.route("/")
def home():
    # just a form so people don't have to use curl/Postman to try it
    return render_template("index.html")


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
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        # keep same column order as training, otherwise sklearn gets confused
        input_df = pd.DataFrame([{f: data[f] for f in FEATURES}])
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][int(pred)]

        return jsonify({
            "prediction": "Heart Disease Detected" if pred == 1 else "No Heart Disease Detected",
            "prediction_label": int(pred),
            "probability": round(float(proba), 4),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Render sets PORT itself, locally it just falls back to 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
