"""
app.py
------
Flask web application for the Flood Prediction System.

Loads the best-performing trained model (selected in train_model.py) and
serves a form where meteorologists / disaster-response coordinators can
enter current weather readings to get an instant flood risk prediction.

Run locally:
    python app.py
Then open http://127.0.0.1:5000
"""

import json
import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

# --- Load trained model, scaler, and metadata at startup ----------------
with open(os.path.join(MODEL_DIR, "best_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(MODEL_DIR, "metadata.json"), "r") as f:
    metadata = json.load(f)

FEATURE_COLUMNS = metadata["feature_columns"]


def build_feature_vector(form):
    """Reads form inputs (in the fixed feature order) and returns a
    single-row DataFrame with matching column names (avoids sklearn's
    'X does not have valid feature names' warning)."""
    values = {col: [float(form.get(col))] for col in FEATURE_COLUMNS}
    return pd.DataFrame(values, columns=FEATURE_COLUMNS)


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        model_name=metadata["best_model_name"],
        model_accuracy=metadata["best_model_accuracy"],
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        X = build_feature_vector(request.form)
        X_scaled = scaler.transform(X)

        prediction = int(model.predict(X_scaled)[0])
        # probability of the "flood" class, if the model supports it
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X_scaled)[0][1]) * 100
        else:
            proba = None

        result = {
            "prediction": "Flood Likely" if prediction == 1 else "No Flood Expected",
            "risk_level": risk_level(proba, prediction),
            "probability": round(proba, 2) if proba is not None else None,
            "inputs": dict(request.form),
        }

        return render_template(
            "result.html",
            result=result,
            model_name=metadata["best_model_name"],
            model_accuracy=metadata["best_model_accuracy"],
        )
    except Exception as e:
        return render_template("result.html", error=str(e))


def risk_level(proba, prediction):
    if proba is None:
        return "High" if prediction == 1 else "Low"
    if proba >= 75:
        return "Very High"
    elif proba >= 50:
        return "High"
    elif proba >= 25:
        return "Moderate"
    else:
        return "Low"


@app.route("/about")
def about():
    return render_template(
        "about.html",
        metadata=metadata,
    )


if __name__ == "__main__":
    # For local development. For IBM Cloud / production deployment,
    # use a WSGI server (gunicorn) - see README.md
    app.run(debug=True, host="0.0.0.0", port=5000)
