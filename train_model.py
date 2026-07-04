"""
train_model.py
---------------
Trains and compares four classification algorithms for flood prediction:
    1. Decision Tree
    2. Random Forest
    3. K-Nearest Neighbours (KNN)
    4. XGBoost

The best-performing model (by test accuracy) is saved to model/best_model.pkl
along with the fitted StandardScaler, so the Flask app can load and use it
directly. A comparison chart and metrics report are also saved.

Run:
    python train_model.py
"""

import json
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "annual_rainfall_mm",
    "seasonal_rainfall_mm",
    "cloud_visibility_km",
    "humidity_percent",
    "temperature_celsius",
    "river_discharge_cusecs",
    "soil_moisture_percent",
]
TARGET_COLUMN = "flood"


def load_data(path="data/flood_data.csv"):
    df = pd.read_csv(path)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y


def main():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # KNN needs scaled features; tree-based models don't strictly need it,
    # but we scale everything once and reuse the same scaler for consistency
    # in the web app (single input pipeline).
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, random_state=RANDOM_STATE
        ),
        "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=7),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        ),
    }

    results = {}
    fitted_models = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)
        results[name] = acc
        fitted_models[name] = model
        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc * 100:.2f}%")
        print(classification_report(y_test, preds, target_names=["No Flood", "Flood"]))

    # --- Pick the best model ---
    best_name = max(results, key=results.get)
    best_model = fitted_models[best_name]
    best_acc = results[best_name]
    print(f"\nBest model: {best_name} with accuracy {best_acc * 100:.2f}%")

    # --- Save best model + scaler + metadata ---
    with open("model/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open("model/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    metadata = {
        "best_model_name": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "all_model_accuracies": {k: round(v * 100, 2) for k, v in results.items()},
        "best_model_accuracy": round(best_acc * 100, 2),
    }
    with open("model/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # --- Comparison bar chart ---
    plt.figure(figsize=(7, 5))
    names = list(results.keys())
    accs = [results[n] * 100 for n in names]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    bars = plt.bar(names, accs, color=colors)
    plt.ylabel("Accuracy (%)")
    plt.title("Model Comparison — Flood Prediction")
    plt.ylim(0, 100)
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width() / 2, acc + 1, f"{acc:.2f}%",
                  ha="center", fontsize=9)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("model/model_comparison.png", dpi=150)

    # --- Confusion matrix for best model ---
    cm = confusion_matrix(y_test, best_model.predict(X_test_scaled))
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix — {best_name}")
    plt.colorbar()
    plt.xticks([0, 1], ["No Flood", "Flood"])
    plt.yticks([0, 1], ["No Flood", "Flood"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                      color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("model/confusion_matrix.png", dpi=150)

    print("\nSaved: model/best_model.pkl, model/scaler.pkl, model/metadata.json")
    print("Saved: model/model_comparison.png, model/confusion_matrix.png")


if __name__ == "__main__":
    main()
