# Rising Waters — Flood Prediction System

A machine learning–powered flood early-warning system. Historical weather
data (annual rainfall, seasonal/monsoon rainfall, cloud visibility, and
related meteorological features) is used to train and compare four
classification algorithms. The best-performing model is served through a
Flask web application so disaster-response teams can get instant flood
risk predictions.

## Features

- **Four classification algorithms compared:** Decision Tree, Random
  Forest, K-Nearest Neighbours (KNN), and XGBoost.
- **Automatic model selection:** the highest-accuracy model on held-out
  test data is saved and used by the web app.
- **Flask web app:** simple form for entering current readings, with a
  visual risk gauge and probability score.
- **Model info page:** shows accuracy comparison across all four
  algorithms.
- **Cloud-ready:** structured for deployment to IBM Cloud (or any
  platform that runs a Python/Flask app), using `gunicorn` as the
  production WSGI server.

## Project structure

```
flood_prediction_project/
├── app.py                  # Flask application
├── train_model.py          # Trains & compares all 4 models, saves the best
├── requirements.txt
├── Procfile                 # for IBM Cloud / Heroku-style deployment
├── data/
│   ├── generate_dataset.py # generates the synthetic training dataset
│   └── flood_data.csv      # historical weather + flood-occurrence data
├── model/
│   ├── best_model.pkl       # trained model (auto-selected)
│   ├── scaler.pkl           # fitted StandardScaler
│   ├── metadata.json        # model name, accuracy, feature list
│   ├── model_comparison.png # accuracy bar chart
│   └── confusion_matrix.png # confusion matrix for the best model
├── templates/
│   ├── base.html
│   ├── index.html            # prediction form
│   ├── result.html           # prediction result + risk gauge
│   └── about.html             # model comparison / info page
└── static/
    ├── style.css
    └── model_comparison.png
```

## About the dataset

No dataset file was supplied with the original project brief, so
`data/generate_dataset.py` builds a realistic **synthetic** historical
weather dataset (3,000 records) with the features described in the
brief — annual rainfall, seasonal rainfall, cloud visibility — plus a
few supporting meteorological features (humidity, temperature, river
discharge, soil moisture) that make the model more robust. The flood
label is generated from a weighted, noisy combination of these features
so the relationships are realistic but not trivially linear.

**To use a real dataset instead:** replace `data/flood_data.csv` with
your own historical data, keeping the same column names (or update
`FEATURE_COLUMNS` in `train_model.py` and `app.py` to match), then
re-run `train_model.py`.

## Setup & run locally

```bash
# 1. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Regenerate the dataset and retrain models
python data/generate_dataset.py
python train_model.py

# 4. Run the web app
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

## Retraining

Running `train_model.py` will:
1. Load `data/flood_data.csv`
2. Split into train/test sets (80/20, stratified)
3. Scale features with `StandardScaler`
4. Train Decision Tree, Random Forest, KNN, and XGBoost
5. Print accuracy + classification report for each
6. Save the best-performing model, scaler, and metadata to `model/`
7. Save a comparison bar chart and confusion matrix as PNGs

## Deploying to IBM Cloud

This app is structured to deploy as a standard Python/Flask app (e.g. on
**IBM Cloud Foundry** or **IBM Code Engine**):

1. Make sure `Procfile` and `requirements.txt` are present (they are).
2. Push the app using the IBM Cloud CLI, e.g.:
   ```bash
   ibmcloud cf push rising-waters -b python_buildpack
   ```
   or containerize it for Code Engine:
   ```bash
   docker build -t rising-waters .
   ibmcloud ce application create --name rising-waters --image <your-registry>/rising-waters
   ```
3. The `Procfile` runs the app with `gunicorn`, which is production-ready
   (unlike Flask's built-in dev server).

## Notes on the demo model performance

The included synthetic dataset yields around 80–85% accuracy across the
four models, with Random Forest performing best on this particular
sample. If you train on a real, larger historical flood dataset with
stronger feature engineering, accuracy — including the 96%+ XGBoost
result referenced in the original project scenario — is achievable;
results depend heavily on the quality and size of the historical
dataset you train on.

## Tech stack

- Python, scikit-learn, XGBoost, NumPy, pandas, Matplotlib
- Flask (web framework)
- HTML/CSS (no JS framework required)
