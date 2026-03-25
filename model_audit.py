import pandas as pd
import numpy as np
import json
import pickle
import os
import sys
import xgboost as xgb
from pathlib import Path

# Self-contained logic to avoid import hell
class SimpleXGBPredictor:
    def __init__(self, model, features):
        self._model = model
        self._features = features
    def predict(self, df):
        X = pd.DataFrame(index=df.index)
        for col in self._features:
            X[col] = df[col].iloc[0] if col in df.columns else 0.0
        dmat = xgb.DMatrix(X.values.astype(np.float32))
        raw = self._model.predict(dmat)
        return float(raw[-1]) if len(raw) > 0 else 0.5

class SimpleRFPredictor:
    def __init__(self, model, features):
        self._model = model
        self._features = features
    def predict(self, df):
        X = pd.DataFrame(index=df.index)
        for col in self._features:
            X[col] = df[col].iloc[0] if col in df.columns else 0.0
        # RF models usually expect specific column order
        # Make sure X has columns in self._features order
        X = X[self._features]
        raw = self._model.predict_proba(X)
        return float(raw[-1][1]) if len(raw) > 0 else 0.5

# Load Dataset
df = pd.read_csv("india_pakistan_flood_balancednew.csv")
floods = df[df['flood_occurred'] == 1]

# Load Models
xgb_booster = xgb.Booster()
xgb_booster.load_model("backend/trained_models/xgb_all.json")
with open("backend/trained_models/xgb_all_features.json", "r") as f:
    xgb_features = json.load(f)
xgb_predictor = SimpleXGBPredictor(xgb_booster, xgb_features)

with open("backend/trained_models/rf_all.pkl", "rb") as f:
    rf_model_obj = pickle.load(f)
with open("backend/trained_models/rf_all_features.json", "r") as f:
    rf_features = json.load(f)
rf_predictor = SimpleRFPredictor(rf_model_obj, rf_features)

def evaluate_samples(samples, label="Random Samples"):
    res = []
    for i in range(len(samples)):
        row = samples.iloc[[i]]
        x_score = xgb_predictor.predict(row)
        r_score = rf_predictor.predict(row)
        e_score = 0.7 * x_score + 0.3 * r_score
        res.append(e_score)
    
    print(f"\n--- {label} ({len(samples)} samples) ---")
    print(f"Mean Ensemble Score: {np.mean(res):.4f}")
    print(f"Max: {max(res):.4f} | Min: {min(res):.4f}")
    print(f"Samples >= 0.5 (HIGH): {sum(1 for s in res if s >= 0.5)}")
    print(f"Samples >= 0.75 (CRITICAL): {sum(1 for s in res if s >= 0.75)}")

# 1. Random
evaluate_samples(df.sample(100), "Random Samples")

# 2. Floods
evaluate_samples(floods.sample(100), "Known Floods")

# 3. High Rain
high_rain = floods.sort_values(by=['rain_mm_weekly'], ascending=False).head(20)
evaluate_samples(high_rain, "Top 20 Rain Events")

# 4. Extreme Synthesis (What the user might have tried)
extreme = pd.DataFrame([{
    "rain_mm_weekly": 450,
    "rain_mm_monthly": 1000,
    "elevation_m": 5,
    "slope_degree": 0.1,
    "dist_major_river_km": 0.1,
    "waterbody_nearby": 1,
    "dam_count_50km": 5,
    "terrain_type": "urban",
    "lat": 18.0,
    "lon": 72.0
}])
evaluate_samples(extreme, "EXTREME Synthesis Case")
