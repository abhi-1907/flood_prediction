import pandas as pd
import numpy as np
import json
import pickle
import xgboost as xgb

# 1. Load Model
xgb_booster = xgb.Booster()
xgb_booster.load_model("backend/trained_models/xgb_all.json")
with open("backend/trained_models/xgb_all_features.json", "r") as f:
    features = json.load(f)

def get_pred(data_dict):
    X = pd.DataFrame([data_dict])
    aligned = pd.DataFrame(index=X.index)
    for col in features:
        aligned[col] = X[col].iloc[0] if col in X.columns else 0.0
    dmat = xgb.DMatrix(aligned.values.astype(np.float32))
    return float(xgb_booster.predict(dmat)[0])

print(f"Baseline (ALL ZEROS): {get_pred({}):.4f}")

# Extreme Case
extreme = {
    "rain_mm_weekly": 450,
    "rain_mm_monthly": 1000,
    "elevation_m": 5,
    "slope_degree": 0.1,
    "dist_major_river_km": 0.1,
    "waterbody_nearby": 1,
    "dam_count_50km": 5,
    "terrain_type_encoded": 3, # urban
    "lat": 18.0,
    "lon": 72.0
}
print(f"Extreme Case: {get_pred(extreme):.4f}")

# What if only Rain is provided?
print(f"Only Rain (450): {get_pred({'rain_mm_weekly': 450}):.4f}")

# What if Rain is high but Elevation is also high?
print(f"High Rain (450) + High Elevation (500m): {get_pred({'rain_mm_weekly': 450, 'elevation_m': 500}):.4f}")

# Feature Importance
importance = xgb_booster.get_score(importance_type='gain')
sorted_imp = sorted(importance.items(), key=lambda x: -x[1])
print("\nTop 5 Features (Gain):")
for k, v in sorted_imp[:5]:
    print(f"  {k}: {v:.2f}")
