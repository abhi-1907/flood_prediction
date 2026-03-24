"""
train_models.py
===============
Offline model training script for FloodSense AI.

Run from the project root (e:/flood-prediction):
    cd e:/flood-prediction
    backend\\venv\\Scripts\\python.exe train_models.py

Reads: india_pakistan_flood_balancednew.csv
Saves to: backend/trained_models/

Feature Groups (from real dataset analysis)
-------------------------------------------
Weather:      rain_mm_weekly, temp_c_mean, rh_percent_mean,
              wind_ms_mean, rain_mm_monthly
Hydrological: dam_count_50km, dist_major_river_km, waterbody_nearby
Terrain:      lat, lon, elevation_m, slope_degree, terrain_type (encoded)

Trained Models
--------------
XGBoost variants:
  xgb_weather.json                  <- weather only
  xgb_hydro.json                    <- hydro only
  xgb_terrain.json                  <- terrain only
  xgb_weather_hydro.json            <- weather + hydro
  xgb_weather_terrain.json          <- weather + terrain
  xgb_hydro_terrain.json            <- hydro + terrain
  xgb_all.json                      <- all features (primary model)

Random Forest variants:
  rf_weather.pkl
  rf_hydro.pkl
  rf_terrain.pkl
  rf_weather_hydro.pkl
  rf_weather_terrain.pkl
  rf_hydro_terrain.pkl
  rf_all.pkl                        <- all features (primary model)
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent
DATASET     = ROOT / "india_pakistan_flood_balancednew.csv"
MODELS_DIR  = ROOT / "backend" / "trained_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL  = "flood_occurred"

# Columns to completely ignore
DROP_COLS   = ["country", "name", "week"]

# ── Feature group definitions ──────────────────────────────────────────────────
WEATHER_FEATURES = [
    "rain_mm_weekly",
    "temp_c_mean",
    "rh_percent_mean",
    "wind_ms_mean",
    "rain_mm_monthly",
]

HYDRO_FEATURES = [
    "dam_count_50km",
    "dist_major_river_km",
    "waterbody_nearby",
]

TERRAIN_FEATURES = [
    "lat",
    "lon",
    "elevation_m",
    "slope_degree",
    "terrain_type_encoded",   # label-encoded version of terrain_type
]

ALL_FEATURES = WEATHER_FEATURES + HYDRO_FEATURES + TERRAIN_FEATURES


# ── Feature group combos to train ─────────────────────────────────────────────
FEATURE_GROUPS: Dict[str, List[str]] = {
    "weather":          WEATHER_FEATURES,
    "hydro":            HYDRO_FEATURES,
    "terrain":          TERRAIN_FEATURES,
    "weather_hydro":    WEATHER_FEATURES + HYDRO_FEATURES,
    "weather_terrain":  WEATHER_FEATURES + TERRAIN_FEATURES,
    "hydro_terrain":    HYDRO_FEATURES + TERRAIN_FEATURES,
    "all":              ALL_FEATURES,
}


# ── Data loading & preprocessing ───────────────────────────────────────────────

def load_and_prepare(path: Path) -> pd.DataFrame:
    """Loads the CSV, encodes categoricals, and drops irrelevant columns."""
    print(f"[DATA] Loading {path.name}...")
    df = pd.read_csv(path)
    print(f"[DATA] Shape: {df.shape}")
    print(f"[DATA] Columns: {list(df.columns)}")

    # Drop ignore columns
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    # Label-encode terrain_type (urban, rural, forest, …)
    if "terrain_type" in df.columns:
        le = LabelEncoder()
        df["terrain_type_encoded"] = le.fit_transform(df["terrain_type"].fillna("unknown"))
        # Save the encoder classes for serving
        classes_path = MODELS_DIR / "terrain_type_classes.json"
        json.dump(list(le.classes_), open(classes_path, "w"))
        print(f"[DATA] terrain_type classes: {list(le.classes_)} → saved to {classes_path}")
        df = df.drop(columns=["terrain_type"])

    # Fill any remaining NaNs
    df = df.fillna(0)

    print(f"[DATA] Target distribution:\n{df[TARGET_COL].value_counts()}\n")
    return df


def extract_Xy(df: pd.DataFrame, features: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extracts X and y, using only available features from the list."""
    available = [f for f in features if f in df.columns]
    missing   = [f for f in features if f not in df.columns]
    if missing:
        print(f"  [WARN] Missing features (will skip): {missing}")
    X = df[available].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(int)
    return X, y, available


# ── XGBoost training ───────────────────────────────────────────────────────────

def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    label:   str,
) -> xgb.Booster:
    """Trains an XGBoost classifier and returns the Booster."""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val,   label=y_val)

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    params = {
        "objective":         "binary:logistic",
        "eval_metric":       ["logloss", "auc"],
        "max_depth":         5,
        "learning_rate":     0.05,
        "subsample":         0.85,
        "colsample_bytree":  0.85,
        "min_child_weight":  5,
        "gamma":             0.1,
        "reg_alpha":         0.1,
        "reg_lambda":        1.0,
        "scale_pos_weight":  scale_pos_weight,
        "seed":              42,
    }

    evals = [(dtrain, "train"), (dval, "val")]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=400,
        evals=evals,
        early_stopping_rounds=25,
        verbose_eval=50,
    )
    print(f"  [XGB-{label}] Best iteration: {model.best_iteration}")
    return model


# ── Random Forest training ─────────────────────────────────────────────────────

def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    label:   str,
) -> RandomForestClassifier:
    """Trains a Random Forest classifier."""
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        oob_score=True,
    )
    model.fit(X_train, y_train)
    print(f"  [RF-{label}] OOB Score: {model.oob_score_:.4f}")
    return model


# ── Evaluation helper ──────────────────────────────────────────────────────────

def evaluate(model, X_test: np.ndarray, y_test: np.ndarray, label: str, is_xgb: bool = False) -> Dict:
    """Prints and returns evaluation metrics."""
    if is_xgb:
        dtest = xgb.DMatrix(X_test)
        y_prob = model.predict(dtest)
        y_pred = (y_prob >= 0.5).astype(int)
    else:
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)
    print(f"\n  === [{label}] Evaluation ===")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["No Flood", "Flood"]))
    return {"accuracy": round(acc, 4), "roc_auc": round(auc, 4)}


# ── Save helpers ───────────────────────────────────────────────────────────────

def save_xgb(model: xgb.Booster, features: List[str], name: str, metrics: Dict) -> None:
    model_path   = MODELS_DIR / f"{name}.json"
    feature_path = MODELS_DIR / f"{name}_features.json"
    meta_path    = MODELS_DIR / f"{name}_meta.json"

    model.save_model(str(model_path))
    json.dump(features, open(feature_path, "w"), indent=2)
    json.dump(metrics,  open(meta_path,    "w"), indent=2)
    print(f"  [SAVE] {model_path.name}  |  features: {features}")


def save_rf(model: RandomForestClassifier, features: List[str], name: str, metrics: Dict) -> None:
    model_path   = MODELS_DIR / f"{name}.pkl"
    feature_path = MODELS_DIR / f"{name}_features.json"
    meta_path    = MODELS_DIR / f"{name}_meta.json"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    json.dump(features, open(feature_path, "w"), indent=2)
    json.dump(metrics,  open(meta_path,    "w"), indent=2)
    print(f"  [SAVE] {model_path.name}  |  features: {features}")


# ── Main training loop ─────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  FloodSense AI – Model Training Pipeline")
    print("=" * 70)

    df = load_and_prepare(DATASET)

    all_metrics: Dict[str, Dict] = {}

    for group_name, feature_list in FEATURE_GROUPS.items():
        print(f"\n{'='*60}")
        print(f"  Feature Group: {group_name.upper()}")
        print(f"  Features: {feature_list}")
        print(f"{'='*60}")

        X, y, actual_features = extract_Xy(df, feature_list)

        if X.shape[1] == 0:
            print(f"  [SKIP] No features available for group '{group_name}'")
            continue

        # Train/Val/Test split  (70 / 15 / 15)
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(
            X, y, test_size=0.30, random_state=42, stratify=y
        )
        X_val, X_te, y_val, y_te = train_test_split(
            X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp
        )

        print(f"  Train/Val/Test: {len(y_tr)} / {len(y_val)} / {len(y_te)}")

        # ── XGBoost ──────────────────────────────────────────────────────────
        print(f"\n  [XGBoost] Training on '{group_name}' features...")
        xgb_model = train_xgboost(X_tr, y_tr, X_val, y_val, group_name)
        xgb_metrics = evaluate(xgb_model, X_te, y_te, f"XGB-{group_name}", is_xgb=True)
        xgb_metrics["feature_group"] = group_name
        xgb_metrics["features"] = actual_features
        save_xgb(xgb_model, actual_features, f"xgb_{group_name}", xgb_metrics)
        all_metrics[f"xgb_{group_name}"] = xgb_metrics

        # ── Random Forest ─────────────────────────────────────────────────────
        print(f"\n  [RandomForest] Training on '{group_name}' features...")
        rf_model = train_random_forest(X_tr, y_tr, group_name)
        rf_metrics = evaluate(rf_model, X_te, y_te, f"RF-{group_name}", is_xgb=False)
        rf_metrics["feature_group"] = group_name
        rf_metrics["features"] = actual_features
        save_rf(rf_model, actual_features, f"rf_{group_name}", rf_metrics)
        all_metrics[f"rf_{group_name}"] = rf_metrics

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE – Summary")
    print("=" * 70)
    print(f"  {'Model':<35} {'Accuracy':>10} {'ROC-AUC':>10}")
    print(f"  {'-'*55}")
    for name, m in sorted(all_metrics.items(), key=lambda x: -x[1].get("roc_auc", 0)):
        print(f"  {name:<35} {m.get('accuracy', 0):>10.4f} {m.get('roc_auc', 0):>10.4f}")

    # Save consolidated metrics
    metrics_path = MODELS_DIR / "training_metrics.json"
    json.dump(all_metrics, open(metrics_path, "w"), indent=2)
    print(f"\n  Full metrics saved to: {metrics_path}")
    print(f"  All models saved in:   {MODELS_DIR}")
    print("=" * 70)

    # Save feature group registry (used by model_registry.py)
    registry_path = MODELS_DIR / "feature_registry.json"
    feature_registry = {
        "weather":  WEATHER_FEATURES,
        "hydro":    HYDRO_FEATURES,
        "terrain":  TERRAIN_FEATURES,
        "all":      ALL_FEATURES,
        "weather_hydro":   WEATHER_FEATURES + HYDRO_FEATURES,
        "weather_terrain": WEATHER_FEATURES + TERRAIN_FEATURES,
        "hydro_terrain":   HYDRO_FEATURES + TERRAIN_FEATURES,
    }
    json.dump(feature_registry, open(registry_path, "w"), indent=2)
    print(f"  Feature registry saved: {registry_path}")


if __name__ == "__main__":
    main()
