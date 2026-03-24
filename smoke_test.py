"""
Smoke test: verifies all prediction agent modules import correctly
and that the XGBoostPredictor._prepare_input works as expected.
Run from backend/ with: venv\Scripts\python.exe ..\smoke_test.py
"""
import sys
sys.path.insert(0, '.')  # backend/ is the Python root

print("Testing imports...")

try:
    from agents.prediction.model_registry import (
        ModelRegistry, ALL_FEATURES, WEATHER_FEATURES,
        HYDRO_FEATURES, TERRAIN_FEATURES, FEATURE_REGISTRY
    )
    print(f"  [OK] model_registry  |  ALL_FEATURES ({len(ALL_FEATURES)}): {ALL_FEATURES}")
except Exception as e:
    print(f"  [FAIL] model_registry: {e}")

try:
    from agents.prediction.model_selector import ModelSelector, infer_group_from_features
    # Test group inference
    w_only = ["rain_mm_weekly", "temp_c_mean"]
    both   = ["rain_mm_weekly", "dam_count_50km", "elevation_m"]
    print(f"  [OK] model_selector  |  infer(weather_only)={infer_group_from_features(w_only)}"
          f"  infer(all)={infer_group_from_features(both)}")
except Exception as e:
    print(f"  [FAIL] model_selector: {e}")

try:
    from agents.prediction.models.xgboost_predictor import XGBoostPredictor
    import pandas as pd, numpy as np
    # Test _prepare_input alignment
    predictor = XGBoostPredictor(model=None, feature_cols=ALL_FEATURES)
    df_test = pd.DataFrame([{
        "rain_mm_weekly": 50.0,
        "temp_c_mean": 30.0,
        "elevation_m": 100.0,
        # rest missing — should be zero-filled
    }])
    X = predictor._prepare_input(df_test)
    assert X.shape == (1, len(ALL_FEATURES)), f"Shape mismatch: {X.shape}"
    print(f"  [OK] XGBoostPredictor._prepare_input  |  shape={X.shape}  dtype={X.dtype}")
    # Check that missing features are 0
    feat_idx = {f: i for i, f in enumerate(ALL_FEATURES)}
    assert X[0, feat_idx["rain_mm_weekly"]] == 50.0
    assert X[0, feat_idx["dam_count_50km"]] == 0.0   # missing → zero
    print(f"  [OK] Feature alignment correct (rain={X[0,0]}, dam_count={X[0, feat_idx['dam_count_50km']]})")
except Exception as e:
    print(f"  [FAIL] XGBoostPredictor: {e}")

try:
    from agents.prediction.models.random_forest_predictor import RandomForestPredictor
    print(f"  [OK] RandomForestPredictor import")
except Exception as e:
    print(f"  [FAIL] RandomForestPredictor: {e}")

try:
    from agents.prediction.ensemble import EnsembleCombiner
    print(f"  [OK] EnsembleCombiner import")
except Exception as e:
    print(f"  [FAIL] EnsembleCombiner: {e}")

try:
    from agents.prediction.prediction_agent import PredictionAgent, TERRAIN_TYPE_ENCODING
    print(f"  [OK] PredictionAgent import  |  terrain_types: {list(TERRAIN_TYPE_ENCODING.keys())}")
except Exception as e:
    print(f"  [FAIL] PredictionAgent: {e}")

# Check actual trained model files exist
from pathlib import Path
import json
models_dir = Path("trained_models")
print(f"\nTrained model files in {models_dir.resolve()}:")
if models_dir.exists():
    for f in sorted(models_dir.glob("*")):
        kb = f.stat().st_size // 1024
        icon = "✓" if f.suffix in (".json", ".pkl") else " "
        print(f"  {icon} {f.name:<45} {kb:>6} KB")
    # Check feature lists are consistent
    feat_reg = json.load(open(models_dir / "feature_registry.json"))
    print(f"\n  Feature registry groups: {list(feat_reg.keys())}")
    print(f"  'all' features: {feat_reg['all']}")
else:
    print("  [WARN] trained_models/ directory not found!")

print("\n[SMOKE TEST COMPLETE]")
