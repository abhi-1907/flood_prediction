import sys, io
sys.path.insert(0, '.')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)

import pandas as pd
import numpy as np

n = 100
np.random.seed(42)
df_raw = pd.DataFrame({
    "country": ["India"]*n, "name": [f"loc_{i}" for i in range(n)],
    "week": ["2001-06"]*n, "flood_occurred": np.random.randint(0,2,n),
    "terrain_type": ["urban"]*n,
    "rain_mm_weekly": np.random.exponential(20,n),
    "temp_c_mean": np.random.normal(30,5,n),
    "rh_percent_mean": np.random.uniform(40,90,n),
    "wind_ms_mean": np.random.exponential(3,n),
    "rain_mm_monthly": np.random.exponential(80,n),
    "dam_count_50km": np.random.randint(0,10,n).astype(float),
    "dist_major_river_km": np.random.exponential(15,n),
    "waterbody_nearby": np.random.randint(0,2,n).astype(float),
    "lat": np.random.uniform(20,35,n), "lon": np.random.uniform(68,88,n),
    "elevation_m": np.random.exponential(100,n),
    "slope_degree": np.random.exponential(2,n),
    "terrain_type_encoded": np.random.randint(0,4,n).astype(float),
})

print(f"RAW SHAPE: {df_raw.shape}")

# 1. Feature Engineer
from agents.preprocessing.transformers.feature_engineer import FeatureEngineer
class MS:
    engineer_features = True
class AL:
    def log(self, **kw): pass

fe = FeatureEngineer()
df_eng = fe.engineer(df_raw.copy(), MS(), AL())
new_cols = sorted(set(df_eng.columns) - set(df_raw.columns))
print(f"[OK] FeatureEngineer added {len(new_cols)} features: {new_cols}")
print(f"     Shape after engineering: {df_eng.shape}")

# 2. Normalizer
from agents.preprocessing.transformers.normalizer import NO_SCALE_COLS
no_scale_ok = all(c in NO_SCALE_COLS for c in ["lat", "lon", "flood_occurred", "terrain_type_encoded", "waterbody_nearby"])
print(f"[{'OK' if no_scale_ok else 'FAIL'}] NO_SCALE_COLS contains all binary/categorical cols")
print(f"     NO_SCALE_COLS: {sorted(NO_SCALE_COLS)}")

# 3. Strategy selector
from agents.preprocessing.strategy_selector import PROTECTED_COLUMNS
prot_ok = all(c in PROTECTED_COLUMNS for c in ["rain_mm_weekly", "elevation_m", "dist_major_river_km"])
print(f"[{'OK' if prot_ok else 'FAIL'}] PROTECTED_COLUMNS contains real dataset columns")
print(f"     PROTECTED_COLUMNS: {sorted(PROTECTED_COLUMNS)}")

# 4. Preprocessing agent imports
from agents.preprocessing.preprocessing_agent import DROP_BEFORE_PREPROCESS, PreprocessingAgent
print(f"[OK] DROP_BEFORE_PREPROCESS: {DROP_BEFORE_PREPROCESS}")
dropped = [c for c in DROP_BEFORE_PREPROCESS if c in df_raw.columns]
print(f"     Would drop from test df: {dropped}")

# 5. ALL_FEATURES coverage
from agents.prediction.model_registry import ALL_FEATURES
junk = set(DROP_BEFORE_PREPROCESS)
df_clean = df_raw.drop(columns=[c for c in junk if c in df_raw.columns])
df_clean = df_clean.drop(columns=df_clean.select_dtypes("object").columns.tolist(), errors="ignore")
present = [f for f in ALL_FEATURES if f in df_clean.columns]
missing = [f for f in ALL_FEATURES if f not in df_clean.columns]
print(f"[{'OK' if not missing else 'WARN'}] {len(present)}/{len(ALL_FEATURES)} training features present after junk drop")
if missing:
    print(f"     MISSING: {missing}")
print(f"     ALL_FEATURES: {ALL_FEATURES}")

print("\n[DONE]")
