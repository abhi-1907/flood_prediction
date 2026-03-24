"""
Model Registry – Central loader and registry for all ML models.

Manages loading, caching, and status of:
  - XGBoost classifiers (7 feature-group variants)
  - Random Forest classifiers (7 feature-group variants)

Feature groups trained on india_pakistan_flood_balancednew.csv:
  weather  : rain_mm_weekly, temp_c_mean, rh_percent_mean,
             wind_ms_mean, rain_mm_monthly
  hydro    : dam_count_50km, dist_major_river_km, waterbody_nearby
  terrain  : lat, lon, elevation_m, slope_degree, terrain_type_encoded

Model files are stored in backend/trained_models/ as:
  xgb_{group}.json          + xgb_{group}_features.json
  rf_{group}.pkl            + rf_{group}_features.json
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from agents.prediction.prediction_schemas import (
    LoadedModel,
    ModelStatus,
    ModelType,
    PredictionMode,
)
from utils.logger import logger


# ── Paths ─────────────────────────────────────────────────────────────────────

MODELS_DIR = Path(__file__).resolve().parents[2] / "trained_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Real feature groups (from training on actual dataset) ─────────────────────

WEATHER_FEATURES: List[str] = [
    "rain_mm_weekly",
    "temp_c_mean",
    "rh_percent_mean",
    "wind_ms_mean",
    "rain_mm_monthly",
]

HYDRO_FEATURES: List[str] = [
    "dam_count_50km",
    "dist_major_river_km",
    "waterbody_nearby",
]

TERRAIN_FEATURES: List[str] = [
    "lat",
    "lon",
    "elevation_m",
    "slope_degree",
    "terrain_type_encoded",
]

ALL_FEATURES: List[str] = WEATHER_FEATURES + HYDRO_FEATURES + TERRAIN_FEATURES

# Feature groups mapped by name
FEATURE_REGISTRY: Dict[str, List[str]] = {
    "weather":          WEATHER_FEATURES,
    "hydro":            HYDRO_FEATURES,
    "terrain":          TERRAIN_FEATURES,
    "weather_hydro":    WEATHER_FEATURES + HYDRO_FEATURES,
    "weather_terrain":  WEATHER_FEATURES + TERRAIN_FEATURES,
    "hydro_terrain":    HYDRO_FEATURES + TERRAIN_FEATURES,
    "all":              ALL_FEATURES,
}

# Primary models to load into memory by default (best performers)
PRIMARY_XGB_GROUP = "all"   # Override if a specialised group performs better
PRIMARY_RF_GROUP  = "all"

# Backward-compat: expose as ModelType keys
MODEL_PATHS: Dict[ModelType, Path] = {
    ModelType.XGBOOST:       MODELS_DIR / f"xgb_{PRIMARY_XGB_GROUP}.json",
    ModelType.RANDOM_FOREST: MODELS_DIR / f"rf_{PRIMARY_RF_GROUP}.pkl",
}

FEATURE_PATHS: Dict[ModelType, Path] = {
    ModelType.XGBOOST:       MODELS_DIR / f"xgb_{PRIMARY_XGB_GROUP}_features.json",
    ModelType.RANDOM_FOREST: MODELS_DIR / f"rf_{PRIMARY_RF_GROUP}_features.json",
}

# Minimum rows needed for a sequence model
MIN_TRAIN_ROWS = 50

# Fallback feature set if no model files exist yet
DEFAULT_FEATURES = ALL_FEATURES


class ModelRegistry:
    """
    Loads, caches, and provides access to all ML models.

    Primary models (loaded at startup):
      - XGBoost: xgb_all.json  (all feature groups combined)
      - RF:      rf_all.pkl   (all feature groups combined)

    Specialised models can be loaded on demand via load_group().
    """

    def __init__(self) -> None:
        self._models:   Dict[ModelType, Any]         = {}
        self._meta:     Dict[ModelType, LoadedModel] = {}
        self._features: Dict[ModelType, List[str]]   = {}

        # Secondary storage: models by feature-group name
        self._group_models:   Dict[str, Any]         = {}
        self._group_features: Dict[str, List[str]]   = {}

    # ── Public API ────────────────────────────────────────────────────────

    async def load_all(self, df: Optional[pd.DataFrame] = None) -> None:
        """Loads XGBoost and Random Forest primary models from disk."""
        logger.info("[ModelRegistry] Loading primary models...")
        for model_type in [ModelType.XGBOOST, ModelType.RANDOM_FOREST]:
            await self._load_single(model_type, df)

        loaded_count = sum(
            1 for m in self._meta.values() if m.status == ModelStatus.LOADED
        )
        logger.info(f"[ModelRegistry] {loaded_count}/2 primary models loaded.")
        logger.info(
            "[ModelRegistry] Available feature groups: "
            + ", ".join(FEATURE_REGISTRY.keys())
        )

    def get(self, model_type: ModelType) -> Optional[Any]:
        """Returns the primary model object, or None if not loaded."""
        return self._models.get(model_type)

    def get_meta(self, model_type: ModelType) -> Optional[LoadedModel]:
        return self._meta.get(model_type)

    def get_features(self, model_type: ModelType) -> List[str]:
        return self._features.get(model_type, DEFAULT_FEATURES)

    def is_loaded(self, model_type: ModelType) -> bool:
        m = self._meta.get(model_type)
        return m is not None and m.status == ModelStatus.LOADED

    def summary(self) -> Dict[str, str]:
        return {mt.value: (m.status.value if m else "not_loaded")
                for mt, m in self._meta.items()}

    def get_feature_group(self, group_name: str) -> List[str]:
        """Returns the feature list for a named group."""
        return FEATURE_REGISTRY.get(group_name, ALL_FEATURES)

    def load_group_model(
        self,
        model_prefix: str,   # "xgb" or "rf"
        group:        str,   # "weather", "hydro", "terrain", "all", etc.
    ) -> Tuple[Optional[Any], List[str]]:
        """
        Loads a specific model variant (e.g. xgb_weather) from disk.
        Returns (model, features).
        """
        key = f"{model_prefix}_{group}"
        if key in self._group_models:
            return self._group_models[key], self._group_features.get(key, [])

        if model_prefix == "xgb":
            path     = MODELS_DIR / f"{key}.json"
            feat_p   = MODELS_DIR / f"{key}_features.json"
            model, features = self._load_xgb(path, feat_p, group)
        else:
            path     = MODELS_DIR / f"{key}.pkl"
            feat_p   = MODELS_DIR / f"{key}_features.json"
            model, features = self._load_rf(path, feat_p, group)

        if model is not None:
            self._group_models[key]   = model
            self._group_features[key] = features

        return model, features

    def save_model(
        self,
        model_type:   ModelType,
        model:        Any,
        feature_cols: List[str],
    ) -> None:
        """Persists a model to disk and refreshes the in-memory cache."""
        path      = MODEL_PATHS[model_type]
        feat_path = FEATURE_PATHS[model_type]

        try:
            if model_type == ModelType.XGBOOST:
                model.save_model(str(path))
            elif model_type == ModelType.RANDOM_FOREST:
                with open(path, "wb") as f:
                    pickle.dump(model, f)

            with open(feat_path, "w") as f:
                json.dump(feature_cols, f)

            self._models[model_type]   = model
            self._features[model_type] = feature_cols
            self._meta[model_type]     = LoadedModel(
                model_type=model_type,
                status=ModelStatus.LOADED,
                path=str(path),
                features=feature_cols,
            )
            logger.info(f"[ModelRegistry] Saved {model_type.value} → {path}")
        except Exception as exc:
            logger.error(f"[ModelRegistry] Failed to save {model_type.value}: {exc}")

    # ── Internal loaders ──────────────────────────────────────────────────

    async def _load_single(
        self,
        model_type: ModelType,
        df:         Optional[pd.DataFrame],
    ) -> None:
        path      = MODEL_PATHS[model_type]
        feat_path = FEATURE_PATHS[model_type]
        features  = self._load_features(feat_path)

        if path.exists():
            model, status, error = self._load_from_disk(model_type, path)
        else:
            logger.warning(
                f"[ModelRegistry] {model_type.value} not found at {path}. "
                "Training default model on synthetic data..."
            )
            model, features, status, error = self._train_default(model_type, df, features)
            if model is not None and status == ModelStatus.LOADED:
                self.save_model(model_type, model, features)

        self._models[model_type]   = model
        self._features[model_type] = features
        self._meta[model_type]     = LoadedModel(
            model_type=model_type,
            status=status,
            path=str(path) if path.exists() else None,
            features=features,
            error=error,
        )

    def _load_from_disk(
        self,
        model_type: ModelType,
        path:       Path,
    ) -> Tuple[Optional[Any], ModelStatus, Optional[str]]:
        try:
            if model_type == ModelType.XGBOOST:
                import xgboost as xgb
                model = xgb.Booster()
                model.load_model(str(path))
            elif model_type == ModelType.RANDOM_FOREST:
                with open(path, "rb") as f:
                    model = pickle.load(f)
            else:
                return None, ModelStatus.FAILED, "Unknown model type"

            logger.info(f"[ModelRegistry] ✓ Loaded {model_type.value} from {path.name}")
            return model, ModelStatus.LOADED, None

        except ImportError as exc:
            logger.warning(f"[ModelRegistry] Missing library for {model_type.value}: {exc}")
            return None, ModelStatus.FAILED, f"Missing library: {exc}"
        except Exception as exc:
            logger.error(f"[ModelRegistry] Failed loading {model_type.value}: {exc}")
            return None, ModelStatus.FAILED, str(exc)

    def _load_xgb(
        self, path: Path, feat_path: Path, group: str
    ) -> Tuple[Optional[Any], List[str]]:
        if not path.exists():
            logger.warning(f"[ModelRegistry] xgb_{group} not found – run train_models.py first")
            return None, FEATURE_REGISTRY.get(group, ALL_FEATURES)
        try:
            import xgboost as xgb
            m = xgb.Booster()
            m.load_model(str(path))
            feat = self._load_features(feat_path)
            logger.info(f"[ModelRegistry] Loaded xgb_{group} ({len(feat)} features)")
            return m, feat
        except Exception as exc:
            logger.error(f"[ModelRegistry] xgb_{group} load error: {exc}")
            return None, FEATURE_REGISTRY.get(group, ALL_FEATURES)

    def _load_rf(
        self, path: Path, feat_path: Path, group: str
    ) -> Tuple[Optional[Any], List[str]]:
        if not path.exists():
            logger.warning(f"[ModelRegistry] rf_{group} not found – run train_models.py first")
            return None, FEATURE_REGISTRY.get(group, ALL_FEATURES)
        try:
            with open(path, "rb") as f:
                m = pickle.load(f)
            feat = self._load_features(feat_path)
            logger.info(f"[ModelRegistry] Loaded rf_{group} ({len(feat)} features)")
            return m, feat
        except Exception as exc:
            logger.error(f"[ModelRegistry] rf_{group} load error: {exc}")
            return None, FEATURE_REGISTRY.get(group, ALL_FEATURES)

    # ── Default / synthetic training ──────────────────────────────────────

    def _train_default(
        self,
        model_type: ModelType,
        df:         Optional[pd.DataFrame],
        features:   List[str],
    ) -> Tuple[Optional[Any], List[str], ModelStatus, Optional[str]]:
        try:
            X, y, used = self._prepare_training_data(df, features)
            if X is None:
                return None, features, ModelStatus.UNTRAINED, "No training data"

            logger.info(
                f"[ModelRegistry] Fallback training {model_type.value} on "
                f"{X.shape[0]} rows × {X.shape[1]} features ..."
            )

            if model_type == ModelType.XGBOOST:
                model = self._train_xgboost_default(X, y)
            elif model_type == ModelType.RANDOM_FOREST:
                model = self._train_rf_default(X, y)
            else:
                return None, used, ModelStatus.FAILED, "Unsupported model type"

            if model is None:
                return None, used, ModelStatus.FAILED, "Training returned None"

            return model, used, ModelStatus.LOADED, None

        except Exception as exc:
            logger.error(f"[ModelRegistry] Fallback training failed: {exc}")
            return None, features, ModelStatus.FAILED, str(exc)

    def _prepare_training_data(
        self,
        df:            Optional[pd.DataFrame],
        hint_features: List[str],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
        if df is not None and isinstance(df, pd.DataFrame) and len(df) >= MIN_TRAIN_ROWS:
            available = [c for c in hint_features if c in df.columns]
            if not available:
                available = [c for c in df.select_dtypes(include="number").columns
                             if c not in ("flood_occurred",)][:13]

            target_col = "flood_occurred" if "flood_occurred" in df.columns else None
            if target_col:
                y_raw = df[target_col].fillna(0).astype(int).values
            else:
                y_raw = np.random.randint(0, 2, len(df))

            X_raw = df[available].fillna(0).values
            return X_raw, y_raw, available

        # Synthetic fallback
        logger.info("[ModelRegistry] Generating synthetic training data (13 features)...")
        n        = 500
        features = list(ALL_FEATURES)
        rng      = np.random.default_rng(42)
        X        = rng.standard_normal((n, len(features)))
        y        = ((X[:, 0] > 1.0) | (X[:, 1] > 1.2)).astype(int)  # rain + rh → flood
        return X, y, features

    @staticmethod
    def _train_xgboost_default(X: np.ndarray, y: np.ndarray) -> Optional[Any]:
        try:
            import xgboost as xgb
            from sklearn.model_selection import train_test_split

            X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            params = {
                "objective":        "binary:logistic",
                "eval_metric":      "logloss",
                "max_depth":        5,
                "learning_rate":    0.1,
                "subsample":        0.85,
                "colsample_bytree": 0.85,
                "seed":             42,
            }
            dtrain = xgb.DMatrix(X_tr, label=y_tr)
            dval   = xgb.DMatrix(X_val, label=y_val)
            model  = xgb.train(
                params, dtrain,
                num_boost_round=200,
                evals=[(dval, "val")],
                early_stopping_rounds=20,
                verbose_eval=False,
            )
            return model
        except ImportError:
            logger.warning("[ModelRegistry] XGBoost not installed.")
            return None

    @staticmethod
    def _train_rf_default(X: np.ndarray, y: np.ndarray) -> Optional[Any]:
        try:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X, y)
            return model
        except ImportError:
            logger.warning("[ModelRegistry] sklearn not installed.")
            return None

    @staticmethod
    def _load_features(feat_path: Path) -> List[str]:
        if feat_path.exists():
            try:
                with open(feat_path) as f:
                    return json.load(f)
            except Exception:
                pass
        return list(ALL_FEATURES)
