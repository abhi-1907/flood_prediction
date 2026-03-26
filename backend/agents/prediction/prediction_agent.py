"""
Prediction Agent – Top-level coordinator for the flood prediction pipeline.

Execution pipeline:
  1. Load / verify models via ModelRegistry (XGBoost + Random Forest)
  2. Select the best trained model variant via ModelSelector
     (based on which features are available in the incoming data)
  3. Run XGBoost and Random Forest predictions in parallel
  4. Combine via EnsembleCombiner (weighted blend + confidence adjustment)
  5. Generate explanation via PredictionExplainer (feature importance + LLM narrative)
  6. Store all artifacts in the session and return PredictionResult

Feature groups trained on real dataset:
  weather  : rain_mm_weekly, temp_c_mean, rh_percent_mean,
             wind_ms_mean, rain_mm_monthly
  hydro    : dam_count_50km, dist_major_river_km, waterbody_nearby
  terrain  : lat, lon, elevation_m, slope_degree, terrain_type_encoded
  all      : all 13 features combined (primary model)

Quick-predict mode (single sensor row):
  The agent also supports quick_predict(**kwargs) where you pass individual
  sensor values directly, without needing a preprocessed DataFrame.
  This is used by the /predict/quick REST endpoint.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from agents.prediction.ensemble import EnsembleCombiner
from agents.prediction.explainer import PredictionExplainer
from agents.prediction.model_registry import (
    ModelRegistry,
    ALL_FEATURES,
    WEATHER_FEATURES,
    HYDRO_FEATURES,
    TERRAIN_FEATURES,
)
from agents.prediction.model_selector import ModelSelector, ModelSelectionPlan, infer_group_from_features
from agents.prediction.models.xgboost_predictor import XGBoostPredictor
from agents.prediction.models.random_forest_predictor import RandomForestPredictor
from agents.prediction.prediction_schemas import (
    EnsemblePrediction,
    ModelPrediction,
    ModelStatus,
    ModelType,
    PredictionMode,
    PredictionResult,
    RiskLevel,
)
from agents.orchestration.memory import Session
from services.gemini_service import GeminiService, get_gemini_service
from utils.logger import logger


# ── Terrain type encoding map (must match training) ───────────────────────────
# Generated during training — classes from LabelEncoder saved in trained_models/
TERRAIN_TYPE_ENCODING: Dict[str, int] = {
    "urban":         3,
    "suburban":      2,
    "rural":         1,
    "forest":        0,
    "agricultural":  0,
    "wetland":       1,
    "unknown":       1,
}


class PredictionAgent:
    """
    Coordinates the full flood prediction pipeline.

    All sub-components are instantiated once and re-used across sessions.
    """

    def __init__(self, gemini_service: Optional[GeminiService] = None) -> None:
        self._gemini    = gemini_service or get_gemini_service()
        self._registry  = ModelRegistry()
        self._selector  = ModelSelector(self._gemini)
        self._combiner  = EnsembleCombiner()
        self._explainer = PredictionExplainer(self._gemini)
        self._models_loaded = False

    # ── Public interface ───────────────────────────────────────────────────

    async def run(
        self,
        session: Session,
        data:    Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> PredictionResult:
        """
        Main agent entry point (called by Orchestrator via ToolRegistry).

        Args:
            session: Active orchestration session.
            data:    Optional DataFrame override. If None, reads
                     "processed_dataset" from session artifacts.

        Returns:
            PredictionResult with ensemble prediction and explanation.
        """
        session_id = session.session_id
        context    = session.context
        warnings:  List[str] = []
        errors:    List[str] = []

        logger.info(f"[PredictionAgent] Starting prediction for session {session_id}")

        # ── 0. Retrieve preprocessed dataset ─────────────────────────────
        df = data if data is not None else session.get_artifact("processed_dataset")

        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            return PredictionResult(
                session_id=session_id,
                status="failed",
                errors=["No processed_dataset found. Run PreprocessingAgent first."],
            )

        if not isinstance(df, pd.DataFrame):
            try:
                df = pd.DataFrame(df)
            except Exception as exc:
                return PredictionResult(
                    session_id=session_id,
                    status="failed",
                    errors=[f"Cannot convert data to DataFrame: {exc}"],
                )

        # Drop non-feature columns from dataset if still present
        drop_cols = ["country", "name", "week", "terrain_type", "flood_occurred"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

        logger.info(
            f"[PredictionAgent] Input: {df.shape[0]} rows × {df.shape[1]} cols | "
            f"cols={list(df.columns)}"
        )

        try:
            result = await self._predict_from_df(df, session, context, warnings, errors)
            return result
        except Exception as exc:
            logger.exception(f"[PredictionAgent] Unhandled error: {exc}")
            return PredictionResult(
                session_id=session_id,
                status="failed",
                errors=[str(exc)],
            )

    async def quick_predict(
        self,
        location:          str,
        lat:               float = 0.0,
        lon:               float = 0.0,
        rain_mm_weekly:    float = 0.0,
        temp_c_mean:       float = 25.0,
        rh_percent_mean:   float = 60.0,
        wind_ms_mean:      float = 2.0,
        rain_mm_monthly:   float = 0.0,
        dam_count_50km:    int   = 0,
        dist_major_river_km: float = 10.0,
        waterbody_nearby:  int   = 0,
        elevation_m:       float = 50.0,
        slope_degree:      float = 1.0,
        terrain_type:      str   = "urban",
        session_id:        str   = "quick",
        **extra_kwargs,
    ) -> PredictionResult:
        """
        Single-row prediction from individual sensor / form values.
        Used by the /predict and /predict/quick REST endpoints.

        Maps the raw user inputs onto the 13 real training features and
        returns a full PredictionResult without needing a session.
        """
        logger.info(f"[PredictionAgent] quick_predict: location={location}")

        # Build a 1-row DataFrame with all training features
        terrain_encoded = TERRAIN_TYPE_ENCODING.get(terrain_type.lower(), 1)
        row = {
            "rain_mm_weekly":      rain_mm_weekly,
            "temp_c_mean":         temp_c_mean,
            "rh_percent_mean":     rh_percent_mean,
            "wind_ms_mean":        wind_ms_mean,
            "rain_mm_monthly":     rain_mm_monthly if rain_mm_monthly > 0 else rain_mm_weekly * 4.3,
            "dam_count_50km":      float(dam_count_50km),
            "dist_major_river_km": dist_major_river_km,
            "waterbody_nearby":    float(waterbody_nearby),
            "lat":                 lat,
            "lon":                 lon,
            "elevation_m":         elevation_m,
            "slope_degree":        slope_degree,
            "terrain_type_encoded": float(terrain_encoded),
        }
        df = pd.DataFrame([row])

        context = {
            "location":       location,
            "latitude":       lat,
            "longitude":      lon,
            "user_type":      extra_kwargs.get("user_type", "general"),
            "original_query": f"Quick prediction for {location}",
        }

        return await self._predict_from_df(df, None, context, [], [])

    # ── Internal prediction engine ─────────────────────────────────────────

    async def _predict_from_df(
        self,
        df:         pd.DataFrame,
        session:    Any,
        context:    Dict[str, Any],
        warnings:   List[str],
        errors:     List[str],
    ) -> PredictionResult:
        """Core prediction logic shared by run() and quick_predict()."""
        
        session_id = session.session_id if session and hasattr(session, "session_id") else "quick_predict_session"

        # ── 1. Load models ────────────────────────────────────────────────
        if not self._models_loaded:
            await self._registry.load_all(df=df)
            self._models_loaded = True

        # ── 2. Select model configuration ─────────────────────────────────
        # Only XGBoost and Random Forest are trained (no LSTM)
        loaded_map = {
            ModelType.XGBOOST:       self._registry.is_loaded(ModelType.XGBOOST),
            ModelType.RANDOM_FOREST: self._registry.is_loaded(ModelType.RANDOM_FOREST),
        }

        # Infer which feature group is best given available columns
        available_cols = list(df.select_dtypes(include="number").columns)
        best_group     = infer_group_from_features(available_cols)
        has_time_series = "week" in df.columns and len(df) > 5   # dataset uses "week" not "date"

        plan = await self._selector.select(
            loaded_models=loaded_map,
            feature_cols=available_cols,
            row_count=len(df),
            has_time_series=has_time_series,
            context={**context, "feature_group": best_group},
            session=session,
        )

        if not plan.models_to_use:
            return PredictionResult(
                session_id=session_id,
                status="failed",
                errors=["No ML models available. Ensure models are trained and saved."],
            )

        logger.info(
            f"[PredictionAgent] Plan: {[m.value for m in plan.models_to_use]} | "
            f"group={best_group} | mode={plan.mode} | horizon={plan.forecast_horizon}d"
        )

        # ── 3. Run predictions ────────────────────────────────────────────
        predictions, rf_interval = await self._run_predictions(df, plan, best_group)

        # ── 4. Ensemble combination ───────────────────────────────────────
        ensemble = self._combiner.combine(
            predictions=predictions,
            weights=plan.weights,
            mode=plan.mode,
            rf_interval=rf_interval,
        )

        # ── 5. Generate explanation ───────────────────────────────────────
        explanation = await self._explainer.explain(
            ensemble=ensemble,
            df=df,
            context=context,
            weights=plan.weights,
        )
        
        # ── 5.1 Store artifacts for backward compatibility ────────────────
        if session:
            session.store_artifact("ensemble_prediction", ensemble.model_dump())
            session.store_artifact("prediction_explanation", explanation.model_dump())

        # ── 6. Collect warnings ───────────────────────────────────────────
        failed_models = [
            p.model_type.value for p in predictions
            if p.status != ModelStatus.LOADED
        ]
        if failed_models:
            warnings.append(
                f"Models unavailable: {', '.join(failed_models)}. "
                f"Prediction based on remaining models."
            )
        if ensemble.confidence < 0.3:
            warnings.append(
                "Low prediction confidence — treat result as preliminary."
            )

        status = "success" if not warnings else "partial"

        logger.info(
            f"[PredictionAgent] Complete: P(flood)={ensemble.flood_probability:.3f} "
            f"| risk={ensemble.risk_level} | conf={ensemble.confidence:.3f}"
        )

        return PredictionResult(
            session_id=session_id,
            status=status,
            ensemble=ensemble,
            explanation=explanation,
            location=context.get("location"),
            latitude=context.get("latitude"),
            longitude=context.get("longitude"),
            prediction_mode=plan.mode,
            forecast_horizon_days=plan.forecast_horizon,
            warnings=warnings,
            errors=errors,
        )

    # ── Parallel model execution ───────────────────────────────────────────

    async def _run_predictions(
        self,
        df:    pd.DataFrame,
        plan:  ModelSelectionPlan,
        group: str = "all",
    ):
        """
        Runs all selected models concurrently.

        Uses the best matching trained model variant for the given feature group.
        Returns (predictions_list, rf_interval).
        """
        predictions: List[ModelPrediction] = []
        rf_interval  = None
        coros        = []

        for model_type in plan.models_to_use:
            # Try loading the specialised variant first, fall back to primary
            if model_type == ModelType.XGBOOST:
                model, features = self._registry.load_group_model("xgb", group)
                if model is None:
                    model    = self._registry.get(ModelType.XGBOOST)
                    features = self._registry.get_features(ModelType.XGBOOST)

                if model is None:
                    predictions.append(ModelPrediction(
                        model_type=model_type,
                        mode=plan.mode,
                        raw_score=0.5,
                        status=ModelStatus.NOT_FOUND,
                        error="XGBoost model not loaded — run train_models.py",
                    ))
                    continue
                coros.append(("xgboost", model_type, asyncio.to_thread(
                    XGBoostPredictor(model, features).predict, df, plan.mode
                )))

            elif model_type == ModelType.RANDOM_FOREST:
                model, features = self._registry.load_group_model("rf", group)
                if model is None:
                    model    = self._registry.get(ModelType.RANDOM_FOREST)
                    features = self._registry.get_features(ModelType.RANDOM_FOREST)

                if model is None:
                    predictions.append(ModelPrediction(
                        model_type=model_type,
                        mode=plan.mode,
                        raw_score=0.5,
                        status=ModelStatus.NOT_FOUND,
                        error="RandomForest model not loaded — run train_models.py",
                    ))
                    continue
                coros.append(("random_forest", model_type, asyncio.to_thread(
                    RandomForestPredictor(model, features).predict, df, plan.mode
                )))

        # Run all concurrently
        if coros:
            results = await asyncio.gather(
                *[coro for _, _, coro in coros],
                return_exceptions=True,
            )
            for (name, mt, _), result in zip(coros, results):
                if isinstance(result, Exception):
                    logger.error(f"[PredictionAgent] {name} raised: {result}")
                    predictions.append(ModelPrediction(
                        model_type=mt,
                        mode=plan.mode,
                        raw_score=0.5,
                        status=ModelStatus.FAILED,
                        error=str(result),
                    ))
                else:
                    predictions.append(result)

        # RF prediction interval for uncertainty band
        if ModelType.RANDOM_FOREST in plan.models_to_use:
            rf_model, rf_features = self._registry.load_group_model("rf", group)
            if rf_model is None:
                rf_model    = self._registry.get(ModelType.RANDOM_FOREST)
                rf_features = self._registry.get_features(ModelType.RANDOM_FOREST)
            if rf_model is not None:
                try:
                    rf_interval = RandomForestPredictor(
                        rf_model, rf_features
                    ).predict_interval(df)
                except Exception:
                    pass

        return predictions, rf_interval
