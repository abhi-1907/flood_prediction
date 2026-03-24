"""
Model Selector – LLM-powered selection of which models and prediction mode to use.

Decides:
  1. Prediction mode: classification / regression / multi_class
  2. Which models to include in the ensemble (based on data availability)
  3. Relative weights for ensemble weighting
  4. Forecast horizon (1/3/7 days)

Selection factors:
  - Data adequacy (rows, features, time-series vs tabular)
  - Available / loaded models (from ModelRegistry)
  - User intent (binary risk vs water level forecast vs risk tiering)
  - Session context (location, user type)
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from agents.prediction.prediction_schemas import (
    ModelType,
    PredictionMode,
)
from agents.prediction.model_registry import FEATURE_REGISTRY  # real feature groups
from services.gemini_service import GeminiService
from utils.logger import logger


# ── Default weights per model ─────────────────────────────────────────────────

DEFAULT_WEIGHTS: Dict[ModelType, float] = {
    ModelType.XGBOOST:       0.55,   # XGB wins on tabular flood data
    ModelType.RANDOM_FOREST: 0.45,
}

# Map feature column sets to which trained model group to prefer
def infer_group_from_features(feature_cols: list[str]) -> str:
    """
    Given available columns, pick the best-matching trained feature group.
    Used to load the right specialised model variant.
    """
    has_weather = any(f in feature_cols for f in ["rain_mm_weekly", "temp_c_mean", "rain_mm_monthly"])
    has_hydro   = any(f in feature_cols for f in ["dam_count_50km", "dist_major_river_km", "waterbody_nearby"])
    has_terrain = any(f in feature_cols for f in ["elevation_m", "slope_degree", "terrain_type_encoded"])

    if has_weather and has_hydro and has_terrain:
        return "all"
    if has_weather and has_hydro:
        return "weather_hydro"
    if has_weather and has_terrain:
        return "weather_terrain"
    if has_hydro and has_terrain:
        return "hydro_terrain"
    if has_weather:
        return "weather"
    if has_hydro:
        return "hydro"
    if has_terrain:
        return "terrain"
    return "all"   # default



class ModelSelectionPlan:
    """Describes which models to use and with what configuration."""
    def __init__(
        self,
        models_to_use:       List[ModelType],
        weights:             Dict[ModelType, float],
        mode:                PredictionMode,
        forecast_horizon:    int,
        rationale:           str,
    ) -> None:
        self.models_to_use    = models_to_use
        self.weights          = weights
        self.mode             = mode
        self.forecast_horizon = forecast_horizon
        self.rationale        = rationale


class ModelSelector:
    """
    Uses LLM + rule-based logic to select the optimal model configuration.

    Phase 1 – Rule-based: checks data shape, feature availability, model status.
    Phase 2 – LLM-enhanced: Gemini refines mode and weights based on context.
    """

    def __init__(self, gemini_service: GeminiService) -> None:
        self._gemini = gemini_service

    async def select(
        self,
        loaded_models:   Dict[ModelType, bool],   # {model_type: is_loaded}
        feature_cols:    List[str],
        row_count:       int,
        has_time_series: bool,
        context:         Dict[str, Any],
    ) -> ModelSelectionPlan:
        """
        Returns an optimal ModelSelectionPlan.

        Args:
            loaded_models:   Dict of model availability.
            feature_cols:    Feature columns in the preprocessed dataset.
            row_count:       Number of rows in the dataset.
            has_time_series: Whether the data has a temporal structure.
            context:         Session context (user_type, intent, location).
        """
        # Phase 1: rule-based model availability filter
        available = [mt for mt, loaded in loaded_models.items() if loaded]
        if not available:
            logger.warning("[ModelSelector] No models available — using rule-based fallback.")
            return self._fallback_plan([], has_time_series, context)

        # Choose mode based on data
        mode = self._infer_mode(feature_cols, context)

        # If only LSTM loaded and not enough data for sequences, exclude it
        if ModelType.LSTM in available and row_count < 50:
            available.remove(ModelType.LSTM)
            logger.info("[ModelSelector] Excluded LSTM — insufficient rows for sequences.")

        if not available:
            return self._fallback_plan([], has_time_series, context)

        # Phase 2: LLM refinement
        try:
            plan = await self._llm_refine(available, feature_cols, row_count, has_time_series, context, mode)
            return plan
        except Exception as exc:
            logger.warning(f"[ModelSelector] LLM refinement failed: {exc}. Using rule-based.")
            return self._rule_based_plan(available, mode, context)

    # ── LLM refinement ────────────────────────────────────────────────────

    async def _llm_refine(
        self,
        available:       List[ModelType],
        feature_cols:    List[str],
        row_count:       int,
        has_time_series: bool,
        context:         Dict[str, Any],
        mode:            PredictionMode,
    ) -> ModelSelectionPlan:
        """Asks Gemini to recommend optimal model configuration."""

        prompt = f"""
You are a flood prediction AI selecting the best ML model configuration.

## Dataset info
- Row count:        {row_count}
- Has time-series:  {has_time_series}
- Feature count:    {len(feature_cols)}
- Key features:     {', '.join(feature_cols[:10])}

## Available models: {[m.value for m in available]}

## Session context
- Location:   {context.get('location', 'unknown')}
- User type:  {context.get('user_type', 'general')}
- User query: {context.get('original_query', '')[:150]}

## Prediction modes
- classification : Binary flood/no-flood (best for public warnings)
- regression     : Continuous water level height (best for civil engineers)
- multi_class    : LOW/MED/HIGH/CRITICAL risk tiers (best for authorities)

## Task
Return ONLY a JSON object:
{{
  "mode":             "<classification | regression | multi_class>",
  "forecast_horizon": <1 | 3 | 7>,
  "models_to_use":    ["xgboost", "random_forest", "lstm"],
  "weights":          {{ "xgboost": 0.4, "random_forest": 0.35, "lstm": 0.25 }},
  "rationale":        "<2 sentence reason for these choices>"
}}

Rules:
- Only include available models in models_to_use.
- Weights must sum to 1.0.
- If user asks about probability or risk → classification.
- If user asks about water level or height → regression.
- If authorities → multi_class.
- LSTM gets higher weight when row_count > 200 and has_time_series=True.
- XGBoost gets highest weight when row_count < 100.
"""
        raw = await self._gemini.generate_json(prompt, use_fast_model=True)

        if not raw or not isinstance(raw, dict):
            return self._rule_based_plan(available, mode, context)

        try:
            mode_str = raw.get("mode", mode.value)
            try:
                mode = PredictionMode(mode_str)
            except ValueError:
                pass

            models_raw = raw.get("models_to_use", [m.value for m in available])
            models_to_use = []
            for m in models_raw:
                try:
                    mt = ModelType(m)
                    if mt in available:
                        models_to_use.append(mt)
                except ValueError:
                    pass

            raw_weights  = raw.get("weights", {})
            weights      = {}
            for mt in models_to_use:
                w = raw_weights.get(mt.value, DEFAULT_WEIGHTS.get(mt, 0.33))
                weights[mt] = float(w)

            # Normalise weights
            total = sum(weights.values())
            if total > 0:
                weights = {mt: w / total for mt, w in weights.items()}

            return ModelSelectionPlan(
                models_to_use    = models_to_use or available,
                weights          = weights or DEFAULT_WEIGHTS,
                mode             = mode,
                forecast_horizon = int(raw.get("forecast_horizon", 1)),
                rationale        = raw.get("rationale", "LLM-selected configuration"),
            )
        except Exception as exc:
            logger.warning(f"[ModelSelector] Failed parsing LLM response: {exc}")
            return self._rule_based_plan(available, mode, context)

    # ── Rule-based helpers ────────────────────────────────────────────────

    @staticmethod
    def _infer_mode(feature_cols: List[str], context: Dict[str, Any]) -> PredictionMode:
        query = context.get("original_query", "").lower()
        user_type = context.get("user_type", "general").lower()

        if "water level" in query or "height" in query or "metre" in query or "meter" in query:
            return PredictionMode.REGRESSION
        if "authorit" in user_type or "official" in user_type or "government" in user_type:
            return PredictionMode.MULTI_CLASS
        return PredictionMode.CLASSIFICATION

    @staticmethod
    def _rule_based_plan(
        available: List[ModelType],
        mode:      PredictionMode,
        context:   Dict[str, Any],
    ) -> ModelSelectionPlan:
        weights = {
            mt: DEFAULT_WEIGHTS.get(mt, 1.0 / len(available))
            for mt in available
        }
        total = sum(weights.values())
        weights = {mt: w / total for mt, w in weights.items()}
        return ModelSelectionPlan(
            models_to_use=available,
            weights=weights,
            mode=mode,
            forecast_horizon=1,
            rationale="Rule-based selection (LLM unavailable)",
        )

    @staticmethod
    def _fallback_plan(
        available: List[ModelType],
        has_ts:    bool,
        context:   Dict[str, Any],
    ) -> ModelSelectionPlan:
        return ModelSelectionPlan(
            models_to_use=[],
            weights={},
            mode=PredictionMode.CLASSIFICATION,
            forecast_horizon=1,
            rationale="No models available — prediction cannot proceed.",
        )
