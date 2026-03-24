"""
Prediction Explainer – Generates human-readable explanations of flood predictions.

Combines two approaches:
  1. Feature-based explanation  – Uses ensemble feature importances and SHAP values
     to identify the top contributing factors.
  2. LLM narrative explanation – Sends the prediction context to Gemini to produce
     a natural-language narrative explaining WHY the prediction is what it is.

Also produces:
  - Ranked FeatureContribution objects for visualization
  - A confidence note explaining what the uncertainty band means
  - Actionable summary sentences for public and authority users
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from agents.prediction.prediction_schemas import (
    EnsemblePrediction,
    FeatureContribution,
    ModelPrediction,
    ModelStatus,
    ModelType,
    PredictionExplanation,
    PredictionMode,
    RiskLevel,
)
from agents.prediction.ensemble import EnsembleCombiner
from services.gemini_service import GeminiService
from utils.logger import logger


# ── Friendly names for feature columns ─────────────────────────────────────────

FEATURE_LABELS: Dict[str, str] = {
    "rainfall_mm":            "Rainfall (mm)",
    "rolling_7d_rainfall":    "7-day cumulative rainfall",
    "rolling_14d_rainfall":   "14-day cumulative rainfall",
    "rolling_30d_rainfall":   "30-day cumulative rainfall",
    "water_level_m":          "River water level (m)",
    "discharge_m3s":          "River discharge (m³/s)",
    "elevation_m":            "Ground elevation (m)",
    "soil_moisture_pct":      "Soil moisture (%)",
    "soil_saturation_idx":    "Soil saturation index",
    "terrain_wetness_idx":    "Terrain wetness index",
    "amc":                    "Antecedent Moisture Condition",
    "is_monsoon":             "Monsoon season flag",
    "season_sin":             "Seasonal cycle (sin)",
    "rain_x_slope":           "Rainfall × terrain slope",
    "rain_x_soilmoist":       "Rainfall × soil moisture",
    "temperature_c":          "Temperature (°C)",
    "humidity_pct":           "Humidity (%)",
    "wind_speed_kmh":         "Wind speed (km/h)",
    "flood_plain_risk":       "Flood plain risk level",
    "low_elevation_flag":     "Low elevation area",
    "rolling_7d_max_rainfall":"7-day peak rainfall",
    "rainfall_trend_7d":      "7-day rainfall trend",
    "mean_slope_deg":         "Terrain slope (°)",
}


class PredictionExplainer:
    """
    Generates both feature-based and LLM-narrative explanations.

    Usage:
        explainer   = PredictionExplainer(gemini_service)
        explanation = await explainer.explain(ensemble, df, context)
    """

    def __init__(self, gemini_service: GeminiService) -> None:
        self._gemini = gemini_service

    # ── Public API ────────────────────────────────────────────────────────

    async def explain(
        self,
        ensemble:   EnsemblePrediction,
        df:         pd.DataFrame,
        context:    Dict[str, Any],
        weights:    Dict[ModelType, float],
    ) -> PredictionExplanation:
        """
        Generates a complete prediction explanation.

        Args:
            ensemble: Combined ensemble prediction.
            df:       The preprocessed input DataFrame.
            context:  Session context (location, user_type, etc.).
            weights:  Model weights from ModelSelector.

        Returns:
            PredictionExplanation with summary, key factors, and LLM narrative.
        """
        # 1. Build feature contributions
        merged_importance = EnsembleCombiner.merged_feature_importance(
            ensemble.model_predictions, weights
        )
        contributions = self._build_contributions(merged_importance, df)

        # 2. Derive key factors (human-readable)
        key_factors = self._key_factors(
            contributions, ensemble, context
        )

        # 3. Build rule-based summary
        summary = self._rule_based_summary(ensemble, context)

        # 4. LLM narrative
        llm_narrative = await self._llm_narrative(
            ensemble, contributions, key_factors, context
        )

        # 5. Confidence note
        confidence_note = self._confidence_note(ensemble)

        logger.info(
            f"[PredictionExplainer] Explanation generated | "
            f"factors={len(key_factors)} | contributions={len(contributions)}"
        )

        return PredictionExplanation(
            summary=summary,
            key_factors=key_factors,
            feature_contributions=contributions,
            llm_narrative=llm_narrative,
            confidence_note=confidence_note,
        )

    # ── Feature contributions ─────────────────────────────────────────────

    def _build_contributions(
        self,
        importance: Dict[str, float],
        df:         pd.DataFrame,
    ) -> List[FeatureContribution]:
        """Converts importance dict → ranked FeatureContribution list."""
        contributions = []
        for rank, (feat, imp) in enumerate(importance.items(), start=1):
            # Get actual value from the last row of the DataFrame
            value = 0.0
            if feat in df.columns:
                val = df[feat].iloc[-1] if len(df) > 0 else 0.0
                value = float(val) if pd.notna(val) else 0.0

            contributions.append(FeatureContribution(
                feature=FEATURE_LABELS.get(feat, feat),
                value=round(value, 4),
                contribution=round(imp, 4),
                rank=rank,
            ))
        return contributions[:10]   # Top 10

    # ── Key factors ───────────────────────────────────────────────────────

    def _key_factors(
        self,
        contributions: List[FeatureContribution],
        ensemble:      EnsemblePrediction,
        context:       Dict[str, Any],
    ) -> List[str]:
        """Derives human-readable key factor sentences from the top contributions."""
        factors = []

        for c in contributions[:5]:
            direction = "increases" if c.contribution > 0 else "decreases"
            factors.append(
                f"{c.feature} (value={c.value:.2f}) — "
                f"{direction} flood risk by {abs(c.contribution)*100:.1f}%"
            )

        # Add contextual factors
        if ensemble.risk_level == RiskLevel.CRITICAL:
            factors.append("⚠️ CRITICAL: Multiple indicators exceed danger thresholds.")
        elif ensemble.risk_level == RiskLevel.HIGH:
            factors.append("⚠️ HIGH: Conditions are favorable for flooding.")

        location = context.get("location", "the target area")
        factors.append(f"Location: {location}")

        return factors

    # ── Rule-based summary ────────────────────────────────────────────────

    @staticmethod
    def _rule_based_summary(
        ensemble: EnsemblePrediction,
        context:  Dict[str, Any],
    ) -> str:
        """Generates a concise one-paragraph summary."""
        location = context.get("location", "the target area")
        prob     = ensemble.flood_probability
        risk     = ensemble.risk_level.value
        conf     = ensemble.confidence
        band     = ensemble.uncertainty_band
        models   = len(ensemble.models_used)

        summary = (
            f"Flood prediction for {location}: "
            f"{prob*100:.1f}% probability ({risk} risk). "
            f"Confidence: {conf*100:.0f}%. "
            f"95% CI: [{band[0]*100:.1f}%–{band[1]*100:.1f}%]. "
            f"Based on {models} model(s): "
            + ", ".join(m.value for m in ensemble.models_used)
            + "."
        )
        return summary

    # ── LLM narrative ─────────────────────────────────────────────────────

    async def _llm_narrative(
        self,
        ensemble:       EnsemblePrediction,
        contributions:  List[FeatureContribution],
        key_factors:    List[str],
        context:        Dict[str, Any],
    ) -> Optional[str]:
        """Asks Gemini to produce a natural-language explanation."""
        try:
            top_features = [
                {"feature": c.feature, "value": c.value, "impact_pct": round(c.contribution * 100, 1)}
                for c in contributions[:5]
            ]

            prompt = f"""
You are an AI flood prediction assistant explaining a prediction result to a user.

## Prediction
- Location:          {context.get('location', 'Unknown')}
- Flood probability: {ensemble.flood_probability*100:.1f}%
- Risk level:        {ensemble.risk_level.value}
- Confidence:        {ensemble.confidence*100:.0f}%
- 95% CI:           [{ensemble.uncertainty_band[0]*100:.1f}%–{ensemble.uncertainty_band[1]*100:.1f}%]
- Models used:       {[m.value for m in ensemble.models_used]}
- User type:         {context.get('user_type', 'general')}

## Top contributing factors
{json.dumps(top_features, indent=2)}

## Task
Write a 3–5 sentence explanation of this prediction in clear, non-technical language.
- Address why the risk is at this level
- Mention the top 2–3 factors and their real-world meaning
- If risk is HIGH or CRITICAL, include a brief safety advisory
- If confidence is low (<50%), note that the prediction has uncertainty
- Be direct and actionable. Do NOT use headers or bullet points.
"""
            narrative = await self._gemini.generate(prompt, use_fast_model=True)
            return narrative.strip() if narrative else None

        except Exception as exc:
            logger.warning(f"[PredictionExplainer] LLM narrative failed: {exc}")
            return None

    # ── Confidence note ───────────────────────────────────────────────────

    @staticmethod
    def _confidence_note(ensemble: EnsemblePrediction) -> str:
        conf = ensemble.confidence
        band = ensemble.uncertainty_band

        if conf >= 0.8:
            return (
                f"High confidence prediction (all models agree). "
                f"The 95% confidence interval is narrow: "
                f"[{band[0]*100:.0f}%–{band[1]*100:.0f}%]."
            )
        elif conf >= 0.5:
            return (
                f"Moderate confidence — some model disagreement exists. "
                f"The actual flood probability likely falls between "
                f"{band[0]*100:.0f}% and {band[1]*100:.0f}%."
            )
        else:
            return (
                f"Low confidence — models show significant disagreement or "
                f"limited data is available. Treat this prediction as preliminary. "
                f"Range: [{band[0]*100:.0f}%–{band[1]*100:.0f}%]."
            )
