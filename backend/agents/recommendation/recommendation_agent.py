"""
Recommendation Agent – Top-level coordinator for the flood recommendation pipeline.

Execution pipeline:
  1. UserProfiler          : Infer user type, vulnerability, and accessibility needs
  2. LocationContextBuilder: Enrich geographic and infrastructure context
  3. RecommendationEngine  : Generate LLM + rule-based recommendations
  4. Safety message         : Short SMS-friendly alert text
  5. Authority brief        : Formal brief for officials (if authority user)
  6. Store artifacts        : All outputs stored in session

This agent is registered in the ToolRegistry under "recommendation" and called
by the Orchestrator after the PredictionAgent completes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from agents.recommendation.recommendation_schemas import (
    LocationContext as LocationContextModel,
    Recommendation,
    RecommendationResult,
    ResourceAllocation,
    UrgencyLevel,
    UserProfile,
    UserType,
)
from agents.recommendation.user_profiler import UserProfiler
from agents.recommendation.location_context import LocationContextBuilder
from agents.recommendation.recommendation_engine import RecommendationEngine
from agents.orchestration.memory import Session
from services.gemini_service import GeminiService, get_gemini_service
from utils.logger import logger


class RecommendationAgent:
    """
    Orchestrates the full recommendation generation pipeline.

    All sub-components are injected and stateless — the agent can be reused
    across multiple sessions.
    """

    def __init__(self, gemini_service: Optional[GeminiService] = None) -> None:
        self._gemini   = gemini_service or get_gemini_service()
        self._profiler = UserProfiler(self._gemini)
        self._location = LocationContextBuilder(self._gemini)
        self._engine   = RecommendationEngine(self._gemini)

    # ── Public interface (called by Orchestrator) ─────────────────────────

    async def run(
        self,
        session: Session,
        **kwargs,
    ) -> RecommendationResult:
        """
        Main agent entry point.

        Args:
            session: Active orchestration session containing prediction results.

        Returns:
            RecommendationResult with personalised recommendations and optional
            resource allocation plan.
        """
        session_id = session.session_id
        context    = session.context
        warnings:  List[str] = []
        errors:    List[str] = []

        logger.info(f"[RecommendationAgent] Starting for session {session_id}")

        try:
            # ── 1. Retrieve prediction results ───────────────────────────
            prediction = session.get_artifact("ensemble_prediction")
            if not prediction or not isinstance(prediction, dict):
                return RecommendationResult(
                    session_id=session_id,
                    status="failed",
                    errors=["No prediction results found. Run PredictionAgent first."],
                )

            risk_level = prediction.get("risk_level", "MEDIUM")
            flood_prob = prediction.get("flood_probability", 0.5)
            confidence = prediction.get("confidence", 0.5)

            # ── 2. Build user profile ─────────────────────────────────────
            profile = await self._profiler.build_profile(context)

            # ── 3. Build location context ─────────────────────────────────
            df = session.get_artifact("processed_dataset")
            loc_ctx = await self._location.build(context, df)

            # ── 4. Extract environmental conditions ──────────────────────
            env_data = self._extract_env_data(df, prediction)

            # ── 5. Generate recommendations ───────────────────────────────
            recs, resources = await self._engine.generate(
                risk_level=risk_level,
                flood_prob=flood_prob,
                confidence=confidence,
                profile=profile,
                location=loc_ctx,
                env_data=env_data,
            )

            if not recs:
                warnings.append("No recommendations generated — using defaults.")
                recs = self._default_recommendations(risk_level, loc_ctx)

            # ── 6. Generate safety SMS message ────────────────────────────
            safety_msg = await self._engine.generate_safety_message(
                risk_level=risk_level,
                flood_prob=flood_prob,
                location=loc_ctx,
                profile=profile,
            )

            # ── 7. Generate authority brief (if applicable) ──────────────
            authority_brief = None
            if profile.user_type in (UserType.AUTHORITY, UserType.RESPONDER):
                authority_brief = await self._engine.generate_authority_brief(
                    risk_level=risk_level,
                    flood_prob=flood_prob,
                    confidence=confidence,
                    recs=recs,
                    resources=resources,
                    location=loc_ctx,
                )

            # ── 8. Generate an overall summary ───────────────────────────
            summary = await self._generate_summary(
                risk_level, flood_prob, recs, profile, loc_ctx
            )

            # ── 9. Determine overall urgency ──────────────────────────────
            urgency = self._overall_urgency(recs)

            # ── 10. Store artifacts ───────────────────────────────────────
            session.store_artifact("recommendations", [r.model_dump() for r in recs])
            session.store_artifact("resource_plan", [r.model_dump() for r in resources])
            session.store_artifact("safety_message", safety_msg)
            if authority_brief:
                session.store_artifact("authority_brief", authority_brief)
            session.store_artifact("recommendation_summary", summary)

            status = "success" if not warnings else "partial"

            logger.info(
                f"[RecommendationAgent] Complete: {len(recs)} recs, "
                f"{len(resources)} resources | urgency={urgency} | "
                f"user_type={profile.user_type}"
            )

            return RecommendationResult(
                session_id=session_id,
                status=status,
                risk_level=risk_level,
                urgency=urgency,
                recommendations=recs,
                resource_plan=resources,
                summary=summary,
                safety_message=safety_msg,
                authority_brief=authority_brief,
                user_profile=profile,
                location_context=loc_ctx,
                warnings=warnings,
                errors=errors,
            )

        except Exception as exc:
            logger.exception(f"[RecommendationAgent] Unhandled error: {exc}")
            return RecommendationResult(
                session_id=session_id,
                status="failed",
                errors=[str(exc)],
            )

    # ── Environmental data extraction ─────────────────────────────────────

    @staticmethod
    def _extract_env_data(
        df:         Optional[pd.DataFrame],
        prediction: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extracts recent environmental readings from the preprocessed dataset."""
        env: Dict[str, Any] = {}

        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
            last = df.iloc[-1]

            env_cols = [
                "rainfall_mm", "water_level_m", "discharge_m3s",
                "soil_moisture_pct", "elevation_m", "temperature_c",
                "humidity_pct", "wind_speed_kmh", "rolling_7d_rainfall",
                "rolling_14d_rainfall", "amc", "flood_risk_proxy",
            ]
            for col in env_cols:
                if col in df.columns:
                    val = last.get(col)
                    if pd.notna(val):
                        env[col] = round(float(val), 2)

        env["flood_probability"] = prediction.get("flood_probability", 0.5)
        env["risk_level"]        = prediction.get("risk_level", "MEDIUM")
        return env

    # ── Overall urgency ───────────────────────────────────────────────────

    @staticmethod
    def _overall_urgency(recs: List[Recommendation]) -> UrgencyLevel:
        """Returns the highest urgency level across all recommendations."""
        urgency_order = [
            UrgencyLevel.EMERGENCY,
            UrgencyLevel.WARNING,
            UrgencyLevel.ADVISORY,
            UrgencyLevel.INFORMATIONAL,
        ]
        for u in urgency_order:
            if any(r.urgency == u for r in recs):
                return u
        return UrgencyLevel.INFORMATIONAL

    # ── Summary generation ────────────────────────────────────────────────

    async def _generate_summary(
        self,
        risk_level: str,
        flood_prob: float,
        recs:       List[Recommendation],
        profile:    UserProfile,
        location:   LocationContextModel,
    ) -> str:
        """Generates a 2–3 sentence overall summary."""
        try:
            rec_titles = [f"{r.priority}. {r.title}" for r in recs[:5]]
            prompt = f"""
Summarize these flood recommendations in 2-3 sentences for a {profile.user_type.value} user.
Location: {location.location_name} | Risk: {risk_level} | Probability: {flood_prob*100:.0f}%

Top recommendations:
{chr(10).join(rec_titles)}

Be concise, direct, and action-oriented. No headers or bullets.
"""
            summary = await self._gemini.generate(prompt, use_fast_model=True)
            return summary.strip() if summary else self._fallback_summary(
                risk_level, flood_prob, recs, location
            )
        except Exception:
            return self._fallback_summary(risk_level, flood_prob, recs, location)

    @staticmethod
    def _fallback_summary(
        risk_level: str,
        flood_prob: float,
        recs:       List[Recommendation],
        location:   LocationContextModel,
    ) -> str:
        loc = location.location_name or "your area"
        return (
            f"Flood risk is {risk_level} for {loc} "
            f"({flood_prob*100:.0f}% probability). "
            f"{len(recs)} recommendations have been generated. "
            f"Emergency contact: {location.emergency_number}."
        )

    # ── Default fallback recommendations ──────────────────────────────────

    @staticmethod
    def _default_recommendations(
        risk_level: str,
        location:   LocationContextModel,
    ) -> List[Recommendation]:
        from agents.recommendation.recommendation_schemas import RecommendationCategory
        return [
            Recommendation(
                id=1,
                category=RecommendationCategory.SAFETY,
                urgency=UrgencyLevel.ADVISORY,
                title="Stay Informed About Flood Risk",
                description=(
                    f"Flood risk is {risk_level} in {location.location_name or 'your area'}. "
                    "Monitor local weather updates and disaster management advisories."
                ),
                action_steps=[
                    "Check local weather forecast regularly",
                    f"Call {location.emergency_number} if you need help",
                    "Keep emergency supplies ready",
                ],
                priority=1,
            ),
        ]
