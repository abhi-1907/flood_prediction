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
        Main agent entry point — uses a single merged LLM call for all outputs.

        All previously separate sub-agent calls (user profiling, location context,
        recommendation generation, summary) are combined into one Gemini call to
        minimise API quota usage.
        """
        session_id = session.session_id
        context    = session.context
        warnings:  List[str] = []
        errors:    List[str] = []

        logger.info(f"[RecommendationAgent] Starting for session {session_id}")

        try:
            # ── 1. Retrieve prediction results ────────────────────────────
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

            # ── 2. Gather context for the merged prompt ───────────────────
            location_name = context.get("location", "Unknown")
            state         = context.get("state", "")
            country       = context.get("country", "India")
            user_type     = context.get("user_type", "general_public")
            has_elderly   = context.get("has_elderly", False)
            has_children  = context.get("has_children", False)
            has_disability = context.get("has_disability", False)
            vehicle_access = context.get("vehicle_access", True)

            df = session.get_artifact("processed_dataset")
            env_data = self._extract_env_data(df, prediction)
            env_str  = ", ".join(f"{k}={v}" for k, v in env_data.items() if k not in ("risk_level",))

            # ── 3. Single merged LLM call ─────────────────────────────────
            prompt = f"""You are a flood disaster management AI assistant.
Generate a complete recommendation report in ONE JSON response.

## Flood Context
- Location: {location_name}, {state}, {country}
- Risk Level: {risk_level}
- Flood Probability: {flood_prob*100:.0f}%
- Confidence: {confidence*100:.0f}%
- Environmental readings: {env_str or 'N/A'}

## User Profile
- User type: {user_type}
- Has elderly: {has_elderly}
- Has children: {has_children}
- Has disability: {has_disability}
- Has vehicle access: {vehicle_access}

## Output Schema (respond ONLY with valid JSON, no prose, no markdown fences)
{{
  "recommendations": [
    {{
      "id": <integer 1–10>,
      "category": "<safety|evacuation|infrastructure|resource|health|communication>",
      "urgency": "<emergency|warning|advisory|informational>",
      "title": "<short action title>",
      "description": "<1-2 sentence description>",
      "action_steps": ["<step 1>", "<step 2>", "<step 3>"],
      "priority": <integer 1–10>
    }}
  ],
  "safety_message": "<SMS-friendly safety message under 160 chars>",
  "summary": "<2-3 sentence overall summary for a {user_type} user>",
  "emergency_contact": "<local emergency number or 112>",
  "risk_factors": ["<factor 1>", "<factor 2>"],
  "resources": [
    {{
      "resource_type": "<personnel|equipment|shelter|medical|food_water>",
      "quantity": <integer>,
      "unit": "<string>",
      "priority": "<high|medium|low>",
      "location": "<where to deploy>"
    }}
  ]
}}

Generate 4–7 recommendations relevant to {risk_level} risk level. Be specific to {location_name}."""

            try:
                raw = await self._gemini.generate_json(prompt)
            except Exception as e:
                logger.warning(f"[RecommendationAgent] Gemini generation error: {e}")
                raw = None

            if not raw or not isinstance(raw, dict) or not raw.get("recommendations"):
                warnings.append("Gemini failed — attempting Ollama (Llama3) local fallback...")
                raw = await self._try_ollama_fallback(prompt)
                
            if not raw or not isinstance(raw, dict) or not raw.get("recommendations"):
                warnings.append("Ollama failed — using emergency static fallback.")
                raw = self._get_emergency_fallback(risk_level, location_name)
            
            # Final safety check to ensure 'raw' is a dict for subsequent .get() calls
            if not isinstance(raw, dict):
                raw = {"recommendations": [], "resources": []}

            # ── 4. Parse LLM output ───────────────────────────────────────
            from agents.recommendation.recommendation_schemas import (
                Recommendation, RecommendationCategory, ResourceAllocation,
            )

            recs: List[Recommendation] = []
            for item in raw.get("recommendations", []):
                try:
                    recs.append(Recommendation(
                        id=item.get("id", len(recs) + 1),
                        category=RecommendationCategory(item.get("category", "safety")),
                        urgency=UrgencyLevel(item.get("urgency", "advisory")),
                        title=item.get("title", "Stay safe"),
                        description=item.get("description", ""),
                        action_steps=item.get("action_steps", []),
                        priority=item.get("priority", 5),
                    ))
                except Exception:
                    pass

            emergency_number = raw.get("emergency_contact", "112")

            resources: List[ResourceAllocation] = []
            for item in raw.get("resources", []):
                try:
                    resources.append(ResourceAllocation(
                        resource_type=item.get("resource_type", "personnel"),
                        quantity=item.get("quantity", 1),
                        deploy_to=item.get("deploy_to", location_name),
                        urgency=UrgencyLevel(item.get("urgency", "advisory")),
                        rationale=item.get("rationale", "Emergency deployment"),
                    ))
                except Exception:
                    pass

            safety_msg = raw.get("safety_message") or (
                f"⚠️ {risk_level} flood risk in {location_name}. "
                f"Flood probability: {flood_prob*100:.0f}%. Call 112 for emergencies."
            )
            summary = raw.get("summary") or self._fallback_summary_text(
                risk_level, flood_prob, recs, location_name, emergency_number
            )

            if not recs:
                warnings.append("No recommendations generated — using defaults.")
                recs = self._default_recommendations_simple(risk_level, location_name, emergency_number)

            # ── 5. Build mock location context for schema compatibility ────
            loc_ctx = LocationContextModel(
                location_name=location_name,
                state=state,
                country=country,
                emergency_number=emergency_number,
                risk_factors=raw.get("risk_factors", []),
            )

            # ── 6. Build user profile ─────────────────────────────────────
            profile = UserProfile(
                user_type=UserType(user_type) if user_type in UserType._value2member_map_ else UserType.PUBLIC,
                has_elderly=has_elderly,
                has_children=has_children,
                has_disability=has_disability,
                vehicle_access=vehicle_access,
            )

            urgency = self._overall_urgency(recs)

            # ── 7. Store artifacts ────────────────────────────────────────
            session.store_artifact("recommendations", [r.model_dump() for r in recs])
            session.store_artifact("resource_plan", [r.model_dump() for r in resources])
            session.store_artifact("safety_message", safety_msg)
            session.store_artifact("recommendation_summary", summary)

            logger.info(
                f"[RecommendationAgent] Complete: {len(recs)} recs (1 LLM call) | "
                f"urgency={urgency} | user_type={profile.user_type}"
            )

            return RecommendationResult(
                session_id=session_id,
                status="success" if not warnings else "partial",
                risk_level=risk_level,
                urgency=urgency,
                recommendations=recs,
                resource_plan=resources,
                summary=summary,
                safety_message=safety_msg,
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

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _fallback_summary_text(
        risk_level: str, flood_prob: float,
        recs: List, location: str, emergency_number: str,
    ) -> str:
        return (
            f"Flood risk is {risk_level} for {location} "
            f"({flood_prob*100:.0f}% probability). "
            f"{len(recs)} recommendations generated. "
            f"Emergency: {emergency_number}."
        )

    @staticmethod
    def _default_recommendations_simple(
        risk_level: str, location_name: str, emergency_number: str,
    ) -> List:
        from agents.recommendation.recommendation_schemas import (
            Recommendation, RecommendationCategory,
        )
        return [
            Recommendation(
                id=1,
                category=RecommendationCategory.SAFETY,
                urgency=UrgencyLevel.ADVISORY,
                title="Stay Informed About Flood Risk",
                description=(
                    f"Flood risk is {risk_level} in {location_name or 'your area'}. "
                    "Monitor local weather updates and disaster management advisories."
                ),
                action_steps=[
                    "Check local weather forecast regularly",
                    f"Call {emergency_number} if you need help",
                    "Keep emergency supplies ready",
                ],
                priority=1,
            ),
        ]


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

    async def _try_ollama_fallback(self, prompt: str) -> Dict[str, Any]:
        """Attempts to generate recommendations using local Ollama (Llama3) via httpx."""
        import httpx
        try:
            logger.info("[RecommendationAgent] Attempting local Ollama fallback...")
            url = "http://localhost:11434/api/generate"
            payload = {
                "model": "llama3:8b",
                "prompt": prompt,
                "stream": False,
                "format": "json"
            }
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, json=payload, timeout=60.0)
                if resp.status_code == 200:
                    data = resp.json()
                    import json
                    return json.loads(data.get("response", "{}"))
        except Exception as e:
            logger.warning(f"[RecommendationAgent] Ollama fallback failed: {str(e) or repr(e)}")
        return {}

    def _get_emergency_fallback(self, risk_level: str, location: str) -> Dict[str, Any]:
        """Provides a robust set of hardcoded recommendations for critical risk."""
        is_high = risk_level in ["HIGH", "CRITICAL"]
        return {
            "status": "partial",
            "summary": f"Emergency safety guidance for {location} due to {risk_level} flood risk.",
            "recommendations": [
                {
                    "id": 1,
                    "category": "safety",
                    "urgency": "emergency" if is_high else "advisory",
                    "title": "Evacuate if in Low-Lying Areas",
                    "description": "Immediate evacuation is recommended for residents in floodplains or near riverbanks.",
                    "action_steps": ["Pack essentials", "Follow designated escape routes", "Move to higher ground"]
                },
                {
                    "id": 2,
                    "category": "safety",
                    "urgency": "warning",
                    "title": "Prepare Emergency Kit",
                    "description": "Ensure you have water, non-perishable food, and medical supplies for at least 72 hours.",
                    "action_steps": ["Check flashlights", "Charge power banks", "Secure important documents"]
                },
                {
                    "id": 3,
                    "category": "infrastructure",
                    "urgency": "advisory",
                    "title": "Protect Property",
                    "description": "Move valuable items to upper floors and turn off electricity/gas if water enters.",
                    "action_steps": ["Use sandbags if available", "Unplug appliances", "Clear drainage holes"]
                }
            ],
            "resources": [
                {
                    "resource_type": "rescue_boats",
                    "quantity": 5,
                    "deploy_to": location,
                    "urgency": "emergency",
                    "rationale": "Required for potential rescue operations in urban flooding."
                }
            ]
        }
