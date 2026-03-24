"""
Context Manager – Builds, enriches, and maintains the shared execution context
for an orchestration session.

The context is a living dict that accumulates information as agents run:
  - Location metadata (lat/lng, district, state, country)
  - User profile (public / authority / first-responder)
  - Detected data types (rainfall, hydro, terrain, timeseries)
  - Risk level (populated after prediction runs)
  - Feature flags (what downstream agents are allowed to do)

Other agents read from the context via session.get_context() so that each
agent knows the full picture of the current request without being re-briefed.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import httpx

from agents.orchestration.memory import Session
from services.gemini_service import GeminiService
from utils.logger import logger


# ── Constants ─────────────────────────────────────────────────────────────────

GEOCODE_URL = "https://nominatim.openstreetmap.org/search"   # Free OSM geocoder


# ── Context Manager ───────────────────────────────────────────────────────────

class ContextManager:
    """
    Builds and enriches the shared execution context in a session.

    Call `initialise()` at the start of each orchestration run to populate
    the context with as much information as possible before the plan runs.
    """

    def __init__(self, gemini_service: GeminiService) -> None:
        self._gemini = gemini_service

    # ── Public API ────────────────────────────────────────────────────────

    async def initialise(
        self,
        session: Session,
        user_query: str,
        uploaded_columns: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Populates the session context with:
          - Extracted location (geocoded to lat/lng)
          - Detected user type
          - Detected data types / available features
          - Feature flags for downstream agents

        Returns the fully populated context dict.
        """
        logger.info(f"[ContextManager] Initialising context for session {session.session_id}")

        # 1. Extract structured metadata via LLM
        meta = await self._extract_metadata(user_query, uploaded_columns)

        # 2. Geocode location if we got a name
        if meta.get("location"):
            geo = await self._geocode(meta["location"])
            meta.update(geo)

        # 3. Derive feature flags
        meta["flags"] = self._derive_flags(meta)

        # 4. Persist into session context
        for key, value in meta.items():
            session.set_context(key, value)

        logger.info(f"[ContextManager] Context initialised: {meta}")
        return meta

    def update_risk(self, session: Session, risk_level: str, probability: float) -> None:
        """Called by the Orchestrator after prediction completes."""
        session.set_context("risk_level", risk_level)
        session.set_context("flood_probability", probability)
        logger.info(
            f"[ContextManager] Risk updated → {risk_level} ({probability:.1%})"
        )

    def update_user_type(self, session: Session, user_type: str) -> None:
        session.set_context("user_type", user_type)

    # ── LLM metadata extraction ───────────────────────────────────────────

    async def _extract_metadata(
        self,
        user_query: str,
        uploaded_columns: Optional[list],
    ) -> Dict[str, Any]:
        """Uses the LLM to extract structured metadata from an unstructured query."""
        columns_text = (
            f"Uploaded CSV columns: {uploaded_columns}" if uploaded_columns
            else "No file uploaded."
        )
        prompt = f"""
You are an assistant for a flood prediction system.
Extract structured metadata from the user query below.
Respond ONLY with a valid JSON object.

User query: "{user_query}"
{columns_text}

Output schema:
{{
  "location": "<place name or null>",
  "user_type": "<general_public | authority | first_responder | unknown>",
  "data_types": ["<list of: rainfall | hydro | terrain | timeseries | unknown>"],
  "wants_recommendations": <true|false>,
  "wants_simulation": <true|false>,
  "wants_alerts": <true|false>,
  "urgency": "<immediate | routine>"
}}
"""
        raw = await self._gemini.generate(prompt)
        import json, re
        cleaned = re.sub(r"```(?:json)?", "", raw).strip()
        try:
            return json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            logger.warning("[ContextManager] Could not parse metadata JSON, using defaults.")
            return {
                "location": None,
                "user_type": "general_public",
                "data_types": [],
                "wants_recommendations": True,
                "wants_simulation": False,
                "wants_alerts": False,
                "urgency": "routine",
            }

    # ── Geocoding ─────────────────────────────────────────────────────────

    @staticmethod
    async def _geocode(location_name: str) -> Dict[str, Any]:
        """
        Converts a place name to lat/lng using OpenStreetMap Nominatim (free).
        Falls back gracefully if the request fails.
        """
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    GEOCODE_URL,
                    params={
                        "q": location_name,
                        "format": "json",
                        "limit": 1,
                        "addressdetails": 1,
                    },
                    headers={"User-Agent": "FloodSenseAI/1.0"},
                )
                resp.raise_for_status()
                results = resp.json()

            if not results:
                logger.warning(f"[ContextManager] Geocode returned no results for '{location_name}'")
                return {}

            top = results[0]
            address = top.get("address", {})
            return {
                "latitude":  float(top["lat"]),
                "longitude": float(top["lon"]),
                "display_name": top.get("display_name", location_name),
                "district":  address.get("county") or address.get("city_district", ""),
                "state":     address.get("state", ""),
                "country":   address.get("country", ""),
                "country_code": address.get("country_code", ""),
            }
        except Exception as exc:
            logger.warning(f"[ContextManager] Geocoding failed for '{location_name}': {exc}")
            return {}

    # ── Feature flags ─────────────────────────────────────────────────────

    @staticmethod
    def _derive_flags(meta: Dict[str, Any]) -> Dict[str, bool]:
        """Derives feature flags that downstream agents use to gate their actions."""
        return {
            "run_recommendation":  meta.get("wants_recommendations", True),
            "run_simulation":      meta.get("wants_simulation", False),
            "run_alerting":        meta.get("wants_alerts", False),
            "is_urgent":           meta.get("urgency") == "immediate",
            "has_location":        bool(meta.get("latitude")),
        }
