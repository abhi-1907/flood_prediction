"""
Location Context Builder – Enriches geographic context for the target area.

Gathers:
  1. Location metadata (state, district, coastal flag)
  2. Terrain parameters from preprocessed data (elevation, flood plain risk)
  3. Infrastructure info (nearest shelter, hospital, emergency numbers)
  4. River/dam proximity
  5. Urban vs rural classification

Uses:
  - Session data (artifacts from ingestion/preprocessing)
  - Rule-based geographic knowledge (Indian states/districts)
  - LLM inference for missing fields
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from agents.recommendation.recommendation_schemas import (
    LocationContext as LocationContextModel,
)
from services.gemini_service import GeminiService
from utils.logger import logger


# ── Indian coastal states/UTs ─────────────────────────────────────────────────

COASTAL_STATES = {
    "kerala", "karnataka", "goa", "maharashtra", "gujarat",
    "tamil nadu", "andhra pradesh", "odisha", "west bengal",
    "puducherry", "daman and diu", "lakshadweep", "andaman and nicobar",
}

# Flood-prone rivers and their states
MAJOR_RIVERS = {
    "brahmaputra": ["assam", "arunachal pradesh"],
    "ganga": ["uttarakhand", "uttar pradesh", "bihar", "west bengal"],
    "yamuna": ["delhi", "uttar pradesh", "haryana"],
    "godavari": ["maharashtra", "telangana", "andhra pradesh"],
    "krishna": ["maharashtra", "karnataka", "andhra pradesh"],
    "mahanadi": ["chhattisgarh", "odisha"],
    "kaveri": ["karnataka", "tamil nadu"],
    "narmada": ["madhya pradesh", "gujarat"],
    "periyar": ["kerala"],
    "pampa": ["kerala"],
    "kosi": ["bihar"],
    "damodar": ["jharkhand", "west bengal"],
}

# State-level emergency helplines
STATE_HELPLINES: Dict[str, str] = {
    "kerala": "1077 (Kerala SDMA)",
    "karnataka": "1070 (Karnataka SDMA)",
    "tamil nadu": "1070",
    "andhra pradesh": "1070",
    "assam": "1070 (Assam SDMA)",
    "bihar": "1070",
    "maharashtra": "1077",
    "odisha": "1070 (Odisha SDMA)",
    "west bengal": "1070",
    "uttar pradesh": "1070",
}


class LocationContextBuilder:
    """
    Builds a rich LocationContext from session data and geographic knowledge.

    Usage:
        builder = LocationContextBuilder(gemini_service)
        loc_ctx = await builder.build(session_context, preprocessed_df)
    """

    def __init__(self, gemini_service: Optional[GeminiService] = None) -> None:
        self._gemini = gemini_service

    async def build(
        self,
        context: Dict[str, Any],
        df:      Optional[pd.DataFrame] = None,
    ) -> LocationContextModel:
        """
        Builds a LocationContext from all available data sources.

        Args:
            context: Session context dict.
            df:      Optional preprocessed DataFrame (for terrain data).

        Returns:
            Populated LocationContextModel.
        """
        loc = LocationContextModel(
            location_name=context.get("location"),
            latitude=context.get("latitude"),
            longitude=context.get("longitude"),
        )

        # Enrich from geographic knowledge
        loc = self._enrich_from_location_name(loc)

        # Enrich from preprocessed data
        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
            loc = self._enrich_from_data(loc, df)

        # LLM enrichment for missing fields
        if self._gemini and self._has_gaps(loc):
            loc = await self._llm_enrich(loc, context)

        # Set emergency number
        if loc.state:
            state_lower = loc.state.lower()
            loc.emergency_number = STATE_HELPLINES.get(state_lower, "112 / 1078 (NDRF)")

        logger.info(
            f"[LocationContextBuilder] Location: {loc.location_name} | "
            f"state={loc.state} | coastal={loc.is_coastal} | "
            f"flood_plain={loc.is_flood_plain} | urban={loc.is_urban}"
        )
        return loc

    # ── Rule-based enrichment ─────────────────────────────────────────────

    def _enrich_from_location_name(
        self,
        loc: LocationContextModel,
    ) -> LocationContextModel:
        """Infers state, coastal flag, and river proximity from the location name."""
        name = (loc.location_name or "").lower()

        # Detect state
        for state in COASTAL_STATES:
            if state in name:
                loc.state      = state.title()
                loc.is_coastal = True
                break

        # Detect river
        for river, states in MAJOR_RIVERS.items():
            if river in name:
                loc.river_name = river.title()
                if not loc.state:
                    loc.state = states[0].title()
                break

        # Known city → state mapping (common flood-prone cities)
        city_state = {
            "kochi": "Kerala", "ernakulam": "Kerala", "alappuzha": "Kerala",
            "patna": "Bihar", "guwahati": "Assam", "mumbai": "Maharashtra",
            "chennai": "Tamil Nadu", "kolkata": "West Bengal",
            "hyderabad": "Telangana", "bhubaneswar": "Odisha",
            "varanasi": "Uttar Pradesh", "delhi": "Delhi",
        }
        for city, state in city_state.items():
            if city in name:
                loc.state   = state
                loc.is_urban = True
                if state.lower() in COASTAL_STATES:
                    loc.is_coastal = True
                break

        return loc

    def _enrich_from_data(
        self,
        loc: LocationContextModel,
        df:  pd.DataFrame,
    ) -> LocationContextModel:
        """Extracts terrain/location info from the preprocessed DataFrame."""
        last = df.iloc[-1] if len(df) > 0 else pd.Series()

        # Elevation
        if "elevation_m" in df.columns:
            val = last.get("elevation_m")
            if pd.notna(val):
                loc.elevation_m = float(val)

        if "mean_elevation_m" in df.columns and loc.elevation_m is None:
            val = last.get("mean_elevation_m")
            if pd.notna(val):
                loc.elevation_m = float(val)

        # Flood plain risk
        if "flood_plain_risk" in df.columns:
            val = last.get("flood_plain_risk")
            if pd.notna(val) and float(val) > 0.5:
                loc.is_flood_plain = True

        # Low elevation → likely flood plain
        if loc.elevation_m is not None and loc.elevation_m < 30:
            loc.is_flood_plain = True

        # Urban area
        if "urban_area" in df.columns:
            val = last.get("urban_area")
            if pd.notna(val):
                loc.is_urban = bool(val)

        # Latitude / longitude from data if missing
        if loc.latitude is None and "latitude" in df.columns:
            val = last.get("latitude")
            if pd.notna(val):
                loc.latitude = float(val)
        if loc.longitude is None and "longitude" in df.columns:
            val = last.get("longitude")
            if pd.notna(val):
                loc.longitude = float(val)

        return loc

    # ── LLM enrichment ────────────────────────────────────────────────────

    async def _llm_enrich(
        self,
        loc:     LocationContextModel,
        context: Dict[str, Any],
    ) -> LocationContextModel:
        """Uses Gemini to fill in missing geographic knowledge."""
        try:
            prompt = f"""
Given this location, provide geographic flood-risk context:
Location: {loc.location_name or context.get('location', 'unknown')}
Lat/Lon:  {loc.latitude}, {loc.longitude}

Return ONLY a JSON object:
{{
  "state":           "<Indian state name>",
  "district":        "<district name>",
  "is_coastal":      <true/false>,
  "is_flood_plain":  <true/false>,
  "is_urban":        <true/false>,
  "river_name":      "<nearest major river or null>",
  "dam_nearby":      <true/false>,
  "nearest_shelter": "<shelter name/type or null>",
  "nearest_hospital":"<hospital name or null>"
}}
"""
            data = await self._gemini.generate_json(prompt, use_fast_model=True)
            if data and isinstance(data, dict):
                if not loc.state and data.get("state"):
                    loc.state = data["state"]
                if not loc.district and data.get("district"):
                    loc.district = data["district"]
                if data.get("is_coastal") is not None:
                    loc.is_coastal = bool(data["is_coastal"])
                if data.get("is_flood_plain") is not None:
                    loc.is_flood_plain = bool(data["is_flood_plain"])
                if data.get("is_urban") is not None:
                    loc.is_urban = bool(data["is_urban"])
                if data.get("river_name"):
                    loc.river_name = data["river_name"]
                if data.get("dam_nearby") is not None:
                    loc.dam_nearby = bool(data["dam_nearby"])
                if data.get("nearest_shelter"):
                    loc.nearest_shelter = data["nearest_shelter"]
                if data.get("nearest_hospital"):
                    loc.nearest_hospital = data["nearest_hospital"]
        except Exception as exc:
            logger.warning(f"[LocationContextBuilder] LLM enrichment failed: {exc}")
        return loc

    @staticmethod
    def _has_gaps(loc: LocationContextModel) -> bool:
        return not loc.state or not loc.district
