"""
Scenario Engine – Generates and manages simulation parameters.

Responsibilities:
  1. Parse user what-if queries ("What if 200mm rain falls in 3 days?")
  2. Generate extreme scenarios (50-yr, 100-yr return period floods)
  3. Calculate derived parameters (runoff, peak discharge)
  4. Build multi-day timeline with hourly progression
  5. Provide LLM-based scenario design for complex queries
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agents.simulation.simulation_schemas import (
    FloodSeverity,
    ScenarioParameters,
    ScenarioType,
    SimulationTimeStep,
)
from services.gemini_service import GeminiService
from utils.logger import logger


# ── Hydrological constants ────────────────────────────────────────────────────

# SCS Curve Numbers (CN) by land use — higher CN = more runoff
CN_TABLE = {
    "urban":        92,
    "residential":  85,
    "agricultural": 75,
    "forest":       60,
    "barren":       90,
    "wetland":      78,
    "default":      80,
}

# Return-period rainfall depths (mm/day) for Indian conditions (approximate)
RETURN_PERIOD_RAINFALL = {
    10:  150,
    25:  220,
    50:  300,
    100: 400,
    200: 500,
    500: 650,
}

# Manning's roughness coefficients
MANNING_N = {
    "river": 0.035,
    "floodplain": 0.06,
    "urban": 0.015,
    "vegetated": 0.08,
}


class ScenarioEngine:
    """
    Generates simulation scenarios from user queries and contextual data.

    Usage:
        engine = ScenarioEngine(gemini_service)
        scenario = await engine.build_scenario(context, env_data)
        timeline = engine.simulate_timeline(scenario, terrain_data)
    """

    def __init__(self, gemini_service: Optional[GeminiService] = None) -> None:
        self._gemini = gemini_service

    # ── Public API ────────────────────────────────────────────────────────

    async def build_scenario(
        self,
        context:    Dict[str, Any],
        env_data:   Dict[str, Any],
    ) -> ScenarioParameters:
        """
        Builds a ScenarioParameters from user query and environmental data.

        Uses LLM to interpret natural-language what-if queries.
        """
        query = context.get("original_query", "").lower()

        # Rule-based scenario detection
        scenario = self._rule_based_parse(query, env_data, context)

        # LLM parse intentionally skipped — rule-based is sufficient and
        # saves one Gemini API call. ScenarioEngine._llm_parse still exists
        # as fallback if re-enabled here.

        logger.info(
            f"[ScenarioEngine] Built scenario: type={scenario.scenario_type} | "
            f"rainfall={scenario.rainfall_mm}mm × {scenario.rainfall_days}d | "
            f"location={scenario.location}"
        )
        return scenario

    def simulate_timeline(
        self,
        scenario:      ScenarioParameters,
        terrain_data:  Dict[str, Any],
    ) -> List[SimulationTimeStep]:
        """
        Generates an hour-by-hour flood progression timeline.

        Uses SCS-CN runoff model + simplified routing for water level estimation.
        """
        total_hours = scenario.rainfall_days * 24
        rainfall_mm = scenario.rainfall_mm or 100.0
        base_wl     = scenario.water_level_m or 2.0
        base_q      = scenario.discharge_m3s or 50.0
        soil_moist  = scenario.soil_moisture_pct or 50.0
        elevation   = terrain_data.get("elevation_m", scenario.elevation_m or 50.0)
        land_use    = terrain_data.get("land_use", "default")

        # SCS-CN runoff calculation
        cn       = CN_TABLE.get(land_use, CN_TABLE["default"])
        s        = (25400 / cn) - 254   # Maximum retention (mm)
        ia       = 0.2 * s               # Initial abstraction (mm)

        # Adjusted for soil moisture (higher moisture → more runoff)
        moisture_factor = 1 + (soil_moist - 50) / 100  # 0.5–1.5x

        timeline: List[SimulationTimeStep] = []

        # Rainfall distribution: SCS Type-II storm (bell-shaped peak)
        peak_hour = total_hours * 0.4   # Peak around 40% into the storm

        for hour in range(total_hours):
            # Hourly rainfall (SCS Type-II shaped)
            t_norm      = hour / max(total_hours, 1)
            rain_factor = self._scs_type2_distribution(t_norm)
            hourly_rain = rainfall_mm * rain_factor / max(total_hours / 24, 1)

            # Cumulative rainfall up to this hour
            cum_rain = sum(
                rainfall_mm * self._scs_type2_distribution(h / max(total_hours, 1))
                / max(total_hours / 24, 1)
                for h in range(hour + 1)
            )

            # SCS-CN runoff (cumulative)
            if cum_rain > ia:
                runoff_mm = ((cum_rain - ia) ** 2) / (cum_rain - ia + s)
            else:
                runoff_mm = 0.0
            runoff_mm *= moisture_factor

            # Water level: base + runoff-derived rise (simplified)
            wl_rise    = runoff_mm / 200   # Rule of thumb: 200mm runoff ≈ 1m rise
            water_level = base_wl + wl_rise

            # Discharge: Manning's equation proxy
            discharge = base_q + runoff_mm * 1.5 * moisture_factor

            # Inundation % (logistic function of depth above danger level)
            danger_level = base_wl + 2.0   # 2m above normal = danger
            depth_above  = max(0, water_level - danger_level)
            inundation   = min(100, 100 * (1 - math.exp(-depth_above / 1.5)))

            # Severity classification
            if depth_above < 0.3:
                severity = FloodSeverity.MINOR
            elif depth_above < 1.0:
                severity = FloodSeverity.MODERATE
            elif depth_above < 3.0:
                severity = FloodSeverity.SEVERE
            else:
                severity = FloodSeverity.EXTREME

            # Risk score (0–100)
            risk = min(100, inundation * 0.5 + depth_above * 20 + hourly_rain * 0.3)

            timeline.append(SimulationTimeStep(
                hour=hour,
                rainfall_mm=round(hourly_rain, 2),
                water_level_m=round(water_level, 3),
                discharge_m3s=round(discharge, 1),
                inundation_pct=round(inundation, 2),
                flood_severity=severity,
                risk_score=round(risk, 1),
            ))

        return timeline

    # ── SCS Type-II storm distribution ─────────────────────────────────────

    @staticmethod
    def _scs_type2_distribution(t_norm: float) -> float:
        """
        SCS Type-II dimensionless hyetograph (bell curve around t=0.4).
        Returns the fraction of total rainfall at normalised time t (0–1).
        """
        peak = 0.4
        sigma = 0.12
        return math.exp(-0.5 * ((t_norm - peak) / sigma) ** 2)

    # ── Rule-based query parsing ──────────────────────────────────────────

    def _rule_based_parse(
        self,
        query:    str,
        env_data: Dict[str, Any],
        context:  Dict[str, Any],
    ) -> ScenarioParameters:
        """Parses common what-if patterns from the user query."""
        scenario = ScenarioParameters(
            location=context.get("location"),
            latitude=context.get("latitude"),
            longitude=context.get("longitude"),
            rainfall_mm=env_data.get("rainfall_mm", 100),
            water_level_m=env_data.get("water_level_m"),
            discharge_m3s=env_data.get("discharge_m3s"),
            soil_moisture_pct=env_data.get("soil_moisture_pct"),
            elevation_m=env_data.get("elevation_m"),
        )

        # Detect scenario type
        if "dam" in query and ("break" in query or "fail" in query or "release" in query):
            scenario.scenario_type = ScenarioType.DAM_BREAK
            scenario.name = "Dam Break/Release Scenario"
            scenario.dam_release_m3s = self._extract_number(query, "m3/s", 500)

        elif any(rp in query for rp in ["100 year", "50 year", "return period", "100-year", "50-year"]):
            scenario.scenario_type = ScenarioType.EXTREME
            rp = self._extract_return_period(query)
            scenario.return_period_years = rp
            scenario.rainfall_mm = RETURN_PERIOD_RAINFALL.get(rp, 300)
            scenario.name = f"{rp}-Year Return Period Flood"

        elif "worst case" in query or "extreme" in query or "maximum" in query:
            scenario.scenario_type = ScenarioType.EXTREME
            scenario.rainfall_mm = 400
            scenario.rainfall_days = 3
            scenario.soil_moisture_pct = 95
            scenario.name = "Worst-Case Extreme Scenario"

        elif "historical" in query or "past" in query or "replay" in query:
            scenario.scenario_type = ScenarioType.HISTORICAL
            scenario.name = "Historical Replay Scenario"

        else:
            scenario.scenario_type = ScenarioType.WHAT_IF
            scenario.name = "Custom What-If Scenario"

        # Extract numeric overrides from query
        rain = self._extract_number(query, "mm", None)
        if rain:
            scenario.rainfall_mm = rain

        days = self._extract_number(query, "day", None)
        if days and days <= 30:
            scenario.rainfall_days = int(days)

        return scenario

    # ── LLM scenario parsing ─────────────────────────────────────────────

    async def _llm_parse(
        self,
        scenario: ScenarioParameters,
        query:    str,
        env_data: Dict[str, Any],
        context:  Dict[str, Any],
    ) -> ScenarioParameters:
        """Uses Gemini to interpret complex what-if queries."""
        try:
            prompt = f"""
Parse this flood simulation query and extract parameters:

Query: "{query}"
Location: {context.get('location', 'unknown')}
Current conditions: {json.dumps(env_data, default=str)[:300]}

Return ONLY a JSON object:
{{
  "scenario_type": "<what_if | historical | extreme | dam_break | multi_day>",
  "name": "<descriptive name, max 8 words>",
  "description": "<1 sentence description>",
  "rainfall_mm": <daily rainfall in mm or null>,
  "rainfall_days": <number of days, 1-30>,
  "water_level_m": <starting water level or null>,
  "discharge_m3s": <river discharge or null>,
  "soil_moisture_pct": <0-100 or null>,
  "dam_release_m3s": <dam release rate or null>,
  "return_period_years": <10/25/50/100/200/500 or null>
}}
"""
            data = await self._gemini.generate_json(prompt, use_fast_model=True)
            if data and isinstance(data, dict):
                if data.get("rainfall_mm"):
                    scenario.rainfall_mm = float(data["rainfall_mm"])
                if data.get("rainfall_days"):
                    scenario.rainfall_days = int(data["rainfall_days"])
                if data.get("water_level_m"):
                    scenario.water_level_m = float(data["water_level_m"])
                if data.get("discharge_m3s"):
                    scenario.discharge_m3s = float(data["discharge_m3s"])
                if data.get("soil_moisture_pct"):
                    scenario.soil_moisture_pct = float(data["soil_moisture_pct"])
                if data.get("dam_release_m3s"):
                    scenario.dam_release_m3s = float(data["dam_release_m3s"])
                if data.get("return_period_years"):
                    scenario.return_period_years = int(data["return_period_years"])
                if data.get("name"):
                    scenario.name = data["name"]
                if data.get("description"):
                    scenario.description = data["description"]
                try:
                    scenario.scenario_type = ScenarioType(data.get("scenario_type", scenario.scenario_type))
                except ValueError:
                    pass
        except Exception as exc:
            logger.warning(f"[ScenarioEngine] LLM parse failed: {exc}")
        return scenario

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _extract_number(text: str, unit: str, default=None):
        """Extracts a number preceding or following a unit keyword."""
        import re
        patterns = [
            rf"(\d+\.?\d*)\s*{unit}",
            rf"{unit}\s*(\d+\.?\d*)",
        ]
        for p in patterns:
            match = re.search(p, text, re.IGNORECASE)
            if match:
                return float(match.group(1))
        return default

    @staticmethod
    def _extract_return_period(text: str) -> int:
        import re
        match = re.search(r"(\d+)[\s-]*year", text, re.IGNORECASE)
        if match:
            rp = int(match.group(1))
            return min(rp, 500)
        return 100
