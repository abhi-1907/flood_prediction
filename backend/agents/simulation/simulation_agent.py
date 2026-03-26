"""
Simulation Agent – Top-level coordinator for the flood simulation pipeline.

Execution pipeline:
  1. ScenarioEngine        : Parse user query → ScenarioParameters
  2. ScenarioEngine        : Build hourly flood timeline (SCS-CN + Type-II)
  3. FloodZoneMapper       : Generate N×N inundation grid
  4. GeoJSONBuilder        : Convert zones → GeoJSON FeatureCollection
  5. ImpactAssessor        : Estimate population/economic/agriculture impact
  6. MapRenderer           : Produce frontend-ready rendering payload
  7. LLM narrative summary : Natural-language explanation of the simulation
  8. Store all artifacts   : Session storage for downstream agents

Registered in ToolRegistry as "simulation".
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from agents.simulation.simulation_schemas import (
    FloodSeverity,
    ScenarioParameters,
    SimulationResult,
    SimulationTimeStep,
)
from agents.simulation.scenario_engine import ScenarioEngine
from agents.simulation.flood_zone_mapper import FloodZoneMapper
from agents.simulation.geojson_builder import GeoJSONBuilder
from agents.simulation.impact_assessor import ImpactAssessor
from agents.simulation.map_renderer import MapRenderer
from agents.orchestration.memory import Session
from services.gemini_service import GeminiService, get_gemini_service
from utils.logger import logger


class SimulationAgent:
    """
    Orchestrates the full flood simulation pipeline.

    Produces spatial inundation maps, hourly progression timelines,
    multi-dimensional impact assessments, and frontend-ready GeoJSON/chart data.
    """

    def __init__(self, gemini_service: Optional[GeminiService] = None) -> None:
        self._gemini     = gemini_service or get_gemini_service()
        self._scenario   = ScenarioEngine(self._gemini)
        self._zone_mapper = FloodZoneMapper()
        self._geojson    = GeoJSONBuilder()
        self._impact     = ImpactAssessor(self._gemini)
        self._renderer   = MapRenderer()

    # ── Public interface (called by Orchestrator) ─────────────────────────

    async def run(
        self,
        session: Session,
        **kwargs,
    ) -> SimulationResult:
        """
        Main agent entry point.

        Args:
            session: Active orchestration session.

        Returns:
            SimulationResult with zones, timeline, impacts, GeoJSON, and map data.
        """
        session_id = session.session_id
        context    = session.context
        warnings:  List[str] = []
        errors:    List[str] = []

        logger.info(f"[SimulationAgent] Starting simulation for session {session_id}")

        try:
            # ── 0. Gather environmental data ──────────────────────────────
            env_data      = self._extract_env_data(session)
            terrain_data  = self._extract_terrain_data(session)

            # ── 1. Build scenario ─────────────────────────────────────────
            scenario = await self._scenario.build_scenario(context, env_data)

            # Override location from session if not set
            if not scenario.latitude:
                scenario.latitude = context.get("latitude")
            if not scenario.longitude:
                scenario.longitude = context.get("longitude")
            if not scenario.location:
                scenario.location = context.get("location")

            session.store_artifact("simulation_scenario", scenario.model_dump())

            # ── 2. Simulate hourly timeline ───────────────────────────────
            timeline = self._scenario.simulate_timeline(scenario, terrain_data)

            if not timeline:
                return SimulationResult(
                    session_id=session_id,
                    status="failed",
                    scenario=scenario,
                    errors=["Timeline generation failed."],
                )

            # ── 3. Generate inundation grid ───────────────────────────────
            zones = self._zone_mapper.generate(
                scenario=scenario,
                timeline=timeline,
                terrain_data=terrain_data,
            )

            # ── 4. Build GeoJSON ──────────────────────────────────────────
            geojson = self._geojson.build(zones, scenario)
            session.store_artifact("simulation_geojson", geojson)

            # ── 5. Assess impact ──────────────────────────────────────────
            impacts = await self._impact.assess(
                zones=zones,
                scenario=scenario,
                timeline=timeline,
            )
            session.store_artifact("simulation_impacts",
                                   [i.model_dump() for i in impacts])

            # ── 6. Render map data ────────────────────────────────────────
            map_data = self._renderer.render(
                zones=zones,
                timeline=timeline,
                geojson=geojson,
                scenario=scenario,
                impacts=impacts,
            )
            session.store_artifact("simulation_map_data", map_data)

            # ── 7. Compute summary statistics ─────────────────────────────
            peak_step = max(timeline, key=lambda s: s.water_level_m)
            total_area_km2 = map_data.get("severity_stats", {}).get("total_area_km2", 0)
            inundated_pct  = map_data.get("severity_stats", {}).get("inundated_pct", 0)

            # ── 8. Single merged LLM call: impact narrative + summary ──────
            summary = await self._generate_merged_narrative(
                scenario, timeline, impacts, peak_step, total_area_km2
            )

            # ── 9. Warnings ───────────────────────────────────────────────
            if peak_step.flood_severity in (FloodSeverity.SEVERE, FloodSeverity.EXTREME):
                warnings.append(
                    f"⚠️ Peak severity: {peak_step.flood_severity.value} "
                    f"at H+{peak_step.hour} with water level {peak_step.water_level_m:.2f}m."
                )

            status = "success" if not errors else "partial"

            logger.info(
                f"[SimulationAgent] Complete: {len(zones)} zones | "
                f"{len(timeline)} steps | {len(impacts)} impacts | "
                f"peak={peak_step.water_level_m:.2f}m @ H+{peak_step.hour}"
            )

            return {
                "session_id": session_id,
                "status": status,
                "scenario_name": scenario.name,
                "location": {"lat": scenario.latitude, "lon": scenario.longitude, "name": scenario.location},
                "flood_risk_score": peak_step.risk_score,
                "affected_area_km2": total_area_km2,
                "peak_depth_m": round(max(z.depth_m for z in zones) if zones else 0, 3),
                "peak_hour": peak_step.hour,
                "peak_discharge_m3s": peak_step.discharge_m3s,
                "inundated_pct": inundated_pct,
                "timeline_chart": [s.model_dump() for s in timeline],
                "impact_summary": [i.model_dump() for i in impacts],
                "geojson": geojson,
                "summary": summary,
                "warnings": warnings,
                "errors": errors,
            }

        except Exception as exc:
            logger.exception(f"[SimulationAgent] Unhandled error: {exc}")
            return SimulationResult(
                session_id=session_id,
                status="failed",
                errors=[str(exc)],
            )

    # ── Data extraction helpers ───────────────────────────────────────────

    @staticmethod
    def _extract_env_data(session: Session) -> Dict[str, Any]:
        """Extracts environmental data from session artifacts."""
        env: Dict[str, Any] = {}

        # From prediction results
        raw_pred = session.get_artifact("prediction_result") or session.get_artifact("ensemble_prediction")
        if raw_pred and isinstance(raw_pred, dict):
            # Handle nested data if we got 'prediction_result'
            pred = raw_pred.get("ensemble") if "ensemble" in raw_pred and isinstance(raw_pred["ensemble"], dict) else raw_pred
            
            env["flood_probability"] = pred.get("flood_probability", 0.5)
            env["risk_level"]        = pred.get("risk_level", "MEDIUM")

        # From preprocessed data
        df = session.get_artifact("processed_dataset")
        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
            last = df.iloc[-1]
            for col in ["rainfall_mm", "water_level_m", "discharge_m3s",
                        "soil_moisture_pct", "elevation_m", "temperature_c"]:
                if col in df.columns:
                    val = last.get(col)
                    if pd.notna(val):
                        env[col] = float(val)

        return env

    @staticmethod
    def _extract_terrain_data(session: Session) -> Dict[str, Any]:
        """Extracts terrain data from session artifacts."""
        terrain: Dict[str, Any] = {"elevation_m": 30.0, "mean_slope_deg": 1.0}

        df = session.get_artifact("processed_dataset")
        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
            last = df.iloc[-1]
            for col in ["elevation_m", "mean_elevation_m", "mean_slope_deg",
                        "flood_plain_risk", "terrain_wetness_idx"]:
                if col in df.columns:
                    val = last.get(col)
                    if pd.notna(val):
                        terrain[col] = float(val)

            # Land use
            if "land_use" in df.columns:
                terrain["land_use"] = str(last.get("land_use", "default"))

            # Urban flag
            if "urban_area" in df.columns:
                terrain["is_urban"] = bool(last.get("urban_area", False))

        return terrain

    # ── Single merged LLM narrative ───────────────────────────────────────

    async def _generate_merged_narrative(
        self,
        scenario:       ScenarioParameters,
        timeline:       List[SimulationTimeStep],
        impacts:        list,
        peak_step:      SimulationTimeStep,
        total_area_km2: float,
    ) -> str:
        """Single LLM call replacing: ImpactAssessor._llm_enhance + _generate_summary.

        Produces a 3-4 sentence narrative covering:
        - Most critical life-safety finding from impacts
        - Time-sensitive action needed
        - Overall simulation summary (peak level, severity, area)
        """
        try:
            impact_lines = "\n".join(
                f"- {i.metric}: {i.value} {i.unit}" for i in impacts[:6]
            )
            prompt = f"""Summarize this flood simulation in 3–4 sentences.

Scenario: {scenario.name}
Location: {scenario.location or 'Target area'}
Rainfall: {scenario.rainfall_mm}mm over {scenario.rainfall_days} day(s)
Peak water level: {peak_step.water_level_m:.2f}m at hour {peak_step.hour}
Peak severity: {peak_step.flood_severity.value}
Area simulated: {total_area_km2:.1f} km²
Timeline: {len(timeline)} hours

Key impacts:
{impact_lines}

Include: (1) most critical life-safety finding, (2) time-sensitive action needed.
Be factual and concise. No headers, no bullets."""

            summary = await self._gemini.generate(prompt, use_fast_model=True)
            return summary.strip() if summary else self._fallback_summary(
                scenario, peak_step, total_area_km2
            )
        except Exception:
            return self._fallback_summary(scenario, peak_step, total_area_km2)

    @staticmethod
    def _fallback_summary(
        scenario:  ScenarioParameters,
        peak_step: SimulationTimeStep,
        area:      float,
    ) -> str:
        return (
            f"Simulation '{scenario.name}' for {scenario.location or 'target area'}: "
            f"{scenario.rainfall_mm}mm rainfall over {scenario.rainfall_days} day(s). "
            f"Peak water level {peak_step.water_level_m:.2f}m reached at hour {peak_step.hour} "
            f"({peak_step.flood_severity.value} severity). "
            f"Total simulated area: {area:.1f} km²."
        )

