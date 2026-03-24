"""
Map Renderer – Generates static and interactive map data for the frontend.

Produces:
  1. Summary statistics for the simulation (peak depth, area, etc.)
  2. Chart-ready data series for the timeline (Recharts format)
  3. Severity statistics for dashboard gauges
  4. The GeoJSON is produced by GeoJSONBuilder; this module enriches it
     with map configuration (bounds, zoom, legend, layer controls)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from agents.simulation.simulation_schemas import (
    FloodSeverity,
    FloodZone,
    ImpactEstimate,
    ScenarioParameters,
    SimulationTimeStep,
)
from utils.logger import logger


class MapRenderer:
    """
    Prepares map and chart data for the frontend simulation view.

    Usage:
        renderer = MapRenderer()
        map_data = renderer.render(zones, timeline, geojson, scenario, impacts)
    """

    def render(
        self,
        zones:    List[FloodZone],
        timeline: List[SimulationTimeStep],
        geojson:  Dict[str, Any],
        scenario: ScenarioParameters,
        impacts:  List[ImpactEstimate],
        cell_km:  float = 0.5,
    ) -> Dict[str, Any]:
        """
        Produces a full frontend-ready rendering payload.

        Returns:
            Dict with keys: map_config, geojson, timeline_chart,
            severity_stats, impact_summary, scenario_info.
        """
        # ── 1. Map bounds and config ──────────────────────────────────────
        map_config = self._map_config(zones, scenario)

        # ── 2. Timeline chart data (Recharts format) ──────────────────────
        timeline_chart = self._timeline_chart(timeline)

        # ── 3. Severity statistics ────────────────────────────────────────
        severity_stats = self._severity_stats(zones, cell_km)

        # ── 4. Impact summary ─────────────────────────────────────────────
        impact_summary = self._impact_summary(impacts)

        # ── 5. Peak statistics ────────────────────────────────────────────
        peak_stats = self._peak_stats(timeline)

        # ── 6. Scenario info card ─────────────────────────────────────────
        scenario_info = {
            "name":         scenario.name,
            "type":         scenario.scenario_type.value,
            "location":     scenario.location,
            "rainfall_mm":  scenario.rainfall_mm,
            "duration_days": scenario.rainfall_days,
            "return_period": scenario.return_period_years,
        }

        # ── 7. Legend data ────────────────────────────────────────────────
        legend = [
            {"label": "Minor (<0.3m)",     "color": "#4FC3F7"},
            {"label": "Moderate (0.3–1m)", "color": "#FFD54F"},
            {"label": "Severe (1–3m)",     "color": "#FF7043"},
            {"label": "Extreme (>3m)",     "color": "#E53935"},
        ]

        result = {
            "map_config":     map_config,
            "geojson":        geojson,
            "timeline_chart": timeline_chart,
            "severity_stats": severity_stats,
            "impact_summary": impact_summary,
            "peak_stats":     peak_stats,
            "scenario_info":  scenario_info,
            "legend":         legend,
        }

        logger.info(
            f"[MapRenderer] Rendered: {len(zones)} zones | "
            f"{len(timeline)} timeline steps | {len(impacts)} impacts"
        )
        return result

    # ── Map configuration ─────────────────────────────────────────────────

    @staticmethod
    def _map_config(
        zones:    List[FloodZone],
        scenario: ScenarioParameters,
    ) -> Dict[str, Any]:
        """Calculates map center, zoom level, and bounding box."""
        if not zones:
            return {
                "center": [scenario.latitude or 10.0, scenario.longitude or 76.0],
                "zoom": 12,
            }

        lats = [z.latitude for z in zones]
        lons = [z.longitude for z in zones]

        return {
            "center": [
                (min(lats) + max(lats)) / 2,
                (min(lons) + max(lons)) / 2,
            ],
            "zoom": 13,
            "bounds": [
                [min(lats), min(lons)],
                [max(lats), max(lons)],
            ],
        }

    # ── Timeline chart data ───────────────────────────────────────────────

    @staticmethod
    def _timeline_chart(
        timeline: List[SimulationTimeStep],
    ) -> List[Dict[str, Any]]:
        """Converts timeline to Recharts-compatible data series."""
        chart = []
        for step in timeline:
            chart.append({
                "hour":           step.hour,
                "label":          f"H+{step.hour}",
                "rainfall_mm":    step.rainfall_mm,
                "water_level_m":  step.water_level_m,
                "discharge_m3s":  step.discharge_m3s,
                "inundation_pct": step.inundation_pct,
                "risk_score":     step.risk_score,
                "severity":       step.flood_severity.value,
            })
        return chart

    # ── Severity statistics ───────────────────────────────────────────────

    @staticmethod
    def _severity_stats(
        zones:   List[FloodZone],
        cell_km: float,
    ) -> Dict[str, Any]:
        """Counts zones and area per severity level."""
        cell_area = cell_km * cell_km
        stats: Dict[str, Any] = {
            "total_cells":     len(zones),
            "inundated_cells": sum(1 for z in zones if z.depth_m > 0.01),
        }

        for sev in FloodSeverity:
            count = sum(1 for z in zones if z.severity == sev and z.depth_m > 0.01)
            stats[f"{sev.value}_count"] = count
            stats[f"{sev.value}_area_km2"] = round(count * cell_area, 2)

        total_inundated = stats["inundated_cells"]
        stats["inundated_pct"] = round(
            total_inundated / max(len(zones), 1) * 100, 1
        )
        stats["total_area_km2"]     = round(len(zones) * cell_area, 2)
        stats["inundated_area_km2"] = round(total_inundated * cell_area, 2)

        return stats

    # ── Impact summary ────────────────────────────────────────────────────

    @staticmethod
    def _impact_summary(impacts: List[ImpactEstimate]) -> List[Dict[str, Any]]:
        """Converts impact estimates to frontend-ready card data."""
        return [
            {
                "category":    i.category.value,
                "metric":      i.metric,
                "value":       i.value,
                "unit":        i.unit,
                "description": i.description,
                "confidence":  i.confidence,
            }
            for i in impacts
        ]

    # ── Peak statistics ───────────────────────────────────────────────────

    @staticmethod
    def _peak_stats(timeline: List[SimulationTimeStep]) -> Dict[str, Any]:
        """Extracts peak values from the simulation timeline."""
        if not timeline:
            return {}

        peak_wl   = max(timeline, key=lambda s: s.water_level_m)
        peak_rain = max(timeline, key=lambda s: s.rainfall_mm)
        peak_risk = max(timeline, key=lambda s: s.risk_score)

        return {
            "peak_water_level_m": round(peak_wl.water_level_m, 3),
            "peak_water_level_hour": peak_wl.hour,
            "peak_rainfall_mm":   round(peak_rain.rainfall_mm, 2),
            "peak_rainfall_hour": peak_rain.hour,
            "peak_risk_score":    round(peak_risk.risk_score, 1),
            "peak_risk_hour":     peak_risk.hour,
            "peak_severity":      peak_wl.flood_severity.value,
            "total_duration_hours": len(timeline),
        }
