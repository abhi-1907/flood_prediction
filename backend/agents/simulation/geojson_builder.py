"""
GeoJSON Builder – Converts FloodZone data into GeoJSON FeatureCollections.

Produces GeoJSON suitable for rendering on Leaflet / Mapbox maps.

Features:
  - Each FloodZone becomes a polygon (rectangle for grid cells)
  - Color-coded by severity (minor→blue, moderate→yellow, severe→orange, extreme→red)
  - Properties include depth, severity, elevation, land_use for tooltips
  - Generates both polygon (heatmap-style) and point (marker) GeoJSON
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from agents.simulation.simulation_schemas import (
    FloodSeverity,
    FloodZone,
    GeoJSONFeature,
    ScenarioParameters,
)
from utils.logger import logger


# ── Severity → color mapping ──────────────────────────────────────────────────

SEVERITY_COLORS = {
    FloodSeverity.MINOR:    "#4FC3F7",   # Light blue
    FloodSeverity.MODERATE: "#FFD54F",   # Amber
    FloodSeverity.SEVERE:   "#FF7043",   # Deep orange
    FloodSeverity.EXTREME:  "#E53935",   # Red
}

SEVERITY_OPACITY = {
    FloodSeverity.MINOR:    0.3,
    FloodSeverity.MODERATE: 0.5,
    FloodSeverity.SEVERE:   0.7,
    FloodSeverity.EXTREME:  0.85,
}


class GeoJSONBuilder:
    """
    Converts FloodZone lists into GeoJSON FeatureCollections.

    Usage:
        builder = GeoJSONBuilder()
        geojson = builder.build(flood_zones, scenario, cell_km=0.5)
    """

    def build(
        self,
        zones:    List[FloodZone],
        scenario: ScenarioParameters,
        cell_km:  float = 0.5,
    ) -> Dict[str, Any]:
        """
        Builds a complete GeoJSON FeatureCollection.

        Args:
            zones:    List of FloodZone objects from FloodZoneMapper.
            scenario: The simulation scenario.
            cell_km:  Grid cell size in km (for polygon dimensions).

        Returns:
            GeoJSON dict suitable for Leaflet/Mapbox.
        """
        features = []

        for zone in zones:
            if zone.depth_m < 0.01:
                continue   # Skip dry cells

            # Build polygon for this cell
            polygon = self._cell_polygon(
                zone.latitude, zone.longitude, cell_km
            )

            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [polygon],
                },
                "properties": {
                    "depth_m":     zone.depth_m,
                    "severity":    zone.severity.value,
                    "elevation_m": zone.elevation_m,
                    "land_use":    zone.land_use,
                    "is_populated": zone.is_populated,
                    "fill_color":  SEVERITY_COLORS.get(zone.severity, "#4FC3F7"),
                    "fill_opacity": SEVERITY_OPACITY.get(zone.severity, 0.5),
                    "tooltip":     (
                        f"Depth: {zone.depth_m:.2f}m | "
                        f"Severity: {zone.severity.value} | "
                        f"Elevation: {zone.elevation_m}m"
                    ),
                },
            }
            features.append(feature)

        # Add centre marker
        if scenario.latitude and scenario.longitude:
            features.append(self._centre_marker(scenario))

        geojson = {
            "type": "FeatureCollection",
            "features": features,
            "properties": {
                "scenario":     scenario.name,
                "total_zones":  len(zones),
                "inundated":    len(features) - (1 if scenario.latitude else 0),
                "cell_km":      cell_km,
            },
        }

        logger.info(
            f"[GeoJSONBuilder] Built FeatureCollection with {len(features)} features"
        )
        return geojson

    def build_timeline_geojson(
        self,
        zones_over_time: Dict[int, List[FloodZone]],
        scenario:        ScenarioParameters,
        cell_km:         float = 0.5,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Builds per-hour GeoJSON for animated timeline rendering.

        Args:
            zones_over_time: {hour: [FloodZone, ...]}
            scenario:        The simulation scenario.
            cell_km:         Grid cell size.

        Returns:
            {hour: geojson_dict}
        """
        result = {}
        for hour, zones in zones_over_time.items():
            result[hour] = self.build(zones, scenario, cell_km)
        return result

    # ── Cell polygon builder ──────────────────────────────────────────────

    @staticmethod
    def _cell_polygon(
        lat:     float,
        lon:     float,
        cell_km: float,
    ) -> List[List[float]]:
        """Builds a rectangular polygon around the cell centre."""
        half_km = cell_km / 2
        dlat    = half_km / 111.32
        dlon    = half_km / (111.32 * math.cos(math.radians(lat)))

        # GeoJSON uses [lon, lat] order
        return [
            [round(lon - dlon, 6), round(lat - dlat, 6)],
            [round(lon + dlon, 6), round(lat - dlat, 6)],
            [round(lon + dlon, 6), round(lat + dlat, 6)],
            [round(lon - dlon, 6), round(lat + dlat, 6)],
            [round(lon - dlon, 6), round(lat - dlat, 6)],   # Close ring
        ]

    @staticmethod
    def _centre_marker(scenario: ScenarioParameters) -> Dict[str, Any]:
        """Creates a GeoJSON Point feature for the scenario centre."""
        return {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [scenario.longitude, scenario.latitude],
            },
            "properties": {
                "marker_type": "centre",
                "name":        scenario.location or "Simulation Centre",
                "scenario":    scenario.name,
                "icon":        "warning",
                "fill_color":  "#FF0000",
            },
        }
