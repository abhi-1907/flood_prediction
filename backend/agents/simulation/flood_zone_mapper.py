"""
Flood Zone Mapper – Generates spatial flood inundation zones.

Creates a grid of FloodZone cells around the target location, calculating
estimated inundation depth for each cell based on:
  - Elevation relative to water level
  - Distance from the river/water source
  - Terrain slope and land use
  - SCS-CN runoff from the ScenarioEngine timeline

The output feeds into GeoJSONBuilder for map visualization.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from agents.simulation.simulation_schemas import (
    FloodSeverity,
    FloodZone,
    ScenarioParameters,
    SimulationTimeStep,
)
from utils.logger import logger


# ── Grid configuration ────────────────────────────────────────────────────────

DEFAULT_GRID_SIZE  = 11      # 11×11 grid = 121 cells
DEFAULT_CELL_KM    = 0.5     # 500m per cell
EARTH_RADIUS_KM    = 6371.0


class FloodZoneMapper:
    """
    Generates a spatial flood inundation grid around the target location.

    Usage:
        mapper = FloodZoneMapper()
        zones  = mapper.generate(scenario, timeline, terrain_data)
    """

    def generate(
        self,
        scenario:     ScenarioParameters,
        timeline:     List[SimulationTimeStep],
        terrain_data: Dict[str, Any],
        grid_size:    int   = DEFAULT_GRID_SIZE,
        cell_km:      float = DEFAULT_CELL_KM,
    ) -> List[FloodZone]:
        """
        Generates flood zones by creating a grid and calculating
        inundation depth at each cell.

        Args:
            scenario:     Simulation scenario with location and parameters.
            timeline:     Simulated hourly timeline from ScenarioEngine.
            terrain_data: Terrain data (elevation, slope, land_use).
            grid_size:    Number of cells per side (grid_size × grid_size).
            cell_km:      Size of each cell in km.

        Returns:
            List of FloodZone objects with inundation depths.
        """
        lat = scenario.latitude or 10.0
        lon = scenario.longitude or 76.0
        base_elev = terrain_data.get("elevation_m", scenario.elevation_m or 30.0)

        # Peak water level from timeline
        if timeline:
            peak_step = max(timeline, key=lambda s: s.water_level_m)
            peak_wl   = peak_step.water_level_m
        else:
            peak_wl = (scenario.water_level_m or 2.0) + (scenario.rainfall_mm or 100) / 200

        zones: List[FloodZone] = []
        half = grid_size // 2

        for i in range(-half, half + 1):
            for j in range(-half, half + 1):
                # Cell coordinates
                cell_lat = lat + self._km_to_deg_lat(i * cell_km)
                cell_lon = lon + self._km_to_deg_lon(j * cell_km, lat)

                # Distance from centre (river/danger point)
                dist_km = math.sqrt((i * cell_km) ** 2 + (j * cell_km) ** 2)

                # Elevation: add some terrain variation
                elev_variation = self._terrain_variation(i, j, base_elev, terrain_data)
                cell_elev = base_elev + elev_variation

                # Inundation depth calculation
                depth_m = self._calculate_depth(
                    peak_wl, cell_elev, dist_km, terrain_data
                )

                # Severity
                severity = self._depth_to_severity(depth_m)

                # Land use inference from distance
                if dist_km < 1.0:
                    land_use = "urban" if terrain_data.get("is_urban") else "residential"
                    is_populated = True
                elif dist_km < 2.5:
                    land_use = "residential"
                    is_populated = True
                else:
                    land_use = "agricultural"
                    is_populated = False

                zones.append(FloodZone(
                    latitude=round(cell_lat, 6),
                    longitude=round(cell_lon, 6),
                    depth_m=round(max(0, depth_m), 3),
                    severity=severity,
                    elevation_m=round(cell_elev, 2),
                    is_populated=is_populated,
                    land_use=land_use,
                ))

        inundated = sum(1 for z in zones if z.depth_m > 0)
        logger.info(
            f"[FloodZoneMapper] Generated {len(zones)} cells | "
            f"{inundated} inundated ({inundated/len(zones)*100:.0f}%) | "
            f"peak_wl={peak_wl:.2f}m"
        )
        return zones

    # ── Depth calculation ─────────────────────────────────────────────────

    @staticmethod
    def _calculate_depth(
        peak_wl:      float,
        cell_elev:    float,
        dist_from_centre_km: float,
        terrain_data: Dict[str, Any],
    ) -> float:
        """
        Estimates inundation depth at a cell.

        depth = (peak_water_level + river_elevation) - cell_elevation - attenuation
        """
        river_elev  = terrain_data.get("elevation_m", 30.0)
        flood_level = river_elev + peak_wl   # Absolute flood level (m ASL)

        # Attenuation: water depth decreases with distance from source
        slope = terrain_data.get("mean_slope_deg", 1.0)
        attenuation = dist_from_centre_km * max(slope, 0.5) * 0.5   # metres

        depth = flood_level - cell_elev - attenuation
        return max(0, depth)

    # ── Terrain variation ─────────────────────────────────────────────────

    @staticmethod
    def _terrain_variation(
        i: int, j: int,
        base_elev: float,
        terrain_data: Dict[str, Any],
    ) -> float:
        """
        Generates realistic terrain variation.
        Uses a combination of distance-decay and pseudo-random variation.
        """
        slope_deg = terrain_data.get("mean_slope_deg", 1.0)
        dist = math.sqrt(i * i + j * j) * 0.5   # km from centre

        # Elevation increases with distance (away from river/valley)
        elevation_rise = dist * math.tan(math.radians(slope_deg)) * 1000 * 0.001

        # Pseudo-random micro-topography (deterministic from grid position)
        micro = ((i * 7 + j * 13 + 17) % 20 - 10) * 0.3

        return elevation_rise + micro

    # ── Severity classification ───────────────────────────────────────────

    @staticmethod
    def _depth_to_severity(depth_m: float) -> FloodSeverity:
        if depth_m < 0.01:
            return FloodSeverity.MINOR
        if depth_m < 0.3:
            return FloodSeverity.MINOR
        if depth_m < 1.0:
            return FloodSeverity.MODERATE
        if depth_m < 3.0:
            return FloodSeverity.SEVERE
        return FloodSeverity.EXTREME

    # ── Coordinate conversion ─────────────────────────────────────────────

    @staticmethod
    def _km_to_deg_lat(km: float) -> float:
        return km / 111.32

    @staticmethod
    def _km_to_deg_lon(km: float, lat: float) -> float:
        return km / (111.32 * math.cos(math.radians(lat)))
