"""
Pydantic response schemas for the Simulation API.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class SimulationRequest(BaseModel):
    """Request to run a what-if flood simulation."""
    session_id:          Optional[str]   = None
    query:               Optional[str]   = None   # Natural language scenario
    location:            Optional[str]   = None
    latitude:            Optional[float] = None
    longitude:           Optional[float] = None
    radius_km:           float           = 50.0
    rainfall_mm:         Optional[float] = None
    rainfall_days:       int             = 1
    water_level_m:       Optional[float] = None
    soil_moisture_pct:   Optional[float] = None
    return_period_years: Optional[int]   = None   # 10, 25, 50, 100, 500


class FloodZoneSummary(BaseModel):
    """Summary of a flood zone cell for API response."""
    lat:         float
    lon:         float
    depth_m:     float
    severity:    str      # minor | moderate | severe | extreme
    land_use:    str
    is_populated: bool = False


class ImpactSummary(BaseModel):
    """A single impact metric for the summary card."""
    metric:      str
    value:       float
    unit:        str
    severity:    str
    description: Optional[str] = None


class SimulationResponse(BaseModel):
    """Full simulation response."""
    session_id:     str
    status:         str
    scenario_name:  Optional[str]  = None
    location:       Optional[str]  = None
    peak_depth_m:   float          = 0.0
    peak_hour:      int            = 0
    total_area_km2: float          = 0.0
    inundated_pct:  float          = 0.0
    summary:        Optional[str]  = None
    # Frontend data
    geojson:        Optional[Dict[str, Any]] = None
    timeline_chart: List[Dict[str, Any]] = []
    severity_stats: Dict[str, Any] = {}
    impact_summary: List[ImpactSummary] = []
    legend:         List[Dict[str, str]] = []
    map_config:     Optional[Dict[str, Any]] = None
    warnings:       List[str] = []
    errors:         List[str] = []
