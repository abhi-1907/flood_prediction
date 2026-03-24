"""
Simulation Agent schemas — Pydantic models shared across the simulation pipeline.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


# ── Enumerations ──────────────────────────────────────────────────────────────

class ScenarioType(str, Enum):
    WHAT_IF       = "what_if"            # User-defined parameter overrides
    HISTORICAL    = "historical"         # Replay a past flood event
    EXTREME       = "extreme"            # Worst-case / return-period scenarios
    DAM_BREAK     = "dam_break"          # Upstream dam failure scenario
    MULTI_DAY     = "multi_day"          # Progressive multi-day rainfall


class FloodSeverity(str, Enum):
    MINOR    = "minor"       # < 0.3m inundation
    MODERATE = "moderate"    # 0.3–1.0m
    SEVERE   = "severe"      # 1.0–3.0m
    EXTREME  = "extreme"     # > 3.0m


class ImpactCategory(str, Enum):
    POPULATION   = "population"
    AGRICULTURE  = "agriculture"
    INFRASTRUCTURE = "infrastructure"
    ECONOMIC     = "economic"
    ENVIRONMENTAL = "environmental"


# ── Scenario definition ──────────────────────────────────────────────────────

class ScenarioParameters(BaseModel):
    """User-defined or LLM-generated scenario to simulate."""
    scenario_type:      ScenarioType = ScenarioType.WHAT_IF
    name:               str          = "Custom Scenario"
    description:        Optional[str] = None

    # Overrideable parameters
    rainfall_mm:        Optional[float] = None    # Daily rainfall override
    rainfall_days:      int             = 1       # Duration of heavy rainfall
    water_level_m:      Optional[float] = None    # Starting water level
    discharge_m3s:      Optional[float] = None    # River discharge override
    soil_moisture_pct:  Optional[float] = None    # Soil saturation level
    dam_release_m3s:    Optional[float] = None    # Dam release rate (dam_break)
    return_period_years: Optional[int]  = None    # e.g. 50-year, 100-year flood
    elevation_m:        Optional[float] = None    # Area elevation override

    # Location
    latitude:           Optional[float] = None
    longitude:          Optional[float] = None
    location:           Optional[str]   = None


# ── Flood zone (one cell/area in the inundation map) ─────────────────────────

class FloodZone(BaseModel):
    """Represents a single cell/area with inundation depth."""
    latitude:       float
    longitude:      float
    depth_m:        float            # Estimated inundation depth (metres)
    severity:       FloodSeverity
    elevation_m:    Optional[float] = None
    is_populated:   bool            = False
    land_use:       Optional[str]   = None   # "residential", "agricultural", etc.


# ── Impact assessment ─────────────────────────────────────────────────────────

class ImpactEstimate(BaseModel):
    """One dimension of flood impact."""
    category:       ImpactCategory
    metric:         str                        # e.g. "people_affected"
    value:          float                      # Numeric estimate
    unit:           str            = ""        # e.g. "persons", "hectares", "INR crore"
    confidence:     float          = 0.5       # 0–1
    description:    str            = ""


# ── Time-step in multi-day simulation ─────────────────────────────────────────

class SimulationTimeStep(BaseModel):
    """One hourly/daily step in a multi-day simulation."""
    hour:           int              # 0-indexed from simulation start
    rainfall_mm:    float
    water_level_m:  float
    discharge_m3s:  float
    inundation_pct: float            # % of area inundated
    flood_severity: FloodSeverity
    risk_score:     float            # 0–100


# ── GeoJSON feature for map rendering ─────────────────────────────────────────

class GeoJSONFeature(BaseModel):
    """One polygon/point in the GeoJSON output."""
    type:       str = "Feature"
    geometry:   Dict[str, Any] = Field(default_factory=dict)
    properties: Dict[str, Any] = Field(default_factory=dict)


# ── Full simulation result ────────────────────────────────────────────────────

class SimulationResult(BaseModel):
    """Final output returned by SimulationAgent to the Orchestrator."""
    session_id:       str
    status:           str                              # "success" | "partial" | "failed"
    scenario:         Optional[ScenarioParameters] = None
    flood_zones:      List[FloodZone]    = Field(default_factory=list)
    timeline:         List[SimulationTimeStep] = Field(default_factory=list)
    impact:           List[ImpactEstimate] = Field(default_factory=list)
    peak_depth_m:     float              = 0.0
    peak_hour:        int                = 0
    total_area_km2:   float              = 0.0
    inundated_pct:    float              = 0.0
    geojson:          Optional[Dict[str, Any]] = None
    summary:          Optional[str]      = None        # LLM narrative
    warnings:         List[str]          = Field(default_factory=list)
    errors:           List[str]          = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True
