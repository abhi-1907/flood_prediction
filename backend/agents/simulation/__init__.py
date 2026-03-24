"""
Simulation Agent package — public API.
"""

from agents.simulation.simulation_agent import SimulationAgent
from agents.simulation.simulation_schemas import (
    FloodSeverity,
    FloodZone,
    GeoJSONFeature,
    ImpactCategory,
    ImpactEstimate,
    ScenarioParameters,
    ScenarioType,
    SimulationResult,
    SimulationTimeStep,
)
from agents.simulation.scenario_engine import ScenarioEngine
from agents.simulation.flood_zone_mapper import FloodZoneMapper
from agents.simulation.geojson_builder import GeoJSONBuilder
from agents.simulation.impact_assessor import ImpactAssessor
from agents.simulation.map_renderer import MapRenderer

__all__ = [
    "SimulationAgent",
    # Schemas
    "FloodSeverity",
    "FloodZone",
    "GeoJSONFeature",
    "ImpactCategory",
    "ImpactEstimate",
    "ScenarioParameters",
    "ScenarioType",
    "SimulationResult",
    "SimulationTimeStep",
    # Core classes
    "ScenarioEngine",
    "FloodZoneMapper",
    "GeoJSONBuilder",
    "ImpactAssessor",
    "MapRenderer",
]
