"""
Simulation Routes – Endpoints for flood simulation and what-if scenarios.

POST /simulation/           – Run a what-if simulation on session data
POST /simulation/scenario   – Run a custom scenario (standalone, no session)
GET  /simulation/{session_id} – Retrieve simulation results
GET  /simulation/{session_id}/geojson – Get GeoJSON for map rendering
GET  /simulation/{session_id}/timeline – Get timeline chart data
GET  /simulation/{session_id}/impact – Get impact assessment
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agents.simulation.simulation_agent import SimulationAgent
from agents.orchestration.memory import agent_memory
from services.gemini_service import get_gemini_service
from utils.logger import logger

router = APIRouter()


# ── Request schemas ───────────────────────────────────────────────────────────

class SimulationRequest(BaseModel):
    session_id: str


class ScenarioRequest(BaseModel):
    query: str                               # "What if 200mm rain falls in Kochi in 3 days?"
    location: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    rainfall_mm: Optional[float] = None
    rainfall_days: int = 1
    water_level_m: Optional[float] = None
    soil_moisture_pct: Optional[float] = None
    return_period_years: Optional[int] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/", summary="Run simulation on session data")
async def run_simulation(request: SimulationRequest):
    """
    Runs the full simulation pipeline on an existing session.
    Uses prediction data and preprocessed datasets for realistic parameters.

    The pipeline:
      1. Builds a scenario from user query + prediction data
      2. Simulates hourly flood timeline (SCS-CN model)
      3. Generates N×N inundation grid
      4. Converts to GeoJSON for map rendering
      5. Estimates multi-dimensional impact
      6. Produces frontend-ready rendering payload
    """
    session = agent_memory.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{request.session_id}' not found.")

    try:
        agent  = SimulationAgent(get_gemini_service())
        result = await agent.run(session)

        return {
            "session_id":    session.session_id,
            "status":        result.status,
            "scenario":      result.scenario.model_dump() if result.scenario else None,
            "peak_depth_m":  result.peak_depth_m,
            "peak_hour":     result.peak_hour,
            "total_area_km2": result.total_area_km2,
            "inundated_pct": result.inundated_pct,
            "impact_count":  len(result.impact),
            "summary":       result.summary,
            "warnings":      result.warnings,
            "errors":        result.errors,
        }
    except Exception as exc:
        logger.error(f"[/simulation] Error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/scenario", summary="Run a custom what-if scenario")
async def run_custom_scenario(request: ScenarioRequest):
    """
    Standalone what-if simulation without requiring a prior session.
    Provide a natural-language query or explicit parameters.

    Examples:
      - "What if 300mm rain falls in 2 days in Kochi?"
      - "Simulate a 100-year flood for Patna"
      - "What happens if the dam releases 500 m³/s?"
    """
    try:
        session = agent_memory.create_session(
            user_query=request.query,
            initial_context={
                "location":     request.location,
                "latitude":     request.latitude,
                "longitude":    request.longitude,
                "original_query": request.query,
            },
        )

        # Store scenario parameters in context for the engine
        if request.rainfall_mm:
            session.context["rainfall_mm"] = request.rainfall_mm
        if request.water_level_m:
            session.context["water_level_m"] = request.water_level_m
        if request.soil_moisture_pct:
            session.context["soil_moisture_pct"] = request.soil_moisture_pct

        agent  = SimulationAgent(get_gemini_service())
        result = await agent.run(session)

        # Return full map data for rendering
        map_data = session.get_artifact("simulation_map_data") or {}

        return {
            "session_id":     session.session_id,
            "status":         result.status,
            "scenario":       result.scenario.model_dump() if result.scenario else None,
            "peak_depth_m":   result.peak_depth_m,
            "peak_hour":      result.peak_hour,
            "total_area_km2": result.total_area_km2,
            "inundated_pct":  result.inundated_pct,
            "summary":        result.summary,
            "timeline_chart": map_data.get("timeline_chart", []),
            "severity_stats": map_data.get("severity_stats", {}),
            "impact_summary": map_data.get("impact_summary", []),
            "legend":         map_data.get("legend", []),
            "warnings":       result.warnings,
        }
    except Exception as exc:
        logger.error(f"[/simulation/scenario] Error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/{session_id}", summary="Get simulation results")
async def get_simulation(session_id: str):
    """Returns the stored simulation summary for a session."""
    session = agent_memory.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    scenario = session.get_artifact("simulation_scenario")
    if not scenario:
        return {"session_id": session_id, "status": "pending", "message": "Simulation not run yet."}

    map_data = session.get_artifact("simulation_map_data") or {}
    return {
        "session_id":     session_id,
        "status":         "complete",
        "scenario":       scenario,
        "peak_stats":     map_data.get("peak_stats", {}),
        "severity_stats": map_data.get("severity_stats", {}),
        "scenario_info":  map_data.get("scenario_info", {}),
    }


@router.get("/{session_id}/geojson", summary="Get GeoJSON for map rendering")
async def get_geojson(session_id: str):
    """Returns the GeoJSON FeatureCollection for Leaflet/Mapbox map rendering."""
    session = agent_memory.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    geojson = session.get_artifact("simulation_geojson")
    if not geojson:
        raise HTTPException(status_code=404, detail="GeoJSON not available. Run simulation first.")

    return geojson


@router.get("/{session_id}/timeline", summary="Get timeline chart data")
async def get_timeline(session_id: str):
    """Returns hourly timeline data in Recharts-compatible format."""
    session = agent_memory.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    map_data = session.get_artifact("simulation_map_data") or {}
    timeline = map_data.get("timeline_chart", [])

    if not timeline:
        return {"session_id": session_id, "status": "pending"}

    return {
        "session_id": session_id,
        "timeline":   timeline,
        "peak_stats": map_data.get("peak_stats", {}),
    }


@router.get("/{session_id}/impact", summary="Get impact assessment")
async def get_impact(session_id: str):
    """Returns the multi-dimensional flood impact assessment."""
    session = agent_memory.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    impacts = session.get_artifact("simulation_impacts")
    if not impacts:
        return {"session_id": session_id, "status": "pending"}

    return {
        "session_id": session_id,
        "impacts":    impacts,
    }
