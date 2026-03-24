"""
Prediction Routes – Endpoints for flood prediction operations.

POST /predict/           – Run prediction on a session's processed data
POST /predict/quick      – Quick single-query prediction (no file upload)
GET  /predict/{session_id} – Retrieve prediction results for a session
GET  /predict/{session_id}/explanation – Get the human-readable explanation
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agents.prediction.prediction_agent import PredictionAgent
from agents.orchestration.memory import agent_memory
from services.gemini_service import get_gemini_service
from utils.logger import logger

router = APIRouter()


# ── Request / Response schemas ────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    session_id: str


class QuickPredictionRequest(BaseModel):
    location: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    rain_mm_weekly: Optional[float] = None
    dist_major_river_km: Optional[float] = None
    elevation_m: Optional[float] = None


class PredictionResponse(BaseModel):
    session_id: str
    status: str
    risk_level: Optional[str] = None
    flood_probability: Optional[float] = None
    confidence: Optional[float] = None
    models_used: list = []
    explanation: Optional[str] = None
    warnings: list = []
    errors: list = []


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/", response_model=PredictionResponse, summary="Run prediction on session data")
async def run_prediction(request: PredictionRequest):
    """
    Runs the full prediction pipeline on an existing session that has
    already gone through ingestion and preprocessing.

    The pipeline:
      1. Loads ML models (XGBoost, Random Forest, LSTM)
      2. Selects optimal model strategy via LLM
      3. Runs parallel inference
      4. Combines via ensemble with weighted averaging
      5. Generates a human-readable explanation
    """
    session = agent_memory.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{request.session_id}' not found.")

    if not session.get_artifact("processed_dataset"):
        raise HTTPException(
            status_code=400,
            detail="No processed data found. Run ingestion and preprocessing first.",
        )

    try:
        agent  = PredictionAgent(get_gemini_service())
        result = await agent.run(session)

        ensemble = session.get_artifact("ensemble_prediction") or {}
        explanation = session.get_artifact("prediction_explanation") or {}

        return PredictionResponse(
            session_id=session.session_id,
            status=result.status,
            risk_level=ensemble.get("risk_level"),
            flood_probability=ensemble.get("flood_probability"),
            confidence=ensemble.get("confidence"),
            models_used=ensemble.get("models_used", []),
            explanation=explanation.get("summary"),
            warnings=result.warnings,
            errors=result.errors,
        )
    except Exception as exc:
        logger.error(f"[/predict] Error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/quick", response_model=PredictionResponse, summary="Quick single-query prediction")
async def quick_prediction(request: QuickPredictionRequest):
    """
    A lightweight prediction endpoint that creates a temporary session
    with the provided parameters. Suitable for quick checks like:
    "What's the flood risk for Kochi with 150mm rainfall?"
    """
    try:
        import pandas as pd
        import numpy as np

        session = agent_memory.create_session(
            user_query=f"Quick prediction for {request.location}",
            initial_context={
                "location":   request.location,
                "latitude":   request.latitude,
                "longitude":  request.longitude,
                "original_query": f"Predict flood risk for {request.location}",
            },
        )

        # Build a minimal DataFrame providing all 13 core features for the new models
        rain_weekly = request.rain_mm_weekly or 50.0
        data = {
            "rain_mm_weekly":      [rain_weekly],
            "temp_c_mean":         [28.0],
            "rh_percent_mean":     [80.0],
            "wind_ms_mean":        [5.0],
            "rain_mm_monthly":     [rain_weekly * 4.3],
            "dam_count_50km":      [0.0],
            "dist_major_river_km": [request.dist_major_river_km or 5.0],
            "waterbody_nearby":    [1.0],
            "lat":                 [request.latitude or 20.0],
            "lon":                 [request.longitude or 80.0],
            "elevation_m":         [request.elevation_m or 30.0],
            "slope_degree":        [2.0],
            "terrain_type_encoded":[0.0],
        }
        df = pd.DataFrame(data)
        session.store_artifact("processed_dataset", df)

        agent  = PredictionAgent(get_gemini_service())
        result = await agent.run(session)

        ensemble    = session.get_artifact("ensemble_prediction") or {}
        explanation = session.get_artifact("prediction_explanation") or {}

        return PredictionResponse(
            session_id=session.session_id,
            status=result.status,
            risk_level=ensemble.get("risk_level"),
            flood_probability=ensemble.get("flood_probability"),
            confidence=ensemble.get("confidence"),
            models_used=ensemble.get("models_used", []),
            explanation=explanation.get("summary"),
            warnings=result.warnings,
            errors=result.errors,
        )
    except Exception as exc:
        logger.error(f"[/predict/quick] Error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/{session_id}", summary="Get prediction results")
async def get_prediction(session_id: str):
    """Retrieves the prediction results for a given session."""
    session = agent_memory.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    ensemble = session.get_artifact("ensemble_prediction")
    if not ensemble:
        return {"session_id": session_id, "status": "pending", "message": "Prediction not run yet."}

    return {
        "session_id":        session_id,
        "status":            "complete",
        "risk_level":        ensemble.get("risk_level"),
        "flood_probability": ensemble.get("flood_probability"),
        "confidence":        ensemble.get("confidence"),
        "confidence_interval": ensemble.get("confidence_interval"),
    }


@router.get("/{session_id}/explanation", summary="Get prediction explanation")
async def get_explanation(session_id: str):
    """Returns the human-readable explanation of the prediction."""
    session = agent_memory.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    explanation = session.get_artifact("prediction_explanation")
    if not explanation:
        return {"session_id": session_id, "status": "pending"}

    return {
        "session_id":   session_id,
        "summary":      explanation.get("summary"),
        "key_factors":  explanation.get("key_factors", []),
        "contributions": explanation.get("feature_contributions", {}),
        "confidence_note": explanation.get("confidence_note"),
    }
