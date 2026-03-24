"""
Recommendation Routes – Endpoints for flood recommendations.

POST /recommendations/          – Generate recommendations for a session
GET  /recommendations/{session_id}  – Retrieve recommendations
GET  /recommendations/{session_id}/safety-message – Get SMS-friendly safety message
GET  /recommendations/{session_id}/authority-brief – Get formal authority brief
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agents.recommendation.recommendation_agent import RecommendationAgent
from agents.orchestration.memory import agent_memory
from services.gemini_service import get_gemini_service
from utils.logger import logger

router = APIRouter()


# ── Request schemas ───────────────────────────────────────────────────────────

class RecommendationRequest(BaseModel):
    session_id: Optional[str] = None
    location: Optional[str] = None
    risk_level: str = "high"
    user_type: str = "general_public"
    has_elderly: bool = False
    has_children: bool = False
    has_disability: bool = False
    vehicle_access: bool = True


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("", summary="Generate flood recommendations")
async def generate_recommendations(request: RecommendationRequest):
    """
    Generates personalised flood recommendations for the given session.
    Requires prediction results to be available in the session.

    The pipeline:
      1. Profiles the user (type, vulnerability, location)
      2. Enriches location context (terrain, infrastructure)
      3. Generates LLM + rule-based recommendations
      4. Produces SMS safety messages and authority briefs
    """
    if request.session_id:
        session = agent_memory.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session '{request.session_id}' not found.")
        # Update session with the provided form overrides
        session.context.update(request.model_dump(exclude={"session_id"}))
    else:
        if not request.location:
             raise HTTPException(status_code=400, detail="Location required if session_id is omitted.")
        
        session = agent_memory.create_session(
            user_query=f"Recommendations for {request.location}",
            initial_context=request.model_dump(exclude={"session_id"}),
        )
        # Fast-track recommendation by mocking the prerequisite prediction
        session.store_artifact("ensemble_prediction", {
            "risk_level": request.risk_level.upper(),
            "flood_probability": 0.85,
        })

    if not session.get_artifact("ensemble_prediction"):
        raise HTTPException(
            status_code=400,
            detail="No prediction results found. Run prediction agent first.",
        )

    try:
        agent  = RecommendationAgent(get_gemini_service())
        result = await agent.run(session)

        return {
            "session_id":     session.session_id,
            "status":         result.status,
            "risk_level":     result.risk_level,
            "urgency":        result.urgency.value if result.urgency else None,
            "recommendations": [r.model_dump() for r in result.recommendations],
            "resource_plan":  [r.model_dump() for r in result.resource_plan],
            "summary":        result.summary,
            "safety_message": result.safety_message,
            "warnings":       result.warnings,
            "errors":         result.errors,
        }
    except Exception as exc:
        logger.error(f"[/recommendations] Error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/{session_id}", summary="Retrieve recommendations")
async def get_recommendations(session_id: str):
    """Returns stored recommendations for a session."""
    session = agent_memory.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    recs = session.get_artifact("recommendations")
    if not recs:
        return {"session_id": session_id, "status": "pending", "message": "Recommendations not generated yet."}

    return {
        "session_id":      session_id,
        "status":          "complete",
        "recommendations": recs,
        "resource_plan":   session.get_artifact("resource_plan") or [],
        "summary":         session.get_artifact("recommendation_summary"),
    }


@router.get("/{session_id}/safety-message", summary="Get SMS safety message")
async def get_safety_message(session_id: str):
    """Returns a short SMS-friendly safety message for the session."""
    session = agent_memory.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    msg = session.get_artifact("safety_message")
    if not msg:
        return {"session_id": session_id, "status": "pending"}

    return {"session_id": session_id, "safety_message": msg}


@router.get("/{session_id}/authority-brief", summary="Get formal authority brief")
async def get_authority_brief(session_id: str):
    """Returns the formal disaster management brief for authorities."""
    session = agent_memory.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    brief = session.get_artifact("authority_brief")
    if not brief:
        return {"session_id": session_id, "status": "pending", "message": "Authority brief not available."}

    return {"session_id": session_id, "authority_brief": brief}
