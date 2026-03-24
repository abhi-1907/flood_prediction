"""
Alert Routes – Endpoints for alert management, subscriptions, and triggering.

POST /alerts/subscribe      – Subscribe to flood alerts for a location
DELETE /alerts/unsubscribe/{subscriber_id} – Unsubscribe
POST /alerts/trigger        – Manually trigger an alert
GET  /alerts/subscribers    – List all subscribers
GET  /alerts/active         – List currently active alerts
GET  /alerts/{session_id}   – Get alert details for a session
POST /alerts/import         – Bulk import subscribers
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agents.alerting.alerting_agent import AlertingAgent
from agents.orchestration.memory import agent_memory
from models.alert_models import AlertSubscribeRequest, AlertTriggerRequest
from services.gemini_service import get_gemini_service
from utils.logger import logger

router = APIRouter()

# Singleton alerting agent (persists subscriber state across requests)
_alerting_agent: Optional[AlertingAgent] = None


def _get_alerting_agent() -> AlertingAgent:
    global _alerting_agent
    if _alerting_agent is None:
        _alerting_agent = AlertingAgent(get_gemini_service(), dry_run=False)
    return _alerting_agent


# ── Additional schemas ────────────────────────────────────────────────────────

class AlertSessionRequest(BaseModel):
    session_id: str


class BulkImportRequest(BaseModel):
    subscribers: List[dict]


# ── Subscription endpoints ────────────────────────────────────────────────────

@router.post("/subscribe", summary="Subscribe to flood alerts")
async def subscribe(request: AlertSubscribeRequest):
    """
    Registers a new subscriber for flood alerts in the given location.
    Alerts will be sent via the chosen channels (email, SMS, push)
    when the flood risk meets or exceeds the subscriber's threshold.
    """
    agent = _get_alerting_agent()
    try:
        sub_id = agent.subscriber_manager.add(
            name=request.name,
            location=request.location,
            latitude=request.latitude,
            longitude=request.longitude,
            email=request.email,
            phone=request.phone,
            push_token=request.push_token,
            channels=[ch.value for ch in request.channels],
            radius_km=request.radius_km,
            min_risk_level=request.min_risk_level,
        )
        return {
            "status":        "subscribed",
            "subscriber_id": sub_id,
            "location":      request.location,
            "channels":      [ch.value for ch in request.channels],
            "radius_km":     request.radius_km,
            "min_risk_level": request.min_risk_level,
        }
    except Exception as exc:
        logger.error(f"[/alerts/subscribe] Error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/unsubscribe/{subscriber_id}", summary="Unsubscribe from alerts")
async def unsubscribe(subscriber_id: str):
    """Deactivates a subscriber (soft delete)."""
    agent   = _get_alerting_agent()
    removed = agent.subscriber_manager.remove(subscriber_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Subscriber '{subscriber_id}' not found.")
    return {"status": "unsubscribed", "subscriber_id": subscriber_id}


@router.get("/subscribers", summary="List all subscribers")
async def list_subscribers():
    """Returns all active subscribers with their channel configurations."""
    agent = _get_alerting_agent()
    subs  = agent.subscriber_manager.list_all()
    return {
        "total": len(subs),
        "subscribers": [s.model_dump() for s in subs],
    }


@router.post("/import", summary="Bulk import subscribers")
async def bulk_import(request: BulkImportRequest):
    """
    Imports subscribers in bulk from a JSON array.
    Each item should have: name, location, latitude, longitude,
    and optionally: email, phone, push_token, channels, radius_km, min_risk_level.
    """
    agent = _get_alerting_agent()
    count = agent.subscriber_manager.import_subscribers(request.subscribers)
    return {
        "status":   "imported",
        "imported":  count,
        "total_now": agent.subscriber_manager.count(),
    }


# ── Alert trigger endpoints ──────────────────────────────────────────────────

@router.post("/trigger", summary="Manually trigger a flood alert")
async def trigger_alert(request: AlertTriggerRequest):
    """
    Manually triggers a flood alert for a location.
    Bypasses the scheduler cooldown — useful for emergency overrides
    and testing.
    """
    agent = _get_alerting_agent()
    try:
        result = await agent.trigger_manual_alert(
            location=request.location,
            risk_level=request.risk_level,
            message=request.message,
        )
        return {
            "status":            result.status,
            "alert_id":          result.alert.alert_id if result.alert else None,
            "severity":          result.alert.severity.value if result.alert else None,
            "total_subscribers": result.total_subscribers,
            "sent_count":        result.sent_count,
            "failed_count":      result.failed_count,
            "channels_used":     result.channels_used,
            "summary":           result.summary,
        }
    except Exception as exc:
        logger.error(f"[/alerts/trigger] Error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/session", summary="Run alerting agent for a session")
async def alert_for_session(request: AlertSessionRequest):
    """
    Runs the full alerting pipeline for an existing session.
    Evaluates whether an alert is necessary based on the prediction results,
    then delivers via all matching channels.
    """
    session = agent_memory.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{request.session_id}' not found.")

    agent = _get_alerting_agent()
    try:
        result = await agent.run(session)
        return {
            "session_id":        session.session_id,
            "status":            result.status,
            "alert_id":          result.alert.alert_id if result.alert else None,
            "severity":          result.alert.severity.value if result.alert else None,
            "total_subscribers": result.total_subscribers,
            "sent_count":        result.sent_count,
            "failed_count":      result.failed_count,
            "channels_used":     result.channels_used,
            "escalation":        result.escalation.value if result.escalation else None,
            "summary":           result.summary,
            "warnings":          result.warnings,
        }
    except Exception as exc:
        logger.error(f"[/alerts/session] Error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/active", summary="List active alerts")
async def list_active_alerts():
    """Returns all currently active (non-expired) alerts."""
    agent  = _get_alerting_agent()
    active = agent.scheduler.get_active_alerts()
    return {
        "active_count": len(active),
        "alerts": [
            {
                "alert_id":    a.alert_id,
                "severity":    a.severity.value,
                "risk_level":  a.risk_level,
                "location":    a.location,
                "created_at":  a.created_at,
                "expires_at":  a.expires_at,
                "deliveries":  len(a.deliveries),
            }
            for a in active
        ],
    }


@router.get("/{session_id}", summary="Get alert details for a session")
async def get_alert(session_id: str):
    """Returns the alert details for a given session."""
    session = agent_memory.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    alert_data = session.get_artifact("alert")
    if not alert_data:
        return {"session_id": session_id, "status": "pending", "message": "No alert generated yet."}

    return {
        "session_id": session_id,
        "status":     "complete",
        **alert_data,
    }
