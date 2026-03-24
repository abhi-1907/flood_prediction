"""
Orchestration Routes – HTTP API endpoints that front the OrchestratorAgent.

POST /orchestrate          – Run the full agentic pipeline (JSON or multipart form)
POST /orchestrate/stream   – Run with Server-Sent Events (SSE) real-time progress
GET  /orchestrate/sessions – List active/archived sessions (admin)
GET  /orchestrate/sessions/{session_id} – Get session detail
"""

from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agents.orchestration.orchestrator import orchestrator
from agents.orchestration.memory import agent_memory

router = APIRouter()


# ── Request / Response schemas ────────────────────────────────────────────────

class OrchestrationRequest(BaseModel):
    query: str
    user_type: str = "general_public"   # general_public | authority | first_responder
    context: Optional[dict] = None


class OrchestrationResponse(BaseModel):
    session_id: str
    prediction: Optional[dict] = None
    recommendations: Optional[str] = None
    simulation: Optional[dict] = None
    alert_status: Optional[dict] = None
    steps_summary: list = []
    elapsed_seconds: float = 0.0


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/", response_model=OrchestrationResponse, summary="Run full agentic pipeline")
async def orchestrate(request: OrchestrationRequest):
    """
    Accepts a free-text query (and optional context) and runs the full
    multi-agent pipeline: ingestion → preprocessing → prediction →
    recommendation → simulation → alerting (as applicable).
    """
    try:
        result = await orchestrator.run(
            user_query=request.query,
            user_type=request.user_type,
            initial_context=request.context,
        )
        return OrchestrationResponse(**{
            k: result.get(k) for k in OrchestrationResponse.model_fields
        })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post(
    "/upload",
    response_model=OrchestrationResponse,
    summary="Run pipeline with a CSV/JSON file upload",
)
async def orchestrate_with_file(
    query:     str      = Form(...),
    user_type: str      = Form("general_public"),
    file:      UploadFile = File(...),
):
    """
    Multipart form endpoint — accepts a data file alongside a query.
    The file is parsed for column names and passed to the ingestion agent.
    """
    try:
        content = await file.read()
        # Quick column extraction for context (header line only)
        try:
            first_line = content.decode("utf-8", errors="ignore").split("\n")[0]
            columns = [c.strip() for c in first_line.split(",")]
        except Exception:
            columns = []

        result = await orchestrator.run(
            user_query=query,
            uploaded_file_bytes=content,
            uploaded_columns=columns,
            user_type=user_type,
        )
        return OrchestrationResponse(**{
            k: result.get(k) for k in OrchestrationResponse.model_fields
        })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/stream", summary="Run pipeline with SSE real-time progress")
async def orchestrate_stream(request: OrchestrationRequest):
    """
    Server-Sent Events (SSE) endpoint.
    Yields JSON event objects as each agent step completes.

    Frontend should use EventSource or fetch() with ReadableStream.
    """
    async def event_generator():
        async for event in orchestrator.stream(
            user_query=request.query,
            user_type=request.user_type,
        ):
            yield f"data: {json.dumps(event)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # Disable Nginx buffering for SSE
        },
    )


# ── Session management (admin) ────────────────────────────────────────────────

@router.get("/sessions", summary="List all orchestration sessions (admin)")
async def list_sessions():
    """Returns a summary of all active and recently archived sessions."""
    return {
        "stats":    agent_memory.stats(),
        "active":   agent_memory.list_active(),
        "archived": agent_memory.list_archived(),
    }


@router.get("/sessions/{session_id}", summary="Get session details")
async def get_session(session_id: str):
    """Returns the full detail of a specific session including all agent steps."""
    session = agent_memory.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return session.to_dict()
