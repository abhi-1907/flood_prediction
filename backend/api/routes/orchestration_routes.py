"""
Orchestration Routes – HTTP API endpoints that front the OrchestratorAgent.

POST /orchestrate          – Run the full agentic pipeline (JSON or multipart form)
POST /orchestrate/stream   – Run with Server-Sent Events (SSE) real-time progress
GET  /orchestrate/sessions – List active/archived sessions (admin)
GET  /orchestrate/sessions/{session_id} – Get session detail
"""

from __future__ import annotations

import json
from typing import Any, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agents.orchestration.orchestrator import orchestrator
from agents.orchestration.memory import agent_memory

router = APIRouter()


# ── Request / Response schemas ────────────────────────────────────────────────

class OrchestrationRequest(BaseModel):
    input_mode: str = "nlp"             # "nlp" | "manual"
    query: Optional[str] = None
    user_type: str = "general_public"   # general_public | authority | first_responder
    context: Optional[dict] = None


class OrchestrationResponse(BaseModel):
    session_id: str
    prediction: Optional[dict] = None
    recommendations: Optional[Any] = None
    simulation: Optional[Any] = None
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
        # ── Mutually exclusive input validation ───────────────────────────────
        query = request.query
        if request.input_mode == "nlp":
            if not query or not query.strip():
                raise ValueError("A query is required when input_mode is 'nlp'.")
            if request.context:
                raise ValueError("Context parameters are not allowed in 'nlp' mode.")
            final_query = query
            final_context = None

        elif request.input_mode == "manual":
            if not request.context:
                raise ValueError("Context parameters are required when input_mode is 'manual'.")
            final_query = query or "Analyze flood risk based on provided parameters."
            final_context = request.context

        else:
            raise ValueError(f"Invalid input_mode: '{request.input_mode}'")

        result = await orchestrator.run(
            user_query=final_query,
            user_type=request.user_type,
            initial_context=final_context,
        )
        return OrchestrationResponse(**{
            k: result.get(k) for k in OrchestrationResponse.model_fields
        })
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post(
    "/upload",
    response_model=OrchestrationResponse,
    summary="Run pipeline with a CSV/JSON file upload",
)
async def orchestrate_with_file(
    input_mode: str      = Form("file"),
    query:      Optional[str] = Form(None),
    user_type:  str      = Form("general_public"),
    file:       UploadFile = File(...),
):
    """
    Multipart form endpoint — accepts a data file alongside a query.
    The file is parsed for column names and passed to the ingestion agent.
    """
    try:
        if input_mode != "file":
             raise ValueError("Upload endpoint requires input_mode='file'.")
             
        content = await file.read()
        # Quick column extraction for context (header line only)
        try:
            first_line = content.decode("utf-8", errors="ignore").split("\n")[0]
            columns = [c.strip() for c in first_line.split(",")]
        except Exception:
            columns = []

        final_query = query or f"Analyze data from uploaded file: {file.filename}"

        result = await orchestrator.run(
            user_query=final_query,
            uploaded_file_bytes=content,
            uploaded_columns=columns,
            user_type=user_type,
        )
        return OrchestrationResponse(**{
            k: result.get(k) for k in OrchestrationResponse.model_fields
        })
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
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
        try:
            # ── Mutually exclusive input validation ───────────────────────────────
            query = request.query
            if request.input_mode == "nlp":
                if not query or not query.strip():
                    raise ValueError("A query is required when input_mode is 'nlp'.")
                if request.context:
                    raise ValueError("Context parameters are not allowed in 'nlp' mode.")
                final_query = query
                final_context = None

            elif request.input_mode == "manual":
                if not request.context:
                    raise ValueError("Context parameters are required when input_mode is 'manual'.")
                final_query = query or "Analyze flood risk based on provided parameters."
                final_context = request.context

            else:
                raise ValueError(f"Invalid input_mode: '{request.input_mode}'")
        except ValueError as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
            yield "data: [DONE]\n\n"
            return
            
        session_id = None
        async for raw in orchestrator.stream(
            user_query=final_query,
            user_type=request.user_type,
            initial_context=final_context,
        ):
            event_type = raw.get("event", "")
            data       = raw.get("data", {})

            if event_type == "plan_ready":
                # Surface session_id from the plan data if available
                if isinstance(data, dict):
                    session_id = data.get("session_id")
                evt = {
                    "type":       "plan_ready",
                    "agent":      None,
                    "status":     "ready",
                    "message":    f"Plan created with {data.get('steps', '?')} steps",
                    "session_id": session_id,
                    "data":       data,
                }

            elif event_type == "step_started":
                agent = data.get("agent", "unknown")
                evt = {
                    "type":    "agent_start",
                    "agent":   agent,
                    "status":  "active",
                    "message": f"Starting {agent}…",
                    "data":    data,
                }

            elif event_type == "step_done":
                agent      = data.get("agent", "unknown")
                step_status = data.get("status", "completed")
                # Map internal StepStatus strings to frontend-friendly names
                if step_status in ("succeeded", "completed", "SUCCEEDED", "COMPLETED"):
                    fe_status = "done"
                elif step_status in ("skipped", "SKIPPED"):
                    fe_status = "skipped"
                elif step_status in ("failed", "FAILED"):
                    fe_status = "error"
                else:
                    fe_status = "done"
                evt = {
                    "type":    "agent_complete",
                    "agent":   agent,
                    "status":  fe_status,
                    "message": f"{agent} {fe_status}",
                    "data":    data.get("result") or data,
                }

            elif event_type == "complete":
                # Whole pipeline finished — include the full result payload
                if isinstance(data, dict):
                    session_id = data.get("session_id", session_id)
                evt = {
                    "type":            "complete",
                    "agent":           None,
                    "status":          "done",
                    "message":         "Pipeline complete",
                    "elapsed_seconds": data.get("elapsed_seconds") if isinstance(data, dict) else None,
                    "session_id":      session_id,
                    "data":            data,
                }

            else:
                # Pass-through any other event types unchanged
                evt = {"type": event_type, **data}

            yield f"data: {json.dumps(evt)}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
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
