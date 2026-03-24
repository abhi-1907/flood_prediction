"""
Data Ingestion Routes – Endpoints for data upload, text parsing, and source management.

POST /ingest/upload   – Upload CSV/JSON/Excel file for ingestion
POST /ingest/text     – Parse raw text or user query for data extraction
POST /ingest/url      – Fetch data from an external URL
GET  /ingest/sources  – List available data sources
GET  /ingest/status/{session_id} – Check ingestion status for a session
"""

from __future__ import annotations

import io
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from agents.data_ingestion.ingestion_agent import DataIngestionAgent
from agents.orchestration.memory import Session, agent_memory
from services.gemini_service import get_gemini_service
from utils.logger import logger

router = APIRouter()


# ── Request / Response schemas ────────────────────────────────────────────────

class TextIngestionRequest(BaseModel):
    text: str
    location: Optional[str] = None
    user_type: str = "general_public"


class UrlIngestionRequest(BaseModel):
    url: str
    location: Optional[str] = None
    data_type: Optional[str] = None   # "csv", "json", "api"


class IngestionStatusResponse(BaseModel):
    session_id: str
    status: str
    sources: List[str] = []
    columns: List[str] = []
    row_count: int = 0
    warnings: List[str] = []


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/upload", summary="Upload a data file for ingestion")
async def upload_file(
    file:     UploadFile = File(...),
    location: str        = Form(""),
    query:    str        = Form(""),
):
    """
    Accepts CSV, JSON, or Excel files. The ingestion agent will:
      - Auto-detect the file format
      - Identify relevant columns (rainfall, water level, etc.)
      - Validate and clean the data
      - Store in the session for downstream agents
    """
    try:
        content = await file.read()
        filename = file.filename or "upload.csv"

        # Create a session
        session = agent_memory.create_session(
            user_query=query or f"Uploaded: {filename}",
            context={"location": location, "uploaded_file": filename},
        )

        agent = DataIngestionAgent(get_gemini_service())
        session.context["raw_data_bytes"] = content
        session.context["raw_data_filename"] = filename
        session.context["original_query"] = query or f"Analyze {filename}"

        result = await agent.run(session)

        return {
            "session_id": session.session_id,
            "status":     result.status,
            "sources":    result.sources_used,
            "columns":    result.identified_columns,
            "row_count":  result.row_count,
            "warnings":   result.warnings,
            "errors":     result.errors,
        }
    except Exception as exc:
        logger.error(f"[/ingest/upload] Error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/text", summary="Parse raw text or query for data extraction")
async def ingest_text(request: TextIngestionRequest):
    """
    Accepts a free-text description or user query. The LLM-powered
    ingestion agent will extract structured data or identify what
    external data sources to fetch from.
    """
    try:
        session = agent_memory.create_session(
            user_query=request.text,
            context={
                "location": request.location,
                "user_type": request.user_type,
                "original_query": request.text,
            },
        )

        agent = DataIngestionAgent(get_gemini_service())
        result = await agent.run(session)

        return {
            "session_id": session.session_id,
            "status":     result.status,
            "sources":    result.sources_used,
            "columns":    result.identified_columns,
            "row_count":  result.row_count,
            "data_preview": result.data_preview,
            "warnings":   result.warnings,
        }
    except Exception as exc:
        logger.error(f"[/ingest/text] Error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/url", summary="Fetch data from an external URL")
async def ingest_from_url(request: UrlIngestionRequest):
    """
    Fetches data from an external URL (e.g., government open data portal,
    weather API, CWC flood bulletin). The ingestion agent handles parsing.
    """
    try:
        session = agent_memory.create_session(
            user_query=f"Fetch data from: {request.url}",
            context={
                "location": request.location,
                "data_url": request.url,
                "data_type": request.data_type,
                "original_query": f"Fetch data from {request.url}",
            },
        )

        agent = DataIngestionAgent(get_gemini_service())
        result = await agent.run(session)

        return {
            "session_id": session.session_id,
            "status":     result.status,
            "sources":    result.sources_used,
            "row_count":  result.row_count,
            "warnings":   result.warnings,
        }
    except Exception as exc:
        logger.error(f"[/ingest/url] Error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/sources", summary="List available data sources")
async def list_sources():
    """Returns the list of supported external data sources and their status."""
    return {
        "sources": [
            {
                "name": "Open-Meteo Weather API",
                "type": "api",
                "status": "available",
                "description": "Historical and forecast weather data (rainfall, temperature, humidity)",
            },
            {
                "name": "India Meteorological Department (IMD)",
                "type": "government",
                "status": "available",
                "description": "Official rainfall and weather warnings for Indian states",
            },
            {
                "name": "Central Water Commission (CWC)",
                "type": "government",
                "status": "available",
                "description": "River water level, discharge, and flood bulletins",
            },

            {
                "name": "CSV / Excel Upload",
                "type": "file_upload",
                "status": "available",
                "description": "Upload your own dataset (rainfall, water level, etc.)",
            },
        ]
    }


@router.get("/status/{session_id}", summary="Check ingestion status")
async def ingestion_status(session_id: str):
    """Returns the current ingestion status for a given session."""
    session = agent_memory.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    ingestion = session.get_artifact("ingestion_result")
    if not ingestion:
        return {"session_id": session_id, "status": "pending", "message": "Ingestion not started yet."}

    return {
        "session_id": session_id,
        "status":     ingestion.get("status", "unknown"),
        "sources":    ingestion.get("sources_used", []),
        "columns":    ingestion.get("identified_columns", []),
        "row_count":  ingestion.get("row_count", 0),
    }


# Import settings at the module level for sources endpoint
from config import settings
