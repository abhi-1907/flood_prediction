"""
Pydantic response schemas for the Data Ingestion API.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class IngestionRequest(BaseModel):
    """Payload for submitting data to the ingestion agent."""
    query:     Optional[str] = None       # Free-text or place name query
    data_type: Optional[str] = None       # "csv", "json", "text"
    location:  Optional[str] = None
    latitude:  Optional[float] = None
    longitude: Optional[float] = None


class IngestionResponse(BaseModel):
    """Response from the ingestion pipeline."""
    session_id:          str
    status:              str              # "success" | "partial" | "failed"
    rows_ingested:       int   = 0
    sources_used:        List[str] = []
    identified_columns:  List[str] = []
    missing_fields:      List[str] = []
    data_preview:        Optional[Dict[str, Any]] = None
    warnings:            List[str] = []
    errors:              List[str] = []


class DataSourceInfo(BaseModel):
    """Information about a supported data source."""
    name:        str
    type:        str                      # "api" | "government" | "file_upload"
    status:      str                      # "available" | "unconfigured" | "error"
    description: str
    url:         Optional[str] = None
