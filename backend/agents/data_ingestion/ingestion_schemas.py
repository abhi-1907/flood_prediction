"""
Pydantic schemas shared across the Data Ingestion pipeline.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Enumerations ──────────────────────────────────────────────────────────────

class DataSourceType(str, Enum):
    USER_CSV       = "user_csv"
    USER_JSON      = "user_json"
    USER_TEXT      = "user_text"
    OPEN_METEO     = "open_meteo"
    TERRAIN        = "terrain"
    HYDROLOGICAL   = "hydrological"
    GOV_DATASET    = "gov_dataset"
    UNKNOWN        = "unknown"


class DataCategory(str, Enum):
    RAINFALL    = "rainfall"
    HYDRO       = "hydro"
    TERRAIN     = "terrain"
    TIMESERIES  = "timeseries"
    MIXED       = "mixed"
    UNKNOWN     = "unknown"


class FieldStatus(str, Enum):
    PRESENT  = "present"
    MISSING  = "missing"
    INVALID  = "invalid"


# ── Field-level schemas ───────────────────────────────────────────────────────

class FieldReport(BaseModel):
    """Assessment of a single data field / column."""
    name:        str
    status:      FieldStatus
    dtype:       Optional[str]  = None
    null_pct:    float          = 0.0      # Percentage of null values
    note:        Optional[str]  = None     # LLM reasoning note


class SchemaReport(BaseModel):
    """
    Complete schema analysis report for an input dataset.
    Produced by SchemaValidator and consumed by SourceIdentifier.
    """
    detected_category:   DataCategory
    detected_sources:    List[DataSourceType] = Field(default_factory=list)
    fields:              List[FieldReport]    = Field(default_factory=list)
    missing_categories:  List[DataCategory]  = Field(default_factory=list)
    row_count:           int = 0
    column_count:        int = 0
    has_timestamp:       bool = False
    has_location:        bool = False
    notes:               Optional[str] = None


# ── Fetch result ──────────────────────────────────────────────────────────────

class FetchResult(BaseModel):
    """Standardised return value from any fetcher."""
    source:        DataSourceType
    category:      DataCategory
    success:       bool
    data:          Optional[Any]  = None    # pandas-serialisable or raw dict
    columns:       List[str]      = Field(default_factory=list)
    row_count:     int            = 0
    error:         Optional[str]  = None
    metadata:      Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


# ── Ingestion result ──────────────────────────────────────────────────────────

class IngestionResult(BaseModel):
    """Final output from the DataIngestionAgent returned to the Orchestrator."""
    session_id:         str
    status:             str                              # "success" | "partial" | "failed"
    schema_report:      Optional[SchemaReport] = None
    sources_used:       List[DataSourceType]   = Field(default_factory=list)
    missing_categories: List[DataCategory]     = Field(default_factory=list)
    rows_collected:     int = 0
    columns:            List[str] = Field(default_factory=list)
    warnings:           List[str] = Field(default_factory=list)
    errors:             List[str] = Field(default_factory=list)
    # The actual merged dataset is stored in session.artifacts["raw_dataset"]

    class Config:
        arbitrary_types_allowed = True
