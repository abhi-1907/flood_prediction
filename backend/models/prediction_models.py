"""
Pydantic response schemas for the Prediction API.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class PredictionRequest(BaseModel):
    """Input payload for triggering a flood prediction."""
    session_id: Optional[str]  = None
    location:   Optional[str]  = None
    latitude:   Optional[float] = None
    longitude:  Optional[float] = None
    query:      Optional[str]  = None   # Free-text e.g. "flood risk in Kochi"
    rainfall_mm:       Optional[float] = None
    water_level_m:     Optional[float] = None
    soil_moisture_pct: Optional[float] = None


class PredictionResult(BaseModel):
    """Structured prediction output returned to the client."""
    session_id:        str
    status:            str                    # "success" | "partial" | "failed"
    location:          Optional[str]  = None
    risk_level:        Optional[str]  = None  # LOW | MEDIUM | HIGH | CRITICAL
    flood_probability: Optional[float] = None # 0.0 – 1.0
    confidence:        Optional[float] = None
    confidence_interval: Optional[List[float]] = None   # [lower, upper]
    models_used:       List[str] = []
    explanation:       Optional[str]  = None
    key_factors:       List[str] = []
    feature_contributions: Dict[str, float] = {}
    timestamp:         Optional[str]  = None
    warnings:          List[str] = []
    errors:            List[str] = []


class QuickPredictionRequest(BaseModel):
    """Lightweight single-datapoint prediction (no session required)."""
    location:          str
    latitude:          Optional[float] = None
    longitude:         Optional[float] = None
    rainfall_mm:       Optional[float] = None
    water_level_m:     Optional[float] = None
    soil_moisture_pct: Optional[float] = None


class ExplanationResponse(BaseModel):
    """Human-readable explanation of a prediction."""
    session_id:           str
    summary:              Optional[str] = None
    key_factors:          List[str] = []
    contributions:        Dict[str, float] = {}
    confidence_note:      Optional[str] = None
    model_selection_reason: Optional[str] = None
