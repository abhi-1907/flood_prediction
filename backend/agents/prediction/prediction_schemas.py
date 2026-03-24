"""
Prediction Agent schemas — Pydantic models and enumerations shared
across the entire Prediction Agent pipeline.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


# ── Enumerations ──────────────────────────────────────────────────────────────

class ModelType(str, Enum):
    XGBOOST       = "xgboost"
    RANDOM_FOREST = "random_forest"
    LSTM          = "lstm"
    ENSEMBLE      = "ensemble"


class PredictionMode(str, Enum):
    CLASSIFICATION = "classification"   # Binary: flood / no-flood
    REGRESSION     = "regression"       # Continuous: water level (m)
    MULTI_CLASS    = "multi_class"      # LOW / MEDIUM / HIGH risk


class RiskLevel(str, Enum):
    LOW      = "LOW"
    MEDIUM   = "MEDIUM"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"


class ModelStatus(str, Enum):
    LOADED    = "loaded"
    NOT_FOUND = "not_found"
    FAILED    = "failed"
    UNTRAINED = "untrained"


# ── Per-model result ──────────────────────────────────────────────────────────

class ModelPrediction(BaseModel):
    """Output from a single model."""
    model_type:      ModelType
    mode:            PredictionMode
    raw_score:       float                      # Probability / regression value
    label:           Optional[str]   = None     # "flood" / "no_flood" / risk level
    confidence:      float           = 0.0      # 0–1 confidence in this prediction
    feature_importance: Dict[str, float] = Field(default_factory=dict)
    status:          ModelStatus     = ModelStatus.LOADED
    error:           Optional[str]   = None


# ── Ensemble result ───────────────────────────────────────────────────────────

class EnsemblePrediction(BaseModel):
    """Combined output from the ensemble of all available models."""
    flood_probability:  float                    # 0–1 weighted average
    risk_level:         RiskLevel
    risk_score:         float                    # 0–100 normalised risk score
    model_predictions:  List[ModelPrediction] = Field(default_factory=list)
    models_used:        List[ModelType]        = Field(default_factory=list)
    confidence:         float                   = 0.0
    uncertainty_band:   Tuple[float, float]    = (0.0, 1.0)   # 95% CI


# ── Explanation ───────────────────────────────────────────────────────────────

class FeatureContribution(BaseModel):
    """SHAP or permutation-based contribution of one feature."""
    feature:      str
    value:        float    # Actual feature value in the input row
    contribution: float    # Impact on the prediction (positive = increases flood risk)
    rank:         int      # 1 = most important


class PredictionExplanation(BaseModel):
    """LLM-generated + SHAP-based explanation of the prediction."""
    summary:              str                                  # Plain-language summary
    key_factors:          List[str]                           # Top 3–5 contributing factors
    feature_contributions: List[FeatureContribution] = Field(default_factory=list)
    llm_narrative:        Optional[str] = None               # Gemini natural-language narrative
    confidence_note:      Optional[str] = None


# ── Prediction result ─────────────────────────────────────────────────────────

class PredictionResult(BaseModel):
    """Final output returned by PredictionAgent to the Orchestrator."""
    session_id:          str
    status:              str                              # "success" | "partial" | "failed"
    ensemble:            Optional[EnsemblePrediction] = None
    explanation:         Optional[PredictionExplanation] = None
    location:            Optional[str]   = None
    latitude:            Optional[float] = None
    longitude:           Optional[float] = None
    prediction_mode:     PredictionMode  = PredictionMode.CLASSIFICATION
    forecast_horizon_days: int           = 1
    warnings:            List[str]       = Field(default_factory=list)
    errors:              List[str]       = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True


# ── Model loading metadata ────────────────────────────────────────────────────

class LoadedModel(BaseModel):
    """Registry entry for a loaded model."""
    model_type:  ModelType
    status:      ModelStatus
    path:        Optional[str]  = None
    features:    List[str]      = Field(default_factory=list)
    mode:        PredictionMode = PredictionMode.CLASSIFICATION
    version:     str            = "1.0"
    error:       Optional[str]  = None

    class Config:
        arbitrary_types_allowed = True
