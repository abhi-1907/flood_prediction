"""
Preprocessing schemas — Pydantic models and enumerations shared
across the entire Preprocessing Agent pipeline.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


# ── Enumerations ──────────────────────────────────────────────────────────────

class ImputationStrategy(str, Enum):
    MEAN          = "mean"
    MEDIAN        = "median"
    FORWARD_FILL  = "forward_fill"
    BACKWARD_FILL = "backward_fill"
    INTERPOLATE   = "interpolate"       # Linear interpolation
    KNN           = "knn"               # K-Nearest Neighbours imputation
    SEASONAL      = "seasonal"          # Season-aware filling (for time-series)
    ZERO          = "zero"
    DROP_ROW      = "drop_row"


class OutlierStrategy(str, Enum):
    CLIP          = "clip"              # IQR or Z-score winsorization
    REMOVE        = "remove"            # Drop the row
    REPLACE_MEAN  = "replace_mean"      # Replace with column mean
    REPLACE_MEDIAN= "replace_median"    # Replace with column median
    ISOLATIONFOREST = "isolation_forest"
    NONE          = "none"              # Do not touch outliers


class ScalingStrategy(str, Enum):
    STANDARD      = "standard"          # Zero mean, unit variance
    MINMAX        = "minmax"            # [0, 1] scaling
    ROBUST        = "robust"            # Median / IQR-based (outlier-resistant)
    LOG           = "log"               # log1p transform
    NONE          = "none"


class DatasetCharacter(str, Enum):
    TIME_SERIES   = "time_series"
    TABULAR       = "tabular"
    MIXED         = "mixed"


# ── Per-column strategy decision ──────────────────────────────────────────────

class ColumnStrategy(BaseModel):
    """LLM-selected strategy for one column."""
    column:             str
    imputation:         ImputationStrategy = ImputationStrategy.MEDIAN
    outlier_strategy:   OutlierStrategy    = OutlierStrategy.CLIP
    scaling:            ScalingStrategy    = ScalingStrategy.STANDARD
    drop:               bool               = False
    note:               Optional[str]      = None


# ── Preprocessing plan ────────────────────────────────────────────────────────

class PreprocessingStrategy(BaseModel):
    """Complete preprocessing plan produced by the StrategySelector."""
    dataset_character:      DatasetCharacter
    global_imputation:      ImputationStrategy   = ImputationStrategy.MEDIAN
    global_outlier:        OutlierStrategy      = OutlierStrategy.CLIP
    global_scaling:        ScalingStrategy      = ScalingStrategy.STANDARD
    column_strategies:     List[ColumnStrategy] = Field(default_factory=list)
    engineer_features:     bool                 = True
    format_time_series:    bool                 = False
    sequence_length:       int                  = 7          # LSTM look-back window
    columns_to_drop:       List[str]            = Field(default_factory=list)
    columns_to_keep:       List[str]            = Field(default_factory=list)
    rationale:             Optional[str]        = None


# ── Audit record ──────────────────────────────────────────────────────────────

class AuditEntry(BaseModel):
    """Records one processing decision for full traceability."""
    step:        str                        # e.g. "missing_value_handler"
    action:      str                        # e.g. "imputed 12 nulls with median"
    column:      Optional[str]  = None
    before_stat: Optional[Any]  = None      # Value/stat before transformation
    after_stat:  Optional[Any]  = None      # Value/stat after transformation
    rows_affected: int          = 0
    strategy:    Optional[str]  = None


class PreprocessingAudit(BaseModel):
    """Full audit trail for one preprocessing run."""
    session_id:  str
    entries:     List[AuditEntry] = Field(default_factory=list)
    input_shape:  Tuple[int, int] = (0, 0)
    output_shape: Tuple[int, int] = (0, 0)
    rows_dropped: int             = 0
    cols_dropped: int             = 0


# ── Preprocessing result ──────────────────────────────────────────────────────

class PreprocessingResult(BaseModel):
    """Returned by PreprocessingAgent to the Orchestrator."""
    session_id:     str
    status:         str                         # "success" | "partial" | "failed"
    strategy:       Optional[PreprocessingStrategy] = None
    audit:          Optional[PreprocessingAudit]    = None
    input_rows:     int = 0
    input_cols:     int = 0
    output_rows:    int = 0
    output_cols:    int = 0
    warnings:       List[str] = Field(default_factory=list)
    errors:         List[str] = Field(default_factory=list)
    feature_columns: List[str] = Field(default_factory=list)
    # The processed DataFrame is stored in session.artifacts["processed_dataset"]

    class Config:
        arbitrary_types_allowed = True
