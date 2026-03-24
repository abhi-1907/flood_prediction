"""
Data Utilities – Common pandas / numpy helper functions for the flood pipeline.

Includes:
  - Safe type casting and coercion
  - DataFrame inspection and summary
  - Column detection helpers
  - Date parsing utilities
  - Outlier detection and capping
  - Missing value reporting
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from utils.logger import logger


# ── Column name patterns ──────────────────────────────────────────────────────

RAINFALL_COLS    = ["rainfall", "rain", "precipitation", "precip", "rainfall_mm"]
WATERLEVEL_COLS  = ["water_level", "water_level_m", "stage", "gauge", "h_m"]
DISCHARGE_COLS   = ["discharge", "flow", "discharge_m3s", "runoff", "q_m3s"]
ELEVATION_COLS   = ["elevation", "elevation_m", "elev", "dem", "altitude"]
TEMPERATURE_COLS = ["temperature", "temp", "temperature_c", "t_c"]
HUMIDITY_COLS    = ["humidity", "humidity_pct", "rh", "relative_humidity"]
SOIL_COLS        = ["soil_moisture", "soil_moisture_pct", "moisture", "swi"]
DATE_COLS        = ["date", "datetime", "timestamp", "time", "date_time"]


# ── Type coercion ─────────────────────────────────────────────────────────────

def safe_float(value: Any, default: float = 0.0) -> float:
    """Converts a value to float, returning `default` on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """Converts a value to int, returning `default` on failure."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_str(value: Any, default: str = "") -> str:
    """Converts a value to str, returning `default` if None or empty."""
    if value is None:
        return default
    s = str(value).strip()
    return s if s else default


# ── Column detection ──────────────────────────────────────────────────────────

def find_column(df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
    """
    Finds the first column in `df` whose name matches any of the patterns
    (case-insensitive prefix/substring match).

    Args:
        df:       DataFrame to search.
        patterns: List of candidate column name substrings.

    Returns:
        Matching column name or None.
    """
    lower_cols = {c.lower(): c for c in df.columns}
    for pat in patterns:
        pat_lower = pat.lower()
        if pat_lower in lower_cols:
            return lower_cols[pat_lower]
        # Substring match
        for col_lower, col_orig in lower_cols.items():
            if pat_lower in col_lower:
                return col_orig
    return None


def detect_feature_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Auto-detects standard column names in a DataFrame.

    Returns:
        Dict mapping semantic role → actual column name.
        e.g. {"rainfall": "Rain_mm", "water_level": "WL"}
    """
    mapping = {}
    checks = {
        "rainfall":     RAINFALL_COLS,
        "water_level":  WATERLEVEL_COLS,
        "discharge":    DISCHARGE_COLS,
        "elevation":    ELEVATION_COLS,
        "temperature":  TEMPERATURE_COLS,
        "humidity":     HUMIDITY_COLS,
        "soil_moisture": SOIL_COLS,
        "date":         DATE_COLS,
    }
    for role, patterns in checks.items():
        found = find_column(df, patterns)
        if found:
            mapping[role] = found
    return mapping


# ── DataFrame inspection ──────────────────────────────────────────────────────

def df_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Returns a quick summary dict of a DataFrame."""
    if df is None or df.empty:
        return {"rows": 0, "columns": 0, "missing_pct": 0.0}

    missing = df.isnull().sum()
    return {
        "rows":         len(df),
        "columns":      len(df.columns),
        "column_names": list(df.columns),
        "dtypes":       {c: str(t) for c, t in df.dtypes.items()},
        "missing":      {c: int(v) for c, v in missing.items() if v > 0},
        "missing_pct":  round(df.isnull().values.mean() * 100, 1),
        "numeric_cols": list(df.select_dtypes(include=np.number).columns),
    }


def check_required_cols(df: pd.DataFrame, required: List[str]) -> List[str]:
    """Returns a list of required columns that are MISSING from the DataFrame."""
    return [c for c in required if c not in df.columns]


# ── Date parsing ──────────────────────────────────────────────────────────────

def parse_date_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Parses a date/datetime column in-place and sorts the DataFrame by it.
    Handles ISO format, %Y-%m-%d, %d/%m/%Y, epoch timestamps.
    """
    formats = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%dT%H:%M:%S"]
    for fmt in formats:
        try:
            df[col] = pd.to_datetime(df[col], format=fmt, errors="raise")
            df = df.sort_values(col).reset_index(drop=True)
            logger.debug(f"[data_utils] Parsed '{col}' with format {fmt}")
            return df
        except Exception:
            continue

    # Fallback: infer
    try:
        df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
        df = df.sort_values(col).reset_index(drop=True)
    except Exception:
        logger.warning(f"[data_utils] Could not parse date column '{col}'")
    return df


# ── Outlier handling ──────────────────────────────────────────────────────────

def cap_outliers(
    df: pd.DataFrame,
    col: str,
    lower_pct: float = 0.01,
    upper_pct: float = 0.99,
) -> pd.DataFrame:
    """
    Caps outliers in a numeric column at the given percentiles.
    Modifies df in place and returns it.
    """
    if col not in df.columns:
        return df
    low  = df[col].quantile(lower_pct)
    high = df[col].quantile(upper_pct)
    df[col] = df[col].clip(lower=low, upper=high)
    return df


def fill_missing(
    df: pd.DataFrame,
    col: str,
    method: str = "ffill",
    fallback: float = 0.0,
) -> pd.DataFrame:
    """
    Fills missing values in a column.

    Methods: "ffill", "bfill", "mean", "zero", "interpolate"
    """
    if col not in df.columns:
        return df

    if method == "ffill":
        df[col] = df[col].ffill()
    elif method == "bfill":
        df[col] = df[col].bfill()
    elif method == "mean":
        df[col] = df[col].fillna(df[col].mean())
    elif method == "zero":
        df[col] = df[col].fillna(0.0)
    elif method == "interpolate":
        df[col] = df[col].interpolate(method="linear", limit_direction="both")

    # Final fallback
    df[col] = df[col].fillna(fallback)
    return df


# ── Normalisation ─────────────────────────────────────────────────────────────

def min_max_scale(series: pd.Series) -> pd.Series:
    """Scales a Series to [0, 1]. Returns original if constant."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return series * 0   # All zeros for constant series
    return (series - mn) / (mx - mn)


def z_score_scale(series: pd.Series) -> pd.Series:
    """Z-score normalises a Series. Returns zeros if std == 0."""
    std = series.std()
    if std == 0:
        return series * 0
    return (series - series.mean()) / std
