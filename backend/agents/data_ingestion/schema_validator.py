"""
Schema Validator – Detects and validates the structure of incoming data.

Handles multiple input formats:
  - CSV / Excel files (bytes or path)
  - JSON payloads (dict or list)
  - Plain-text descriptions / user queries

The validator produces a SchemaReport that tells the SourceIdentifier exactly
what data is present, what field quality looks like, and whether the dataset
has timestamp/location columns.
"""

from __future__ import annotations

import io
import json
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np

from agents.data_ingestion.ingestion_schemas import (
    DataCategory,
    DataSourceType,
    FieldReport,
    FieldStatus,
    SchemaReport,
)
from utils.logger import logger


# ── Column name heuristics ───────────────────────────────────────────────────
# Keywords match partial column names to categorise data.
# Real dataset columns: rain_mm_weekly, rain_mm_monthly, temp_c_mean,
#   rh_percent_mean, wind_ms_mean, dam_count_50km, dist_major_river_km,
#   waterbody_nearby, lat, lon, elevation_m, slope_degree, terrain_type_encoded

RAINFALL_KEYWORDS  = {"rainfall", "rain", "precipitation", "precip", "mm", "rf",
                      "weekly", "monthly"}   # rain_mm_weekly, rain_mm_monthly
HYDRO_KEYWORDS     = {"water_level", "discharge", "flow", "river", "stream",
                      "gauge", "runoff", "soil_moisture", "groundwater",
                      "dam_count", "dist_major", "waterbody", "dam"}  # real cols
TERRAIN_KEYWORDS   = {"elevation", "dem", "altitude", "slope", "aspect",
                      "land_use", "ndvi", "terrain", "topo",
                      "slope_degree", "terrain_type"}  # real cols
TIMESTAMP_KEYWORDS = {"date", "time", "datetime", "timestamp", "year",
                      "month", "day", "hour",
                      "week"}   # real dataset temporal column (YYYY-MM)
LOCATION_KEYWORDS  = {"lat", "lon", "latitude", "longitude", "location",
                      "district", "state", "place", "city", "village"}


# mapping aliases to standard system names
COLUMN_MAPPING = {
    "latitude":           ["lat", "latitude", "y", "coords_y"],
    "longitude":          ["lon", "long", "longitude", "x", "coords_x"],
    "rain_mm_weekly":     ["rain", "rainfall", "precip", "precipitation", "rain_mm", "weekly_rain", "weekly_precip"],
    "rain_mm_monthly":    ["monthly_rain", "monthly_precip", "rain_monthly", "precip_monthly"],
    "elevation_m":        ["elevation", "elev", "altitude", "alt", "dem", "height"],
    "slope_degree":       ["slope", "slope_deg", "gradient"],
    "dist_major_river_km":["river_dist", "dist_river", "distance_river", "river_km"],
    "waterbody_nearby":   ["waterbody", "water_body", "nearby_water", "lake_nearby", "river_nearby"],
    "terrain_type":       ["terrain", "land_use", "landcover", "land_type"],
}

class SchemaValidator:
    """
    Validates and profiles the schema of any incoming data source.

    Usage:
        validator = SchemaValidator()
        report = validator.validate_csv(file_bytes)
        report = validator.validate_json(json_data)
        report = validator.validate_text("Rainfall data for Kochi 2023")
    """

    # ── Public API ────────────────────────────────────────────────────────

    def validate_csv(
        self,
        file_bytes: bytes,
        filename: str = "upload.csv",
        context: Optional[Dict[str, Any]] = None,
    ) -> SchemaReport:
        """Parses a CSV file and produces a SchemaReport."""
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
            df = self._rename_columns(df)
            return self._analyse_dataframe(df, DataSourceType.USER_CSV, context)
        except Exception as exc:
            logger.error(f"[SchemaValidator] CSV parse error: {exc}")
            return self._empty_report(notes=f"CSV parse error: {exc}")

    def validate_json(self, data: Union[dict, list], context: Optional[Dict[str, Any]] = None) -> SchemaReport:
        """Analyses a JSON dict or list-of-dicts and produces a SchemaReport."""
        try:
            if isinstance(data, dict):
                df = pd.json_normalize(data)
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                return self._empty_report(notes="Unsupported JSON structure.")
            df = self._rename_columns(df)
            return self._analyse_dataframe(df, DataSourceType.USER_JSON, context)
        except Exception as exc:
            logger.error(f"[SchemaValidator] JSON parse error: {exc}")
            return self._empty_report(notes=f"JSON parse error: {exc}")

    def validate_dataframe(self, df: pd.DataFrame, context: Optional[Dict[str, Any]] = None) -> SchemaReport:
        """Analyses an already-loaded DataFrame with optional context."""
        df = self._rename_columns(df)
        return self._analyse_dataframe(df, DataSourceType.USER_CSV, context)

    def validate_text(self, text: str) -> SchemaReport:
        """
        Produces a minimal SchemaReport for a plain-text query.
        The actual data will be fetched by the fetchers.
        """
        text_lower = text.lower()
        has_location = any(kw in text_lower for kw in LOCATION_KEYWORDS)
        return SchemaReport(
            detected_category=DataCategory.UNKNOWN,
            detected_sources=[DataSourceType.USER_TEXT],
            fields=[],
            missing_categories=[
                DataCategory.RAINFALL,
                DataCategory.HYDRO,
                DataCategory.TERRAIN,
            ],
            row_count=0,
            column_count=0,
            has_timestamp=False,
            has_location=has_location,
            notes="Plain text query — all data must be fetched from external sources.",
        )

    # ── Core analysis ─────────────────────────────────────────────────────

    def map_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Public helper to get a renamed DataFrame."""
        return self._rename_columns(df)

    def _analyse_dataframe(
        self,
        df: pd.DataFrame,
        source: DataSourceType,
        context: Optional[Dict[str, Any]] = None,
    ) -> SchemaReport:
        """Profiles a DataFrame and returns a SchemaReport."""
        fields        = self._analyse_fields(df)
        category      = self._detect_category(df)
        missing       = self._detect_missing_categories(df, category, context)
        has_timestamp = self._has_timestamp(df)
        has_location  = self._has_location(df) or self._has_context_location(context)

        logger.info(
            f"[SchemaValidator] Analysed {len(df)} rows × {len(df.columns)} cols | "
            f"category={category} | missing={missing}"
        )

        return SchemaReport(
            detected_category=category,
            detected_sources=[source],
            fields=fields,
            missing_categories=missing,
            row_count=len(df),
            column_count=len(df.columns),
            has_timestamp=has_timestamp,
            has_location=has_location,
        )

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Maps common aliases in the DataFrame to standard system names."""
        rename_map = {}
        cols = [c.lower().strip() for c in df.columns]
        
        for standard, aliases in COLUMN_MAPPING.items():
            for alias in aliases:
                if alias in cols:
                    # Find original casing
                    orig_idx = cols.index(alias)
                    orig_name = df.columns[orig_idx]
                    rename_map[orig_name] = standard
                    break
        
        if rename_map:
            logger.info(f"[SchemaValidator] Renaming columns: {rename_map}")
            return df.rename(columns=rename_map)
        return df

    def _has_context_location(self, context: Optional[Dict[str, Any]]) -> bool:
        if not context: return False
        return bool(context.get("latitude") and context.get("longitude"))

    def _analyse_fields(self, df: pd.DataFrame) -> List[FieldReport]:
        """Produces a FieldReport for every column in the DataFrame."""
        reports = []
        for col in df.columns:
            null_pct = float(df[col].isna().mean() * 100)
            dtype    = str(df[col].dtype)

            # Determine status
            if null_pct == 100:
                status = FieldStatus.MISSING
                note   = "All values are null"
            elif null_pct > 50:
                status = FieldStatus.INVALID
                note   = f"{null_pct:.1f}% null — potentially unusable"
            elif not self._is_valid_dtype(df[col]):
                status = FieldStatus.INVALID
                note   = f"Unexpected dtype or unparseable values"
            else:
                status = FieldStatus.PRESENT
                note   = None

            reports.append(
                FieldReport(
                    name=col,
                    status=status,
                    dtype=dtype,
                    null_pct=round(null_pct, 2),
                    note=note,
                )
            )
        return reports

    # ── Category detection ────────────────────────────────────────────────

    def _detect_category(self, df: pd.DataFrame) -> DataCategory:
        cols = [c.lower().replace(" ", "_") for c in df.columns]
        
        has_rainfall = any(any(kw in c for kw in RAINFALL_KEYWORDS) for c in cols)
        has_hydro    = any(any(kw in c for kw in HYDRO_KEYWORDS) for c in cols)
        has_terrain  = any(any(kw in c for kw in TERRAIN_KEYWORDS) for c in cols)

        count = sum([has_rainfall, has_hydro, has_terrain])
        if count >= 2:
            return DataCategory.MIXED
        if has_rainfall:
            if self._has_timestamp(df):
                return DataCategory.TIMESERIES
            return DataCategory.RAINFALL
        if has_hydro:
            return DataCategory.HYDRO
        if has_terrain:
            return DataCategory.TERRAIN
        return DataCategory.UNKNOWN

    def _detect_missing_categories(
        self,
        df: pd.DataFrame,
        present: DataCategory,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[DataCategory]:
        """Returns which data categories are absent from BOTH DataFrame and Context."""
        cols = [c.lower().replace(" ", "_") for c in df.columns]
        ctx = context or {}
        missing = []
        
        # RAINFALL check
        has_rain_df = any(any(kw in c for kw in RAINFALL_KEYWORDS) for c in cols)
        has_rain_ctx = ctx.get("rain_mm_weekly") is not None or ctx.get("rain_mm_monthly") is not None
        if not (has_rain_df or has_rain_ctx):
            missing.append(DataCategory.RAINFALL)
            
        # HYDRO check
        has_hydro_df = any(any(kw in c for kw in HYDRO_KEYWORDS) for c in cols)
        has_hydro_ctx = ctx.get("dist_major_river_km") is not None or ctx.get("waterbody_nearby") is not None
        if not (has_hydro_df or has_hydro_ctx):
            missing.append(DataCategory.HYDRO)
            
        # TERRAIN check
        has_terrain_df = any(any(kw in c for kw in TERRAIN_KEYWORDS) for c in cols)
        has_terrain_ctx = any(ctx.get(k) is not None for k in ["elevation_m", "slope_degree", "terrain_type"])
        if not (has_terrain_df or has_terrain_ctx):
            missing.append(DataCategory.TERRAIN)
            
        return missing

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _has_timestamp(df: pd.DataFrame) -> bool:
        return any(
            kw in col.lower()
            for col in df.columns
            for kw in TIMESTAMP_KEYWORDS
        )

    @staticmethod
    def _has_location(df: pd.DataFrame) -> bool:
        return any(
            kw in col.lower()
            for col in df.columns
            for kw in LOCATION_KEYWORDS
        )

    @staticmethod
    def _is_valid_dtype(series: pd.Series) -> bool:
        """Returns False if the series is all strings when numeric is expected."""
        if series.dtype == object:
            # Try converting to numeric — if more than 50% fail, flag as invalid
            numeric = pd.to_numeric(series, errors="coerce")
            return numeric.notna().mean() > 0.5
        return True

    @staticmethod
    def _empty_report(notes: str = "") -> SchemaReport:
        return SchemaReport(
            detected_category=DataCategory.UNKNOWN,
            detected_sources=[DataSourceType.UNKNOWN],
            fields=[],
            missing_categories=[
                DataCategory.RAINFALL,
                DataCategory.HYDRO,
                DataCategory.TERRAIN,
            ],
            notes=notes,
        )
