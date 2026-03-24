"""
Data Merger – Intelligently combines multiple FetchResult DataFrames into one
unified dataset suitable for downstream preprocessing and ML modelling.

Merge strategy:
  1. Each source produces a time-indexed (by date) DataFrame.
  2. All DFs are outer-joined on "date" to preserve the full time range.
  3. Location columns (latitude, longitude) are propagated from the primary source.
  4. Duplicate columns (from multiple sources) are resolved by:
       a. Preferring non-null values
       b. Averaging numeric duplicates
  5. A provenance column records which source(s) contributed each row.
  6. A basic quality score (0–100) is computed per row based on completeness.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from agents.data_ingestion.ingestion_schemas import FetchResult, DataCategory
from utils.logger import logger


# ── Priority weights per source (higher = preferred for dedup) ────────────────

SOURCE_PRIORITY: Dict[str, int] = {
    "open_meteo":   4,
    "hydrological": 3,
    "terrain":      2,
    "gov_dataset":  1,
}

# Columns that should NOT be duplicated across merges
UNIQUE_COLUMNS = {"date", "week", "lat", "lon", "latitude", "longitude"}

# Core columns expected in the final dataset — aligned with real training features
# Dataset: india_pakistan_flood_balancednew.csv
CORE_COLUMNS = [
    # Weather features
    "rain_mm_weekly",
    "temp_c_mean",
    "rh_percent_mean",
    "wind_ms_mean",
    "rain_mm_monthly",
    # Hydrological features
    "dam_count_50km",
    "dist_major_river_km",
    "waterbody_nearby",
    # Terrain features
    "lat",
    "lon",
    "elevation_m",
    "slope_degree",
    "terrain_type_encoded",
]


class DataMerger:
    """
    Combines FetchResult DataFrames from multiple sources into one
    unified, deduplicated dataset.

    Usage:
        merger = DataMerger()
        merged_df = merger.merge(fetch_results)
    """

    def merge(
        self,
        fetch_results: List[FetchResult],
        primary_lat:   Optional[float] = None,
        primary_lon:   Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Merges all successful FetchResults into a single DataFrame.

        Args:
            fetch_results: List of FetchResult objects from fetcher agents.
            primary_lat:   Override latitude for location columns.
            primary_lon:   Override longitude for location columns.

        Returns:
            Merged, deduplicated pandas DataFrame. Returns an empty DataFrame
            if no successful results are provided.
        """
        successful = [r for r in fetch_results if r.success and r.data is not None]

        if not successful:
            logger.warning("[DataMerger] No successful fetch results to merge.")
            return pd.DataFrame()

        logger.info(
            f"[DataMerger] Merging {len(successful)} sources: "
            + ", ".join(r.source.value for r in successful)
        )

        # Separate time-series from static (terrain / single-row)
        time_series = [r for r in successful if self._is_time_series(r)]
        static      = [r for r in successful if not self._is_time_series(r)]

        # Build the time-series backbone (outer join on date)
        merged = self._merge_time_series(time_series)

        # Broadcast static data (terrain) across all time rows
        for result in static:
            merged = self._broadcast_static(merged, result)

        # Handle the edge case of no time-series data (static only)
        if merged.empty and static:
            merged = pd.DataFrame(static[0].data if not isinstance(static[0].data, pd.DataFrame)
                                  else static[0].data)

        # Fill lat/lon if not present
        if primary_lat is not None and "latitude" not in merged.columns:
            merged["latitude"]  = primary_lat
        if primary_lon is not None and "longitude" not in merged.columns:
            merged["longitude"] = primary_lon

        # Sort chronologically
        if "date" in merged.columns:
            merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
            merged = merged.sort_values("date").reset_index(drop=True)

        # Compute quality score per row
        merged["data_quality_score"] = merged.apply(
            lambda row: self._quality_score(row, CORE_COLUMNS), axis=1
        )

        # Add provenance
        merged["data_sources"] = ", ".join(r.source.value for r in successful)

        logger.info(
            f"[DataMerger] Final dataset: {len(merged)} rows × {len(merged.columns)} cols | "
            f"avg quality={merged['data_quality_score'].mean():.1f}/100"
        )

        return merged

    # ── Time-series merge ─────────────────────────────────────────────────

    def _merge_time_series(self, results: List[FetchResult]) -> pd.DataFrame:
        """Outer-joins all time-series DataFrames on the 'date' column."""
        if not results:
            return pd.DataFrame()

        # Sort by source priority
        results_sorted = sorted(
            results,
            key=lambda r: SOURCE_PRIORITY.get(r.source.value, 0),
            reverse=True,
        )

        merged = self._to_df(results_sorted[0])
        for result in results_sorted[1:]:
            right = self._to_df(result)
            if right.empty:
                continue
            if "date" in merged.columns and "date" in right.columns:
                # Identify overlapping non-key columns
                overlap = [
                    c for c in right.columns
                    if c in merged.columns and c not in UNIQUE_COLUMNS
                ]
                suffix = f"_{result.source.value}"
                merged = pd.merge(merged, right, on="date", how="outer",
                                  suffixes=("", suffix))
                # Resolve duplicates by preferring non-null, then averaging
                for col in overlap:
                    dup_col = f"{col}{suffix}"
                    if dup_col in merged.columns:
                        merged[col] = merged.apply(
                            lambda r, c=col, d=dup_col: self._coalesce(r[c], r[d]),
                            axis=1,
                        )
                        merged.drop(columns=[dup_col], inplace=True)
            else:
                merged = pd.concat([merged, right], ignore_index=True)

        return merged

    # ── Static broadcast ──────────────────────────────────────────────────

    def _broadcast_static(
        self,
        time_df: pd.DataFrame,
        static_result: FetchResult,
    ) -> pd.DataFrame:
        """
        Broadcasts a single-row (static) DataFrame across all time rows.
        Used to attach terrain attributes to every time-point.
        """
        static_df = self._to_df(static_result)
        if static_df.empty or time_df.empty:
            return time_df

        static_row = static_df.iloc[0]
        for col in static_df.columns:
            if col not in UNIQUE_COLUMNS and col not in time_df.columns:
                time_df[col] = static_row.get(col, None)

        return time_df

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _to_df(result: FetchResult) -> pd.DataFrame:
        """Safely converts a FetchResult.data to a DataFrame."""
        if isinstance(result.data, pd.DataFrame):
            return result.data.copy()
        if isinstance(result.data, (list, dict)):
            try:
                return pd.DataFrame(result.data)
            except Exception:
                return pd.DataFrame()
        return pd.DataFrame()

    @staticmethod
    def _is_time_series(result: FetchResult) -> bool:
        """Returns True if the result contains multiple time-indexed rows."""
        if result.row_count > 1:
            return True
        df = result.data
        if isinstance(df, pd.DataFrame):
            # Check for both 'date' (live API fetchers) and 'week' (dataset column)
            has_temporal = "date" in df.columns or "week" in df.columns
            return has_temporal and len(df) > 1
        return False

    @staticmethod
    def _coalesce(a: Any, b: Any) -> Any:
        """Returns the first non-null value; if both numeric, returns average."""
        a_null = pd.isna(a) if not isinstance(a, (list, dict)) else False
        b_null = pd.isna(b) if not isinstance(b, (list, dict)) else False

        if a_null and b_null:
            return np.nan
        if a_null:
            return b
        if b_null:
            return a
        # Both present — average if numeric
        try:
            return (float(a) + float(b)) / 2
        except (TypeError, ValueError):
            return a   # Keep primary source value for non-numeric

    @staticmethod
    def _quality_score(row: pd.Series, expected_cols: List[str]) -> float:
        """
        Computes a 0–100 completeness score for a single DataFrame row,
        based on how many of the expected core columns are non-null.
        """
        present = [c for c in expected_cols if c in row.index and pd.notna(row[c])]
        return round(len(present) / len(expected_cols) * 100, 1)
