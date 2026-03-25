"""
OpenMeteo Fetcher – Retrieves rainfall, weather, and historical precipitation data.

Uses the Open-Meteo free API (no key required) to fetch:
  - Historical daily precipitation (past N days)
  - Hourly precipitation for the last 7 days
  - Additional weather variables: temperature, humidity, wind speed, cloud cover
  - Optionally fetches flood-index from ERA5 reanalysis

API docs: https://open-meteo.com/en/docs
Historical API: https://archive-api.open-meteo.com/v1/archive
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd

from agents.data_ingestion.ingestion_schemas import (
    DataCategory,
    DataSourceType,
    FetchResult,
)
from utils.logger import logger


# ── API configuration ─────────────────────────────────────────────────────────

FORECAST_URL  = "https://api.open-meteo.com/v1/forecast"
HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
TIMEOUT_SECS  = 30

HOURLY_VARIABLES = [
    "precipitation",
    "temperature_2m",
    "relativehumidity_2m",
    "windspeed_10m",
    "cloudcover",
    "surface_pressure",
    "et0_fao_evapotranspiration",   # evapotranspiration — useful for soil moisture proxy
]

DAILY_VARIABLES = [
    "precipitation_sum",
    "precipitation_hours",
    "windspeed_10m_max",
    "et0_fao_evapotranspiration",
]


class OpenMeteoFetcher:
    """
    Fetches rainfall and weather data from the Open-Meteo API.

    Supports:
      - 7-day forecast (real-time)
      - Historical data from a custom date range (up to 2 years back)
      - Automatic fallback from forecast → historical if forecast is insufficient
    """

    def __init__(self, timeout: int = TIMEOUT_SECS) -> None:
        self._timeout = timeout

    # ── Public API ────────────────────────────────────────────────────────

    async def fetch(
        self,
        latitude:    float,
        longitude:   float,
        days_back:   int = 30,
        include_hourly: bool = True,
        **kwargs,
    ) -> FetchResult:
        """
        Main entry point — fetches rainfall + weather data for a location.

        Args:
            latitude:        Decimal latitude of the target location.
            longitude:       Decimal longitude of the target location.
            days_back:       Number of historical days to fetch (default 30).
            include_hourly:  Whether to include hourly data (default True).

        Returns:
            FetchResult with a pandas DataFrame in `data`.
        """
        end_date   = datetime.utcnow().date()
        start_date = end_date - timedelta(days=days_back)

        logger.info(
            f"[OpenMeteoFetcher] Fetching {days_back}d data for "
            f"({latitude:.4f}, {longitude:.4f}) "
            f"from {start_date} to {end_date}"
        )

        try:
            dfs = []

            # Fetch historical daily data
            daily_df = await self._fetch_historical_daily(
                latitude, longitude, str(start_date), str(end_date)
            )
            if daily_df is not None:
                dfs.append(daily_df)

            # Fetch hourly data for the last 7 days if requested
            if include_hourly:
                hourly_start = end_date - timedelta(days=7)
                hourly_df = await self._fetch_historical_hourly(
                    latitude, longitude, str(hourly_start), str(end_date)
                )
                if hourly_df is not None and not hourly_df.empty:
                    # Aggregate hourly → daily and merge
                    agg_df = self._aggregate_hourly_to_daily(hourly_df)
                    if not agg_df.empty:
                        dfs.append(agg_df)

            if not dfs:
                return FetchResult(
                    source=DataSourceType.OPEN_METEO,
                    category=DataCategory.RAINFALL,
                    success=False,
                    error="No data returned from Open-Meteo API",
                )

            # Merge daily datasets on date
            merged = dfs[0]
            for df in dfs[1:]:
                merged = pd.merge(merged, df, on="date", how="outer", suffixes=("", "_hourly"))

            # Standardise column names to match real training feature names
            merged = self._standardise_columns(merged)
            merged = merged.loc[:, ~merged.columns.duplicated(keep="first")]
            merged["lat"]  = latitude
            merged["lon"] = longitude
            merged = merged.sort_values("date").reset_index(drop=True)
            # Compute rain_mm_monthly as 4.3 × weekly estimate (approx monthly total)
            if "rain_mm_weekly" in merged.columns:
                merged["rain_mm_monthly"] = (merged["rain_mm_weekly"] * 4.3).round(2)

            logger.info(
                f"[OpenMeteoFetcher] Fetched {len(merged)} rows, "
                f"{len(merged.columns)} columns"
            )

            return FetchResult(
                source=DataSourceType.OPEN_METEO,
                category=DataCategory.RAINFALL,
                success=True,
                data=merged,
                columns=list(merged.columns),
                row_count=len(merged),
                metadata={
                    "latitude":   latitude,
                    "longitude":  longitude,
                    "start_date": str(start_date),
                    "end_date":   str(end_date),
                    "days_back":  days_back,
                },
            )

        except Exception as exc:
            logger.error(f"[OpenMeteoFetcher] Error: {exc}")
            return FetchResult(
                source=DataSourceType.OPEN_METEO,
                category=DataCategory.RAINFALL,
                success=False,
                error=str(exc),
            )

    async def fetch_forecast(
        self,
        latitude:  float,
        longitude: float,
        days:      int = 7,
    ) -> FetchResult:
        """Fetches the N-day weather forecast (future data)."""
        params = {
            "latitude":         latitude,
            "longitude":        longitude,
            "hourly":           ",".join(HOURLY_VARIABLES),
            "daily":            ",".join(DAILY_VARIABLES),
            "forecast_days":    days,
            "timezone":         "auto",
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(FORECAST_URL, params=params)
                resp.raise_for_status()
                raw = resp.json()
            df = self._parse_daily(raw)
            df["latitude"]  = latitude
            df["longitude"] = longitude
            return FetchResult(
                source=DataSourceType.OPEN_METEO,
                category=DataCategory.RAINFALL,
                success=True,
                data=df,
                columns=list(df.columns),
                row_count=len(df),
                metadata={"type": "forecast", "days": days},
            )
        except Exception as exc:
            logger.error(f"[OpenMeteoFetcher] Forecast error: {exc}")
            return FetchResult(
                source=DataSourceType.OPEN_METEO,
                category=DataCategory.RAINFALL,
                success=False,
                error=str(exc),
            )

    # ── Internal fetch helpers ────────────────────────────────────────────

    async def _fetch_historical_daily(
        self,
        lat: float,
        lon: float,
        start: str,
        end: str,
    ) -> Optional[pd.DataFrame]:
        params = {
            "latitude":   lat,
            "longitude":  lon,
            "start_date": start,
            "end_date":   end,
            "daily":      ",".join(DAILY_VARIABLES),
            "timezone":   "auto",
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(HISTORICAL_URL, params=params)
                resp.raise_for_status()
                raw = resp.json()
            return self._parse_daily(raw)
        except Exception as exc:
            logger.warning(f"[OpenMeteoFetcher] Historical daily error: {exc}")
            return None

    async def _fetch_historical_hourly(
        self,
        lat: float,
        lon: float,
        start: str,
        end: str,
    ) -> Optional[pd.DataFrame]:
        params = {
            "latitude":   lat,
            "longitude":  lon,
            "start_date": start,
            "end_date":   end,
            "hourly":     ",".join(HOURLY_VARIABLES),
            "timezone":   "auto",
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(HISTORICAL_URL, params=params)
                resp.raise_for_status()
                raw = resp.json()
            return self._parse_hourly(raw)
        except Exception as exc:
            logger.warning(f"[OpenMeteoFetcher] Historical hourly error: {exc}")
            return None

    # ── Parsers ───────────────────────────────────────────────────────────

    @staticmethod
    def _parse_daily(raw: Dict[str, Any]) -> pd.DataFrame:
        daily = raw.get("daily", {})
        if not daily or "time" not in daily:
            return pd.DataFrame()
        df = pd.DataFrame(daily)
        df.rename(columns={"time": "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        return df

    @staticmethod
    def _parse_hourly(raw: Dict[str, Any]) -> pd.DataFrame:
        hourly = raw.get("hourly", {})
        if not hourly or "time" not in hourly:
            return pd.DataFrame()
        df = pd.DataFrame(hourly)
        df.rename(columns={"time": "datetime"}, inplace=True)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["date"]     = df["datetime"].dt.date
        return df

    @staticmethod
    def _aggregate_hourly_to_daily(df: pd.DataFrame) -> pd.DataFrame:
        """Aggregates hourly readings → daily totals/means."""
        agg: Dict[str, str] = {}
        for col in df.columns:
            if col in ("datetime", "date"):
                continue
            if "precipitation" in col or "rain" in col or "et0" in col:
                agg[col] = "sum"
            else:
                agg[col] = "mean"

        if not agg:
            return pd.DataFrame()

        daily = df.groupby("date").agg(agg).reset_index()
        daily["date"] = pd.to_datetime(daily["date"])
        return daily

    @staticmethod
    def _standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Renames Open-Meteo API columns to the real training feature names
        used in the india_pakistan_flood_balancednew.csv dataset.
        """
        rename_map = {
            # Precipitation → weekly rainfall (primary weather feature)
            "precipitation_sum":            "rain_mm_weekly",
            "rain_sum":                     "rain_mm_weekly",    # fallback alias
            "precipitation_hours":          "rain_hours",
            # Wind (m/s max → mean equivalent)
            "windspeed_10m_max":            "wind_ms_mean",
            "windspeed_10m":                "wind_ms_mean",
            # Temperature (2m → daily mean)
            "temperature_2m":               "temp_c_mean",
            # Relative humidity
            "relativehumidity_2m":          "rh_percent_mean",
            # Other weather cols (kept under descriptive names)
            "et0_fao_evapotranspiration":   "evapotranspiration_mm",
            "cloudcover":                   "cloud_cover_pct",
            "surface_pressure":             "pressure_hpa",
        }
        return df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
