"""
Hydrological Data Fetcher – Retrieves river discharge, water level, and soil moisture data.

Sources (in priority order):
  1. India-WRIS (Water Resources Information System) – primary for Indian locations
  2. Central Water Commission (CWC) Open Data – river gauge station data
  3. NASA GLDAS – Global Land Data Assimilation System (global coverage, soil moisture)
  4. USGS NWIS – National Water Information System (reference / US coverage)

Note: Government APIs often require registration or tokens. Free-tier endpoints
      and open data portals are used wherever available. Graceful fallback logic
      is included for all sources.
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


# ── API endpoints ─────────────────────────────────────────────────────────────

# NASA GLDAS (VIC 0.25-degree, free, no key for basic access)
GLDAS_BASE_URL   = "https://hydro1.gesdisc.eosdis.nasa.gov/daac-bin/access/timeseries.cgi"

# USGS NWIS (completely free, no key)
USGS_NWIS_URL    = "https://waterservices.usgs.gov/nwis/iv/"

# India-WRIS open API (public endpoint for water level data)
INDIA_WRIS_URL   = "https://indiawris.gov.in/wris/api/"

# CWC open data portal
CWC_PORTAL_URL   = "https://cwc.gov.in/api/"

TIMEOUT = 25


class HydroFetcher:
    """
    Fetches hydrological data for a given location.

    The fetcher attempts multiple sources in sequence and merges what is
    available, producing a standardised DataFrame with columns aligned to
    the real training features:
      - dist_major_river_km  (distance to nearest major river)
      - dam_count_50km       (number of dams within 50km)
      - waterbody_nearby     (1 if a waterbody is within 5km, else 0)
    Legacy columns (water_level_m, discharge_m3s) may still appear from
    real API responses and are passed through for reference.
    """

    def __init__(self) -> None:
        pass

    # ── Public API ────────────────────────────────────────────────────────

    async def fetch(
        self,
        latitude:   float,
        longitude:  float,
        days_back:  int = 30,
        **kwargs,
    ) -> FetchResult:
        """
        Fetches hydrological data for a location using available sources.

        Args:
            latitude:    Location latitude.
            longitude:   Location longitude.
            days_back:   Number of historical days to fetch.

        Returns:
            FetchResult with a combined hydrological DataFrame.
        """
        logger.info(
            f"[HydroFetcher] Fetching hydro data for "
            f"({latitude:.4f}, {longitude:.4f}), days_back={days_back}"
        )

        end_date   = datetime.utcnow().date()
        start_date = end_date - timedelta(days=days_back)

        dfs: List[pd.DataFrame] = []
        sources_tried: List[str] = []

        # ── Attempt 1: NASA GLDAS soil moisture ──────────────────────────
        gldas_df = await self._fetch_gldas(latitude, longitude, str(start_date), str(end_date))
        if gldas_df is not None and not gldas_df.empty:
            dfs.append(gldas_df)
            sources_tried.append("gldas")

        # ── Attempt 2: USGS NWIS (best for global/US, useful as reference) ─
        usgs_df = await self._fetch_usgs_nearest(latitude, longitude, start_date, end_date)
        if usgs_df is not None and not usgs_df.empty:
            dfs.append(usgs_df)
            sources_tried.append("usgs")

        # ── Attempt 3: India-WRIS for Indian coordinates ──────────────────
        if self._is_india(latitude, longitude):
            wris_df = await self._fetch_india_wris(latitude, longitude, start_date, end_date)
            if wris_df is not None and not wris_df.empty:
                dfs.append(wris_df)
                sources_tried.append("india_wris")

        if not dfs:
            logger.warning("[HydroFetcher] All hydro sources failed — using synthetic proxy.")
            dfs = [self._synthetic_proxy(latitude, longitude, start_date, end_date)]
            sources_tried.append("synthetic_proxy")

        # Merge all available DataFrames
        merged = self._merge_dataframes(dfs)

        logger.info(
            f"[HydroFetcher] Collected {len(merged)} rows from: {sources_tried}"
        )

        return FetchResult(
            source=DataSourceType.HYDROLOGICAL,
            category=DataCategory.HYDRO,
            success=True,
            data=merged,
            columns=list(merged.columns),
            row_count=len(merged),
            metadata={
                "sources_tried": sources_tried,
                "latitude":      latitude,
                "longitude":     longitude,
                "start_date":    str(start_date),
                "end_date":      str(end_date),
            },
        )

    # ── Source fetchers ───────────────────────────────────────────────────

    async def _fetch_gldas(
        self,
        lat: float,
        lon: float,
        start: str,
        end: str,
    ) -> Optional[pd.DataFrame]:
        """
        Fetches NASA GLDAS soil moisture and runoff data.
        Uses the GESDISC OGC-compatible timeseries endpoint.
        """
        try:
            params = {
                "type":       "asc2",
                "location":   f"GEOM:POINT({lon},{lat})",
                "variable":   "GLDAS_NOAH025_3H_2_1:SoilMoi0_10cm_inst,GLDAS_NOAH025_3H_2_1:Qs_acc",
                "startDate":  start,
                "endDate":    end,
            }
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                resp = await client.get(GLDAS_BASE_URL, params=params)
                if resp.status_code != 200:
                    return None

            lines = [ln for ln in resp.text.split("\n") if ln and not ln.startswith("#")]
            if len(lines) < 2:
                return None

            import io
            df = pd.read_csv(io.StringIO("\n".join(lines)), sep=",")
            df.columns = [c.strip() for c in df.columns]

            # Standardise
            date_col = next((c for c in df.columns if "date" in c.lower() or "time" in c.lower()), None)
            if date_col:
                df.rename(columns={date_col: "date"}, inplace=True)
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.dropna(subset=["date"])

            col_map = {}
            for c in df.columns:
                if "soilmoi" in c.lower():
                    col_map[c] = "soil_moisture_pct"
                elif "qs_acc" in c.lower() or "runoff" in c.lower():
                    col_map[c] = "surface_runoff_mm"
            df.rename(columns=col_map, inplace=True)

            return df

        except Exception as exc:
            logger.warning(f"[HydroFetcher] GLDAS failed: {exc}")
            return None

    async def _fetch_usgs_nearest(
        self,
        lat: float,
        lon: float,
        start,
        end,
    ) -> Optional[pd.DataFrame]:
        """
        Queries USGS NWIS for the nearest active stream gauge station.
        Best coverage for the US; useful globally as a structural reference.
        """
        try:
            params = {
                "format":        "json",
                "parameterCd":   "00060,00065",   # discharge + gage height
                "startDT":       str(start),
                "endDT":         str(end),
                "siteStatus":    "active",
                "bBox":         f"{lon-1},{lat-1},{lon+1},{lat+1}",
                "siteType":      "ST",
            }
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                resp = await client.get(USGS_NWIS_URL, params=params)
                if resp.status_code != 200:
                    return None
                raw = resp.json()

            time_series = raw.get("value", {}).get("timeSeries", [])
            if not time_series:
                return None

            records = []
            for ts in time_series[:1]:   # Use first station
                var_desc = ts.get("variable", {}).get("variableCode", [{}])
                code = var_desc[0].get("value", "") if var_desc else ""
                for val in ts.get("values", [{}])[0].get("value", []):
                    records.append({
                        "date":    val.get("dateTime", "")[:10],
                        "discharge_m3s" if "60" in code else "water_level_m":
                            self._to_metric(float(val.get("value", 0)), code),
                    })

            if not records:
                return None
            df = pd.DataFrame(records)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.groupby("date").mean(numeric_only=True).reset_index()
            return df

        except Exception as exc:
            logger.warning(f"[HydroFetcher] USGS NWIS failed: {exc}")
            return None

    async def _fetch_india_wris(
        self,
        lat: float,
        lon: float,
        start,
        end,
    ) -> Optional[pd.DataFrame]:
        """
        Queries India-WRIS for river gauge data near the given coordinates.
        NOTE: India-WRIS public API is sometimes rate-limited; this is a
        best-effort attempt.
        """
        try:
            params = {
                "lat":       lat,
                "lon":       lon,
                "from_date": str(start),
                "to_date":   str(end),
                "radius":    50,   # km search radius
            }
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                resp = await client.get(
                    f"{INDIA_WRIS_URL}water_level",
                    params=params,
                    headers={"Accept": "application/json"},
                )
                if resp.status_code != 200:
                    return None
                raw = resp.json()

            data = raw.get("data", raw.get("results", []))
            if not data:
                return None

            df = pd.DataFrame(data)
            # Flexible rename
            for old, new in [
                ("observation_date", "date"),
                ("water_level",      "water_level_m"),
                ("discharge",        "discharge_m3s"),
                ("river",            "river_name"),
                ("station_name",     "station"),
            ]:
                if old in df.columns:
                    df.rename(columns={old: new}, inplace=True)

            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.dropna(subset=["date"])

            return df

        except Exception as exc:
            logger.warning(f"[HydroFetcher] India-WRIS failed: {exc}")
            return None

    # ── Synthetic proxy ───────────────────────────────────────────────────

    @staticmethod
    def _synthetic_proxy(lat, lon, start, end) -> pd.DataFrame:
        """
        Generates a plausible synthetic hydrological baseline when all
        real sources fail.  Produces the three real hydro training features:
          - dist_major_river_km  : estimated river proximity
          - dam_count_50km       : estimated dam count
          - waterbody_nearby     : binary flag
        These are static (per-location) values, so returned as a single-row DF.
        """
        import numpy as np
        # Rough heuristics for India/Pakistan regions
        rng = np.random.default_rng(seed=int(abs(lat * 100 + lon)))
        return pd.DataFrame([{
            "dist_major_river_km": round(float(rng.uniform(2.0, 50.0)), 2),
            "dam_count_50km":      int(rng.integers(0, 8)),
            "waterbody_nearby":    int(rng.random() > 0.5),
            "lat":                 lat,
            "lon":                 lon,
            "data_source":         "synthetic_estimated",
        }])

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _is_india(lat: float, lon: float) -> bool:
        """Rough bounding box check for India."""
        return 6.5 <= lat <= 37.5 and 68.0 <= lon <= 97.5

    @staticmethod
    def _to_metric(value: float, param_code: str) -> float:
        """Converts USGS imperial units to metric."""
        if "60" in param_code:   # Discharge: cfs → m³/s
            return round(value * 0.028316, 3)
        if "65" in param_code:   # Gage height: ft → m
            return round(value * 0.3048, 3)
        return value

    @staticmethod
    def _merge_dataframes(dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """Outer-joins multiple hydro DataFrames on the 'date' column."""
        if not dfs:
            return pd.DataFrame()
        merged = dfs[0]
        for df in dfs[1:]:
            if "date" in df.columns and "date" in merged.columns:
                merged = pd.merge(merged, df, on="date", how="outer", suffixes=("", "_extra"))
            else:
                merged = pd.concat([merged, df], ignore_index=True)
        if "date" in merged.columns:
            merged = merged.sort_values("date").reset_index(drop=True)
        return merged
