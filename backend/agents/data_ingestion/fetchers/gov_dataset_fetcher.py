"""
Government Dataset Fetcher – Retrieves data from public / government open data portals.

Supported portals:
  1. data.gov.in      – India's national open data portal
  2. IMD (India Meteorological Department) – Historical weather/rainfall archives
  3. India-NDMA       – National Disaster Management Authority datasets
  4. Kerala SDMA      – State Disaster Management Authority (flood-prone state)
  5. OpenDataSoft     – Aggregator with many government datasets

All sources are queried asynchronously with graceful per-source fallback.
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

DATA_GOV_IN_URL   = "https://api.data.gov.in/resource"
IMD_ARCHIVE_URL   = "https://www.imdpune.gov.in/cmpg/Griddata"
NDMA_PORTAL_URL   = "https://ndma.gov.in/en/open-data"
KERALA_SDMA_URL   = "https://sdma.kerala.gov.in/api/flood"
FLOODMAP_URL      = "https://global-flood-database.cloudtostreet.ai/"

TIMEOUT = 25

# data.gov.in resource IDs for flood-relevant datasets
GOV_IN_RESOURCES = {
    "imd_rainfall":   "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69",
    "cwc_water_level":"551f4e4e-bbf5-404a-a0ce-79deae3fcb07",
    "district_flood": "93e2e5bf-8b6a-4cbc-afe3-a6ed7adeff1d",
}


class GovDatasetFetcher:
    """
    Queries government open data portals for flood-relevant datasets.

    Produces a DataFrame with columns including:
      date, district, state, rainfall_mm (IMD-sourced), flood_reported (bool),
      damages_inr, casualties, river_name (where available)
    """

    def __init__(self, api_key: str = "") -> None:
        # data.gov.in requires an API key for higher rate limits (free to obtain)
        self._gov_in_key = api_key or "579b464db66ec23d318f7"   # Demo key (limited)

    # ── Public API ────────────────────────────────────────────────────────

    async def fetch(
        self,
        latitude:  float,
        longitude: float,
        location:  str   = "",
        state:     str   = "",
        days_back: int   = 60,
        **kwargs,
    ) -> FetchResult:
        """
        Fetches government dataset records for the given location.

        Args:
            latitude:   Target latitude (used to filter location-specific records).
            longitude:  Target longitude.
            location:   Human-readable place name (district / city).
            state:      State name for state-level filtering.
            days_back:  Number of historical days to fetch.

        Returns:
            FetchResult containing a merged DataFrame of government data.
        """
        logger.info(
            f"[GovDatasetFetcher] Fetching gov data for "
            f"{location or f'({latitude:.3f},{longitude:.3f})'}, state={state}"
        )

        end_date   = datetime.utcnow().date()
        start_date = end_date - timedelta(days=days_back)

        dfs:           List[pd.DataFrame] = []
        sources_tried: List[str]          = []

        # ── 1. data.gov.in: IMD rainfall archive ─────────────────────────
        imd_df = await self._fetch_data_gov_in(
            "imd_rainfall", state, str(start_date), str(end_date)
        )
        if imd_df is not None and not imd_df.empty:
            dfs.append(imd_df)
            sources_tried.append("data.gov.in/imd_rainfall")

        # ── 2. data.gov.in: flood incident reports ────────────────────────
        flood_df = await self._fetch_data_gov_in(
            "district_flood", state, str(start_date), str(end_date)
        )
        if flood_df is not None and not flood_df.empty:
            dfs.append(flood_df)
            sources_tried.append("data.gov.in/district_flood")

        # ── 3. Kerala SDMA (if Kerala is detected) ────────────────────────
        if self._is_kerala(latitude, longitude) or "kerala" in location.lower():
            kerala_df = await self._fetch_kerala_sdma(str(start_date), str(end_date))
            if kerala_df is not None and not kerala_df.empty:
                dfs.append(kerala_df)
                sources_tried.append("kerala_sdma")

        # ── 4. Global Flood Database for historical flood events ──────────
        gfd_df = await self._fetch_global_flood_db(latitude, longitude)
        if gfd_df is not None and not gfd_df.empty:
            dfs.append(gfd_df)
            sources_tried.append("global_flood_database")

        if not dfs:
            logger.warning("[GovDatasetFetcher] All gov sources returned no data.")
            return FetchResult(
                source=DataSourceType.GOV_DATASET,
                category=DataCategory.MIXED,
                success=False,
                error="No government data available for this location/time range.",
                metadata={"sources_tried": sources_tried},
            )

        merged = self._merge_gov_data(dfs)
        logger.info(
            f"[GovDatasetFetcher] Collected {len(merged)} rows from: {sources_tried}"
        )

        return FetchResult(
            source=DataSourceType.GOV_DATASET,
            category=DataCategory.MIXED,
            success=True,
            data=merged,
            columns=list(merged.columns),
            row_count=len(merged),
            metadata={"sources_tried": sources_tried},
        )

    # ── Source fetchers ───────────────────────────────────────────────────

    async def _fetch_data_gov_in(
        self,
        resource_name: str,
        state:         str,
        start:         str,
        end:           str,
        limit:         int = 1000,
    ) -> Optional[pd.DataFrame]:
        """Queries the data.gov.in REST API for a given resource."""
        resource_id = GOV_IN_RESOURCES.get(resource_name)
        if not resource_id:
            return None

        params: Dict[str, Any] = {
            "api-key": self._gov_in_key,
            "format":  "json",
            "limit":   limit,
        }
        if state:
            params["filters[state]"] = state
        if start:
            params["filters[date][gte]"] = start
        if end:
            params["filters[date][lte]"] = end

        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                resp = await client.get(
                    f"{DATA_GOV_IN_URL}/{resource_id}",
                    params=params,
                )
                if resp.status_code == 401:
                    logger.warning("[GovDatasetFetcher] data.gov.in API key invalid.")
                    return None
                resp.raise_for_status()
                raw = resp.json()

            records = raw.get("records", raw.get("data", []))
            if not records:
                return None

            df = pd.DataFrame(records)
            df = self._standardise_columns(df)
            return df

        except Exception as exc:
            logger.warning(f"[GovDatasetFetcher] data.gov.in/{resource_name}: {exc}")
            return None

    async def _fetch_kerala_sdma(
        self,
        start: str,
        end:   str,
    ) -> Optional[pd.DataFrame]:
        """Queries Kerala State Disaster Management Authority flood API."""
        try:
            params = {"from_date": start, "to_date": end, "format": "json"}
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                resp = await client.get(KERALA_SDMA_URL, params=params)
                if resp.status_code != 200:
                    return None
                raw = resp.json()

            data = raw.get("flood_data", raw.get("data", []))
            if not data:
                return None

            df = pd.DataFrame(data)
            df = self._standardise_columns(df)
            df["state"] = "Kerala"
            return df

        except Exception as exc:
            logger.warning(f"[GovDatasetFetcher] Kerala SDMA: {exc}")
            return None

    async def _fetch_global_flood_db(
        self,
        lat: float,
        lon: float,
        radius_deg: float = 1.0,
    ) -> Optional[pd.DataFrame]:
        """
        Queries the Cloudtostreet Global Flood Database for historical events
        near the target coordinates (public API, no key required for basic use).
        """
        try:
            params = {
                "lat":    lat,
                "lon":    lon,
                "radius": radius_deg,
                "format": "json",
            }
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                resp = await client.get(
                    f"{FLOODMAP_URL}events",
                    params=params,
                    headers={"Accept": "application/json"},
                )
                if resp.status_code != 200:
                    return None
                raw = resp.json()

            events = raw.get("events", raw.get("results", []))
            if not events:
                return None

            rows = []
            for ev in events:
                rows.append({
                    "date":                  ev.get("began", ev.get("start_date")),
                    "flood_event":           True,
                    "duration_days":         ev.get("duration"),
                    "severity":              ev.get("severity"),
                    "area_affected_km2":     ev.get("flooded_area"),
                    "displaced_persons":     ev.get("displaced"),
                    "source":                "global_flood_db",
                })
            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            return df.dropna(subset=["date"])

        except Exception as exc:
            logger.warning(f"[GovDatasetFetcher] Global Flood DB: {exc}")
            return None

    # ── Standardisation & merging ─────────────────────────────────────────

    @staticmethod
    def _standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Renames common government API column variants to the real training
        feature names used in india_pakistan_flood_balancednew.csv.
        """
        rename_map = {
            "Date":           "date",
            "DATE":           "date",
            # Rainfall → real training feature name
            "Rainfall":       "rain_mm_weekly",
            "RAINFALL":       "rain_mm_weekly",
            "Rain_mm":        "rain_mm_weekly",
            "rainfall_mm":    "rain_mm_weekly",     # old standardised name → new
            "District":       "district",
            "DISTRICT":       "district",
            "State":          "state",
            "STATE":          "state",
            # Location → real column names (lat/lon, not latitude/longitude)
            "Latitude":       "lat",
            "Longitude":      "lon",
            "latitude":       "lat",
            "longitude":      "lon",
            # Flood label → real target column
            "FloodReported":  "flood_occurred",
            "flood_reported": "flood_occurred",
            "WaterLevel":     "water_level_m",
            "Water_Level":    "water_level_m",
            "CasualtyCount":  "casualties",
            "Damage_INR":     "damages_inr",
        }
        df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    @staticmethod
    def _merge_gov_data(dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """Concatenates government DataFrames, deduplicating on date+district."""
        merged = pd.concat(dfs, ignore_index=True)
        # Try deduplication if we have key columns
        subset = [c for c in ["date", "district", "state"] if c in merged.columns]
        if subset:
            merged = merged.drop_duplicates(subset=subset, keep="first")
        if "date" in merged.columns:
            merged = merged.sort_values("date").reset_index(drop=True)
        return merged

    @staticmethod
    def _is_kerala(lat: float, lon: float) -> bool:
        """Rough bounding box check for Kerala state."""
        return 8.1 <= lat <= 12.8 and 74.8 <= lon <= 77.5
