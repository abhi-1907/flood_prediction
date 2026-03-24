"""
Terrain Fetcher – Retrieves elevation and terrain data from multiple sources.

Sources used (in priority order):
  1. Google Maps Elevation API – precise spot elevation values
  2. Open-Elevation (free, open-source) – fallback if Google Maps key unavailable
  3. OpenTopoData (SRTM 90m) – another free fallback

Also derives:
  - Slope estimate (from elevation gradient across bounding box)
  - Basic land-use context from OpenStreetMap (overpass API)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx
import pandas as pd
import math

from agents.data_ingestion.ingestion_schemas import (
    DataCategory,
    DataSourceType,
    FetchResult,
)
from config import settings
from utils.logger import logger


# ── API URLs ──────────────────────────────────────────────────────────────────

OPEN_ELEVATION_URL    = "https://api.open-elevation.com/api/v1/lookup"
OPENTOPODATA_URL      = "https://api.opentopodata.org/v1/srtm90m"
OSM_OVERPASS_URL      = "https://overpass-api.de/api/interpreter"
TIMEOUT               = 20


class TerrainFetcher:
    """
    Fetches elevation, slope, and terrain context for a given lat/lng.

    Returns a single-row DataFrame with terrain attributes, plus a
    bounding-box grid of elevation points for slope estimation.
    """


    def __init__(self) -> None:
        pass

    # ── Public API ────────────────────────────────────────────────────────

    async def fetch(
        self,
        latitude:   float,
        longitude:  float,
        radius_km:  float = 30.0,
        grid_size:  int   = 5,       # NxN grid of elevation samples
        **kwargs,
    ) -> FetchResult:
        """
        Fetches terrain data for a location and a surrounding bounding box.

        Args:
            latitude:    Centre latitude.
            longitude:   Centre longitude.
            radius_km:   Radius of surrounding area to sample (km).
            grid_size:   Width/height of the elevation grid (e.g. 5 → 25 points).

        Returns:
            FetchResult with a DataFrame containing terrain attributes.
        """
        logger.info(
            f"[TerrainFetcher] Fetching terrain for "
            f"({latitude:.4f}, {longitude:.4f}), radius={radius_km}km"
        )

        try:
            # Build sample grid
            locations = self._build_grid(latitude, longitude, radius_km, grid_size)

            # Fetch elevations
            elevations = await self._get_elevations(locations)

            if not elevations:
                return FetchResult(
                    source=DataSourceType.TERRAIN,
                    category=DataCategory.TERRAIN,
                    success=False,
                    error="Could not fetch any elevation data.",
                )

            # Build DataFrame
            df = pd.DataFrame(elevations)
            df["slope_deg"] = self._estimate_slope(df)

            # Fetch land-use categories for centre point
            land_use = await self._fetch_land_use(latitude, longitude)
            df["land_use"]       = land_use.get("primary", "unknown")
            df["water_bodies"]   = land_use.get("water_bodies", False)
            df["urban"]          = land_use.get("urban", False)
            df["forest"]         = land_use.get("forest", False)

            # Summary row for the centre point — use REAL dataset column names
            centre = df.iloc[len(df) // 2] if len(df) > 0 else df.iloc[0]
            # Estimate dist_major_river_km from land-use context (waterbody = close = ~5km)
            waterbody_flag = int(bool(land_use.get("water_bodies", False)))
            dist_river_est = round(5.0 if waterbody_flag else 20.0, 1)   # rough heuristic
            summary = pd.DataFrame([{
                # Location — real dataset uses lat/lon (not latitude/longitude)
                "lat":               latitude,
                "lon":               longitude,
                # Real terrain features expected by model
                "elevation_m":       float(centre["elevation_m"]),
                "slope_degree":      float(df["slope_deg"].mean()),
                # Real hydro features derivable from terrain context
                "dist_major_river_km": dist_river_est,
                "waterbody_nearby":  float(waterbody_flag),
                "dam_count_50km":    0.0,   # unknown without specific API; default 0
                # Additional context (not model features; may be dropped in preprocessing)
                "terrain_type":      land_use.get("primary", "unknown"),
                "max_elevation_m":   float(df["elevation_m"].max()),
                "min_elevation_m":   float(df["elevation_m"].min()),
                "mean_elevation_m":  float(df["elevation_m"].mean()),
                "urban_area":        land_use.get("urban", False),
                "forest_cover":      land_use.get("forest", False),
                "flood_plain_risk":  self._estimate_flood_plain_risk(df),
            }])

            logger.info(
                f"[TerrainFetcher] Done — elevation range: "
                f"{df['elevation_m'].min():.1f}–{df['elevation_m'].max():.1f} m"
            )

            return FetchResult(
                source=DataSourceType.TERRAIN,
                category=DataCategory.TERRAIN,
                success=True,
                data=summary,
                columns=list(summary.columns),
                row_count=len(summary),
                metadata={
                    "grid_df":    df.to_dict(orient="records"),
                    "radius_km":  radius_km,
                    "grid_size":  grid_size,
                    "land_use":   land_use,
                },
            )

        except Exception as exc:
            logger.error(f"[TerrainFetcher] Error: {exc}")
            return FetchResult(
                source=DataSourceType.TERRAIN,
                category=DataCategory.TERRAIN,
                success=False,
                error=str(exc),
            )

    # ── Grid generation ───────────────────────────────────────────────────

    @staticmethod
    def _build_grid(
        lat: float, lon: float, radius_km: float, n: int
    ) -> List[Dict[str, float]]:
        """Creates an NxN grid of lat/lng points around the centre."""
        deg_lat = radius_km / 111.0
        deg_lon = radius_km / (111.0 * math.cos(math.radians(lat)))
        step_lat = (2 * deg_lat) / (n - 1) if n > 1 else 0
        step_lon = (2 * deg_lon) / (n - 1) if n > 1 else 0

        points = []
        for i in range(n):
            for j in range(n):
                points.append({
                    "latitude":  lat - deg_lat + i * step_lat,
                    "longitude": lon - deg_lon + j * step_lon,
                })
        return points

    # ── Elevation fetching ────────────────────────────────────────────────

    async def _get_elevations(
        self, locations: List[Dict[str, float]]
    ) -> List[Dict[str, Any]]:
        """Tries Open-Elevation → OpenTopoData in order."""
        result = await self._open_elevation(locations)
        if result:
            return result

        result = await self._opentopodata(locations)
        return result or []


    async def _open_elevation(
        self, locations: List[Dict[str, float]]
    ) -> List[Dict[str, Any]]:
        """Open-Elevation API (free, no key)."""
        try:
            payload = {"locations": locations}
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                resp = await client.post(OPEN_ELEVATION_URL, json=payload)
                resp.raise_for_status()
                raw = resp.json()
            return [
                {
                    "latitude":    r["latitude"],
                    "longitude":   r["longitude"],
                    "elevation_m": r["elevation"],
                    "resolution":  90,
                }
                for r in raw.get("results", [])
            ]
        except Exception as exc:
            logger.warning(f"[TerrainFetcher] Open-Elevation failed: {exc}")
            return []

    async def _opentopodata(
        self, locations: List[Dict[str, float]]
    ) -> List[Dict[str, Any]]:
        """OpenTopoData SRTM-90m (free, no key, rate limited)."""
        # Split into batches of 100 (API limit)
        batches = [locations[i:i+100] for i in range(0, len(locations), 100)]
        results = []
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                for batch in batches:
                    loc_str = "|".join(f"{p['latitude']},{p['longitude']}" for p in batch)
                    resp = await client.get(OPENTOPODATA_URL, params={"locations": loc_str})
                    resp.raise_for_status()
                    raw = resp.json()
                    for r in raw.get("results", []):
                        results.append({
                            "latitude":    r["location"]["lat"],
                            "longitude":   r["location"]["lng"],
                            "elevation_m": r.get("elevation") or 0.0,
                            "resolution":  90,
                        })
            return results
        except Exception as exc:
            logger.warning(f"[TerrainFetcher] OpenTopoData failed: {exc}")
            return []

    # ── Land use ─────────────────────────────────────────────────────────

    async def _fetch_land_use(
        self, lat: float, lon: float, radius_m: int = 5000
    ) -> Dict[str, Any]:
        """Queries OpenStreetMap overpass for basic land-use tags."""
        query = f"""
[out:json][timeout:15];
(
  way["natural"="water"](around:{radius_m},{lat},{lon});
  way["landuse"="residential"](around:{radius_m},{lat},{lon});
  way["landuse"="forest"](around:{radius_m},{lat},{lon});
  way["natural"="wetland"](around:{radius_m},{lat},{lon});
  way["natural"="floodplain"](around:{radius_m},{lat},{lon});
);
out count;
"""
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(OSM_OVERPASS_URL, data=query)
                resp.raise_for_status()
                raw = resp.json()

            tags = {el.get("type", "") for el in raw.get("elements", [])}
            count = raw.get("elements", [])

            return {
                "primary":      "mixed",
                "water_bodies": any("water" in str(e) for e in count),
                "urban":        any("residential" in str(e) for e in count),
                "forest":       any("forest" in str(e) for e in count),
                "wetland":      any("wetland" in str(e) for e in count),
                "floodplain":   any("floodplain" in str(e) for e in count),
            }
        except Exception as exc:
            logger.warning(f"[TerrainFetcher] OSM land-use query failed: {exc}")
            return {
                "primary": "unknown",
                "water_bodies": False,
                "urban": False,
                "forest": False,
                "wetland": False,
                "floodplain": False,
            }

    # ── Derived metrics ───────────────────────────────────────────────────

    @staticmethod
    def _estimate_slope(df: pd.DataFrame) -> pd.Series:
        """
        Approximates slope in degrees from elevation differences between adjacent rows.
        Horizontal distance is assumed to be constant (grid spacing).
        """
        elev = df["elevation_m"].values
        diffs = pd.Series(elev).diff().abs().fillna(0)
        # Assume ~1 km between adjacent grid points — convert to degrees
        return (diffs / 1000).apply(lambda g: math.degrees(math.atan(g)))

    @staticmethod
    def _estimate_flood_plain_risk(df: pd.DataFrame) -> str:
        """
        Heuristic flood-plain risk based on elevation range and slope.
        LOW / MODERATE / HIGH
        """
        elev_range = df["elevation_m"].max() - df["elevation_m"].min()
        mean_slope = df["slope_deg"].mean() if "slope_deg" in df.columns else 5.0

        if mean_slope < 1.0 and elev_range < 10:
            return "HIGH"
        if mean_slope < 3.0 and elev_range < 50:
            return "MODERATE"
        return "LOW"
