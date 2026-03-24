"""
Geo Utilities – Shared geospatial helper functions.

Includes:
  - Haversine distance calculation
  - Bounding box construction from a centre point + radius
  - Grid coordinate generation (N×N lat/lon grid)
  - GeoJSON feature builders (polygon, point, line)
  - Coordinate validation
  - Indian state / city → lat/lon lookup
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

# ── Constants ─────────────────────────────────────────────────────────────────

EARTH_RADIUS_KM = 6371.0

# Common Indian city → (lat, lon)
INDIA_CITY_COORDS: Dict[str, Tuple[float, float]] = {
    "kochi":       (9.9312, 76.2673),
    "cochin":      (9.9312, 76.2673),
    "mumbai":      (19.0760, 72.8777),
    "delhi":       (28.6139, 77.2090),
    "chennai":     (13.0827, 80.2707),
    "kolkata":     (22.5726, 88.3639),
    "hyderabad":   (17.3850, 78.4867),
    "bangalore":   (12.9716, 77.5946),
    "bengaluru":   (12.9716, 77.5946),
    "pune":        (18.5204, 73.8567),
    "ahmedabad":   (23.0225, 72.5714),
    "patna":       (25.5941, 85.1376),
    "guwahati":    (26.1445, 91.7362),
    "bhubaneswar": (20.2961, 85.8245),
    "visakhapatnam": (17.6868, 83.2185),
    "surat":       (21.1702, 72.8311),
    "varanasi":    (25.3176, 82.9739),
    "lucknow":     (26.8467, 80.9462),
    "thiruvananthapuram": (8.5241, 76.9366),
    "trivandrum":  (8.5241, 76.9366),
    "mangalore":   (12.9141, 74.8560),
    "kozhikode":   (11.2588, 75.7804),
    "calicut":     (11.2588, 75.7804),
    "thrissur":    (10.5276, 76.2144),
    "bhopal":      (23.2599, 77.4126),
    "nagpur":      (21.1458, 79.0882),
    "jaipur":      (26.9124, 75.7873),
    "chandigarh":  (30.7333, 76.7794),
    "dehradun":    (30.3165, 78.0322),
    "shillong":    (25.5788, 91.8933),
    "imphal":      (24.8170, 93.9368),
}


# ── Distance ──────────────────────────────────────────────────────────────────

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Returns the great-circle distance in km between two lat/lon points."""
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return EARTH_RADIUS_KM * 2 * math.asin(math.sqrt(max(0, a)))


def bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Returns the bearing in degrees (0–360, clockwise from North)."""
    dlon = math.radians(lon2 - lon1)
    lat1r, lat2r = math.radians(lat1), math.radians(lat2)
    x = math.sin(dlon) * math.cos(lat2r)
    y = math.cos(lat1r) * math.sin(lat2r) - math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


# ── Bounding box ──────────────────────────────────────────────────────────────

def bounding_box(
    lat: float, lon: float, radius_km: float
) -> Dict[str, float]:
    """
    Returns a bounding box dict for the given centre and radius.

    Returns:
        {"min_lat": ..., "max_lat": ..., "min_lon": ..., "max_lon": ...}
    """
    delta_lat = math.degrees(radius_km / EARTH_RADIUS_KM)
    delta_lon = math.degrees(
        radius_km / (EARTH_RADIUS_KM * math.cos(math.radians(lat)))
    )
    return {
        "min_lat": lat - delta_lat,
        "max_lat": lat + delta_lat,
        "min_lon": lon - delta_lon,
        "max_lon": lon + delta_lon,
    }


# ── Grid generation ───────────────────────────────────────────────────────────

def generate_grid(
    lat: float,
    lon: float,
    radius_km: float,
    n_cells: int = 11,
) -> List[Tuple[float, float]]:
    """
    Generates an N×N grid of (lat, lon) centre points covering the given radius.

    Args:
        lat:       Centre latitude.
        lon:       Centre longitude.
        radius_km: Radius to cover.
        n_cells:   Number of cells per dimension (default 11 → 121 total).

    Returns:
        List of (lat, lon) tuples for each grid cell centre.
    """
    bb = bounding_box(lat, lon, radius_km)
    lat_step = (bb["max_lat"] - bb["min_lat"]) / n_cells
    lon_step = (bb["max_lon"] - bb["min_lon"]) / n_cells

    points = []
    for row in range(n_cells):
        for col in range(n_cells):
            cell_lat = bb["min_lat"] + (row + 0.5) * lat_step
            cell_lon = bb["min_lon"] + (col + 0.5) * lon_step
            points.append((round(cell_lat, 6), round(cell_lon, 6)))
    return points


# ── GeoJSON helpers ───────────────────────────────────────────────────────────

def geojson_point(
    lat: float, lon: float, properties: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Creates a GeoJSON Point feature."""
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": properties or {},
    }


def geojson_polygon(
    coords: List[Tuple[float, float]],
    properties: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Creates a GeoJSON Polygon feature.

    Args:
        coords: List of (lat, lon) tuples (will be closed automatically).
        properties: Optional feature properties.
    """
    ring = [[lon, lat] for lat, lon in coords]
    if ring[0] != ring[-1]:
        ring.append(ring[0])   # Close the ring
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [ring]},
        "properties": properties or {},
    }


def geojson_feature_collection(
    features: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Wraps a list of GeoJSON features into a FeatureCollection."""
    return {"type": "FeatureCollection", "features": features}


# ── Coordinate helpers ────────────────────────────────────────────────────────

def is_valid_coordinate(lat: float, lon: float) -> bool:
    """Validates a lat/lon pair."""
    return -90 <= lat <= 90 and -180 <= lon <= 180


def offset_coordinate(
    lat: float, lon: float, delta_km: float, bearing_deg: float
) -> Tuple[float, float]:
    """
    Returns a new coordinate offset from the given point by delta_km in the given bearing.
    """
    d = delta_km / EARTH_RADIUS_KM
    lat_r = math.radians(lat)
    lon_r = math.radians(lon)
    bearing_r = math.radians(bearing_deg)

    new_lat = math.asin(
        math.sin(lat_r) * math.cos(d) +
        math.cos(lat_r) * math.sin(d) * math.cos(bearing_r)
    )
    new_lon = lon_r + math.atan2(
        math.sin(bearing_r) * math.sin(d) * math.cos(lat_r),
        math.cos(d) - math.sin(lat_r) * math.sin(new_lat),
    )
    return math.degrees(new_lat), math.degrees(new_lon)


def lookup_city_coords(city: str) -> Optional[Tuple[float, float]]:
    """Looks up (lat, lon) for an Indian city name. Returns None if not found."""
    key = city.lower().strip()
    return INDIA_CITY_COORDS.get(key)


def parse_location(location: str) -> Dict[str, Any]:
    """
    Parses a location string into lat/lon if possible.

    Tries:
      1. Known Indian cities lookup
      2. Returns None coordinates if unrecognised

    Returns:
        dict with keys: "location", "latitude", "longitude", "resolved"
    """
    coords = lookup_city_coords(location)
    return {
        "location":  location,
        "latitude":  coords[0] if coords else None,
        "longitude": coords[1] if coords else None,
        "resolved":  coords is not None,
    }
