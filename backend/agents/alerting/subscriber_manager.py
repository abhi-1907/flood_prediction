"""
Subscriber Manager – CRUD operations for alert subscribers.

Manages:
  - In-memory subscriber registry (production: swap for DB)
  - Location-based subscriber matching (haversine distance)
  - Risk-level filtering (only alert subscribers at or above their threshold)
  - Authority vs public subscriber classification
  - Subscriber import/export
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from agents.alerting.alerting_schemas import Subscriber
from utils.logger import logger


# ── Risk level ordering ───────────────────────────────────────────────────────

RISK_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}

EARTH_RADIUS_KM = 6371.0


class SubscriberManager:
    """
    Manages alert subscribers with location-based matching.

    In-memory storage for development; swap _store for a DB adapter in production.

    Usage:
        manager = SubscriberManager()
        sub_id  = manager.add(name="Alice", email="a@b.com", location="Kochi", ...)
        matched = manager.find_by_location(lat=9.93, lon=76.26, risk_level="HIGH")
    """

    def __init__(self) -> None:
        self._store: Dict[str, Subscriber] = {}

    # ── CRUD ──────────────────────────────────────────────────────────────

    def add(
        self,
        name:           str,
        location:       str,
        latitude:       float,
        longitude:      float,
        email:          Optional[str]   = None,
        phone:          Optional[str]   = None,
        push_token:     Optional[str]   = None,
        channels:       Optional[List[str]] = None,
        radius_km:      float           = 25.0,
        min_risk_level: str             = "MEDIUM",
        is_authority:   bool            = False,
    ) -> str:
        """Registers a new subscriber and returns their ID."""
        sub_id = str(uuid.uuid4())[:8]
        sub = Subscriber(
            id=sub_id,
            name=name,
            email=email,
            phone=phone,
            push_token=push_token,
            location=location,
            latitude=latitude,
            longitude=longitude,
            radius_km=radius_km,
            channels=channels or ["email"],
            min_risk_level=min_risk_level,
            is_authority=is_authority,
            is_active=True,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._store[sub_id] = sub
        logger.info(f"[SubscriberManager] Added subscriber {sub_id}: {name} @ {location}")
        return sub_id

    def remove(self, subscriber_id: str) -> bool:
        """Deactivates a subscriber (soft delete)."""
        if subscriber_id in self._store:
            self._store[subscriber_id].is_active = False
            logger.info(f"[SubscriberManager] Deactivated subscriber {subscriber_id}")
            return True
        return False

    def get(self, subscriber_id: str) -> Optional[Subscriber]:
        return self._store.get(subscriber_id)

    def list_all(self) -> List[Subscriber]:
        return [s for s in self._store.values() if s.is_active]

    def count(self) -> int:
        return sum(1 for s in self._store.values() if s.is_active)

    # ── Location-based matching ───────────────────────────────────────────

    def find_by_location(
        self,
        latitude:   float,
        longitude:  float,
        risk_level: str,
    ) -> List[Subscriber]:
        """
        Finds all active subscribers within range of the given coordinates
        whose min_risk_level threshold is met.

        Args:
            latitude:   Alert location latitude.
            longitude:  Alert location longitude.
            risk_level: Current risk level (LOW/MEDIUM/HIGH/CRITICAL).

        Returns:
            List of matching Subscribers, sorted by distance (nearest first).
        """
        alert_risk_order = RISK_ORDER.get(risk_level.upper(), 1)
        matched: List[tuple] = []

        for sub in self._store.values():
            if not sub.is_active:
                continue

            # Risk threshold check
            sub_threshold = RISK_ORDER.get(sub.min_risk_level.upper(), 1)
            if alert_risk_order < sub_threshold:
                continue

            # Distance check (haversine)
            dist = self._haversine(latitude, longitude, sub.latitude, sub.longitude)
            if dist <= sub.radius_km:
                matched.append((dist, sub))

        # Sort by distance (nearest first)
        matched.sort(key=lambda x: x[0])
        result = [sub for _, sub in matched]

        logger.info(
            f"[SubscriberManager] Location match: {len(result)}/{self.count()} "
            f"subscribers within range for risk={risk_level}"
        )
        return result

    def find_authorities(
        self,
        latitude:  float,
        longitude: float,
    ) -> List[Subscriber]:
        """Finds authority subscribers near the given location (no risk threshold)."""
        matched = []
        for sub in self._store.values():
            if not sub.is_active or not sub.is_authority:
                continue
            dist = self._haversine(latitude, longitude, sub.latitude, sub.longitude)
            if dist <= sub.radius_km * 2:   # Wider radius for authorities
                matched.append(sub)
        return matched

    # ── Bulk operations ───────────────────────────────────────────────────

    def import_subscribers(self, data: List[Dict[str, Any]]) -> int:
        """Bulk imports subscribers from a list of dicts."""
        count = 0
        for item in data:
            try:
                self.add(
                    name=item.get("name", "Unknown"),
                    location=item.get("location", ""),
                    latitude=float(item.get("latitude", 0)),
                    longitude=float(item.get("longitude", 0)),
                    email=item.get("email"),
                    phone=item.get("phone"),
                    push_token=item.get("push_token"),
                    channels=item.get("channels", ["email"]),
                    radius_km=float(item.get("radius_km", 25)),
                    min_risk_level=item.get("min_risk_level", "MEDIUM"),
                    is_authority=bool(item.get("is_authority", False)),
                )
                count += 1
            except Exception as exc:
                logger.warning(f"[SubscriberManager] Import failed for item: {exc}")
        logger.info(f"[SubscriberManager] Imported {count}/{len(data)} subscribers")
        return count

    def export_subscribers(self) -> List[Dict[str, Any]]:
        """Exports all active subscribers as dicts."""
        return [s.model_dump() for s in self._store.values() if s.is_active]

    # ── Haversine distance ────────────────────────────────────────────────

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Returns distance in km between two lat/lon points."""
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(dlon / 2) ** 2)
        return EARTH_RADIUS_KM * 2 * math.asin(math.sqrt(a))
