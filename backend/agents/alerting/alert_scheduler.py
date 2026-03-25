"""
Alert Scheduler – Manages alert timing, de-duplication, escalation, and cooldown.

Prevents alert fatigue through:
  1. Cooldown periods (don't re-alert within N minutes for same location)
  2. Escalation logic (upgrade channel/audience if risk increases)
  3. De-duplicate check (same location + same risk level = skip)
  4. Auto-expiry of alerts after a configurable duration
  5. Time-of-day awareness (avoid alerts at 2 AM unless CRITICAL)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from agents.alerting.alerting_schemas import (
    Alert,
    AlertSeverity,
    AlertStatus,
    EscalationLevel,
)
from utils.logger import logger


# ── Configuration ─────────────────────────────────────────────────────────────

COOLDOWN_MINUTES = {
    AlertSeverity.GREEN:  360,    # 6 hours
    AlertSeverity.YELLOW: 120,    # 2 hours
    AlertSeverity.ORANGE: 60,     # 1 hour
    AlertSeverity.RED:    1,      # 1 minute (dev) — reduce alert fatigue in prod by raising to 15+
}

ESCALATION_MAP = {
    EscalationLevel.NONE:    EscalationLevel.LEVEL_1,
    EscalationLevel.LEVEL_1: EscalationLevel.LEVEL_2,
    EscalationLevel.LEVEL_2: EscalationLevel.LEVEL_3,
    EscalationLevel.LEVEL_3: EscalationLevel.LEVEL_3,   # Max
}

SEVERITY_ORDER = {
    AlertSeverity.GREEN:  0,
    AlertSeverity.YELLOW: 1,
    AlertSeverity.ORANGE: 2,
    AlertSeverity.RED:    3,
}

RISK_TO_SEVERITY = {
    "LOW":      AlertSeverity.GREEN,
    "MEDIUM":   AlertSeverity.YELLOW,
    "HIGH":     AlertSeverity.ORANGE,
    "CRITICAL": AlertSeverity.RED,
}

# Quiet hours: don't alert between 11PM and 6AM unless CRITICAL
QUIET_HOURS = (23, 6)


class AlertScheduler:
    """
    Manages alert timing, de-duplication, and escalation.

    Usage:
        scheduler = AlertScheduler()
        should_alert, severity, escalation = scheduler.evaluate(
            location="Kochi", risk_level="HIGH"
        )
    """

    def __init__(self) -> None:
        self._history: Dict[str, Alert] = {}   # location_key → last Alert

    # ── Public API ────────────────────────────────────────────────────────

    def evaluate(
        self,
        location:   str,
        risk_level: str,
        latitude:   Optional[float] = None,
        longitude:  Optional[float] = None,
    ) -> Tuple[bool, AlertSeverity, EscalationLevel]:
        """
        Decides whether an alert should be sent.

        Returns:
            (should_alert, severity, escalation_level)
        """
        severity   = RISK_TO_SEVERITY.get(risk_level.upper(), AlertSeverity.YELLOW)
        loc_key    = self._location_key(location, latitude, longitude)
        now        = datetime.now(timezone.utc)

        # Check cooldown
        last_alert = self._history.get(loc_key)
        if last_alert:
            cooldown   = timedelta(minutes=COOLDOWN_MINUTES.get(severity, 60))
            last_time  = self._parse_time(last_alert.created_at)
            time_since = now - last_time

            if time_since < cooldown:
                # Check for escalation (risk level increased since last alert)
                last_sev_order = SEVERITY_ORDER.get(last_alert.severity, 0)
                curr_sev_order = SEVERITY_ORDER.get(severity, 0)

                if curr_sev_order > last_sev_order:
                    # Risk increased — escalate
                    escalation = ESCALATION_MAP.get(
                        last_alert.escalation, EscalationLevel.LEVEL_2
                    )
                    logger.info(
                        f"[AlertScheduler] ESCALATED {location}: "
                        f"{last_alert.severity.value} → {severity.value} | "
                        f"escalation={escalation.value}"
                    )
                    return True, severity, escalation

                # Same or lower severity within cooldown — skip
                logger.info(
                    f"[AlertScheduler] COOLDOWN active for {location}: "
                    f"{time_since.seconds // 60}min < {cooldown.seconds // 60}min"
                )
                return False, severity, EscalationLevel.NONE

        # Quiet hours check (skip non-critical during night)
        if self._is_quiet_hours(now) and severity != AlertSeverity.RED:
            logger.info(
                f"[AlertScheduler] QUIET HOURS — deferring {severity.value} "
                f"alert for {location}"
            )
            return False, severity, EscalationLevel.NONE

        # GREEN severity — no alert needed
        if severity == AlertSeverity.GREEN:
            logger.info(f"[AlertScheduler] GREEN severity for {location} — no alert")
            return False, severity, EscalationLevel.NONE

        logger.info(
            f"[AlertScheduler] ALERT approved for {location}: "
            f"severity={severity.value}"
        )
        return True, severity, EscalationLevel.LEVEL_1

    def record_alert(self, alert: Alert) -> None:
        """Records an alert in history for cooldown tracking."""
        loc_key = self._location_key(
            alert.location, alert.latitude, alert.longitude
        )
        self._history[loc_key] = alert
        logger.info(f"[AlertScheduler] Recorded alert {alert.alert_id} for {alert.location}")

    def clear_history(self, location: Optional[str] = None) -> None:
        """Clears alert history (all or for a specific location)."""
        if location:
            keys = [k for k in self._history if location.lower() in k.lower()]
            for k in keys:
                del self._history[k]
        else:
            self._history.clear()

    def get_active_alerts(self) -> List[Alert]:
        """Returns all alerts that haven't expired."""
        now    = datetime.now(timezone.utc)
        active = []
        for alert in self._history.values():
            if alert.expires_at:
                expires = self._parse_time(alert.expires_at)
                if now < expires:
                    active.append(alert)
            else:
                # No expiry set — keep for 24 hours
                created = self._parse_time(alert.created_at)
                if (now - created) < timedelta(hours=24):
                    active.append(alert)
        return active

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _location_key(
        location:  str,
        latitude:  Optional[float] = None,
        longitude: Optional[float] = None,
    ) -> str:
        if latitude is not None and longitude is not None:
            return f"{round(latitude, 2)}_{round(longitude, 2)}"
        return location.lower().strip().replace(" ", "_")

    @staticmethod
    def _parse_time(iso_str: str) -> datetime:
        try:
            return datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        except Exception:
            return datetime.now(timezone.utc) - timedelta(hours=24)

    @staticmethod
    def _is_quiet_hours(now: datetime) -> bool:
        hour = now.hour
        start, end = QUIET_HOURS
        if start > end:
            return hour >= start or hour < end
        return start <= hour < end
