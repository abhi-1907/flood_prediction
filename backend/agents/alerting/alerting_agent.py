"""
Alerting Agent – Top-level coordinator for the multi-channel flood alert pipeline.

Execution pipeline:
  1. Evaluate alert necessity (AlertScheduler: cooldown, escalation, quiet hours)
  2. Find matching subscribers (SubscriberManager: haversine location matching)
  3. Compose alert content (AlertComposer: SMS, HTML email, push, webhook)
  4. Deliver to all channels in parallel (Email, SMS, Push, Webhook)
  5. Track delivery records and escalation state
  6. Store alert artifacts in session

Registered in ToolRegistry as "alerting".
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from agents.alerting.alerting_schemas import (
    Alert,
    AlertContent,
    AlertingResult,
    AlertSeverity,
    AlertStatus,
    DeliveryRecord,
    EscalationLevel,
    Subscriber,
)
from agents.alerting.subscriber_manager import SubscriberManager
from agents.alerting.alert_composer import AlertComposer
from agents.alerting.alert_scheduler import AlertScheduler
from agents.alerting.channels.email_channel import EmailChannel
from agents.alerting.channels.sms_channel import SMSChannel
from agents.alerting.channels.push_channel import PushChannel
from agents.alerting.channels.webhook_channel import WebhookChannel
from agents.orchestration.memory import Session
from services.gemini_service import GeminiService, get_gemini_service
from utils.logger import logger


# ── Module-level singleton ────────────────────────────────────────────────────
# All consumers (orchestrator, API routes) should use get_alerting_agent() so
# they share the same SubscriberManager, AlertScheduler, and cooldown state.

_alerting_agent_instance: Optional["AlertingAgent"] = None


def get_alerting_agent() -> "AlertingAgent":
    """Returns the process-wide AlertingAgent singleton."""
    global _alerting_agent_instance
    if _alerting_agent_instance is None:
        _alerting_agent_instance = AlertingAgent()
    return _alerting_agent_instance


class AlertingAgent:
    """
    Orchestrates the full multi-channel alert delivery pipeline.

    Components are instantiated once and reused across sessions.
    All delivery channels support dry-run mode for testing.
    """

    def __init__(
        self,
        gemini_service: Optional[GeminiService] = None,
        dry_run:        bool = False,
    ) -> None:
        self._gemini      = gemini_service or get_gemini_service()
        self._subscribers  = SubscriberManager()
        self._composer     = AlertComposer(self._gemini)
        self._scheduler    = AlertScheduler()
        self._email        = EmailChannel(dry_run=dry_run)
        self._sms          = SMSChannel(dry_run=dry_run)
        self._push         = PushChannel(dry_run=dry_run)
        self._webhook      = WebhookChannel(dry_run=dry_run)

    # ── Public accessors ──────────────────────────────────────────────────

    @property
    def subscriber_manager(self) -> SubscriberManager:
        """Expose for API endpoints (subscribe/unsubscribe)."""
        return self._subscribers

    @property
    def scheduler(self) -> AlertScheduler:
        return self._scheduler

    # ── Main entry point ──────────────────────────────────────────────────

    async def run(
        self,
        session: Session,
        **kwargs,
    ) -> AlertingResult:
        """
        Main agent entry point.

        Reads prediction results from the session, evaluates whether an alert
        is needed, and delivers to all matching subscribers.
        """
        session_id = session.session_id
        context    = session.context
        warnings:  List[str] = []
        errors:    List[str] = []

        logger.info(f"[AlertingAgent] Starting for session {session_id}")

        try:
            # ── 1. Get prediction data ────────────────────────────────────
            prediction = session.get_artifact("ensemble_prediction")
            if not prediction or not isinstance(prediction, dict):
                return AlertingResult(
                    session_id=session_id,
                    status="skipped",
                    warnings=["No prediction data — skipping alert evaluation."],
                )

            risk_level = prediction.get("risk_level", "MEDIUM")
            flood_prob = prediction.get("flood_probability", 0.5)
            confidence = prediction.get("confidence", 0.5)
            location   = context.get("location", "Unknown")
            latitude   = context.get("latitude")
            longitude  = context.get("longitude")

            # ── 2. Evaluate alert necessity ───────────────────────────────
            should_alert, severity, escalation = self._scheduler.evaluate(
                location=location,
                risk_level=risk_level,
                latitude=latitude,
                longitude=longitude,
            )

            if not should_alert:
                logger.info(
                    f"[AlertingAgent] Alert suppressed for {location} "
                    f"(severity={severity.value}, cooldown/quiet-hours)"
                )
                return AlertingResult(
                    session_id=session_id,
                    status="suppressed",
                    escalation=EscalationLevel.NONE,
                    summary=f"Alert suppressed for {location} — cooldown active or risk too low.",
                    warnings=["Alert suppressed by scheduler."],
                )

            # ── 3. Find matching subscribers ──────────────────────────────
            if latitude is not None and longitude is not None:
                subscribers = self._subscribers.find_by_location(
                    latitude=latitude,
                    longitude=longitude,
                    risk_level=risk_level,
                )
            else:
                subscribers = self._subscribers.list_all()

            # Add authorities on escalation
            if escalation in (EscalationLevel.LEVEL_2, EscalationLevel.LEVEL_3):
                if latitude is not None and longitude is not None:
                    authorities = self._subscribers.find_authorities(latitude, longitude)
                    existing_ids = {s.id for s in subscribers}
                    for auth in authorities:
                        if auth.id not in existing_ids:
                            subscribers.append(auth)

            if not subscribers:
                warnings.append("No subscribers found for this location.")
                # Still create the alert for the session record

            # ── 4. Compose alert content ──────────────────────────────────
            risk_data = {
                "flood_probability": flood_prob,
                "risk_level":       risk_level,
                "confidence":       confidence,
                "key_factors":      session.get_artifact("prediction_explanation", {}).get(
                    "key_factors", []
                ) if isinstance(session.get_artifact("prediction_explanation"), dict) else [],
            }

            location_data = {
                "location":          location,
                "latitude":          latitude,
                "longitude":         longitude,
                "emergency_number":  context.get("emergency_number", "112"),
                "state":             context.get("state", ""),
            }

            # Compose public alert
            public_content = await self._composer.compose(
                severity=severity,
                risk_data=risk_data,
                location_data=location_data,
                for_authority=False,
            )

            # Compose authority alert (if escalated)
            authority_content = None
            if escalation in (EscalationLevel.LEVEL_2, EscalationLevel.LEVEL_3):
                authority_content = await self._composer.compose(
                    severity=severity,
                    risk_data=risk_data,
                    location_data=location_data,
                    for_authority=True,
                )

            # ── 5. Create alert record ────────────────────────────────────
            alert = Alert(
                alert_id=str(uuid.uuid4())[:12],
                session_id=session_id,
                severity=severity,
                risk_level=risk_level,
                location=location,
                latitude=latitude,
                longitude=longitude,
                flood_probability=flood_prob,
                content=public_content,
                escalation=escalation,
                created_at=datetime.now(timezone.utc).isoformat(),
                expires_at=(datetime.now(timezone.utc) + timedelta(hours=6)).isoformat(),
            )

            # ── 6. Deliver to all channels in parallel ────────────────────
            delivery_coros = []
            for sub in subscribers:
                content = authority_content if sub.is_authority and authority_content else public_content
                delivery_coros.extend(
                    self._build_delivery_coros(sub, content)
                )

            # Add webhook delivery (global)
            delivery_coros.append(self._webhook.send(content=public_content))

            if delivery_coros:
                results = await asyncio.gather(
                    *delivery_coros, return_exceptions=True
                )
                for result in results:
                    if isinstance(result, DeliveryRecord):
                        alert.deliveries.append(result)
                    elif isinstance(result, Exception):
                        logger.error(f"[AlertingAgent] Delivery error: {result}")

            # ── 7. Record in scheduler for cooldown ───────────────────────
            self._scheduler.record_alert(alert)

            # ── 8. Calculate stats ────────────────────────────────────────
            sent_count   = sum(1 for d in alert.deliveries if d.status == AlertStatus.SENT)
            failed_count = sum(1 for d in alert.deliveries if d.status == AlertStatus.FAILED)
            channels_used = list(set(d.channel for d in alert.deliveries if d.status == AlertStatus.SENT))

            # ── 9. Store artifacts ────────────────────────────────────────
            session.store_artifact("alert", {
                "alert_id":       alert.alert_id,
                "severity":       severity.value,
                "risk_level":     risk_level,
                "location":       location,
                "escalation":     escalation.value,
                "subscribers":    len(subscribers),
                "sent":           sent_count,
                "failed":         failed_count,
                "channels":       channels_used,
            })
            session.store_artifact("safety_message", public_content.body_text)

            summary = (
                f"Alert {alert.alert_id}: {severity.value} severity for {location}. "
                f"Sent to {sent_count}/{len(subscribers)} subscribers via "
                f"{', '.join(channels_used) or 'no channels'}. "
                f"Escalation: {escalation.value}."
            )

            logger.info(f"[AlertingAgent] {summary}")

            return AlertingResult(
                session_id=session_id,
                status="success" if sent_count > 0 else "partial",
                alert=alert,
                total_subscribers=len(subscribers),
                sent_count=sent_count,
                failed_count=failed_count,
                channels_used=channels_used,
                escalation=escalation,
                summary=summary,
                warnings=warnings,
                errors=errors,
            )

        except Exception as exc:
            logger.exception(f"[AlertingAgent] Unhandled error: {exc}")
            return AlertingResult(
                session_id=session_id,
                status="failed",
                errors=[str(exc)],
            )

    # ── Manual alert trigger (from API endpoint) ──────────────────────────

    async def trigger_manual_alert(
        self,
        location:   str,
        risk_level: str,
        message:    Optional[str] = None,
        latitude:   Optional[float] = None,
        longitude:  Optional[float] = None,
    ) -> AlertingResult:
        """
        Triggers an alert manually (e.g., from the API or admin dashboard).
        Bypasses the scheduler cooldown.
        """
        session_id = f"manual-{str(uuid.uuid4())[:8]}"

        severity = {
            "LOW": AlertSeverity.YELLOW,
            "MEDIUM": AlertSeverity.YELLOW,
            "HIGH": AlertSeverity.ORANGE,
            "CRITICAL": AlertSeverity.RED,
        }.get(risk_level.upper(), AlertSeverity.ORANGE)

        # Find subscribers
        if latitude is not None and longitude is not None:
            subscribers = self._subscribers.find_by_location(latitude, longitude, risk_level)
        else:
            subscribers = self._subscribers.list_all()

        # Compose content
        content = await self._composer.compose(
            severity=severity,
            risk_data={"flood_probability": 0.8, "risk_level": risk_level, "confidence": 0.6},
            location_data={"location": location, "latitude": latitude, "longitude": longitude,
                           "emergency_number": "112"},
            for_authority=False,
        )

        if message:
            content.body_text = message

        # Create alert
        alert = Alert(
            alert_id=f"manual-{str(uuid.uuid4())[:8]}",
            session_id=session_id,
            severity=severity,
            risk_level=risk_level,
            location=location,
            latitude=latitude,
            longitude=longitude,
            content=content,
            escalation=EscalationLevel.LEVEL_1,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        # Deliver
        delivery_coros = []
        for sub in subscribers:
            delivery_coros.extend(self._build_delivery_coros(sub, content))

        if delivery_coros:
            results = await asyncio.gather(*delivery_coros, return_exceptions=True)
            for result in results:
                if isinstance(result, DeliveryRecord):
                    alert.deliveries.append(result)

        sent_count = sum(1 for d in alert.deliveries if d.status == AlertStatus.SENT)
        channels_used = list(set(d.channel for d in alert.deliveries if d.status == AlertStatus.SENT))

        return AlertingResult(
            session_id=session_id,
            status="success" if sent_count > 0 else "partial",
            alert=alert,
            total_subscribers=len(subscribers),
            sent_count=sent_count,
            failed_count=len(alert.deliveries) - sent_count,
            channels_used=channels_used,
            summary=f"Manual alert sent to {sent_count} subscribers for {location}.",
        )

    # ── Delivery coroutine builder ────────────────────────────────────────

    def _build_delivery_coros(
        self,
        subscriber: Subscriber,
        content:    AlertContent,
    ) -> List:
        """Builds async delivery coroutines for all channels of a subscriber."""
        coros = []

        for ch in subscriber.channels:
            ch_lower = ch.lower()
            if ch_lower == "email" and subscriber.email:
                coros.append(self._email.send(
                    to=subscriber.email,
                    content=content,
                    subscriber_id=subscriber.id,
                ))
            elif ch_lower == "sms" and subscriber.phone:
                coros.append(self._sms.send(
                    to=subscriber.phone,
                    content=content,
                    subscriber_id=subscriber.id,
                ))
            elif ch_lower == "push" and subscriber.push_token:
                coros.append(self._push.send(
                    token=subscriber.push_token,
                    content=content,
                    subscriber_id=subscriber.id,
                ))

        return coros
