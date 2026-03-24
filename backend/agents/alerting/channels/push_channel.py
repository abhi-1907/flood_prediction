"""
Push Channel – Sends flood alert push notifications via Firebase Cloud Messaging.

Supports:
  - Firebase Admin SDK (FCM) for push delivery
  - Individual and topic-based messaging
  - Async sending via asyncio.to_thread
  - Dry-run mode for testing
  - Data payload for rich notification on client
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from agents.alerting.alerting_schemas import AlertContent, AlertStatus, DeliveryRecord
from utils.logger import logger


class PushChannel:
    """
    Firebase Cloud Messaging push notification delivery channel.

    Configuration:
      FIREBASE_CREDENTIALS_PATH — path to service account JSON
      (or initialize Firebase Admin elsewhere in app startup)

    Usage:
        channel = PushChannel()
        record  = await channel.send(token="fcm-token-xxx", content=alert_content)
    """

    def __init__(self, dry_run: bool = False) -> None:
        from config import settings
        self._dry_run     = dry_run
        self._initialized = False
        self._creds_path  = settings.FIREBASE_CREDENTIALS_PATH

    async def send(
        self,
        token:         str,
        content:       AlertContent,
        subscriber_id: str = "",
        max_retries:   int = 2,
    ) -> DeliveryRecord:
        """
        Sends a push notification.

        Args:
            token:         FCM device registration token.
            content:       AlertContent — uses push_title and push_body.
            subscriber_id: For tracking.
            max_retries:   Number of retry attempts.

        Returns:
            DeliveryRecord with status.
        """
        record = DeliveryRecord(
            subscriber_id=subscriber_id,
            channel="push",
            status=AlertStatus.PENDING,
        )

        if not token:
            record.status = AlertStatus.FAILED
            record.error  = "No push token provided"
            return record

        if self._dry_run:
            logger.info(
                f"[PushChannel] DRY RUN → {token[:20]}...: {content.push_title}"
            )
            record.status = AlertStatus.SENT
            return record

        for attempt in range(1, max_retries + 1):
            try:
                await asyncio.to_thread(
                    self._send_fcm, token, content
                )
                record.status     = AlertStatus.SENT
                record.retry_count = attempt - 1
                from datetime import datetime, timezone
                record.sent_at = datetime.now(timezone.utc).isoformat()
                logger.info(f"[PushChannel] Sent to {token[:20]}... (attempt {attempt})")
                return record

            except Exception as exc:
                logger.warning(
                    f"[PushChannel] Attempt {attempt}/{max_retries} failed: {exc}"
                )
                record.retry_count = attempt
                if attempt < max_retries:
                    await asyncio.sleep(1)

        record.status = AlertStatus.FAILED
        record.error  = f"Failed after {max_retries} attempts"
        return record

    async def send_to_topic(
        self,
        topic:   str,
        content: AlertContent,
    ) -> DeliveryRecord:
        """Sends a push notification to all devices subscribed to a topic."""
        record = DeliveryRecord(
            subscriber_id=f"topic:{topic}",
            channel="push",
            status=AlertStatus.PENDING,
        )

        if self._dry_run:
            logger.info(f"[PushChannel] DRY RUN → topic:{topic}: {content.push_title}")
            record.status = AlertStatus.SENT
            return record

        try:
            await asyncio.to_thread(
                self._send_fcm_topic, topic, content
            )
            record.status = AlertStatus.SENT
            return record
        except Exception as exc:
            record.status = AlertStatus.FAILED
            record.error  = str(exc)
            return record

    def _send_fcm(self, token: str, content: AlertContent) -> None:
        """Synchronous FCM send to individual device."""
        try:
            import firebase_admin
            from firebase_admin import messaging

            if not self._initialized:
                self._init_firebase()

            message = messaging.Message(
                notification=messaging.Notification(
                    title=content.push_title or content.subject,
                    body=content.push_body or content.body_text[:200],
                ),
                data={
                    "severity":   str(content.metadata.get("severity", "")),
                    "risk_level": str(content.metadata.get("risk_level", "")),
                    "location":   str(content.metadata.get("location", "")),
                    "flood_probability": str(content.metadata.get("flood_probability", "")),
                    "click_action": "OPEN_FLOOD_ALERT",
                },
                token=token,
                android=messaging.AndroidConfig(
                    priority="high",
                    notification=messaging.AndroidNotification(
                        sound="emergency",
                        channel_id="flood_alerts",
                        priority="max",
                    ),
                ),
                apns=messaging.APNSConfig(
                    payload=messaging.APNSPayload(
                        aps=messaging.Aps(
                            sound="default",
                            badge=1,
                            content_available=True,
                        ),
                    ),
                ),
            )
            messaging.send(message)
        except ImportError:
            logger.warning("[PushChannel] firebase-admin not installed — skipping")
            raise

    def _send_fcm_topic(self, topic: str, content: AlertContent) -> None:
        """Synchronous FCM send to topic."""
        try:
            import firebase_admin
            from firebase_admin import messaging

            if not self._initialized:
                self._init_firebase()

            message = messaging.Message(
                notification=messaging.Notification(
                    title=content.push_title or content.subject,
                    body=content.push_body or content.body_text[:200],
                ),
                topic=topic,
            )
            messaging.send(message)
        except ImportError:
            logger.warning("[PushChannel] firebase-admin not installed — skipping")
            raise

    def _init_firebase(self) -> None:
        """Initializes Firebase Admin SDK."""
        try:
            import firebase_admin
            from firebase_admin import credentials

            if not firebase_admin._apps:
                if self._creds_path:
                    cred = credentials.Certificate(self._creds_path)
                    firebase_admin.initialize_app(cred)
                else:
                    firebase_admin.initialize_app()
            self._initialized = True
        except Exception as exc:
            logger.warning(f"[PushChannel] Firebase init failed: {exc}")
