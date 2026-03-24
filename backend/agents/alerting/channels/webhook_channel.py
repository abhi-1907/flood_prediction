"""
Webhook Channel – Delivers flood alerts to external systems via HTTP POST.

Supports:
  - Custom webhook URLs (Slack, Discord, custom dashboards)
  - Configurable headers and auth tokens
  - JSON payload delivery
  - Retry with exponential backoff
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

import httpx

from agents.alerting.alerting_schemas import AlertContent, AlertStatus, DeliveryRecord
from utils.logger import logger


class WebhookChannel:
    """
    HTTP webhook delivery channel.

    Configuration via environment variables:
      WEBHOOK_URL, WEBHOOK_AUTH_TOKEN (optional Bearer token)

    Usage:
        channel = WebhookChannel(url="https://hooks.slack.com/...")
        record  = await channel.send(content=alert_content)
    """

    def __init__(
        self,
        url:       Optional[str] = None,
        auth_token: Optional[str] = None,
        dry_run:   bool = False,
    ) -> None:
        from config import settings
        self._url   = url or settings.WEBHOOK_URL
        self._token = auth_token or settings.WEBHOOK_AUTH_TOKEN
        self._dry_run = dry_run

    async def send(
        self,
        content:       AlertContent,
        subscriber_id: str = "webhook",
        url_override:  Optional[str] = None,
        max_retries:   int = 3,
    ) -> DeliveryRecord:
        """
        Sends alert payload to a webhook URL.

        Args:
            content:       AlertContent — uses webhook_payload.
            subscriber_id: For tracking.
            url_override:  Override the default URL.
            max_retries:   Number of retry attempts.

        Returns:
            DeliveryRecord with status.
        """
        record = DeliveryRecord(
            subscriber_id=subscriber_id,
            channel="webhook",
            status=AlertStatus.PENDING,
        )

        url = url_override or self._url
        if not url:
            record.status = AlertStatus.FAILED
            record.error  = "No webhook URL configured"
            return record

        payload = content.webhook_payload or {
            "text": content.body_text,
            "subject": content.subject,
        }

        if self._dry_run:
            logger.info(f"[WebhookChannel] DRY RUN → {url}")
            record.status = AlertStatus.SENT
            return record

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        for attempt in range(1, max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    resp = await client.post(url, json=payload, headers=headers)
                    resp.raise_for_status()

                record.status     = AlertStatus.SENT
                record.retry_count = attempt - 1
                from datetime import datetime, timezone
                record.sent_at = datetime.now(timezone.utc).isoformat()
                logger.info(
                    f"[WebhookChannel] Sent to {url} (status {resp.status_code})"
                )
                return record

            except Exception as exc:
                logger.warning(
                    f"[WebhookChannel] Attempt {attempt}/{max_retries} failed: {exc}"
                )
                record.retry_count = attempt
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)

        record.status = AlertStatus.FAILED
        record.error  = f"Failed after {max_retries} attempts"
        return record
