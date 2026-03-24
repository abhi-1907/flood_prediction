"""
Email Channel – Sends flood alert emails via SMTP.

Supports:
  - HTML + plain text multipart emails
  - TLS/SSL SMTP connections
  - Async sending via asyncio.to_thread
  - Retry with configurable attempts
  - Dry-run mode for testing
"""

from __future__ import annotations

import asyncio
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

from agents.alerting.alerting_schemas import AlertContent, AlertStatus, DeliveryRecord
from utils.logger import logger


class EmailChannel:
    """
    SMTP-based email delivery channel.

    Configuration via environment variables:
      SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_FROM

    Usage:
        channel = EmailChannel()
        record  = await channel.send(to="user@email.com", content=alert_content)
    """

    def __init__(self, dry_run: bool = False) -> None:
        from config import settings
        self._host     = settings.SMTP_HOST
        self._port     = settings.SMTP_PORT
        self._user     = settings.SMTP_USER
        self._password = settings.SMTP_PASSWORD
        self._from     = settings.SMTP_FROM or settings.SMTP_USER
        self._dry_run  = dry_run

    async def send(
        self,
        to:           str,
        content:      AlertContent,
        subscriber_id: str = "",
        max_retries:  int = 3,
    ) -> DeliveryRecord:
        """
        Sends an alert email.

        Args:
            to:           Recipient email address.
            content:      AlertContent with subject, body_text, body_html.
            subscriber_id: For tracking.
            max_retries:  Number of retry attempts.

        Returns:
            DeliveryRecord with status.
        """
        record = DeliveryRecord(
            subscriber_id=subscriber_id,
            channel="email",
            status=AlertStatus.PENDING,
        )

        if not to:
            record.status = AlertStatus.FAILED
            record.error  = "No email address provided"
            return record

        if self._dry_run:
            logger.info(f"[EmailChannel] DRY RUN → {to}: {content.subject}")
            record.status = AlertStatus.SENT
            return record

        if not self._user or not self._password:
            logger.warning("[EmailChannel] SMTP credentials not configured — using dry run")
            record.status = AlertStatus.SENT
            record.error  = "DRY RUN (no SMTP credentials)"
            return record

        for attempt in range(1, max_retries + 1):
            try:
                await asyncio.to_thread(
                    self._send_smtp, to, content
                )
                record.status     = AlertStatus.SENT
                record.retry_count = attempt - 1
                from datetime import datetime, timezone
                record.sent_at = datetime.now(timezone.utc).isoformat()
                logger.info(f"[EmailChannel] Sent to {to} (attempt {attempt})")
                return record

            except Exception as exc:
                logger.warning(
                    f"[EmailChannel] Attempt {attempt}/{max_retries} to {to} failed: {exc}"
                )
                record.retry_count = attempt
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)   # Exponential backoff

        record.status = AlertStatus.FAILED
        record.error  = f"Failed after {max_retries} attempts"
        return record

    def _send_smtp(self, to: str, content: AlertContent) -> None:
        """Synchronous SMTP send."""
        msg = MIMEMultipart("alternative")
        msg["Subject"] = content.subject
        msg["From"]    = self._from
        msg["To"]      = to

        # Plain text part
        msg.attach(MIMEText(content.body_text, "plain", "utf-8"))

        # HTML part (preferred)
        if content.body_html:
            msg.attach(MIMEText(content.body_html, "html", "utf-8"))

        with smtplib.SMTP(self._host, self._port) as server:
            server.starttls()
            server.login(self._user, self._password)
            server.sendmail(self._from, to, msg.as_string())
