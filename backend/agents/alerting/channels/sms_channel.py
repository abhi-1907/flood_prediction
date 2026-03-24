"""
SMS Channel – No-op stub (SMS provider removed).

Twilio was removed because it is a paid service and no comparable free
alternative for production SMS delivery exists.

The channel logs the alert text to the application log so operators can
see what *would* have been sent, and returns a SKIPPED DeliveryRecord so
the rest of the alerting pipeline (email, push, webhook) continues
unaffected.

To restore SMS in the future, replace _send_sms() with any provider's
HTTP API call and set the required credentials in .env.
"""

from __future__ import annotations

from agents.alerting.alerting_schemas import AlertContent, AlertStatus, DeliveryRecord
from utils.logger import logger


class SMSChannel:
    """
    No-op SMS delivery channel.

    Always returns AlertStatus.SKIPPED so the alerting pipeline does not
    fail. If SMS credentials are added later, replace this stub with a
    real provider implementation.
    """

    def __init__(self, dry_run: bool = False) -> None:
        # dry_run kept for API compatibility with other channels
        self._dry_run = dry_run
        logger.debug(
            "[SMSChannel] Initialised in no-op mode — SMS alerts are disabled."
        )

    async def send(
        self,
        to: str,
        content: AlertContent,
        subscriber_id: str = "",
        max_retries: int = 3,  # kept for API compatibility
    ) -> DeliveryRecord:
        """
        Logs the SMS alert text and returns SKIPPED.

        Args:
            to:            Recipient phone number (unused).
            content:       AlertContent — body_text is logged for visibility.
            subscriber_id: For tracking.
            max_retries:   Unused (API compatibility).

        Returns:
            DeliveryRecord with status=SKIPPED.
        """
        phone = self._normalise_phone(to) if to else "(no number)"
        message = content.body_text[:300]

        logger.info(
            f"[SMSChannel] SKIPPED (no SMS provider configured) → {phone} | "
            f"Message preview: {message[:80]}{'...' if len(message) > 80 else ''}"
        )

        return DeliveryRecord(
            subscriber_id=subscriber_id,
            channel="sms",
            status=AlertStatus.SKIPPED,
            error="SMS disabled — no provider configured. "
                  "Add a provider (e.g. AWS SNS free tier) to enable SMS alerts.",
        )

    @staticmethod
    def _normalise_phone(phone: str) -> str:
        """Normalises to E.164 format (kept for potential future use)."""
        phone = phone.strip().replace(" ", "").replace("-", "")
        if not phone.startswith("+"):
            if phone.startswith("0"):
                phone = "+91" + phone[1:]    # Indian default
            elif len(phone) == 10:
                phone = "+91" + phone
            else:
                phone = "+" + phone
        return phone
