"""
Alerting Agent package — public API.
"""

from agents.alerting.alerting_agent import AlertingAgent
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
from agents.alerting.channels import EmailChannel, SMSChannel, PushChannel, WebhookChannel

__all__ = [
    "AlertingAgent",
    # Schemas
    "Alert",
    "AlertContent",
    "AlertingResult",
    "AlertSeverity",
    "AlertStatus",
    "DeliveryRecord",
    "EscalationLevel",
    "Subscriber",
    # Core classes
    "SubscriberManager",
    "AlertComposer",
    "AlertScheduler",
    # Channels
    "EmailChannel",
    "SMSChannel",
    "PushChannel",
    "WebhookChannel",
]
