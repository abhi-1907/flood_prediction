"""
Alerting Agent schemas — Pydantic models for alert lifecycle management.

Extends the base alert_models.py with delivery tracking, escalation,
and templated alert content.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Enumerations ──────────────────────────────────────────────────────────────

class AlertSeverity(str, Enum):
    GREEN  = "GREEN"     # Normal — no alert
    YELLOW = "YELLOW"    # Watch — be prepared
    ORANGE = "ORANGE"    # Warning — take action
    RED    = "RED"       # Emergency — immediate action


class AlertStatus(str, Enum):
    PENDING   = "pending"
    SENT      = "sent"
    DELIVERED = "delivered"
    FAILED    = "failed"
    ESCALATED = "escalated"
    CANCELLED = "cancelled"
    SKIPPED   = "skipped"   # Channel disabled / no provider configured


class EscalationLevel(str, Enum):
    NONE    = "none"
    LEVEL_1 = "level_1"    # Initial alert
    LEVEL_2 = "level_2"    # Repeated alert + additional channels
    LEVEL_3 = "level_3"    # Escalate to authorities


# ── Subscriber ────────────────────────────────────────────────────────────────

class Subscriber(BaseModel):
    """A registered alert subscriber."""
    id:             str
    name:           str
    email:          Optional[str]   = None
    phone:          Optional[str]   = None
    push_token:     Optional[str]   = None
    location:       str
    latitude:       Optional[float] = None
    longitude:      Optional[float] = None
    radius_km:      float           = 25.0
    channels:       List[str]       = Field(default_factory=lambda: ["email"])
    min_risk_level: str             = "MEDIUM"
    is_authority:   bool            = False
    is_active:      bool            = True
    created_at:     str             = ""


# ── Alert content ─────────────────────────────────────────────────────────────

class AlertContent(BaseModel):
    """Rendered alert content ready for delivery."""
    subject:        str
    body_text:      str                        # Plain text (SMS / fallback)
    body_html:      Optional[str]   = None     # HTML for email
    push_title:     Optional[str]   = None     # Push notification title
    push_body:      Optional[str]   = None     # Push notification body
    webhook_payload: Optional[Dict[str, Any]] = None
    metadata:       Dict[str, Any]  = Field(default_factory=dict)


# ── Delivery record ──────────────────────────────────────────────────────────

class DeliveryRecord(BaseModel):
    """Tracks one delivery attempt to one subscriber on one channel."""
    subscriber_id:  str
    channel:        str                        # "email" / "sms" / "push" / "webhook"
    status:         AlertStatus     = AlertStatus.PENDING
    sent_at:        Optional[str]   = None
    error:          Optional[str]   = None
    retry_count:    int             = 0


# ── Alert object ──────────────────────────────────────────────────────────────

class Alert(BaseModel):
    """A single alert event with content and delivery tracking."""
    alert_id:       str
    session_id:     str
    severity:       AlertSeverity
    risk_level:     str
    location:       str
    latitude:       Optional[float] = None
    longitude:      Optional[float] = None
    flood_probability: float        = 0.0
    content:        Optional[AlertContent] = None
    deliveries:     List[DeliveryRecord] = Field(default_factory=list)
    escalation:     EscalationLevel = EscalationLevel.LEVEL_1
    created_at:     str             = ""
    expires_at:     Optional[str]   = None


# ── Full alerting result ──────────────────────────────────────────────────────

class AlertingResult(BaseModel):
    """Final output returned by AlertingAgent to the Orchestrator."""
    session_id:     str
    status:         str                              # "success" | "partial" | "failed"
    alert:          Optional[Alert]     = None
    total_subscribers: int              = 0
    sent_count:     int                 = 0
    failed_count:   int                 = 0
    channels_used:  List[str]           = Field(default_factory=list)
    escalation:     EscalationLevel     = EscalationLevel.NONE
    summary:        Optional[str]       = None
    warnings:       List[str]           = Field(default_factory=list)
    errors:         List[str]           = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True
