"""
Pydantic schemas for alert subscription and triggering.
"""
from pydantic import BaseModel, EmailStr
from typing import Optional, List
from enum import Enum


class AlertChannel(str, Enum):
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"


class AlertSubscribeRequest(BaseModel):
    name: str
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    push_token: Optional[str] = None
    location: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    radius_km: float = 25.0
    channels: List[AlertChannel] = [AlertChannel.EMAIL]
    min_risk_level: str = "MODERATE"    # Only alert at this level or above


class AlertTriggerRequest(BaseModel):
    location: str
    risk_level: str
    message: Optional[str] = None      # Optional manual override message
