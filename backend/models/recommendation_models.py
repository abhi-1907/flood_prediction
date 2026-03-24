"""
Pydantic response schemas for the Recommendation API.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class UserType(str, Enum):
    GENERAL_PUBLIC  = "general_public"
    AUTHORITY       = "authority"
    FIRST_RESPONDER = "first_responder"
    ENGINEER        = "engineer"
    RESEARCHER      = "researcher"


class UrgencyLevel(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


class RecommendationRequest(BaseModel):
    """Request to generate recommendations for a session."""
    session_id: Optional[str] = None
    location:   Optional[str] = None
    risk_level: Optional[str] = None
    user_type:  UserType = UserType.GENERAL_PUBLIC
    latitude:   Optional[float] = None
    longitude:  Optional[float] = None
    # Additional context
    has_elderly:    bool = False
    has_children:   bool = False
    has_disability: bool = False
    vehicle_access: bool = True


class RecommendationItem(BaseModel):
    """A single structured recommendation."""
    category:     str                      # evacuation | safety | medical | resource
    priority:     str                      # immediate | soon | monitor
    title:        str
    description:  str
    action_steps: List[str] = []
    target_group: Optional[str] = None


class ResourceAllocation(BaseModel):
    """Resource allocation for authorities."""
    resource_type: str                     # rescue_boats | relief_camps | medical | NDRF
    quantity:      Optional[int] = None
    location:      Optional[str] = None
    priority:      str = "high"
    notes:         Optional[str] = None


class RecommendationResponse(BaseModel):
    """Full recommendation response for a session."""
    session_id:      str
    status:          str
    location:        Optional[str]  = None
    user_type:       UserType       = UserType.GENERAL_PUBLIC
    risk_level:      Optional[str]  = None
    urgency:         Optional[UrgencyLevel] = None
    recommendations: List[RecommendationItem] = []
    resource_plan:   List[ResourceAllocation] = []
    safety_message:  Optional[str]  = None    # SMS-friendly
    authority_brief: Optional[str]  = None    # Formal brief
    summary:         Optional[str]  = None
    safe_zones:      List[str] = []
    emergency_contacts: List[Dict[str, str]] = []
    warnings:        List[str] = []
    errors:          List[str] = []
