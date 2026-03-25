"""
Recommendation Agent schemas — Pydantic models shared across the pipeline.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Enumerations ──────────────────────────────────────────────────────────────

class UserType(str, Enum):
    PUBLIC     = "public"           # Citizens / general public
    AUTHORITY  = "authority"        # District admin / NDMA / SDMA officials
    ENGINEER   = "engineer"         # Civil / hydraulic engineers
    RESPONDER  = "responder"        # NDRF / fire / rescue teams
    RESEARCHER = "researcher"       # Academics / scientists


class UrgencyLevel(str, Enum):
    INFORMATIONAL = "informational"   # No immediate danger
    ADVISORY      = "advisory"        # Be prepared
    WARNING       = "warning"         # Take action soon
    EMERGENCY     = "emergency"       # Act NOW


class RecommendationCategory(str, Enum):
    EVACUATION    = "evacuation"
    SAFETY        = "safety"
    RESOURCE      = "resource_allocation"
    INFRASTRUCTURE= "infrastructure"
    HEALTH        = "health"
    AGRICULTURE   = "agriculture"
    COMMUNICATION = "communication"
    SHELTER       = "shelter"
    WATER_SUPPLY  = "water_supply"
    POST_FLOOD    = "post_flood"


# ── User profile ──────────────────────────────────────────────────────────────

class UserProfile(BaseModel):
    """Describes who is receiving the recommendation."""
    user_type:       UserType       = UserType.PUBLIC
    language:        str            = "en"
    location:        Optional[str]  = None
    latitude:        Optional[float]= None
    longitude:       Optional[float]= None
    has_disability:  bool           = False
    has_elderly:     bool           = False
    has_children:    bool           = False
    is_mobile:       bool           = True        # Can travel
    has_vehicle:     bool           = True
    near_river:      bool           = False
    floor_level:     int            = 0           # 0 = ground floor


# ── Location context ──────────────────────────────────────────────────────────

class LocationContext(BaseModel):
    """Geographic & infrastructure context for the target area."""
    location_name:   Optional[str]  = None
    state:           Optional[str]  = None
    district:        Optional[str]  = None
    latitude:        Optional[float]= None
    longitude:       Optional[float]= None
    elevation_m:     Optional[float]= None
    is_coastal:      bool           = False
    is_flood_plain:  bool           = False
    is_urban:        bool           = False
    nearest_shelter: Optional[str]  = None
    nearest_hospital:Optional[str]  = None
    river_name:      Optional[str]  = None
    dam_nearby:      bool           = False
    emergency_number:str            = "112"       # India: 112 / NDRF: 011-24363260


# ── Single recommendation ────────────────────────────────────────────────────

class Recommendation(BaseModel):
    """One actionable recommendation item."""
    id:           int
    category:     RecommendationCategory
    urgency:      UrgencyLevel
    title:        str
    description:  str                              # 2–3 sentence explanation
    action_steps: List[str] = Field(default_factory=list)   # Numbered steps
    
    # Regional language support
    title_regional:        Optional[str] = None
    description_regional:  Optional[str] = None
    action_steps_regional: List[str]     = Field(default_factory=list)

    for_user_type: UserType  = UserType.PUBLIC
    priority:     int        = 1                   # 1 = highest


# ── Resource allocation (for authorities) ─────────────────────────────────────

class ResourceAllocation(BaseModel):
    """Suggested resource deployment for disaster management authorities."""
    resource_type:  str              # e.g. "rescue_boats", "medical_teams"
    quantity:       int
    deploy_to:      str              # Location / area name
    urgency:        UrgencyLevel
    rationale:      str
    estimated_cost_inr: Optional[float] = None


# ── Full recommendation result ────────────────────────────────────────────────

class RecommendationResult(BaseModel):
    """Final output returned by RecommendationAgent to the Orchestrator."""
    session_id:       str
    status:           str                                # "success" | "partial" | "failed"
    risk_level:       Optional[str]       = None
    urgency:          UrgencyLevel        = UrgencyLevel.INFORMATIONAL
    recommendations:  List[Recommendation]= Field(default_factory=list)
    resource_plan:    List[ResourceAllocation] = Field(default_factory=list)
    summary:          Optional[str]       = None         # LLM-generated summary
    safety_message:   Optional[str]       = None         # Short SMS-friendly message
    authority_brief:  Optional[str]       = None         # Formal brief for officials
    user_profile:     Optional[UserProfile] = None
    location_context: Optional[LocationContext] = None
    warnings:         List[str]           = Field(default_factory=list)
    errors:           List[str]           = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True
