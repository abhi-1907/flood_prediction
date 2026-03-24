"""
Recommendation Agent package — public API.
"""

from agents.recommendation.recommendation_agent import RecommendationAgent
from agents.recommendation.recommendation_schemas import (
    LocationContext,
    Recommendation,
    RecommendationCategory,
    RecommendationResult,
    ResourceAllocation,
    UrgencyLevel,
    UserProfile,
    UserType,
)
from agents.recommendation.user_profiler import UserProfiler
from agents.recommendation.location_context import LocationContextBuilder
from agents.recommendation.recommendation_engine import RecommendationEngine

__all__ = [
    "RecommendationAgent",
    # Schemas
    "LocationContext",
    "Recommendation",
    "RecommendationCategory",
    "RecommendationResult",
    "ResourceAllocation",
    "UrgencyLevel",
    "UserProfile",
    "UserType",
    # Core classes
    "UserProfiler",
    "LocationContextBuilder",
    "RecommendationEngine",
]
