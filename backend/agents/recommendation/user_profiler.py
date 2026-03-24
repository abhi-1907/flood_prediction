"""
User Profiler – Infers user type, needs, and accessibility from session context.

Determines:
  1. User type (public / authority / engineer / responder / researcher)
  2. Vulnerability factors (elderly, children, disability, mobility)
  3. Housing situation (floor level, near river, vehicle access)
  4. Language preference
  5. Communication channel suitability (SMS, app, email)

Uses a combination of:
  - Explicit fields from session context
  - LLM inference from the user's original query
  - Rule-based keyword detection
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from agents.recommendation.recommendation_schemas import UserProfile, UserType
from services.gemini_service import GeminiService
from utils.logger import logger


# ── Keyword-based detection ───────────────────────────────────────────────────

AUTHORITY_KEYWORDS = {
    "official", "collector", "district", "authority", "sdma", "ndma",
    "government", "ndrf", "police", "admin", "commissioner", "deploy",
    "evacuation plan", "resource allocation", "emergency response",
}

ENGINEER_KEYWORDS = {
    "engineer", "infrastructure", "dam", "embankment", "drainage",
    "spillway", "culvert", "bridge", "canal", "levee", "sluice",
}

RESPONDER_KEYWORDS = {
    "rescue", "ndrf", "fire", "ambulance", "search and rescue",
    "humanitarian", "relief", "first responder", "medical team",
}

RESEARCHER_KEYWORDS = {
    "research", "dataset", "model", "accuracy", "statistical",
    "study", "journal", "publication", "analysis", "thesis",
}

VULNERABILITY_KEYWORDS = {
    "elderly": {"elderly", "old", "senior", "grandparent", "aged"},
    "children": {"child", "children", "infant", "baby", "school"},
    "disability": {"disabled", "wheelchair", "blind", "deaf", "disability"},
}


class UserProfiler:
    """
    Builds a UserProfile from session context and optional LLM inference.

    Usage:
        profiler = UserProfiler(gemini_service)
        profile  = await profiler.build_profile(session_context)
    """

    def __init__(self, gemini_service: Optional[GeminiService] = None) -> None:
        self._gemini = gemini_service

    async def build_profile(
        self,
        context: Dict[str, Any],
    ) -> UserProfile:
        """
        Infers a complete UserProfile from the session context.

        Args:
            context: Session context dict (location, user_type, original_query, etc.)

        Returns:
            Populated UserProfile.
        """
        # Start with explicit fields from context
        profile = UserProfile(
            location=context.get("location"),
            latitude=context.get("latitude"),
            longitude=context.get("longitude"),
            language=context.get("language", "en"),
        )

        # Detect user type
        profile.user_type = self._detect_user_type(context)

        # Detect vulnerability flags
        query = context.get("original_query", "").lower()
        profile.has_elderly   = self._check_keywords(query, VULNERABILITY_KEYWORDS["elderly"])
        profile.has_children  = self._check_keywords(query, VULNERABILITY_KEYWORDS["children"])
        profile.has_disability= self._check_keywords(query, VULNERABILITY_KEYWORDS["disability"])

        # Explicit overrides from context
        if context.get("user_type"):
            try:
                profile.user_type = UserType(context["user_type"])
            except ValueError:
                pass

        if context.get("near_river") is not None:
            profile.near_river = bool(context["near_river"])

        if context.get("floor_level") is not None:
            profile.floor_level = int(context["floor_level"])

        # LLM-enhanced profiling (if query is ambiguous)
        if self._gemini and self._is_ambiguous(context):
            profile = await self._llm_enhance(profile, context)

        logger.info(
            f"[UserProfiler] Profile: type={profile.user_type} | "
            f"location={profile.location} | "
            f"vulnerable={profile.has_elderly or profile.has_children or profile.has_disability}"
        )
        return profile

    # ── User type detection ───────────────────────────────────────────────

    def _detect_user_type(self, context: Dict[str, Any]) -> UserType:
        """Rule-based user type detection from query keywords."""
        query = context.get("original_query", "").lower()

        if self._check_keywords(query, AUTHORITY_KEYWORDS):
            return UserType.AUTHORITY
        if self._check_keywords(query, RESPONDER_KEYWORDS):
            return UserType.RESPONDER
        if self._check_keywords(query, ENGINEER_KEYWORDS):
            return UserType.ENGINEER
        if self._check_keywords(query, RESEARCHER_KEYWORDS):
            return UserType.RESEARCHER

        return UserType.PUBLIC

    # ── LLM enhancement ───────────────────────────────────────────────────

    async def _llm_enhance(
        self,
        profile: UserProfile,
        context: Dict[str, Any],
    ) -> UserProfile:
        """Uses Gemini to refine the user profile when the query is ambiguous."""
        try:
            prompt = f"""
Analyze this flood-related query and infer the user profile.

Query: "{context.get('original_query', '')}"
Location: {context.get('location', 'unknown')}

Return ONLY a JSON object:
{{
  "user_type": "<public | authority | engineer | responder | researcher>",
  "near_river": <true | false>,
  "has_elderly": <true | false>,
  "has_children": <true | false>,
  "floor_level": <0-10>,
  "is_mobile": <true | false>
}}
"""
            data = await self._gemini.generate_json(prompt, use_fast_model=True)
            if data and isinstance(data, dict):
                try:
                    profile.user_type = UserType(data.get("user_type", profile.user_type))
                except ValueError:
                    pass
                profile.near_river  = data.get("near_river", profile.near_river)
                profile.has_elderly = data.get("has_elderly", profile.has_elderly)
                profile.has_children= data.get("has_children", profile.has_children)
                profile.floor_level = data.get("floor_level", profile.floor_level)
                profile.is_mobile   = data.get("is_mobile", profile.is_mobile)
        except Exception as exc:
            logger.warning(f"[UserProfiler] LLM enhancement failed: {exc}")
        return profile

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _check_keywords(text: str, keywords: set) -> bool:
        return any(kw in text for kw in keywords)

    @staticmethod
    def _is_ambiguous(context: Dict[str, Any]) -> bool:
        """Returns True if the query doesn't clearly indicate user type."""
        query = context.get("original_query", "")
        return len(query) > 20 and not context.get("user_type")
