"""
Recommendation Engine – Core LLM + rule-based recommendation generator.

Two-phase approach:
  Phase 1: Rule-based safety recommendations (instant, no-LLM fallback)
  Phase 2: LLM-generated contextual recommendations (Gemini, personalised)

Also generates:
  - Resource allocation plans (for authority users)
  - SMS-friendly safety messages
  - Formal authority briefs
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agents.recommendation.recommendation_schemas import (
    LocationContext,
    Recommendation,
    RecommendationCategory,
    ResourceAllocation,
    UrgencyLevel,
    UserProfile,
    UserType,
)
from services.gemini_service import GeminiService
from utils.logger import logger


# ── Template paths ────────────────────────────────────────────────────────────

TEMPLATES_DIR = Path(__file__).parent / "templates"


class RecommendationEngine:
    """
    Generates flood recommendations using LLM + rule-based approaches.

    Usage:
        engine = RecommendationEngine(gemini_service)
        recs, resources = await engine.generate(
            risk_level="HIGH",
            flood_prob=0.72,
            profile=user_profile,
            location=loc_context,
            env_data={"rainfall_mm": 150, ...}
        )
    """

    def __init__(self, gemini_service: GeminiService) -> None:
        self._gemini = gemini_service

    # ── Public API ─────────────────────────────────────────────────────────

    async def generate(
        self,
        risk_level:     str,
        flood_prob:     float,
        confidence:     float,
        profile:        UserProfile,
        location:       LocationContext,
        env_data:       Dict[str, Any],
    ) -> Tuple[List[Recommendation], List[ResourceAllocation]]:
        """
        Generates recommendations and resource plans.

        Returns:
            (recommendations_list, resource_allocations_list)
        """
        urgency = self._risk_to_urgency(risk_level)

        # Phase 1: rule-based (always runs — instant fallback)
        rule_recs = self._rule_based_recommendations(
            risk_level, flood_prob, urgency, profile, location, env_data
        )

        # Phase 2: LLM-generated (contextual, personalised)
        llm_recs, resources = await self._llm_recommendations(
            risk_level, flood_prob, confidence, urgency, profile, location, env_data
        )

        # Merge: LLM recs take priority; add rule-based ones not covered
        final_recs = self._merge_recommendations(llm_recs, rule_recs)

        logger.info(
            f"[RecommendationEngine] Generated {len(final_recs)} recommendations "
            f"+ {len(resources)} resource allocations | urgency={urgency}"
        )
        return final_recs, resources

    async def generate_safety_message(
        self,
        risk_level:  str,
        flood_prob:  float,
        location:    LocationContext,
        profile:     UserProfile,
    ) -> str:
        """Generates a short SMS-friendly safety message (< 160 chars)."""
        urgency_emoji = {
            "LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"
        }.get(risk_level, "⚠️")

        loc_name = location.location_name or "your area"
        emergency = location.emergency_number

        if risk_level in ("HIGH", "CRITICAL"):
            sms = (
                f"{urgency_emoji} FLOOD {risk_level} for {loc_name}. "
                f"Probability: {flood_prob*100:.0f}%. "
                f"Move to higher ground immediately. "
                f"Emergency: {emergency}"
            )
        elif risk_level == "MEDIUM":
            sms = (
                f"{urgency_emoji} FLOOD ADVISORY for {loc_name}. "
                f"Probability: {flood_prob*100:.0f}%. "
                f"Prepare essential supplies and stay alert. "
                f"Emergency: {emergency}"
            )
        else:
            sms = (
                f"{urgency_emoji} Flood risk LOW for {loc_name}. "
                f"No immediate action needed. Stay informed via local news."
            )

        return sms[:300]

    async def generate_authority_brief(
        self,
        risk_level:  str,
        flood_prob:  float,
        confidence:  float,
        recs:        List[Recommendation],
        resources:   List[ResourceAllocation],
        location:    LocationContext,
    ) -> str:
        """Generates a formal authority brief suitable for official meetings."""
        try:
            prompt = f"""
Write a short formal disaster management brief (150-200 words) for {location.state or 'State'} authorities.

Subject: Flood Risk Assessment — {location.location_name or 'Target Area'}

Key Data:
- Flood probability: {flood_prob*100:.1f}%
- Risk level: {risk_level}
- Model confidence: {confidence*100:.0f}%
- Location: {location.location_name}, {location.district or ''}, {location.state or ''}
- Coastal: {location.is_coastal} | Flood plain: {location.is_flood_plain}
- River: {location.river_name or 'N/A'}

Recommendations count: {len(recs)}
Resource allocations: {len(resources)}

Write in formal government report style. Include:
1. Situation assessment (2 sentences)
2. Immediate actions required (3-4 bullet points)
3. Resource requirements summary (2-3 points)
4. Closing with emergency contact info
"""
            brief = await self._gemini.generate(prompt, use_fast_model=True)
            return brief.strip() if brief else self._fallback_brief(
                risk_level, flood_prob, location
            )
        except Exception as exc:
            logger.warning(f"[RecommendationEngine] Authority brief failed: {exc}")
            return self._fallback_brief(risk_level, flood_prob, location)

    # ── Rule-based recommendations ────────────────────────────────────────

    def _rule_based_recommendations(
        self,
        risk_level: str,
        flood_prob: float,
        urgency:    UrgencyLevel,
        profile:    UserProfile,
        location:   LocationContext,
        env_data:   Dict[str, Any],
    ) -> List[Recommendation]:
        """Generates instant rule-based recommendations (LLM fallback)."""
        recs: List[Recommendation] = []
        idx  = 1

        # ── Evacuation (HIGH / CRITICAL) ──
        if risk_level in ("HIGH", "CRITICAL"):
            evac_steps = [
                "Switch off electricity and gas connections",
                "Pack essential documents (Aadhaar, insurance, medications)",
                "Move to the nearest shelter or higher ground",
            ]
            if profile.has_elderly or profile.has_disability:
                evac_steps.insert(0, "Assist elderly/disabled family members first")
            if not profile.has_vehicle:
                evac_steps.append("Contact emergency services for transport assistance")

            recs.append(Recommendation(
                id=idx, category=RecommendationCategory.EVACUATION,
                urgency=UrgencyLevel.EMERGENCY,
                title="Evacuate to Higher Ground Immediately",
                description=(
                    f"Flood probability is {flood_prob*100:.0f}% in {location.location_name or 'your area'}. "
                    "Move to the nearest elevated shelter without delay. "
                    "Do not attempt to walk or drive through floodwater."
                ),
                action_steps=evac_steps,
                for_user_type=profile.user_type,
                priority=1,
            ))
            idx += 1

        # ── Shelter ──
        if risk_level in ("MEDIUM", "HIGH", "CRITICAL"):
            shelter = location.nearest_shelter or "the nearest community centre or school"
            recs.append(Recommendation(
                id=idx, category=RecommendationCategory.SHELTER,
                urgency=urgency,
                title="Identify and Move to Nearest Shelter",
                description=f"Proceed to {shelter}. Carry essential supplies for 72 hours.",
                action_steps=[
                    f"Go to {shelter}",
                    "Carry food, water, medicines, and phone charger",
                    "Register with local disaster management authorities",
                ],
                for_user_type=profile.user_type,
                priority=2,
            ))
            idx += 1

        # ── Water safety ──
        recs.append(Recommendation(
            id=idx, category=RecommendationCategory.HEALTH,
            urgency=UrgencyLevel.ADVISORY if risk_level != "LOW" else UrgencyLevel.INFORMATIONAL,
            title="Ensure Safe Drinking Water",
            description=(
                "Floodwater contaminates drinking water sources. "
                "Store clean water NOW and boil any water before drinking."
            ),
            action_steps=[
                "Fill clean containers with tap water immediately",
                "Use water purification tablets if available",
                "Avoid consuming floodwater-contaminated food",
            ],
            for_user_type=profile.user_type,
            priority=3,
        ))
        idx += 1

        # ── Communication ──
        recs.append(Recommendation(
            id=idx, category=RecommendationCategory.COMMUNICATION,
            urgency=UrgencyLevel.ADVISORY,
            title="Stay Informed and Contactable",
            description=(
                f"Keep your phone charged. Emergency helpline: {location.emergency_number}. "
                "Follow local disaster management updates."
            ),
            action_steps=[
                "Save emergency numbers on your phone",
                f"Call {location.emergency_number} for immediate help",
                "Monitor All India Radio or local TV for official updates",
                "Share your location with family members",
            ],
            for_user_type=profile.user_type,
            priority=4,
        ))
        idx += 1

        # ── Agriculture (if env_data has soil/crop info or rural area) ──
        if not location.is_urban and risk_level in ("MEDIUM", "HIGH", "CRITICAL"):
            recs.append(Recommendation(
                id=idx, category=RecommendationCategory.AGRICULTURE,
                urgency=urgency,
                title="Protect Crops and Livestock",
                description=(
                    "Move livestock to higher ground. Photograph crops for insurance claims. "
                    "Contact the district agriculture officer for crop damage assessment."
                ),
                action_steps=[
                    "Move livestock to elevated land",
                    "Harvest any ready crops if time permits",
                    "Document damage with photos for insurance",
                ],
                for_user_type=profile.user_type,
                priority=5,
            ))
            idx += 1

        # ── River proximity ──
        if location.river_name or profile.near_river:
            river = location.river_name or "the nearby river"
            recs.append(Recommendation(
                id=idx, category=RecommendationCategory.SAFETY,
                urgency=urgency,
                title=f"Monitor {river} Water Levels",
                description=(
                    f"Stay away from {river} banks. Water levels can rise rapidly "
                    "without warning, especially during heavy rainfall upstream."
                ),
                action_steps=[
                    f"Do not go near {river} banks",
                    "Check CWC flood bulletins (cwc.gov.in)",
                    "Be prepared to evacuate if water rises",
                ],
                for_user_type=profile.user_type,
                priority=2 if risk_level in ("HIGH", "CRITICAL") else 4,
            ))
            idx += 1

        return recs

    # ── LLM-generated recommendations ─────────────────────────────────────

    async def _llm_recommendations(
        self,
        risk_level: str,
        flood_prob: float,
        confidence: float,
        urgency:    UrgencyLevel,
        profile:    UserProfile,
        location:   LocationContext,
        env_data:   Dict[str, Any],
    ) -> Tuple[List[Recommendation], List[ResourceAllocation]]:
        """Generates contextual recommendations via Gemini."""
        resources: List[ResourceAllocation] = []

        # Select the right prompt template
        is_authority = profile.user_type in (UserType.AUTHORITY, UserType.RESPONDER)
        template_file = "authority_prompt.txt" if is_authority else "public_prompt.txt"

        try:
            template = (TEMPLATES_DIR / template_file).read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.warning(f"[RecommendationEngine] Template not found: {template_file}")
            return [], resources

        # Format environmental conditions
        env_lines = "\n".join(
            f"- {k}: {v}" for k, v in env_data.items()
        ) or "- No detailed environmental data available"

        # Fill the template
        prompt = template.format(
            location=location.location_name or "Unknown",
            flood_probability=round(flood_prob * 100, 1),
            risk_level=risk_level,
            urgency=urgency.value,
            confidence=round(confidence * 100, 0),
            environmental_conditions=env_lines,
            has_elderly=profile.has_elderly,
            has_children=profile.has_children,
            has_disability=profile.has_disability,
            near_river=profile.near_river,
            floor_level=profile.floor_level,
            has_vehicle=profile.has_vehicle,
            state=location.state or "Unknown",
            district=location.district or "Unknown",
            is_coastal=location.is_coastal,
            is_flood_plain=location.is_flood_plain,
            is_urban=location.is_urban,
            nearest_shelter=location.nearest_shelter or "nearest community centre",
            emergency_number=location.emergency_number,
            river_name=location.river_name or "N/A",
            dam_nearby=location.dam_nearby,
            models_used="ensemble (XGBoost + Random Forest + LSTM)",
        )

        try:
            raw = await self._gemini.generate_json(prompt, use_fast_model=False)
        except Exception as exc:
            logger.warning(f"[RecommendationEngine] LLM generation failed: {exc}")
            return [], resources

        if not raw:
            return [], resources

        # Parse response
        recs_data = []
        res_data  = []

        if isinstance(raw, list):
            recs_data = raw   # Public prompt returns a list
        elif isinstance(raw, dict):
            recs_data = raw.get("recommendations", [])
            res_data  = raw.get("resource_plan", [])

        llm_recs = self._parse_recommendations(recs_data, profile.user_type)
        resources = self._parse_resources(res_data)

        return llm_recs, resources

    # ── Parsing helpers ───────────────────────────────────────────────────

    @staticmethod
    def _parse_recommendations(
        data:      List[Dict],
        user_type: UserType,
    ) -> List[Recommendation]:
        recs = []
        for i, item in enumerate(data, start=1):
            try:
                category = RecommendationCategory.SAFETY
                try:
                    category = RecommendationCategory(item.get("category", "safety"))
                except ValueError:
                    pass

                urgency = UrgencyLevel.ADVISORY
                try:
                    urgency = UrgencyLevel(item.get("urgency", "advisory"))
                except ValueError:
                    pass

                recs.append(Recommendation(
                    id=i,
                    category=category,
                    urgency=urgency,
                    title=item.get("title", f"Recommendation {i}")[:80],
                    description=item.get("description", "")[:500],
                    action_steps=item.get("action_steps", [])[:6],
                    for_user_type=user_type,
                    priority=int(item.get("priority", i)),
                ))
            except Exception:
                continue
        return recs

    @staticmethod
    def _parse_resources(data: List[Dict]) -> List[ResourceAllocation]:
        resources = []
        for item in data:
            try:
                urgency = UrgencyLevel.ADVISORY
                try:
                    urgency = UrgencyLevel(item.get("urgency", "advisory"))
                except ValueError:
                    pass

                resources.append(ResourceAllocation(
                    resource_type=item.get("resource_type", "unknown"),
                    quantity=int(item.get("quantity", 1)),
                    deploy_to=item.get("deploy_to", "target area"),
                    urgency=urgency,
                    rationale=item.get("rationale", ""),
                    estimated_cost_inr=item.get("estimated_cost_inr"),
                ))
            except Exception:
                continue
        return resources

    # ── Merge strategies ──────────────────────────────────────────────────

    @staticmethod
    def _merge_recommendations(
        llm_recs:  List[Recommendation],
        rule_recs: List[Recommendation],
    ) -> List[Recommendation]:
        """Merges LLM and rule-based recs, preferring LLM but filling gaps."""
        if not llm_recs:
            return rule_recs

        # Categories covered by LLM
        llm_categories = {r.category for r in llm_recs}

        # Add rule-based recs whose category isn't covered by LLM
        merged = list(llm_recs)
        for r in rule_recs:
            if r.category not in llm_categories:
                r.id = len(merged) + 1
                merged.append(r)

        # Re-sort by priority
        merged.sort(key=lambda r: r.priority)

        # Re-number IDs
        for i, r in enumerate(merged, start=1):
            r.id = i

        return merged

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _risk_to_urgency(risk_level: str) -> UrgencyLevel:
        return {
            "LOW":      UrgencyLevel.INFORMATIONAL,
            "MEDIUM":   UrgencyLevel.ADVISORY,
            "HIGH":     UrgencyLevel.WARNING,
            "CRITICAL": UrgencyLevel.EMERGENCY,
        }.get(risk_level, UrgencyLevel.ADVISORY)

    @staticmethod
    def _fallback_brief(risk_level: str, flood_prob: float, location: LocationContext) -> str:
        return (
            f"FLOOD RISK BRIEF — {location.location_name or 'Target Area'}\n"
            f"Risk Level: {risk_level} | Probability: {flood_prob*100:.0f}%\n"
            f"Location: {location.state or 'N/A'}, {location.district or 'N/A'}\n"
            f"Emergency Contact: {location.emergency_number}\n"
            f"Action Required: Deploy resources as per standard operating procedures."
        )
