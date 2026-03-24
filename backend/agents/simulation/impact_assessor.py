"""
Impact Assessor – Estimates population, economic, agriculture, and infrastructure
impact from the simulated flood scenario.

Dimensions:
  1. Population    : People affected / displaced / at risk
  2. Agriculture   : Crop area inundated, estimated crop damage
  3. Infrastructure: Roads, bridges, buildings potentially affected
  4. Economic      : Estimated loss in INR crore
  5. Environmental : Ecological damage and water contamination risk

Uses Indian census-based population density estimates and NDMA cost guidelines.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from agents.simulation.simulation_schemas import (
    FloodSeverity,
    FloodZone,
    ImpactCategory,
    ImpactEstimate,
    ScenarioParameters,
    SimulationTimeStep,
)
from services.gemini_service import GeminiService
from utils.logger import logger


# ── Indian population density (persons/km²) by area type ─────────────────────

POP_DENSITY = {
    "urban":        5000,
    "residential":  2500,
    "agricultural": 300,
    "forest":       50,
    "barren":       20,
    "default":      800,
}

# NDMA damage cost estimates (INR crore per km² fully inundated)
DAMAGE_COST_PER_KM2 = {
    "urban":        25.0,
    "residential":  12.0,
    "agricultural": 3.0,
    "forest":       0.5,
    "default":      5.0,
}

# Crop damage (INR lakh per hectare, moderate inundation)
CROP_DAMAGE_PER_HA = 0.8   # INR lakh


class ImpactAssessor:
    """
    Estimates multi-dimensional flood impact from simulated zones.

    Usage:
        assessor = ImpactAssessor(gemini_service)
        impacts  = await assessor.assess(zones, scenario, timeline)
    """

    def __init__(self, gemini_service: Optional[GeminiService] = None) -> None:
        self._gemini = gemini_service

    async def assess(
        self,
        zones:    List[FloodZone],
        scenario: ScenarioParameters,
        timeline: List[SimulationTimeStep],
        cell_km:  float = 0.5,
    ) -> List[ImpactEstimate]:
        """
        Computes multi-dimensional impact estimates.

        Args:
            zones:    List of FloodZone cells with inundation depths.
            scenario: The simulation scenario.
            timeline: Hourly timeline from ScenarioEngine.
            cell_km:  Grid cell size in km.

        Returns:
            List of ImpactEstimate objects across all categories.
        """
        impacts: List[ImpactEstimate] = []
        cell_area_km2 = cell_km * cell_km

        # Filter to inundated zones only
        inundated = [z for z in zones if z.depth_m > 0.01]
        if not inundated:
            logger.info("[ImpactAssessor] No inundated zones — zero impact.")
            return impacts

        total_area_km2    = len(inundated) * cell_area_km2
        populated_zones   = [z for z in inundated if z.is_populated]
        agricultural_zones= [z for z in inundated if z.land_use == "agricultural"]

        # ── 1. Population impact ──────────────────────────────────────────
        pop_affected = 0
        pop_displaced = 0
        for z in populated_zones:
            density  = POP_DENSITY.get(z.land_use, POP_DENSITY["default"])
            pop_cell = density * cell_area_km2

            if z.severity in (FloodSeverity.SEVERE, FloodSeverity.EXTREME):
                pop_displaced += pop_cell
                pop_affected  += pop_cell
            elif z.severity == FloodSeverity.MODERATE:
                pop_affected  += pop_cell
                pop_displaced += pop_cell * 0.3
            else:
                pop_affected  += pop_cell * 0.1

        impacts.append(ImpactEstimate(
            category=ImpactCategory.POPULATION,
            metric="people_affected",
            value=round(pop_affected),
            unit="persons",
            confidence=0.4,
            description=f"Estimated {round(pop_affected):,} people in inundated areas.",
        ))
        impacts.append(ImpactEstimate(
            category=ImpactCategory.POPULATION,
            metric="people_displaced",
            value=round(pop_displaced),
            unit="persons",
            confidence=0.4,
            description=f"Estimated {round(pop_displaced):,} people may need evacuation.",
        ))

        # ── 2. Agriculture impact ─────────────────────────────────────────
        agri_area_ha = len(agricultural_zones) * cell_area_km2 * 100  # ha
        crop_damage_lakh = agri_area_ha * CROP_DAMAGE_PER_HA
        impacts.append(ImpactEstimate(
            category=ImpactCategory.AGRICULTURE,
            metric="crop_area_affected",
            value=round(agri_area_ha, 1),
            unit="hectares",
            confidence=0.35,
            description=f"Estimated {agri_area_ha:.0f} hectares of cropland inundated.",
        ))
        impacts.append(ImpactEstimate(
            category=ImpactCategory.AGRICULTURE,
            metric="crop_damage_estimate",
            value=round(crop_damage_lakh, 2),
            unit="INR lakh",
            confidence=0.3,
            description=f"Estimated crop damage: ₹{crop_damage_lakh:.1f} lakh.",
        ))

        # ── 3. Infrastructure impact ──────────────────────────────────────
        buildings_affected = round(pop_affected / 4.5)   # ~4.5 persons per household
        road_km_affected   = total_area_km2 * 1.5         # Approximate road density

        impacts.append(ImpactEstimate(
            category=ImpactCategory.INFRASTRUCTURE,
            metric="buildings_affected",
            value=buildings_affected,
            unit="buildings",
            confidence=0.35,
            description=f"Approximately {buildings_affected:,} buildings in flood zone.",
        ))
        impacts.append(ImpactEstimate(
            category=ImpactCategory.INFRASTRUCTURE,
            metric="road_km_affected",
            value=round(road_km_affected, 1),
            unit="km",
            confidence=0.3,
            description=f"Estimated {road_km_affected:.1f} km of roads may be waterlogged.",
        ))

        # ── 4. Economic impact ────────────────────────────────────────────
        total_economic_crore = 0.0
        for z in inundated:
            cost  = DAMAGE_COST_PER_KM2.get(z.land_use, DAMAGE_COST_PER_KM2["default"])
            sev_factor = {
                FloodSeverity.MINOR: 0.1,
                FloodSeverity.MODERATE: 0.4,
                FloodSeverity.SEVERE: 0.8,
                FloodSeverity.EXTREME: 1.0,
            }.get(z.severity, 0.3)
            total_economic_crore += cost * cell_area_km2 * sev_factor

        impacts.append(ImpactEstimate(
            category=ImpactCategory.ECONOMIC,
            metric="estimated_damage",
            value=round(total_economic_crore, 2),
            unit="INR crore",
            confidence=0.25,
            description=f"Estimated total economic damage: ₹{total_economic_crore:.1f} crore.",
        ))

        # ── 5. Environmental impact ───────────────────────────────────────
        env_risk_score = min(100, total_area_km2 * 5 + len(
            [z for z in inundated if z.severity in (FloodSeverity.SEVERE, FloodSeverity.EXTREME)]
        ) * 3)

        impacts.append(ImpactEstimate(
            category=ImpactCategory.ENVIRONMENTAL,
            metric="contamination_risk",
            value=round(env_risk_score, 1),
            unit="score (0-100)",
            confidence=0.3,
            description=(
                f"Water contamination risk score: {env_risk_score:.0f}/100. "
                "Floodwater may carry sewage, chemicals, and debris."
            ),
        ))

        # LLM-enhanced summary (optional)
        if self._gemini:
            llm_impact = await self._llm_enhance(impacts, scenario)
            if llm_impact:
                impacts.append(llm_impact)

        logger.info(
            f"[ImpactAssessor] {len(impacts)} impact estimates | "
            f"pop_affected={pop_affected:.0f} | area={total_area_km2:.1f}km² | "
            f"economic=₹{total_economic_crore:.1f}cr"
        )
        return impacts

    # ── LLM enhancement ───────────────────────────────────────────────────

    async def _llm_enhance(
        self,
        impacts:  List[ImpactEstimate],
        scenario: ScenarioParameters,
    ) -> Optional[ImpactEstimate]:
        """Uses Gemini to add a qualitative risk note."""
        try:
            summary = "\n".join(
                f"- {i.metric}: {i.value} {i.unit}" for i in impacts
            )
            prompt = f"""
Given these flood simulation impact estimates for {scenario.location or 'the target area'}:
{summary}

Scenario: {scenario.name} | Rainfall: {scenario.rainfall_mm}mm × {scenario.rainfall_days} days

Write a 2-sentence qualitative assessment of the most critical risk.
Focus on life-safety and time-sensitive actions. Be specific.
Return ONLY the assessment text, no JSON.
"""
            narrative = await self._gemini.generate(prompt, use_fast_model=True)
            if narrative:
                return ImpactEstimate(
                    category=ImpactCategory.POPULATION,
                    metric="qualitative_assessment",
                    value=0,
                    unit="narrative",
                    confidence=0.5,
                    description=narrative.strip()[:500],
                )
        except Exception as exc:
            logger.warning(f"[ImpactAssessor] LLM enhancement failed: {exc}")
        return None
