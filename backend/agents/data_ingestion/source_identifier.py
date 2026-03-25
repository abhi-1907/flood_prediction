"""
Source Identifier – LLM-powered classifier that analyses incoming data and determines:
  1. What data categories are present (rainfall, hydro, terrain, timeseries)
  2. Which required fields are missing or invalid
  3. What external sources need to be queried to fill the gaps
  4. Which fetcher strategy to apply for each missing category

This is the "intelligence" step that bridges SchemaValidator output →
fetcher dispatch decisions. It uses the Schema report + Gemini context-awareness
to reason about what a complete flood-prediction dataset needs.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from agents.data_ingestion.ingestion_schemas import (
    DataCategory,
    DataSourceType,
    FetchResult,
    SchemaReport,
)
from services.gemini_service import GeminiService
from utils.logger import logger


# ── Required fields by category ─────────────────────────────────────────────
# Aligned with the real training features from india_pakistan_flood_balancednew.csv
# Weather:      rain_mm_weekly, temp_c_mean, rh_percent_mean, wind_ms_mean, rain_mm_monthly
# Hydro:        dam_count_50km, dist_major_river_km, waterbody_nearby
# Terrain:      lat, lon, elevation_m, slope_degree, terrain_type_encoded

REQUIRED_FIELDS: Dict[DataCategory, List[str]] = {
    DataCategory.RAINFALL:   ["rain_mm_weekly", "rain_mm_monthly"],
    DataCategory.HYDRO:      ["dist_major_river_km", "dam_count_50km", "waterbody_nearby"],
    DataCategory.TERRAIN:    ["lat", "lon", "elevation_m", "slope_degree"],
    DataCategory.TIMESERIES: ["week", "rain_mm_weekly"],
    DataCategory.MIXED:      ["rain_mm_weekly", "dist_major_river_km", "elevation_m", "slope_degree"],
}

# Source priority per missing category
CATEGORY_TO_SOURCE: Dict[DataCategory, List[DataSourceType]] = {
    DataCategory.RAINFALL:  [DataSourceType.OPEN_METEO, DataSourceType.GOV_DATASET],
    DataCategory.HYDRO:     [DataSourceType.HYDROLOGICAL, DataSourceType.GOV_DATASET],
    DataCategory.TERRAIN:   [DataSourceType.TERRAIN],
    DataCategory.TIMESERIES:[DataSourceType.OPEN_METEO],
}


class SourceIdentifier:
    """
    LLM-powered identifier that decides which external fetchers to invoke.

    Two-phase approach:
      Phase 1 – Rule-based: reads SchemaReport and flags categories that are
                             missing entirely.
      Phase 2 – LLM-based:  uses Gemini to reason about ambiguous gaps and
                             produce a structured fetch plan with rationale.
    """

    def __init__(self, gemini_service: GeminiService) -> None:
        self._gemini = gemini_service

    # ── Public API ────────────────────────────────────────────────────────

    async def identify(
        self,
        schema_report: SchemaReport,
        user_query: str,
        context: Dict[str, Any],
        session=None,   # Optional[Session] — avoids import cycle
    ) -> "IngestionPlan":
        """
        Produces an IngestionPlan describing exactly which fetchers to call.

        If the orchestrator already pre-computed a fetch plan via the unified
        startup call, reads it from session to skip the LLM reasoning call.
        """
        # Fast path: pre-computed fetch plan from unified startup call
        if session is not None:
            precomputed = session.get_artifact("fetch_plan")
            if precomputed and isinstance(precomputed, list):
                logger.info("[SourceIdentifier] Using pre-computed fetch plan (no LLM call).")
                tasks = []
                for item in precomputed:
                    try:
                        task = FetchTask(
                            source=DataSourceType(item.get("source", "open_meteo")),
                            category=DataCategory(item.get("category", "rainfall")),
                            priority=int(item.get("priority", 2)),
                            params=item.get("params", {}),
                            rationale=item.get("rationale", "Pre-computed by startup call."),
                        )
                        tasks.append(task)
                    except (ValueError, KeyError) as e:
                        logger.warning(f"[SourceIdentifier] Bad pre-computed task: {item} — {e}")
                if tasks:
                    return IngestionPlan(
                        tasks=sorted(tasks, key=lambda t: t.priority),
                        context=context,
                        schema_report=schema_report,
                    )

        # Phase 1: Rule-based gap analysis
        rule_gaps = self._rule_based_gaps(schema_report, context)
        logger.info(f"[SourceIdentifier] Rule-based gaps: {[g.value for g in rule_gaps]}")

        # Optimization: If no gaps or perfectly clear rule-based plan, skip LLM
        if not rule_gaps and schema_report.detected_sources:
            logger.info("[SourceIdentifier] No gaps detected — skipping LLM reasoning.")
            return IngestionPlan(tasks=[], context=context, schema_report=schema_report)

        # Phase 2: LLM-enhanced reasoning (only if needed or complex gaps)
        # Note: If we are in "file upload" mode, we often prefer rules to save quota
        plan = await self._llm_reasoning(
            schema_report, user_query, context, rule_gaps
        )
        logger.info(
            f"[SourceIdentifier] Fetch plan: "
            + ", ".join(t.source.value for t in plan.tasks)
        )
        return plan

    async def assess_quality(
        self,
        fetch_results: List[FetchResult],
        schema_report: SchemaReport,
    ) -> Dict[str, Any]:
        """
        After fetchers run, evaluates whether the combined dataset is sufficient
        for flood prediction. Returns a quality dict with a GO/NO-GO decision.
        """
        total_rows = sum(r.row_count for r in fetch_results if r.success)
        successful = [r for r in fetch_results if r.success]
        failed     = [r for r in fetch_results if not r.success]
        categories = list({r.category for r in successful})

        has_rainfall = DataCategory.RAINFALL in categories or DataCategory.TIMESERIES in categories or DataCategory.MIXED in categories
        has_hydro    = DataCategory.HYDRO in categories or DataCategory.MIXED in categories
        has_terrain  = DataCategory.TERRAIN in categories or DataCategory.MIXED in categories

        go_no_go = "GO" if (has_rainfall or has_hydro or has_terrain) and total_rows > 0 else "NO_GO"

        return {
            "go_no_go":        go_no_go,
            "total_rows":      total_rows,
            "categories":      [c.value for c in categories],
            "has_rainfall":    has_rainfall,
            "has_hydro":       has_hydro,
            "has_terrain":     has_terrain,
            "successful_sources": [r.source.value for r in successful],
            "failed_sources":     [r.source.value for r in failed],
            "warnings":        self._generate_warnings(has_rainfall, has_hydro, has_terrain),
        }

    # ── Rule-based gap analysis ───────────────────────────────────────────

    def _rule_based_gaps(
        self,
        report: SchemaReport,
        context: Dict[str, Any],
    ) -> List[DataCategory]:
        """Determines which data categories are absent from BOTH report and context."""
        gaps = list(report.missing_categories)

        # Remove categories that are already satisfied by context variables
        if DataCategory.RAINFALL in gaps:
            if context.get("rain_mm_weekly") is not None or context.get("rain_mm_monthly") is not None:
                gaps.remove(DataCategory.RAINFALL)
        
        if DataCategory.HYDRO in gaps:
            if context.get("dist_major_river_km") is not None or context.get("waterbody_nearby") is not None:
                gaps.remove(DataCategory.HYDRO)
                
        if DataCategory.TERRAIN in gaps:
            if all(context.get(k) is not None for k in ["elevation_m", "slope_degree", "terrain_type"]):
                gaps.remove(DataCategory.TERRAIN)

        # If user gave us zero data in report, it might still be covered by context
        if report.row_count == 0 and not gaps:
            # If everything was in context, we are good.
            pass

        return list(set(gaps))

    # ── LLM reasoning ────────────────────────────────────────────────────

    async def _llm_reasoning(
        self,
        schema_report: SchemaReport,
        user_query: str,
        context: Dict[str, Any],
        rule_gaps: List[DataCategory],
    ) -> "IngestionPlan":
        """Uses Gemini to produce a refined fetch plan with rationale."""

        schema_summary = {
            "category":          schema_report.detected_category.value,
            "rows":              schema_report.row_count,
            "columns":           schema_report.column_count,
            "has_timestamp":     schema_report.has_timestamp,
            "has_location":      schema_report.has_location,
            "missing_categories": [c.value for c in schema_report.missing_categories],
            "invalid_fields": [
                f.name for f in schema_report.fields
                if f.status.value in ("missing", "invalid")
            ],
        }

        # Identify fields already satisfied by the user (manual/textbox)
        context_satisfied = [
            k for k in ["rain_mm_weekly", "rain_mm_monthly", "elevation_m", "slope_degree", "dist_major_river_km", "waterbody_nearby", "terrain_type"]
            if context.get(k) is not None
        ]

        prompt = f"""
You are a data engineering AI for a flood prediction system.

User query: "{user_query}"

Location context:
  - Place: {context.get('location', 'unknown')}
  - Latitude: {context.get('latitude', 'unknown')}
  - Longitude: {context.get('longitude', 'unknown')}

Existing data schema analysis:
{json.dumps(schema_summary, indent=2)}

Manual User Inputs (ALREADY PRESENT): {context_satisfied}

Rule-based gaps identified: {[g.value for g in rule_gaps]}

Available external data sources:
  - open_meteo   : Free rainfall, weather, and historical precipitation (Open-Meteo API)
  - terrain      : Elevation and terrain data (Google Maps Elevation / SRTM DEM)
  - hydrological : River discharge, water levels (India-WRIS / CWC / USGS)
  - gov_dataset  : Government open datasets (data.gov.in, IMD, Kerala SDMA)

Task: Generate a fetch plan as a JSON array. Each item has:
{{
  "source":    "<open_meteo | terrain | hydrological | gov_dataset>",
  "category":  "<rainfall | hydro | terrain | timeseries>",
  "priority":  <1-3, 1=highest>,
  "params":    {{<key-value pairs to pass to the fetcher>}},
  "rationale": "<one-line reason this source is needed>"
}}

Strict Fetch Rules:
1. ONLY include sources that are genuinely needed to fill gaps.
2. If a data category is covered by "Manual User Inputs" above, do NOT include a source for it.
3. If the user provided a file with multiple rows, prefer that data over everything else.
4. If terrain data is missing and location is known, include terrain.
5. Respond ONLY with the JSON array, no prose.
"""
        raw = await self._gemini.generate_json(prompt, use_fast_model=True)
        tasks = []

        if isinstance(raw, list):
            for item in raw:
                try:
                    task = FetchTask(
                        source=DataSourceType(item.get("source", "unknown")),
                        category=DataCategory(item.get("category", "unknown")),
                        priority=int(item.get("priority", 2)),
                        params=item.get("params", {}),
                        rationale=item.get("rationale", ""),
                    )
                    tasks.append(task)
                except (ValueError, KeyError) as e:
                    logger.warning(f"[SourceIdentifier] Bad task item: {item} — {e}")
        else:
            # LLM failed — fall back to rule-based
            logger.warning("[SourceIdentifier] LLM returned non-list; using rule-based fallback.")
            tasks = self._fallback_tasks(rule_gaps, context)

        return IngestionPlan(
            tasks=sorted(tasks, key=lambda t: t.priority),
            context=context,
            schema_report=schema_report,
        )

    # ── Fallback ──────────────────────────────────────────────────────────

    def _fallback_tasks(
        self,
        gaps: List[DataCategory],
        context: Dict[str, Any],
    ) -> List["FetchTask"]:
        """Generates a minimal fetch plan from rule-based gaps without LLM."""
        tasks = []
        lat = context.get("latitude")
        lon = context.get("longitude")

        if DataCategory.RAINFALL in gaps or DataCategory.TIMESERIES in gaps:
            tasks.append(FetchTask(
                source=DataSourceType.OPEN_METEO,
                category=DataCategory.RAINFALL,
                priority=1,
                params={"latitude": lat, "longitude": lon},
                rationale="Rainfall data missing — fetch from Open-Meteo",
            ))
        if DataCategory.HYDRO in gaps:
            tasks.append(FetchTask(
                source=DataSourceType.HYDROLOGICAL,
                category=DataCategory.HYDRO,
                priority=1,
                params={"latitude": lat, "longitude": lon},
                rationale="Hydrological data missing",
            ))
        if DataCategory.TERRAIN in gaps and lat and lon:
            tasks.append(FetchTask(
                source=DataSourceType.TERRAIN,
                category=DataCategory.TERRAIN,
                priority=2,
                params={"latitude": lat, "longitude": lon},
                rationale="Terrain/elevation data missing",
            ))
        return tasks

    # ── Quality warning generation ────────────────────────────────────────

    @staticmethod
    def _generate_warnings(
        has_rainfall: bool,
        has_hydro: bool,
        has_terrain: bool,
    ) -> List[str]:
        warnings = []
        if not has_rainfall:
            warnings.append("No rainfall data — prediction accuracy will be reduced.")
        if not has_hydro:
            warnings.append("No hydrological data — cannot use hydro-specific models.")
        if not has_terrain:
            warnings.append("No terrain data — flood zone simulation will be approximate.")
        return warnings


# ── Supporting data classes ───────────────────────────────────────────────────

class FetchTask:
    """A single fetch task within an IngestionPlan."""
    def __init__(
        self,
        source:    DataSourceType,
        category:  DataCategory,
        priority:  int,
        params:    Dict[str, Any],
        rationale: str,
    ) -> None:
        self.source    = source
        self.category  = category
        self.priority  = priority
        self.params    = params
        self.rationale = rationale

    def __repr__(self) -> str:
        return f"FetchTask(source={self.source.value}, cat={self.category.value}, p={self.priority})"


class IngestionPlan:
    """Ordered list of fetch tasks to execute for a given user request."""
    def __init__(
        self,
        tasks:         List[FetchTask],
        context:       Dict[str, Any],
        schema_report: SchemaReport,
    ) -> None:
        self.tasks         = tasks
        self.context       = context
        self.schema_report = schema_report

    def tasks_for_source(self, source: DataSourceType) -> List[FetchTask]:
        return [t for t in self.tasks if t.source == source]

    def summary(self) -> str:
        return " | ".join(str(t) for t in self.tasks)
