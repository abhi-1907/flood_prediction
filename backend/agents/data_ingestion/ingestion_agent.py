"""
Data Ingestion Agent – Top-level coordinator for the entire data gathering pipeline.

Responsibilities:
  1. Receive raw user input (query text, uploaded file bytes, session context)
  2. Validate and profile the input schema (SchemaValidator)
  3. Identify data gaps and plan which external sources to fetch (SourceIdentifier)
  4. Execute all required fetchers in parallel (asyncio.gather)
  5. Merge all collected data into a unified DataFrame (DataMerger)
  6. Run quality assessment and emit GO/NO-GO decision
  7. Store the merged dataset in the session as an artifact
  8. Return a structured IngestionResult to the Orchestrator

This agent is registered in the ToolRegistry under the name "data_ingestion"
and is called via `session`-aware dependency injection from the Orchestrator.
"""

from __future__ import annotations

import asyncio
import io
from typing import Any, Dict, List, Optional

import pandas as pd

from agents.data_ingestion.data_merger import DataMerger
from agents.data_ingestion.ingestion_schemas import (
    DataCategory,
    DataSourceType,
    FetchResult,
    IngestionResult,
    SchemaReport,
)
from agents.data_ingestion.schema_validator import SchemaValidator
from agents.data_ingestion.source_identifier import SourceIdentifier
from agents.data_ingestion.fetchers.openmeteo_fetcher import OpenMeteoFetcher
from agents.data_ingestion.fetchers.terrain_fetcher import TerrainFetcher
from agents.data_ingestion.fetchers.hydro_fetcher import HydroFetcher
from agents.data_ingestion.fetchers.gov_dataset_fetcher import GovDatasetFetcher
from agents.orchestration.memory import Session
from services.gemini_service import GeminiService, get_gemini_service
from utils.logger import logger


class DataIngestionAgent:
    """
    Coordinates the full data ingestion pipeline for one orchestration session.

    All fetchers are instantiated once and re-used across calls (stateless).
    """

    def __init__(self, gemini_service: Optional[GeminiService] = None) -> None:
        self._gemini    = gemini_service or get_gemini_service()
        self._validator = SchemaValidator()
        self._identifier = SourceIdentifier(self._gemini)
        self._merger     = DataMerger()

        # Fetchers (stateless — can be shared)
        self._meteo_fetcher = OpenMeteoFetcher()
        self._terrain_fetcher = TerrainFetcher()
        self._hydro_fetcher   = HydroFetcher()
        self._gov_fetcher     = GovDatasetFetcher()

    # ── Public interface (called by Orchestrator via ToolRegistry) ────────

    async def run(
        self,
        session: Session,
        query:                Optional[str]   = None,
        uploaded_file_bytes:  Optional[bytes] = None,
        days_back:            int             = 30,
        **kwargs,
    ) -> IngestionResult:
        """
        Main agent entry point.

        Args:
            session:             Active orchestration session (contains context).
            query:               User's free-text query (used when no file given).
            uploaded_file_bytes: Raw bytes from a CSV/JSON upload (optional).
            days_back:           Historical window to fetch (default 30 days).

        Returns:
            IngestionResult with status, schema report, and quality assessment.
            The merged DataFrame is stored in session.artifacts["raw_dataset"].
        """
        session_id = session.session_id
        context    = session.context
        warnings:  List[str] = []
        errors:    List[str] = []

        logger.info(f"[DataIngestionAgent] Starting ingestion for session {session_id}")

        # ── Step 1: Validate / profile incoming data ──────────────────────
        schema_report = await self._validate_input(
            session,
            query=query or session.user_query,
            file_bytes=uploaded_file_bytes or session.get_artifact("uploaded_file_bytes"),
        )
        session.store_artifact("schema_report", schema_report)
        logger.info(
            f"[DataIngestionAgent] Schema: {schema_report.detected_category} | "
            f"rows={schema_report.row_count} | missing={schema_report.missing_categories}"
        )

        # ── Step 2: Identify gaps and build fetch plan ────────────────────
        plan = await self._identifier.identify(
            schema_report=schema_report,
            user_query=query or session.user_query,
            context=context,
        )
        session.store_artifact("ingestion_plan", plan.summary())

        if not plan.tasks:
            logger.info("[DataIngestionAgent] No external fetches needed — user data is complete.")
            # If user gave a complete dataset, use it directly
            user_df = session.get_artifact("user_dataframe")
            if user_df is not None:
                session.store_artifact("raw_dataset", user_df)
                return IngestionResult(
                    session_id=session_id,
                    status="success",
                    schema_report=schema_report,
                    sources_used=[DataSourceType.USER_CSV],
                    rows_collected=len(user_df),
                    columns=list(user_df.columns),
                )

        # ── Step 3: Execute fetchers in parallel ──────────────────────────
        fetch_results = await self._run_fetchers(plan, context, days_back)

        # Include user-uploaded data as a FetchResult
        user_upload_result = session.get_artifact("user_fetch_result")
        if user_upload_result is not None:
            fetch_results.insert(0, user_upload_result)

        # ── Step 4: Quality assessment ────────────────────────────────────
        quality = await self._identifier.assess_quality(fetch_results, schema_report)
        session.store_artifact("ingestion_quality", quality)
        logger.info(f"[DataIngestionAgent] Quality: {quality['go_no_go']} | {quality}")

        if quality["go_no_go"] == "NO_GO":
            errors.append("Insufficient data collected for flood prediction.")
            return IngestionResult(
                session_id=session_id,
                status="failed",
                schema_report=schema_report,
                sources_used=[r.source for r in fetch_results if r.success],
                warnings=quality.get("warnings", []),
                errors=errors,
            )

        warnings.extend(quality.get("warnings", []))

        # ── Step 5: Merge all collected data ──────────────────────────────
        lat = context.get("latitude")
        lon = context.get("longitude")
        merged_df = self._merger.merge(
            fetch_results, primary_lat=lat, primary_lon=lon
        )

        if merged_df.empty:
            errors.append("Data merge produced an empty dataset.")
            return IngestionResult(
                session_id=session_id,
                status="failed",
                schema_report=schema_report,
                errors=errors,
            )

        # ── Step 6: Store merged dataset as session artifact ──────────────
        session.store_artifact("raw_dataset", merged_df)
        logger.info(
            f"[DataIngestionAgent] Stored raw_dataset: "
            f"{len(merged_df)} rows × {len(merged_df.columns)} cols"
        )

        status = "success" if not warnings else "partial"

        return IngestionResult(
            session_id=session_id,
            status=status,
            schema_report=schema_report,
            sources_used=[r.source for r in fetch_results if r.success],
            missing_categories=schema_report.missing_categories,
            rows_collected=len(merged_df),
            columns=list(merged_df.columns),
            warnings=warnings,
            errors=errors,
        )

    # ── Input validation ──────────────────────────────────────────────────

    async def _validate_input(
        self,
        session: Session,
        query: str,
        file_bytes: Optional[bytes],
    ) -> SchemaReport:
        """Detects input type and runs the appropriate validator."""

        if file_bytes:
            # Try CSV first, fallback to JSON
            try:
                schema = self._validator.validate_csv(file_bytes)
                # Parse and store user's DataFrame
                df = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
                session.store_artifact("user_dataframe", df)
                session.store_artifact(
                    "user_fetch_result",
                    FetchResult(
                        source=DataSourceType.USER_CSV,
                        category=schema.detected_category,
                        success=True,
                        data=df,
                        columns=list(df.columns),
                        row_count=len(df),
                    ),
                )
                return schema
            except Exception:
                pass

            try:
                import json
                data = json.loads(file_bytes.decode("utf-8", errors="ignore"))
                schema = self._validator.validate_json(data)
                df = pd.json_normalize(data) if isinstance(data, dict) else pd.DataFrame(data)
                session.store_artifact("user_dataframe", df)
                session.store_artifact(
                    "user_fetch_result",
                    FetchResult(
                        source=DataSourceType.USER_JSON,
                        category=schema.detected_category,
                        success=True,
                        data=df,
                        columns=list(df.columns),
                        row_count=len(df),
                    ),
                )
                return schema
            except Exception:
                pass

        # No file — plain text query
        return self._validator.validate_text(query)

    # ── Parallel fetcher execution ─────────────────────────────────────────

    async def _run_fetchers(
        self,
        plan: Any,
        context: Dict[str, Any],
        days_back: int,
    ) -> List[FetchResult]:
        """Executes all planned fetch tasks concurrently with asyncio.gather."""
        lat = context.get("latitude")
        lon = context.get("longitude")
        location  = context.get("location", "")
        state     = context.get("state", "")

        # Build one coroutine per fetch task
        coros = []
        for task in plan.tasks:
            params = dict(task.params)

            # Inject lat/lon from session context if not in task params
            if lat and "latitude" not in params:
                params["latitude"] = lat
            if lon and "longitude" not in params:
                params["longitude"] = lon
            params["days_back"] = days_back

            if task.source == DataSourceType.OPEN_METEO:
                coros.append(("open_meteo", self._meteo_fetcher.fetch(**params)))
            elif task.source == DataSourceType.TERRAIN:
                coros.append(("terrain", self._terrain_fetcher.fetch(**params)))
            elif task.source == DataSourceType.HYDROLOGICAL:
                coros.append(("hydro", self._hydro_fetcher.fetch(**params)))
            elif task.source == DataSourceType.GOV_DATASET:
                gov_params = {**params, "location": location, "state": state}
                coros.append(("gov", self._gov_fetcher.fetch(**gov_params)))

        if not coros:
            logger.warning("[DataIngestionAgent] No fetch coroutines to execute.")
            return []

        logger.info(
            f"[DataIngestionAgent] Launching {len(coros)} parallel fetches: "
            + ", ".join(name for name, _ in coros)
        )

        # Run all concurrently, capture exceptions without crashing
        results: List[FetchResult] = []
        raw_results = await asyncio.gather(
            *[coro for _, coro in coros],
            return_exceptions=True,
        )

        for (name, _), res in zip(coros, raw_results):
            if isinstance(res, Exception):
                logger.error(f"[DataIngestionAgent] Fetcher '{name}' raised: {res}")
                results.append(
                    FetchResult(
                        source=DataSourceType.UNKNOWN,
                        category=DataCategory.UNKNOWN,
                        success=False,
                        error=str(res),
                    )
                )
            else:
                results.append(res)

        successful = sum(1 for r in results if r.success)
        logger.info(
            f"[DataIngestionAgent] {successful}/{len(results)} fetches succeeded."
        )
        return results
