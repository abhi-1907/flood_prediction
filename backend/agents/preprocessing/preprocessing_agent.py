"""
Preprocessing Agent – Top-level coordinator for the entire data preprocessing pipeline.

Execution pipeline (in order):
  1. Drop non-feature columns (country, name, week, terrain_type, flood_occurred)
  2. StrategySelector    : LLM+rule analysis → PreprocessingStrategy
  3. RowDiscardHandler   : Remove truly unreliable rows first
  4. MissingValueHandler : Impute remaining nulls per column strategy
  5. OutlierHandler      : Detect and resolve outliers per strategy
  6. FeatureEngineer     : Add derived features (rainfall intensity, terrain wetness, etc.)
  7. Normalizer          : Scale numeric features; store fitted scalers
  8. AuditLogger.compile : Compile full audit report

All intermediate + final artifacts are stored in the session:
  - "strategy"            : PreprocessingStrategy (plan)
  - "processed_dataset"   : Final cleaned/scaled DataFrame (for tabular models)
  - "fitted_scalers"      : Dict of fitted scaler objects
  - "preprocessing_audit" : Full PreprocessingAudit report

Real dataset columns expected:
  Weather  : rain_mm_weekly, temp_c_mean, rh_percent_mean, wind_ms_mean, rain_mm_monthly
  Hydro    : dam_count_50km, dist_major_river_km, waterbody_nearby
  Terrain  : lat, lon, elevation_m, slope_degree, terrain_type_encoded
  Dropped  : country, name, week, terrain_type, flood_occurred (target — not a feature)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from agents.preprocessing.audit_logger import AuditLogger
from agents.preprocessing.preprocessing_schemas import PreprocessingResult, PreprocessingStrategy
from agents.preprocessing.strategy_selector import StrategySelector
from agents.preprocessing.cleaners.missing_value_handler import MissingValueHandler
from agents.preprocessing.cleaners.outlier_handler import OutlierHandler
from agents.preprocessing.cleaners.row_discard_handler import RowDiscardHandler
from agents.preprocessing.transformers.normalizer import Normalizer
from agents.preprocessing.transformers.feature_engineer import FeatureEngineer
from agents.orchestration.memory import Session
from services.gemini_service import GeminiService, get_gemini_service
from utils.logger import logger
from agents.prediction.model_registry import ALL_FEATURES


# Columns dropped before any preprocessing (non-feature / ID / target columns)
DROP_BEFORE_PREPROCESS = ["country", "name", "week", "terrain_type", "flood_occurred"]


class PreprocessingAgent:
    """
    Orchestrates the complete data preprocessing pipeline.

    All sub-components are injected and stateless — the agent can be reused
    across multiple sessions without side effects.
    """

    def __init__(self, gemini_service: Optional[GeminiService] = None) -> None:
        self._gemini    = gemini_service or get_gemini_service()
        self._selector  = StrategySelector(self._gemini)
        self._discarder = RowDiscardHandler()
        self._imputer   = MissingValueHandler()
        self._outlier   = OutlierHandler()
        self._engineer  = FeatureEngineer()
        self._normalizer= Normalizer()

    # ── Public interface (called by Orchestrator via ToolRegistry) ────────

    async def run(
        self,
        session:  Session,
        data:     Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> PreprocessingResult:
        """
        Main agent entry point.

        Args:
            session: Active orchestration session (used to read/write artifacts).
            data:    Optional DataFrame override. If None, reads "raw_dataset"
                     from session artifacts.
            **kwargs: Passed through for future extensibility.

        Returns:
            PreprocessingResult with status, shapes, audit, and feature columns.
            Stores processed DataFrame in session.artifacts["processed_dataset"].
        """
        session_id = session.session_id
        warnings:  List[str] = []
        errors:    List[str] = []

        logger.info(f"[PreprocessingAgent] Starting preprocessing for session {session_id}")

        # ── 0. Retrieve raw dataset ───────────────────────────────────────
        df = data if data is not None else session.get_artifact("raw_dataset")

        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            return PreprocessingResult(
                session_id=session_id,
                status="failed",
                errors=["No raw_dataset found in session. Run DataIngestionAgent first."],
            )

        if not isinstance(df, pd.DataFrame):
            try:
                df = pd.DataFrame(df)
            except Exception as exc:
                return PreprocessingResult(
                    session_id=session_id,
                    status="failed",
                    errors=[f"Cannot convert data to DataFrame: {exc}"],
                )

        input_rows, input_cols = df.shape
        logger.info(f"[PreprocessingAgent] Input: {input_rows} rows × {input_cols} cols")

        # ── 1. Drop non-feature columns (ID / target / temporal) ─────────
        drop_now = [c for c in DROP_BEFORE_PREPROCESS if c in df.columns]
        if drop_now:
            df = df.drop(columns=drop_now)
            logger.info(f"[PreprocessingAgent] Dropped non-feature columns: {drop_now}")
        # Also remove any remaining string/object columns
        string_cols = df.select_dtypes(include="object").columns.tolist()
        if string_cols:
            df = df.drop(columns=string_cols)
            logger.info(f"[PreprocessingAgent] Dropped string cols: {string_cols}")

        # ── 2. Initialise audit logger ────────────────────────────────────
        audit_logger = AuditLogger(session_id=session_id)
        audit_logger.log_shape("preprocessing_agent", df, "After dropping non-feature columns")

        try:
            # ── 3. Select preprocessing strategy ─────────────────────────
            strategy = await self._selector.select(
                df=df,
                session_context=session.context,
                data_quality=session.get_artifact("ingestion_quality"),
            )
            session.store_artifact("strategy", strategy)
            audit_logger.log(
                step="strategy_selector",
                action=f"Strategy selected: char={strategy.dataset_character} | "
                       f"impute={strategy.global_imputation} | "
                       f"outlier={strategy.global_outlier} | "
                       f"scale={strategy.global_scaling} | "
                       f"ts={strategy.format_time_series}",
            )
            if strategy.rationale:
                audit_logger.log(
                    step="strategy_selector",
                    action=f"LLM rationale: {strategy.rationale}",
                )

            # ── 4. Drop strategy-flagged columns ──────────────────────────
            cols_to_drop = [c for c in strategy.columns_to_drop if c in df.columns]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
                audit_logger.log(
                    step="column_dropper",
                    action=f"Dropped {len(cols_to_drop)} low-value columns: {cols_to_drop}",
                    rows_affected=0,
                )
                logger.info(f"[PreprocessingAgent] Dropped columns: {cols_to_drop}")

            # ── 5. Row discard (remove obviously bad rows FIRST) ──────────
            df = self._discarder.handle(df, strategy, audit_logger)

            if df.empty:
                return PreprocessingResult(
                    session_id=session_id,
                    status="failed",
                    errors=["All rows discarded after row-discard step."],
                    audit=audit_logger.compile(
                        pd.DataFrame(columns=df.columns), df
                    ),
                )

            # ── 6. Impute missing values ───────────────────────────────────
            df = self._imputer.handle(df, strategy, audit_logger)

            # ── 7. Handle outliers ─────────────────────────────────────────
            df = self._outlier.handle(df, strategy, audit_logger)

            # ── 8. Feature engineering ────────────────────────────────────
            df = self._engineer.engineer(df, strategy, audit_logger)

            # ── 9. Normalise / scale ──────────────────────────────────────
            df, fitted_scalers = self._normalizer.fit_transform(df, strategy, audit_logger)
            session.store_artifact("fitted_scalers", fitted_scalers)

            # ── 10. Filter to training features + store ───────────────────
            # Keep the 13 real training features first, then any valid engineered
            # extras. flood_risk_proxy is EDA-only — excluded from model input.
            training_cols = [c for c in ALL_FEATURES if c in df.columns]
            extra_cols    = [
                c for c in df.columns
                if c not in ALL_FEATURES
                and c not in ("flood_occurred", "flood_risk_proxy")
            ]
            df = df[training_cols + extra_cols]
            session.store_artifact("processed_dataset", df)
            logger.info(
                f"[PreprocessingAgent] Stored processed_dataset: "
                f"{df.shape[0]} rows × {df.shape[1]} cols | "
                f"training_feats={len(training_cols)}/{len(ALL_FEATURES)}"
            )



            # ── 11. Compile audit report ──────────────────────────────────
            # Keep a copy of original shape for audit
            original_df = session.get_artifact("raw_dataset") or pd.DataFrame()
            audit = audit_logger.compile(
                input_df=original_df if isinstance(original_df, pd.DataFrame) else pd.DataFrame(),
                output_df=df,
            )
            session.store_artifact("preprocessing_audit", audit)

            # Feature columns = all numeric columns in final dataset
            feature_cols = list(df.select_dtypes(include="number").columns)

            output_rows, output_cols = df.shape
            status = "success" if not warnings else "partial"

            logger.info(
                f"[PreprocessingAgent] Complete: {input_rows}×{input_cols} → "
                f"{output_rows}×{output_cols} | status={status}"
            )

            return PreprocessingResult(
                session_id=session_id,
                status=status,
                strategy=strategy,
                audit=audit,
                input_rows=input_rows,
                input_cols=input_cols,
                output_rows=output_rows,
                output_cols=output_cols,
                warnings=warnings,
                errors=errors,
                feature_columns=feature_cols,
            )

        except Exception as exc:
            logger.exception(f"[PreprocessingAgent] Unhandled error: {exc}")
            errors.append(str(exc))
            return PreprocessingResult(
                session_id=session_id,
                status="failed",
                input_rows=input_rows,
                input_cols=input_cols,
                errors=errors,
            )
