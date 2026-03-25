"""
Strategy Selector – LLM-powered preprocessing decision engine.

Given a raw DataFrame (with profiled statistics), the StrategySelector asks
Gemini to recommend the optimal preprocessing strategy for flood prediction:

  1. Which imputation method to use per column (median, KNN, seasonal, etc.)
  2. How to handle outliers (clip, remove, isolation forest, etc.)
  3. Which scaling approach to apply (standard, robust, log, etc.)
  4. Whether to drop any columns (too sparse, high cardinality, irrelevant)
  5. Whether time-series formatting is needed (for LSTM)
  6. Which features to engineer

The selector also runs a fast rule-based fallback for cases where the LLM
call fails or returns malformed JSON.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from agents.preprocessing.preprocessing_schemas import (
    ColumnStrategy,
    DatasetCharacter,
    ImputationStrategy,
    OutlierStrategy,
    PreprocessingStrategy,
    ScalingStrategy,
)
from services.gemini_service import GeminiService
from utils.logger import logger


# ── Heuristic thresholds ──────────────────────────────────────────────────────

HIGH_NULL_PCT      = 60.0    # Columns above this null % are candidates for drop
LOW_VARIANCE_THR   = 1e-6    # Near-zero variance → likely constant → drop
HIGH_CARDINALITY   = 0.95    # > 95% unique values in a string column → drop
OUTLIER_Z_THR      = 3.5     # Z-score threshold for heuristic outlier flagging

# Core flood-prediction columns from real dataset — never drop or zero-impute these
# Dataset: india_pakistan_flood_balancednew.csv
PROTECTED_COLUMNS = {
    # Geo / location
    "lat", "lon",
    # Weather features
    "rain_mm_weekly", "rain_mm_monthly",
    "temp_c_mean", "rh_percent_mean", "wind_ms_mean",
    # Hydrological features
    "dam_count_50km", "dist_major_river_km", "waterbody_nearby",
    # Terrain features
    "elevation_m", "slope_degree", "terrain_type_encoded",
    # Target
    "flood_occurred",
}


class StrategySelector:
    """
    Recommends an optimal preprocessing strategy via two phases:

    Phase 1 – Rule-based profiling: quick statistical analysis of each column.
    Phase 2 – LLM enhancement:     Gemini reasons about the dataset character
                                    and refines the strategy with domain knowledge.
    """

    def __init__(self, gemini_service: GeminiService) -> None:
        self._gemini = gemini_service

    # ── Public API ────────────────────────────────────────────────────────

    async def select(
        self,
        df: pd.DataFrame,
        session_context: Dict[str, Any],
        data_quality: Optional[Dict[str, Any]] = None,
    ) -> PreprocessingStrategy:
        """
        Produces a full PreprocessingStrategy for the given DataFrame.

        Args:
            df:              The raw merged DataFrame from the ingestion agent.
            session_context: Session context dict (location, user_type, etc.).
            data_quality:    Optional quality assessment from IngestionAgent.

        Returns:
            PreprocessingStrategy with per-column strategies and global settings.
        """
        logger.info(
            f"[StrategySelector] Profiling {df.shape[0]} rows × "
            f"{df.shape[1]} columns"
        )

        # Phase 1: rule-based profiling (no LLM call — saves quota)
        profile = self._profile_dataframe(df)
        rule_strategy = self._rule_based_strategy(df, profile)

        logger.info(
            f"[StrategySelector] Strategy (rule-based): "
            f"char={rule_strategy.dataset_character} | "
            f"impute={rule_strategy.global_imputation} | "
            f"outlier={rule_strategy.global_outlier} | "
            f"scale={rule_strategy.global_scaling}"
        )
        return rule_strategy

    # ── Phase 1: Rule-based profiling ────────────────────────────────────

    def _profile_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Statistical profile of each column for LLM context."""
        profile: Dict[str, Any] = {}

        for col in df.columns:
            series = df[col]
            p: Dict[str, Any] = {
                "dtype":     str(series.dtype),
                "null_pct":  round(series.isna().mean() * 100, 2),
                "unique_pct": round(series.nunique() / max(len(series), 1) * 100, 2),
            }

            if pd.api.types.is_numeric_dtype(series) and series.dtype != bool:
                non_null = series.dropna()
                if len(non_null) > 0:
                    p.update({
                        "mean":   round(float(non_null.mean()), 4),
                        "std":    round(float(non_null.std()), 4),
                        "min":    round(float(non_null.min()), 4),
                        "max":    round(float(non_null.max()), 4),
                        "q25":    round(float(non_null.quantile(0.25)), 4),
                        "q75":    round(float(non_null.quantile(0.75)), 4),
                        "skew":   round(float(non_null.skew()), 4),
                        "outlier_count": int(
                            (np.abs((non_null - non_null.mean()) / (non_null.std() + 1e-9))
                             > OUTLIER_Z_THR).sum()
                        ),
                        "is_near_zero_variance": bool(non_null.var() < LOW_VARIANCE_THR),
                    })
            elif series.dtype == bool:
                # Boolean columns: treat as binary categorical
                p.update({
                    "is_binary": True,
                    "true_pct": round(float(series.mean() * 100), 2),
                    "is_near_zero_variance": bool(series.var() < LOW_VARIANCE_THR),
                })
            profile[col] = p

        # Dataset has 'week' column (YYYY-MM) instead of 'date'
        has_date = any(
            c.lower() in ("date", "week", "time", "timestamp", "datetime")
            for c in df.columns
        )
        profile["__meta__"] = {
            "has_date": has_date,
            "rows": df.shape[0],
            "cols": df.shape[1],
            "dataset_character": "time_series" if has_date else "tabular",
        }
        return profile

    def _rule_based_strategy(
        self,
        df: pd.DataFrame,
        profile: Dict[str, Any],
    ) -> PreprocessingStrategy:
        """Builds a safe default strategy using statistical heuristics alone."""
        has_date = profile["__meta__"]["has_date"]
        char = DatasetCharacter.TIME_SERIES if has_date else DatasetCharacter.TABULAR

        col_strategies: List[ColumnStrategy] = []
        columns_to_drop: List[str] = []

        for col in df.columns:
            if col == "__meta__":
                continue
            p = profile.get(col, {})
            null_pct  = p.get("null_pct", 0)
            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            is_protected = col.lower() in PROTECTED_COLUMNS

            # Drop decision
            should_drop = (
                not is_protected and (
                    null_pct > HIGH_NULL_PCT or
                    p.get("is_near_zero_variance", False) or
                    (not is_numeric and p.get("unique_pct", 0) > HIGH_CARDINALITY * 100)
                )
            )

            if should_drop:
                columns_to_drop.append(col)
                continue

            # Imputation
            if null_pct == 0:
                impute = ImputationStrategy.MEAN   # No nulls — no-op but specify
            elif has_date and null_pct < 30:
                impute = ImputationStrategy.INTERPOLATE
            elif null_pct < 20:
                impute = ImputationStrategy.MEDIAN
            elif null_pct < 50:
                impute = ImputationStrategy.FORWARD_FILL if has_date else ImputationStrategy.KNN
            else:
                impute = ImputationStrategy.MEDIAN

            # Outlier
            outlier_count = p.get("outlier_count", 0)
            if outlier_count > df.shape[0] * 0.1:
                outlier = OutlierStrategy.CLIP
            elif outlier_count > 0:
                outlier = OutlierStrategy.CLIP
            else:
                outlier = OutlierStrategy.NONE

            # Scaling — use robust for high-skew or outlier-rich columns
            skew  = abs(p.get("skew", 0))
            if skew > 2:
                scaling = ScalingStrategy.LOG
            elif outlier_count > 0:
                scaling = ScalingStrategy.ROBUST
            else:
                scaling = ScalingStrategy.NONE

            col_strategies.append(ColumnStrategy(
                column=col,
                imputation=impute,
                outlier_strategy=outlier,
                scaling=scaling,
                drop=False,
            ))

        return PreprocessingStrategy(
            dataset_character=char,
            global_imputation=ImputationStrategy.MEDIAN,
            global_outlier=OutlierStrategy.CLIP,
            global_scaling=ScalingStrategy.NONE,
            column_strategies=col_strategies,
            engineer_features=True,
            format_time_series=has_date,
            columns_to_drop=columns_to_drop,
        )

    # ── Phase 2: LLM enhancement ─────────────────────────────────────────

    async def _llm_enhance(
        self,
        df: pd.DataFrame,
        profile: Dict[str, Any],
        rule_strategy: PreprocessingStrategy,
        context: Dict[str, Any],
    ) -> PreprocessingStrategy:
        """Asks Gemini to review and refine the rule-based strategy."""

        # Compact profile (avoid token overflow)
        compact_profile = {
            col: {k: v for k, v in stats.items()
                  if k in ("null_pct", "outlier_count", "skew", "dtype",
                            "is_near_zero_variance", "min", "max")}
            for col, stats in profile.items()
            if col != "__meta__"
        }

        col_summaries = json.dumps(compact_profile, default=str)
        rule_summary  = json.dumps({
            "dataset_character":  rule_strategy.dataset_character,
            "global_imputation":  rule_strategy.global_imputation,
            "global_outlier":     rule_strategy.global_outlier,
            "global_scaling":     rule_strategy.global_scaling,
            "format_time_series": rule_strategy.format_time_series,
            "columns_to_drop":    rule_strategy.columns_to_drop,
        }, default=str)

        prompt = f"""
You are a data preprocessing AI for a flood prediction system.

## Dataset overview
- Shape: {df.shape[0]} rows × {df.shape[1]} columns
- Dataset type: {profile['__meta__']['dataset_character']}
- Location: {context.get('location', 'unknown')}

## Column profiles (null%, outliers, skew, dtype)
{col_summaries}

## Rule-based strategy (your starting point)
{rule_summary}

## Task
Review the rule-based strategy and produce an improved preprocessing strategy.
Consider flood prediction domain knowledge for the India/Pakistan dataset:
  - rain_mm_weekly and rain_mm_monthly are the most critical weather columns — log transform recommended (right-skewed).
  - dist_major_river_km: lower = closer = higher flood risk. Robust scale recommended.
  - elevation_m and slope_degree: terrain context — standard or robust scale.
  - temp_c_mean, rh_percent_mean, wind_ms_mean: normal distribution, standard scale.
  - dam_count_50km, waterbody_nearby: count/binary columns — do NOT log transform.
  - terrain_type_encoded: categorical integer (0-3) — do NOT scale.
  - flood_occurred: binary target — NEVER drop, NEVER scale.
  - week column (YYYY-MM): temporal — should be dropped after feature extraction or kept as-is.
  - country, name: string ID columns — drop them.
  - Use robust scaler for columns with heavy tails (rainfall, river distance).
  - Do NOT request LSTM formatting (no LSTM model is trained for this dataset).

Respond ONLY with a JSON object matching this schema:
{{
  "dataset_character": "<time_series | tabular | mixed>",
  "global_imputation": "<mean | median | forward_fill | interpolate | knn | seasonal | zero>",
  "global_outlier":    "<clip | remove | replace_median | isolation_forest | none>",
  "global_scaling":    "<standard | minmax | robust | log | none>",
  "format_time_series": <true | false>,
  "sequence_length":   <integer 7-30>,
  "engineer_features": <true | false>,
  "columns_to_drop":   ["<col_name>", ...],
  "column_overrides": [
    {{
      "column":           "<column_name>",
      "imputation":       "<strategy>",
      "outlier_strategy": "<strategy>",
      "scaling":          "<strategy>",
      "drop":             <true|false>,
      "note":             "<one-line reason>"
    }}
  ],
  "rationale": "<2-3 sentence explanation of key decisions>"
}}
"""
        raw = await self._gemini.generate_json(prompt, use_fast_model=False)

        if not raw or not isinstance(raw, dict):
            logger.warning("[StrategySelector] LLM returned invalid JSON — using rule-based strategy.")
            return rule_strategy

        try:
            return self._merge_llm_into_strategy(rule_strategy, raw)
        except Exception as exc:
            logger.warning(f"[StrategySelector] Failed to parse LLM strategy: {exc}")
            return rule_strategy

    def _merge_llm_into_strategy(
        self,
        base: PreprocessingStrategy,
        llm_data: Dict[str, Any],
    ) -> PreprocessingStrategy:
        """Merges valid LLM overrides into the rule-based base strategy."""

        # Global overrides
        try:
            base.dataset_character = DatasetCharacter(
                llm_data.get("dataset_character", base.dataset_character)
            )
        except ValueError:
            pass

        for attr, enum_cls, key in [
            ("global_imputation", ImputationStrategy, "global_imputation"),
            ("global_outlier",    OutlierStrategy,    "global_outlier"),
            ("global_scaling",    ScalingStrategy,    "global_scaling"),
        ]:
            try:
                setattr(base, attr, enum_cls(llm_data.get(key, getattr(base, attr))))
            except ValueError:
                pass

        base.format_time_series = bool(llm_data.get("format_time_series", base.format_time_series))
        base.engineer_features  = bool(llm_data.get("engineer_features",  base.engineer_features))
        base.sequence_length    = int(llm_data.get("sequence_length",     base.sequence_length))
        base.rationale          = llm_data.get("rationale")

        # Merge column drops
        llm_drops = llm_data.get("columns_to_drop", [])
        for col in llm_drops:
            if col not in PROTECTED_COLUMNS and col not in base.columns_to_drop:
                base.columns_to_drop.append(col)

        # Apply column-level overrides
        col_map = {cs.column: cs for cs in base.column_strategies}
        for override in llm_data.get("column_overrides", []):
            col = override.get("column")
            if not col:
                continue
            if col in col_map:
                cs = col_map[col]
                try:
                    cs.imputation = ImputationStrategy(
                        override.get("imputation", cs.imputation)
                    )
                except ValueError:
                    pass
                try:
                    cs.outlier_strategy = OutlierStrategy(
                        override.get("outlier_strategy", cs.outlier_strategy)
                    )
                except ValueError:
                    pass
                try:
                    cs.scaling = ScalingStrategy(
                        override.get("scaling", cs.scaling)
                    )
                except ValueError:
                    pass
                cs.drop = bool(override.get("drop", cs.drop))
                cs.note = override.get("note")

        base.column_strategies = list(col_map.values())
        return base
