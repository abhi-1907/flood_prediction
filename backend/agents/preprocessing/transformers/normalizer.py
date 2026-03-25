"""
Normalizer / Scaler – Applies feature scaling to numeric columns.

Supported strategies:
  - standard   : Z-score normalisation (mean=0, std=1)  [sklearn StandardScaler]
  - minmax     : [0, 1] range scaling                    [sklearn MinMaxScaler]
  - robust     : Median / IQR-based scaling              [sklearn RobustScaler]
                 (outlier-resistant — recommended for flood data)
  - log        : log1p transform (handles right-skewed distributions like rainfall)
  - none       : Pass-through (no scaling)

The fitted scaler objects are stored in the session so the SAME transformation
can be applied to new inference data (critical for reproducible predictions).

Non-numeric and excluded columns (date, lat, lon) are never scaled.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from agents.preprocessing.audit_logger import AuditLogger
from agents.preprocessing.preprocessing_schemas import (
    ColumnStrategy,
    PreprocessingStrategy,
    ScalingStrategy,
)
from utils.logger import logger


# ──# Columns that must NEVER be scaled
# Includes: binary flags, categorical encodings, target, ID/temporal strings
NO_SCALE_COLS = {
    # Geo (pass-through — model uses raw lat/lon)
    "lat", "lon",
    # Legacy names (kept for compatibility)
    "latitude", "longitude",
    # Binary / categorical — scaling would break meaning
    "flood_occurred",          # binary target
    "waterbody_nearby",        # binary flag
    "terrain_type_encoded",    # categorical 0-3 integer
    "heavy_rain_flag",         # engineered binary
    "very_heavy_rain_flag",    # engineered binary
    "low_elevation_flag",      # engineered binary
    "is_monsoon",              # engineered binary
    # String/temporal — never numeric
    "week", "country", "name",
    # Internal audit cols
    "data_quality_score", "data_sources", "data_source",
    "outlier_flag", "flood_risk_proxy",
}


class Normalizer:
    """
    Scales numeric features using the strategy from the PreprocessingStrategy.

    After fitting, the scaler objects are stored in `self.fitted_scalers` as
    a dict of {column_name: scaler_instance} so they can be persisted and reused.

    Usage:
        normalizer = Normalizer()
        df_scaled, scalers = normalizer.fit_transform(df, strategy, audit_logger)
        df_new_scaled      = normalizer.transform(df_new, scalers, audit_logger)
    """

    def __init__(self) -> None:
        self.fitted_scalers: Dict[str, any] = {}

    # ── Public API ────────────────────────────────────────────────────────

    def fit_transform(
        self,
        df:           pd.DataFrame,
        strategy:     PreprocessingStrategy,
        audit_logger: AuditLogger,
    ) -> Tuple[pd.DataFrame, Dict[str, any]]:
        """
        Fits a scaler per column and transforms the DataFrame.

        Args:
            df:           Clean DataFrame (no nulls, no critical outliers).
            strategy:     Preprocessing strategy with scaling choices.
            audit_logger: Audit logger.

        Returns:
            (scaled DataFrame, fitted scalers dict)
        """
        df = df.copy()
        audit_logger.log_shape("normalizer", df, "Input to Normalizer")

        col_strategy_map: Dict[str, ColumnStrategy] = {
            cs.column: cs for cs in strategy.column_strategies
        }

        numeric_cols = [
            c for c in df.select_dtypes(include="number").columns
            if c.lower() not in NO_SCALE_COLS
        ]

        for col in numeric_cols:
            cs     = col_strategy_map.get(col)
            scale  = cs.scaling if cs else strategy.global_scaling

            if scale == ScalingStrategy.NONE:
                continue

            # ── Safety Guard: Never fit_transform on a 1-row dataset ─────
            # (StandardScaler on 1 row results in all 0.0s)
            if df.shape[0] <= 1:
                # logger.debug(f"[Normalizer] Skipping scaling fit for 1-row inference: {col}")
                continue

            before_mean = round(float(df[col].mean()), 4)
            before_std  = round(float(df[col].std()),  4)

            try:
                scaler, df_col = self._apply_scaling(df[[col]], scale)
                df[col]        = df_col.values.ravel()
                self.fitted_scalers[col] = scaler

                after_mean = round(float(df[col].mean()), 4)
                after_std  = round(float(df[col].std()),  4)

                audit_logger.log(
                    step="normalizer",
                    action=f"Scaled '{col}' with {scale}",
                    column=col,
                    before_stat={"mean": before_mean, "std": before_std},
                    after_stat= {"mean": after_mean,  "std": after_std},
                    strategy=scale,
                )
            except Exception as exc:
                logger.warning(f"[Normalizer] Failed to scale '{col}': {exc}")

        audit_logger.log_shape("normalizer", df, "Output of Normalizer")
        logger.info(
            f"[Normalizer] Scaled {len(self.fitted_scalers)} columns. "
            f"Shape: {df.shape}"
        )
        return df, self.fitted_scalers

    def transform(
        self,
        df:           pd.DataFrame,
        scalers:      Dict[str, any],
        audit_logger: AuditLogger,
    ) -> pd.DataFrame:
        """Applies pre-fitted scalers to new data (inference time)."""
        df = df.copy()
        for col, scaler in scalers.items():
            if col not in df.columns:
                continue
            try:
                if scaler == "log1p":
                    df[col] = np.log1p(df[col].clip(lower=0))
                else:
                    df[col] = scaler.transform(df[[col]]).ravel()
            except Exception as exc:
                logger.warning(f"[Normalizer] Transform failed for '{col}': {exc}")
        return df

    # ── Scaling helpers ───────────────────────────────────────────────────

    @staticmethod
    def _apply_scaling(
        df_col: pd.DataFrame,
        strategy: ScalingStrategy,
    ) -> Tuple[any, pd.DataFrame]:
        """Returns (fitted_scaler_or_tag, transformed_DataFrame_column)."""

        if strategy == ScalingStrategy.LOG:
            # log1p clipping to avoid log(negative)
            col_name = df_col.columns[0]
            df_col   = df_col.copy()
            df_col[col_name] = np.log1p(df_col[col_name].clip(lower=0))
            return "log1p", df_col

        try:
            from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

            scaler_cls = {
                ScalingStrategy.STANDARD: StandardScaler,
                ScalingStrategy.MINMAX:   MinMaxScaler,
                ScalingStrategy.ROBUST:   RobustScaler,
            }.get(strategy, StandardScaler)

            scaler  = scaler_cls()
            scaled  = scaler.fit_transform(df_col.fillna(df_col.mean()))
            df_out  = pd.DataFrame(scaled, columns=df_col.columns, index=df_col.index)
            return scaler, df_out

        except ImportError:
            # Fallback: manual standard scaling without sklearn
            logger.warning("[Normalizer] sklearn not available. Using manual z-score.")
            col_name = df_col.columns[0]
            mean = df_col[col_name].mean()
            std  = df_col[col_name].std()
            df_col = df_col.copy()
            df_col[col_name] = (df_col[col_name] - mean) / (std + 1e-9)
            return {"type": "manual_zscore", "mean": mean, "std": std}, df_col
