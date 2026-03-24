"""
Outlier Handler – Detects and resolves statistical outliers per column.

Supported strategies:
  - clip              : IQR-based winsorization (Q1 - 1.5×IQR, Q3 + 1.5×IQR)
  - remove            : Drop rows where any tracked column is an outlier
  - replace_mean      : Replace outlier values with column mean
  - replace_median    : Replace with column median
  - isolation_forest  : Unsupervised ML-based outlier detection
  - none              : Pass-through (log and skip)

Important design note:
  - 'clip' is the default for flood prediction because extreme rainfall events
    ARE real data — we don't want to remove them, just winsorise to avoid
    breaking ML models entirely.
  - 'isolation_forest' identifies multivariate outliers (anomaly rows) rather
    than single-column spikes.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from agents.preprocessing.audit_logger import AuditLogger
from agents.preprocessing.preprocessing_schemas import (
    ColumnStrategy,
    OutlierStrategy,
    PreprocessingStrategy,
)
from utils.logger import logger


# ── Protected columns — never clip/remove ────────────────────────────────────

SKIP_OUTLIER_COLS = {"date", "latitude", "longitude", "data_quality_score",
                     "data_sources", "data_source"}


class OutlierHandler:
    """
    Detects and handles outliers in numeric columns.

    Usage:
        handler = OutlierHandler()
        df_clean = handler.handle(df, strategy, audit_logger)
    """

    def handle(
        self,
        df:           pd.DataFrame,
        strategy:     PreprocessingStrategy,
        audit_logger: AuditLogger,
    ) -> pd.DataFrame:
        """
        Applies outlier handling to all eligible numeric columns.

        Args:
            df:           DataFrame with potentially imputed values.
            strategy:     Preprocessing strategy.
            audit_logger: Audit logger for traceability.

        Returns:
            DataFrame with outliers resolved.
        """
        df = df.copy()
        audit_logger.log_shape("outlier_handler", df, "Input to OutlierHandler")

        col_strategy_map: Dict[str, ColumnStrategy] = {
            cs.column: cs for cs in strategy.column_strategies
        }

        numeric_cols = [
            c for c in df.select_dtypes(include="number").columns
            if c.lower() not in SKIP_OUTLIER_COLS
        ]

        # Isolation Forest applies to all numeric columns at once
        iso_cols = [
            c for c in numeric_cols
            if (col_strategy_map.get(c) or ColumnStrategy(
                    column=c,
                    outlier_strategy=strategy.global_outlier)).outlier_strategy
               == OutlierStrategy.ISOLATIONFOREST
        ]
        if iso_cols:
            df = self._isolation_forest(df, iso_cols, audit_logger)

        # Per-column strategies
        for col in numeric_cols:
            cs = col_strategy_map.get(col)
            col_strategy = cs.outlier_strategy if cs else strategy.global_outlier

            if col_strategy == OutlierStrategy.ISOLATIONFOREST:
                continue   # Already handled above

            outlier_mask = self._iqr_mask(df[col])
            n_outliers   = int(outlier_mask.sum())

            if n_outliers == 0 or col_strategy == OutlierStrategy.NONE:
                continue

            audit_logger.log(
                step="outlier_handler",
                action=f"Found {n_outliers} outliers ({n_outliers/len(df)*100:.1f}%)",
                column=col,
                before_stat={"n_outliers": n_outliers,
                             "min": round(float(df[col].min()), 4),
                             "max": round(float(df[col].max()), 4)},
                rows_affected=n_outliers,
                strategy=col_strategy,
            )

            df = self._apply(df, col, col_strategy, outlier_mask)

            audit_logger.log(
                step="outlier_handler",
                action=f"After outlier handling: range [{df[col].min():.3f}, {df[col].max():.3f}]",
                column=col,
                after_stat={"min": round(float(df[col].min()), 4),
                            "max": round(float(df[col].max()), 4)},
            )

        audit_logger.log_shape("outlier_handler", df, "Output of OutlierHandler")
        logger.info(f"[OutlierHandler] Processed {len(numeric_cols)} numeric columns")
        return df

    # ── Strategy dispatch ─────────────────────────────────────────────────

    def _apply(
        self,
        df:           pd.DataFrame,
        col:          str,
        strat:        OutlierStrategy,
        outlier_mask: pd.Series,
    ) -> pd.DataFrame:
        """Applies the chosen outlier strategy to a single column."""

        if strat == OutlierStrategy.CLIP:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr    = q3 - q1
            lower  = q1 - 1.5 * iqr
            upper  = q3 + 1.5 * iqr
            df[col] = df[col].clip(lower=lower, upper=upper)

        elif strat == OutlierStrategy.REMOVE:
            df = df[~outlier_mask].reset_index(drop=True)

        elif strat == OutlierStrategy.REPLACE_MEAN:
            df.loc[outlier_mask, col] = df[col].mean()

        elif strat == OutlierStrategy.REPLACE_MEDIAN:
            df.loc[outlier_mask, col] = df[col].median()

        return df

    # ── IQR outlier mask ──────────────────────────────────────────────────

    @staticmethod
    def _iqr_mask(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
        """Returns a boolean mask: True where the value is an IQR outlier."""
        q1, q3 = series.quantile([0.25, 0.75])
        iqr    = q3 - q1
        return (series < (q1 - multiplier * iqr)) | (series > (q3 + multiplier * iqr))

    @staticmethod
    def _zscore_mask(series: pd.Series, threshold: float = 3.5) -> pd.Series:
        """Returns a boolean mask: True where |z-score| exceeds threshold."""
        mean, std = series.mean(), series.std()
        return (np.abs((series - mean) / (std + 1e-9)) > threshold)

    # ── Isolation Forest ──────────────────────────────────────────────────

    @staticmethod
    def _isolation_forest(
        df:           pd.DataFrame,
        cols:         List[str],
        audit_logger: AuditLogger,
        contamination: float = 0.05,
    ) -> pd.DataFrame:
        """
        Uses sklearn IsolationForest to detect multivariate anomaly rows.
        Anomalous rows are NOT dropped — their outlier_flag column is set to True
        so downstream steps can handle them.
        """
        try:
            from sklearn.ensemble import IsolationForest

            sub = df[cols].dropna()
            if len(sub) < 20:
                return df

            iso     = IsolationForest(contamination=contamination, random_state=42)
            labels  = iso.fit_predict(sub)   # -1 = anomaly, 1 = inlier
            anomalies = (labels == -1)
            n_anomalies = int(anomalies.sum())

            df["outlier_flag"] = False
            df.loc[sub.index[anomalies], "outlier_flag"] = True

            audit_logger.log(
                step="outlier_handler",
                action=f"IsolationForest detected {n_anomalies} anomaly rows "
                       f"({n_anomalies/len(df)*100:.1f}%) — flagged in 'outlier_flag' column",
                rows_affected=n_anomalies,
                strategy=OutlierStrategy.ISOLATIONFOREST,
            )
            logger.info(f"[OutlierHandler] IsolationForest: {n_anomalies} anomalies flagged.")

        except ImportError:
            logger.warning("[OutlierHandler] sklearn not available — skipping IsolationForest.")
        except Exception as exc:
            logger.warning(f"[OutlierHandler] IsolationForest failed: {exc}")

        return df
