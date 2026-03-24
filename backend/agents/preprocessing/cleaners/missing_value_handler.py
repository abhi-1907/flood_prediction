"""
Missing Value Handler – Handles null / NaN values for every column using the
strategy recommended by the StrategySelector.

Supported strategies:
  - mean / median / zero          : Simple statistical fills
  - forward_fill / backward_fill  : Time-series propagation
  - interpolate                   : Linear or time-based interpolation
  - seasonal                      : Day-of-week or month-aware filling
  - knn                           : K-Nearest Neighbours imputation (scikit-learn)
  - drop_row                      : Remove rows where this column is null

Each operation is logged to the AuditLogger for full traceability.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from agents.preprocessing.audit_logger import AuditLogger
from agents.preprocessing.preprocessing_schemas import (
    ColumnStrategy,
    ImputationStrategy,
    PreprocessingStrategy,
)
from utils.logger import logger


class MissingValueHandler:
    """
    Fills missing values in a DataFrame using per-column strategies.

    Usage:
        handler = MissingValueHandler()
        df_clean = handler.handle(df, strategy, audit_logger)
    """

    def handle(
        self,
        df:            pd.DataFrame,
        strategy:      PreprocessingStrategy,
        audit_logger:  AuditLogger,
    ) -> pd.DataFrame:
        """
        Applies imputation strategies to all columns in the DataFrame.

        Args:
            df:           Input DataFrame (may contain NaNs).
            strategy:     Full preprocessing strategy with per-column configs.
            audit_logger: Audit logger to record every transformation.

        Returns:
            DataFrame with missing values filled.
        """
        df = df.copy()
        total_nulls_before = int(df.isna().sum().sum())
        audit_logger.log_shape("missing_value_handler", df, "Input to MissingValueHandler")

        # Build column → strategy map
        col_strategy_map: Dict[str, ColumnStrategy] = {
            cs.column: cs for cs in strategy.column_strategies
        }

        for col in df.columns:
            null_count = int(df[col].isna().sum())
            if null_count == 0:
                continue

            # Get per-column strategy; fall back to global
            cs = col_strategy_map.get(col)
            impute_strategy = cs.imputation if cs else strategy.global_imputation

            # Log before
            audit_logger.log(
                step="missing_value_handler",
                action=f"Imputing {null_count} nulls using {impute_strategy}",
                column=col,
                before_stat=null_count,
                rows_affected=null_count,
                strategy=impute_strategy,
            )

            try:
                df = self._apply(df, col, impute_strategy, strategy)
            except Exception as exc:
                logger.warning(
                    f"[MissingValueHandler] Failed to impute '{col}' with "
                    f"{impute_strategy}: {exc}. Falling back to median."
                )
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(method="ffill", inplace=True)
                    df[col].fillna(method="bfill", inplace=True)

            remaining = int(df[col].isna().sum())
            audit_logger.log(
                step="missing_value_handler",
                action=f"After imputation: {remaining} nulls remain in '{col}'",
                column=col,
                after_stat=remaining,
            )

        total_nulls_after = int(df.isna().sum().sum())
        audit_logger.log(
            step="missing_value_handler",
            action=f"Global nulls: {total_nulls_before} → {total_nulls_after}",
            before_stat=total_nulls_before,
            after_stat=total_nulls_after,
        )
        logger.info(
            f"[MissingValueHandler] Nulls: {total_nulls_before} → {total_nulls_after}"
        )
        return df

    # ── Strategy dispatch ─────────────────────────────────────────────────

    def _apply(
        self,
        df:       pd.DataFrame,
        col:      str,
        strategy: ImputationStrategy,
        full_strategy: PreprocessingStrategy,
    ) -> pd.DataFrame:
        """Dispatches to the right imputation method."""

        if strategy == ImputationStrategy.MEAN:
            df[col] = df[col].fillna(df[col].mean())

        elif strategy == ImputationStrategy.MEDIAN:
            df[col] = df[col].fillna(df[col].median())

        elif strategy == ImputationStrategy.ZERO:
            df[col] = df[col].fillna(0)

        elif strategy == ImputationStrategy.FORWARD_FILL:
            df[col] = df[col].fillna(method="ffill").fillna(method="bfill")

        elif strategy == ImputationStrategy.BACKWARD_FILL:
            df[col] = df[col].fillna(method="bfill").fillna(method="ffill")

        elif strategy == ImputationStrategy.INTERPOLATE:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Time-based interpolation if date column exists
                if "date" in df.columns:
                    df_sorted = df.set_index("date").sort_index()
                    df_sorted[col] = df_sorted[col].interpolate(method="time")
                    df[col] = df_sorted[col].values
                else:
                    df[col] = df[col].interpolate(method="linear", limit_direction="both")
            else:
                df[col] = df[col].fillna(method="ffill").fillna(method="bfill")

        elif strategy == ImputationStrategy.SEASONAL:
            df = self._seasonal_fill(df, col)

        elif strategy == ImputationStrategy.KNN:
            df = self._knn_impute(df, col)

        elif strategy == ImputationStrategy.DROP_ROW:
            before = len(df)
            df = df.dropna(subset=[col])
            after  = len(df)
            logger.info(f"[MissingValueHandler] Dropped {before - after} rows for null '{col}'")

        # Final safety net — fill any remaining NaN with median
        if df[col].isna().any() and pd.api.types.is_numeric_dtype(df[col]):
            median_val = df[col].median()
            df[col]    = df[col].fillna(median_val if pd.notna(median_val) else 0)

        return df

    # ── Seasonal fill ─────────────────────────────────────────────────────

    @staticmethod
    def _seasonal_fill(df: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Fills nulls using the historical average for that month (season-aware).
        Falls back to global median if no date column is present.
        """
        if "date" not in df.columns:
            return df.assign(**{col: df[col].fillna(df[col].median())})

        df = df.copy()
        df["__month__"] = pd.to_datetime(df["date"], errors="coerce").dt.month
        monthly_mean   = df.groupby("__month__")[col].transform("mean")
        df[col]        = df[col].fillna(monthly_mean)
        df[col]        = df[col].fillna(df[col].median())   # catch residual NaNs
        df.drop(columns=["__month__"], inplace=True)
        return df

    # ── KNN imputation ────────────────────────────────────────────────────

    @staticmethod
    def _knn_impute(df: pd.DataFrame, col: str, n_neighbors: int = 5) -> pd.DataFrame:
        """
        K-Nearest Neighbours imputation using scikit-learn's KNNImputer.
        Applied only to the target column; other numeric columns used as features.
        Falls back to median if sklearn is unavailable.
        """
        try:
            from sklearn.impute import KNNImputer

            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            if col not in numeric_cols:
                return df

            imputer   = KNNImputer(n_neighbors=min(n_neighbors, len(df) - 1))
            imputed   = imputer.fit_transform(df[numeric_cols])
            df_imp    = pd.DataFrame(imputed, columns=numeric_cols, index=df.index)
            df[col]   = df_imp[col]
            return df

        except ImportError:
            logger.warning("[MissingValueHandler] sklearn not available — using median for KNN.")
            return df.assign(**{col: df[col].fillna(df[col].median())})
        except Exception as exc:
            logger.warning(f"[MissingValueHandler] KNN failed for '{col}': {exc}")
            return df.assign(**{col: df[col].fillna(df[col].median())})
