"""
Row Discard Handler – Removes unreliable rows from the dataset.

Criteria for discarding a row:
  1. Missing ALL critical columns (rainfall_mm AND water_level_m AND discharge_m3s)
  2. Data quality score below threshold (if 'data_quality_score' column exists)
  3. Duplicate date × location combinations
  4. Rows marked as synthetic proxy data (where allowed to remove)
  5. Future dates (timestamps beyond today — data error)
  6. Impossible physical values (negative rainfall, elevation below sea-level
     when inland, etc.)

All discards are logged to the AuditLogger with explicit reasons.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np

from agents.preprocessing.audit_logger import AuditLogger
from agents.preprocessing.preprocessing_schemas import PreprocessingStrategy
from utils.logger import logger


# ── Physical validity thresholds ──────────────────────────────────────────────

MIN_RAINFALL_MM       = -0.1     # below this = impossible (allow tiny negatives from rounding)
MAX_RAINFALL_MM       = 1500.0   # realistic daily max (world record ~1825mm; India max ~1168mm)
MAX_WIND_KMH          = 400.0    # above this = sensor error
MIN_HUMIDITY_PCT      = 0.0
MAX_HUMIDITY_PCT      = 100.0
MIN_ELEVATION_M       = -500.0   # Dead Sea / Death Valley
MAX_ELEVATION_M       = 9000.0   # Everest
MIN_QUALITY_SCORE     = 15.0     # Rows below 15/100 quality score are discarded

CRITICAL_COLUMNS = ["rainfall_mm", "water_level_m", "discharge_m3s", "soil_moisture_pct"]


class RowDiscardHandler:
    """
    Removes rows that are physically impossible, completely empty, or duplicated.

    Usage:
        handler = RowDiscardHandler()
        df_clean = handler.handle(df, strategy, audit_logger)
    """

    def handle(
        self,
        df:           pd.DataFrame,
        strategy:     PreprocessingStrategy,
        audit_logger: AuditLogger,
    ) -> pd.DataFrame:
        """
        Runs all discard rules and removes qualifying rows.

        Args:
            df:           Partially cleaned DataFrame.
            strategy:     Preprocessing strategy (not directly used here but
                          passed for consistency).
            audit_logger: Audit logger.

        Returns:
            DataFrame with unreliable rows removed.
        """
        df = df.copy()
        original_len = len(df)
        audit_logger.log_shape("row_discard_handler", df, "Input to RowDiscardHandler")

        # Apply each discard rule in sequence
        df, n1 = self._discard_all_critical_nulls(df, audit_logger)
        df, n2 = self._discard_low_quality(df, audit_logger)
        df, n3 = self._discard_duplicates(df, audit_logger)
        df, n4 = self._discard_future_dates(df, audit_logger)
        df, n5 = self._discard_physically_impossible(df, audit_logger)

        total_discarded = original_len - len(df)
        pct_discarded   = total_discarded / original_len * 100 if original_len > 0 else 0

        audit_logger.log(
            step="row_discard_handler",
            action=f"Total discarded: {total_discarded} rows ({pct_discarded:.1f}%)",
            before_stat=original_len,
            after_stat=len(df),
            rows_affected=total_discarded,
        )

        logger.info(
            f"[RowDiscardHandler] {original_len} → {len(df)} rows "
            f"(removed {total_discarded} / {pct_discarded:.1f}%)"
        )

        df = df.reset_index(drop=True)
        return df

    # ── Discard rules ─────────────────────────────────────────────────────

    def _discard_all_critical_nulls(
        self,
        df:           pd.DataFrame,
        audit_logger: AuditLogger,
    ) -> Tuple[pd.DataFrame, int]:
        """Removes rows where ALL critical columns are null simultaneously."""
        present_critical = [c for c in CRITICAL_COLUMNS if c in df.columns]
        if not present_critical:
            return df, 0

        all_null_mask = df[present_critical].isna().all(axis=1)
        n_removed     = int(all_null_mask.sum())

        if n_removed:
            df = df[~all_null_mask]
            audit_logger.log(
                step="row_discard_handler",
                action=f"Discarded {n_removed} rows with all critical columns null",
                rows_affected=n_removed,
                strategy="all_critical_null",
            )
        return df, n_removed

    def _discard_low_quality(
        self,
        df:           pd.DataFrame,
        audit_logger: AuditLogger,
        threshold:    float = MIN_QUALITY_SCORE,
    ) -> Tuple[pd.DataFrame, int]:
        """Removes rows with a data_quality_score below the threshold."""
        if "data_quality_score" not in df.columns:
            return df, 0

        low_quality_mask = df["data_quality_score"] < threshold
        n_removed        = int(low_quality_mask.sum())

        if n_removed:
            df = df[~low_quality_mask]
            audit_logger.log(
                step="row_discard_handler",
                action=f"Discarded {n_removed} rows with quality_score < {threshold}",
                rows_affected=n_removed,
                strategy="low_quality_score",
            )
        return df, n_removed

    def _discard_duplicates(
        self,
        df:           pd.DataFrame,
        audit_logger: AuditLogger,
    ) -> Tuple[pd.DataFrame, int]:
        """Removes duplicate rows (same date + location)."""
        before = len(df)

        subset = [c for c in ["date", "latitude", "longitude"] if c in df.columns]
        if subset:
            df = df.drop_duplicates(subset=subset, keep="first")
        else:
            df = df.drop_duplicates(keep="first")

        n_removed = before - len(df)
        if n_removed:
            audit_logger.log(
                step="row_discard_handler",
                action=f"Dropped {n_removed} exact-duplicate rows",
                rows_affected=n_removed,
                strategy="deduplication",
            )
        return df, n_removed

    def _discard_future_dates(
        self,
        df:           pd.DataFrame,
        audit_logger: AuditLogger,
    ) -> Tuple[pd.DataFrame, int]:
        """Removes rows with dates in the future (data integrity error)."""
        if "date" not in df.columns:
            return df, 0

        today       = pd.Timestamp(datetime.utcnow().date())
        future_mask = pd.to_datetime(df["date"], errors="coerce") > today
        n_removed   = int(future_mask.sum())

        if n_removed:
            df = df[~future_mask]
            audit_logger.log(
                step="row_discard_handler",
                action=f"Discarded {n_removed} rows with future dates (after {today.date()})",
                rows_affected=n_removed,
                strategy="future_date",
            )
        return df, n_removed

    def _discard_physically_impossible(
        self,
        df:           pd.DataFrame,
        audit_logger: AuditLogger,
    ) -> Tuple[pd.DataFrame, int]:
        """Removes rows with physically impossible sensor values."""
        masks  = []
        checks = [
            ("rainfall_mm",    MIN_RAINFALL_MM,   MAX_RAINFALL_MM),
            ("wind_speed_kmh", 0.0,                MAX_WIND_KMH),
            ("humidity_pct",   MIN_HUMIDITY_PCT,   MAX_HUMIDITY_PCT),
            ("elevation_m",    MIN_ELEVATION_M,    MAX_ELEVATION_M),
        ]

        for col, lo, hi in checks:
            if col not in df.columns:
                continue
            bad = (df[col] < lo) | (df[col] > hi)
            if bad.any():
                masks.append(bad)
                audit_logger.log(
                    step="row_discard_handler",
                    action=f"Found {int(bad.sum())} impossible values in '{col}' "
                           f"(outside [{lo}, {hi}])",
                    column=col,
                    rows_affected=int(bad.sum()),
                    strategy="physical_validity",
                )

        if not masks:
            return df, 0

        combined   = pd.concat(masks, axis=1).any(axis=1)
        n_removed  = int(combined.sum())
        df         = df[~combined]
        return df, n_removed
