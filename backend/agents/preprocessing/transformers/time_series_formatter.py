"""
Time Series Formatter – Reshapes a flat DataFrame into 3-D LSTM sequences.

LSTM models expect input shape: (samples, time_steps, features)

This transformer:
  1. Sorts the dataset by date
  2. Selects only numeric feature columns (excluding targets and metadata)
  3. Slides a window of `sequence_length` time-steps over the data
  4. Produces X (3D array) and y (1D target label array)
  5. Optionally applies train/val/test split by date (chronological split)

Returned artefacts stored in the session:
  - "lstm_X"           : numpy array (n_samples, seq_len, n_features)
  - "lstm_y"           : numpy array (n_samples,)
  - "lstm_feature_cols": list of feature column names
  - "lstm_dates"       : list of dates corresponding to each sample
  - "lstm_splits"      : dict with train/val/test slice indices
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from agents.preprocessing.audit_logger import AuditLogger
from agents.preprocessing.preprocessing_schemas import PreprocessingStrategy
from utils.logger import logger


# ── Configuration ─────────────────────────────────────────────────────────────

# Columns to exclude from the LSTM feature set
EXCLUDE_FROM_FEATURES = {
    "date", "data_sources", "data_source", "data_quality_score",
    "outlier_flag", "flood_risk_proxy",   # These are derived targets/metadata
    "flood_reported",                     # This is a target label
}

# Possible target column names (in priority order)
TARGET_CANDIDATES = [
    "flood_reported",
    "flood_risk_proxy",
    "water_level_m",   # Regression target if no classification target
]


class TimeSeriesFormatter:
    """
    Converts a flat, time-indexed DataFrame into windowed LSTM sequences.

    Usage:
        formatter = TimeSeriesFormatter()
        outputs   = formatter.format(df, strategy, audit_logger, session)
    """

    # ── Public API ────────────────────────────────────────────────────────

    def format(
        self,
        df:            pd.DataFrame,
        strategy:      PreprocessingStrategy,
        audit_logger:  AuditLogger,
        sequence_length: Optional[int] = None,
        target_col:    Optional[str]   = None,
        val_frac:      float           = 0.15,
        test_frac:     float           = 0.15,
    ) -> Dict[str, Any]:
        """
        Reshapes the DataFrame into LSTM-ready sequences.

        Args:
            df:              Scaled DataFrame with feature engineering applied.
            strategy:        Preprocessing strategy (provides sequence_length).
            audit_logger:    Audit logger.
            sequence_length: Override sequence window length (defaults to strategy value).
            target_col:      Override target column name.
            val_frac:        Fraction of samples for validation set.
            test_frac:       Fraction of samples for test set.

        Returns:
            Dict with:
              - X:             numpy array (n_samples, seq_len, n_features)
              - y:             numpy array (n_samples,)
              - feature_cols:  list of column names
              - dates:         list of dates for each sample
              - splits:        {"train": (0, n_train), "val": ..., "test": ...}
              - seq_len:       actual sequence length used
        """
        seq_len = sequence_length or strategy.sequence_length
        audit_logger.log_shape("time_series_formatter", df, "Input to TimeSeriesFormatter")

        # ── 1. Sort by date ───────────────────────────────────────────────
        if "date" in df.columns:
            df = df.sort_values("date").reset_index(drop=True)
        else:
            logger.warning("[TimeSeriesFormatter] No 'date' column — using row order.")

        # ── 2. Select target column ────────────────────────────────────────
        y_col = target_col or self._find_target(df)
        if y_col and y_col not in df.columns:
            y_col = None

        # ── 3. Select feature columns ─────────────────────────────────────
        feature_cols = [
            c for c in df.select_dtypes(include="number").columns
            if c not in EXCLUDE_FROM_FEATURES
               and c != (y_col or "")
        ]

        if not feature_cols:
            logger.error("[TimeSeriesFormatter] No numeric feature columns found.")
            return {}

        # ── 4. Extract arrays ─────────────────────────────────────────────
        X_raw = df[feature_cols].values.astype(np.float32)
        y_raw = df[y_col].values.astype(np.float32) if y_col else None

        if X_raw.shape[0] < seq_len + 1:
            logger.warning(
                f"[TimeSeriesFormatter] Not enough rows ({X_raw.shape[0]}) "
                f"for sequence_length={seq_len}. Reducing to {X_raw.shape[0] // 2}."
            )
            seq_len = max(1, X_raw.shape[0] // 2)

        # ── 5. Create sliding windows ─────────────────────────────────────
        X_seq, y_seq, dates_seq = self._sliding_window(
            X_raw, y_raw, df, seq_len
        )

        # ── 6. Chronological train / val / test split ─────────────────────
        n_total  = X_seq.shape[0]
        n_test   = max(1, int(n_total * test_frac))
        n_val    = max(1, int(n_total * val_frac))
        n_train  = n_total - n_val - n_test

        splits = {
            "train": (0,       n_train),
            "val":   (n_train, n_train + n_val),
            "test":  (n_train + n_val, n_total),
        }

        audit_logger.log(
            step="time_series_formatter",
            action=f"Created {n_total} sequences ({seq_len}-step window) | "
                   f"features={len(feature_cols)} | "
                   f"train={n_train}, val={n_val}, test={n_test}",
            before_stat={"rows": len(df), "feature_cols": len(feature_cols)},
            after_stat={"sequences": n_total, "shape": list(X_seq.shape)},
        )

        logger.info(
            f"[TimeSeriesFormatter] X: {X_seq.shape} | "
            f"y: {y_seq.shape if y_seq is not None else 'None'} | "
            f"seq_len={seq_len} | splits={splits}"
        )

        return {
            "X":            X_seq,
            "y":            y_seq,
            "feature_cols": feature_cols,
            "dates":        dates_seq,
            "splits":       splits,
            "seq_len":      seq_len,
            "target_col":   y_col,
        }

    # ── Sliding-window construction ───────────────────────────────────────

    @staticmethod
    def _sliding_window(
        X:      np.ndarray,
        y:      Optional[np.ndarray],
        df:     pd.DataFrame,
        seq_len: int,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List]:
        """
        Creates overlapping windows of shape (seq_len, n_features).

        The target for each window is the value AFTER the window ends
        (t+1 prediction horizon).
        """
        n_samples  = X.shape[0] - seq_len
        n_features = X.shape[1]

        X_windows = np.lib.stride_tricks.sliding_window_view(
            X, window_shape=(seq_len, n_features)
        ).reshape(n_samples, seq_len, n_features)

        y_windows = None
        if y is not None:
            y_windows = y[seq_len:]   # Shift by seq_len (predict next step)
            y_windows = y_windows[:n_samples]

        # Match dates to the END of each window
        dates = []
        if "date" in df.columns:
            date_vals = pd.to_datetime(df["date"], errors="coerce").tolist()
            dates = date_vals[seq_len: seq_len + n_samples]

        return X_windows, y_windows, dates

    # ── Target detection ──────────────────────────────────────────────────

    @staticmethod
    def _find_target(df: pd.DataFrame) -> Optional[str]:
        """Returns the best available target column."""
        for candidate in TARGET_CANDIDATES:
            if candidate in df.columns:
                return candidate
        return None

    # ── Reconstruction helpers ────────────────────────────────────────────

    @staticmethod
    def get_split(
        formatted: Dict[str, Any],
        split: str = "train",
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Returns (X_split, y_split) for train/val/test."""
        start, end = formatted["splits"][split]
        X_split    = formatted["X"][start:end]
        y_split    = formatted["y"][start:end] if formatted["y"] is not None else None
        return X_split, y_split
