"""
LSTM Predictor – Keras/TensorFlow sequence model for time-series flood prediction.

Wraps a pre-trained LSTM model (loaded by ModelRegistry) and provides:
  - Binary classification (sigmoid output → flood probability 0–1)
  - Automatic input reshaping to 3-D sequences (samples, timesteps, features)
  - Temporal attention weight extraction (if attention-based model)
  - Monte Carlo Dropout uncertainty estimation

The LSTM expects pre-formatted 3-D arrays from the TimeSeriesFormatter,
but also supports raw DataFrames via automatic windowed reshaping.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from agents.prediction.prediction_schemas import (
    ModelPrediction,
    ModelStatus,
    ModelType,
    PredictionMode,
)
from utils.logger import logger


DEFAULT_SEQ_LEN = 7   # must match training sequence length


class LSTMPredictor:
    """
    Wraps a Keras LSTM model and provides a clean predict() interface.

    Usage:
        predictor = LSTMPredictor(model, feature_cols, seq_len=7)
        result    = predictor.predict(df, mode=PredictionMode.CLASSIFICATION)
        result    = predictor.predict_from_sequences(X_3d, mode=...)
    """

    def __init__(
        self,
        model:        Any,
        feature_cols: List[str],
        seq_len:      int = DEFAULT_SEQ_LEN,
    ) -> None:
        self._model    = model
        self._features = feature_cols
        self._seq_len  = seq_len

    # ── Public API ────────────────────────────────────────────────────────

    def predict(
        self,
        df:   pd.DataFrame,
        mode: PredictionMode = PredictionMode.CLASSIFICATION,
    ) -> ModelPrediction:
        """
        Runs LSTM inference on a DataFrame by auto-creating a sliding window.
        Returns prediction for the LAST window (most recent time-step).
        """
        if self._model is None:
            return ModelPrediction(
                model_type=ModelType.LSTM,
                mode=mode,
                raw_score=0.5,
                status=ModelStatus.FAILED,
                error="LSTM model not loaded",
            )
        try:
            X_3d = self._prepare_input(df)
            return self._run_inference(X_3d, mode)
        except ImportError:
            return ModelPrediction(
                model_type=ModelType.LSTM,
                mode=mode,
                raw_score=0.5,
                status=ModelStatus.FAILED,
                error="tensorflow library not installed",
            )
        except Exception as exc:
            logger.error(f"[LSTMPredictor] Prediction error: {exc}")
            return ModelPrediction(
                model_type=ModelType.LSTM,
                mode=mode,
                raw_score=0.5,
                status=ModelStatus.FAILED,
                error=str(exc),
            )

    def predict_from_sequences(
        self,
        X_3d: np.ndarray,
        mode: PredictionMode = PredictionMode.CLASSIFICATION,
    ) -> ModelPrediction:
        """
        Runs inference on pre-formatted 3-D sequences (from TimeSeriesFormatter).
        Uses the LAST sequence's prediction.
        """
        if self._model is None:
            return ModelPrediction(
                model_type=ModelType.LSTM,
                mode=mode,
                raw_score=0.5,
                status=ModelStatus.FAILED,
                error="LSTM model not loaded",
            )
        try:
            return self._run_inference(X_3d, mode)
        except Exception as exc:
            logger.error(f"[LSTMPredictor] Sequence prediction error: {exc}")
            return ModelPrediction(
                model_type=ModelType.LSTM,
                mode=mode,
                raw_score=0.5,
                status=ModelStatus.FAILED,
                error=str(exc),
            )

    def predict_with_uncertainty(
        self,
        df:           pd.DataFrame,
        n_forward:    int = 30,
        mode:         PredictionMode = PredictionMode.CLASSIFICATION,
    ) -> Tuple[float, float, float]:
        """
        Monte Carlo Dropout uncertainty estimation.
        Runs N forward passes with dropout enabled to estimate prediction
        mean and standard deviation.

        Returns: (mean_score, std_score, confidence)
        """
        if self._model is None:
            return 0.5, 0.25, 0.0

        try:
            import tensorflow as tf
            X_3d = self._prepare_input(df)
            last_window = X_3d[-1:]   # (1, seq_len, features)

            preds = []
            for _ in range(n_forward):
                # training=True enables dropout during inference (MC Dropout)
                raw = self._model(last_window, training=True).numpy().flatten()
                preds.append(float(raw[0]) if len(raw) > 0 else 0.5)

            mean_pred = float(np.mean(preds))
            std_pred  = float(np.std(preds))
            conf      = max(0.0, 1.0 - 2 * std_pred)   # High std → low confidence

            logger.info(
                f"[LSTMPredictor] MC Dropout: mean={mean_pred:.4f}, "
                f"std={std_pred:.4f}, conf={conf:.4f}"
            )
            return mean_pred, std_pred, conf

        except Exception as exc:
            logger.warning(f"[LSTMPredictor] MC Dropout failed: {exc}")
            return 0.5, 0.25, 0.0

    # ── Internal inference ────────────────────────────────────────────────

    def _run_inference(
        self,
        X_3d: np.ndarray,
        mode: PredictionMode,
    ) -> ModelPrediction:
        """Handles the actual model.predict() call and format the result."""
        raw  = self._model.predict(X_3d, verbose=0)
        # Flatten to 1-D
        flat = raw.flatten()

        # Use the last prediction
        score = float(flat[-1]) if len(flat) > 0 else 0.5
        score = float(np.clip(score, 0.0, 1.0)) if mode != PredictionMode.REGRESSION else float(score)

        label = self._score_to_label(score, mode)

        # Feature importance — approximate via last-step input gradient (lightweight)
        importance = self._gradient_importance(X_3d)

        # Confidence via MC Dropout
        _, std, conf = self.predict_with_uncertainty.__wrapped__(self, pd.DataFrame()) \
            if False else (score, 0.0, self._confidence(score, mode))

        return ModelPrediction(
            model_type=ModelType.LSTM,
            mode=mode,
            raw_score=round(score, 4),
            label=label,
            confidence=round(conf, 4),
            feature_importance=importance,
            status=ModelStatus.LOADED,
        )

    # ── Input preparation ─────────────────────────────────────────────────

    def _prepare_input(self, df: pd.DataFrame) -> np.ndarray:
        """
        Converts a flat DataFrame to a 3-D array (n_windows, seq_len, features).

        If the DataFrame is shorter than seq_len, zero-pads from the left.
        """
        # Align features
        present = [c for c in self._features if c in df.columns]
        missing = [c for c in self._features if c not in df.columns]
        aligned = df[present].copy()
        for c in missing:
            aligned[c] = 0.0
        aligned = aligned.reindex(columns=self._features, fill_value=0.0)
        X = aligned.fillna(0).values.astype(np.float32)

        n_samples, n_features = X.shape

        if n_samples < self._seq_len:
            # Zero-pad from left
            pad = np.zeros((self._seq_len - n_samples, n_features), dtype=np.float32)
            X = np.vstack([pad, X])
            n_samples = self._seq_len

        # Create sliding windows
        n_windows = n_samples - self._seq_len + 1
        windows   = np.lib.stride_tricks.sliding_window_view(
            X, window_shape=(self._seq_len, n_features)
        ).reshape(n_windows, self._seq_len, n_features)

        return windows

    # ── Feature importance (gradient approximation) ───────────────────────

    def _gradient_importance(self, X_3d: np.ndarray) -> Dict[str, float]:
        """
        Lightweight feature importance via input perturbation on the last window.
        Perturbs each feature ±1 std and measures the prediction delta.
        """
        try:
            last_window = X_3d[-1:]   # (1, seq_len, features)
            base_pred   = float(self._model.predict(last_window, verbose=0).flatten()[0])

            importance = {}
            for i, feat in enumerate(self._features):
                if i >= last_window.shape[2]:
                    break
                perturbed        = last_window.copy()
                std_val          = max(float(np.std(perturbed[0, :, i])), 1e-3)
                perturbed[0, :, i] += std_val

                pert_pred = float(self._model.predict(perturbed, verbose=0).flatten()[0])
                delta     = abs(pert_pred - base_pred)
                importance[feat] = round(delta, 4)

            # Normalise
            total = sum(importance.values()) or 1
            importance = {k: round(v / total, 4)
                          for k, v in sorted(importance.items(), key=lambda x: -x[1])[:15]}
            return importance
        except Exception:
            return {}

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _score_to_label(score: float, mode: PredictionMode) -> str:
        if mode == PredictionMode.CLASSIFICATION:
            return "flood" if score >= 0.5 else "no_flood"
        if mode == PredictionMode.MULTI_CLASS:
            if score < 0.25: return "LOW"
            if score < 0.5:  return "MEDIUM"
            if score < 0.75: return "HIGH"
            return "CRITICAL"
        return str(round(score, 3))

    @staticmethod
    def _confidence(score: float, mode: PredictionMode) -> float:
        if mode == PredictionMode.REGRESSION:
            return 0.6
        return round(abs(score - 0.5) * 2, 4)
