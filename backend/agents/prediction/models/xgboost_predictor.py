"""
XGBoost Predictor – Gradient-boosted tree model for flood classification/regression.

Wraps the XGBoost Booster (trained via model_registry) and provides:
  - Binary classification (flood probability 0–1)
  - Regression (water level or continuous risk score)
  - Feature importance extraction (gain-based)
  - Input preparation with automatic feature alignment
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


class XGBoostPredictor:
    """
    Wraps an XGBoost Booster and provides a clean predict() interface.

    Usage:
        predictor = XGBoostPredictor(model, feature_cols)
        result = predictor.predict(df, mode=PredictionMode.CLASSIFICATION)
    """

    def __init__(self, model: Any, feature_cols: List[str]) -> None:
        self._model    = model
        self._features = feature_cols

    # ── Public API ────────────────────────────────────────────────────────

    def predict(
        self,
        df:   pd.DataFrame,
        mode: PredictionMode = PredictionMode.CLASSIFICATION,
    ) -> ModelPrediction:
        """
        Runs XGBoost inference on the input DataFrame.

        Returns the LAST ROW's prediction (most recent time-step).
        """
        if self._model is None:
            return ModelPrediction(
                model_type=ModelType.XGBOOST,
                mode=mode,
                raw_score=0.5,
                status=ModelStatus.FAILED,
                error="Model not loaded",
            )
        try:
            import xgboost as xgb

            X    = self._prepare_input(df)
            dmat = xgb.DMatrix(X)
            raw  = self._model.predict(dmat)

            # Use the last sample's prediction (most recent row)
            score = float(raw[-1]) if len(raw) > 0 else 0.5
            score = float(np.clip(score, 0.0, 1.0)) if mode == PredictionMode.CLASSIFICATION else float(score)

            # Feature importance (gain-based)
            importance = self._get_importance()
            label      = self._score_to_label(score, mode)

            return ModelPrediction(
                model_type=ModelType.XGBOOST,
                mode=mode,
                raw_score=round(score, 4),
                label=label,
                confidence=self._confidence(score, mode),
                feature_importance=importance,
                status=ModelStatus.LOADED,
            )
        except ImportError:
            return ModelPrediction(
                model_type=ModelType.XGBOOST,
                mode=mode,
                raw_score=0.5,
                status=ModelStatus.FAILED,
                error="xgboost library not installed",
            )
        except Exception as exc:
            logger.error(f"[XGBoostPredictor] Prediction error: {exc}")
            return ModelPrediction(
                model_type=ModelType.XGBOOST,
                mode=mode,
                raw_score=0.5,
                status=ModelStatus.FAILED,
                error=str(exc),
            )

    def predict_proba_all(
        self,
        df: pd.DataFrame,
    ) -> np.ndarray:
        """Returns flood probability for ALL rows (used by calibration / SHAP)."""
        try:
            import xgboost as xgb
            X    = self._prepare_input(df)
            dmat = xgb.DMatrix(X)
            return np.clip(self._model.predict(dmat), 0, 1)
        except Exception:
            return np.full(len(df), 0.5)

    # ── Internal helpers ──────────────────────────────────────────────────

    def _prepare_input(self, df: pd.DataFrame) -> np.ndarray:
        """Aligns DataFrame columns to the model's exact training feature order.

        For any feature expected by the model but not present in the input,
        fills with 0.0 (safe default — matches training fillna(0) strategy).
        """
        aligned = pd.DataFrame(index=df.index)
        for col in self._features:
            if col in df.columns:
                aligned[col] = df[col].fillna(0.0)
            else:
                aligned[col] = 0.0   # Missing feature → zero-fill
        return aligned.values.astype(np.float32)


    def _get_importance(self) -> Dict[str, float]:
        """Extracts gain-based feature importance scores."""
        try:
            scores = self._model.get_score(importance_type="gain")
            total  = sum(scores.values()) or 1
            return {k: round(v / total, 4) for k, v in sorted(
                scores.items(), key=lambda x: -x[1]
            )[:15]}
        except Exception:
            return {}

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
        """Confidence = distance from decision boundary (0.5)."""
        if mode == PredictionMode.REGRESSION:
            return 0.7   # Default confidence for regression
        return round(abs(score - 0.5) * 2, 4)   # 0 at boundary, 1 at extremes
