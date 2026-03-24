"""
Random Forest Predictor – Ensemble tree model for flood classification/regression.

Provides:
  - Binary classification (probability via predict_proba)
  - Regression (predict continuous value)
  - Feature importance (MDI — Mean Decrease Impurity)
  - Prediction interval estimation via individual tree variance
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


class RandomForestPredictor:
    """
    Wraps a scikit-learn RandomForestClassifier or Regressor.

    Usage:
        predictor = RandomForestPredictor(model, feature_cols)
        result    = predictor.predict(df, mode=PredictionMode.CLASSIFICATION)
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
        """Runs Random Forest inference on the most recent input row."""
        if self._model is None:
            return ModelPrediction(
                model_type=ModelType.RANDOM_FOREST,
                mode=mode,
                raw_score=0.5,
                status=ModelStatus.FAILED,
                error="Model not loaded",
            )
        try:
            X = self._prepare_input(df)

            if mode == PredictionMode.REGRESSION:
                raw   = self._model.predict(X)
                score = float(raw[-1]) if len(raw) > 0 else 0.0
                label = str(round(score, 3))
                conf  = 0.7
            else:
                # Classification — use predict_proba
                proba = self._model.predict_proba(X)
                # proba shape: (n_samples, n_classes); class 1 = flood
                n_classes = proba.shape[1]
                if n_classes >= 2:
                    score = float(proba[-1, 1])   # P(flood) for last row
                else:
                    score = float(proba[-1, 0])
                label = self._score_to_label(score, mode)
                conf  = self._confidence(score, mode, proba[-1])

            importance = self._get_importance()

            return ModelPrediction(
                model_type=ModelType.RANDOM_FOREST,
                mode=mode,
                raw_score=round(score, 4),
                label=label,
                confidence=round(conf, 4),
                feature_importance=importance,
                status=ModelStatus.LOADED,
            )

        except Exception as exc:
            logger.error(f"[RandomForestPredictor] Prediction error: {exc}")
            return ModelPrediction(
                model_type=ModelType.RANDOM_FOREST,
                mode=mode,
                raw_score=0.5,
                status=ModelStatus.FAILED,
                error=str(exc),
            )

    def predict_proba_all(
        self, df: pd.DataFrame
    ) -> np.ndarray:
        """Returns P(flood) for ALL rows (used for SHAP / calibration)."""
        try:
            X     = self._prepare_input(df)
            proba = self._model.predict_proba(X)
            return proba[:, 1] if proba.shape[1] >= 2 else proba[:, 0]
        except Exception:
            return np.full(len(df), 0.5)

    def predict_interval(
        self,
        df:    pd.DataFrame,
        quantile: float = 0.95,
    ) -> Tuple[float, float]:
        """
        Estimates a prediction interval via individual tree variance.
        Returns (lower, upper) bounds at the given quantile.
        """
        try:
            X   = self._prepare_input(df)[-1:]
            tree_preds = np.array([
                tree.predict_proba(X)[0, 1]
                for tree in self._model.estimators_
            ])
            lower = float(np.percentile(tree_preds, (1 - quantile) / 2 * 100))
            upper = float(np.percentile(tree_preds, (1 + quantile) / 2 * 100))
            return lower, upper
        except Exception:
            return 0.0, 1.0

    # ── Internal helpers ──────────────────────────────────────────────────

    def _prepare_input(self, df: pd.DataFrame) -> np.ndarray:
        """Aligns and fills the input DataFrame to the training feature set."""
        present = [c for c in self._features if c in df.columns]
        missing = [c for c in self._features if c not in df.columns]
        aligned = df[present].copy()
        for c in missing:
            aligned[c] = 0.0
        ordered = [c for c in self._features]
        aligned = aligned.reindex(columns=ordered, fill_value=0.0)
        return aligned.fillna(0).values.astype(np.float32)

    def _get_importance(self) -> Dict[str, float]:
        """Extracts MDI feature importances from the RF model."""
        try:
            importances = self._model.feature_importances_
            total       = importances.sum() or 1
            pairs       = sorted(
                zip(self._features, importances / total),
                key=lambda x: -x[1],
            )
            return {k: round(float(v), 4) for k, v in pairs[:15]}
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
    def _confidence(
        score: float,
        mode:  PredictionMode,
        proba: np.ndarray,
    ) -> float:
        if mode == PredictionMode.REGRESSION:
            return 0.7
        # Margin = difference between top-2 class probabilities (like margin sampling)
        sorted_proba = np.sort(proba)[::-1]
        if len(sorted_proba) >= 2:
            return round(float(sorted_proba[0] - sorted_proba[1]), 4)
        return round(abs(score - 0.5) * 2, 4)
