"""
Ensemble Combiner – Weighted aggregation of individual model predictions.

Combines XGBoost and Random Forest predictions into a single flood
probability, risk level, and confidence score.

Techniques:
  1. Weighted average      – Uses model weights from ModelSelector (XGB 55%, RF 45%)
  2. Confidence weighting  – Models with higher individual confidence get more weight
  3. Disagreement penalty  – Confidence reduced when models disagree significantly
  4. Risk level mapping    – Probability → LOW / MEDIUM / HIGH / CRITICAL
  5. Uncertainty band      – 95% CI estimated from RF individual tree variance

Models used: XGBoost (xgb_all.json), Random Forest (rf_all.pkl)
External feature groups: weather, hydro, terrain, combined (see model_registry.py)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agents.prediction.prediction_schemas import (
    EnsemblePrediction,
    ModelPrediction,
    ModelStatus,
    ModelType,
    PredictionMode,
    RiskLevel,
)
from utils.logger import logger


# ── Risk thresholds ────────────────────────────────────────────────────────────

RISK_THRESHOLDS = {
    RiskLevel.LOW:      0.25,
    RiskLevel.MEDIUM:   0.50,
    RiskLevel.HIGH:     0.75,
    RiskLevel.CRITICAL: 1.00,
}


class EnsembleCombiner:
    """
    Combines XGBoost and Random Forest ModelPrediction objects.

    Usage:
        combiner = EnsembleCombiner()
        ensemble = combiner.combine(
            predictions=[xgb_pred, rf_pred],
            weights={ModelType.XGBOOST: 0.55, ModelType.RANDOM_FOREST: 0.45},
            mode=PredictionMode.CLASSIFICATION,
            rf_interval=(lower, upper),  # optional
        )
    """

    # ── Public API ────────────────────────────────────────────────────────

    def combine(
        self,
        predictions:      List[ModelPrediction],
        weights:          Dict[ModelType, float],
        mode:             PredictionMode = PredictionMode.CLASSIFICATION,
        rf_interval:      Optional[Tuple[float, float]] = None,
        lstm_uncertainty: Optional[Tuple[float, float, float]] = None,  # kept for API compat, ignored
    ) -> EnsemblePrediction:
        """
        Produces a single ensemble prediction from XGBoost + Random Forest.

        Args:
            predictions:  List of per-model predictions (XGBoost, RF).
            weights:      {ModelType: weight} from ModelSelector.
            mode:         Prediction mode (classification / multi_class / regression).
            rf_interval:  Optional (lower, upper) bounds from RF tree variance.

        Returns:
            EnsemblePrediction with combined probability, risk, and uncertainty.
        """
        # Filter to successful predictions only
        valid = [p for p in predictions if p.status == ModelStatus.LOADED]
        if not valid:
            logger.warning("[EnsembleCombiner] No valid model predictions to combine.")
            return EnsemblePrediction(
                flood_probability=0.5,
                risk_level=RiskLevel.MEDIUM,
                risk_score=50.0,
                model_predictions=predictions,
                models_used=[],
                confidence=0.0,
                uncertainty_band=(0.0, 1.0),
            )

        # ── Step 1: Weighted average ──────────────────────────────────────
        w_avg = self._weighted_average(valid, weights)

        # ── Step 2: Confidence-weighted adjustment ────────────────────────
        conf_weighted = self._confidence_weighted_average(valid, weights)

        # Blend: 70% weight-based + 30% confidence-based
        flood_prob = 0.7 * w_avg + 0.3 * conf_weighted
        flood_prob = float(np.clip(flood_prob, 0.0, 1.0))

        # ── Step 3: Overall confidence (with disagreement penalty) ────────
        confidence = self._compute_confidence(valid, flood_prob)

        # ── Step 4: Risk level & score ────────────────────────────────────
        risk_level = self._probability_to_risk(flood_prob)
        risk_score = round(flood_prob * 100, 1)

        # ── Step 5: Uncertainty band ──────────────────────────────────────
        uncertainty = self._compute_uncertainty(
            valid, flood_prob, rf_interval, lstm_uncertainty
        )

        # Merge feature importances across models
        models_used = [p.model_type for p in valid]

        logger.info(
            f"[EnsembleCombiner] P(flood)={flood_prob:.4f} | "
            f"risk={risk_level} | score={risk_score} | "
            f"conf={confidence:.3f} | CI={uncertainty} | "
            f"models={[m.value for m in models_used]}"
        )

        return EnsemblePrediction(
            flood_probability=round(flood_prob, 4),
            risk_level=risk_level,
            risk_score=risk_score,
            model_predictions=predictions,
            models_used=models_used,
            confidence=round(confidence, 4),
            uncertainty_band=uncertainty,
        )

    # ── Weighted average ──────────────────────────────────────────────────

    @staticmethod
    def _weighted_average(
        valid:   List[ModelPrediction],
        weights: Dict[ModelType, float],
    ) -> float:
        """Simple weighted average using ModelSelector weights."""
        total_w = 0.0
        total_s = 0.0
        for p in valid:
            w = weights.get(p.model_type, 1.0 / len(valid))
            total_w += w
            total_s += p.raw_score * w

        return total_s / total_w if total_w > 0 else 0.5

    @staticmethod
    def _confidence_weighted_average(
        valid:   List[ModelPrediction],
        weights: Dict[ModelType, float],
    ) -> float:
        """
        Weights each model's prediction by the product of:
          - its selector weight
          - its individual confidence score
        """
        total_w = 0.0
        total_s = 0.0
        for p in valid:
            w   = weights.get(p.model_type, 0.33) * max(p.confidence, 0.1)
            total_w += w
            total_s += p.raw_score * w
        return total_s / total_w if total_w > 0 else 0.5

    # ── Confidence & disagreement ─────────────────────────────────────────

    @staticmethod
    def _compute_confidence(
        valid: List[ModelPrediction],
        ensemble_prob: float,
    ) -> float:
        """
        Ensemble confidence = average individual confidence minus a disagreement
        penalty (standard deviation of scores).
        """
        if not valid:
            return 0.0

        avg_conf = np.mean([p.confidence for p in valid])
        scores   = [p.raw_score for p in valid]
        std      = float(np.std(scores)) if len(scores) > 1 else 0.0

        # Disagreement penalty: higher std → lower confidence
        penalty = min(std * 2, 0.4)   # Cap penalty at 0.4
        final   = max(0.0, float(avg_conf) - penalty)
        return final

    # ── Uncertainty band ──────────────────────────────────────────────────

    @staticmethod
    def _compute_uncertainty(
        valid:            List[ModelPrediction],
        ensemble_prob:    float,
        rf_interval:      Optional[Tuple[float, float]] = None,
        lstm_uncertainty: Optional[Tuple[float, float, float]] = None,
    ) -> Tuple[float, float]:
        """
        Computes a 95% confidence interval around the ensemble prediction.

        Sources:
          - RF: tree-level prediction variance
          - LSTM: Monte Carlo Dropout standard deviation
          - Fallback: ±2 × standard deviation of model scores
        """
        lower, upper = 0.0, 1.0

        if rf_interval:
            lower = max(lower, rf_interval[0])
            upper = min(upper, rf_interval[1])

        if lstm_uncertainty:
            mean, std, _ = lstm_uncertainty
            lower = max(lower, mean - 2 * std)
            upper = min(upper, mean + 2 * std)

        # Fallback: use model score spread
        if lower == 0.0 and upper == 1.0:
            scores = [p.raw_score for p in valid]
            if len(scores) > 1:
                std = float(np.std(scores))
                lower = max(0.0, ensemble_prob - 2 * std)
                upper = min(1.0, ensemble_prob + 2 * std)
            else:
                lower = max(0.0, ensemble_prob - 0.15)
                upper = min(1.0, ensemble_prob + 0.15)

        return (round(lower, 4), round(upper, 4))

    # ── Risk mapping ──────────────────────────────────────────────────────

    @staticmethod
    def _probability_to_risk(prob: float) -> RiskLevel:
        if prob < 0.25:
            return RiskLevel.LOW
        if prob < 0.50:
            return RiskLevel.MEDIUM
        if prob < 0.75:
            return RiskLevel.HIGH
        return RiskLevel.CRITICAL

    # ── Merged feature importances ────────────────────────────────────────

    @staticmethod
    def merged_feature_importance(
        predictions: List[ModelPrediction],
        weights:     Dict[ModelType, float],
    ) -> Dict[str, float]:
        """
        Produces a single feature importance dict by weighted-averaging
        each model's importance scores.
        """
        merged: Dict[str, float] = {}
        total_weight: float = 0.0

        for p in predictions:
            if p.status != ModelStatus.LOADED or not p.feature_importance:
                continue
            w = weights.get(p.model_type, 0.33)
            total_weight += w
            for feat, imp in p.feature_importance.items():
                merged[feat] = merged.get(feat, 0.0) + imp * w

        if total_weight > 0:
            merged = {k: round(v / total_weight, 4) for k, v in merged.items()}

        return dict(sorted(merged.items(), key=lambda x: -x[1])[:15])
