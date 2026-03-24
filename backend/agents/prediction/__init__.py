"""
Prediction Agent package — public API.
"""

from agents.prediction.prediction_agent import PredictionAgent
from agents.prediction.prediction_schemas import (
    EnsemblePrediction,
    FeatureContribution,
    LoadedModel,
    ModelPrediction,
    ModelStatus,
    ModelType,
    PredictionExplanation,
    PredictionMode,
    PredictionResult,
    RiskLevel,
)
from agents.prediction.model_registry import ModelRegistry
from agents.prediction.model_selector import ModelSelector
from agents.prediction.ensemble import EnsembleCombiner
from agents.prediction.explainer import PredictionExplainer

__all__ = [
    "PredictionAgent",
    # Schemas
    "EnsemblePrediction",
    "FeatureContribution",
    "LoadedModel",
    "ModelPrediction",
    "ModelStatus",
    "ModelType",
    "PredictionExplanation",
    "PredictionMode",
    "PredictionResult",
    "RiskLevel",
    # Core classes
    "ModelRegistry",
    "ModelSelector",
    "EnsembleCombiner",
    "PredictionExplainer",
]
