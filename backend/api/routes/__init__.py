"""API routes package — re-exports for main.py router registration."""

from api.routes import (
    health_routes,
    orchestration_routes,
    ingestion_routes,
    prediction_routes,
    recommendation_routes,
    simulation_routes,
    alert_routes,
)

__all__ = [
    "health_routes",
    "orchestration_routes",
    "ingestion_routes",
    "prediction_routes",
    "recommendation_routes",
    "simulation_routes",
    "alert_routes",
]
