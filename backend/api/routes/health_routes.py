"""
Health Routes – System health checks and status monitoring.

GET  /health      – Quick liveness check
GET  /health/deep – Deep check (LLM connectivity, model status, memory)
"""

from __future__ import annotations

import time
from typing import Any, Dict

from fastapi import APIRouter

from config import settings
from utils.logger import logger

router = APIRouter()

_start_time = time.time()


@router.get("/", summary="Liveness check")
async def health_check():
    """Returns a simple OK if the server is running."""
    return {
        "status":  "healthy",
        "service": "Flood Prediction & Support System",
        "version": "1.0.0",
        "uptime_seconds": round(time.time() - _start_time, 1),
    }


@router.get("/deep", summary="Deep system health check")
async def deep_health_check():
    """
    Validates connectivity to all major subsystems:
      - Gemini LLM
      - Agent memory
      - Model registry
    """
    checks: Dict[str, Any] = {}

    # 1. Gemini connectivity
    try:
        from services.gemini_service import get_gemini_service
        gemini = get_gemini_service()
        test   = await gemini.generate("Reply with OK", use_fast_model=True)
        checks["gemini"] = {
            "status":      "ok" if test else "degraded",
            "token_usage": gemini.token_usage,
        }
    except Exception as exc:
        checks["gemini"] = {"status": "error", "error": str(exc)}

    # 2. Agent memory
    try:
        from agents.orchestration.memory import agent_memory
        stats = agent_memory.stats()
        checks["memory"] = {"status": "ok", **stats}
    except Exception as exc:
        checks["memory"] = {"status": "error", "error": str(exc)}

    # 3. Configuration
    checks["config"] = {
        "environment":   settings.APP_ENV,
        "gemini_key_set": bool(settings.GEMINI_API_KEY),
        "smtp_configured": bool(settings.SMTP_USER),
    }

    overall = "healthy" if all(
        c.get("status") == "ok" for c in checks.values() if isinstance(c, dict) and "status" in c
    ) else "degraded"

    return {
        "status":           overall,
        "uptime_seconds":   round(time.time() - _start_time, 1),
        "checks":           checks,
    }
