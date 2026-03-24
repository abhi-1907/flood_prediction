"""
FastAPI application entry point for the Flood Prediction & Support System.

Start with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Interactive docs:
    http://localhost:8000/docs
    http://localhost:8000/redoc
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from config import settings
from api.middleware import ErrorEnvelopeMiddleware, RequestLoggingMiddleware
from api.routes import (
    alert_routes,
    health_routes,
    ingestion_routes,
    orchestration_routes,
    prediction_routes,
    recommendation_routes,
    simulation_routes,
)
from utils.logger import logger


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler — startup validation and graceful shutdown."""

    # ── Startup ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  FloodSense AI — Flood Prediction & Support System")
    logger.info(f"  Environment : {settings.APP_ENV}")
    logger.info(f"  Gemini key  : {'✓ set' if settings.GEMINI_API_KEY else '✗ MISSING'}")
    logger.info(f"  SMTP        : {'✓ configured' if settings.SMTP_USER else '○ dry-run mode'}")
    logger.info(f"  Firebase    : {'✓ configured' if settings.FIREBASE_CREDENTIALS_PATH else '○ dry-run mode'}")
    logger.info(f"  Storage     : {settings.STORAGE_PATH}")
    logger.info("=" * 60)

    if not settings.GEMINI_API_KEY:
        logger.warning(
            "GEMINI_API_KEY is not set. LLM-powered features will fail. "
            "Add it to your .env file."
        )

    # Pre-warm the Gemini service singleton so the first request is faster
    try:
        from services.gemini_service import get_gemini_service
        get_gemini_service()
        logger.info("Gemini service initialised.")
    except Exception as exc:
        logger.warning(f"Gemini service init failed: {exc}")

    # Ensure storage directories exist
    try:
        from services.storage_service import get_storage
        storage = get_storage()
        logger.info(f"Storage service ready: {storage.stats()['base_path']}")
    except Exception as exc:
        logger.warning(f"Storage service init failed: {exc}")

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────
    logger.info("FloodSense AI backend shutting down gracefully.")

    try:
        from services.storage_service import get_storage
        cleaned = get_storage().cleanup_old_sessions(settings.SESSION_MAX_AGE_HOURS)
        if cleaned:
            logger.info(f"Cleaned up {cleaned} old session directories.")
    except Exception:
        pass


# ── Application ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Flood Prediction & Support System",
    description=(
        "An agentic AI system for intelligent flood prediction, scenario simulation, "
        "and multi-channel alerting.\n\n"
        "**Agents:** Orchestration · Data Ingestion · Preprocessing · Prediction · "
        "Recommendation · Simulation · Alerting\n\n"
        "**Powered by:** Google Gemini 1.5 Pro/Flash · XGBoost · Random Forest · LSTM · "
        "SCS-CN Hydrological Model · Firebase FCM · Twilio"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "FloodSense AI",
        "url": "http://localhost:5173",
    },
    license_info={
        "name": "MIT",
    },
)

# ── Middleware (order matters — first added = outermost) ──────────────────────
# 1. Error envelope — catches anything that slips through
app.add_middleware(ErrorEnvelopeMiddleware)

# 2. Request logging + X-Request-ID injection
app.add_middleware(RequestLoggingMiddleware)

# 3. CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health_routes.router,         prefix="/health",          tags=["Health"])
app.include_router(orchestration_routes.router,  prefix="/orchestrate",     tags=["Orchestration"])
app.include_router(ingestion_routes.router,      prefix="/ingest",          tags=["Data Ingestion"])
app.include_router(prediction_routes.router,     prefix="/predict",         tags=["Prediction"])
app.include_router(recommendation_routes.router, prefix="/recommendations", tags=["Recommendations"])
app.include_router(simulation_routes.router,     prefix="/simulation",      tags=["Simulation"])
app.include_router(alert_routes.router,          prefix="/alerts",          tags=["Alerts"])


# ── Root redirect ─────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    """Redirects / → /docs for convenience."""
    return RedirectResponse(url="/docs")
