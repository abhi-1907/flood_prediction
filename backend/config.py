"""
Global configuration loaded from environment variables / .env file.

All modules should import `settings` from here rather than calling
os.getenv() directly, ensuring a single source of truth.
"""
from pydantic_settings import BaseSettings
from typing import List, Optional


class Settings(BaseSettings):
    # ── LLM ───────────────────────────────────────────────────────────────
    GEMINI_API_KEY: str = ""

    # ── External APIs ─────────────────────────────────────────────────────
    OPENMETEO_BASE_URL: str = "https://api.open-meteo.com/v1"
    # Open-Meteo is completely free and requires no API key.

    # ── Alerting – Email (SMTP) ────────────────────────────────────────────
    SMTP_HOST:     str = "smtp.gmail.com"
    SMTP_PORT:     int = 587
    SMTP_USER:     str = ""
    SMTP_PASSWORD: str = ""
    SMTP_FROM:     str = ""       # Defaults to SMTP_USER if blank

    # ── Alerting – SMS ─────────────────────────────────────────────────────
    # SMS is currently disabled (no provider configured).
    # To enable SMS alerts, add a provider (e.g. AWS SNS free tier) here.

    # ── Alerting – Push (Firebase) ─────────────────────────────────────────
    FIREBASE_CREDENTIALS_PATH: str = ""   # Path to service account JSON

    # ── Alerting – Webhooks ────────────────────────────────────────────────
    WEBHOOK_URL:        str = ""   # Default webhook endpoint (e.g. Slack)
    WEBHOOK_AUTH_TOKEN: str = ""   # Bearer token for webhook auth

    # ── App ────────────────────────────────────────────────────────────────
    APP_ENV:              str       = "development"
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://localhost:3000"]
    STORAGE_PATH:         str       = "./data"
    LOG_LEVEL:            str       = "DEBUG"    # DEBUG | INFO | WARNING | ERROR

    # ── Session management ────────────────────────────────────────────────
    SESSION_MAX_AGE_HOURS: int = 24   # Archive sessions older than this

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
