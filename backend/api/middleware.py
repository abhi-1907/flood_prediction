"""
API Middleware – CORS, request ID injection, structured access logging,
timing measurement, and error envelope formatting.

Usage:
    Registered in main.py via app.middleware("http")(log_and_time_requests)
"""

from __future__ import annotations

import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from utils.logger import logger


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that:
      1. Injects a unique X-Request-ID header for tracing
      2. Logs each request with method, path, status, and duration
      3. Adds timing headers (X-Process-Time-Ms)
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())[:12]
        request.state.request_id = request_id

        # Start timer
        start = time.perf_counter()

        # Process request
        try:
            response = await call_next(request)
        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.error(
                f"[{request_id}] {request.method} {request.url.path} → 500 "
                f"({duration_ms:.0f}ms) ERROR: {exc}"
            )
            raise

        # Calculate duration
        duration_ms = (time.perf_counter() - start) * 1000

        # Add headers
        response.headers["X-Request-ID"]      = request_id
        response.headers["X-Process-Time-Ms"] = f"{duration_ms:.0f}"

        # Log (skip noisy health checks at debug level)
        path = request.url.path
        if "/health" in path:
            logger.debug(
                f"[{request_id}] {request.method} {path} → {response.status_code} "
                f"({duration_ms:.0f}ms)"
            )
        else:
            logger.info(
                f"[{request_id}] {request.method} {path} → {response.status_code} "
                f"({duration_ms:.0f}ms)"
            )

        return response


class ErrorEnvelopeMiddleware(BaseHTTPMiddleware):
    """
    Catches unhandled exceptions and wraps them in a consistent JSON envelope:
    {"error": True, "status_code": 500, "detail": "..."}
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except Exception as exc:
            from fastapi.responses import JSONResponse
            logger.exception(f"Unhandled exception: {exc}")
            return JSONResponse(
                status_code=500,
                content={
                    "error":       True,
                    "status_code": 500,
                    "detail":      str(exc),
                    "request_id":  getattr(request.state, "request_id", None),
                },
            )
