"""
Logger – Centralised structured logging using Loguru.

All modules import `logger` from here to ensure consistent formatting,
log levels, and optional file output across the entire backend.
"""
import sys
from loguru import logger

# Remove the default handler
logger.remove()

# Console handler – coloured, human-readable
logger.add(
    sys.stderr,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    ),
    level="DEBUG",
    colorize=True,
)

# Optional rotating file handler (uncomment for production)
# logger.add(
#     "logs/floodsense_{time}.log",
#     rotation="50 MB",
#     retention="14 days",
#     compression="gz",
#     level="INFO",
#     enqueue=True,          # thread-safe
# )

__all__ = ["logger"]
