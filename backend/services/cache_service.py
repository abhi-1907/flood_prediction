"""
Cache Service – TTL-based in-memory cache for expensive API and model results.

Prevents redundant calls to:
  - Open-Meteo weather API
  - Google Maps Elevation API
  - Gemini LLM (for identical prompts)
  - ML model inference on same feature vectors

In production: swap `_store` for a Redis-backed adapter without changing the API.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, Optional, Tuple

from utils.logger import logger


class CacheService:
    """
    Thread-safe in-memory TTL cache.

    Usage:
        cache = CacheService(default_ttl=300)
        cache.set("weather:kochi", data, ttl=600)
        data = cache.get("weather:kochi")
    """

    def __init__(self, default_ttl: int = 300) -> None:
        """
        Args:
            default_ttl: Default time-to-live in seconds (default 5 min).
        """
        self._store: Dict[str, Tuple[Any, float]] = {}   # key → (value, expiry_ts)
        self._default_ttl = default_ttl
        self._hits   = 0
        self._misses = 0

    # ── Core API ──────────────────────────────────────────────────────────

    def get(self, key: str) -> Optional[Any]:
        """Returns the cached value or None if expired/missing."""
        entry = self._store.get(key)
        if entry is None:
            self._misses += 1
            return None

        value, expiry = entry
        if time.time() > expiry:
            del self._store[key]
            self._misses += 1
            return None

        self._hits += 1
        logger.debug(f"[CacheService] HIT: {key}")
        return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Stores a value with a TTL."""
        ttl = ttl if ttl is not None else self._default_ttl
        self._store[key] = (value, time.time() + ttl)
        logger.debug(f"[CacheService] SET: {key} (TTL={ttl}s)")

    def delete(self, key: str) -> bool:
        """Deletes a cache entry. Returns True if it existed."""
        if key in self._store:
            del self._store[key]
            return True
        return False

    def exists(self, key: str) -> bool:
        """Returns True if the key is in cache and not expired."""
        return self.get(key) is not None

    def clear(self) -> None:
        """Clears all cached entries."""
        self._store.clear()
        logger.info("[CacheService] Cache cleared")

    # ── Hash-based key helpers ────────────────────────────────────────────

    @staticmethod
    def make_key(*args: Any, prefix: str = "cache") -> str:
        """
        Generates a stable cache key from arbitrary arguments.

        Usage:
            key = CacheService.make_key("weather", lat, lon, prefix="api")
        """
        raw = json.dumps(args, sort_keys=True, default=str)
        digest = hashlib.md5(raw.encode()).hexdigest()[:12]
        return f"{prefix}:{digest}"

    # ── Bulk operations ───────────────────────────────────────────────────

    def evict_expired(self) -> int:
        """Removes all expired entries. Returns the number evicted."""
        now = time.time()
        expired_keys = [k for k, (_, exp) in self._store.items() if now > exp]
        for k in expired_keys:
            del self._store[k]
        if expired_keys:
            logger.info(f"[CacheService] Evicted {len(expired_keys)} expired entries")
        return len(expired_keys)

    # ── Statistics ────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """Returns hit/miss statistics and current size."""
        total = self._hits + self._misses
        return {
            "size":      len(self._store),
            "hits":      self._hits,
            "misses":    self._misses,
            "hit_rate":  round(self._hits / total, 3) if total else 0.0,
        }


# ── Module-level singleton ────────────────────────────────────────────────────

_cache: Optional[CacheService] = None


def get_cache(default_ttl: int = 300) -> CacheService:
    """Returns the module-level singleton CacheService."""
    global _cache
    if _cache is None:
        _cache = CacheService(default_ttl=default_ttl)
    return _cache
