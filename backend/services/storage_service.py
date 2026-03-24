"""
Storage Service – File I/O for uploaded datasets, model artifacts, and outputs.

Manages:
  - Saving and loading uploaded CSVs, JSON, Excel files
  - Writing agent output artifacts to disk (GeoJSON, reports, charts)
  - Session-based directory structure: data/<session_id>/
  - Cleanup of old session files
"""

from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from config import settings
from utils.logger import logger


class StorageService:
    """
    Manages reading and writing of dataset files and model output artifacts.

    Directory layout:
        {STORAGE_PATH}/
          uploads/                   # Raw uploaded files
          sessions/<session_id>/     # Per-session artifacts
          exports/                   # Public-facing outputs

    Usage:
        storage = StorageService()
        path = storage.save_upload(session_id, filename, content_bytes)
        df   = storage.load_dataframe(path)
        storage.save_artifact(session_id, "geojson", geojson_dict)
    """

    def __init__(self, base_path: Optional[str] = None) -> None:
        self._base  = Path(base_path or settings.STORAGE_PATH)
        self._uploads_dir = self._base / "uploads"
        self._sessions_dir = self._base / "sessions"
        self._exports_dir = self._base / "exports"
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        for d in [self._uploads_dir, self._sessions_dir, self._exports_dir]:
            d.mkdir(parents=True, exist_ok=True)

    # ── Upload handling ───────────────────────────────────────────────────

    def save_upload(
        self,
        session_id: str,
        filename:   str,
        content:    bytes,
    ) -> Path:
        """
        Saves an uploaded file to disk.

        Returns:
            Path to the saved file.
        """
        session_dir = self._uploads_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        dest = session_dir / filename
        dest.write_bytes(content)
        logger.info(f"[StorageService] Saved upload: {dest} ({len(content)} bytes)")
        return dest

    def load_dataframe(self, path: Union[str, Path]) -> Optional[pd.DataFrame]:
        """
        Loads a CSV, JSON, or Excel file into a DataFrame.

        Returns:
            DataFrame or None on failure.
        """
        path = Path(path)
        if not path.exists():
            logger.warning(f"[StorageService] File not found: {path}")
            return None

        suffix = path.suffix.lower()
        try:
            if suffix == ".csv":
                return pd.read_csv(path)
            elif suffix in (".xls", ".xlsx"):
                return pd.read_excel(path)
            elif suffix == ".json":
                return pd.read_json(path)
            else:
                # Try CSV as fallback
                return pd.read_csv(path)
        except Exception as exc:
            logger.error(f"[StorageService] Failed to load {path}: {exc}")
            return None

    # ── Session artifacts ─────────────────────────────────────────────────

    def save_artifact(
        self,
        session_id: str,
        name:       str,
        data:       Any,
    ) -> Path:
        """
        Saves an artifact (dict, list, string) as JSON for a session.

        Returns:
            Path to the saved file.
        """
        session_dir = self._sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        dest = session_dir / f"{name}.json"

        try:
            if isinstance(data, pd.DataFrame):
                data.to_json(dest, orient="records", indent=2)
            else:
                dest.write_text(
                    json.dumps(data, indent=2, default=str),
                    encoding="utf-8",
                )
            logger.info(f"[StorageService] Saved artifact: {dest}")
        except Exception as exc:
            logger.error(f"[StorageService] Failed to save artifact {name}: {exc}")

        return dest

    def load_artifact(
        self,
        session_id: str,
        name:       str,
    ) -> Optional[Any]:
        """Loads a JSON artifact for a session. Returns None if not found."""
        path = self._sessions_dir / session_id / f"{name}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error(f"[StorageService] Failed to load artifact {name}: {exc}")
            return None

    def list_session_artifacts(self, session_id: str) -> List[str]:
        """Lists all artifact names for a session."""
        session_dir = self._sessions_dir / session_id
        if not session_dir.exists():
            return []
        return [f.stem for f in session_dir.glob("*.json")]

    # ── Export handling ───────────────────────────────────────────────────

    def save_export(self, filename: str, data: Union[str, bytes, Any]) -> Path:
        """
        Saves a public export file (e.g., GeoJSON, PDF report).

        Returns:
            Path to the saved file.
        """
        dest = self._exports_dir / filename
        if isinstance(data, bytes):
            dest.write_bytes(data)
        elif isinstance(data, str):
            dest.write_text(data, encoding="utf-8")
        else:
            dest.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        logger.info(f"[StorageService] Saved export: {dest}")
        return dest

    # ── Cleanup ───────────────────────────────────────────────────────────

    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """
        Deletes session directories older than `max_age_hours`.

        Returns:
            Number of sessions removed.
        """
        count   = 0
        cutoff  = time.time() - max_age_hours * 3600
        for session_dir in self._sessions_dir.iterdir():
            if session_dir.is_dir() and session_dir.stat().st_mtime < cutoff:
                shutil.rmtree(session_dir, ignore_errors=True)
                count += 1
        if count:
            logger.info(f"[StorageService] Cleaned up {count} old sessions")
        return count

    def get_session_size_bytes(self, session_id: str) -> int:
        """Returns total disk usage for a session's artifacts in bytes."""
        session_dir = self._sessions_dir / session_id
        if not session_dir.exists():
            return 0
        return sum(f.stat().st_size for f in session_dir.rglob("*") if f.is_file())

    # ── Stats ─────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """Returns storage statistics."""
        sessions = list(self._sessions_dir.iterdir()) if self._sessions_dir.exists() else []
        return {
            "base_path":     str(self._base),
            "total_sessions": len(sessions),
            "uploads_dir":   str(self._uploads_dir),
            "exports_dir":   str(self._exports_dir),
        }


# ── Module-level singleton ────────────────────────────────────────────────────

_storage: Optional[StorageService] = None


def get_storage() -> StorageService:
    """Returns the module-level singleton StorageService."""
    global _storage
    if _storage is None:
        _storage = StorageService()
    return _storage
