"""
Audit Logger – Records every preprocessing decision in a structured, human-readable trail.

Every transformation step (imputation, outlier removal, scaling, feature engineering)
is logged here with before/after statistics so the system is fully auditable and
explainable. The audit log is stored in the session and returned to the user.

This module is stateless — it receives and stores AuditEntry objects passed to it
by each cleaner/transformer, then compiles them into a PreprocessingAudit report.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from agents.preprocessing.preprocessing_schemas import (
    AuditEntry,
    PreprocessingAudit,
)
from utils.logger import logger


class AuditLogger:
    """
    Collects and formats audit entries from all preprocessing steps.

    Usage:
        audit_logger = AuditLogger(session_id="abc123")
        audit_logger.log(step="missing_value_handler",
                         action="imputed 12 nulls with median",
                         column="rainfall_mm",
                         rows_affected=12,
                         before_stat=12,
                         after_stat=0)
        audit = audit_logger.compile(input_df, output_df)
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._entries: List[AuditEntry] = []
        self._started_at = datetime.utcnow()

    # ── Logging ───────────────────────────────────────────────────────────

    def log(
        self,
        step:         str,
        action:       str,
        column:       Optional[str] = None,
        before_stat:  Optional[Any] = None,
        after_stat:   Optional[Any] = None,
        rows_affected: int          = 0,
        strategy:     Optional[str] = None,
    ) -> None:
        """Records one preprocessing decision."""
        entry = AuditEntry(
            step=step,
            action=action,
            column=column,
            before_stat=before_stat,
            after_stat=after_stat,
            rows_affected=rows_affected,
            strategy=strategy,
        )
        self._entries.append(entry)
        logger.debug(
            f"[AuditLogger] [{step}] {action}"
            + (f" | col={column}" if column else "")
            + (f" | rows={rows_affected}" if rows_affected else "")
        )

    def log_column_stats(
        self,
        step:   str,
        df:     pd.DataFrame,
        column: str,
        action: str,
    ) -> None:
        """Convenience method that auto-extracts before stats and logs them."""
        if column not in df.columns:
            return
        self.log(
            step=step,
            action=action,
            column=column,
            before_stat={
                "null_count": int(df[column].isna().sum()),
                "mean":       round(float(df[column].mean()), 4) if pd.api.types.is_numeric_dtype(df[column]) else None,
                "std":        round(float(df[column].std()), 4)  if pd.api.types.is_numeric_dtype(df[column]) else None,
                "min":        round(float(df[column].min()), 4)  if pd.api.types.is_numeric_dtype(df[column]) else None,
                "max":        round(float(df[column].max()), 4)  if pd.api.types.is_numeric_dtype(df[column]) else None,
            },
        )

    def log_shape(self, step: str, df: pd.DataFrame, action: str = "") -> None:
        """Logs the current shape of the DataFrame as a milestone."""
        self.log(
            step=step,
            action=action or f"DataFrame shape: {df.shape[0]} rows × {df.shape[1]} cols",
            before_stat={"rows": df.shape[0], "cols": df.shape[1]},
        )

    # ── Compile ───────────────────────────────────────────────────────────

    def compile(
        self,
        input_df:  pd.DataFrame,
        output_df: pd.DataFrame,
    ) -> PreprocessingAudit:
        """Compiles all logged entries into a final PreprocessingAudit report."""
        input_rows, input_cols   = input_df.shape
        output_rows, output_cols = output_df.shape

        audit = PreprocessingAudit(
            session_id   = self.session_id,
            entries      = list(self._entries),
            input_shape  = (input_rows, input_cols),
            output_shape = (output_rows, output_cols),
            rows_dropped = max(0, input_rows - output_rows),
            cols_dropped = max(0, input_cols - output_cols),
        )

        elapsed = (datetime.utcnow() - self._started_at).total_seconds()
        logger.info(
            f"[AuditLogger] Preprocessing complete | "
            f"{input_rows}×{input_cols} → {output_rows}×{output_cols} | "
            f"{audit.rows_dropped} rows dropped | "
            f"{len(self._entries)} audit entries | "
            f"{elapsed:.2f}s"
        )
        return audit

    # ── Human-readable report ─────────────────────────────────────────────

    def to_markdown(self) -> str:
        """Returns a Markdown-formatted audit report for display to users."""
        lines = [
            f"# Preprocessing Audit — Session `{self.session_id}`\n",
            f"**Total steps logged:** {len(self._entries)}\n",
            "| Step | Column | Action | Rows Affected | Strategy |",
            "|------|--------|--------|---------------|----------|",
        ]
        for e in self._entries:
            lines.append(
                f"| {e.step} | {e.column or '—'} | {e.action} "
                f"| {e.rows_affected} | {e.strategy or '—'} |"
            )
        return "\n".join(lines)

    def get_entries(self) -> List[Dict]:
        """Returns all entries as plain dicts (for JSON serialisation)."""
        return [e.model_dump() for e in self._entries]

    def clear(self) -> None:
        self._entries.clear()
