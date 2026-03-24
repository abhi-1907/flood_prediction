"""
Memory – Manages short-term and long-term memory for the agentic system.

Stores:
  - Conversation history (user ↔ LLM messages)
  - Agent execution steps with inputs/outputs
  - Intermediate data artifacts (DataFrames, predictions, GeoJSON)
  - Session metadata and flags

Architecture:
  - Each user request creates a Session (with a UUID).
  - The session accumulates AgentStep records as the plan executes.
  - Finished sessions are archived for auditability.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


# ── Enumerations ──────────────────────────────────────────────────────────────

class StepStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    SUCCEEDED = "succeeded"
    FAILED    = "failed"
    SKIPPED   = "skipped"


class MessageRole(str, Enum):
    USER      = "user"
    ASSISTANT = "assistant"
    SYSTEM    = "system"
    AGENT     = "agent"


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Message:
    """A single entry in the conversation history."""
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentStep:
    """Records one step executed by a specialist agent."""
    step_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    agent_name: str = ""
    action: str = ""
    input_data: Any = None
    output_data: Any = None
    status: StepStatus = StepStatus.PENDING
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def start(self) -> None:
        self.status = StepStatus.RUNNING
        self.started_at = datetime.utcnow()

    def succeed(self, output: Any) -> None:
        self.output_data = output
        self.status = StepStatus.SUCCEEDED
        self.finished_at = datetime.utcnow()
        if self.started_at:
            self.duration_ms = (self.finished_at - self.started_at).total_seconds() * 1000

    def fail(self, error: str) -> None:
        self.error = error
        self.status = StepStatus.FAILED
        self.finished_at = datetime.utcnow()


@dataclass
class Session:
    """
    A complete orchestration session for a single user request.

    Holds:
      - session_id     : Unique identifier
      - user_query     : Original user input
      - conversation   : Full message history
      - steps          : Ordered list of agent execution steps
      - artifacts      : Named data artifacts (DataFrames, GeoJSON, etc.)
      - context        : Arbitrary key-value context (location, risk level, etc.)
      - created_at     : Session creation time
      - completed_at   : Session completion time (None if still running)
      - final_result   : Final structured output returned to the caller
    """
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_query: str = ""
    conversation: List[Message] = field(default_factory=list)
    steps: List[AgentStep] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    final_result: Optional[Any] = None

    # ── Conversation helpers ───────────────────────────────────────────────

    def add_message(self, role: MessageRole, content: str, **metadata) -> Message:
        msg = Message(role=role, content=content, metadata=metadata)
        self.conversation.append(msg)
        return msg

    def get_conversation_text(self) -> str:
        """Returns the full conversation as a formatted string for LLM context."""
        lines = []
        for msg in self.conversation:
            lines.append(f"[{msg.role.upper()}] {msg.content}")
        return "\n".join(lines)

    # ── Step helpers ──────────────────────────────────────────────────────

    def add_step(self, agent_name: str, action: str, input_data: Any = None) -> AgentStep:
        step = AgentStep(
            agent_name=agent_name,
            action=action,
            input_data=input_data,
        )
        self.steps.append(step)
        return step

    def get_last_step_output(self, agent_name: str) -> Optional[Any]:
        """Returns the most recent successful output from a given agent."""
        for step in reversed(self.steps):
            if step.agent_name == agent_name and step.status == StepStatus.SUCCEEDED:
                return step.output_data
        return None

    def get_steps_summary(self) -> List[Dict[str, Any]]:
        return [
            {
                "step_id": s.step_id,
                "agent": s.agent_name,
                "action": s.action,
                "status": s.status,
                "duration_ms": s.duration_ms,
                "error": s.error,
            }
            for s in self.steps
        ]

    # ── Artifact helpers ──────────────────────────────────────────────────

    def store_artifact(self, key: str, value: Any) -> None:
        self.artifacts[key] = value

    def get_artifact(self, key: str, default: Any = None) -> Any:
        return self.artifacts.get(key, default)

    # ── Context helpers ───────────────────────────────────────────────────

    def set_context(self, key: str, value: Any) -> None:
        self.context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        return self.context.get(key, default)

    # ── Completion ────────────────────────────────────────────────────────

    def complete(self, result: Any) -> None:
        self.final_result = result
        self.completed_at = datetime.utcnow()

    @property
    def is_complete(self) -> bool:
        return self.completed_at is not None

    @property
    def elapsed_seconds(self) -> Optional[float]:
        if self.completed_at:
            return (self.completed_at - self.created_at).total_seconds()
        return (datetime.utcnow() - self.created_at).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_query": self.user_query,
            "steps": self.get_steps_summary(),
            "context": self.context,
            "elapsed_seconds": self.elapsed_seconds,
            "is_complete": self.is_complete,
            "final_result": self.final_result,
        }


# ── Memory Store ──────────────────────────────────────────────────────────────

class AgentMemory:
    """
    In-process memory store for all active and archived orchestration sessions.

    Usage:
        memory = AgentMemory()
        session = memory.create_session(user_query="Flood risk in Kochi?")
        # ... agents update session ...
        memory.archive_session(session.session_id)
    """

    def __init__(self, max_active: int = 50, max_archived: int = 200) -> None:
        self._active: Dict[str, Session] = {}
        self._archived: Dict[str, Session] = {}
        self._max_active = max_active
        self._max_archived = max_archived

    # ── Session lifecycle ─────────────────────────────────────────────────

    def create_session(self, user_query: str, initial_context: Optional[Dict] = None) -> Session:
        """Creates a brand-new session and registers it as active."""
        session = Session(user_query=user_query)
        session.add_message(MessageRole.USER, user_query)
        if initial_context:
            session.context.update(initial_context)
        self._active[session.session_id] = session
        self._evict_if_needed()
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        return self._active.get(session_id) or self._archived.get(session_id)

    def archive_session(self, session_id: str) -> None:
        """Moves a completed session from active → archived."""
        if session_id in self._active:
            session = self._active.pop(session_id)
            self._archived[session.session_id] = session
            if len(self._archived) > self._max_archived:
                # Remove oldest archived session
                oldest = next(iter(self._archived))
                del self._archived[oldest]

    def delete_session(self, session_id: str) -> None:
        self._active.pop(session_id, None)
        self._archived.pop(session_id, None)

    # ── Convenience accessors ─────────────────────────────────────────────

    def list_active(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self._active.values()]

    def list_archived(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self._archived.values()]

    def stats(self) -> Dict[str, int]:
        return {
            "active_sessions": len(self._active),
            "archived_sessions": len(self._archived),
        }

    # ── Internals ─────────────────────────────────────────────────────────

    def _evict_if_needed(self) -> None:
        """Removes the oldest active session if the limit is exceeded."""
        if len(self._active) > self._max_active:
            oldest_id = next(iter(self._active))
            self.archive_session(oldest_id)


# ── Module-level singleton (imported by other modules) ────────────────────────
agent_memory = AgentMemory()
