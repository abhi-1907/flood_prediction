"""
Orchestration Agent package exports.

Expose the key classes and the module-level singleton so other modules
can import from `agents.orchestration` directly.
"""

from agents.orchestration.memory import (
    AgentMemory,
    AgentStep,
    Message,
    MessageRole,
    Session,
    StepStatus,
    agent_memory,
)
from agents.orchestration.planner import Planner
from agents.orchestration.tool_registry import ToolRegistry, ToolSchema, ToolResult
from agents.orchestration.context_manager import ContextManager
from agents.orchestration.orchestrator import OrchestratorAgent, orchestrator

__all__ = [
    # Memory
    "AgentMemory",
    "AgentStep",
    "Message",
    "MessageRole",
    "Session",
    "StepStatus",
    "agent_memory",
    # Planner
    "Planner",
    # Tool registry
    "ToolRegistry",
    "ToolSchema",
    "ToolResult",
    # Context
    "ContextManager",
    # Orchestrator
    "OrchestratorAgent",
    "orchestrator",
]
