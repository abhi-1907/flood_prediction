"""
Tool Registry – Catalogue of all callable agent tools available to the Orchestrator.

Each "tool" in the registry is an async callable that the Orchestrator can invoke
by name as part of executing a plan step. This follows the function-calling / tool-use
pattern popularised by agentic LLM frameworks.

Responsibilities:
  - Register agent entry-point callables with metadata (name, description, params schema)
  - Look up a tool by agent name
  - Validate that required input keys are present before invoking
  - Return a standardised ToolResult object

This design makes it trivially easy to add new agents later — just register
a new tool.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable, Dict, List, Optional


# ── Tool metadata ─────────────────────────────────────────────────────────────

@dataclass
class ToolSchema:
    """Metadata for a registered tool (mirrors OpenAI function schema convention)."""
    name: str
    description: str
    agent_class: str                            # Python class path (for docs)
    required_inputs: List[str] = field(default_factory=list)
    optional_inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)


@dataclass
class ToolResult:
    """Standardised return value from any tool invocation."""
    tool_name: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── Registry ──────────────────────────────────────────────────────────────────

class ToolRegistry:
    """
    Central catalogue of all agent tools available to the Orchestrator.

    Usage:
        registry = ToolRegistry()
        registry.register(
            name="data_ingestion",
            schema=ToolSchema(...),
            callable=ingestion_agent.run,
        )
        result = await registry.invoke("data_ingestion", inputs={...})
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Callable[..., Awaitable[Any]]] = {}
        self._schemas: Dict[str, ToolSchema] = {}

    # ── Registration ──────────────────────────────────────────────────────

    def register(
        self,
        name: str,
        schema: ToolSchema,
        callable_fn: Callable[..., Awaitable[Any]],
    ) -> None:
        """Register an async callable as a named tool."""
        self._tools[name] = callable_fn
        self._schemas[name] = schema

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)
        self._schemas.pop(name, None)

    # ── Lookup ────────────────────────────────────────────────────────────

    def get_schema(self, name: str) -> Optional[ToolSchema]:
        return self._schemas.get(name)

    def list_tools(self) -> List[ToolSchema]:
        return list(self._schemas.values())

    def list_tool_names(self) -> List[str]:
        return list(self._tools.keys())

    def is_registered(self, name: str) -> bool:
        return name in self._tools

    # ── Invocation ────────────────────────────────────────────────────────

    async def invoke(
        self,
        tool_name: str,
        inputs: Dict[str, Any],
        timeout: Optional[float] = 300.0,
    ) -> ToolResult:
        """
        Invokes a registered tool by name with the given inputs.

        Args:
            tool_name: Name of the registered tool/agent.
            inputs:    Dict of input arguments to pass to the tool.
            timeout:   Max seconds to wait; raises TimeoutError if exceeded.

        Returns:
            A ToolResult with success flag, output or error.
        """
        if not self.is_registered(tool_name):
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"No tool named '{tool_name}' is registered.",
            )

        # Validate required inputs
        schema = self._schemas[tool_name]
        missing = [k for k in schema.required_inputs if k not in inputs]
        if missing:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"Missing required inputs for '{tool_name}': {missing}",
            )

        try:
            fn = self._tools[tool_name]
            if timeout:
                output = await asyncio.wait_for(fn(**inputs), timeout=timeout)
            else:
                output = await fn(**inputs)
            return ToolResult(tool_name=tool_name, success=True, output=output)
        except asyncio.TimeoutError:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"Tool '{tool_name}' timed out after {timeout}s.",
            )
        except Exception as exc:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
            )

    # ── LLM description ───────────────────────────────────────────────────

    def describe_for_llm(self) -> str:
        """Returns a compact text description of all tools for use in LLM prompts."""
        lines = []
        for schema in self._schemas.values():
            req = ", ".join(schema.required_inputs) or "none"
            opt = ", ".join(schema.optional_inputs) or "none"
            lines.append(
                f"- {schema.name}: {schema.description}\n"
                f"  Required inputs: {req} | Optional: {opt} | Outputs: {', '.join(schema.outputs)}"
            )
        return "\n".join(lines)
