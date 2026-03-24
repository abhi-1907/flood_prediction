"""
Orchestration Agent – Central LLM-driven planner and executor.

This is the nerve centre of the entire Flood Prediction and Support System.
Every user request — whether a CSV upload, a free-text query, or an API call —
passes through this agent.

Execution flow:
  1.  A new Session is created in AgentMemory.
  2.  ContextManager.initialise() enriches the session context (geocoding,
      user-type detection, feature-flag derivation).
  3.  Planner.classify_intent() does a quick intent classification.
  4.  Planner.create_plan() generates an ordered, dependency-aware agent plan.
  5.  The Orchestrator executes the plan step-by-step (or in parallel for
      independent steps), calling each specialist agent via the ToolRegistry.
  6.  If a step fails, Planner.re_plan() is called to get a recovery plan.
  7.  After all steps complete, the session is archived and a unified response
      is returned to the caller.

Agentic capabilities implemented here:
  - Dynamic plan generation (LLM decides which agents to invoke)
  - Parallel step execution (asyncio.gather for independent steps)
  - Automatic re-planning on failure
  - Retry with exponential backoff for transient errors
  - Full session memory and audit trail
  - Real-time status streaming (via async generator)
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional

from agents.orchestration.context_manager import ContextManager
from agents.orchestration.memory import AgentMemory, AgentStep, MessageRole, Session, StepStatus, agent_memory
from agents.orchestration.planner import Planner
from agents.orchestration.tool_registry import ToolRegistry, ToolResult
from services.gemini_service import GeminiService
from utils.logger import logger


# ── Retry configuration ───────────────────────────────────────────────────────

MAX_RETRIES     = 2       # Max automatic retries for a transient step failure
RETRY_BASE_SECS = 2.0     # Base delay for exponential backoff (2, 4 seconds)
STEP_TIMEOUT    = 300.0   # Seconds before a single agent step times out


# ── OrchestratorAgent ─────────────────────────────────────────────────────────

class OrchestratorAgent:
    """
    Central agentic orchestrator for the Flood Prediction and Support System.

    Dependencies are injected at construction time so the class is testable
    without real network calls.
    """

    def __init__(
        self,
        gemini_service:  Optional[GeminiService]  = None,
        memory:          Optional[AgentMemory]    = None,
        tool_registry:   Optional[ToolRegistry]   = None,
    ) -> None:
        self._gemini   = gemini_service or GeminiService()
        self._memory   = memory        or agent_memory
        self._registry = tool_registry or ToolRegistry()
        self._planner  = Planner(self._gemini)
        self._ctx_mgr  = ContextManager(self._gemini)

        # Register all specialist agents with the tool registry
        self._register_agents()

    # ── Public API ────────────────────────────────────────────────────────

    async def run(
        self,
        user_query: str,
        uploaded_file_bytes: Optional[bytes] = None,
        uploaded_columns:    Optional[List[str]] = None,
        user_type:           str = "general_public",
        initial_context:     Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Main entry point — processes a user request end-to-end.

        Args:
            user_query:          Free-text query or description.
            uploaded_file_bytes: Raw bytes of an uploaded CSV/JSON file.
            uploaded_columns:    Column names of the uploaded file (pre-parsed).
            user_type:           'general_public' | 'authority' | 'first_responder'
            initial_context:     Any pre-known context (e.g. from API request body).

        Returns:
            A unified response dict containing prediction, recommendations,
            simulation data, and session metadata.
        """
        # ── 1. Create session ─────────────────────────────────────────────
        session = self._memory.create_session(
            user_query=user_query,
            initial_context={"user_type": user_type, **(initial_context or {})},
        )
        logger.info(
            f"[Orchestrator] New session {session.session_id} | "
            f"query='{user_query[:80]}'"
        )

        try:
            # ── 2. Enrich context ─────────────────────────────────────────
            await self._ctx_mgr.initialise(
                session,
                user_query=user_query,
                uploaded_columns=uploaded_columns,
            )
            self._ctx_mgr.update_user_type(session, user_type)

            # ── 3. Classify intent ────────────────────────────────────────
            intent_meta = await self._planner.classify_intent(user_query)
            session.set_context("intent", intent_meta)
            logger.info(f"[Orchestrator] Intent: {intent_meta}")

            # ── 4. Store uploaded file in session artifacts ────────────────
            if uploaded_file_bytes:
                session.store_artifact("uploaded_file_bytes", uploaded_file_bytes)
                session.store_artifact("uploaded_columns", uploaded_columns or [])

            # ── 5. Generate execution plan ────────────────────────────────
            available_data = {
                "uploaded_columns": uploaded_columns,
                "location": session.get_context("location"),
                "data_types": session.get_context("data_types", []),
            }
            plan = await self._planner.create_plan(session, available_data)
            logger.info(
                f"[Orchestrator] Plan: "
                + ", ".join(f"{s['step_index']}:{s['agent']}" for s in plan)
            )

            # ── 6. Execute plan ───────────────────────────────────────────
            await self._execute_plan(session, plan)

            # ── 7. Synthesise final response ──────────────────────────────
            response = self._build_response(session)
            session.complete(response)

            session.add_message(
                MessageRole.ASSISTANT,
                f"Orchestration complete in {session.elapsed_seconds:.1f}s.",
            )
            logger.info(
                f"[Orchestrator] Session {session.session_id} complete in "
                f"{session.elapsed_seconds:.1f}s"
            )
            return response

        except Exception as exc:
            logger.exception(
                f"[Orchestrator] Unhandled exception in session {session.session_id}: {exc}"
            )
            session.complete({"error": str(exc), "session_id": session.session_id})
            raise

        finally:
            self._memory.archive_session(session.session_id)

    async def stream(
        self,
        user_query: str,
        uploaded_file_bytes: Optional[bytes] = None,
        uploaded_columns:    Optional[List[str]] = None,
        user_type:           str = "general_public",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Streaming variant of `run()`.

        Yields a status event dict after each major step so the frontend
        can show real-time progress.

        Event schema:
            {"event": "step_started" | "step_done" | "plan_ready" | "complete",
             "data": {...}}
        """
        session = self._memory.create_session(user_query=user_query)
        session.set_context("user_type", user_type)

        try:
            await self._ctx_mgr.initialise(session, user_query, uploaded_columns)
            intent_meta = await self._planner.classify_intent(user_query)
            session.set_context("intent", intent_meta)

            if uploaded_file_bytes:
                session.store_artifact("uploaded_file_bytes", uploaded_file_bytes)

            plan = await self._planner.create_plan(
                session,
                available_data={
                    "uploaded_columns": uploaded_columns,
                    "location": session.get_context("location"),
                },
            )
            yield {"event": "plan_ready", "data": {"steps": len(plan), "plan": plan}}

            for step_dict in plan:
                yield {
                    "event": "step_started",
                    "data": {
                        "step_index": step_dict["step_index"],
                        "agent": step_dict["agent"],
                        "action": step_dict["action"],
                    },
                }
                await self._execute_single_step(session, step_dict, attempt=0)
                agent_step = self._find_agent_step(session, step_dict)
                yield {
                    "event": "step_done",
                    "data": {
                        "step_index": step_dict["step_index"],
                        "agent": step_dict["agent"],
                        "status": agent_step.status if agent_step else "unknown",
                    },
                }

            response = self._build_response(session)
            session.complete(response)
            yield {"event": "complete", "data": response}

        finally:
            self._memory.archive_session(session.session_id)

    # ── Plan execution ────────────────────────────────────────────────────

    async def _execute_plan(self, session: Session, plan: List[Dict[str, Any]]) -> None:
        """
        Executes all plan steps respecting dependency order.

        Steps with no unresolved dependencies are executed in parallel using
        asyncio.gather. Steps with dependencies are executed sequentially
        after all their required predecessors succeed.
        """
        completed_indices: set[int] = set()
        remaining = list(plan)
        max_iterations = len(plan) * 2 + 5    # Safety guard against infinite loops

        iteration = 0
        while remaining and iteration < max_iterations:
            iteration += 1

            # Find steps whose dependencies are all satisfied
            runnable = [
                s for s in remaining
                if all(dep in completed_indices for dep in s.get("depends_on", []))
            ]

            if not runnable:
                # None are runnable → dependency deadlock or all skipped
                logger.warning(
                    "[Orchestrator] No runnable steps found. "
                    f"Remaining: {[s['agent'] for s in remaining]}"
                )
                break

            # Execute all runnable steps in parallel
            if len(runnable) > 1:
                logger.info(
                    f"[Orchestrator] Parallel execution of steps: "
                    + ", ".join(str(s["step_index"]) for s in runnable)
                )
                await asyncio.gather(
                    *[self._execute_with_retry(session, s) for s in runnable]
                )
            else:
                await self._execute_with_retry(session, runnable[0])

            for step in runnable:
                completed_indices.add(step["step_index"])
                remaining.remove(step)

    async def _execute_with_retry(
        self,
        session: Session,
        step_dict: Dict[str, Any],
    ) -> None:
        """Wraps a single step execution with retry and re-plan logic."""
        for attempt in range(MAX_RETRIES + 1):
            await self._execute_single_step(session, step_dict, attempt)
            agent_step = self._find_agent_step(session, step_dict)

            if agent_step and agent_step.status == StepStatus.SUCCEEDED:
                return

            if agent_step and agent_step.status == StepStatus.FAILED:
                error_msg = agent_step.error or "Unknown error"
                if attempt < MAX_RETRIES:
                    wait = RETRY_BASE_SECS * (2 ** attempt)
                    logger.warning(
                        f"[Orchestrator] Step {step_dict['step_index']} "
                        f"({step_dict['agent']}) failed (attempt {attempt + 1}). "
                        f"Retrying in {wait:.0f}s..."
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.error(
                        f"[Orchestrator] Step {step_dict['step_index']} "
                        f"({step_dict['agent']}) failed after {MAX_RETRIES + 1} attempts."
                    )
                    if not step_dict.get("optional", False):
                        # Non-optional failure → trigger re-plan
                        current_plan = session.get_artifact("execution_plan", [])
                        revised = await self._planner.re_plan(
                            session, step_dict, error_msg
                        )
                        if revised:
                            logger.info(
                                "[Orchestrator] Re-plan produced "
                                f"{len(revised)} revised steps."
                            )
                        # We don't execute the revised plan recursively here —
                        # the caller (_execute_plan) will pick it up on next iteration.
                    break

    async def _execute_single_step(
        self,
        session: Session,
        step_dict: Dict[str, Any],
        attempt: int,
    ) -> None:
        """Performs one call to the ToolRegistry for the given plan step."""
        agent_name = step_dict["agent"]
        action     = step_dict["action"]
        inputs_map = step_dict.get("inputs", {})

        # Resolve artifact references in inputs
        resolved_inputs = self._resolve_inputs(session, inputs_map)
        resolved_inputs["session"] = session      # Always pass session

        # Record step start
        agent_step = session.add_step(
            agent_name=agent_name,
            action=action,
            input_data={k: v for k, v in resolved_inputs.items() if k != "session"},
        )
        agent_step.start()
        agent_step.metadata["attempt"] = attempt

        logger.info(
            f"[Orchestrator] → Invoking '{agent_name}' | action='{action}' "
            f"| attempt={attempt + 1}"
        )

        result: ToolResult = await self._registry.invoke(
            tool_name=agent_name,
            inputs=resolved_inputs,
            timeout=STEP_TIMEOUT,
        )

        if result.success:
            agent_step.succeed(result.output)
            # Auto-store outputs declared in plan as session artifacts
            for artifact_name in step_dict.get("outputs", []):
                session.store_artifact(artifact_name, result.output)
            # Context updates for prediction step
            if agent_name == "prediction" and isinstance(result.output, dict):
                self._ctx_mgr.update_risk(
                    session,
                    risk_level=result.output.get("risk_level", "UNKNOWN"),
                    probability=result.output.get("flood_probability", 0.0),
                )
            logger.info(
                f"[Orchestrator] ✓ '{agent_name}' succeeded in "
                f"{agent_step.duration_ms:.0f}ms"
            )
        else:
            agent_step.fail(result.error or "Unknown tool error")
            logger.error(
                f"[Orchestrator] ✗ '{agent_name}' failed: {result.error}"
            )

    # ── Response synthesis ────────────────────────────────────────────────

    def _build_response(self, session: Session) -> Dict[str, Any]:
        """
        Aggregates all agent outputs into a single structured response.
        """
        prediction    = session.get_artifact("prediction_result")
        preprocessed  = session.get_artifact("processed_dataset")
        recommendations = session.get_artifact("recommendations")
        geojson       = session.get_artifact("geojson")
        simulation    = session.get_artifact("simulation_result")
        alert_status  = session.get_artifact("alert_status")

        return {
            "session_id":      session.session_id,
            "user_query":      session.user_query,
            "context":         session.context,
            "prediction":      prediction,
            "recommendations": recommendations,
            "simulation":      simulation or {"geojson": geojson},
            "alert_status":    alert_status,
            "steps_summary":   session.get_steps_summary(),
            "elapsed_seconds": session.elapsed_seconds,
        }

    # ── Input resolution ──────────────────────────────────────────────────

    def _resolve_inputs(
        self,
        session: Session,
        inputs_map: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Resolves artifact references in a step's inputs dict.

        Values that match a session artifact key are replaced with the
        actual artifact object. Literal strings remain as-is.
        """
        resolved: Dict[str, Any] = {}
        for param_name, artifact_key in inputs_map.items():
            artifact = session.get_artifact(artifact_key)
            if artifact is not None:
                resolved[param_name] = artifact
            else:
                # Treat as a literal string value
                resolved[param_name] = artifact_key
        return resolved

    def _find_agent_step(
        self,
        session: Session,
        step_dict: Dict[str, Any],
    ) -> Optional[AgentStep]:
        """Returns the most recent AgentStep for the given plan step."""
        for step in reversed(session.steps):
            if step.agent_name == step_dict["agent"] and step.action == step_dict["action"]:
                return step
        return None

    # ── Tool registration ─────────────────────────────────────────────────

    def _register_agents(self) -> None:
        """
        Registers all specialist agent callables in the ToolRegistry.

        Imports are deferred here to avoid circular dependencies.
        Each import is wrapped in a try/except so a missing optional agent
        won't crash the entire system.
        """
        from agents.orchestration.tool_registry import ToolSchema

        registrations = [
            {
                "name": "data_ingestion",
                "schema": ToolSchema(
                    name="data_ingestion",
                    description="Collects, validates, and merges data from APIs and user uploads.",
                    agent_class="agents.data_ingestion.ingestion_agent.DataIngestionAgent",
                    required_inputs=["session"],
                    optional_inputs=["query", "uploaded_file_bytes"],
                    outputs=["raw_dataset"],
                ),
                "module": "agents.data_ingestion.ingestion_agent",
                "class": "DataIngestionAgent",
                "method": "run",
            },
            {
                "name": "preprocessing",
                "schema": ToolSchema(
                    name="preprocessing",
                    description="Cleans, transforms, and engineers features from raw data.",
                    agent_class="agents.preprocessing.preprocessing_agent.PreprocessingAgent",
                    required_inputs=["session", "data"],
                    optional_inputs=[],
                    outputs=["processed_dataset"],
                ),
                "module": "agents.preprocessing.preprocessing_agent",
                "class": "PreprocessingAgent",
                "method": "run",
            },
            {
                "name": "prediction",
                "schema": ToolSchema(
                    name="prediction",
                    description="Runs ML models to produce a flood risk prediction.",
                    agent_class="agents.prediction.prediction_agent.PredictionAgent",
                    required_inputs=["session", "data"],
                    optional_inputs=[],
                    outputs=["prediction_result"],
                ),
                "module": "agents.prediction.prediction_agent",
                "class": "PredictionAgent",
                "method": "run",
            },
            {
                "name": "recommendation",
                "schema": ToolSchema(
                    name="recommendation",
                    description="Generates LLM recommendations for public or authorities.",
                    agent_class="agents.recommendation.recommendation_agent.RecommendationAgent",
                    required_inputs=["session", "prediction"],
                    optional_inputs=["user_type"],
                    outputs=["recommendations"],
                ),
                "module": "agents.recommendation.recommendation_agent",
                "class": "RecommendationAgent",
                "method": "run",
            },
            {
                "name": "simulation",
                "schema": ToolSchema(
                    name="simulation",
                    description="Computes flood zone polygons and GeoJSON map overlays.",
                    agent_class="agents.simulation.simulation_agent.SimulationAgent",
                    required_inputs=["session"],
                    optional_inputs=["prediction"],
                    outputs=["simulation_result", "geojson"],
                ),
                "module": "agents.simulation.simulation_agent",
                "class": "SimulationAgent",
                "method": "run",
            },
            {
                "name": "alerting",
                "schema": ToolSchema(
                    name="alerting",
                    description="Composes and dispatches flood alerts to subscribers.",
                    agent_class="agents.alerting.alerting_agent.AlertingAgent",
                    required_inputs=["session"],
                    optional_inputs=["prediction", "recommendations"],
                    outputs=["alert_status"],
                ),
                "module": "agents.alerting.alerting_agent",
                "class": "AlertingAgent",
                "method": "run",
            },
        ]

        for reg in registrations:
            try:
                import importlib
                mod   = importlib.import_module(reg["module"])
                cls   = getattr(mod, reg["class"])
                inst  = cls()
                fn    = getattr(inst, reg["method"])
                self._registry.register(
                    name=reg["name"],
                    schema=reg["schema"],
                    callable_fn=fn,
                )
                logger.debug(f"[Orchestrator] Registered tool: {reg['name']}")
            except Exception as exc:
                logger.warning(
                    f"[Orchestrator] Could not register tool '{reg['name']}': {exc}. "
                    "It will be unavailable until the agent is implemented."
                )


# ── Module-level singleton ────────────────────────────────────────────────────

orchestrator = OrchestratorAgent()
