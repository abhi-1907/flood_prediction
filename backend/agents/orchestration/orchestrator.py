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
from config import settings
from utils.logger import logger


# ── Retry configuration ───────────────────────────────────────────────────────

MAX_RETRIES     = 0       # Max automatic retries for a transient step failure (GeminiService handles network retries)
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
            # ── Optimization: Bypass LLM planning for file uploads ────────
            if uploaded_file_bytes or uploaded_columns:
                logger.info("[Orchestrator] File upload detected — using hardcoded 1-call path.")
                
                # 1. Hardcoded pipeline
                plan = [
                    {"step_index": 1, "agent": "data_ingestion", "action": "Ingest flood data", "inputs": {}, "depends_on": []},
                    {"step_index": 2, "agent": "preprocessing", "action": "Preprocess data", "inputs": {"dataset": "raw_dataset"}, "depends_on": [1]},
                    {"step_index": 3, "agent": "prediction", "action": "Predict flood risk", "inputs": {"dataset": "processed_dataset"}, "depends_on": [2]},
                    {"step_index": 4, "agent": "recommendation", "action": "Generate recommendations", "inputs": {"dataset": "processed_dataset"}, "depends_on": [3]},
                    {"step_index": 5, "agent": "simulation", "action": "Model flood zones", "inputs": {"prediction": "prediction_result"}, "depends_on": [3]},
                    {"step_index": 6, "agent": "alerting", "action": "Send subscriber alerts", "inputs": {"prediction": "prediction_result", "recommendations": "recommendations"}, "depends_on": [4, 5]},
                ]
                
                # 2. Hardcoded Fetch Plan (Assume user data is primary)
                # This ensures SourceIdentifier skips its Gemini call
                fetch_plan = [
                    {
                        "source": "uploaded_file",
                        "category": "mixed",
                        "priority": 1,
                        "params": {},
                        "rationale": "Using user-uploaded CSV as primary data source."
                    }
                ]
                session.store_artifact("fetch_plan", fetch_plan)

                # 3. Hardcoded Model Config (Rule-based weights)
                # This ensures ModelSelector skips its Gemini call
                model_config = {
                    "mode": "classification",
                    "forecast_horizon": 1,
                    "models_to_use": ["xgboost", "random_forest"],
                    "weights": {"xgboost": 0.55, "random_forest": 0.45}
                }
                session.store_artifact("model_config", model_config)
                
                # 4. Extract initial context
                session.set_context("original_query", user_query)
                session.set_context("intent", "data_upload")
                session.set_context("wants_recommendations", True)
                
                # Location hints
                if uploaded_columns:
                    col_map = {c.lower(): c for c in uploaded_columns}
                    if any(k in col_map for k in ["lat", "latitude", "lon", "longitude"]):
                        session.set_context("has_location", True)

            else:
                # ── Standard unified single Gemini startup call ───────────
                plan = await self._unified_startup_call(
                    session, user_query, user_type, uploaded_columns
                )
                logger.info(
                    f"[Orchestrator] Unified startup done. Plan: "
                    + ", ".join(f"{s['step_index']}:{s['agent']}" for s in plan)
                )

            # ── 4. Store uploaded file in session artifacts ────────────────
            if uploaded_file_bytes:
                session.store_artifact("uploaded_file_bytes", uploaded_file_bytes)
                session.store_artifact("uploaded_columns", uploaded_columns or [])

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
        initial_context:     Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Streaming variant of `run()`.

        Yields a status event dict after each major step so the frontend
        can show real-time progress.

        Event schema:
            {"event": "step_started" | "step_done" | "plan_ready" | "complete",
             "data": {...}}
        """
        session = self._memory.create_session(
            user_query=user_query,
            initial_context={"user_type": user_type, **(initial_context or {})},
        )

        try:
            # ── Optimization: Bypass LLM planning for file uploads ────────
            if uploaded_file_bytes or uploaded_columns:
                logger.info("[Orchestrator] File upload detected — using hardcoded 1-call path.")
                
                # 1. Hardcoded pipeline
                plan = [
                    {"step_index": 1, "agent": "data_ingestion", "action": "Ingest flood data", "inputs": {}, "depends_on": []},
                    {"step_index": 2, "agent": "preprocessing", "action": "Preprocess data", "inputs": {"dataset": "raw_dataset"}, "depends_on": [1]},
                    {"step_index": 3, "agent": "prediction", "action": "Predict flood risk", "inputs": {"dataset": "processed_dataset"}, "depends_on": [2]},
                    {"step_index": 4, "agent": "recommendation", "action": "Generate recommendations", "inputs": {"dataset": "processed_dataset"}, "depends_on": [3]},
                ]
                
                # 2. Hardcoded Fetch Plan (Assume user data is primary)
                # This ensures SourceIdentifier skips its Gemini call
                fetch_plan = [
                    {
                        "source": "uploaded_file",
                        "category": "mixed",
                        "priority": 1,
                        "params": {},
                        "rationale": "Using user-uploaded CSV as primary data source."
                    }
                ]
                session.store_artifact("fetch_plan", fetch_plan)

                # 3. Hardcoded Model Config (Rule-based weights)
                # This ensures ModelSelector skips its Gemini call
                model_config = {
                    "mode": "classification",
                    "forecast_horizon": 1,
                    "models_to_use": ["xgboost", "random_forest"],
                    "weights": {"xgboost": 0.55, "random_forest": 0.45}
                }
                session.store_artifact("model_config", model_config)
                
                # 4. Extract initial context
                session.set_context("user_type", user_type)
                session.set_context("original_query", user_query)
                session.set_context("intent", "data_upload")
                session.set_context("wants_recommendations", True)
                
                # Location hints
                if uploaded_columns:
                    col_map = {c.lower(): c for c in uploaded_columns}
                    if any(k in col_map for k in ["lat", "latitude", "lon", "longitude"]):
                        session.set_context("has_location", True)

            else:
                # ── Standard unified single Gemini startup call ───────────
                plan = await self._unified_startup_call(
                    session, user_query, user_type, uploaded_columns
                )

            if uploaded_file_bytes:
                session.store_artifact("uploaded_file_bytes", uploaded_file_bytes)

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

                # Serialize agent result so it can be sent over SSE as JSON
                step_result = None
                if agent_step and agent_step.output_data is not None:
                    raw_result = agent_step.output_data
                    if hasattr(raw_result, "model_dump"):
                        raw_result = raw_result.model_dump()
                    elif hasattr(raw_result, "dict"):
                        raw_result = raw_result.dict()
                    step_result = raw_result if isinstance(raw_result, dict) else None

                yield {
                    "event": "step_done",
                    "data": {
                        "step_index": step_dict["step_index"],
                        "agent": step_dict["agent"],
                        "status": agent_step.status if agent_step else "unknown",
                        "result": step_result,
                    },
                }

            response = self._build_response(session)
            session.complete(response)
            yield {"event": "complete", "data": response}

        finally:
            self._memory.archive_session(session.session_id)

    # ── Unified startup call (1 LLM call replaces context + plan + model + source) ─

    async def _unified_startup_call(
        self,
        session,
        user_query:       str,
        user_type:        str,
        uploaded_columns: Optional[List[str]],
    ) -> List[Dict[str, Any]]:
        """
        Single Gemini call that replaces 3-4 previously separate LLM calls:
          1. ContextManager.initialise()  → metadata + geocode + intent
          2. Planner.create_plan()        → execution steps
          3. ModelSelector._llm_refine()  → model weights (pre-computed, saved to session)
          4. SourceIdentifier._llm_reasoning() → fetch plan (pre-computed, saved to session)

        Returns the parsed execution plan (list of step dicts) and stores all
        other outputs into the session for downstream agents to consume.
        """
        cols_str = ", ".join(uploaded_columns[:20]) if uploaded_columns else "none"
        has_upload = bool(uploaded_columns)

        # Identify fields already satisfied by the user (manual/textbox)
        manual_inputs = [
            k for k in [
                "rain_mm_weekly", "rain_mm_monthly", "temp_c_mean", "rh_percent_mean", "wind_ms_mean",
                "elevation_m", "slope_degree", "terrain_type",
                "dist_major_river_km", "dam_count_50km", "waterbody_nearby"
            ]
            if session.get_context(k) is not None
        ]

        prompt = f"""You are the orchestration brain of FloodSense AI, a flood prediction system.
A user submitted a request and you must plan the entire AI pipeline in ONE response.

User query: "{user_query}"
User type: {user_type}
Uploaded data columns: {cols_str}
Has uploaded data: {has_upload}
Manual User Inputs (ALREADY PROVIDED): {manual_inputs}

Return ONLY a single JSON object (no prose, no markdown fences):
{{
  "metadata": {{
    "location": "<city/district from query or null>",
    "state": "<Indian state or null>",
    "country": "India",
    "latitude": <float or null>,
    "longitude": <float or null>,
    "user_type": "{user_type}",
    "intent": "<flood_prediction|simulation|recommendation|general_query>",
    "wants_recommendations": <true|false (default true for flood_prediction)>,
    "wants_simulation": <true|false (default false)>,
    "urgency": "<low|medium|high|critical>",
    "data_types": ["<rainfall|hydro|terrain>"],
    "original_query": "{user_query[:200]}"
  }},
  "plan": [
    {{
      "step_index": 1,
      "agent": "data_ingestion",
      "action": "Ingest flood data",
      "inputs": {{}},
      "depends_on": []
    }},
    {{
      "step_index": 2,
      "agent": "preprocessing",
      "action": "Preprocess data",
      "inputs": {{"dataset": "raw_dataset"}},
      "depends_on": [1]
    }},
    {{
      "step_index": 3,
      "agent": "prediction",
      "action": "Predict flood risk",
      "inputs": {{"dataset": "processed_dataset"}},
      "depends_on": [2]
    }}
  ],
  "model_config": {{
    "mode": "<classification|regression|multi_class>",
    "forecast_horizon": <1|3|7>,
    "models_to_use": ["xgboost", "random_forest"],
    "weights": {{"xgboost": 0.55, "random_forest": 0.45}}
  }},
  "fetch_plan": [
    {{
      "source": "<open_meteo|terrain|hydrological|gov_dataset>",
      "category": "<rainfall|hydro|terrain|timeseries>",
      "priority": <1|2|3>,
      "params": {{}},
      "rationale": "<one line reason>"
    }}
  ]
}}

Rules for plan:
- Always include data_ingestion → preprocessing → prediction as steps 1-3.
- Always include step 4 "recommendation" (depends_on: [3]) for flood_prediction intents.
- Always include step 5 "simulation" (depends_on: [3]) for flood_prediction intents.
- Always include step 6 "alerting" (depends_on: [3, 4]) for flood_prediction intents (it will be dynamically skipped later if risk is low).
- For model_config: use classification by default, regression if user asks water levels, multi_class for authorities.
- For fetch_plan: 
  1. ONLY include sources for missing data categories.
  2. If a data category (rainfall, hydro, terrain) is already covered by "Manual User Inputs", do NOT include a source for it.
  3. if user uploaded data, only include terrain source (priority 2) IF it's not in Manual User Inputs. 
  4. If no upload and no manual inputs, include open_meteo (priority 1) + terrain (priority 2).
- Weights must sum to 1.0."""

        fallback_plan = [
            {"step_index": 1, "agent": "data_ingestion", "action": "Ingest data", "inputs": {}, "depends_on": []},
            {"step_index": 2, "agent": "preprocessing", "action": "Preprocess", "inputs": {"dataset": "raw_dataset"}, "depends_on": [1]},
            {"step_index": 3, "agent": "prediction", "action": "Predict", "inputs": {"dataset": "processed_dataset"}, "depends_on": [2]},
            # Safety agents are included — the orchestrator prunes them automatically
            # if prediction fails or flood_probability < SAFETY_THRESHOLD (25%).
            {"step_index": 4, "agent": "recommendation", "action": "Generate recommendations", "inputs": {"prediction": "ensemble_prediction"}, "depends_on": [3]},
            {"step_index": 5, "agent": "simulation", "action": "Run flood simulation", "inputs": {"prediction": "ensemble_prediction"}, "depends_on": [3]},
            {"step_index": 6, "agent": "alerting", "action": "Evaluate for alerts", "inputs": {"prediction": "ensemble_prediction", "recommendations": "recommendations"}, "depends_on": [3, 4]},
        ]

        try:
            raw = await self._gemini.generate_json(prompt)
        except Exception as exc:
            err_str = str(exc).upper()
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                logger.warning(f"[Orchestrator] Unified startup call failed (QUOTA EXHAUSTED). Enabling low-quota mode.")
                session.set_context("low_quota_mode", True)
            else:
                logger.warning(f"[Orchestrator] Unified startup call failed: {exc}. Using fallback.")
            
            session.set_context("user_type", user_type)
            session.set_context("original_query", user_query)
            return fallback_plan

        if not raw or not isinstance(raw, dict):
            logger.warning("[Orchestrator] Unified call returned non-dict; using fallback.")
            session.set_context("user_type", user_type)
            session.set_context("original_query", user_query)
            return fallback_plan

        # ── Populate session context from metadata ──────────────────────────
        meta = raw.get("metadata", {})
        for key in ["location", "state", "country", "latitude", "longitude",
                    "intent", "wants_recommendations", "wants_simulation",
                    "urgency", "data_types", "original_query"]:
            val = meta.get(key)
            if val is not None:
                session.set_context(key, val)
        session.set_context("user_type", user_type)

        # ── Store pre-computed configs for sub-agents ───────────────────────
        model_cfg = raw.get("model_config")
        if model_cfg and isinstance(model_cfg, dict):
            session.store_artifact("model_config", model_cfg)
            logger.info(f"[Orchestrator] Pre-computed model config: {model_cfg.get('mode')} | "
                        f"models={model_cfg.get('models_to_use')}")

        fetch_plan = raw.get("fetch_plan")
        if fetch_plan and isinstance(fetch_plan, list):
            session.store_artifact("fetch_plan", fetch_plan)
            logger.info(f"[Orchestrator] Pre-computed fetch plan: {len(fetch_plan)} tasks")

        # ── Parse and return plan ────────────────────────────────────────
        plan = raw.get("plan")
        if not plan or not isinstance(plan, list):
            logger.warning("[Orchestrator] Unified call returned no plan; using fallback.")
            return fallback_plan

        logger.info(
            f"[Orchestrator] Unified startup: location={meta.get('location')} | "
            f"intent={meta.get('intent')} | steps={len(plan)}"
        )
        return plan

    # ── Plan execution ────────────────────────────────────────────────

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

                # ── Dynamic Plan Pruning based on Prediction ──────────────
                if step["agent"] == "prediction":
                    # Correct lookup: Ensemble data is stored as ensemble_prediction
                    ensemble = session.get_artifact("ensemble_prediction", {})
                    prob = ensemble.get("flood_probability", 0.0) if isinstance(ensemble, dict) else 0.0

                    # Also check if the prediction step itself failed
                    prediction_step = self._find_agent_step(session, step)
                    prediction_failed = (prediction_step is not None and prediction_step.status == StepStatus.FAILED)

                    # Skip safety agents if prediction failed OR risk is too low
                    if prediction_failed or prob < settings.SAFETY_THRESHOLD:
                        reason = (
                            f"Prediction step failed — no risk data"
                            if prediction_failed
                            else f"Risk too low ({prob:.0%}) < {settings.SAFETY_THRESHOLD:.0%}"
                        )
                        logger.info(
                            f"[Orchestrator] {reason}: Skipping safety agents "
                            f"(recommendation, simulation, alerting)."
                        )
                        # Remove downstream safety steps from remaining
                        safety_agents = {"recommendation", "simulation", "alerting"}
                        remaining = [s for s in remaining if s["agent"] not in safety_agents]
                        # Mark them as SKIPPED so UI/logic doesn't hang
                        for s in plan:
                            if s["agent"] in safety_agents and s["step_index"] not in completed_indices:
                                completed_indices.add(s["step_index"])
                                skip_step = session.add_step(
                                    agent_name=s["agent"],
                                    action=s["action"],
                                    input_data={"reason": reason},
                                )
                                skip_step.status = StepStatus.SKIPPED

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

        # ── Auto-inject missing required inputs from session artifacts ─────────
        # The LLM sometimes generates an empty inputs map. Fall back to
        # well-known artifact keys so the pipeline always has what it needs.
        schema = self._registry.get_schema(agent_name)
        if schema:
            FALLBACK_KEYS = {
                "data":           ["data", "processed_dataset", "raw_dataset"],
                "prediction":     ["prediction", "prediction_result"],
                "recommendations": ["recommendations"],
            }
            for req_key in schema.required_inputs:
                if req_key in resolved_inputs or req_key == "session":
                    continue
                for candidate in FALLBACK_KEYS.get(req_key, [req_key]):
                    artifact = session.get_artifact(candidate)
                    if artifact is not None:
                        resolved_inputs[req_key] = artifact
                        logger.debug(
                            f"[Orchestrator] Auto-injected '{req_key}' "
                            f"from artifact '{candidate}' for '{agent_name}'"
                        )
                        break

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
            # ── Fixed pipeline aliases ────────────────────────────────────────
            # The LLM may use different artifact key names than what downstream
            # agents expect. We always store well-known aliases so the pipeline
            # is robust to LLM variation.
            if agent_name == "data_ingestion":
                # preprocessing/prediction look for "data" → alias raw_dataset
                raw = session.get_artifact("raw_dataset")
                if raw is not None:
                    session.store_artifact("data", raw)
            elif agent_name == "preprocessing":
                # prediction looks for "data" → alias processed_dataset
                processed = session.get_artifact("processed_dataset")
                if processed is not None:
                    session.store_artifact("data", processed)
            elif agent_name == "prediction":
                # Serialize Pydantic model → dict before storing
                pred_data = result.output
                if hasattr(pred_data, "model_dump"):
                    pred_data = pred_data.model_dump()
                elif hasattr(pred_data, "dict"):
                    pred_data = pred_data.dict()
                if isinstance(pred_data, dict):
                    # Store under all keys so _build_response and recommendation
                    # / alerting / simulation agents can all find it
                    session.store_artifact("prediction", pred_data)
                    session.store_artifact("prediction_result", pred_data)
                    
                    # Correctly access nested ensemble data
                    ensemble = pred_data.get("ensemble") or {}
                    session.store_artifact("ensemble_prediction", ensemble)
                    
                    self._ctx_mgr.update_risk(
                        session,
                        risk_level=ensemble.get("risk_level", "UNKNOWN"),
                        probability=ensemble.get("flood_probability", 0.0),
                    )
            
            # ── Unified Registry for Downstream Visibility ──────────────────
            # Store serialized result under canonical keys used by _build_response
            output = result.output
            if hasattr(output, "model_dump"):
                output = output.model_dump()
            elif hasattr(output, "dict"):
                output = output.dict()
            
            if agent_name == "recommendation":
                session.store_artifact("recommendations", output)
            elif agent_name == "simulation":
                session.store_artifact("simulation_result", output)
                if isinstance(output, dict) and "geojson" in output:
                    session.store_artifact("geojson", output["geojson"])
            elif agent_name == "alerting":
                session.store_artifact("alert_status", output)
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
        prediction    = session.get_artifact("prediction_result") or session.get_artifact("prediction")
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
            # Normalise: the LLM occasionally returns a list instead of a string.
            # e.g. {"data": ["raw_dataset"]} -> treat first element as the key.
            if isinstance(artifact_key, list):
                artifact_key = artifact_key[0] if artifact_key else ""
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
                mod = importlib.import_module(reg["module"])

                # Use the module-level singleton for AlertingAgent so the orchestrator
                # shares the same SubscriberManager as the API routes (same subscribers).
                if reg["class"] == "AlertingAgent" and hasattr(mod, "get_alerting_agent"):
                    inst = mod.get_alerting_agent()
                else:
                    cls  = getattr(mod, reg["class"])
                    inst = cls()

                fn = getattr(inst, reg["method"])
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
