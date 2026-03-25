"""
Planner – LLM-powered task decomposition for the Orchestration Agent.

Given a user query and the current session context, the Planner asks Gemini
to generate a structured, ordered execution plan consisting of named agent
steps. Each step specifies:
  - agent : which specialist agent to invoke
  - action: what the agent should do
  - inputs : what data to pass (references to session artifacts or literals)
  - depends_on : step IDs this step must wait for

The Planner also handles plan revision — if an agent step fails or produces
unexpected output the Orchestrator can call re_plan() to get a revised plan.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from agents.orchestration.memory import Session, MessageRole
from services.gemini_service import GeminiService
from utils.logger import logger


# ── Plan schema ───────────────────────────────────────────────────────────────

PLAN_SCHEMA_DESCRIPTION = """
Return a JSON array of plan steps. Each step is an object with these fields:
{
  "step_index": <integer, 1-based>,
  "agent": "<one of: data_ingestion | preprocessing | prediction | recommendation | simulation | alerting>",
  "action": "<short description of what the agent must do>",
  "inputs": {"<key>": "<artifact_name or literal value>"},
  "outputs": ["<artifact names this step produces>"],
  "depends_on": [<step_index integers this step depends on; [] if first step>],
  "optional": <true|false  — if true, failure can be tolerated>
}

Respond ONLY with the JSON array. No prose, no markdown fences.
"""

AVAILABLE_AGENTS = {
    "data_ingestion":  "Collects, validates, and merges data from APIs and user uploads.",
    "preprocessing":   "Cleans, transforms, and engineers features from raw data.",
    "prediction":      "Runs one or more ML models to produce a flood risk prediction.",
    "recommendation":  "Generates LLM recommendations tailored to user type and location.",
    "simulation":      "Computes flood zone polygons and builds GeoJSON / map tiles.",
    "alerting":        "Composes and dispatches alerts to subscribed users.",
}


# ── Planner class ─────────────────────────────────────────────────────────────

class Planner:
    """
    Uses Gemini to generate and revise a step-by-step execution plan.

    The plan is re-generated each time the context changes significantly
    (e.g. after an agent failure or a mid-run user clarification).
    """

    def __init__(self, gemini_service: GeminiService) -> None:
        self._gemini = gemini_service

    # ── Public API ────────────────────────────────────────────────────────

    async def create_plan(
        self,
        session: Session,
        available_data: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generates an execution plan for the user query in `session`.

        Args:
            session:        The active orchestration session.
            available_data: Dict describing what data is already available
                            (e.g. uploaded CSV columns, location name).

        Returns:
            A list of step dicts conforming to PLAN_SCHEMA_DESCRIPTION.
        """
        prompt = self._build_plan_prompt(session, available_data)
        logger.info(f"[Planner] Generating plan for session {session.session_id}")

        raw_response = await self._gemini.generate(prompt)
        plan = self._parse_plan(raw_response, session)

        session.add_message(
            MessageRole.ASSISTANT,
            f"Execution plan generated with {len(plan)} steps.",
            plan=plan,
        )
        session.store_artifact("execution_plan", plan)
        logger.info(f"[Planner] Plan has {len(plan)} steps: {[s['agent'] for s in plan]}")
        return plan

    async def re_plan(
        self,
        session: Session,
        failed_step: Dict[str, Any],
        error_message: str,
    ) -> List[Dict[str, Any]]:
        """
        Asks the LLM to revise the plan after a step failure.

        Args:
            session:       The active session containing the original plan.
            failed_step:   The step dict that failed.
            error_message: The error string from the failing agent.

        Returns:
            A revised list of plan steps (only remaining steps).
        """
        prompt = self._build_replan_prompt(session, failed_step, error_message)
        logger.warning(
            f"[Planner] Re-planning after failure in step {failed_step.get('step_index')} "
            f"({failed_step.get('agent')}): {error_message[:120]}"
        )

        raw_response = await self._gemini.generate(prompt)
        revised_plan = self._parse_plan(raw_response, session)

        session.add_message(
            MessageRole.ASSISTANT,
            f"Plan revised after failure. New plan has {len(revised_plan)} remaining steps.",
            original_failure=failed_step,
        )
        session.store_artifact("execution_plan", revised_plan)
        return revised_plan

    async def classify_intent(self, user_query: str, session=None) -> Dict[str, Any]:
        """Returns intent metadata.

        If the session already has intent (populated by ContextManager's merged
        metadata+intent call), return it directly — no extra LLM call needed.
        """
        if session is not None:
            intent = session.get_context("intent")
            if intent:
                return {
                    "intent":    intent,
                    "location":  session.get_context("location"),
                    "data_type": session.get_context("data_type"),
                    "urgency":   session.get_context("urgency", "routine"),
                }
        # Fallback default — avoids an LLM call
        return {
            "intent":    "prediction_query",
            "location":  None,
            "data_type": None,
            "urgency":   "routine",
        }

    # ── Prompt builders ───────────────────────────────────────────────────

    def _build_plan_prompt(
        self,
        session: Session,
        available_data: Optional[Dict[str, Any]],
    ) -> str:
        agent_descriptions = "\n".join(
            f"  - {name}: {desc}" for name, desc in AVAILABLE_AGENTS.items()
        )
        data_section = (
            f"Available data artifacts: {json.dumps(available_data, default=str)}"
            if available_data
            else "No data uploaded yet — the data_ingestion agent must fetch data."
        )
        conversation_history = session.get_conversation_text()

        return f"""
You are the intelligent orchestration planner for a flood prediction and early-warning system.

## User Request
{session.user_query}

## Conversation History
{conversation_history}

## Available Specialist Agents
{agent_descriptions}

## Data Context
{data_section}

## Session Context
{json.dumps(session.context, default=str)}

## Instructions
Analyse the user request and produce an execution plan.
- Only include agents that are necessary.
- Steps that can run in parallel share the same `step_index` value.
- Always start with data_ingestion if fresh data needs to be fetched.
- prediction must always precede recommendation, simulation, and alerting.
- alerting is optional unless the user explicitly wants alerts.

{PLAN_SCHEMA_DESCRIPTION}
"""

    def _build_replan_prompt(
        self,
        session: Session,
        failed_step: Dict[str, Any],
        error_message: str,
    ) -> str:
        original_plan = session.get_artifact("execution_plan", [])
        completed_steps = [
            s for s in original_plan
            if s.get("step_index", 0) < failed_step.get("step_index", 0)
        ]

        return f"""
You are the orchestration planner for a flood prediction system.

## Original User Request
{session.user_query}

## Steps Already Completed Successfully
{json.dumps(completed_steps, default=str)}

## Failed Step
{json.dumps(failed_step, default=str)}

## Error
{error_message}

## Instructions
The above step has failed. Revise the remaining execution plan to:
1. Either retry the failed step with a different configuration, OR skip it if not critical.
2. List only the REMAINING steps (do not re-list completed ones).
3. Re-number step_index starting from {failed_step.get('step_index', 1)}.

{PLAN_SCHEMA_DESCRIPTION}
"""

    # ── Response parser ───────────────────────────────────────────────────

    def _parse_plan(self, raw: str, session: Session) -> List[Dict[str, Any]]:
        """Extracts and validates the JSON plan from the LLM response."""
        # Strip any accidental markdown code fences
        cleaned = re.sub(r"```(?:json)?", "", raw).strip()
        try:
            plan = json.loads(cleaned)
            if not isinstance(plan, list):
                raise ValueError("Plan must be a JSON array.")
            # Basic field validation
            for step in plan:
                step.setdefault("depends_on", [])
                step.setdefault("optional", False)
                step.setdefault("inputs", {})
                step.setdefault("outputs", [])
            return plan
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error(f"[Planner] Failed to parse plan JSON: {exc}\nRaw:\n{raw[:500]}")
            session.add_message(
                MessageRole.SYSTEM,
                f"Warning: Planner returned malformed JSON. Using fallback plan.",
            )
            return self._fallback_plan()

    @staticmethod
    def _fallback_plan() -> List[Dict[str, Any]]:
        """Minimal safe plan used when LLM response cannot be parsed."""
        return [
            {
                "step_index": 1,
                "agent": "data_ingestion",
                "action": "Collect all available data for the requested location",
                "inputs": {},
                "outputs": ["raw_dataset"],
                "depends_on": [],
                "optional": False,
            },
            {
                "step_index": 2,
                "agent": "preprocessing",
                "action": "Clean and prepare the ingested dataset",
                "inputs": {"data": "raw_dataset"},
                "outputs": ["processed_dataset"],
                "depends_on": [1],
                "optional": False,
            },
            {
                "step_index": 3,
                "agent": "prediction",
                "action": "Run flood risk prediction on preprocessed data",
                "inputs": {"data": "processed_dataset"},
                "outputs": ["prediction_result"],
                "depends_on": [2],
                "optional": False,
            },
            {
                "step_index": 4,
                "agent": "recommendation",
                "action": "Generate safety recommendations based on prediction",
                "inputs": {"prediction": "prediction_result"},
                "outputs": ["recommendations"],
                "depends_on": [3],
                "optional": True,
            },
        ]
