"""
Gemini Service – Centralised, reusable wrapper for the Google Gemini API.

Uses the new `google-genai` SDK (google.genai) which replaced the deprecated
`google.generativeai` package. All agents should import `get_gemini_service()`
rather than calling the SDK directly.

Features:
  - Async text generation with automatic retry / exponential backoff
  - JSON-mode helper (generate_json)
  - Multi-turn chat session wrapper
  - Token usage tracking
  - Configurable model selection
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types as genai_types

from config import settings
from utils.logger import logger


# ── Default models ────────────────────────────────────────────────────────────

DEFAULT_MODEL = "gemini-2.0-flash"   # Available and stable on this API key
FAST_MODEL    = "gemini-2.0-flash-lite" # Higher-quota / faster for structural tasks

# Retry settings
MAX_RETRIES      = 1   # Fail fast to hit fallbacks and save quota
RETRY_BASE_SECS  = 4.0


class GeminiService:
    """
    Async wrapper around the google-genai SDK.

    All agents should use this service so we get unified retry logic,
    logging, and token tracking.
    """

    def __init__(
        self,
        api_key:    Optional[str] = None,
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        _key = api_key or settings.GEMINI_API_KEY
        if not _key:
            raise ValueError(
                "GEMINI_API_KEY is not set. "
                "Add it to your .env file or pass it explicitly."
            )
        # New SDK: create a Client instance (v1beta default works for gemini-2.x models)
        self._client     = genai.Client(api_key=_key)
        self._model_name = model_name
        self._fast_model = FAST_MODEL
        self._token_usage: Dict[str, int] = {"prompt": 0, "completion": 0}
        logger.info(f"[GeminiService] Initialised with model={model_name}")

    # ── Core generation ───────────────────────────────────────────────────

    async def generate(
        self,
        prompt: str,
        use_fast_model: bool = False,
        temperature: float = 0.2,
        max_output_tokens: int = 8192,
    ) -> str:
        """
        Single-turn text generation with retry logic.

        Args:
            prompt:             The full prompt string.
            use_fast_model:     Use FAST_MODEL instead of the main model.
            temperature:        Sampling temperature (lower = more deterministic).
            max_output_tokens:  Cap on response length.

        Returns:
            The generated text string.
        """
        model = self._fast_model if use_fast_model else self._model_name
        config = genai_types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

        for attempt in range(MAX_RETRIES):
            try:
                response = await asyncio.to_thread(
                    self._client.models.generate_content,
                    model=model,
                    contents=prompt,
                    config=config,
                )
                text = response.text or ""
                self._track_usage(response)
                logger.debug(
                    f"[GeminiService] Generated {len(text)} chars "
                    f"(attempt {attempt + 1})"
                )
                return text

            except Exception as exc:
                wait = RETRY_BASE_SECS * (2 ** attempt)
                logger.warning(
                    f"[GeminiService] Generation failed (attempt {attempt + 1}): "
                    f"{exc}. Retrying in {wait:.0f}s..."
                )
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(wait)
                else:
                    logger.error("[GeminiService] All retries exhausted.")
                    raise

    async def generate_json(
        self,
        prompt: str,
        use_fast_model: bool = True,
    ) -> Any:
        """
        Generates a response and parses it as JSON.

        If parsing fails, returns an empty dict so agents can handle gracefully.
        """
        raw = await self.generate(prompt, use_fast_model=use_fast_model, temperature=0.1)
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()
        try:
            return json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            logger.warning(f"[GeminiService] JSON parse failed. Raw:\n{cleaned[:300]}")
            return {}

    # ── Chat sessions ─────────────────────────────────────────────────────

    def start_chat(
        self,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> "GeminiChatSession":
        """Creates a multi-turn chat session."""
        return GeminiChatSession(
            client=self._client,
            model=self._model_name,
            history=history or [],
            service=self,
        )

    # ── Embeddings ────────────────────────────────────────────────────────

    async def embed(self, text: str) -> List[float]:
        """
        Generates a text embedding vector using the embeddings model.
        Useful for semantic similarity in the recommendation agent.
        """
        result = await asyncio.to_thread(
            self._client.models.embed_content,
            model="text-embedding-004",
            contents=text,
        )
        return result.embeddings[0].values

    # ── Token tracking ────────────────────────────────────────────────────

    def _track_usage(self, response: Any) -> None:
        try:
            meta = response.usage_metadata
            self._token_usage["prompt"]     += meta.prompt_token_count or 0
            self._token_usage["completion"] += meta.candidates_token_count or 0
        except Exception:
            pass

    @property
    def token_usage(self) -> Dict[str, int]:
        """Returns cumulative token counts since service initialisation."""
        return dict(self._token_usage)


# ── Chat session wrapper ──────────────────────────────────────────────────────

class GeminiChatSession:
    """Wraps multi-turn chat using the new google-genai SDK."""

    def __init__(
        self,
        client: Any,
        model: str,
        history: List[Dict[str, str]],
        service: GeminiService,
    ) -> None:
        self._client  = client
        self._model   = model
        self._history: List[Dict[str, str]] = list(history)
        self._service = service

    async def send(self, message: str, temperature: float = 0.3) -> str:
        """Sends a message and returns the assistant reply."""
        self._history.append({"role": "user", "parts": [{"text": message}]})

        config = genai_types.GenerateContentConfig(temperature=temperature)
        for attempt in range(MAX_RETRIES):
            try:
                response = await asyncio.to_thread(
                    self._client.models.generate_content,
                    model=self._model,
                    contents=self._history,
                    config=config,
                )
                reply = response.text or ""
                self._history.append({"role": "model", "parts": [{"text": reply}]})
                return reply
            except Exception as exc:
                wait = RETRY_BASE_SECS * (2 ** attempt)
                logger.warning(f"[GeminiChatSession] Error (attempt {attempt+1}): {exc}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(wait)
                else:
                    raise
        return ""

    @property
    def history(self) -> List[Dict]:
        return list(self._history)


# ── Module-level singleton ────────────────────────────────────────────────────

_gemini_service: Optional[GeminiService] = None


def get_gemini_service() -> GeminiService:
    """Returns the module-level singleton, creating it on first call."""
    global _gemini_service
    if _gemini_service is None:
        _gemini_service = GeminiService()
    return _gemini_service
