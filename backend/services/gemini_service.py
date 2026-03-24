"""
Gemini Service – Centralised, reusable wrapper for Google Gemini API calls.

Features:
  - Async and sync generation methods
  - Configurable model selection (gemini-1.5-pro, gemini-1.5-flash, etc.)
  - Automatic retry with exponential backoff on quota/server errors
  - Chat session management (multi-turn conversations)
  - Token usage tracking
  - Structured JSON generation helper
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict, List, Optional

import google.generativeai as genai

from config import settings
from utils.logger import logger


# ── Default models ────────────────────────────────────────────────────────────

DEFAULT_MODEL       = "gemini-1.5-pro"
FAST_MODEL          = "gemini-1.5-flash"    # Cheaper / faster for simple tasks

# Retry settings
MAX_RETRIES         = 3
RETRY_BASE_SECS     = 2.0

# Safety settings (less restrictive for professional/emergency use-cases)
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
]


class GeminiService:
    """
    Thin async wrapper around the google-generativeai SDK.

    All agents should import and use this service rather than calling
    genai directly, so we get unified retry logic, logging, and token tracking.
    """

    def __init__(
        self,
        api_key:     Optional[str] = None,
        model_name:  str = DEFAULT_MODEL,
    ) -> None:
        _key = api_key or settings.GEMINI_API_KEY
        if not _key:
            raise ValueError(
                "GEMINI_API_KEY is not set. "
                "Add it to your .env file or pass it explicitly."
            )
        genai.configure(api_key=_key)
        self._model_name = model_name
        self._model      = genai.GenerativeModel(
            model_name=model_name,
            safety_settings=SAFETY_SETTINGS,
        )
        self._fast_model = genai.GenerativeModel(
            model_name=FAST_MODEL,
            safety_settings=SAFETY_SETTINGS,
        )
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
            use_fast_model:     Use gemini-1.5-flash instead of the main model.
            temperature:        Sampling temperature (lower = more deterministic).
            max_output_tokens:  Cap on response length.

        Returns:
            The generated text string.
        """
        model = self._fast_model if use_fast_model else self._model
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

        for attempt in range(MAX_RETRIES):
            try:
                response = await asyncio.to_thread(
                    model.generate_content,
                    prompt,
                    generation_config=generation_config,
                )
                text = response.text
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

        If parsing fails, returns an empty dict rather than raising,
        so agents can handle the failure gracefully.
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
        chat = self._model.start_chat(history=history or [])
        return GeminiChatSession(chat, self)

    # ── Embeddings ────────────────────────────────────────────────────────

    async def embed(self, text: str) -> List[float]:
        """
        Generates a text embedding vector.
        Useful for semantic similarity comparisons in the recommendation agent.
        """
        result = await asyncio.to_thread(
            genai.embed_content,
            model="models/text-embedding-004",
            content=text,
            task_type="semantic_similarity",
        )
        return result["embedding"]

    # ── Token tracking ────────────────────────────────────────────────────

    def _track_usage(self, response: Any) -> None:
        try:
            usage = response.usage_metadata
            self._token_usage["prompt"]     += usage.prompt_token_count or 0
            self._token_usage["completion"] += usage.candidates_token_count or 0
        except Exception:
            pass

    @property
    def token_usage(self) -> Dict[str, int]:
        """Returns cumulative token counts since service initialisation."""
        return dict(self._token_usage)


# ── Chat session wrapper ──────────────────────────────────────────────────────

class GeminiChatSession:
    """Wraps a genai ChatSession for multi-turn conversations."""

    def __init__(self, chat: Any, service: GeminiService) -> None:
        self._chat    = chat
        self._service = service

    async def send(self, message: str, temperature: float = 0.3) -> str:
        """Sends a message and returns the assistant reply."""
        for attempt in range(MAX_RETRIES):
            try:
                gen_config = genai.types.GenerationConfig(temperature=temperature)
                response = await asyncio.to_thread(
                    self._chat.send_message,
                    message,
                    generation_config=gen_config,
                )
                return response.text
            except Exception as exc:
                wait = RETRY_BASE_SECS * (2 ** attempt)
                logger.warning(f"[GeminiChatSession] Error (attempt {attempt+1}): {exc}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(wait)
                else:
                    raise

    @property
    def history(self) -> List[Dict]:
        return self._chat.history


# ── Module-level singleton ────────────────────────────────────────────────────

_gemini_service: Optional[GeminiService] = None


def get_gemini_service() -> GeminiService:
    """Returns the module-level singleton, creating it on first call."""
    global _gemini_service
    if _gemini_service is None:
        _gemini_service = GeminiService()
    return _gemini_service
