"""
Pluggable LLM provider interface.

The rest of the app (SQL generation, RAG synthesis, routing, eval)
talks to `get_llm()` and never imports anthropic/openai/groq directly.
Swapping providers is a one-line config change (LLM_PROVIDER=openai),
not a code change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.core.config import LLMProvider, Settings, settings


class LLMClient(ABC):
    """Minimal common interface every provider adapter implements."""

    @abstractmethod
    def complete(self, system: str, user: str) -> str:
        """Single-turn completion: system prompt + user message -> text."""
        raise NotImplementedError


class AnthropicClient(LLMClient):
    def __init__(self, cfg: Settings):
        from anthropic import Anthropic  # local import: optional dependency

        self._client = Anthropic(api_key=cfg.active_api_key())
        self._model = cfg.llm_model
        self._max_tokens = cfg.llm_max_tokens
        self._temperature = cfg.llm_temperature

    def complete(self, system: str, user: str) -> str:
        # Newer Anthropic SDKs no longer accept `temperature` in
        # messages.create for this model family.
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")


class OpenAIClient(LLMClient):
    def __init__(self, cfg: Settings):
        from openai import OpenAI  # local import: optional dependency

        self._client = OpenAI(api_key=cfg.active_api_key())
        self._model = cfg.llm_model
        self._max_tokens = cfg.llm_max_tokens
        self._temperature = cfg.llm_temperature

    def complete(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""


class GroqClient(LLMClient):
    def __init__(self, cfg: Settings):
        from groq import Groq  # local import: optional dependency

        self._client = Groq(api_key=cfg.active_api_key())
        self._model = cfg.llm_model
        self._max_tokens = cfg.llm_max_tokens
        self._temperature = cfg.llm_temperature

    def complete(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""


_PROVIDER_MAP: dict[LLMProvider, type[LLMClient]] = {
    LLMProvider.ANTHROPIC: AnthropicClient,
    LLMProvider.OPENAI: OpenAIClient,
    LLMProvider.GROQ: GroqClient,
}


def get_llm(cfg: Settings | None = None) -> LLMClient:
    cfg = cfg or settings
    try:
        cls = _PROVIDER_MAP[cfg.llm_provider]
    except KeyError as e:
        raise ValueError(f"Unknown LLM provider: {cfg.llm_provider}") from e
    return cls(cfg)
