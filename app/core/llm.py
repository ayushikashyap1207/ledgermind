"""
Pluggable LLM provider interface.

The rest of the app (SQL generation, RAG synthesis, routing, eval)
talks to `get_llm()` and never imports anthropic/openai/groq directly.
Swapping providers is a one-line config change (LLM_PROVIDER=openai),
not a code change.
"""

from __future__ import annotations

import time
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


class OllamaClient(LLMClient):
    """Local OpenAI-compatible endpoint, for offline/self-hosted deployments."""

    def __init__(self, cfg: Settings):
        from openai import OpenAI  # local import: optional dependency

        self._client = OpenAI(api_key="ollama", base_url=cfg.ollama_base_url)
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


class GuardrailViolation(RuntimeError):
    """Raised when Bedrock Guardrails blocks a model response."""


class BedrockClient(LLMClient):
    """Amazon Bedrock Converse API, with optional Guardrails."""

    def __init__(self, cfg: Settings, client=None):
        self._model = cfg.llm_model
        self._max_tokens = cfg.llm_max_tokens
        self._temperature = cfg.llm_temperature
        self._guardrail_enabled = bool(
            cfg.bedrock_guardrail_enabled and cfg.bedrock_guardrail_id
        )
        self._guardrail_id = cfg.bedrock_guardrail_id
        self._guardrail_version = cfg.bedrock_guardrail_version

        if client is not None:
            self._client = client
        else:
            try:
                import boto3
            except ImportError as exc:  # pragma: no cover - dependency optional
                raise RuntimeError(
                    "Bedrock is selected but boto3 is not installed. "
                    "Run: pip install -r requirements-aws.txt"
                ) from exc
            session = boto3.Session(
                profile_name=cfg.aws_profile or None,
                region_name=cfg.aws_region,
            )
            self._client = session.client(
                "bedrock-runtime",
                endpoint_url=cfg.aws_endpoint_url or None,
            )

    def complete(self, system: str, user: str) -> str:
        kwargs = {
            "modelId": self._model,
            "messages": [{"role": "user", "content": [{"text": user}]}],
            "system": [{"text": system}],
            "inferenceConfig": {
                "maxTokens": self._max_tokens,
                "temperature": self._temperature,
            },
        }
        if self._guardrail_enabled:
            kwargs["guardrailConfig"] = {
                "guardrailIdentifier": self._guardrail_id,
                "guardrailVersion": self._guardrail_version,
            }

        response = self._client.converse(**kwargs)

        if response.get("stopReason") == "guardrail_intervened":
            raise GuardrailViolation(
                "The response was blocked by the configured Bedrock Guardrail."
            )

        content = response.get("output", {}).get("message", {}).get("content", [])
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict)
        )


class RetryingLLMClient(LLMClient):
    """Retry transient provider failures with exponential backoff."""

    def __init__(self, inner: LLMClient, max_retries: int, backoff_seconds: float):
        self._inner = inner
        self._max_retries = max(0, max_retries)
        self._backoff_seconds = max(0.0, backoff_seconds)

    @property
    def inner(self) -> LLMClient:
        return self._inner

    def complete(self, system: str, user: str) -> str:
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return self._inner.complete(system=system, user=user)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= self._max_retries or not _is_retryable(exc):
                    raise
                time.sleep(self._backoff_seconds * (2 ** attempt))
        raise last_error  # pragma: no cover


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, GuardrailViolation):
        return False
    status = getattr(exc, "status_code", None)
    if status in {408, 409, 429, 500, 502, 503, 504}:
        return True
    message = str(exc).lower()
    return any(
        token in message
        for token in ("rate limit", "timeout", "temporarily unavailable", "connection reset")
    )


_PROVIDER_MAP: dict[LLMProvider, type[LLMClient]] = {
    LLMProvider.ANTHROPIC: AnthropicClient,
    LLMProvider.OPENAI: OpenAIClient,
    LLMProvider.GROQ: GroqClient,
    LLMProvider.OLLAMA: OllamaClient,
    LLMProvider.BEDROCK: BedrockClient,
}


def get_llm(cfg: Settings | None = None) -> LLMClient:
    cfg = cfg or settings
    try:
        cls = _PROVIDER_MAP[cfg.llm_provider]
    except KeyError as e:
        raise ValueError(f"Unknown LLM provider: {cfg.llm_provider}") from e
    client = cls(cfg)
    if cfg.llm_max_retries > 0:
        return RetryingLLMClient(
            inner=client,
            max_retries=cfg.llm_max_retries,
            backoff_seconds=cfg.llm_retry_backoff_seconds,
        )
    return client
