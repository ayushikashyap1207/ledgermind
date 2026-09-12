"""
Central configuration for LedgerMind.

Everything that varies between environments (which LLM provider to use,
where the DB lives, how strict the SQL safety layer is) lives here and
is driven by environment variables / a .env file. Nothing in the rest
of the codebase should read os.environ directly — import `settings`
from this module instead, so there's exactly one source of truth.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GROQ = "groq"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- App ---
    app_name: str = "LedgerMind"
    environment: Literal["local", "test", "production"] = "local"
    log_level: str = "INFO"

    # --- LLM provider (pluggable: swap via config, not code) ---
    llm_provider: LLMProvider = LLMProvider.ANTHROPIC
    llm_model: str = "claude-sonnet-4-6"
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    groq_api_key: str | None = None
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024
    sql_fallback_enabled: bool = True

    # --- Structured data (Text-to-SQL) ---
    # sqlite for local dev per spec; postgres in docker-compose / production.
    database_url: str = "sqlite:///./ledgermind.db"
    # A second, read-only connection string used for actually EXECUTING
    # generated SQL. In Postgres this points at a role with SELECT-only
    # grants (defense in depth beyond the app-level validator). In SQLite
    # local dev there's no native read-only role, so the executor opens
    # the file in `mode=ro` via a URI connection instead — see
    # app/sql/executor.py.
    database_readonly_url: str | None = None
    sql_row_limit: int = 500
    sql_query_timeout_seconds: int = 10

    # --- Unstructured data (RAG) ---
    chroma_persist_dir: str = "./chroma_data"
    chroma_collection_name: str = "ledgermind_docs"
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_use_http: bool = False
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 800
    chunk_overlap: int = 150
    retrieval_k: int = 8
    retrieval_k_after_mmr: int = 4

    # --- Evaluation / tracing ---
    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "ledgermind"

    @field_validator("sql_row_limit")
    @classmethod
    def _row_limit_must_be_positive_and_bounded(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("sql_row_limit must be positive")
        if v > 5000:
            raise ValueError("sql_row_limit > 5000 defeats the point of a cap")
        return v

    def active_api_key(self) -> str | None:
        """Return the API key for whichever provider is currently selected."""
        return {
            LLMProvider.ANTHROPIC: self.anthropic_api_key,
            LLMProvider.OPENAI: self.openai_api_key,
            LLMProvider.GROQ: self.groq_api_key,
        }[self.llm_provider]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
