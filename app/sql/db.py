from __future__ import annotations

from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings


@lru_cache
def get_engine() -> Engine:
    """Read/write engine — used for schema setup and synthetic data load only.
    Generated (LLM) queries must NEVER go through this engine; they go
    through get_readonly_engine() in executor.py."""
    connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
    return create_engine(settings.database_url, connect_args=connect_args)


def get_sessionmaker() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine())
