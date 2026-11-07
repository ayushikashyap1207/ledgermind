"""
Executes ALREADY-VALIDATED SQL against a read-only connection.

This is layer 2 of the defense-in-depth story: even if the safety
validator in safety.py had a bug and let something mutating through,
this connection physically cannot write.

- Postgres (production/docker-compose): DATABASE_READONLY_URL points at
  a role created with only SELECT grants (see docker/init-readonly-role.sql).
- SQLite (local dev): opened via a `file:...?mode=ro` URI, which SQLite
  enforces at the OS/file-handle level — write attempts raise
  sqlite3.OperationalError: attempt to write a readonly database.

Callers MUST go through safety.validate_sql() first. This module does
not re-validate; it trusts its caller by contract, same as any
prepared-statement boundary.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from functools import lru_cache

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import OperationalError

from app.core.config import settings


class QueryTimeout(RuntimeError):
    """The query exceeded its execution deadline."""


class QueryResultTooLarge(RuntimeError):
    """The query returned more rows than the configured cap."""


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[dict]
    row_count: int


@lru_cache
def get_readonly_engine() -> Engine:
    if settings.database_readonly_url:
        return create_engine(settings.database_readonly_url)

    if settings.database_url.startswith("sqlite"):
        # sqlite:///./ledgermind.db -> ./ledgermind.db
        path = settings.database_url.split("sqlite:///", 1)[-1]
        ro_url = f"sqlite+pysqlite:///file:{path}?mode=ro&uri=true"
        return create_engine(ro_url, connect_args={"check_same_thread": False})

    raise RuntimeError(
        "No DATABASE_READONLY_URL configured and DATABASE_URL is not sqlite; "
        "refusing to execute against a read-write connection."
    )


def _sqlite_driver_connection(conn):
    fairy = conn.connection
    return getattr(fairy, "driver_connection", None)


def execute_validated_sql(
    safe_sql: str, timeout_seconds: float | None = None
) -> QueryResult:
    timeout_seconds = (
        timeout_seconds
        if timeout_seconds is not None
        else settings.sql_query_timeout_seconds
    )
    if timeout_seconds <= 0:
        timeout_seconds = settings.sql_query_timeout_seconds

    engine = get_readonly_engine()
    deadline = time.monotonic() + timeout_seconds

    try:
        with engine.connect() as conn:
            if engine.dialect.name == "postgresql":
                # Session-level guard; the connection is read-only anyway.
                conn.execute(
                    text(f"SET statement_timeout = {int(timeout_seconds * 1000)}")
                )

            raw_connection = _sqlite_driver_connection(conn)
            progress_handler_installed = False
            if isinstance(raw_connection, sqlite3.Connection):
                def _abort_after_deadline() -> int:
                    return 1 if time.monotonic() > deadline else 0

                raw_connection.set_progress_handler(_abort_after_deadline, 1000)
                progress_handler_installed = True

            try:
                result = conn.execute(text(safe_sql))
                columns = list(result.keys())
                rows = result.fetchmany(settings.sql_row_limit + 1)
            finally:
                if progress_handler_installed:
                    raw_connection.set_progress_handler(None, 0)

            if len(rows) > settings.sql_row_limit:
                raise QueryResultTooLarge(
                    f"Query returned more than the configured row cap "
                    f"({settings.sql_row_limit})."
                )

            materialized = [dict(zip(columns, row)) for row in rows]
            return QueryResult(
                columns=columns, rows=materialized, row_count=len(materialized)
            )
    except OperationalError as exc:
        message = str(exc).lower()
        if "interrupted" in message:
            raise QueryTimeout(
                f"SQL query exceeded the {timeout_seconds:g}s timeout."
            ) from exc
        if "readonly" in message:
            raise PermissionError(
                "Blocked at the connection level: this is a read-only database handle."
            ) from exc
        raise
