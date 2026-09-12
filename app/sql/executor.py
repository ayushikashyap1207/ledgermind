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

from dataclasses import dataclass
from functools import lru_cache

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import OperationalError

from app.core.config import settings


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


def execute_validated_sql(safe_sql: str, timeout_seconds: int | None = None) -> QueryResult:
    timeout_seconds = timeout_seconds or settings.sql_query_timeout_seconds
    engine = get_readonly_engine()
    try:
        with engine.connect() as conn:
            result = conn.execute(text(safe_sql))
            columns = list(result.keys())
            rows = [dict(zip(columns, row)) for row in result.fetchall()]
            return QueryResult(columns=columns, rows=rows, row_count=len(rows))
    except OperationalError as exc:
        if "readonly" in str(exc).lower():
            raise PermissionError(
                "Blocked at the connection level: this is a read-only database handle."
            ) from exc
        raise
