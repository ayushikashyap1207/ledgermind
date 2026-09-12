from __future__ import annotations

from dataclasses import dataclass

from app.core.tracing import trace_chain
from app.sql.executor import QueryResult, execute_validated_sql
from app.sql.safety import validate_sql
from app.sql.text_to_sql import generate_sql


@dataclass
class SQLAnswer:
    question: str
    status: str  # "ok" | "not_answerable" | "rejected" | "execution_error"
    generated_sql: str | None
    executed_sql: str | None
    result: QueryResult | None
    error: str | None
    violations: list[str] | None


@trace_chain("sql_chain")
def answer_from_sql(question: str) -> SQLAnswer:
    gen = generate_sql(question)

    if gen.not_answerable or not gen.raw_sql:
        return SQLAnswer(question, "not_answerable", None, None, None, None, None)

    validation = validate_sql(gen.raw_sql)
    if not validation.is_safe:
        return SQLAnswer(
            question, "rejected", gen.raw_sql, None, None, validation.reason, validation.violations,
        )

    try:
        result = execute_validated_sql(validation.safe_sql)
    except Exception as exc:  # noqa: BLE001
        return SQLAnswer(question, "execution_error", gen.raw_sql, validation.safe_sql, None, str(exc), None)

    return SQLAnswer(question, "ok", gen.raw_sql, validation.safe_sql, result, None, None)
