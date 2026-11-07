from __future__ import annotations

from dataclasses import dataclass

from app.core.tracing import trace_chain
from app.sql.executor import QueryResult, execute_validated_sql
from app.sql.safety import validate_sql
from app.sql.text_to_sql import generate_sql, repair_sql


@dataclass
class SQLAnswer:
    question: str
    status: str  # "ok" | "not_answerable" | "rejected" | "execution_error"
    generated_sql: str | None
    executed_sql: str | None
    result: QueryResult | None
    error: str | None
    violations: list[str] | None
    repair_attempted: bool = False
    repair_succeeded: bool = False


@trace_chain("sql_chain")
def answer_from_sql(question: str) -> SQLAnswer:
    gen = generate_sql(question)

    if gen.not_answerable or not gen.raw_sql:
        return SQLAnswer(
            question=question,
            status="not_answerable",
            generated_sql=None,
            executed_sql=None,
            result=None,
            error=None,
            violations=None,
        )

    validation = validate_sql(gen.raw_sql)
    if not validation.is_safe:
        return SQLAnswer(
            question=question,
            status="rejected",
            generated_sql=gen.raw_sql,
            executed_sql=None,
            result=None,
            error=validation.reason,
            violations=validation.violations,
        )

    try:
        result = execute_validated_sql(validation.safe_sql)
    except Exception as exc:  # noqa: BLE001
        try:
            repaired = repair_sql(question, gen.raw_sql, str(exc))
        except Exception as repair_exc:  # noqa: BLE001
            return SQLAnswer(
                question=question,
                status="execution_error",
                generated_sql=gen.raw_sql,
                executed_sql=validation.safe_sql,
                result=None,
                error=f"{exc} (repair failed: {repair_exc})",
                violations=None,
                repair_attempted=True,
            )

        if repaired.not_answerable or not repaired.raw_sql:
            return SQLAnswer(
                question=question,
                status="execution_error",
                generated_sql=gen.raw_sql,
                executed_sql=validation.safe_sql,
                result=None,
                error=str(exc),
                violations=None,
                repair_attempted=True,
            )

        repair_validation = validate_sql(repaired.raw_sql)
        if not repair_validation.is_safe:
            return SQLAnswer(
                question=question,
                status="execution_error",
                generated_sql=gen.raw_sql,
                executed_sql=validation.safe_sql,
                result=None,
                error=f"{exc} (repair rejected: {repair_validation.reason})",
                violations=repair_validation.violations,
                repair_attempted=True,
            )

        try:
            result = execute_validated_sql(repair_validation.safe_sql)
        except Exception as repair_exec_exc:  # noqa: BLE001
            return SQLAnswer(
                question=question,
                status="execution_error",
                generated_sql=repaired.raw_sql,
                executed_sql=repair_validation.safe_sql,
                result=None,
                error=f"original: {exc}; repair: {repair_exec_exc}",
                violations=None,
                repair_attempted=True,
            )

        return SQLAnswer(
            question=question,
            status="ok",
            generated_sql=repaired.raw_sql,
            executed_sql=repair_validation.safe_sql,
            result=result,
            error=None,
            violations=None,
            repair_attempted=True,
            repair_succeeded=True,
        )

    return SQLAnswer(
        question=question,
        status="ok",
        generated_sql=gen.raw_sql,
        executed_sql=validation.safe_sql,
        result=result,
        error=None,
        violations=None,
    )
