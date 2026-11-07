import app.sql.pipeline as pipeline
from app.sql.executor import QueryResult
from app.sql.safety import ValidationResult
from app.sql.text_to_sql import SQLGenerationResult, repair_sql


class FakeLLM:
    def __init__(self, response: str):
        self.response = response
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        return self.response


def test_repair_sql_parses_fenced_completion():
    llm = FakeLLM("```sql\nSELECT i.invoice_number FROM invoices i\n```")

    result = repair_sql("Which invoices are overdue?", "SELECT bad", "no such column", llm=llm)

    assert result.raw_sql == "SELECT i.invoice_number FROM invoices i"
    assert not result.not_answerable
    assert llm.calls == 1


def test_answer_from_sql_repairs_execution_error(monkeypatch):
    calls = {"execute": 0}

    monkeypatch.setattr(
        pipeline,
        "generate_sql",
        lambda question: SQLGenerationResult(question, "SELECT bad", False),
    )
    monkeypatch.setattr(
        pipeline,
        "validate_sql",
        lambda sql: ValidationResult(is_safe=True, safe_sql=sql),
    )

    def fake_execute(sql: str) -> QueryResult:
        calls["execute"] += 1
        if calls["execute"] == 1:
            raise RuntimeError("ambiguous column name")
        return QueryResult(columns=["n"], rows=[{"n": 1}], row_count=1)

    monkeypatch.setattr(pipeline, "execute_validated_sql", fake_execute)
    monkeypatch.setattr(
        pipeline,
        "repair_sql",
        lambda question, failed_sql, error: SQLGenerationResult(
            question, "SELECT fixed", False
        ),
    )

    answer = pipeline.answer_from_sql("How many invoices are overdue?")

    assert answer.status == "ok"
    assert answer.repair_attempted
    assert answer.repair_succeeded
    assert answer.executed_sql == "SELECT fixed"
    assert answer.result.rows == [{"n": 1}]


def test_answer_from_sql_reports_failed_repair(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "generate_sql",
        lambda question: SQLGenerationResult(question, "SELECT bad", False),
    )
    monkeypatch.setattr(
        pipeline,
        "validate_sql",
        lambda sql: ValidationResult(is_safe=True, safe_sql=sql),
    )

    def always_fail(sql: str):
        raise RuntimeError("still broken")

    monkeypatch.setattr(pipeline, "execute_validated_sql", always_fail)
    monkeypatch.setattr(
        pipeline,
        "repair_sql",
        lambda question, failed_sql, error: SQLGenerationResult(
            question, "SELECT still_bad", False
        ),
    )

    answer = pipeline.answer_from_sql("How many invoices are overdue?")

    assert answer.status == "execution_error"
    assert answer.repair_attempted
    assert not answer.repair_succeeded
    assert "still broken" in answer.error


def test_unsafe_generated_sql_is_not_repaired(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "generate_sql",
        lambda question: SQLGenerationResult(question, "DROP TABLE invoices", False),
    )
    monkeypatch.setattr(
        pipeline,
        "validate_sql",
        lambda sql: ValidationResult(
            is_safe=False, reason="Only SELECT statements are allowed"
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "repair_sql",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unsafe SQL must not be repaired")
        ),
    )

    answer = pipeline.answer_from_sql("Delete every invoice")

    assert answer.status == "rejected"
    assert not answer.repair_attempted
