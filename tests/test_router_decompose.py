from app.router.classifier import RouteDecision
from app.router.decompose import DecomposedQuestion, decompose_question
from app.sql.pipeline import SQLAnswer
import app.router.pipeline as pipeline


class FakeLLM:
    def __init__(self, response: str | None = None, error: Exception | None = None):
        self.response = response
        self.error = error

    def complete(self, system: str, user: str) -> str:
        if self.error is not None:
            raise self.error
        return self.response or ""


def test_llm_decomposition_returns_two_subquestions():
    llm = FakeLLM(
        '{"sql_question":"How many pending transactions are there?",'
        '"rag_question":"What approval threshold applies?",'
        '"reason":"Ledger count plus policy."}'
    )

    result = decompose_question(
        "How many pending transactions and what approval threshold applies?", llm=llm
    )

    assert result is not None
    assert result.method == "llm"
    assert result.sql_question == "How many pending transactions are there?"
    assert result.rag_question == "What approval threshold applies?"


def test_fenced_json_is_parsed():
    llm = FakeLLM(
        '```json\n{"sql_question":"Total overdue invoices?",'
        '"rag_question":"What are the MSA payment terms?",'
        '"reason":"Both."}\n```'
    )

    result = decompose_question("Total overdue invoices and MSA payment terms?", llm=llm)

    assert result is not None
    assert result.method == "llm"
    assert result.sql_question == "Total overdue invoices?"
    assert result.rag_question == "What are the MSA payment terms?"


def test_malformed_json_falls_back_to_heuristic_split():
    llm = FakeLLM("I cannot split this.")

    result = decompose_question("Top vendor by invoiced amount plus concentration cap?", llm=llm)

    assert result is not None
    assert result.method == "heuristic"
    assert result.sql_question == "Top vendor by invoiced amount"
    assert result.rag_question == "Concentration cap?"


def test_provider_error_falls_back_to_heuristic_split():
    llm = FakeLLM(error=RuntimeError("provider unavailable"))

    result = decompose_question(
        "Count vendors in India and cite risk memo exposure threshold.", llm=llm
    )

    assert result is not None
    assert result.method == "heuristic"
    assert result.sql_question == "Count vendors in India"
    assert result.rag_question == "Cite risk memo exposure threshold?"


def test_sql_statement_output_falls_back_to_heuristic():
    llm = FakeLLM(
        '{"sql_question":"SELECT SUM(amount) FROM transactions WHERE status=\'void\'",'
        '"rag_question":"What is the unresolved audit exception cap?",'
        '"reason":"SQL leaked into the question field."}'
    )

    result = decompose_question(
        "Void transaction total and unresolved audit exception cap?", llm=llm
    )

    assert result is not None
    assert result.method == "heuristic"
    assert result.sql_question == "Void transaction total"
    assert result.rag_question == "Unresolved audit exception cap?"


def test_identical_subquestions_are_rejected():
    llm = FakeLLM(
        '{"sql_question":"Same question?","rag_question":"Same question?",'
        '"reason":"No real split."}'
    )

    result = decompose_question("Tell me about the company", llm=llm)

    assert result is None


def test_answer_query_uses_decomposed_subquestions(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "classify_question",
        lambda q: RouteDecision("needs_both", "both", 0.99, "llm"),
    )
    monkeypatch.setattr(
        pipeline,
        "decompose_question",
        lambda q: DecomposedQuestion("sql sub-question", "rag sub-question", "test", "llm"),
    )

    seen: dict[str, str] = {}

    def fake_sql(question: str) -> SQLAnswer:
        seen["sql"] = question
        return SQLAnswer(
            question=question,
            status="not_answerable",
            generated_sql=None,
            executed_sql=None,
            result=None,
            error=None,
            violations=None,
        )

    def fake_retrieve(question: str, top_k: int = 3, tenant_id: str = "default"):
        seen["rag"] = question
        return []

    monkeypatch.setattr(pipeline, "answer_from_sql", fake_sql)
    monkeypatch.setattr(pipeline, "retrieve", fake_retrieve)

    answer = pipeline.answer_query("compound question")

    assert seen == {"sql": "sql sub-question", "rag": "rag sub-question"}
    assert answer.sql_question == "sql sub-question"
    assert answer.rag_question == "rag sub-question"
    assert answer.decomposition_method == "llm"
