from __future__ import annotations

from dataclasses import dataclass

from app.core.tracing import trace_chain
from app.rag.pipeline import RetrievedChunk, retrieve
from app.router.classifier import classify_question
from app.router.synthesis import synthesize_answer
from app.sql.pipeline import SQLAnswer, answer_from_sql


@dataclass
class QueryAnswer:
    route: str
    route_reason: str
    answer: str
    citations: list[dict]
    sql_status: str | None
    rag_hits: int


@trace_chain("router_query")
def answer_query(question: str, top_k: int = 3) -> QueryAnswer:
    decision = classify_question(question)

    sql_answer: SQLAnswer | None = None
    rag_chunks: list[RetrievedChunk] = []

    if decision.route in {"needs_sql", "needs_both"}:
        try:
            sql_answer = answer_from_sql(question)
        except Exception as exc:  # noqa: BLE001
            sql_answer = SQLAnswer(
                question=question,
                status="execution_error",
                generated_sql=None,
                executed_sql=None,
                result=None,
                error=str(exc),
                violations=None,
            )

    if decision.route in {"needs_rag", "needs_both"}:
        rag_chunks = retrieve(question, top_k=top_k)

    merged = synthesize_answer(
        question=question,
        route=decision.route,
        sql_answer=sql_answer,
        rag_chunks=rag_chunks,
    )

    return QueryAnswer(
        route=decision.route,
        route_reason=decision.reason,
        answer=merged.text,
        citations=merged.citations,
        sql_status=sql_answer.status if sql_answer is not None else None,
        rag_hits=len(rag_chunks),
    )
