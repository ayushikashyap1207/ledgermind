from __future__ import annotations

from dataclasses import dataclass, field

from app.core.tracing import trace_chain
from app.rag.pipeline import RetrievedChunk, retrieve
from app.router.classifier import classify_question
from app.router.decompose import decompose_question
from app.router.synthesis import synthesize_answer
from app.sql.pipeline import SQLAnswer, answer_from_sql


@dataclass
class QueryAnswer:
    route: str
    route_reason: str
    route_confidence: float
    route_method: str
    answer: str
    citations: list[dict]
    sql_status: str | None
    rag_hits: int
    sql_question: str | None = None
    rag_question: str | None = None
    decomposition_method: str | None = None
    answer_method: str = "extractive"
    claims: list[dict] = field(default_factory=list)
    insufficient_evidence: bool = False
    conflicts: list[dict] = field(default_factory=list)


@trace_chain("router_query")
def answer_query(question: str, top_k: int = 3) -> QueryAnswer:
    decision = classify_question(question)

    sql_question = question
    rag_question = question
    decomposition_method: str | None = None

    if decision.route == "needs_both":
        decomposed = decompose_question(question)
        if decomposed is not None:
            sql_question = decomposed.sql_question
            rag_question = decomposed.rag_question
            decomposition_method = decomposed.method

    sql_answer: SQLAnswer | None = None
    rag_chunks: list[RetrievedChunk] = []

    if decision.route in {"needs_sql", "needs_both"}:
        try:
            sql_answer = answer_from_sql(sql_question)
        except Exception as exc:  # noqa: BLE001
            sql_answer = SQLAnswer(
                question=sql_question,
                status="execution_error",
                generated_sql=None,
                executed_sql=None,
                result=None,
                error=str(exc),
                violations=None,
            )

    if decision.route in {"needs_rag", "needs_both"}:
        rag_chunks = retrieve(rag_question, top_k=top_k)

    merged = synthesize_answer(
        question=question,
        route=decision.route,
        sql_answer=sql_answer,
        rag_chunks=rag_chunks,
    )

    return QueryAnswer(
        route=decision.route,
        route_reason=decision.reason,
        route_confidence=decision.confidence,
        route_method=decision.method,
        answer=merged.text,
        citations=merged.citations,
        sql_status=sql_answer.status if sql_answer is not None else None,
        rag_hits=len(rag_chunks),
        sql_question=sql_question if decision.route == "needs_both" else None,
        rag_question=rag_question if decision.route == "needs_both" else None,
        decomposition_method=decomposition_method,
        answer_method=merged.method,
        claims=merged.claims,
        insufficient_evidence=merged.insufficient_evidence,
        conflicts=merged.conflicts,
    )
