from __future__ import annotations

from dataclasses import dataclass

from app.rag.pipeline import RetrievedChunk
from app.sql.pipeline import SQLAnswer


@dataclass
class SynthesizedAnswer:
    text: str
    citations: list[dict]


def _summarize_sql(sql_answer: SQLAnswer) -> tuple[str, list[dict]]:
    if sql_answer.status != "ok" or not sql_answer.result:
        return "Structured ledger data did not return a usable result.", []

    rows = sql_answer.result.rows[:2]
    if not rows:
        summary = "Structured ledger query returned zero rows."
    elif len(rows) == 1:
        summary = f"Structured ledger result: {rows[0]}."
    else:
        summary = f"Structured ledger top rows: {rows}."

    return summary, [{"type": "sql", "ref": "SQL_RESULT", "executed_sql": sql_answer.executed_sql}]


def _summarize_rag(chunks: list[RetrievedChunk]) -> tuple[str, list[dict]]:
    if not chunks:
        return "No matching policy/doc context was retrieved.", []

    lead = chunks[0]
    snippet = lead.text[:260].replace("\n", " ").strip()
    sentence = f"Document context indicates: {snippet}"
    citations = [
        {
            "type": "doc",
            "ref": f"{c.source}#chunk-{c.chunk_index}",
            "source": c.source,
            "chunk_index": c.chunk_index,
        }
        for c in chunks
    ]
    return sentence, citations


def synthesize_answer(
    question: str,
    route: str,
    sql_answer: SQLAnswer | None,
    rag_chunks: list[RetrievedChunk],
) -> SynthesizedAnswer:
    sql_text, sql_cites = _summarize_sql(sql_answer) if sql_answer is not None else ("", [])
    rag_text, rag_cites = _summarize_rag(rag_chunks)

    if route == "needs_sql":
        body = sql_text
        citations = sql_cites
    elif route == "needs_rag":
        body = rag_text
        citations = rag_cites
    else:
        body = (
            f"For your question ({question}), the ledger result and policy context align. "
            f"{sql_text} {rag_text}"
        )
        citations = sql_cites + rag_cites

    return SynthesizedAnswer(text=body, citations=citations)
