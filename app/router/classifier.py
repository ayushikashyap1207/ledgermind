from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RouteDecision:
    route: str  # needs_sql | needs_rag | needs_both
    reason: str


_SQL_HINTS = {
    "count", "sum", "average", "avg", "total", "top", "highest", "lowest",
    "invoice", "invoices", "vendor", "vendors", "transaction", "transactions",
    "ledger", "quarter", "account", "accounts", "how many", "what is the total",
}

_RAG_HINTS = {
    "policy", "memo", "agreement", "msa", "transcript", "earnings call", "document",
    "according", "retention", "approval threshold", "net 45", "late fee", "receipt",
    "reimbursement", "gross margin", "announced", "sla", "payment terms",
}


def classify_question(question: str) -> RouteDecision:
    q = question.lower()
    sql_score = sum(1 for hint in _SQL_HINTS if hint in q)
    rag_score = sum(1 for hint in _RAG_HINTS if hint in q)

    if rag_score > 0 and ("policy" in q or "msa" in q or "memo" in q or "earnings" in q):
        if any(word in q for word in ["count", "sum", "total", "how many"]):
            return RouteDecision(route="needs_both", reason="Contains explicit metric + document request.")
        return RouteDecision(route="needs_rag", reason="Primarily asks for policy/document evidence.")

    if sql_score > 0 and rag_score > 0:
        return RouteDecision(route="needs_both", reason="Contains both structured-metric and policy/doc signals.")
    if rag_score > 0:
        return RouteDecision(route="needs_rag", reason="Looks document/policy-seeking.")
    return RouteDecision(route="needs_sql", reason="Looks like structured ledger analytics.")
