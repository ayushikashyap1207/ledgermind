from __future__ import annotations

import json
from dataclasses import dataclass

from app.router.pipeline import answer_query


@dataclass(frozen=True)
class RouterCase:
    id: str
    question: str
    expected_route: str
    expects_sql_citation: bool
    expects_doc_citation: bool


CASES: list[RouterCase] = [
    RouterCase("m01", "How many invoices are overdue?", "needs_sql", True, False),
    RouterCase("m02", "What is the total transaction amount for void status?", "needs_sql", True, False),
    RouterCase("m03", "What is the procurement approval threshold?", "needs_rag", False, True),
    RouterCase("m04", "What is the reimbursement SLA after approval?", "needs_rag", False, True),
    RouterCase("m05", "What is the late payment fee in the vendor MSA?", "needs_rag", False, True),
    RouterCase("m06", "How many pending transactions do we have and what does the policy say about approval threshold?", "needs_both", True, True),
    RouterCase("m07", "Give the total overdue invoice amount and cite payment terms from vendor agreement.", "needs_both", True, True),
    RouterCase("m08", "Count vendors in India and cite risk memo exposure threshold.", "needs_both", True, True),
    RouterCase("m09", "What gross margin target was announced for FY2026?", "needs_rag", False, True),
    RouterCase("m10", "Which vendor has highest invoice total and include memo limit context.", "needs_both", True, True),
]


def run_router_eval() -> dict:
    rows = []
    passed = 0

    for case in CASES:
        ans = answer_query(case.question, top_k=3)
        has_sql = any(c.get("type") == "sql" for c in ans.citations)
        has_doc = any(c.get("type") == "doc" for c in ans.citations)
        ok = (
            ans.route == case.expected_route
            and has_sql == case.expects_sql_citation
            and has_doc == case.expects_doc_citation
        )
        if ok:
            passed += 1

        rows.append(
            {
                "id": case.id,
                "question": case.question,
                "expected_route": case.expected_route,
                "actual_route": ans.route,
                "has_sql_citation": has_sql,
                "has_doc_citation": has_doc,
                "passed": ok,
            }
        )

    return {"phase": "phase3_router", "passed": passed, "total": len(CASES), "results": rows}


if __name__ == "__main__":
    report = run_router_eval()
    print(json.dumps(report, indent=2))
    print(f"\n{report['passed']}/{report['total']} router cases passed")
