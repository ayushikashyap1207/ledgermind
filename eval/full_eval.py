from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass

from app.core.llm import get_llm
from app.core.config import settings
from app.rag.pipeline import retrieve
from app.router.pipeline import answer_query
from eval.rag_eval import CASES as RAG_CASES
from eval.sql_eval import run_full_eval


@dataclass(frozen=True)
class EndToEndCase:
    id: str
    question: str
    route: str
    expected_source_suffix: str | None = None
    expects_sql_citation: bool = False
    expects_doc_citation: bool = False


E2E_CASES: list[EndToEndCase] = [
    EndToEndCase(
        f"e{i:02d}",
        q.question,
        "needs_rag",
        expected_source_suffix=q.expected_source_suffix,
        expects_doc_citation=True,
    )
    for i, q in enumerate(RAG_CASES, start=1)
] + [
    EndToEndCase("e16", "How many invoices are overdue?", "needs_sql", expects_sql_citation=True),
    EndToEndCase("e17", "What is the total amount of overdue invoices?", "needs_sql", expects_sql_citation=True),
    EndToEndCase("e18", "How many pending transactions are there?", "needs_sql", expects_sql_citation=True),
    EndToEndCase("e19", "How many vendors are based in India?", "needs_sql", expects_sql_citation=True),
    EndToEndCase("e20", "What is average invoice amount?", "needs_sql", expects_sql_citation=True),
    EndToEndCase(
        "e21",
        "How many pending transactions and what approval threshold applies?",
        "needs_both",
        expected_source_suffix="procurement_policy.txt",
        expects_sql_citation=True,
        expects_doc_citation=True,
    ),
    EndToEndCase(
        "e22",
        "Total overdue invoices and MSA payment terms?",
        "needs_both",
        expected_source_suffix="vendor_msa_excerpt.txt",
        expects_sql_citation=True,
        expects_doc_citation=True,
    ),
    EndToEndCase(
        "e23",
        "Vendor count in Germany and retention expectation?",
        "needs_both",
        expected_source_suffix="earnings_call_q2_2026.txt",
        expects_sql_citation=True,
        expects_doc_citation=True,
    ),
    EndToEndCase(
        "e24",
        "Top vendor by invoiced amount plus concentration cap?",
        "needs_both",
        expected_source_suffix="risk_committee_memo.txt",
        expects_sql_citation=True,
        expects_doc_citation=True,
    ),
    EndToEndCase(
        "e25",
        "Void transaction total and unresolved audit exception cap?",
        "needs_both",
        expected_source_suffix="risk_committee_memo.txt",
        expects_sql_citation=True,
        expects_doc_citation=True,
    ),
    EndToEndCase("e26", "How many revenue accounts exist?", "needs_sql", expects_sql_citation=True),
    EndToEndCase(
        "e27",
        "What is the gross margin target for FY2026?",
        "needs_rag",
        expected_source_suffix="earnings_call_q2_2026.txt",
        expects_doc_citation=True,
    ),
    EndToEndCase(
        "e28",
        "What is the reimbursement timeline for approved expenses?",
        "needs_rag",
        expected_source_suffix="expense_policy.txt",
        expects_doc_citation=True,
    ),
    EndToEndCase(
        "e29",
        "What is late payment fee percent in MSA?",
        "needs_rag",
        expected_source_suffix="vendor_msa_excerpt.txt",
        expects_doc_citation=True,
    ),
    EndToEndCase(
        "e30",
        "What is max single-vendor exposure threshold?",
        "needs_rag",
        expected_source_suffix="risk_committee_memo.txt",
        expects_doc_citation=True,
    ),
]


def _citation_check(answer, case: EndToEndCase) -> dict:
    """Verify the expected citation types are present and that a doc
    citation points at the expected source, not just any document."""
    has_sql = any(c.get("type") == "sql" for c in answer.citations)
    doc_sources = [
        str(c.get("source") or c.get("ref") or "")
        for c in answer.citations
        if c.get("type") == "doc"
    ]

    sql_type_ok = has_sql == case.expects_sql_citation
    doc_type_ok = bool(doc_sources) == case.expects_doc_citation

    source_ok = True
    if case.expects_doc_citation and case.expected_source_suffix:
        source_ok = any(case.expected_source_suffix in source for source in doc_sources)

    return {
        "has_sql_citation": has_sql,
        "has_doc_citation": bool(doc_sources),
        "doc_sources": doc_sources,
        "sql_type_ok": sql_type_ok,
        "doc_type_ok": doc_type_ok,
        "expected_source_present": source_ok if case.expects_doc_citation else None,
        "passed": sql_type_ok and doc_type_ok and source_ok,
    }


def _judge_faithfulness(question: str, answer: str, evidence: list[str]) -> dict:
    rubric = {
        "criteria": [
            "Answer is grounded in provided evidence only",
            "No fabricated specific values",
            "Citations are relevant to claims",
        ],
        "scale": "1-5",
    }

    if settings.active_api_key():
        llm = get_llm()
        prompt = (
            "You are a strict faithfulness judge. Return JSON with keys: score, faithful, rationale. "
            "Score 1-5 where 5 is fully grounded.\n"
            f"Rubric: {json.dumps(rubric)}\n"
            f"Question: {question}\n"
            f"Answer: {answer}\n"
            f"Evidence: {evidence}\n"
        )
        try:
            raw = llm.complete(system="Judge groundedness only.", user=prompt)
        except Exception as exc:  # noqa: BLE001
            return {
                "score": None,
                "faithful": None,
                "rationale": f"Judge unavailable: {exc}",
                "rubric": rubric,
                "judge_mode": "error",
            }
        try:
            out = json.loads(raw)
            out["rubric"] = rubric
            out["judge_mode"] = "llm"
            return out
        except Exception:
            return {"score": 2, "faithful": False, "rationale": raw, "rubric": rubric, "judge_mode": "llm_parse_fallback"}

    faithful = bool(evidence) and ("result" in answer.lower() or "context" in answer.lower())
    return {
        "score": 4 if faithful else 2,
        "faithful": faithful,
        "rationale": "Heuristic fallback due to missing API key.",
        "rubric": rubric,
        "judge_mode": "heuristic",
    }


def _retrieval_metrics() -> dict:
    k = 3
    tp = 0
    precision_vals = []
    recall_vals = []

    for case in RAG_CASES:
        hits = retrieve(case.question, top_k=k)
        top_sources = [h.source.split("/")[-1] for h in hits]
        found = case.expected_source_suffix in top_sources
        if found:
            tp += 1
            precision_vals.append(1.0 / k)
            recall_vals.append(1.0)
        else:
            precision_vals.append(0.0)
            recall_vals.append(0.0)

    return {
        "k": k,
        "precision_at_k": round(sum(precision_vals) / len(precision_vals), 4),
        "recall_at_k": round(sum(recall_vals) / len(recall_vals), 4),
        "hits": tp,
        "total": len(RAG_CASES),
    }


def run_full_stack_eval(
    skip_sql_suite: bool = False, skip_judge: bool = False
) -> dict:
    sql_eval = None if skip_sql_suite else run_full_eval()
    retrieval = _retrieval_metrics()

    faithfulness_rows = []
    latencies_ms = []

    for case in E2E_CASES:
        t0 = time.perf_counter()
        ans = answer_query(case.question, top_k=3)
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(latency_ms)

        evidence = []
        for cite in ans.citations:
            if cite.get("type") == "doc":
                evidence_id = cite.get("evidence_id", cite.get("ref", "doc"))
                evidence.append(f"[{evidence_id}] {cite.get('text', '')}")
            if cite.get("type") == "sql":
                evidence_id = cite.get("evidence_id", "S1")
                evidence.append(
                    f"[{evidence_id}] executed_sql={cite.get('executed_sql')} "
                    f"columns={cite.get('columns')} rows={cite.get('rows')}"
                )

        if skip_judge:
            judge = {
                "score": None,
                "faithful": None,
                "rationale": "Judge skipped by --skip-judge.",
                "judge_mode": "skipped",
            }
        else:
            judge = _judge_faithfulness(case.question, ans.answer, evidence)
        citation = _citation_check(ans, case)
        faithfulness_rows.append(
            {
                "id": case.id,
                "question": case.question,
                "expected_route": case.route,
                "actual_route": ans.route,
                "route_correct": ans.route == case.route,
                "route_confidence": ans.route_confidence,
                "route_method": ans.route_method,
                "sql_question": ans.sql_question,
                "rag_question": ans.rag_question,
                "decomposition_method": ans.decomposition_method,
                "answer_method": ans.answer_method,
                "insufficient_evidence": ans.insufficient_evidence,
                "sql_status": ans.sql_status,
                "rag_hits": ans.rag_hits,
                "citations": citation,
                "citation_correct": citation["passed"],
                "faithfulness": judge,
                "latency_ms": round(latency_ms, 2),
            }
        )

    judged_rows = [
        row for row in faithfulness_rows if row["faithfulness"].get("faithful") is not None
    ]
    faithful_n = sum(1 for row in judged_rows if row["faithfulness"].get("faithful"))
    skipped_judge = sum(
        1
        for row in faithfulness_rows
        if row["faithfulness"].get("judge_mode") == "skipped"
    )
    judge_errors = len(faithfulness_rows) - len(judged_rows) - skipped_judge
    route_n = sum(1 for row in faithfulness_rows if row["route_correct"])
    citation_n = sum(1 for row in faithfulness_rows if row["citation_correct"])
    total_rows = len(faithfulness_rows)

    return {
        "phase": "phase4_eval_suite",
        "retrieval": retrieval,
        "sql_execution_correctness": (
            {
                "passed": sql_eval["passed"],
                "total": sql_eval["total"],
                "accuracy": round(sql_eval["passed"] / sql_eval["total"], 4),
            }
            if sql_eval is not None
            else None
        ),
        "routing_accuracy": {
            "correct": route_n,
            "total": total_rows,
            "accuracy": round(route_n / total_rows, 4),
            "rows": [
                {
                    "id": row["id"],
                    "question": row["question"],
                    "expected_route": row["expected_route"],
                    "actual_route": row["actual_route"],
                    "route_confidence": row["route_confidence"],
                    "route_method": row["route_method"],
                    "decomposition_method": row["decomposition_method"],
                    "passed": row["route_correct"],
                }
                for row in faithfulness_rows
            ],
        },
        "citation_correctness": {
            "correct": citation_n,
            "total": total_rows,
            "accuracy": round(citation_n / total_rows, 4),
            "rows": [
                {
                    "id": row["id"],
                    "question": row["question"],
                    "passed": row["citation_correct"],
                    **row["citations"],
                }
                for row in faithfulness_rows
            ],
        },
        "answer_faithfulness": {
            "faithful": faithful_n,
            "judged": len(judged_rows),
            "skipped": skipped_judge,
            "errors": judge_errors,
            "total": total_rows,
            "rate": round(faithful_n / len(judged_rows), 4) if judged_rows else None,
            "judgments": faithfulness_rows,
        },
        "latency": {
            "p50_ms": round(statistics.median(latencies_ms), 2),
            "mean_ms": round(sum(latencies_ms) / len(latencies_ms), 2),
            "max_ms": round(max(latencies_ms), 2),
            "total_cases": len(latencies_ms),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-sql-suite",
        action="store_true",
        help="Skip the 15-question SQL suite (saves tokens).",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip LLM faithfulness judging (saves tokens).",
    )
    args = parser.parse_args()

    report = run_full_stack_eval(
        skip_sql_suite=args.skip_sql_suite,
        skip_judge=args.skip_judge,
    )
    print(json.dumps(report, indent=2))
