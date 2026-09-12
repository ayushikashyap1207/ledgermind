from __future__ import annotations

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


E2E_CASES: list[EndToEndCase] = [
    EndToEndCase(f"e{i:02d}", q.question, "needs_rag") for i, q in enumerate(RAG_CASES, start=1)
] + [
    EndToEndCase("e16", "How many invoices are overdue?", "needs_sql"),
    EndToEndCase("e17", "What is the total amount of overdue invoices?", "needs_sql"),
    EndToEndCase("e18", "How many pending transactions are there?", "needs_sql"),
    EndToEndCase("e19", "How many vendors are based in India?", "needs_sql"),
    EndToEndCase("e20", "What is average invoice amount?", "needs_sql"),
    EndToEndCase("e21", "How many pending transactions and what approval threshold applies?", "needs_both"),
    EndToEndCase("e22", "Total overdue invoices and MSA payment terms?", "needs_both"),
    EndToEndCase("e23", "Vendor count in Germany and retention expectation?", "needs_both"),
    EndToEndCase("e24", "Top vendor by invoiced amount plus concentration cap?", "needs_both"),
    EndToEndCase("e25", "Void transaction total and unresolved audit exception cap?", "needs_both"),
    EndToEndCase("e26", "How many revenue accounts exist?", "needs_sql"),
    EndToEndCase("e27", "What is the gross margin target for FY2026?", "needs_rag"),
    EndToEndCase("e28", "What is the reimbursement timeline for approved expenses?", "needs_rag"),
    EndToEndCase("e29", "What is late payment fee percent in MSA?", "needs_rag"),
    EndToEndCase("e30", "What is max single-vendor exposure threshold?", "needs_rag"),
]


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
        raw = llm.complete(system="Judge groundedness only.", user=prompt)
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


def run_full_stack_eval() -> dict:
    sql_eval = run_full_eval()
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
                evidence.append(cite.get("ref", ""))
            if cite.get("type") == "sql":
                evidence.append("SQL_RESULT")

        judge = _judge_faithfulness(case.question, ans.answer, evidence)
        faithfulness_rows.append(
            {
                "id": case.id,
                "question": case.question,
                "faithfulness": judge,
                "latency_ms": round(latency_ms, 2),
            }
        )

    faithful_n = sum(1 for row in faithfulness_rows if row["faithfulness"].get("faithful"))

    return {
        "phase": "phase4_eval_suite",
        "retrieval": retrieval,
        "sql_execution_correctness": {
            "passed": sql_eval["passed"],
            "total": sql_eval["total"],
            "accuracy": round(sql_eval["passed"] / sql_eval["total"], 4),
        },
        "answer_faithfulness": {
            "faithful": faithful_n,
            "total": len(faithfulness_rows),
            "rate": round(faithful_n / len(faithfulness_rows), 4),
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
    report = run_full_stack_eval()
    print(json.dumps(report, indent=2))
