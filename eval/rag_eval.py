from __future__ import annotations

import json
from dataclasses import dataclass

from app.rag.pipeline import ingest_documents, retrieve


@dataclass(frozen=True)
class RetrievalCase:
    id: str
    question: str
    expected_source_suffix: str


CASES: list[RetrievalCase] = [
    RetrievalCase("r01", "What is the company procurement approval threshold?", "procurement_policy.txt"),
    RetrievalCase("r02", "How many signatures are required for contracts above 250k?", "procurement_policy.txt"),
    RetrievalCase("r03", "What expense category must include original receipts?", "expense_policy.txt"),
    RetrievalCase("r04", "What is the reimbursement SLA for approved expenses?", "expense_policy.txt"),
    RetrievalCase("r05", "What is the announced gross margin target for FY2026?", "earnings_call_q2_2026.txt"),
    RetrievalCase("r06", "Which region showed fastest ARR growth in the latest call?", "earnings_call_q2_2026.txt"),
    RetrievalCase("r07", "When are vendor invoices due under Net terms in the MSA?", "vendor_msa_excerpt.txt"),
    RetrievalCase("r08", "What happens if invoice disputes are not raised in time?", "vendor_msa_excerpt.txt"),
    RetrievalCase("r09", "What concentration limit exists for cloud infrastructure spend?", "risk_committee_memo.txt"),
    RetrievalCase("r10", "What cap applies to unresolved audit exceptions per quarter?", "risk_committee_memo.txt"),
    RetrievalCase("r11", "Who approves emergency purchases over 50k?", "procurement_policy.txt"),
    RetrievalCase("r12", "By when should managers approve expense reports?", "expense_policy.txt"),
    RetrievalCase("r13", "What retention expectation is stated for enterprise customers?", "earnings_call_q2_2026.txt"),
    RetrievalCase("r14", "What is the late payment fee percentage in vendor terms?", "vendor_msa_excerpt.txt"),
    RetrievalCase("r15", "What is the maximum single-vendor exposure threshold?", "risk_committee_memo.txt"),
]


def run_rag_eval(
    corpus_paths: list[str] | None = None,
    chunk_size: int = 700,
    chunk_overlap: int = 120,
    top_k: int = 3,
    reset: bool = True,
) -> dict:
    corpus_paths = corpus_paths or ["docs/rag_corpus"]
    ingest = ingest_documents(
        paths=corpus_paths,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        reset=reset,
    )

    results = []
    passed = 0

    for case in CASES:
        hits = retrieve(case.question, top_k=top_k)
        top_sources = [h.source.split("/")[-1] for h in hits]
        ok = case.expected_source_suffix in top_sources[:top_k]
        if ok:
            passed += 1
        results.append(
            {
                "id": case.id,
                "question": case.question,
                "expected_source": case.expected_source_suffix,
                "top_sources": top_sources,
                "passed": ok,
            }
        )

    return {
        "phase": "phase2_rag",
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunk_count": ingest.chunks,
        "passed": passed,
        "total": len(CASES),
        "results": results,
    }


if __name__ == "__main__":
    report = run_rag_eval()
    print(json.dumps(report, indent=2))
    print(f"\n{report['passed']}/{report['total']} retrieval cases passed (top-3)")
