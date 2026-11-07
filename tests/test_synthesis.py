from app.rag.pipeline import RetrievedChunk
from app.router.synthesis import _unsupported_numbers, synthesize_answer
from app.sql.executor import QueryResult
from app.sql.pipeline import SQLAnswer


class FakeLLM:
    def __init__(self, responses):
        if not isinstance(responses, list):
            responses = [responses]
        self.responses = responses
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        response = self.responses[min(self.calls - 1, len(self.responses) - 1)]
        if isinstance(response, Exception):
            raise response
        return response


class ExplodingLLM:
    def complete(self, system: str, user: str) -> str:
        raise AssertionError("synthesis LLM should not be called without evidence")


def make_chunk(text: str, index: int = 0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"docs/policy.txt::chunk-{index}",
        source="docs/policy.txt",
        chunk_index=index,
        text=text,
        score=0.9,
    )


def make_sql_answer(rows: list[dict], columns: list[str] | None = None) -> SQLAnswer:
    result = QueryResult(
        columns=columns or (list(rows[0].keys()) if rows else []),
        rows=rows,
        row_count=len(rows),
    )
    return SQLAnswer(
        question="q",
        status="ok",
        generated_sql="SELECT ...",
        executed_sql="SELECT ...",
        result=result,
        error=None,
        violations=None,
    )


VALID_RESPONSE = """
{
  "answer": "The procurement approval threshold is 25,000 USD [D1].",
  "claims": [
    {"text": "The procurement approval threshold is 25,000 USD.", "citations": ["D1"]}
  ],
  "insufficient_evidence": false,
  "missing_evidence": "",
  "conflicts": []
}
"""


def test_valid_llm_synthesis_is_returned_with_claims():
    chunk = make_chunk("The procurement approval threshold is 25,000 USD.")
    llm = FakeLLM(VALID_RESPONSE)

    result = synthesize_answer(
        "What is the procurement approval threshold?",
        "needs_rag",
        None,
        [chunk],
        llm=llm,
    )

    assert result.method == "llm"
    assert "25,000 USD [D1]" in result.text
    assert result.citations[0]["evidence_id"] == "D1"
    assert result.claims == [
        {
            "text": "The procurement approval threshold is 25,000 USD.",
            "citations": ["D1"],
        }
    ]
    assert not result.insufficient_evidence


def test_malformed_json_falls_back_to_extractive_answer():
    chunk = make_chunk(
        "Section 1. The procurement approval threshold is 25,000 USD. "
        "Unrelated material follows here."
    )
    llm = FakeLLM("not json at all")

    result = synthesize_answer(
        "What is the procurement approval threshold?",
        "needs_rag",
        None,
        [chunk],
        llm=llm,
    )

    assert result.method == "extractive"
    assert "25,000 USD" in result.text
    assert "[D1]" in result.text


def test_unsupported_number_is_repaired_on_second_attempt():
    chunk = make_chunk("The procurement approval threshold is 25,000 USD.")
    invalid = """
    {
      "answer": "The procurement approval threshold is 99,000 USD [D1].",
      "claims": [{"text": "The threshold is 99,000 USD.", "citations": ["D1"]}],
      "insufficient_evidence": false,
      "missing_evidence": "",
      "conflicts": []
    }
    """
    llm = FakeLLM([invalid, VALID_RESPONSE])

    result = synthesize_answer(
        "What is the procurement approval threshold?",
        "needs_rag",
        None,
        [chunk],
        llm=llm,
    )

    assert llm.calls == 2
    assert result.method == "llm"
    assert "25,000" in result.text
    assert "99,000" not in result.text


def test_persistently_unsupported_number_falls_back():
    chunk = make_chunk("The procurement approval threshold is 25,000 USD.")
    invalid = """
    {
      "answer": "The procurement approval threshold is 99,000 USD [D1].",
      "claims": [{"text": "The threshold is 99,000 USD.", "citations": ["D1"]}],
      "insufficient_evidence": false,
      "missing_evidence": "",
      "conflicts": []
    }
    """
    llm = FakeLLM([invalid, invalid])

    result = synthesize_answer(
        "What is the procurement approval threshold?",
        "needs_rag",
        None,
        [chunk],
        llm=llm,
    )

    assert llm.calls == 2
    assert result.method == "extractive"
    assert "25,000" in result.text


def test_claim_citing_the_wrong_source_is_repaired():
    procurement = make_chunk("The procurement approval threshold is 25,000 USD.", index=0)
    expense = make_chunk(
        "Approved expense reports are reimbursed within 10 business days.", index=1
    )
    wrong_source = """
    {
      "answer": "The procurement approval threshold is 25,000 USD [D2].",
      "claims": [
        {"text": "The procurement approval threshold is 25,000 USD.", "citations": ["D2"]}
      ],
      "insufficient_evidence": false,
      "missing_evidence": "",
      "conflicts": []
    }
    """
    corrected = """
    {
      "answer": "The procurement approval threshold is 25,000 USD [D1].",
      "claims": [
        {"text": "The procurement approval threshold is 25,000 USD.", "citations": ["D1"]}
      ],
      "insufficient_evidence": false,
      "missing_evidence": "",
      "conflicts": []
    }
    """
    llm = FakeLLM([wrong_source, corrected])

    result = synthesize_answer(
        "What is the procurement approval threshold?",
        "needs_rag",
        None,
        [procurement, expense],
        llm=llm,
    )

    assert llm.calls == 2
    assert result.method == "llm"
    assert "[D1]" in result.text
    assert "[D2]" not in result.text


def test_persistently_wrong_source_falls_back_to_extractive():
    procurement = make_chunk("The procurement approval threshold is 25,000 USD.", index=0)
    expense = make_chunk(
        "Approved expense reports are reimbursed within 10 business days.", index=1
    )
    wrong_source = """
    {
      "answer": "The procurement approval threshold is 25,000 USD [D2].",
      "claims": [
        {"text": "The procurement approval threshold is 25,000 USD.", "citations": ["D2"]}
      ],
      "insufficient_evidence": false,
      "missing_evidence": "",
      "conflicts": []
    }
    """
    llm = FakeLLM([wrong_source, wrong_source])

    result = synthesize_answer(
        "What is the procurement approval threshold?",
        "needs_rag",
        None,
        [procurement, expense],
        llm=llm,
    )

    assert result.method == "extractive"
    assert "25,000" in result.text
    assert "[D1]" in result.text


def test_insufficient_evidence_is_passed_through():
    chunk = make_chunk("The policy covers expense reports but not FY2027 targets.")
    response = """
    {
      "answer": "The evidence does not state an FY2027 target.",
      "claims": [],
      "insufficient_evidence": true,
      "missing_evidence": "No FY2027 target was retrieved.",
      "conflicts": []
    }
    """
    llm = FakeLLM(response)

    result = synthesize_answer(
        "What is the FY2027 target?",
        "needs_rag",
        None,
        [chunk],
        llm=llm,
    )

    assert result.method == "llm"
    assert result.insufficient_evidence
    assert result.missing_evidence == "No FY2027 target was retrieved."


def test_no_evidence_returns_abstention_without_calling_llm():
    sql_answer = SQLAnswer(
        question="q",
        status="not_answerable",
        generated_sql=None,
        executed_sql=None,
        result=None,
        error=None,
        violations=None,
    )

    result = synthesize_answer(
        "How many invoices are overdue?",
        "needs_sql",
        sql_answer,
        [],
        llm=ExplodingLLM(),
    )

    assert result.method == "extractive"
    assert result.insufficient_evidence
    assert result.citations == []


def test_single_row_sql_uses_fast_path_without_llm():
    sql_answer = make_sql_answer([{"n": 12}], columns=["n"])

    result = synthesize_answer(
        "How many invoices are overdue?",
        "needs_sql",
        sql_answer,
        [],
        llm=ExplodingLLM(),
    )

    assert result.method == "extractive"
    assert "n=12" in result.text
    assert result.citations[0]["evidence_id"] == "S1"


def test_sql_only_fallback_formats_the_result():
    sql_answer = make_sql_answer([{"n": 12}], columns=["n"])
    llm = FakeLLM(RuntimeError("provider unavailable"))

    result = synthesize_answer(
        "How many invoices are overdue?",
        "needs_sql",
        sql_answer,
        [],
        llm=llm,
    )

    assert result.method == "extractive"
    assert "n=12" in result.text
    assert "[S1]" in result.text
    assert result.citations[0]["evidence_id"] == "S1"


def test_sql_evidence_must_be_cited_in_needs_both():
    chunk = make_chunk("The procurement approval threshold is 25,000 USD.")
    sql_answer = make_sql_answer([{"n": 98}], columns=["n"])
    response = """
    {
      "answer": "The procurement approval threshold is 25,000 USD [D1].",
      "claims": [
        {"text": "The procurement approval threshold is 25,000 USD.", "citations": ["D1"]}
      ],
      "insufficient_evidence": false,
      "missing_evidence": "",
      "conflicts": []
    }
    """
    llm = FakeLLM([response, response])

    result = synthesize_answer(
        "How many pending transactions and what approval threshold applies?",
        "needs_both",
        sql_answer,
        [chunk],
        llm=llm,
    )

    assert result.method == "extractive"
    assert "[S1]" in result.text
    assert "n=98" in result.text


def test_claim_must_appear_in_the_answer():
    chunk = make_chunk("The procurement approval threshold is 25,000 USD.")
    response = """
    {
      "answer": "The procurement approval threshold is 25,000 USD [D1].",
      "claims": [
        {"text": "A completely unrelated claim about expenses.", "citations": ["D1"]}
      ],
      "insufficient_evidence": false,
      "missing_evidence": "",
      "conflicts": []
    }
    """
    llm = FakeLLM([response, response])

    result = synthesize_answer(
        "What is the procurement approval threshold?",
        "needs_rag",
        None,
        [chunk],
        llm=llm,
    )

    assert result.method == "extractive"
    assert "25,000" in result.text


def test_needs_both_conflicts_are_passed_through():
    chunk = make_chunk("The policy requires 250,000 USD approval for this spend.")
    sql_answer = make_sql_answer([{"amount": 100}], columns=["amount"])
    response = """
    {
      "answer": "The ledger shows 100 [S1], while the policy threshold is 250,000 USD [D1].",
      "claims": [
        {"text": "The ledger shows 100.", "citations": ["S1"]},
        {"text": "The policy threshold is 250,000 USD.", "citations": ["D1"]}
      ],
      "insufficient_evidence": false,
      "missing_evidence": "",
      "conflicts": [
        {"description": "Ledger amount is below the policy threshold.", "values": ["100", "250000"]}
      ]
    }
    """
    llm = FakeLLM(response)

    result = synthesize_answer(
        "Does the ledger spend comply with the policy threshold?",
        "needs_both",
        sql_answer,
        [chunk],
        llm=llm,
    )

    assert result.method == "llm"
    assert result.conflicts
    assert result.conflicts[0]["values"] == ["100", "250000"]


def test_unsupported_numbers_ignore_citation_markers():
    assert _unsupported_numbers(
        "The threshold is 25,000 [D1] and 12 [S1].",
        "The threshold is 25,000 USD. The ledger count is 12.",
    ) == []


def test_unsupported_numbers_detects_a_real_mismatch():
    assert _unsupported_numbers(
        "The threshold is 99,000 USD [D1].",
        "The threshold is 25,000 USD.",
    ) == ["99000"]


def test_unsupported_numbers_accepts_scale_words():
    assert _unsupported_numbers(
        "The threshold is 18,000,000 USD [D1].",
        "The maximum single-vendor exposure threshold is 18 million USD.",
    ) == []
    assert _unsupported_numbers(
        "The threshold is 250,000 USD [D1].",
        "Purchases above 250k require approval.",
    ) == []
