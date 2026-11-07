from app.router.classifier import classify_question


class FakeLLM:
    def __init__(self, response: str | None = None, error: Exception | None = None):
        self.response = response
        self.error = error

    def complete(self, system: str, user: str) -> str:
        if self.error is not None:
            raise self.error
        return self.response or ""


def test_llm_structured_route_is_used():
    llm = FakeLLM('{"route":"needs_rag","confidence":0.91,"reason":"Policy question."}')

    decision = classify_question("Who approves emergency purchases over 50k?", llm=llm)

    assert decision.route == "needs_rag"
    assert decision.confidence == 0.91
    assert decision.reason == "Policy question."
    assert decision.method == "llm"


def test_markdown_fenced_json_is_parsed():
    llm = FakeLLM('```json\n{"route":"needs_both","confidence":0.8,"reason":"Needs ledger and policy."}\n```')

    decision = classify_question("Total overdue invoices and MSA payment terms?", llm=llm)

    assert decision.route == "needs_both"
    assert decision.confidence == 0.8
    assert decision.method == "llm"


def test_invalid_json_falls_back_to_keywords():
    llm = FakeLLM("I think this is a document question.")

    decision = classify_question("What is the procurement approval threshold?", llm=llm)

    assert decision.method == "keyword"
    assert decision.route == "needs_rag"


def test_invalid_route_falls_back_to_keywords():
    llm = FakeLLM('{"route":"needs_everything","confidence":0.99,"reason":"Nope."}')

    decision = classify_question("How many invoices are overdue?", llm=llm)

    assert decision.method == "keyword"
    assert decision.route == "needs_sql"


def test_provider_error_falls_back_to_keywords():
    llm = FakeLLM(error=RuntimeError("provider unavailable"))

    decision = classify_question("How many invoices are overdue?", llm=llm)

    assert decision.method == "keyword"
    assert decision.route == "needs_sql"


def test_unknown_question_defaults_to_both():
    llm = FakeLLM(error=RuntimeError("provider unavailable"))

    decision = classify_question("Tell me about the company", llm=llm)

    assert decision.method == "keyword"
    assert decision.route == "needs_both"
